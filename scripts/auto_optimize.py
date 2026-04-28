#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Iteratively drive the Cursor CLI to optimize a coding task.

The script runs many rounds of ``cursor-agent --print`` (headless mode) using
the *Opus 4.7 1M Max Thinking* model with **max mode** turned on, asking the
agent to make autonomous progress on a user-defined task. Between rounds it
runs a user-supplied *metric command* (which prints a single number where
"higher is better") and stops early if the metric has not improved for a
configurable number of rounds.

Each round is fully scripted - the script tells the agent to:

  1. Read the local skill at
     ``/root/.cursor/skills/hipkittens-primus-turbo-backend/SKILL.md``
     (or whatever ``--skill-path`` you point it at), so it inherits the
     project context without you having to repeat it every round.
  2. Decide on its own what to do this round.
  3. Run validation tests / benchmarks before claiming progress.
  4. Commit any improvement on the current branch.

After each round the script measures the new metric, compares it against the
historical best, and either updates ``best_sha`` / resets the patience counter
or warns and increments. When patience runs out the script prints a clear
``EARLY-STOP`` banner and exits.

Example
-------

::

    cd /workspace/code/Primus-Turbo
    python3 scripts/auto_optimize.py \\
        --rounds 80 \\
        --patience 5 \\
        --task "扩大 HipKittens BF16 grouped GEMM 的 allow-list 覆盖，并提升 TFLOPS" \\
        --metric-cmd 'HIP_VISIBLE_DEVICES=7 \\
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \\
python3 -m pytest tests/pytorch/ops/test_grouped_gemm.py -k hipkitten \\
--tb=no -q 2>&1 | tail -1 | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+"' \\
        --metric-name "hipkitten_pass_count"

Run ``python3 scripts/auto_optimize.py --help`` for the full set of options.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "claude-opus-4-7-thinking-max"
DEFAULT_SKILL = "/root/.cursor/skills/hipkittens-primus-turbo-backend/SKILL.md"
DEFAULT_TASK = (
    "**最终目标**（6 段独立判定，每段都必须 PASS；看 metric stderr 的 "
    "`Goals:` 6 行 PASS/FAIL）：\n"
    "  (1) **BF16_fwd vs HIPBLASLT** ≥ 0.97 —— dense BF16 GEMM forward (8 shape)\n"
    "  (2) **BF16_bwd vs HIPBLASLT** ≥ 0.97 —— dense BF16 GEMM backward (8 shape)\n"
    "  (3) **FP8_fwd  vs HIPBLASLT** ≥ 0.97 —— dense FP8 tensorwise GEMM forward (8 shape)\n"
    "  (4) **FP8_bwd  vs TRITON**    ≥ 1.20 —— dense FP8 tensorwise GEMM backward (8 shape)\n"
    "  (5) **grp_BF16 vs TRITON**    ≥ 1.20 —— grouped BF16 GEMM (16 shape, DeepSeek-V3 + gpt_oss_20B)\n"
    "  (6) **grp_FP8  vs TRITON**    ≥ 1.20 —— grouped FP8 tensorwise GEMM (同 16 shape)\n\n"
    "6 段全 PASS 才算交付；metric score (= 6 段 geomean 的 geomean × 1000) 只"
    "是 auto_optimize 的 ranking 信号，**不是验收标准**。\n\n"
    "═══════════════════════════════════════════════════════════════════════\n"
    "**核心架构方向（必读，决定本轮所有判断）**\n"
    "═══════════════════════════════════════════════════════════════════════\n\n"
    "参考 turbo 自己的 triton GEMM：\n"
    "  `primus_turbo/triton/gemm/gemm_kernel.py::offline_select_bf16` (line 239-281)\n"
    "  + `gemm_triton_kernel` 主入口 (line 626-823)\n"
    "它**完全没有 cache / JSON / pickle / dict lookup**，全是 if/else 规则：\n"
    "  ```\n"
    "  BM, BN, BK = 256, 256, 64                           # 90% shape 默认 tile\n"
    "  group_m = 8 if min(tiles_m, tiles_n) < 16 else 4    # ← 规则，不查表\n"
    "  use_bk64 = is_tn and (K > 3584 or min(M,N) > 4608)  # ← 规则\n"
    "  ```\n"
    "kernel 永远能跑 —— 永远有一组规则给出 default config。triton 的 `offline_select_bf16` "
    "docstring 写得很清楚：186 个 bench entry 是**线下分析时**用的，归纳成 if/else 规则后，"
    "**runtime 不再需要那张表**。这就是 user 想要的 'autotune 是 turbo 这边可设的，"
    "一般不开，有一组根据 mnk 的默认配置'。\n\n"
    "**HipKittens backend 必须改成同样的模式**：\n"
    "  1. 在 `primus_turbo/pytorch/kernels/hipkitten/` 下新建 `config.py`，写一个纯函数：\n"
    "       `select_default_config(M, N, K, layout, dtype) -> Config`\n"
    "     （Config 含 group_m / num_xcds / kernel variant id）。**依据**：把\n"
    "     `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/bench_bf16_no_jit_final.json` (~58 行)\n"
    "     `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/.autotune_cache.json` (~177 行)\n"
    "     一次性 dump 出来线下看，归纳 (M,N,K,layout) → cfg 的分布成几条 if/else。\n"
    "     这些 .json 是**离线 bench 笔记**，runtime **不再 import / parse**。\n"
    "  2. `can_handle` 只查硬约束：dtype 支持、layout (NT/NN/TN/TT)、alignment\n"
    "     （HK kernel 对 M/N/K 的 tile 对齐要求）。**不再查 cache 是否命中**。\n"
    "     永远命中 default 规则 → 永远不 reject。\n"
    "  3. dispatch 路径 = `select_default_config(...)` + `kernel(args, group_m=g, num_xcds=x)`，\n"
    "     **零 IO、零 cache、零 dict、零 pickle、零 JSON parse**。\n"
    "  4. autotune 是 turbo 这边的**可选机制**（像 `@triton.autotune`）。\n"
    "     **默认关闭**。开了之后是开发者工具：跑 sweep → 看 winning config →\n"
    "     **开发者手动**把结论写回 `select_default_config` 的规则里，**结果不存盘**。\n"
    "     不是 runtime 路径 —— 第一刀做完后再考虑，可选。\n\n"
    "═══════════════════════════════════════════════════════════════════════\n"
    "**ROUND 1 起点 = 545** / 6 goals = 0 PASS（架构第一/二刀完成后的实测）：\n"
    "  • BF16_fwd  (8)   geomean=0.937 vs HIPBLASLT  ← 离 0.97 一步之遥\n"
    "  • BF16_bwd  (8)   geomean=0.739 vs HIPBLASLT  ← bwd 走 3 次 dispatch，还有 host overhead\n"
    "  • FP8_fwd   (8)   geomean=0.918 vs HIPBLASLT  ← 离 0.97 一步之遥\n"
    "  • FP8_bwd   (8)   geomean=0.775 vs TRITON     ← 距 1.20 远；HK FP8 CRR kernel 慢\n"
    "  • grp_BF16 (16)   geomean=0.098 vs TRITON     ← gpt_oss_20B 的 8/16 因 N=K=2880 不对齐 256/128 被 reject\n"
    "  • grp_FP8  (16)   geomean=?     vs TRITON     ← 还没测过；HK FP8 .so 没有 grouped binding，per-group launch 必慢\n"
    "新架构下 reject 全部源于**真硬约束**（dtype / layout / tile alignment），不再有 cache miss 类\n"
    "假 reject。**收紧 can_handle 反而扣分**：reject → ratio clip 0.01 → 整段 geomean 塌掉。\n\n"
    "**严禁的『假优化』模式**（违反 = 本轮立即作废）：\n"
    "  ✗ **runtime 读任何 .json / .pkl / .autotune_cache.json**（开发期 bench 用、归纳完后\n"
    "    runtime 不再 import；任何 hipkitten 路径上看到 `json.load`/`pickle.load` 都是违规）\n"
    "  ✗ **任何形式 cache：dict / weakref / data_ptr / _version / LRU / TTL** —— quant 输出 /\n"
    "    preshuffle / group_offs / grid_x_hint / scale / autograd 中间产物全在禁单里\n"
    "  ✗ case-by-case 形状表：`if (M,N,K)==(X,Y,Z): return cfg`（**通用规则**才允许，\n"
    "    例如 `if K>=4096`、`if min(tiles)<16`、`if N>=K`）\n"
    "  ✗ 收紧 can_handle 把难 shape 排除掉（geomean clip 0.01，分数立刻塌）\n"
    "  ✗ **grouped 路径严禁任何形式的 balanced 假设**（命名 + 资格判定都不准）：\n"
    "    1) `can_handle` / dispatch 路径**禁止**加任何形如 `_is_balanced_group_lens` /\n"
    "       `all(g==g[0] for g in group_lens)` 的检查去 reject 非均匀 `group_lens` —— 真实\n"
    "       MoE 训练里 group_lens 永远是稀疏不均匀的，reject 一个 shape → ratio clip 0.01\n"
    "       → 整段 geomean 塌掉。HK kernel 必须能跑任意 group_lens；当前 HK BF16 grouped\n"
    "       launcher 只接受均匀 M-per-group，是 launcher 实现细节，所以 `execute` 用\n"
    "       `_uniform_group_m` 做 *fast-path detection*（detect 不到 → per-group `dense_run`\n"
    "       fallback），**不是 reject**。\n"
    "    2) **命名同样禁止传播 balanced 污染**：HK 仓库 BF16 .so 当前暴露的 `grouped_*_balanced`\n"
    "       是历史遗留命名，Primus 内部已经统一别名为 `grouped_*`（loader 通过\n"
    "       `_resolve_grouped_attr` 兼容老 .so，新代码不暴露 `_balanced` 后缀）。任何**新加**\n"
    "       的 HK 仓库 binding（FP8 grouped、新增 layout、新加变体）一律命名 `grouped_*`，\n"
    "       不准带 `_balanced` 后缀；任何**新加**的 Primus 端 grouped API 也不准带 `_balanced` 后缀。\n"
    "       这条规则覆盖 BF16 grouped、新加的 FP8 grouped、未来一切 grouped 入口。\n"
    "  ✗ 只改 metric/test/config.py 文件让数字变好\n"
    "  ✗ 加 pytest.skip / 删 parametrize / 提高 SNR 阈值\n\n"
    "**两仓库分工**：\n"
    "  - /workspace/code/HipKittens —— kernel/launcher 仓：tile/wave layout/swizzle/MFMA。\n"
    "    改 .cpp 后进 analysis/{bf16,fp8}_gemm/mi350x/ 跑 `source ../../../env.src && make -j` 重编\n"
    "    tk_*_layouts.so（Primus 自动 reload）。**.json 文件保留作为离线 bench 笔记，\n"
    "    runtime 不读**。\n"
    "  - /workspace/code/Primus-Turbo —— dispatch 仓：写 `select_default_config` 的通用规则\n"
    "    + can_handle 硬约束检查；不写形状表。\n\n"
    "**FROZEN（不可修改）**：\n"
    "  - scripts/_metric_hk_ratio.py / scripts/auto_optimize.py / scripts/run_dod_metric.sh\n"
    "  - tests/pytorch/ops/test_*.py（不能加 skip / 删 parametrize / 调 SNR 阈值）\n"
    "  - benchmark/ops/config.py（shape ground truth）\n"
    "  - /root/.cursor/skills/hipkittens-primus-turbo-backend/SKILL.md"
)
# Loop metric: real benchmark of HIPKITTEN vs default-backend TFLOPS on a
# fixed 16-shape LLM-typical suite (BF16 + FP8 tensorwise dense). Score is
# int(geomean(hk_tflops / ref_tflops) * 1000); target >= 900 (= 90%).
# HIPKITTEN reject -> ratio clipped to 0.01 -> ~100x geomean penalty so the
# agent can't game the score by narrowing can_handle. ~10s wall.
DEFAULT_METRIC_CMD = "python3 scripts/_metric_hk_ratio.py"
# Final acceptance is the full DoD pytest suite (all 4 files, both default and
# --deterministic-only). Too slow for every round; the agent / user runs this
# occasionally to confirm we haven't regressed the broader sweeps. Empty by
# default so the loop doesn't measure it.
DEFAULT_DETERMINISTIC_CMD = ""
# DoD checkpoint: every N rounds we run the full 4-file pytest suite as a
# regression guardrail. The fast metric (DEFAULT_METRIC_CMD) doesn't catch
# every shape — e.g. the user-facing op layer assertions and the
# non-HIPKITTEN backend regressions are only covered in pytest.
DEFAULT_DOD_EVERY = 5
DEFAULT_DOD_CMD = "bash scripts/run_dod_metric.sh --full"
CLI_CONFIG_PATH = Path(os.path.expanduser("~/.cursor/cli-config.json"))


@dataclass
class RoundResult:
    index: int
    started_at: str
    finished_at: str
    duration_s: float
    metric: Optional[float]
    best_so_far: Optional[float]
    improved: bool
    head_sha_before: str
    head_sha_after: str
    cursor_exit_code: int
    log_dir: str

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 2),
            "metric": self.metric,
            "best_so_far": self.best_so_far,
            "improved": self.improved,
            "head_sha_before": self.head_sha_before,
            "head_sha_after": self.head_sha_after,
            "cursor_exit_code": self.cursor_exit_code,
            "log_dir": self.log_dir,
        }


@dataclass
class TrajectoryState:
    rounds: list[RoundResult] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_sha: Optional[str] = None
    rounds_without_improvement: int = 0
    dod_checkpoints: list[dict] = field(default_factory=list)
    last_dod_score: Optional[int] = None
    last_dod_sha: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--rounds", type=int, default=80, help="Maximum number of optimization rounds.")
    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop early once this many consecutive rounds end without metric improvement.",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Improvement must exceed best_so_far + min_delta to count.",
    )
    p.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="High-level optimization goal injected into every round's prompt.",
    )
    p.add_argument(
        "--skill-path",
        type=str,
        default=DEFAULT_SKILL,
        help="Path to the project skill .md file the agent must read first.",
    )
    p.add_argument(
        "--metric-cmd",
        type=str,
        default=DEFAULT_METRIC_CMD,
        help=(
            "Shell command that prints a single float metric to stdout (higher is better). "
            "Run after every round and at startup for the baseline."
        ),
    )
    p.add_argument(
        "--metric-name",
        type=str,
        default="hk_ratio_score",
        help="Human-readable metric label (used in logs only).",
    )
    p.add_argument(
        "--deterministic-cmd",
        type=str,
        default=DEFAULT_DETERMINISTIC_CMD,
        help=(
            "Shell command for the deterministic-only suite that completes the DoD bar. "
            "Quoted into the agent prompt so it knows the second half of 'all pass'. "
            "Set to empty string to omit."
        ),
    )
    p.add_argument(
        "--dod-every",
        type=int,
        default=DEFAULT_DOD_EVERY,
        help=(
            "Run the full-DoD pytest checkpoint every N rounds (0 disables). "
            "Catches regressions the fast probe metric can't see."
        ),
    )
    p.add_argument(
        "--dod-cmd",
        type=str,
        default=DEFAULT_DOD_CMD,
        help=(
            "Shell command run on each DoD checkpoint. Should print a single "
            "integer score where score >= 0 means all-pass."
        ),
    )
    p.add_argument(
        "--dod-timeout",
        type=int,
        default=60 * 60,
        help=(
            "Seconds to allow each DoD checkpoint (default 60 min). "
            "Full 4-file pytest with -n=#idle-pool-GPUs can legitimately "
            "take 40+ min on a busy box."
        ),
    )
    p.add_argument(
        "--gpu-pool",
        type=str,
        default="0,2,3",
        help=(
            "Comma-separated list of GPU ids the loop is allowed to use. "
            "Default 0,2,3 (GPU 1 is currently reserved for other workloads). "
            "Exported as HIPKITTEN_GPU_POOL, honored by scripts/_metric_hk_ratio.py "
            "and scripts/run_dod_metric.sh (idle picks intersect this pool). "
            "Empty string disables (let scripts see all GPUs)."
        ),
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Working directory passed to cursor-agent and metric command.",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Cursor CLI model slug.")
    p.add_argument(
        "--no-max-mode",
        action="store_true",
        help="Do not toggle maxMode in ~/.cursor/cli-config.json (default: turn it on for the run).",
    )
    p.add_argument(
        "--round-timeout",
        type=int,
        default=60 * 30,
        help="Seconds to allow each cursor-agent invocation before killing it (default 30 min).",
    )
    p.add_argument(
        "--metric-timeout",
        type=int,
        default=60 * 15,
        help="Seconds to allow the metric command (default 15 min).",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Where to put per-round logs. Defaults to <workspace>/auto_optimize_logs/<timestamp>.",
    )
    p.add_argument(
        "--prompt-extra",
        type=str,
        default="",
        help="Additional text appended to every round's prompt (e.g. extra constraints).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip cursor-agent invocations, just measure the metric N times. Useful for testing.",
    )
    return p.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def banner(msg: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def section(msg: str) -> None:
    print(f"\n--- {msg} ---", flush=True)


def run_metric(cmd: str, cwd: str, timeout: int) -> Optional[float]:
    section(f"measuring metric: {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[metric] TIMEOUT after {timeout}s", flush=True)
        return None
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0:
        print(
            f"[metric] non-zero exit {result.returncode}; "
            f"stdout={out[-300:]!r} stderr={err[-300:]!r}",
            flush=True,
        )
    if not out:
        print(f"[metric] empty stdout; stderr={err[-300:]!r}", flush=True)
        return None
    last_line = out.splitlines()[-1].strip()
    try:
        value = float(last_line)
    except ValueError:
        print(f"[metric] could not parse {last_line!r} as float", flush=True)
        return None
    print(f"[metric] = {value}", flush=True)
    return value


def get_head_sha(cwd: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
        ).strip()
        return out
    except subprocess.CalledProcessError:
        return ""


def get_recent_log(cwd: str, n: int = 5) -> str:
    try:
        return subprocess.check_output(
            ["git", "log", f"-{n}", "--oneline"], cwd=cwd, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def get_short_status(cwd: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=cwd, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def build_prompt(
    args: argparse.Namespace,
    state: TrajectoryState,
    round_idx: int,
    baseline_metric: Optional[float],
    head_sha: str,
    recent_log: str,
    short_status: str,
) -> str:
    """Build the per-round prompt fed to cursor-agent.

    The prompt is in Chinese to match the user's preferred working language.
    It always:
      * tells the agent to read the project skill first;
      * gives it the metric and history;
      * lets it decide what to optimize;
      * reminds it of hard constraints (FP8 tests, autotune=False, etc.);
      * asks it to commit any progress.
    """
    last = state.rounds[-1] if state.rounds else None
    last_metric = last.metric if last else baseline_metric
    last_improved = "是" if (last and last.improved) else "否"

    history_lines = []
    for r in state.rounds[-5:]:
        history_lines.append(
            f"  - 第 {r.index} 轮: metric={r.metric}, "
            f"best={r.best_so_far}, improved={r.improved}, "
            f"sha {r.head_sha_before[:8]}->{r.head_sha_after[:8]}"
        )
    history_block = "\n".join(history_lines) if history_lines else "  (尚无历史)"

    gpu_pool = args.gpu_pool or "(unrestricted)"

    dod_block = ""
    if args.dod_every > 0:
        last_dod_line = (
            f"上一次 DoD score = {state.last_dod_score}（SHA {(state.last_dod_sha or '')[:8]}）"
            if state.last_dod_score is not None
            else "尚未跑过 DoD checkpoint"
        )
        dod_block = (
            f"\n【DoD 检查点（每 {args.dod_every} 轮自动跑一次，不需要你手动跑）】\n"
            f"脚本会在第 {args.dod_every}, {2*args.dod_every}, ... 轮结束后自动执行：\n"
            f"  {args.dod_cmd}\n"
            f"它跑 4 文件全套 pytest（test_gemm{{,_fp8}}, test_grouped_gemm{{,_fp8}}），"
            f"任何 failed > 0 都会让脚本立刻 EARLY-STOP。所以你 commit 的时候要小心：\n"
            f"  - 如果你的改动只触及 HIPKITTEN 路径（grouped_gemm_impl.GroupedGEMMHipKittenBackend、"
            f"gemm_fp8_impl.GEMMFP8HipKittenBackend、kernels/hipkitten/* 等），快 metric 通常足够。\n"
            f"  - 如果你触及任何**共用代码**（autograd 入口、dispatcher、quantize_fp8_*、"
            f"grouped_gemm.py 顶层、torch.library custom_op 注册等），你**必须**怀疑会影响"
            f"非 HIPKITTEN 后端，主动跑一次 `{args.dod_cmd}`（约 5-10 分钟）确认 0 failed 后再 commit；"
            f"否则脚本下次 checkpoint 会因为你的改动 EARLY-STOP，整个 run 报废。\n"
            f"  - {last_dod_line}\n"
        )

    return f"""你是 HipKittens × Primus-Turbo 联合优化协作者。本次是第 {round_idx} / {args.rounds} 轮，由脚本自动调度。
你**同时拥有两个仓库的写权限**：
  • Primus-Turbo: {args.workspace} (本轮工作目录、metric 在这里跑)
  • HipKittens : /workspace/code/HipKittens  (kernel 源码 + 离线 bench 笔记 .json，runtime 不读)
当前 Primus-Turbo git HEAD: {head_sha}

【强制第一步】
请先用读文件工具完整读取这份本地 skill：
  {args.skill_path}
里面有 HipKittens + Primus-Turbo 集成的所有上下文（路径、env、cache 结构、白名单、坑）。读完再决定本轮该做什么。

【优化目标】
{args.task}

【本轮的快速验收命令】
metric 命令（约 10-30 秒、单 GPU、自动选空闲卡）：
  {args.metric_cmd}
含义：在 48 个 LLM 典型 shape（BF16/FP8 dense fwd+bwd 各 8 + BF16 grouped 16）上跑
HIPKITTEN vs reference 的 TFLOPS 实测。score = int(geomean²(hk/ref) × 1000)。
**新架构下 HIPKITTEN 不应该 reject 任何 shape** —— select_default_config 永远返回 cfg，
kernel 永远能跑。reject 一个 shape → ratio clip 0.01 → geomean 大幅下跌，所以收紧
can_handle 反而扣分。新架构下，本轮的真实压力来自 (a) dispatch overhead 是否清掉、
(b) HK kernel 跟 reference 的相对速度。

【**首要数据源** — metric 的 stderr 表】
metric 命令的 stderr 会打印一张逐 shape 表，列出 dtype / (M,N,K) / hk_tflops / ref_tflops /
ratio / status。**本轮第一步先跑一次 metric**，从那张表里找出 ratio < 0.9 的 shape，
按 ratio 升序选 1 个作为本轮攻坚目标。**不许凭印象选 shape**，必须用上一轮 metric 数据。
跑命令：`{args.metric_cmd} 2>&1 | tee /tmp/metric_round_{round_idx}.log`。
**改完一次、commit 前一次** —— 不要每改一行都跑。
{dod_block}

【度量指标】指标名: {args.metric_name}（数值越高越好；900 = 90% 是 DoD）
- 基线 (优化开始前) = {baseline_metric}
- 历史最佳 = {state.best_metric}
- 上一轮 = {last_metric}（improved={last_improved}）
- 已连续 {state.rounds_without_improvement} 轮未提升（patience={args.patience}）

【近期 Primus-Turbo git log】
{recent_log or "(空)"}

【当前 Primus-Turbo working tree 状态】
{short_status or "(干净)"}

【近 5 轮记录】
{history_block}

【优化方向 — phased 路线】（"严禁假优化"清单已在【优化目标】里给出，请重读）
  ✓ 第一刀 / 第二刀 已完成（commit `1970d91`）：rule-based dispatch + 删 runtime JSON cache lookups。
    `select_default_config` 在 `primus_turbo/pytorch/kernels/hipkitten/config.py`，runtime 不再 parse
    任何 JSON/pickle。后面所有的提速都建立在这个基础上 —— **不要回退它**。
  ☐ 第三刀（接下来 1-2 轮）：BF16/FP8 fwd 0.94 → 0.97，BF16_bwd 0.74 → 0.97。
    这两段都是 dispatch host overhead 残余 + select_default_config 规则不够细的问题：
      • BF16/FP8 fwd 4096x4096x4096 仅 0.888 —— 看是 (a) host overhead vs (b) tile cfg 不优；
      • BF16_bwd 走 fwd + dA + dB 三次 dispatch，host overhead 累积；
        可写专用 fused autograd 入口跳过中间 op-registration。
    依据：用 stderr 表锁定 ratio < 0.97 的具体 shape，再调规则。
  ☐ 第四刀：grp_BF16 0.098 → 1.20。8/16 reject 全部是 gpt_oss_20B (N=2880 / K=2880 不对齐 256/128 tile)。
    两条路：(a) Primus 端写通用 pad-and-copy 处理非对齐 grouped (会有 overhead，但能解 8 个 case)；
    (b) 在 HipKittens 仓库出 256/128/?(non-256 N) 兼容 tile 模板 (更彻底)。先 (a) 验证可行性。
    **再次提醒：can_handle 严禁判 balanced**（见上面"严禁的『假优化』模式"）。
  ☐ 第五刀（**新加，重点**）：grp_FP8 vs TRITON >= 1.20。HipKittens FP8 .so 当前**没有**
    原生 grouped binding，Primus 端 `GroupedGEMMFP8HipKittenBackend.execute` 是 per-group
    `dense_run` 循环 fallback —— B=32 时 launch overhead 让它在 Triton 持久 kernel 面前
    崩盘。**修法**（命名严格遵守『假优化』清单的禁 _balanced 规则）：
      1. 在 `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` 末尾加 3 条
         `m.def("grouped_<layout>", &grouped_<layout>, ...)` binding，layout ∈ {{rcr, rrr, crr}}。
         **不要**带 `_balanced` 后缀（命名一律 `grouped_rcr` / `grouped_rrr` / `grouped_crr`）。
         BF16 .so 暴露的 `grouped_*_balanced` 是历史遗留，新 binding 不传播这个污染 ——
         Primus 端 loader 用 `_resolve_grouped_attr` 已经兼容两种命名，所以 BF16 老 .so 仍能
         加载。binding 实现层面：MoE 真实流量 group_lens 不均匀，所以最好把 launcher 写成
         接受 cumulative offsets 直接消费（而不是假设均匀 M-per-group）—— 这样 Primus 端
         永远不需要 fast-path detection。
      2. cd 到 mi350x，`source ../../../env.src && make -j`，重编 `tk_fp8_layouts.so`，确认
         `python3 -c "import tk_fp8_layouts; print([x for x in dir(tk_fp8_layouts) if 'grouped' in x])"`
         打印 `['grouped_crr', 'grouped_rcr', 'grouped_rrr']`（无 `_balanced` 后缀）。
      3. Primus 端 `kernels/grouped_gemm/grouped_gemm_fp8_impl.py::GroupedGEMMFP8HipKittenBackend.execute`
         去掉 per-group for 循环，改成一次 `hipkitten.grouped_run(hk, cfg, ...)` 调用。loader
         自动通过 `_resolve_grouped_attr` 把新 `grouped_*` attribute 拿出来。
      4. 跑 fp32 数值 probe（max_abs + SNR）确认 deepseek-v3 / gpt_oss_20B FP8 grouped
         通过；跑 metric 验证 grp_FP8 ratio 提升；commit。
  ☐ HipKittens kernel 改写（任意时刻可做，不依赖前 5 刀）：改 .cpp 的 tile/wave/swizzle/
    MFMA 排布，进 analysis/{{bf16,fp8}}_gemm/mi350x `source ../../../env.src && make -j`，
    生成 tk_{{bf16,fp8}}_layouts.so —— Primus 自动加载新 .so。提速来源：bank conflict、
    ds_read 吞吐、K_STEP 覆盖 K=128/256 倍数。
  ☐ 可选第六刀：写 opt-in autotune 工具（开发期 sweep，结果只 print 不存盘 —— 开发者手动
    把 winning config 写回 select_default_config 规则）。

【典型流程示例（按新架构）】
1. 跑 metric 看哪些 shape ratio < 0.9（stderr 表里有 dtype/(M,N,K)/layout/ratio/status）
2. 在 select_default_config 里加 / 调一条规则覆盖这族 shape（**规则**，不是 if shape==X return Y）。
   依据：HK 仓库的 .json 数据（开发期看，归纳完后 runtime 不 import）。
3. 跑 fp32 数值 probe（max_abs + SNR）确认改动数值正确。
4. 跑 metric 验证 ratio 提升。
5. commit 在 Primus-Turbo（如果同时也改了 HK kernel cpp，分别 commit 在两个仓库）。
   commit message 要贴 (a) 改了哪条规则 (b) before/after ratio (c) SNR 数字。

【两个仓库的 commit】
- 改了 HipKittens：进 /workspace/code/HipKittens 用 git add/commit；不要 push。
- 改了 Primus-Turbo：进 {args.workspace} commit；不要 push。
- 如果两边都改了，本轮 git log 在 Primus-Turbo 显示的是 Primus 这边的 commit；
  在你的本轮小结里**列出 HipKittens commit SHA**，方便用户回溯。

【脚本机制硬约束 - 不可违反】
- **GPU 池**：本次 run 只允许使用 `HIPKITTEN_GPU_POOL={gpu_pool}`（已经写进环境变量；GPU 1
  当前被其他作业占用，绝不许动）。metric / DoD / 你自己的任何 benchmark/probe 都**只能从这个池
  里挑卡**，跑前先 `rocm-smi --showuse --showpids` 看哪张空闲。**绝不许手动 export
  HIP_VISIBLE_DEVICES** 去用池外的卡。
- BackendType.HIPKITTEN 必须保持 `BackendEntry(..., autotune=False)` —— 它是手动 backend，
  不能进 autotune 池。
- 任何动 dispatch / can_handle / group_m 规则的修改都必须配一个小 python 数值 probe
  （比 fp32 reference 算 max_abs + SNR），把两个数值贴进 commit message。
- **每轮在每个仓库最多 1 个 focused commit**；message 用 `feat:` / `fix:` / `perf:` /
  `refactor:` 风格；commit 后**绝不 push**任何 remote。
- 不要修改 ~/.cursor/cli-config.json 或全局 git config。
- 不要用 `git rebase -i / git add -i` 这类交互命令。

【输出要求】
本轮结束前给一段 markdown 小结，包含：
- 本轮选择的目标 + 选了哪个方向（kernel / cache / 规则 dispatch）
- HipKittens 与 Primus-Turbo 各自改了哪些文件
- metric 命令的结果数字（before / after）
- 两个仓库各自的 commit SHA / message（如果都改了）
- 下一轮建议下游做什么

完成后退出（headless 模式会自动结束）。

{args.prompt_extra}"""


def maybe_set_max_mode(enabled: bool) -> Optional[dict]:
    """Toggle maxMode in cli-config.json. Returns the original snapshot, or None."""
    if not CLI_CONFIG_PATH.exists():
        print(f"[max-mode] {CLI_CONFIG_PATH} not found - skipping toggle.", flush=True)
        return None
    original = json.loads(CLI_CONFIG_PATH.read_text())
    snapshot = json.loads(json.dumps(original))  # deep copy
    changed = False
    if original.get("maxMode") != enabled:
        original["maxMode"] = enabled
        changed = True
    if isinstance(original.get("model"), dict) and original["model"].get("maxMode") != enabled:
        original["model"]["maxMode"] = enabled
        changed = True
    if changed:
        CLI_CONFIG_PATH.write_text(json.dumps(original, indent=2))
        print(
            f"[max-mode] set maxMode={enabled} in {CLI_CONFIG_PATH}",
            flush=True,
        )
    return snapshot


def restore_cli_config(snapshot: Optional[dict]) -> None:
    if snapshot is None:
        return
    try:
        CLI_CONFIG_PATH.write_text(json.dumps(snapshot, indent=2))
        print(f"[max-mode] restored {CLI_CONFIG_PATH}", flush=True)
    except OSError as exc:
        print(f"[max-mode] failed to restore cli-config.json: {exc}", flush=True)


def run_cursor_round(
    args: argparse.Namespace,
    prompt: str,
    log_dir: Path,
) -> int:
    """Run a single cursor-agent round, streaming output into log_dir/cursor.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.md").write_text(prompt)

    cmd = [
        "cursor-agent",
        "--print",
        "--force",
        "--trust",
        "--model",
        args.model,
        "--workspace",
        args.workspace,
        "--output-format",
        "text",
        prompt,
    ]
    print(
        f"[cursor] launching: cursor-agent --print --force --trust --model {args.model} ...",
        flush=True,
    )
    log_path = log_dir / "cursor.log"
    with log_path.open("w") as logf:
        logf.write(f"# Command: {shlex.join(cmd[:-1])} <prompt>\n")
        logf.write(f"# Started at: {now_iso()}\n\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=args.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            print("[cursor] cursor-agent not on PATH - aborting.", flush=True)
            return 127
        try:
            assert proc.stdout is not None
            start = time.monotonic()
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                # Mirror to console so the user can follow live progress.
                sys.stdout.write(line)
                sys.stdout.flush()
                if time.monotonic() - start > args.round_timeout:
                    print(
                        f"\n[cursor] round timeout {args.round_timeout}s exceeded; "
                        "sending SIGTERM",
                        flush=True,
                    )
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return proc.returncode if proc.returncode is not None else 124
            proc.wait()
        except KeyboardInterrupt:
            print("\n[cursor] interrupted by user; killing cursor-agent", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        return proc.returncode if proc.returncode is not None else 1


def run_dod_checkpoint(
    cmd: str,
    cwd: str,
    timeout: int,
    log_path: Path,
) -> tuple[Optional[int], int]:
    """Run the full DoD pytest gate, log raw output, return (score, exit_code).

    Streams stdout/stderr line-by-line to log_path so a `tail -f dod.log`
    watcher sees real-time progress (subprocess.PIPE buffering meant we
    only saw output after the run terminated, which made a 30-min timeout
    appear as a stuck zero-output process).

    score is parsed from the last line of captured stdout (single integer,
    >= 0 = all pass). Returns (None, rc) on timeout or unparseable output.
    """
    import threading
    log_path.parent.mkdir(parents=True, exist_ok=True)
    section(f"DoD checkpoint: {cmd[:120]}{'...' if len(cmd) > 120 else ''}")

    import os as _os
    import signal as _signal

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # new pgrp so we can kill the whole tree
    )
    pgid = _os.getpgid(proc.pid)

    captured_lines: list[str] = []

    def _pump() -> None:
        with log_path.open("w") as logf:
            logf.write(f"# Command: {cmd}\n# Started at: {now_iso()}\n# pgid: {pgid}\n\n")
            logf.flush()
            assert proc.stdout is not None
            for line in proc.stdout:
                captured_lines.append(line)
                logf.write(line)
                logf.flush()

    pump_thread = threading.Thread(target=_pump, daemon=True)
    pump_thread.start()

    def _killtree() -> None:
        try:
            _os.killpg(pgid, _signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            _os.killpg(pgid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[dod] TIMEOUT after {timeout}s, killing pytest tree (pgid={pgid})", flush=True)
        _killtree()
        pump_thread.join(timeout=5)
        return None, 124

    pump_thread.join(timeout=10)

    out = "".join(captured_lines).strip()
    last = out.splitlines()[-1].strip() if out else ""
    try:
        score = int(last)
    except ValueError:
        print(f"[dod] could not parse {last!r} as int (rc={rc})", flush=True)
        return None, rc
    print(f"[dod] score={score} rc={rc}", flush=True)
    return score, rc


def write_summary(summary_path: Path, args: argparse.Namespace, state: TrajectoryState, baseline: Optional[float]) -> None:
    summary = {
        "started_at": getattr(write_summary, "_start", now_iso()),
        "metric_name": args.metric_name,
        "metric_cmd": args.metric_cmd,
        "dod_cmd": args.dod_cmd if args.dod_every > 0 else None,
        "dod_every": args.dod_every,
        "model": args.model,
        "max_mode": not args.no_max_mode,
        "rounds_planned": args.rounds,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "baseline_metric": baseline,
        "best_metric": state.best_metric,
        "best_sha": state.best_sha,
        "rounds_run": len(state.rounds),
        "rounds": [r.as_dict() for r in state.rounds],
        "dod_checkpoints": state.dod_checkpoints,
        "last_dod_score": state.last_dod_score,
        "last_dod_sha": state.last_dod_sha,
    }
    summary_path.write_text(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f"workspace {workspace} is not a directory", file=sys.stderr)
        return 2

    if shutil.which("cursor-agent") is None and not args.dry_run:
        print("cursor-agent not on PATH; install Cursor CLI first", file=sys.stderr)
        return 2

    skill_path = Path(args.skill_path)
    if not skill_path.exists():
        print(
            f"[skill] WARNING: skill file {skill_path} not found - the agent will be told to read it anyway.",
            flush=True,
        )

    log_dir = Path(args.log_dir) if args.log_dir else (
        workspace / "auto_optimize_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.json"
    write_summary._start = now_iso()  # type: ignore[attr-defined]

    cli_snapshot = None if args.no_max_mode else maybe_set_max_mode(True)

    if args.gpu_pool:
        os.environ["HIPKITTEN_GPU_POOL"] = args.gpu_pool
        print(f"[gpu-pool] HIPKITTEN_GPU_POOL={args.gpu_pool}", flush=True)

    state = TrajectoryState()
    baseline = None

    try:
        banner(f"AUTO-OPTIMIZE start | rounds={args.rounds} | patience={args.patience} | log_dir={log_dir}")
        section("baseline metric")
        baseline = run_metric(args.metric_cmd, str(workspace), args.metric_timeout)
        state.best_metric = baseline
        state.best_sha = get_head_sha(str(workspace))
        write_summary(summary_path, args, state, baseline)

        for i in range(1, args.rounds + 1):
            banner(
                f"ROUND {i}/{args.rounds} | best={state.best_metric} "
                f"| no_improve_streak={state.rounds_without_improvement}/{args.patience}"
            )
            sha_before = get_head_sha(str(workspace))
            recent_log = get_recent_log(str(workspace))
            short_status = get_short_status(str(workspace))
            prompt = build_prompt(
                args, state, i, baseline, sha_before, recent_log, short_status
            )

            round_dir = log_dir / f"round_{i:03d}"
            started_at = now_iso()
            t0 = time.monotonic()
            if args.dry_run:
                print("[dry-run] skipping cursor-agent", flush=True)
                round_dir.mkdir(parents=True, exist_ok=True)
                (round_dir / "prompt.md").write_text(prompt)
                cursor_exit = 0
            else:
                cursor_exit = run_cursor_round(args, prompt, round_dir)
            duration = time.monotonic() - t0

            sha_after = get_head_sha(str(workspace))
            metric = run_metric(args.metric_cmd, str(workspace), args.metric_timeout)

            improved = False
            if metric is not None:
                if state.best_metric is None or metric > (state.best_metric + args.min_delta):
                    state.best_metric = metric
                    state.best_sha = sha_after
                    state.rounds_without_improvement = 0
                    improved = True
                else:
                    state.rounds_without_improvement += 1
            else:
                state.rounds_without_improvement += 1

            result = RoundResult(
                index=i,
                started_at=started_at,
                finished_at=now_iso(),
                duration_s=duration,
                metric=metric,
                best_so_far=state.best_metric,
                improved=improved,
                head_sha_before=sha_before,
                head_sha_after=sha_after,
                cursor_exit_code=cursor_exit,
                log_dir=str(round_dir.relative_to(log_dir.parent)) if round_dir.exists() else "",
            )
            state.rounds.append(result)
            write_summary(summary_path, args, state, baseline)

            print(
                f"[round {i}] metric={metric} best={state.best_metric} improved={improved} "
                f"streak={state.rounds_without_improvement}/{args.patience} "
                f"duration={duration:.1f}s",
                flush=True,
            )

            # Periodic DoD checkpoint: every N rounds, run the slow 4-file
            # pytest gate to catch regressions the fast probe can't see.
            if args.dod_every > 0 and i % args.dod_every == 0:
                banner(f"DoD CHECKPOINT after round {i}")
                dod_log = round_dir / "dod.log"
                dod_score, dod_rc = run_dod_checkpoint(
                    args.dod_cmd, str(workspace), args.dod_timeout, dod_log
                )
                state.dod_checkpoints.append({
                    "after_round": i,
                    "sha": sha_after,
                    "score": dod_score,
                    "exit_code": dod_rc,
                    "log_path": str(dod_log.relative_to(log_dir.parent)),
                    "at": now_iso(),
                })
                if dod_score is not None and dod_score >= 0:
                    state.last_dod_score = dod_score
                    state.last_dod_sha = sha_after
                write_summary(summary_path, args, state, baseline)
                if dod_score is None or dod_score < 0:
                    banner(
                        f"EARLY-STOP: DoD checkpoint regressed (score={dod_score}, rc={dod_rc}) "
                        f"after round {i}. Last green DoD SHA = {state.last_dod_sha or '(never green)'}. "
                        f"Inspect {dod_log} for details."
                    )
                    return 0

            if state.rounds_without_improvement >= args.patience:
                banner(
                    f"EARLY-STOP: no improvement for {args.patience} consecutive rounds. "
                    f"Best metric={state.best_metric} at SHA {state.best_sha}."
                )
                break

        banner(
            f"AUTO-OPTIMIZE done | rounds_run={len(state.rounds)} | "
            f"baseline={baseline} | best={state.best_metric} | best_sha={state.best_sha}"
        )
        return 0
    except KeyboardInterrupt:
        banner("AUTO-OPTIMIZE interrupted by user (Ctrl+C)")
        return 130
    finally:
        write_summary(summary_path, args, state, baseline)
        restore_cli_config(cli_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
