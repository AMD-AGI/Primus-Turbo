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
    "**最终目标**（三段独立判定，每段都必须达成；看 metric stderr 的 "
    "`Goals:` 三行 PASS/FAIL）：\n"
    "  (1) **BF16_overall** ≥ 0.97 —— 含 dense (8 shape) + grouped (16 shape) "
    "对 Primus 默认 dispatch（BF16→HIPBLASLT, grouped BF16→AITER）的 TFLOPS "
    "geomean ratio。\n"
    "  (2) **FP8_forward**  ≥ 0.97 —— FP8 tensorwise dense forward 对 "
    "HIPBLASLT 的 TFLOPS geomean ratio (8 shape)。\n"
    "  (3) **FP8_backward** ≥ FP8_forward × 0.97 —— FP8 tensorwise dense "
    "backward (a_grad + b_grad 两次 GEMM) 隔离后的 TFLOPS geomean ratio，"
    "相对于 FP8_forward 不允许掉超过 3% (因 backward 比 forward 多一次 "
    "quant + 多一次 GEMM，路径更长，所以判定标准是相对的，不是绝对 0.97)。\n\n"
    "三段全 PASS 才算交付；overall metric score (= geomean × 1000) 只是 "
    "auto_optimize 的 ranking 信号，**不是验收标准**。\n\n"
    "**Baseline = 56** / 3 goals = 0 PASS（2026-04-28，cache-free dispatch + "
    "40-shape suite with FP8 backward）。这是把 hipkitten 模块所有 lru_cache / "
    "dict / weakref 全部清光后的诚实起点 —— 每次 forward 重 import + reparse "
    "autotune JSON 多吃 ~0.6 ms dispatch overhead，会盖过小 GEMM 的 kernel "
    "时间。这是规则的物理代价，不能用 cache 绕。\n"
    "现状 per-section breakdown:\n"
    "  • dense_BF16     (8)  geomean=0.321  ← dispatch overhead 把小 GEMM 拖死\n"
    "  • dense_FP8_fwd  (8)  geomean=0.366  ← 同上\n"
    "  • dense_FP8_bwd  (8)  geomean=0.015  ← **7/8 reject**：backward 走 "
    "grad_a (NN/RRR layout) + grad_b (TN/CRR layout)，跟 forward 的 RCR 不同；"
    "HK FP8 binding 当前 can_handle 不覆盖这些 layout\n"
    "  • grp_BF16      (16)  geomean=0.018  ← **14/16 reject**：HK grouped kernel "
    "缺 K=2048~2880 / M=2048 / N=5760,2880 的 autotune entry\n"
    "  • BF16_overall  (24)  geomean=0.047 (= dense+grouped, 被 grouped 拖塌)\n"
    "唯二跑通的 grouped case 是 DeepSeek-V3-GateUP-{B16,B32}-M4096（ratio "
    "1.06~1.13），说明 HK grouped kernel 本身没问题，缺的是 (a) `can_handle` "
    "覆盖更小的 K / 更小的 M-per-group / 不规则 N，(b) HipKittens 的 grouped "
    "autotune cache 扩到这些 shape。dense 端要在 kernel 侧/launch 侧把 dispatch "
    "路径变短（比如让 HK kernel 单 launch 自己做更多事），不能靠回退到任何 "
    "host-side cache。FP8 backward 端要扩 HK FP8 binding 的 layout 覆盖（RRR / "
    "CRR entry），同时让 backward 路径里的 quantize_fp8 不走 host sync。\n\n"
    "**两仓库分工**：\n"
    "  - /workspace/code/HipKittens —— 底层 kernel 仓：tile/wave layout/swizzle/MFMA、"
    "autotune cache、kernel launcher。改 .cpp 后进 analysis/{bf16,fp8}_gemm/mi350x/ "
    "或对应 grouped 目录跑 `source ../../../env.src && make -j` 重编 "
    "tk_*_layouts.so（Primus 自动 reload）。\n"
    "  - /workspace/code/Primus-Turbo —— dispatch 框架仓：写**通用规则**(`if K%128==0`、"
    "`if N>=K`)，不写形状表。扩 can_handle 覆盖、改 group_m / kernel 变体的启发式。\n\n"
    "**FROZEN（不可修改）文件清单**：\n"
    "  - scripts/_metric_hk_ratio.py —— metric 评分脚本，改它就是作弊\n"
    "  - scripts/auto_optimize.py / scripts/run_dod_metric.sh —— 调度器本身\n"
    "  - tests/pytorch/ops/test_*.py —— 不能加 skip / 删 parametrize / 调 SNR 阈值\n"
    "  - benchmark/ops/config.py —— shape 表的 ground truth\n"
    "  - /root/.cursor/skills/hipkittens-primus-turbo-backend/SKILL.md —— 上下文文档\n\n"
    "**严禁的『假优化』模式**（违反 = 本轮立即作废）：\n"
    "  ✗ case-by-case 形状表：`if (M,N,K)==(X,Y,Z): return cfg`（autotune .json 是数据，允许）\n"
    "  ✗ 收紧 can_handle 把难 shape 排除掉（geomean clip 0.01，分数立刻塌）\n"
    "  ✗ 只改 metric/test 文件让数字变好\n"
    "  ✗ 加 pytest.skip / 删 parametrize / 提高 SNR 阈值\n"
    "  ✗ **永远禁止 cache：dict / weakref / data_ptr / _version / LRU / TTL "
    "任何形式都不行（quant 输出 / preshuffle / group_offs / grid_x_hint / "
    "scale / autograd 中间产物 全在禁单里）**"
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
            f"gemm_fp8_impl.GEMMFP8HipKittenBackend、cache 文件读取等），快 metric 通常足够。\n"
            f"  - 如果你触及任何**共用代码**（autograd 入口、dispatcher、quantize_fp8_*、"
            f"grouped_gemm.py 顶层、torch.library custom_op 注册等），你**必须**怀疑会影响"
            f"非 HIPKITTEN 后端，主动跑一次 `{args.dod_cmd}`（约 5-10 分钟）确认 0 failed 后再 commit；"
            f"否则脚本下次 checkpoint 会因为你的改动 EARLY-STOP，整个 run 报废。\n"
            f"  - {last_dod_line}\n"
        )

    return f"""你是 HipKittens × Primus-Turbo 联合优化协作者。本次是第 {round_idx} / {args.rounds} 轮，由脚本自动调度。
你**同时拥有两个仓库的写权限**：
  • Primus-Turbo: {args.workspace} (本轮工作目录、metric 在这里跑)
  • HipKittens : /workspace/code/HipKittens  (kernel 源码、autotune cache 在这里)
当前 Primus-Turbo git HEAD: {head_sha}

【强制第一步】
请先用读文件工具完整读取这份本地 skill：
  {args.skill_path}
里面有 HipKittens + Primus-Turbo 集成的所有上下文（路径、env、cache 结构、白名单、坑）。读完再决定本轮该做什么。

【优化目标】
{args.task}

【本轮的快速验收命令】
metric 命令（约 10 秒、单 GPU、自动选空闲卡）：
  {args.metric_cmd}
含义：在 16 个 LLM 典型 dense shape 上跑 HIPKITTEN vs Primus 默认 dispatch 的 TFLOPS 实测，
score = int(geomean(hk_tflops / ref_tflops) * 1000)。target ≥ 900 (= 90%)。
HIPKITTEN reject 一个 shape → 该 shape ratio 被 clip 到 0.01 → geomean 大幅下跌，
所以**收紧 can_handle 反而会扣分**，必须靠真功夫扩覆盖 + 提速。

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

【真优化方向 — 任选一个具体小目标】（"严禁假优化"清单已在【优化目标】里给出，请重读）
  ✓ HipKittens kernel 改写：例如 /workspace/code/HipKittens/analysis/fp8_gemm/mi350x/
    kernel_fp8_layouts.cpp 的 K-step / wave grid / swizzle / MFMA 排布。
    改完进 analysis/{{bf16,fp8}}_gemm/mi350x 跑 `source ../../../env.src && make`，
    会在原地生成 tk_{{bf16,fp8}}_layouts.so —— Primus 自动加载新 .so，无需 rebuild Primus。
    最常见的提速来源：fix bank conflict、提升 ds_read 吞吐、调 K_STEP 更好覆盖 K=128/256 倍数。
  ✓ HipKittens autotune 扩 cache：跑 bench_bf16_vs_torch.py / autotune.py 把缺的 (M,N,K)
    填进 bench_*.json / .autotune_cache.json（数据不是分支，允许）。注意只跑 LLM 典型 shape，
    别把 cache 撑爆。
  ✓ HipKittens kernel 接受更宽的 K：当前 FP8 kernel 模板化 K，导致 K=4096/8192/14336 走得通、
    K=11008/53248 走不通。改 launcher 让 launcher 在运行时切换 K-block，能直接增加 can_handle 命中。
  ✓ Primus 规则式 can_handle/dispatch：例如把 "_can_use_hipkitten_kernel" 的 K%TILE_K 限制改为
    K%128==0（如果 kernel 支持），或对未在 cache 中的 shape 用 default group_m=4, xcd=8 兜底
    （**通用规则**，不是查表）。
  ✓ Primus 改 _hipkitten_fp8_group_m / _hipkitten_grouped_cfg 的 fallback 启发式（用形状特征
    例如 N>=K 走 group_m=2, otherwise group_m=4 之类**规则**，不是 if shape==X return 2）。

【典型流程示例（FP8 提速）】
1. 跑 metric 看哪 8 个 FP8 shape ratio < 0.9（stderr 表里有）
2. 选其中 1 个 shape (例如 4096x4096x4096) 进 HipKittens 仓库直接 bench 那个 kernel：
   ```bash
   cd /workspace/code/HipKittens/analysis/fp8_gemm/mi350x
   source ../../../env.src
   python3 bench_vs_hipblaslt.py  # 看 group_m 是不是没扫到 32
   ```
3. 想清楚瓶颈（例如 4-wave kernel 在 N<8192 时 wave fan-out 不够），动 cpp:
   ```bash
   $EDITOR kernel_fp8_layouts.cpp
   make -j  # 重编 tk_fp8_layouts.so
   ```
4. 回 Primus 跑 metric。提升了就 commit 在 HipKittens 仓库（feat/perf:），同时也 commit
   在 Primus-Turbo 仓库（哪怕只是空 commit "perf(hk): trigger HK fp8 kernel rebench"，
   方便 auto_optimize 跟踪历史）。

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
