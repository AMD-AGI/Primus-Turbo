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
    "  (1) **BF16_fwd vs HIPBLASLT** ≥ 0.97 —— dense BF16 GEMM forward (8 shape) [w=1]\n"
    "  (2) **BF16_bwd vs HIPBLASLT** ≥ 0.97 —— dense BF16 GEMM backward (8 shape) [w=1]\n"
    "  (3) **FP8_fwd  vs HIPBLASLT** ≥ 0.97 —— dense FP8 tensorwise GEMM forward (8 shape) [w=2]\n"
    "  (4) **FP8_bwd  vs TRITON**    ≥ 1.20 —— dense FP8 tensorwise GEMM backward (8 shape) [w=2]\n"
    "  (5) **grp_BF16 vs TRITON**    ≥ 1.20 —— grouped BF16 GEMM (16 shape, DeepSeek-V3 + gpt_oss_20B) [w=4]\n"
    "  (6) **grp_FP8  vs TRITON**    ≥ 1.20 —— grouped FP8 tensorwise GEMM (同 16 shape) [w=4]\n\n"
    "**Score = 1000 × weighted-geomean(min(progress_i / target_i, 1.0))**，权重\n"
    "`BF16fwd:1 BF16bwd:1 FP8fwd:2 FP8bwd:2 grpBF16:4 grpFP8:4`（sum=14）。\n"
    "解读：\n"
    "  • 每段 progress 上限锁 1.0：把 BF16_fwd 从 0.97 推到 1.05 **不加分**；要花\n"
    "    cycle 在还没 PASS 的段上才有 score 收益。\n"
    "  • grp_BF16 / grp_FP8 权重 4 ——把 grouped 从 0.5 推到 1.0 比把 BF16_fwd 从 0.95 推到 1.0\n"
    "    多 4× 分。FP8 dense 权重 2，BF16 dense 权重 1。这条权重表是 user 拍板的\n"
    "    优先级体现 —— **挑分最大的方向打**，不要均匀分配 cycles。\n"
    "  • 6 段全 PASS（每段 progress=1.0）= score 1000；任何段 progress<1 ⇒ score<1000。\n"
    "  • Reject 把 ratio 钉到 0.01：grp_FP8 中 1/16 reject = 该段 geomean 大跌 → 段权重 4×\n"
    "    放大冲击 ⇒ score 直接跌穿。所以**收紧 can_handle 在新打分里更亏**。\n"
    "6 段全 PASS 才算交付；score 只是 ranking 信号，**不是验收标准**。\n\n"
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
    "**起点（架构第一/二刀完成后的近期实测，权重表上线后会偏低）**：\n"
    "  • BF16_fwd  (8)   geomean ≈ 0.94-1.05 vs HIPBLASLT  [w=1] ← 容易 PASS，progress 已 cap 1.0\n"
    "  • BF16_bwd  (8)   geomean ≈ 0.74      vs HIPBLASLT  [w=1] ← bwd 走 3 次 dispatch，host overhead\n"
    "  • FP8_fwd   (8)   geomean ≈ 0.93      vs HIPBLASLT  [w=2] ← 离 0.97 一步之遥\n"
    "  • FP8_bwd   (8)   geomean ≈ 1.07      vs TRITON     [w=2] ← 距 1.20 还有 12% gap\n"
    "  • grp_BF16 (16)   geomean ≈ 0.10-0.40 vs TRITON     [w=4] ← gpt_oss_20B (N=K=2880) 不对齐\n"
    "  • grp_FP8  (16)   geomean ≈ 0.05-0.10 vs TRITON     [w=4] ← HK FP8 .so 缺 grouped binding\n"
    "**权重 4× 决定攻坚顺序**：grp_FP8 + grp_BF16 是 score 大头（合占 8/14 = 57%）。\n"
    "**唯一允许的路径（user 拍板）**：HK 仓库新加 **persistent + CPU-sync-free** grouped\n"
    "kernel（mirror Triton 的 `grouped_gemm_kernel.py`，详见第四+五刀），**不许** multi-stream\n"
    "/ per-group launch / pad-and-copy 凑数。pad-and-copy 在非对齐 N/K 上能挤出局部分数但**永远**\n"
    "无法跨过 1.20 vs TRITON 的目标 —— Triton 是 single-launch persistent，per-group launch\n"
    "永远输给它。所以两段必须**一并**用 persistent kernel 解。建议路径：\n"
    "  ① HK BF16 grouped binding 重写为 persistent + accept group_offs（轮 N）→ grp_BF16 +28%\n"
    "  ② HK FP8 grouped binding 新加（同款 persistent，mirror BF16）（轮 N+1）→ grp_FP8 +28%\n"
    "  ③ FP8_fwd / FP8_bwd 凑零头 → +14%\n"
    "  ④ BF16_bwd 最后做（dispatch overhead 可能 1 次重写解决多段）→ +7%\n"
    "新架构下 reject 全部源于**真硬约束**（dtype / layout / tile alignment），不再有 cache miss 类\n"
    "假 reject。**收紧 can_handle 反而扣分**：reject → ratio clip 0.01 → 整段 geomean 塌掉，\n"
    "在 grp_BF16/grp_FP8 (w=4) 段上一次 reject 就能让 score 掉 100+ 分。\n\n"
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
    "  ✗ **grouped 路径严禁 multi-stream / per-group launch / cudaStream 池**（user 拍板）：\n"
    "    1) **禁止**任何形式的 `for g in range(B): launch_dense_kernel(...)` 一组一发，\n"
    "       哪怕是分摊到 `torch.cuda.Stream()` 池里『并行』也不准。这是当前 Primus 端\n"
    "       `GroupedGEMMFP8HipKittenBackend.execute` 的 fallback 路径，必须**整体替换**\n"
    "       为单次 launch 的 persistent kernel（见第五刀）。MI355X 有 256 个 CU + 8 个 XCD，\n"
    "       一个 persistent kernel 单次 launch 用满 256 个 program 就能 saturate；切多 stream\n"
    "       既增加 host overhead 又破坏 XCD chiplet 调度。\n"
    "    2) **禁止**给 HK kernel binding 加 `stream` 参数 / `hipStream_t` 透传 / device-side\n"
    "       stream pool —— 这是 multi-stream 的前置铺垫。HK 仓库 BF16/FP8 binding 只接受\n"
    "       默认流（hipStream_t = 0）；如果 Primus 在非默认流上调用，是 Primus 端的责任在\n"
    "       binding 调用前后做 stream sync（不是把 stream 塞进 binding 签名）。\n"
    "  ✗ **grouped 路径严禁 CPU sync**（user 拍板）：\n"
    "    1) **禁止**任何形式的 host-side 读 `group_lens` / `group_offs`：`.item()` /\n"
    "       `.tolist()` / `.cpu()` / `int(t)` / `cudaMemcpyDeviceToHost` 全在禁单。哪怕是\n"
    "       为了拿 `B = group_lens.shape[0]` 也不许（B 直接从 b.shape[0] 静态拿；其他元数据\n"
    "       全在 device tensor 里）。\n"
    "    2) **禁止** `group_offs.cumsum()` 跑在 CPU 上 —— prefix sum 必须 device tensor 输入\n"
    "       device tensor 输出（`torch.cumsum(group_lens.cuda(), dim=0)`）。\n"
    "    3) Persistent kernel 必须在 GPU 内自己消费 `group_offs_ptr` —— 用 `tl.load` /\n"
    "       device pointer arithmetic 读 prefix sum，做 group-id 的 O(G) linear scan 或\n"
    "       O(log G) binary search。**参考 `primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py`\n"
    "       的 `_grouped_bf16_persistent_gemm_kernel` (line 182-330)** —— Triton 里同样的\n"
    "       `_g in range(G): tl.load(group_offs_ptr + _g)` 模式，HK 仓库要 mirror。\n"
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
    # Cross-round agent session reuse — see --reuse-chat-window-secs.
    # ``chat_id`` is a dashed UUID (cursor-agent's session_id) we pass to
    # ``--resume`` so the agent retains tool history / scratch reasoning
    # / file reads across rounds. ``chat_started_at`` is monotonic time
    # when the chat was originally opened; once that's older than the
    # window we drop the chat_id and let the next round start fresh.
    chat_id: Optional[str] = None
    chat_started_at: Optional[float] = None
    chat_round_count: int = 0


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
        default=60 * 60,
        help="Seconds to allow each cursor-agent invocation before killing it (default 60 min).",
    )
    p.add_argument(
        "--reuse-chat-window-secs",
        type=int,
        default=60 * 90,
        help=(
            "Reuse the same cursor-agent chat session (via --resume <session_id>) "
            "across rounds for this many seconds before starting a fresh chat. "
            "Default 5400s = 1.5h. The agent keeps its tool history, scratch "
            "reasoning, and file reads across rounds in the window — so a "
            "multi-round task (write a HK .cpp kernel → compile → numerical "
            "probe → metric → tweak → recompile) doesn't have to re-onboard "
            "from SKILL.md / git log every round. After the window the chat "
            "is dropped and the next round starts fresh (full prompt)."
        ),
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


def _pick_idle_gpu(pool_csv: str) -> Optional[str]:
    """Pick the smallest idle GPU id from ``pool_csv`` (e.g. "0,2,3").

    Same algorithm as scripts/_metric_hk_ratio.py::_pick_idle_gpu — a GPU
    counts as busy if a KFD process holds > 100MB of VRAM on it OR its
    rocm-smi `GPU use (%)` is > 30. The 30% threshold catches GPUs that
    are pinned at 100% by other containers / non-KFD processes (which
    show up in --showuse but with VRAM=0 in --showpids).

    Returns the chosen id as a string ("2"), or None if rocm-smi is
    unavailable AND the pool is empty / unparseable.

    The auto_optimize loop calls this once per round and pins
    ``HIP_VISIBLE_DEVICES`` into every cursor-agent / metric subprocess
    so the agent literally cannot see any other GPU — no matter what
    it does in shell (``HIP_VISIBLE_DEVICES=0`` from inside the agent
    just remaps to the one visible card we gave it).
    """
    import re
    VRAM_THR = 100 * 1024 * 1024
    USE_PCT_THR = 30
    pool: Optional[set[int]] = None
    if pool_csv.strip():
        pool = set()
        for tok in pool_csv.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                pool.add(int(tok))
            except ValueError:
                pass
        if not pool:
            pool = None
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showpids"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except Exception:
        if pool:
            return str(min(pool))
        return None
    all_gpus = sorted({int(m) for m in re.findall(r"^GPU\[(\d+)\]", out, flags=re.M)})
    if pool is not None:
        all_gpus = [g for g in all_gpus if g in pool]
    busy: set[int] = set()
    for m in re.finditer(
        r"^GPU\[(\d+)\]\s*:\s*GPU use \(%\):\s*(\d+)", out, flags=re.M,
    ):
        gid, pct = int(m.group(1)), int(m.group(2))
        if pct > USE_PCT_THR:
            busy.add(gid)
    in_kfd = False
    for line in out.splitlines():
        if "KFD process information" in line:
            in_kfd = True
            continue
        if not in_kfd:
            continue
        if line.startswith("=") or "PROCESS NAME" in line:
            continue
        cols = line.split()
        if len(cols) < 4 or not cols[0].isdigit():
            continue
        try:
            vram = int(cols[3])
        except ValueError:
            continue
        if vram <= VRAM_THR:
            continue
        for gid in re.findall(r"\d+", cols[2]):
            busy.add(int(gid))
    idle = [g for g in all_gpus if g not in busy]
    if idle:
        return str(idle[0])
    return str(all_gpus[0]) if all_gpus else None


def _subprocess_env_with_pinned_gpu(pool_csv: str) -> tuple[dict, Optional[str]]:
    """Build a subprocess env dict with HIP_VISIBLE_DEVICES pinned to an
    idle GPU from ``pool_csv``.

    Returns (env_dict, picked_id). When ``picked_id`` is None we fall
    through to the parent's env unchanged (pool was empty or rocm-smi
    failed) — the called script can still do its own selection.

    HIPKITTEN_GPU_POOL is also forwarded so child scripts (e.g.
    _metric_hk_ratio.py) keep their pool semantics.
    """
    env = os.environ.copy()
    if pool_csv.strip():
        env["HIPKITTEN_GPU_POOL"] = pool_csv
    picked = _pick_idle_gpu(pool_csv)
    if picked is not None:
        env["HIP_VISIBLE_DEVICES"] = picked
    return env, picked


def run_metric(cmd: str, cwd: str, timeout: int, gpu_pool: str = "") -> Optional[float]:
    section(f"measuring metric: {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    env, picked = _subprocess_env_with_pinned_gpu(gpu_pool)
    if picked is not None:
        print(f"[metric] pinned HIP_VISIBLE_DEVICES={picked}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
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


def _build_resume_prompt(
    args: argparse.Namespace,
    state: TrajectoryState,
    round_idx: int,
    baseline_metric: Optional[float],
    head_sha: str,
    recent_log: str,
    short_status: str,
    pinned_gpu: Optional[str] = None,
) -> str:
    """Lightweight follow-up prompt — same chat as previous round.

    Assumes the agent already has the task body / skill / FROZEN list /
    phased roadmap / commit policy in context. Only delivers the
    per-round delta. Roughly 1/8 the length of the cold-start prompt.
    """
    last = state.rounds[-1] if state.rounds else None
    last_metric = last.metric if last else baseline_metric
    last_improved = "是" if (last and last.improved) else "否"

    history_lines = []
    for r in state.rounds[-3:]:  # 接续模式只回看最近 3 轮 — 完整历史 agent 自己记得
        history_lines.append(
            f"  - 第 {r.index} 轮: metric={r.metric}, best={r.best_so_far}, "
            f"improved={r.improved}, sha {r.head_sha_before[:8]}->{r.head_sha_after[:8]}"
        )
    history_block = "\n".join(history_lines) if history_lines else "  (尚无历史)"

    chat_age_min = 0.0
    if state.chat_started_at is not None:
        chat_age_min = (time.monotonic() - state.chat_started_at) / 60.0

    if pinned_gpu is not None:
        gpu_line = (
            f"- **本轮 GPU**: HIP_VISIBLE_DEVICES={pinned_gpu}（auto_optimize.py 已 pin；"
            f"shell 里不要再写 `HIP_VISIBLE_DEVICES=N`，写了也只能用这张）。"
        )
    else:
        gpu_line = ""

    dod_line = ""
    if args.dod_every > 0 and state.last_dod_score is not None:
        dod_line = (
            f"上次 DoD={state.last_dod_score} (sha {(state.last_dod_sha or '')[:8]}). "
            f"提醒：每 {args.dod_every} 轮自动跑一次，failed > 0 立刻 EARLY-STOP。"
        )

    return f"""【第 {round_idx} / {args.rounds} 轮 — 接续上一轮 chat】
本 chat session 已运行 {state.chat_round_count + 1} 轮 / {chat_age_min:.0f} 分钟（窗口上限
{args.reuse_chat_window_secs / 60:.0f} 分钟，超过会换新 chat）。你已知道 task / 6 段目标 /
FROZEN 列表 / phased 路线 / 严禁假优化清单 —— **不要重新读 SKILL.md，不要重 quote 上面这些**，
直接干活。如果你忘了某条规则，自己回去翻本 chat 历史，不要再让用户复述。

【本轮数据增量】
- Primus-Turbo HEAD: {head_sha}
- 历史最佳 metric = {state.best_metric}
- 上一轮 metric = {last_metric}（improved={last_improved}）
- 已连续 {state.rounds_without_improvement} 轮未提升（patience={args.patience}）
- baseline (开始时) = {baseline_metric}
{gpu_line}
{('- ' + dod_line) if dod_line else ''}

【近 3 轮】
{history_block}

【近期 git log】
{recent_log or "(空)"}

【working tree 状态】
{short_status or "(干净)"}

【本轮指令】
1. 第一步：跑 metric `{args.metric_cmd} 2>&1 | tee /tmp/metric_round_{round_idx}.log`，
   看 stderr 表里 ratio 最低 + 权重最高的 shape (grp_FP8 / grp_BF16 weight=4，
   FP8_fwd / FP8_bwd weight=2，BF16_fwd / BF16_bwd weight=1). 选 1 个攻坚目标。
2. 你**记得**上一轮在做什么 —— 如果是跨轮长任务（写 HK .cpp kernel / 编译 / 数值 probe），
   continue 那条主线，不要换方向。如果上一轮已 commit 完整工作，再选新目标。
3. 改完跑 metric 验证，再 commit。每仓库最多 1 commit/轮。
4. 末尾 markdown 小结：本轮目标 / 改了什么 / before-after metric / commit SHA / 下一轮建议。

{args.prompt_extra}"""


def build_prompt(
    args: argparse.Namespace,
    state: TrajectoryState,
    round_idx: int,
    baseline_metric: Optional[float],
    head_sha: str,
    recent_log: str,
    short_status: str,
    is_resume: bool = False,
    pinned_gpu: Optional[str] = None,
) -> str:
    """Build the per-round prompt fed to cursor-agent.

    The prompt is in Chinese to match the user's preferred working language.

    When ``is_resume=False`` (first round of a chat-window): includes the
    full task body + skill-read instruction + FROZEN list + phased
    roadmap + commit policy + output requirements. ~200 line cold-start
    prompt.

    When ``is_resume=True`` (subsequent rounds within the same chat
    window): ships only the *delta* — round counter, current HEAD SHA,
    last metric, best metric, working-tree status, recent git log.
    Skips the task body / skill / roadmap / FROZEN list because the
    agent has already seen all of them in the same chat session and
    keeping ~10K-token boilerplate every round both burns context and
    makes the user's actual delta hard for the model to find.

    The boundary between the two modes is ``--reuse-chat-window-secs``;
    once a chat exceeds that wall age the loop drops ``state.chat_id``
    and the next round starts a fresh cold-start chat.
    """
    if is_resume:
        return _build_resume_prompt(
            args, state, round_idx, baseline_metric, head_sha, recent_log, short_status,
            pinned_gpu=pinned_gpu,
        )
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
    if pinned_gpu is not None:
        pinned_gpu_block = (
            f"  **本轮已为你 pin** `HIP_VISIBLE_DEVICES={pinned_gpu}`（auto_optimize.py 在每轮开始前\n"
            f"  从 `HIPKITTEN_GPU_POOL={gpu_pool}` 里挑了一张 use% < 30 且无 KFD VRAM 占用的 idle GPU\n"
            f"  并通过 subprocess env 传进 cursor-agent 进程）。在本进程下 ROCm 只能看到这 1 张物理\n"
            f"  卡，shell 命令里**不要再写** `HIP_VISIBLE_DEVICES=N`：写了也没用（你看到的 device 0\n"
            f"  就是这张 pin 好的卡，不是物理 GPU 0）。**绝不许**手动 export `HIP_VISIBLE_DEVICES`。"
        )
    else:
        pinned_gpu_block = (
            f"  rocm-smi 不可用或 pool 为空 — 没 pin。**跑前必须** `rocm-smi --showuse --showpids`\n"
            f"  在 pool 里挑 GPU use% < 30 且 KFD VRAM ≈ 0 的卡，metric 脚本会自动选；不要硬编码\n"
            f"  `HIP_VISIBLE_DEVICES=N`。"
        )

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
  ☐ 第四刀 + 第五刀（user 拍板，**唯一允许的 grouped 实现路径**）：HK 仓库新加
    **persistent + CPU-sync-free** grouped kernel binding（BF16 + FP8 各 3 个 layout）。
    硬件参数：MI355X = **256 个 CU + 8 个 XCD**（per_xcd = 32 CU），grid_size = 256 个 program。
    **样板**：完全照抄 `primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py::_grouped_bf16_persistent_gemm_kernel`
    (line 182-330) 的设计 —— **本轮第一步先把这个文件读完**，理解 5 个核心点：
      (i)  Grid = NUM_SMS（= 256 在 MI355X 上）；kernel 是 **persistent**：单次 launch、program 内
           跨多 group × 多 tile 滚动 (`for global_tile_id in range(pid, total_tiles, NUM_SMS)`)。
      (ii) `group_offs_ptr` 是 device int64 tensor (shape `[G+1]`，prefix sum of group_lens)。
           Kernel 内**唯一**的 group 元数据来源是 `tl.load(group_offs_ptr + g)`。**没有任何 host
           回读** —— host 只 launch 一次，剩下全在 device 上跑。
      (iii) 每个 program 进 kernel 后做 O(G) linear scan 累加每组 tile 数得到 total_tiles +
           映射 global_tile_id → (group_idx, local_tile)。G ≤ 32 时这个 scan 在 L1/L2 命中
           后基本免费 (line 230-255)。
      (iv) Group-local tile → (pid_m, pid_n) 用跟 dense GEMM 一样的 GROUP_SIZE_M swizzle
           (line 263-268)；K-loop 跟 dense 完全一致。
      (v)  Chiplet (XCD) 重排：`_chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS=8, CHUNK_SIZE)`
           (line 152-165) 把 PID 重排让相邻 program 走相邻 XCD，提升 L2 reuse。MI355X
           NUM_XCDS=8。
    **实施分两轮**（必须按这个顺序，单 chat session 内连续推进）：
    轮 N (HK BF16 grouped 重写)：
      1. 在 `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/` 找到现有 BF16 grouped
         launcher（mirror Triton 的 BF16 persistent 实现），把 binding 改成接受
         `(a, b, c, group_offs)` —— 其中 group_offs 是 `[G+1] int64` device tensor，**绝不**
         接受任何 group_lens / 假设 uniform-M / 把 group 信息从 host 传进来。binding 命名
         `grouped_rcr` / `grouped_rrr` / `grouped_crr`（无 `_balanced` 后缀）。
      2. Primus 端 `GroupedGEMMHipKittenBackend.execute`：删 `_uniform_group_m` fast-path、
         删 padded uniform-M 分支、删 per-group `dense_run` fallback，**全部**走单次
         `hipkitten.grouped_run(hk, cfg, a, b, out, group_offs)` 调用；group_offs 用
         `torch.cumsum(group_lens, dim=0)` 算（device → device，无 sync）。
      3. 数值 probe：fp32 reference vs 新 kernel，max_abs + SNR；commit。
    轮 N+1 (HK FP8 grouped 新加)：
      1. 在 `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` 末尾
         新加 3 条 `m.def("grouped_<layout>", ...)` binding（rcr/rrr/crr），实现 mirror 上轮
         BF16 grouped persistent launcher 的结构：grid=256, group_offs device 输入, single
         launch 跑完所有 group × 所有 tile, FP8 quantize/dequantize 用现有 dense 路径的逻辑。
         **绝对不准**给 binding 加 `stream` / `hipStream_t` 参数；**绝对不准**借机引入
         multi-stream pool。如果上轮 commit 里看到任何 `stream` 参数 / multi-stream 准备工作，
         **revert 之**。
      2. `source ../../../env.src && make -j` 重编，确认 `dir(tk_fp8_layouts)` 有 `grouped_crr/rcr/rrr`。
      3. Primus 端 `GroupedGEMMFP8HipKittenBackend.execute`：删 per-group `for` 循环 +
         per-stream / per-CudaStream 调度，**全部**改成单次 `hipkitten.grouped_run(...)`。
      4. 数值 probe + metric verify + commit。
    **明确禁止**（违反 = 本轮立即作废，且整个 commit 必须 revert）：
      ✗ multi-stream / `torch.cuda.Stream()` 池 / per-group hipStream_t / device-side stream
      ✗ host-side 读 group_lens / group_offs（`.item()` / `.tolist()` / `.cpu()` / `int(...)`）
      ✗ `for g in range(B): launch_dense_kernel(...)` 一组一发的 fallback（per-group launch）
      ✗ 给 HK binding 加 `stream` / `hipStream_t` 参数（即便 default=0 也禁）
      ✗ 假设 uniform M-per-group / can_handle 判 balanced
      ✗ runtime 读任何 .json / .pkl / .autotune_cache.json
      ✗ 加 dict/lru_cache/weakref/data_ptr cache 任何形式
    Acceptance：grp_BF16 / grp_FP8 两段同时 PASS（>= 1.20 vs TRITON），且 `git grep` 在两个
    仓库里**找不到** `cudaStream` / `hipStream_t` / `torch.cuda.Stream` / `.item()` 在
    grouped 路径上的新增使用。
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

【**chat-window 跨轮接续**（agent 必读）】
本 auto_optimize run 启用 `--reuse-chat-window-secs 5400`（默认 1.5 小时）。意思是：
  • 第一轮 cold-start 一个新 cursor-agent chat session，记下 session_id；
  • 后续轮如果**距 chat 起点 < 1.5 小时**，下一轮直接 `--resume <session_id>`
    （**接续相同对话**，agent 保留所有 file reads / tool history / scratch 推理）；
  • 一旦超过 1.5 小时窗口，丢掉 chat_id，下一轮 cold-start 新 chat。
对你（agent）的影响：
  ✓ 跨轮的**长任务**（写 HK .cpp → 编译 → 数值 probe → metric → 调 cfg → 重编 → ...）
    可以**自然跨多轮**完成 —— 你**记得**自己上一轮试过什么、卡在哪一步。不要换方向。
  ✓ resume 模式下 prompt 极短，**不再重复** task body / SKILL.md / FROZEN list / 路线 ——
    这些你都已经在本 chat 里见过。**禁止再去读 SKILL.md** 或要求用户复述规则；忘了
    自己回去翻本 chat history。
  ✓ chat 滚动到第 ~10-15 轮时窗口会过期，那一轮你会看到完整 cold-start prompt ——
    这是**新 chat**，之前的 file reads 都不再可见，你必须**从 commit log 推断**之前进展。
  ✗ 不要在 resume 轮里"重做"已 commit 的工作 —— 看 git log 确认上一轮是否 commit；
    如果是，**接着推进**，不是重做。

【脚本机制硬约束 - 不可违反】
- **GPU 池**：本次 run 只允许使用 `HIPKITTEN_GPU_POOL={gpu_pool}`（已经写进环境变量；GPU 1
  当前被其他作业占用，绝不许动）。
{pinned_gpu_block}
  你**只**能在 pin 给你的这一张卡上跑任何 benchmark / probe / metric。判 idle 的标准：
  rocm-smi `GPU use (%)` ≤ 30 **且** KFD process VRAM ≈ 0 — 光看 KFD list 不够，有些
  workload 跑在 100% 但 KFD VRAM=0（这就是上一次手动选 GPU 0 撞上正在跑的卡的原因）。
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


def _format_tool_call_summary(ev: dict) -> Optional[str]:
    """Render a one-line summary of a stream-json tool_call event."""
    sub = ev.get("subtype")
    if sub != "started":
        return None  # only show on start; completed lines are noisy
    tc = ev.get("tool_call") or {}
    # tool_call is a dict whose first key encodes the tool family.
    tool_kind = next(iter(tc.keys()), "tool")
    inner = tc.get(tool_kind) or {}
    args_obj = inner.get("args") or {}
    desc = inner.get("description") or args_obj.get("description") or ""
    if tool_kind == "shellToolCall":
        cmd = (args_obj.get("command") or "").splitlines()[0][:160]
        return f"  [tool] shell: {cmd}{' — ' + desc if desc else ''}"
    if tool_kind == "readToolCall":
        return f"  [tool] read {args_obj.get('path', '?')[:160]}"
    if tool_kind == "editToolCall":
        return f"  [tool] edit {args_obj.get('path', '?')[:160]}"
    if tool_kind == "writeToolCall":
        return f"  [tool] write {args_obj.get('path', '?')[:160]}"
    if tool_kind == "globToolCall":
        return f"  [tool] glob {args_obj.get('globPattern', '?')[:160]}"
    if tool_kind == "grepToolCall":
        return f"  [tool] grep {args_obj.get('pattern', '?')[:160]}"
    return f"  [tool] {tool_kind}{(' — ' + desc) if desc else ''}"


def run_cursor_round(
    args: argparse.Namespace,
    prompt: str,
    log_dir: Path,
    resume_chat_id: Optional[str] = None,
    pinned_gpu: Optional[str] = None,
) -> tuple[int, Optional[str]]:
    """Run one cursor-agent round.

    Returns ``(exit_code, session_id)`` where ``session_id`` is the
    dashed UUID emitted on the very first ``system/init`` event. Caller
    persists it across rounds so the next round can ``--resume`` instead
    of cold-starting a brand-new chat.

    Output is piped via ``--output-format stream-json`` so each line is
    a JSON event (init / user / assistant / tool_call / result). The
    raw event stream is logged to ``cursor.log``; the human-readable
    text content is mirrored to stdout for live monitoring.
    """
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
        "stream-json",
    ]
    if resume_chat_id:
        cmd += ["--resume", resume_chat_id]
    cmd.append(prompt)

    pretty_cmd = " ".join(shlex.quote(c) for c in cmd[:-1])
    if resume_chat_id:
        print(
            f"[cursor] resuming chat {resume_chat_id[:8]}…: {pretty_cmd} <prompt>",
            flush=True,
        )
    else:
        print(f"[cursor] new chat: {pretty_cmd} <prompt>", flush=True)

    log_path = log_dir / "cursor.log"
    raw_path = log_dir / "cursor.jsonl"
    captured_session_id: Optional[str] = None
    last_assistant_text = ""

    child_env = os.environ.copy()
    if pinned_gpu is not None:
        child_env["HIP_VISIBLE_DEVICES"] = pinned_gpu

    with log_path.open("w") as logf, raw_path.open("w") as rawf:
        logf.write(f"# Command: {pretty_cmd} <prompt>\n")
        logf.write(f"# Resume chat_id: {resume_chat_id or '(none — fresh chat)'}\n")
        logf.write(f"# Pinned GPU: HIP_VISIBLE_DEVICES={pinned_gpu or '(unset — agent picks)'}\n")
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
                env=child_env,
            )
        except FileNotFoundError:
            print("[cursor] cursor-agent not on PATH - aborting.", flush=True)
            return 127, None
        try:
            assert proc.stdout is not None
            start = time.monotonic()
            for line in proc.stdout:
                rawf.write(line)
                rawf.flush()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    ev = json.loads(stripped)
                except json.JSONDecodeError:
                    # Non-JSON line (e.g. cursor-agent error trace) — passthrough.
                    logf.write(line)
                    logf.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                ev_type = ev.get("type")
                if ev_type == "system" and ev.get("subtype") == "init":
                    sid = ev.get("session_id")
                    if sid and not captured_session_id:
                        captured_session_id = sid
                        msg = f"  [cursor] session_id={sid}"
                        logf.write(msg + "\n")
                        sys.stdout.write(msg + "\n")
                        sys.stdout.flush()
                elif ev_type == "assistant":
                    for blk in ev.get("message", {}).get("content", []) or []:
                        if blk.get("type") == "text":
                            txt = blk.get("text", "")
                            last_assistant_text = txt
                            logf.write(txt + "\n")
                            sys.stdout.write(txt + "\n")
                            sys.stdout.flush()
                elif ev_type == "thinking":
                    # Thinking deltas only go to log (too noisy for stdout).
                    delta = ev.get("text", "")
                    if delta:
                        logf.write(delta)
                        logf.flush()
                elif ev_type == "tool_call":
                    summary = _format_tool_call_summary(ev)
                    if summary:
                        logf.write(summary + "\n")
                        sys.stdout.write(summary + "\n")
                        sys.stdout.flush()
                elif ev_type == "result":
                    sid = ev.get("session_id")
                    if sid:
                        captured_session_id = sid  # final overrides init for safety
                    dur = ev.get("duration_ms", 0)
                    msg = f"  [cursor] result: duration={dur}ms is_error={ev.get('is_error', False)}"
                    logf.write(msg + "\n")
                    sys.stdout.write(msg + "\n")
                    sys.stdout.flush()
                logf.flush()
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
                    return (
                        proc.returncode if proc.returncode is not None else 124,
                        captured_session_id,
                    )
            proc.wait()
        except KeyboardInterrupt:
            print("\n[cursor] interrupted by user; killing cursor-agent", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        rc = proc.returncode if proc.returncode is not None else 1
        return rc, captured_session_id


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
        baseline = run_metric(
            args.metric_cmd, str(workspace), args.metric_timeout,
            gpu_pool=args.gpu_pool,
        )
        state.best_metric = baseline
        state.best_sha = get_head_sha(str(workspace))
        write_summary(summary_path, args, state, baseline)

        for i in range(1, args.rounds + 1):
            # Decide whether to resume the previous chat or cold-start a new
            # one. Cold-start when (a) no chat yet, (b) last cursor invocation
            # didn't return a session_id (failure / signaled before init), or
            # (c) the chat window has elapsed.
            now = time.monotonic()
            chat_age_secs = (
                (now - state.chat_started_at)
                if state.chat_started_at is not None
                else None
            )
            cold_start = (
                state.chat_id is None
                or chat_age_secs is None
                or chat_age_secs >= args.reuse_chat_window_secs
            )
            if cold_start and state.chat_id is not None:
                age_min = (chat_age_secs or 0) / 60.0
                print(
                    f"[chat-window] previous chat {state.chat_id[:8]} aged "
                    f"{age_min:.0f} min (>= {args.reuse_chat_window_secs/60:.0f} min "
                    f"window) — starting fresh chat for round {i}.",
                    flush=True,
                )
                state.chat_id = None
                state.chat_started_at = None
                state.chat_round_count = 0

            chat_status = (
                f"resume {state.chat_id[:8]} (age {(chat_age_secs or 0)/60:.0f}min, "
                f"{state.chat_round_count} rounds in chat)"
                if not cold_start
                else "cold-start (new chat)"
            )
            banner(
                f"ROUND {i}/{args.rounds} | best={state.best_metric} "
                f"| no_improve_streak={state.rounds_without_improvement}/{args.patience} "
                f"| {chat_status}"
            )
            sha_before = get_head_sha(str(workspace))
            recent_log = get_recent_log(str(workspace))
            short_status = get_short_status(str(workspace))
            pinned_gpu = _pick_idle_gpu(args.gpu_pool) if args.gpu_pool else None
            if pinned_gpu is not None:
                print(
                    f"[gpu-pin] round {i} pinned HIP_VISIBLE_DEVICES={pinned_gpu} "
                    f"(picked from pool {args.gpu_pool})",
                    flush=True,
                )
            prompt = build_prompt(
                args, state, i, baseline, sha_before, recent_log, short_status,
                is_resume=(not cold_start),
                pinned_gpu=pinned_gpu,
            )

            round_dir = log_dir / f"round_{i:03d}"
            started_at = now_iso()
            t0 = time.monotonic()
            if args.dry_run:
                print("[dry-run] skipping cursor-agent", flush=True)
                round_dir.mkdir(parents=True, exist_ok=True)
                (round_dir / "prompt.md").write_text(prompt)
                cursor_exit = 0
                returned_session_id: Optional[str] = state.chat_id  # keep state
            else:
                cursor_exit, returned_session_id = run_cursor_round(
                    args, prompt, round_dir,
                    resume_chat_id=state.chat_id if not cold_start else None,
                    pinned_gpu=pinned_gpu,
                )

            # Update chat-window state: cold start banks the new session_id;
            # resume reuses the old one (returned_session_id should match).
            if cold_start:
                state.chat_id = returned_session_id  # may be None on failure
                state.chat_started_at = t0 if returned_session_id else None
                state.chat_round_count = 1 if returned_session_id else 0
            else:
                state.chat_round_count += 1
                # Sanity-check resume preserved the chat_id.
                if returned_session_id and returned_session_id != state.chat_id:
                    print(
                        f"[chat-window] WARNING resumed chat returned new "
                        f"session_id {returned_session_id} (expected {state.chat_id}); "
                        f"adopting new id.",
                        flush=True,
                    )
                    state.chat_id = returned_session_id
            duration = time.monotonic() - t0

            sha_after = get_head_sha(str(workspace))
            metric = run_metric(
                args.metric_cmd, str(workspace), args.metric_timeout,
                gpu_pool=args.gpu_pool,
            )

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
