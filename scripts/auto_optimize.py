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
    "**任务范围**：本轮**只**搞 HipKittens grouped GEMM kernel —— BF16 + FP8 tensorwise，\n"
    "forward + variable-K backward + dA backward 全部归 grouped 任务。**不要碰 dense GEMM**\n"
    "（dense 路径有自己的优化通道，本 run 不测 dense）。\n\n"
    "**最终目标**（2 段独立判定，每段都必须 PASS；看 metric stderr 的 `Goals:` 2 行 PASS/FAIL）：\n"
    "  (1) **grp_BF16 vs TRITON** ≥ 1.20 —— grouped BF16 GEMM (16 shape, DeepSeek-V3 + gpt_oss_20B)\n"
    "  (2) **grp_FP8  vs TRITON** ≥ 1.20 —— grouped FP8 tensorwise GEMM (同 16 shape)\n\n"
    "**Score = 1000 × geomean(min(progress_i / 1.20, 1.0))**，两段等权重 (1, 1)。\n"
    "解读：\n"
    "  • 每段 progress 上限锁 1.0：把 grp_BF16 从 1.20 推到 1.50 **不加分**；要花\n"
    "    cycle 在还没 PASS 的段上才有 score 收益。\n"
    "  • 2 段全 PASS（每段 progress=1.0）= score 1000；任何段 progress<1 ⇒ score<1000。\n"
    "  • Reject（HIPKITTEN raise / NaN / Inf）把 ratio 钉到 0.01：grp_FP8 中 1/16 reject\n"
    "    = 该段 geomean 大跌 → 段进度塌掉 → score 直接跌穿。**收紧 can_handle 排除难\n"
    "    shape = 扣分**，不是好优化。\n"
    "2 段全 PASS 才算交付；score 只是 ranking 信号，**不是验收标准**。\n\n"
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
    "**起点（host-pad + uniform 判断 + per-group fallback 全删除之后的实测）**：\n"
    "  • grp_BF16 (16)   8/16 case FAIL —— gpt_oss misaligned (N=2880/5760, K=2880) 数值错\n"
    "                    avg fwd ≈ 515 TF (vs Triton 1119 ⇒ ratio ≈ 0.46) — 两段都没接近 1.20\n"
    "  • grp_FP8  (16)   16/16 RuntimeError —— backward dA 路径用 trans_b=False (RRR layout)，\n"
    "                    HipKittens FP8 binding 缺 grouped_rrr，Primus raise → ratio clip 0.01\n"
    "**第一轮必做**（grp_FP8 整段被一个缺失 binding 卡死）：在 HipKittens 仓库\n"
    "  /workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp\n"
    "里 mirror BF16 grouped 的实现写一个 persistent `grouped_rrr` (含 _dscale 变体)，\n"
    "**接受任意 group_lens**（device 端 O(G) scan group_offs；不许 host 判 uniform）+\n"
    "**接受任意 (N, K)**（column-masked C store + LDS K-tail + per-group SRD）。\n"
    "**唯一允许的路径（user 拍板）**：HK 仓库新加 **persistent + CPU-sync-free** grouped\n"
    "kernel（mirror Triton 的 `primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py::\n"
    "_grouped_bf16_persistent_gemm_kernel`，line 182-330）。**不许** multi-stream / per-group\n"
    "launch / pad-and-copy 凑数。Triton 是 single-launch persistent + device-side group_offs\n"
    "scan，per-group launch 永远输给它，pad-and-copy 在非对齐 N/K 上能挤出局部分数但**永远**\n"
    "无法跨过 1.20 vs TRITON 的目标。两段必须**一并**用 persistent kernel 解。\n"
    "建议路径：\n"
    "  ① HK FP8 grouped binding 新加 grouped_rrr (+ _dscale)（轮 1）→ grp_FP8 unblock\n"
    "  ② HK BF16 grouped main kernel 修 misaligned (N, K)（column-masked C + LDS K-tail）\n"
    "    （轮 2-3）→ grp_BF16 8/16 FAIL → 16/16 PASS\n"
    "  ③ HK FP8 grouped main kernel 修 misaligned (mirror BF16)（轮 4-5）→ grp_FP8 性能爬升\n"
    "  ④ 两段一起 persistent kernel 性能调优（XCD 调度、wave layout、swizzle）（轮 6+）\n"
    "    → 推到 1.20x Triton\n"
    "**收紧 can_handle 反而扣分**：reject → ratio clip 0.01 → 整段 geomean 塌掉，\n"
    "1 个 reject 就能让 score 掉 100+ 分。\n\n"
    "**严禁的『假优化』模式**（违反 = 本轮立即作废）：\n"
    "  ✗ **runtime 读任何 .json / .pkl / .autotune_cache.json**（开发期 bench 用、归纳完后\n"
    "    runtime 不再 import；任何 hipkitten 路径上看到 `json.load`/`pickle.load` 都是违规）\n"
    "  ✗ **任何形式 cache：dict / weakref / data_ptr / _version / LRU / TTL** —— quant 输出 /\n"
    "    preshuffle / group_offs / grid_x_hint / scale / autograd 中间产物全在禁单里\n"
    "  ✗ case-by-case 形状表：`if (M,N,K)==(X,Y,Z): return cfg`（**通用规则**才允许，\n"
    "    例如 `if K>=4096`、`if min(tiles)<16`、`if N>=K`）\n"
    "  ✗ 收紧 can_handle 把难 shape 排除掉（geomean clip 0.01，分数立刻塌）\n"
    "  ✗ **grouped 路径严禁 host-side padding**（user 拍板，新规）：\n"
    "    1) **禁止** `torch.empty(...) + .copy_() + .zero_()` 套路把输入 K/N 凑齐对齐再\n"
    "       喂 kernel —— 这是变相绕过 misaligned 支持。agent 必须在 HK kernel 端 native\n"
    "       处理（column-masked C store + LDS K-tail + per-group SRD）。这一条覆盖\n"
    "       forward / dB / 任何 grouped fallback 所有路径，**没有例外**。\n"
    "    2) **禁止** `_pad_2d` / `_pad_2d_into` / `padded_shape(...)` 等 host-pad helper\n"
    "       在 dispatch 路径上调用。这些 helper 即便保留作为开发期 probe 也不许出现\n"
    "       在 `grouped_gemm{,_fp8}_impl.py` 的 `execute` 函数里。\n"
    "  ✗ **grouped 路径严禁 uniform-M 判断**（user 拍板，新规）：\n"
    "    1) **禁止** 任何形如 `_uniform_group_m(...)` / `if a.shape[0] % bs == 0` /\n"
    "       `m_uniform is not None: fast-path else: fallback` 的分支。host 端**永远走\n"
    "       同一条 persistent grouped launch**，kernel 端用 device-side `group_offs`\n"
    "       处理任意 group_lens（uniform 或非 uniform 都是同一条路径，不许二分）。\n"
    "    2) `m_avg = a_total // bs` **只能** 用作 `select_default_config(...)` 选 cfg\n"
    "       （group_m / num_xcds / kernel variant），**绝不能** 用作分支条件。cfg 选不\n"
    "       准 = 慢，但仍然 correct，因为 kernel 端 `group_offs` 自己 O(G) scan。\n"
    "  ✗ **grouped 路径严禁任何形式的 balanced 假设**（命名 + 资格判定都不准）：\n"
    "    1) `can_handle` / dispatch 路径**禁止**加任何形如 `_is_balanced_group_lens` /\n"
    "       `all(g==g[0] for g in group_lens)` 的检查去 reject 非均匀 `group_lens` —— 真实\n"
    "       MoE 训练里 group_lens 永远是稀疏不均匀的，reject 一个 shape → ratio clip 0.01\n"
    "       → 整段 geomean 塌掉。HK kernel 必须用**单条 persistent grouped launch** 处理\n"
    "       任意 group_lens（uniform 或非 uniform 都同一条路径），host 端不许判断、\n"
    "       不许 fast-path/fallback 二分（重复参考上一条 uniform 严禁）。\n"
    "    2) **命名同样禁止传播 balanced 污染**：HK 仓库 BF16 .so 当前暴露的 `grouped_*_balanced`\n"
    "       是历史遗留命名，Primus 内部已经统一别名为 `grouped_*`（loader 通过\n"
    "       `_resolve_grouped_attr` 兼容老 .so，新代码不暴露 `_balanced` 后缀）。任何**新加**\n"
    "       的 HK 仓库 binding（FP8 grouped、新增 layout、新加变体）一律命名 `grouped_*`，\n"
    "       不准带 `_balanced` 后缀；任何**新加**的 Primus 端 grouped API 也不准带 `_balanced` 后缀。\n"
    "       这条规则覆盖 BF16 grouped、新加的 FP8 grouped、未来一切 grouped 入口。\n"
    "  ✗ **grouped 路径严禁 multi-stream / per-group launch / cudaStream 池**（user 拍板）：\n"
    "    1) **禁止**任何形式的 `for g in range(B): launch_dense_kernel(...)` 一组一发 ——\n"
    "       不论 uniform 还是非 uniform，不论 fast-path 还是 fallback，**唯一允许**的\n"
    "       grouped 入口是 single-launch persistent grouped kernel（kernel 内部消费\n"
    "       `group_offs`）。哪怕分摊到 `torch.cuda.Stream()` 池『并行』也不准。MI355X\n"
    "       有 256 个 CU + 8 个 XCD，一个 persistent kernel 单次 launch 用满 256 个\n"
    "       program 就能 saturate；切多 stream 既增加 host overhead 又破坏 XCD chiplet\n"
    "       调度。\n"
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
    "  - scripts/_metric_grouped_only.py / scripts/_metric_hk_ratio.py / scripts/auto_optimize.py /\n"
    "    scripts/run_dod_metric.sh（任何 metric / 调度脚本都属于 FROZEN，agent 不许动）\n"
    "  - tests/pytorch/ops/test_*.py（不能加 skip / 删 parametrize / 调 SNR 阈值）\n"
    "  - benchmark/ops/config.py / benchmark/ops/bench_grouped_gemm_turbo.py（shape ground truth）\n"
    "  - /root/.cursor/skills/hipkittens-primus-turbo-backend/SKILL.md"
)
# Loop metric: focused grouped-only benchmark of HIPKITTEN vs TRITON on the
# 16 MoE shape suite (DeepSeek-V3 + gpt_oss_20B), measured for both BF16
# grouped GEMM and FP8 tensorwise grouped GEMM (32 cases). Score is
# int(weighted_geomean(min(g_i/1.20, 1.0)) * 1000); 1000 = both sections
# at >= 1.20x TRITON (= PASS). HIPKITTEN reject clips that shape's ratio
# to 0.01 — section geomean drops sharply, agent can't game by narrowing
# can_handle. ~15-25s wall.
#
# The 6-segment dense+grouped metric (scripts/_metric_hk_ratio.py) is
# preserved for backward compat / future runs that re-include dense GEMM
# objectives. For the current run the user has scoped the work to
# grouped GEMM kernels only — measuring dense sections every round
# would just waste cycles and add noise.
DEFAULT_METRIC_CMD = "python3 scripts/_metric_grouped_only.py"
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
    p.add_argument("--rounds", type=int, default=40, help="Maximum number of optimization rounds.")
    p.add_argument(
        "--patience",
        type=int,
        default=8,
        help=(
            "Stop early once this many consecutive rounds end without metric improvement. "
            "Default 8 (was 5): writing a new persistent grouped HK kernel typically spans "
            "3-5 rounds (write .cpp → compile → numerical probe → fix → re-bench) before "
            "the metric reflects the win, so a tight patience prematurely kills long tasks."
        ),
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
        "--task-file",
        type=str,
        default=None,
        help=(
            "Path to a UTF-8 text file whose contents replace --task. Use this when "
            "the task body contains shell-active characters (backticks, $, parens) "
            "that get mangled by `--task \"$VAR\"` quoting."
        ),
    )
    p.add_argument(
        "--prompt-extra-file",
        type=str,
        default=None,
        help=(
            "Path to a UTF-8 text file whose contents replace --prompt-extra. "
            "Same shell-safety rationale as --task-file."
        ),
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
        default="grouped_only_score",
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
        "--dod-strict",
        action="store_true",
        default=False,
        help=(
            "If set, EARLY-STOP the run when a DoD checkpoint regresses "
            "(score < 0). Default is OFF — DoD score is informational only "
            "(recorded in summary.json + dod_checkpoints, surfaced to the "
            "agent prompt) so a pre-existing red baseline can't kill the run."
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
        "--focus-model",
        type=str,
        default="all",
        choices=["all", "gpt_oss", "deepseek", "dsv3"],
        help=(
            "Which model's shapes drive the score / Goals geomean. 'all' (default) "
            "counts both DSV3 and gpt_oss; 'gpt_oss' / 'deepseek' restricts the "
            "geomean to that model only. The other model is still benchmarked + "
            "correctness-checked every round (visible as [watch] rows in the metric "
            "table) so silent regressions on the un-focused model still surface in "
            "stderr — they just don't move the score. Use to spend rounds on the "
            "model whose gap-to-target dominates without distractor cycles on the "
            "well-tuned path. Propagates as METRIC_MODEL_FILTER env to the metric."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip cursor-agent invocations, just measure the metric N times. Useful for testing.",
    )
    p.add_argument(
        "--resume-state",
        type=str,
        default="",
        help=(
            "Path to a previous run's summary.json. When set, the loop:\n"
            "  - Skips the startup baseline metric measurement\n"
            "  - Restores baseline_metric / best_metric / best_sha / rounds list /\n"
            "    last_dod_score / last_dod_sha from the file\n"
            "  - Continues the round counter past the saved rounds_run (so logs go\n"
            "    into round_006, round_007, ... instead of overwriting)\n"
            "Use case: bumping --rounds / --patience mid-run without losing 1h+ of\n"
            "metric history and HK commits. Chat session_id is NOT restored — the\n"
            "next round cold-starts a new cursor-agent chat (which is fine since the\n"
            "agent reads git log + task body to recover context)."
        ),
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

    focus_resume_block = ""
    focus_pick_hint = ""
    if args.focus_model and args.focus_model != "all":
        focus_resume_block = (
            f"\n【focus={args.focus_model}】metric score 只统计 {args.focus_model} 段；DSV3 / 另一段在表里以 [watch] 出现，"
            f"不计入 score 但 correctness FAIL 仍是硬错误。**不要再啃未 focus 那段的 kernel** —— score 不奖励。"
        )
        focus_pick_hint = (
            f"  ⚠️ focus={args.focus_model}：选攻坚 shape 时**只看不带 [watch] 的行**。"
        )

    dod_line = ""
    if args.dod_every > 0 and state.last_dod_score is not None:
        if args.dod_strict:
            dod_tail = f"提醒：每 {args.dod_every} 轮自动跑一次，failed > 0 立刻 EARLY-STOP。"
        else:
            dod_tail = (
                f"提醒：每 {args.dod_every} 轮自动跑一次，仅作参考记录，不会 EARLY-STOP；"
                f"但 failed 数应尽量收敛。"
            )
        dod_line = (
            f"上次 DoD={state.last_dod_score} (sha {(state.last_dod_sha or '')[:8]}). "
            f"{dod_tail}"
        )

    return f"""【第 {round_idx} / {args.rounds} 轮 — 接续上一轮 chat】
本 chat session 已运行 {state.chat_round_count + 1} 轮 / {chat_age_min:.0f} 分钟（窗口上限
{args.reuse_chat_window_secs / 60:.0f} 分钟，超过会换新 chat）。你已知道 task body / 优化方向 /
FROZEN 列表 / 严禁假优化清单（host-pad / uniform 判断 / per-group launch / CPU sync 等）——
**不要重新读 SKILL.md，不要重 quote 上面这些**，直接干活。如果你忘了某条规则，
自己回去翻本 chat 历史，不要再让用户复述。

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
{focus_resume_block}
【本轮指令】
1. 第一步：跑 metric `{args.metric_cmd} 2>&1 | tee /tmp/metric_round_{round_idx}.log`，
   看 stderr 表里 ratio < target 的 shape，按 ratio 升序选 1 个攻坚目标 ——
   不许凭印象选 shape，必须用本轮 metric 数据决定。{focus_pick_hint}
2. 你**记得**上一轮在做什么 —— 如果是跨轮长任务（写 HK .cpp kernel / 编译 / 数值 probe），
   continue 那条主线，不要换方向。如果上一轮已 commit 完整工作，再选新目标。
3. 改完跑 metric 验证，再 commit。每仓库最多 1 commit/轮。
4. **backward 改动**（dB / dA / variable-K）metric 看不到 —— 必须自跑
   `bench_grouped_gemm_turbo.py --dtype {{bf16,fp8}}`，把 fwd+bwd TFLOPS 和
   correctness 贴进 commit message。
5. 末尾 markdown 小结：本轮目标 / 改了什么 / before-after metric / commit SHA / 下一轮建议。

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
        if args.dod_strict:
            gate_line = (
                f"它跑 curated DoD smoke regression set（test_dod_smoke.py，约 610 cases，dense + "
                f"grouped 全覆盖），任何 failed > 0 都会让脚本立刻 EARLY-STOP。\n"
            )
            commit_warn = (
                f"    后再 commit；否则脚本下次 checkpoint 会因为你的改动 EARLY-STOP，整个 run 报废。\n"
            )
        else:
            gate_line = (
                f"它跑 curated DoD smoke regression set（test_dod_smoke.py，约 610 cases，dense + "
                f"grouped 全覆盖），分数仅作参考记录到 summary.json，不会 EARLY-STOP（baseline\n"
                f"已经红的 case 不影响整轮自动化）。但 failed 数仍是回归信号，请尽量收敛。\n"
            )
            commit_warn = (
                f"    后再 commit。脚本不再因 DoD 红而停，但 failed 数会写进 summary.json + 下轮 prompt，\n"
                f"    便于追踪你这轮是否引入了新回归。\n"
            )
        dod_block = (
            f"\n【DoD 检查点（每 {args.dod_every} 轮自动跑一次，不需要你手动跑）】\n"
            f"脚本会在第 {args.dod_every}, {2*args.dod_every}, ... 轮结束后自动执行：\n"
            f"  {args.dod_cmd}\n"
            f"{gate_line}"
            f"本 run 任务范围**只**是 grouped GEMM (BF16 + FP8)，但 smoke 还是测了 dense ——\n"
            f"为了兜住跨 backend 共用代码（autograd 入口、dispatcher、quantize_fp8、torch.library\n"
            f"custom_op 注册等）的意外回归。所以 commit 的时候要小心：\n"
            f"  - 改动**只**触及 grouped HIPKITTEN 路径（grouped_gemm_impl.py / grouped_gemm_fp8_impl.py /\n"
            f"    kernels/hipkitten/grouped*）：快 metric 通常足够。\n"
            f"  - 改动触及任何**共用代码**（autograd / dispatcher / quantize_fp8 / grouped_gemm.py 顶层 /\n"
            f"    custom_op 注册）：必须自己跑一次 `{args.dod_cmd}`（约 5-10 分钟）确认 0 failed\n"
            f"{commit_warn}"
            f"  - {last_dod_line}\n"
        )

    focus_block = ""
    if args.focus_model and args.focus_model != "all":
        focus_block = (
            f"\n【**本 run 焦点：`{args.focus_model}`**（user 拍板，metric 已切到 focused 模式）】\n"
            f"metric 的 score / Goals geomean **只统计 {args.focus_model} shapes**，另一段（DSV3 或\n"
            f"gpt_oss）虽然每轮仍然跑 + correctness check，但在 stderr 表里被打 `[watch]` 标签、\n"
            f"**不计入 score**。这意味着：\n"
            f"  • 本轮选攻坚 shape 时**只看不带 [watch] 的行**；DSV3 ratio 怎么动都不影响 score。\n"
            f"  • 但 **[watch] correctness FAIL 仍是硬错误** —— 不许为了 focused 段提分而让\n"
            f"    DSV3 数值跑挂；watch 段 `correct_fail > 0` 一定要修。\n"
            f"  • 不许动 metric / shape suite 把 [watch] 行删掉来 fake 提速 —— FROZEN 列里都标了。\n"
            f"  • 接续旧 chat 的轮里如果 agent 又开始啃**未 focus 那段**的 kernel（典型：BF16 RRR\n"
            f"    K-tail fuse），先停手；本 run 的 score 不再奖励那条线。\n"
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
{focus_block}
【本轮的快速验收命令】
metric 命令（grouped-only，约 15-25 秒、单 GPU、自动选空闲卡）：
  {args.metric_cmd}
metric 跑的 shape 数 / score 公式 / target —— 看 task body。**通用规则**：
  • score 越高越好；1000 = 全部 PASS；< 1000 = 还有段没到 target。
  • HIPKITTEN reject（raise / NaN / Inf）会让那个 shape 的 ratio clip 到 0.01 →
    geomean 大跌 → 段权重放大冲击。**收紧 can_handle 排除难 shape = 扣分**。
  • 真实压力来自 (a) HK kernel 相对 reference 的速度 + (b) dispatch / quantize /
    launch overhead 是否清掉。

【**首要数据源** — metric 的 stderr 表】
metric 命令的 stderr 会打印一张逐 shape 表，列出 name / hk_tflops / ref_tflops / ratio /
status，以及每段 geomean + Goals: PASS/FAIL 块。**本轮第一步先跑一次 metric**，
从那张表里找出 ratio < target 的 shape，按 ratio 升序选 1 个作为本轮攻坚目标。
**不许凭印象选 shape**，必须用上一轮 metric 数据决定。
跑命令：`{args.metric_cmd} 2>&1 | tee /tmp/metric_round_{round_idx}.log`。
**改完一次、commit 前一次** —— 不要每改一行都跑。

【grouped backward 验证（agent 自查，不归 metric）】
本 metric 只测 grouped FORWARD（追求快）。任何动 backward 路径（dB / dA / variable-K）
的改动，agent **必须**自己跑：
  PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \\
  PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \\
    python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype bf16 --output /tmp/hk_bf16.csv
  （fp8 同理）—— bench 含 fwd + bwd + correctness check (allclose / SNR)。
fwd-OK 但 bwd-broken 的回归 metric 看不到，所以 backward 改动**必须**贴 bench 输出
到 commit message 才算交付。
{dod_block}

【度量指标】指标名: {args.metric_name}（数值越高越好；1000 = 全部段 PASS）
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
    if args.task_file:
        try:
            args.task = Path(args.task_file).read_text(encoding="utf-8")
            print(f"[task-file] loaded {len(args.task)} chars from {args.task_file}", flush=True)
        except OSError as exc:
            print(f"--task-file {args.task_file} unreadable: {exc}", file=sys.stderr)
            return 2
    if args.prompt_extra_file:
        try:
            args.prompt_extra = Path(args.prompt_extra_file).read_text(encoding="utf-8")
            print(
                f"[prompt-extra-file] loaded {len(args.prompt_extra)} chars from "
                f"{args.prompt_extra_file}", flush=True,
            )
        except OSError as exc:
            print(f"--prompt-extra-file {args.prompt_extra_file} unreadable: {exc}", file=sys.stderr)
            return 2
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f"workspace {workspace} is not a directory", file=sys.stderr)
        return 2

    if shutil.which("cursor-agent") is None and not args.dry_run:
        print("cursor-agent not on PATH; install Cursor CLI first", file=sys.stderr)
        return 2

    # Propagate focus to the metric subprocess via env var (the metric reads
    # METRIC_MODEL_FILTER directly). Setting in os.environ here means
    # _subprocess_env_with_pinned_gpu's os.environ.copy() picks it up
    # automatically, no need to thread the flag through run_metric.
    if args.focus_model and args.focus_model != "all":
        os.environ["METRIC_MODEL_FILTER"] = args.focus_model
        print(
            f"[focus] METRIC_MODEL_FILTER={args.focus_model} "
            f"(score / Goals geomean restricted to {args.focus_model} shapes; "
            "other model still benchmarked + correctness-gated as [watch])",
            flush=True,
        )
    else:
        os.environ.pop("METRIC_MODEL_FILTER", None)

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
    baseline: Optional[float] = None
    round_offset = 0

    if args.resume_state:
        try:
            saved = json.loads(Path(args.resume_state).read_text())
        except Exception as exc:
            print(f"--resume-state {args.resume_state} unreadable: {exc}", file=sys.stderr)
            return 2
        baseline = saved.get("baseline_metric")
        state.best_metric = saved.get("best_metric")
        state.best_sha = saved.get("best_sha")
        state.last_dod_score = saved.get("last_dod_score")
        state.last_dod_sha = saved.get("last_dod_sha")
        state.dod_checkpoints = list(saved.get("dod_checkpoints") or [])
        for r_dict in saved.get("rounds") or []:
            state.rounds.append(RoundResult(
                index=r_dict["index"],
                started_at=r_dict["started_at"],
                finished_at=r_dict["finished_at"],
                duration_s=r_dict["duration_s"],
                metric=r_dict.get("metric"),
                best_so_far=r_dict.get("best_so_far"),
                improved=r_dict.get("improved", False),
                head_sha_before=r_dict.get("head_sha_before", ""),
                head_sha_after=r_dict.get("head_sha_after", ""),
                cursor_exit_code=r_dict.get("cursor_exit_code", 0),
                log_dir=r_dict.get("log_dir", ""),
            ))
        round_offset = saved.get("rounds_run", len(state.rounds))
        # Reconstruct rounds_without_improvement by tail-counting non-improving
        # rounds at the end of the restored history.
        streak = 0
        for r in reversed(state.rounds):
            if r.improved:
                break
            streak += 1
        state.rounds_without_improvement = streak
        write_summary(summary_path, args, state, baseline)

    try:
        if args.resume_state:
            remaining = max(args.rounds - round_offset, 0)
            banner(
                f"AUTO-OPTIMIZE RESUME | restored {round_offset} rounds | "
                f"--rounds={args.rounds} (absolute cap; running rounds "
                f"{round_offset + 1}..{args.rounds}, {remaining} more) | "
                f"patience={args.patience} | streak={state.rounds_without_improvement} | "
                f"baseline={baseline} | best={state.best_metric} | log_dir={log_dir}"
            )
            if remaining <= 0:
                banner(f"AUTO-OPTIMIZE: --rounds {args.rounds} <= already-run {round_offset}, nothing to do.")
                return 0
            range_iter = range(round_offset + 1, args.rounds + 1)
        else:
            banner(f"AUTO-OPTIMIZE start | rounds={args.rounds} | patience={args.patience} | log_dir={log_dir}")
            section("baseline metric")
            baseline = run_metric(
                args.metric_cmd, str(workspace), args.metric_timeout,
                gpu_pool=args.gpu_pool,
            )
            state.best_metric = baseline
            state.best_sha = get_head_sha(str(workspace))
            write_summary(summary_path, args, state, baseline)
            range_iter = range(1, args.rounds + 1)

        for i in range_iter:
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
                # ``last_dod_score`` / ``last_dod_sha`` always track the most
                # recent checkpoint (green or red) so the next-round prompt
                # can surface "you're at score=X" regardless. The early-stop
                # decision (gated by --dod-strict) is made separately below.
                if dod_score is not None:
                    state.last_dod_score = dod_score
                    state.last_dod_sha = sha_after
                write_summary(summary_path, args, state, baseline)
                regressed = dod_score is None or dod_score < 0
                if regressed and args.dod_strict:
                    banner(
                        f"EARLY-STOP: DoD checkpoint regressed (score={dod_score}, rc={dod_rc}) "
                        f"after round {i}. --dod-strict was set. "
                        f"Inspect {dod_log} for details."
                    )
                    return 0
                if regressed:
                    banner(
                        f"DoD checkpoint regressed after round {i} "
                        f"(score={dod_score}, rc={dod_rc}) — recorded as informational; "
                        f"continuing because --dod-strict is OFF. "
                        f"Inspect {dod_log} for details."
                    )

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
