#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Iteratively drive the Cursor CLI to optimize the grouped MXFP8 GEMM kernels
on the ``dev/kyle_mxfp8_gg`` branch of Primus-Turbo.

The script runs many rounds of ``cursor-agent --print`` (headless mode) with
*max mode* turned on, asking the agent to make autonomous progress.  Between
rounds it runs a user-supplied *metric command* (which prints a single number,
higher is better) and stops early if the metric has not improved for a
configurable number of rounds.

Each round is fully scripted -- the prompt tells the agent to:

  1. Read the local skill at
     ``.claude/skills/mxfp8-persistent-optimization/SKILL.md``
     so it inherits the project context every round.
  2. Decide on its own what to do this round.
  3. Run validation tests / benchmarks before claiming progress.
  4. Commit any improvement on the current branch (no push).

After each round the script measures the new metric, compares it against the
historical best, and either updates ``best_sha`` / resets the patience counter
or warns and increments.  When patience runs out the script prints a clear
``EARLY-STOP`` banner and exits.

The framework is forked from the auto_optimize.py shipped on
``dev/kyle_hipkitten_bf16``; the changes here are the project-specific
defaults (skill path, task description, metric command, prompt body) and the
DoD wording that matches the MXFP8 grouped-GEMM context.

Example
-------

::

    cd /workspace/code/Primus-Turbo
    python3 scripts/auto_optimize.py --rounds 20 --patience 5

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

DEFAULT_MODEL = "claude-opus-4-7-thinking-max"
DEFAULT_SKILL = (
    "/workspace/code/Primus-Turbo/.claude/skills/"
    "mxfp8-persistent-optimization/SKILL.md"
)
# Tenant is sharing the node on GPUs 0-3, so the loop and any agent-driven
# probes must stay on 4-7.  Both the metric script and the per-round prompt
# inherit this; override via env if the topology changes.
DEFAULT_GPU_POOL = "4,5,6,7"
DEFAULT_TASK = (
    "持续优化 Primus-Turbo 在 gfx950 (MI355) 上的 grouped MXFP8 GEMM kernel "
    "(forward + variable-K wgrad)，目标是 forward / backward TFLOPS 持续向 "
    "Triton FP8 tensorwise 看齐 (理想超过)。当前在 _metric_mxfp8.py "
    "上的 baseline score 来自 SHAPES (DeepSeek-V3 + gpt_oss_20B) 上 fwd+bwd "
    "TFLOPS 之和减去 SNR/stress/exception 罚分；任何改动都不能让相应测试 "
    "失败、benchmark 不能从 16/16 PASS 退化、确定性 stress 不能显著恶化。"
    "**每轮快速验收**：scripts/_metric_mxfp8.py 输出的 score 不能下降；"
    "**最终验收 (DoD)**：tests/pytorch/ops/test_grouped_gemm_fp8.py 中的 "
    "test_grouped_gemm_fp8_mx_blockwise 全 pass，且 "
    "benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8 --granularity mxfp8 "
    "16/16 PASS。"
)
DEFAULT_METRIC_CMD = "python3 scripts/_metric_mxfp8.py"
# Deep-check acceptance: full mxfp8 pytest sweep.  Run every N rounds (not
# every round) because it's slow.  A failure forces improved=False and does
# NOT update best_metric/best_sha, even if the cheap metric went up — it
# guards against optimizations that game the metric while breaking the
# numerical/SNR contract enforced by the test sweep.
DEFAULT_DEEP_CHECK_CMD = (
    "pytest tests/pytorch/ops/test_grouped_gemm_fp8.py"
    "::test_grouped_gemm_fp8_mx_blockwise --tb=short -q"
)
DEFAULT_DEEP_CHECK_EVERY = 5
CLI_CONFIG_PATH = Path(os.path.expanduser("~/.cursor/cli-config.json"))

# Preview / logging truncation (unchanged behavior).
METRIC_CMD_PREVIEW_LEN = 120
METRIC_IO_TAIL_CHARS = 300
METRIC_STDERR_TAIL_LINES = 12
DEEP_CHECK_TAIL_LINE_CAP = 200
DEEP_CHECK_CONSOLE_TAIL_LINES = 12

# Set once per process run for summary.json started_at (replaces setattr on write_summary).
_SESSION_STARTED_AT: str | None = None


def _preview_cmd(cmd: str, max_len: int = METRIC_CMD_PREVIEW_LEN) -> str:
    """Shorten long shell commands for console section headers."""
    if len(cmd) <= max_len:
        return cmd
    return cmd[:max_len] + "..."


def _metric_stderr_tail(err: str, *, lines: int = METRIC_STDERR_TAIL_LINES) -> str:
    """Last N lines of stderr for operator visibility."""
    if not err:
        return ""
    return "\n".join(err.splitlines()[-lines:])


@dataclass
class RoundResult:
    index: int
    started_at: str
    finished_at: str
    duration_s: float
    metric: float | None
    best_so_far: float | None
    improved: bool
    head_sha_before: str
    head_sha_after: str
    cursor_exit_code: int
    log_dir: str
    # Deep-check fields are None when the round didn't run a deep check.
    deep_check_ran: bool = False
    deep_check_passed: bool | None = None
    deep_check_exit_code: int | None = None

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
            "deep_check_ran": self.deep_check_ran,
            "deep_check_passed": self.deep_check_passed,
            "deep_check_exit_code": self.deep_check_exit_code,
        }


@dataclass
class TrajectoryState:
    rounds: list[RoundResult] = field(default_factory=list)
    best_metric: float | None = None
    best_sha: str | None = None
    rounds_without_improvement: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--rounds", type=int, default=20, help="Maximum number of optimization rounds.")
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
            "Shell command that prints one numeric metric on the last non-empty stdout line "
            "(higher is better; default MXFP8 helper prints one integer). "
            "Run after every round and at startup for the baseline."
        ),
    )
    p.add_argument(
        "--metric-name",
        type=str,
        default="mxfp8_score",
        help="Human-readable metric label (used in logs only).",
    )
    p.add_argument(
        "--deep-check-cmd",
        type=str,
        default=DEFAULT_DEEP_CHECK_CMD,
        help=(
            "Shell command run every --deep-check-every rounds as the slow "
            "DoD acceptance gate (default: the mxfp8 pytest sweep).  A "
            "non-zero exit code forces improved=False for that round and "
            "does NOT update best_metric, regardless of the cheap metric.  "
            "Set to empty string to disable."
        ),
    )
    p.add_argument(
        "--deep-check-every",
        type=int,
        default=DEFAULT_DEEP_CHECK_EVERY,
        help=(
            "Run --deep-check-cmd every N rounds (default %(default)s).  "
            "Has no effect if --deep-check-cmd is empty."
        ),
    )
    p.add_argument(
        "--deep-check-timeout",
        type=int,
        default=60 * 30,
        help="Seconds to allow the deep-check command before killing it (default 30 min).",
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Working directory passed to cursor-agent and metric command.",
    )
    p.add_argument(
        "--gpu-pool",
        type=str,
        default=DEFAULT_GPU_POOL,
        help=(
            "Comma-separated list of GPU ids the loop is allowed to use "
            "(default: %(default)s).  Exported as MXFP8_GPU_POOL to the "
            "metric / deep-check / agent so they all stay on the same set."
        ),
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
        help=(
            "Seconds to allow each cursor-agent invocation before killing it "
            "(default 60 min).  Opus thinking-max can spend ~35-45 min per "
            "round so 30 min was too tight: SIGTERM was hitting cursor-agent "
            "mid-summary and truncating the cursor.log output."
        ),
    )
    p.add_argument(
        "--metric-timeout",
        type=int,
        default=60 * 10,
        help="Seconds to allow the metric command (default 10 min).",
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


def _env_with_pool(gpu_pool: str, restrict_visible: bool = False) -> dict:
    """Return a copy of os.environ with the GPU pool propagated.

    Always sets MXFP8_GPU_POOL so downstream scripts (notably
    ``_metric_mxfp8.py``) can auto-pick from the pool.

    If restrict_visible=True, also sets HIP_VISIBLE_DEVICES to the pool so
    the subprocess CANNOT see GPUs outside the pool.  Use this for things
    that don't do their own auto-pick (e.g. pytest in single-process mode).
    Don't use it for the metric script -- it does its own auto-pick and
    setting HIP_VISIBLE_DEVICES would expose all 4 pool GPUs and let cuda
    default-pick GPU 0 (which may be busy).
    """
    env = os.environ.copy()
    env["MXFP8_GPU_POOL"] = gpu_pool
    if restrict_visible:
        env["HIP_VISIBLE_DEVICES"] = gpu_pool
    return env


def run_metric(cmd: str, cwd: str, timeout: int, gpu_pool: str) -> float | None:
    section(f"measuring metric: {_preview_cmd(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_env_with_pool(gpu_pool),
        )
    except subprocess.TimeoutExpired:
        print(f"[metric] TIMEOUT after {timeout}s", flush=True)
        return None
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0:
        print(
            f"[metric] non-zero exit {result.returncode}; "
            f"stdout={out[-METRIC_IO_TAIL_CHARS:]!r} stderr={err[-METRIC_IO_TAIL_CHARS:]!r}",
            flush=True,
        )
    if not out:
        print(f"[metric] empty stdout; stderr={err[-METRIC_IO_TAIL_CHARS:]!r}", flush=True)
        return None
    last_line = out.splitlines()[-1].strip()
    try:
        value = float(last_line)
    except ValueError:
        print(f"[metric] could not parse {last_line!r} as float", flush=True)
        return None
    print(f"[metric] = {value}", flush=True)
    if err:
        print(_metric_stderr_tail(err), flush=True)
    return value


def run_deep_check(cmd: str, cwd: str, timeout: int, log_path: Path,
                   gpu_pool: str) -> tuple[bool, int]:
    """Run the deep-check command, stream its output to log_path.

    Returns ``(passed, exit_code)`` where ``passed`` is true iff exit_code == 0.
    """
    section(f"deep-check: {_preview_cmd(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        logf.write(f"# Command: {cmd}\n# Started at: {now_iso()}\n\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env=_env_with_pool(gpu_pool, restrict_visible=True),
            )
        except FileNotFoundError as exc:
            print(f"[deep-check] failed to spawn: {exc}", flush=True)
            return False, 127
        tail: list[str] = []
        try:
            assert proc.stdout is not None
            start = time.monotonic()
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                tail.append(line.rstrip("\n"))
                if len(tail) > DEEP_CHECK_TAIL_LINE_CAP:
                    tail = tail[-DEEP_CHECK_TAIL_LINE_CAP:]
                if time.monotonic() - start > timeout:
                    print(
                        f"\n[deep-check] timeout {timeout}s exceeded; sending SIGTERM",
                        flush=True,
                    )
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return False, proc.returncode if proc.returncode is not None else 124
            proc.wait()
        except KeyboardInterrupt:
            print("\n[deep-check] interrupted by user; killing", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        rc = proc.returncode if proc.returncode is not None else 1
    if tail:
        for line in tail[-DEEP_CHECK_CONSOLE_TAIL_LINES:]:
            print(f"[deep-check] {line}", flush=True)
    print(f"[deep-check] exit_code={rc} ({'PASS' if rc == 0 else 'FAIL'})", flush=True)
    return rc == 0, rc


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
    baseline_metric: float | None,
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
      * reminds it of hard constraints (no race regressions, no test deletion);
      * asks it to commit any progress on the current branch (no push).
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

    if args.deep_check_cmd:
        deep_check_block = (
            f"\n【深度验收 - 每 {args.deep_check_every} 轮自动跑一次】\n"
            f"  {args.deep_check_cmd}\n"
            f"  这条命令通过 = 0 退出码；失败时本轮 improved 强制为 False，且 "
            f"best_metric 不会更新。**你 commit 之前必须自己跑一次**，确认 0 failed。\n"
        )
    else:
        deep_check_block = ""

    return f"""你是 Primus-Turbo 仓库 (分支 dev/kyle_mxfp8_gg) 的自动优化协作者。本次是第 {round_idx} / {args.rounds} 轮，由脚本自动调度。
工作目录: {args.workspace}
当前 git HEAD: {head_sha}

【强制第一步】
请先用读文件工具完整读取这份本地 skill：
  {args.skill_path}
里面写明了 grouped MXFP8 优化的允许与禁止方向 (持久化 kernel only / 不允许 flat fallback / 不允许 uniform-group fast path / 不允许主机端 sync 等)、rocprofv3 用法、MI355 资源现状、最近最佳已知性能。读完再决定本轮该做什么。

【优化目标】
{args.task}

【硬性 DoD - 任何一轮违反即视为失败】
1. scripts/_metric_mxfp8.py 输出的 score **必须不下降**。score = int(sum_tflops*10) - 1000*snr_fail - 100*stress_bad - 2000*exception；其中 sum_tflops 由几个 DeepSeek-V3 + gpt_oss_20B 形状的 fwd+bwd TFLOPS 累加得来。
2. tests/pytorch/ops/test_grouped_gemm_fp8.py 中 test_grouped_gemm_fp8_mx_blockwise 任何一个 case 都不能 fail。可单独跑：
     `pytest tests/pytorch/ops/test_grouped_gemm_fp8.py::test_grouped_gemm_fp8_mx_blockwise --tb=short -q`
3. benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8 --granularity mxfp8 应保持 16/16 PASS（gpt_oss_20B-Down 系列已在 cfd2616 修好，不要让它退化）。
4. 确定性 stress 必须收紧：metric 里 `stress_bad / 100` **目标 ≤ 2/100**（≤2%）。当前 baseline 大约在 2-6/100，主要是 G=4 M=1024 N=2048 K=2048 E4M3 这条 FWD `out` race（cfd2616 引入 non-volatile FWD C-store 时换来的副作用，dB race 是更早就存在的）。skill 里"Recommended Next Experiments"对修这条 race 有几个候选方向：在 epilogue 末尾加显式 L1/L2 fence、把 read_c 与 store_c 之间的 c_frags 寄存器复用拆开、或者干脆在 forward 也保持 volatile 但用更粗粒度的 wait_vmcnt 批次。任何修复都必须既保持 metric score 上升 (即 TFLOPS 不退化) 又把 stress_bad/100 拉低到 ≤2。绝不允许通过减少 STRESS_ITERS 或抬 STRESS_THRESH 来骗过这条门槛。

本轮的 metric 命令 (越大越好) =
  {args.metric_cmd}
{deep_check_block}
你必须确保：本轮结束时 metric 不下降；如果改动了 csrc / primus_turbo / scripts，commit 之前自己再跑一次 metric 命令确认 score >= 当前 best。
**绝不允许通过删测试 / 加 pytest.skip / 改 SNR 阈值这种方式骗过 DoD。**

【度量指标】指标名: {args.metric_name}（数值越高越好）
- 基线 (优化开始前) = {baseline_metric}
- 历史最佳 = {state.best_metric}
- 上一轮 = {last_metric}（improved={last_improved}）
- 已连续 {state.rounds_without_improvement} 轮未提升（patience={args.patience}）

【近期 git log】
{recent_log or "(空)"}

【当前 working tree 状态】
{short_status or "(干净)"}

【近 5 轮记录】
{history_block}

【你需要自主决策的事】
1. 先读 skill，理解当前阶段卡在哪、哪些方向被允许、哪些被明确禁止。
2. 自己挑一个**具体且可验证**的小步骤去做（不要一次铺得太大）。skill 里列出的"Recommended Next Experiments"是优先方向；下面是当前文件结构里几个真实可改的点：
   - csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h - GemmTile / phase_mfma_lds_ldg / store_c_subtile，inner-loop 调度 (主回路 wait_vmcnt<0> 是已知最大 stall)。
   - csrc/kernels/grouped_gemm/turbo/turbo_grouped_gemm_mxfp8_kernel.h - persistent forward kernel (compute_tile + 外层 tile_id 循环)。
   - csrc/kernels/grouped_gemm/turbo/turbo_grouped_gemm_mxfp8_wgrad_kernel.h - persistent variable-K wgrad kernel；目前保留 volatile C-store 是因为 dB 有 pre-existing race，谁能吃下这条 race 谁就再拿一段 perf。
   - primus_turbo/pytorch/ops/grouped_gemm_fp8.py - GroupedGemmFP8MXFunc wrapper (quantize_fp8_with_trans / preshuffle 启动开销)。
   - csrc/kernels/grouped_gemm/turbo_grouped_gemm.cu - workspace 与 preshuffle launch (有融合空间)。
3. 改完一定要：
   - 重编译：`rm -f build/temp/csrc/kernels/grouped_gemm/turbo_grouped_gemm.o; touch <你改的 .h>; HIP_VISIBLE_DEVICES=$IDLE pip install --no-build-isolation -e .` (skill 里有完整命令)。
   - 跑 metric：`HIP_VISIBLE_DEVICES=$IDLE python3 scripts/_metric_mxfp8.py --verbose 2>&1 | tail -20` 看 score 是否上升。
   - 跑 stress：`HIP_VISIBLE_DEVICES=$IDLE STOP_AFTER_BAD=1 N_ITERS=100 python3 .claude/probes/stress_grouped_mx_bwd_determinism.py` 看 BAD 是否爆。
   - 必要时跑 pytest mx_blockwise 子集 (慢)。
4. 如果改动确实有进展（metric 不降且测试 0 fail），就 git commit；若没把握就 git restore 还原，**不要留下脏 working tree**。

【硬性约束 - 不可违反】
- **本节点和别的租户共用，被允许使用的 GPU 池 = `{args.gpu_pool}`** (env: MXFP8_GPU_POOL)。绝对不要碰这个池外的卡。`scripts/_metric_mxfp8.py` 会从这个池里 auto-pick 一张空闲卡 (`rocm-smi --showuse --showpids` 看 KFD VRAM > 100 MiB 视为 busy)；直接跑 pytest / .claude/probes/* 时也务必从池里挑一张：例如 `HIP_VISIBLE_DEVICES=$(MXFP8_GPU_POOL={args.gpu_pool} python3 -c "from scripts._metric_mxfp8 import _pick_idle_gpu; print(_pick_idle_gpu())")`。
- skill 里的"Not allowed as final solutions"清单严禁触碰：不允许恢复 flat 内核、不允许 uniform-group fast path、不允许 host/device sync 当作 race fix、不允许 cache flush/invalidate 当 workaround、不允许 s_nop / 计时 padding、不允许 majority voting / 重复执行。
- 绝不删除 tests/pytorch/ops/test_grouped_gemm_fp8.py，不要给 mx_blockwise 加 pytest.skip。
- 绝不改 SNR 阈值（既不在测试里改，也不在 _metric_mxfp8.py 里改）。
- 不要修改 ~/.cursor/cli-config.json 或全局 git config。
- 不要 push 到 remote (本仓库只允许在本地 dev/kyle_mxfp8_gg 提交)。
- 不要用 `git rebase -i / git add -i` 这类交互命令。
- 小步前进：**每轮只 commit 1 个 focused commit**，commit message 用 `feat:` / `fix:` / `perf:` / `test:` / `refactor:` / `opt:` 风格，并在 body 里写明：动机 / 实测 score 变化 / stress BAD 计数变化。
- 任何 inner-loop wait_vmcnt 放宽尝试都必须搭配长程 stress (`N_ITERS=1000`) 验证。skill 里记录过 vmcnt<4> leak 的事。

【输出要求】
本轮结束前，在你的最后一条文本里给出一段 markdown 小结，包含：
- 本轮选择的目标
- 实际改了哪些文件
- 验证用的命令与简明结果（score / stress / pytest 三档）
- 是否 commit、commit SHA / message
- 下一轮你建议下游做什么

完成后退出（headless 模式会自动结束）。

{args.prompt_extra}"""


def maybe_set_max_mode(enabled: bool) -> dict | None:
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


def restore_cli_config(snapshot: dict | None) -> None:
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
                env=_env_with_pool(args.gpu_pool),
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


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    state: TrajectoryState,
    baseline: float | None,
) -> None:
    summary = {
        "started_at": _SESSION_STARTED_AT or now_iso(),
        "metric_name": args.metric_name,
        "metric_cmd": args.metric_cmd,
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
    global _SESSION_STARTED_AT
    _SESSION_STARTED_AT = now_iso()

    cli_snapshot = None if args.no_max_mode else maybe_set_max_mode(True)

    state = TrajectoryState()
    baseline = None

    try:
        banner(
            f"AUTO-OPTIMIZE start | rounds={args.rounds} | patience={args.patience} | log_dir={log_dir}"
        )
        section(f"baseline metric (gpu_pool={args.gpu_pool})")
        baseline = run_metric(args.metric_cmd, str(workspace), args.metric_timeout, args.gpu_pool)
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
            metric = run_metric(args.metric_cmd, str(workspace), args.metric_timeout, args.gpu_pool)

            # Deep check (slow pytest sweep) every N rounds.  A failure here
            # cancels any "improved" verdict and does NOT update best_metric,
            # regardless of how the cheap metric moved.
            deep_ran = False
            deep_passed: bool | None = None
            deep_exit: int | None = None
            should_deep_check = (
                args.deep_check_cmd
                and args.deep_check_every > 0
                and (i % args.deep_check_every == 0)
                and not args.dry_run
            )
            if should_deep_check:
                deep_log = round_dir / "deep_check.log"
                deep_passed, deep_exit = run_deep_check(
                    args.deep_check_cmd, str(workspace),
                    args.deep_check_timeout, deep_log, args.gpu_pool,
                )
                deep_ran = True

            improved = False
            metric_qualifies = (
                metric is not None
                and (state.best_metric is None or metric > (state.best_metric + args.min_delta))
            )
            if metric_qualifies and (not deep_ran or deep_passed):
                state.best_metric = metric
                state.best_sha = sha_after
                state.rounds_without_improvement = 0
                improved = True
            else:
                state.rounds_without_improvement += 1
                if metric_qualifies and deep_ran and not deep_passed:
                    print(
                        f"[round {i}] cheap metric improved to {metric} but deep check FAILED "
                        f"(exit={deep_exit}); refusing to update best_metric.",
                        flush=True,
                    )

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
                deep_check_ran=deep_ran,
                deep_check_passed=deep_passed,
                deep_check_exit_code=deep_exit,
            )
            state.rounds.append(result)
            write_summary(summary_path, args, state, baseline)

            deep_str = ""
            if deep_ran:
                deep_str = (
                    f" deep_check={'PASS' if deep_passed else 'FAIL'}"
                    f"(exit={deep_exit})"
                )
            print(
                f"[round {i}] metric={metric} best={state.best_metric} improved={improved}"
                f"{deep_str} streak={state.rounds_without_improvement}/{args.patience} "
                f"duration={duration:.1f}s",
                flush=True,
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
