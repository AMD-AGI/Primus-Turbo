#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lightweight grouped MXFP8 perf + correctness probe used as the
``auto_optimize.py`` loop metric.

Single-process, no pytest, no xdist, ~10-20 s wall.  Drives the persistent
forward + variable-K wgrad MXFP8 grouped-GEMM kernels through three
representative shapes, measures TFLOPS, checks SNR vs the bf16 reference,
and runs a short determinism stress on the worst-known shape.

**Stdout contract:** on normal completion (including "no CUDA" fallback),
exactly **one line** is printed to stdout: the integer ``score``.
All diagnostics (pick banner, traceback, per-shape notes, summary line)
go to stderr.

Score formula:

    score = int(round(sum_tflops * 10)) \\
          - SNR_FAIL_PENALTY    * snr_fail_count \\
          - STRESS_BAD_PENALTY  * stress_bad_count \\
          - EXCEPTION_PENALTY   * exception_count

Higher is better.  Defaults are tuned so:

  * ``sum_tflops`` for the current persistent kernel (post non-volatile
    C-store fix) is around 7000-9000 -> score ~70000-90000.
  * One SNR failure costs the same as ~100 TFLOPS = a meaningful chunk of
    one shape's contribution, so any correctness regression sinks the
    metric well below the best-so-far.
  * One stress BAD costs ~10 TFLOPS, so the loop rewards reducing
    determinism failure rate.

If ``HIP_VISIBLE_DEVICES`` is unset, picks an idle GPU from ``MXFP8_GPU_POOL``
via ``rocm-smi --showuse --showpids`` (KFD VRAM above the busy threshold).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager

# Score weights -- tweak via env if you want a different penalty profile.
SNR_FAIL_PENALTY   = int(os.environ.get("MXFP8_SNR_FAIL_PENALTY", "1000"))
STRESS_BAD_PENALTY = int(os.environ.get("MXFP8_STRESS_BAD_PENALTY", "100"))
EXCEPTION_PENALTY  = int(os.environ.get("MXFP8_EXCEPTION_PENALTY", "2000"))

# SNR thresholds (dB).  Match the values used in
# benchmark/ops/bench_grouped_gemm_turbo.py: 25 for E4M3, 20 for E5M2.
SNR_E4M3 = float(os.environ.get("MXFP8_SNR_E4M3", "25.0"))
SNR_E5M2 = float(os.environ.get("MXFP8_SNR_E5M2", "20.0"))

# Per-call timing knobs.  Each shape is measured by running PERF_TRIALS
# independent batches of PERF_BATCH_ITERS calls between two CUDA events,
# and we report the MIN average-per-call across batches.
#
# Min-of-N is the right reduction here because cross-batch latency variance
# on a shared MI355 host comes almost entirely from one-off scheduler /
# memory-bandwidth interference from other GPUs on the same node, not from
# real kernel slowdown -- the fastest of N batches reflects the kernel's
# true steady-state throughput.  Empirically PERF_TRIALS=8 brings the
# cross-run score variance from ~50% down to ~10%.
PERF_WARMUP        = int(os.environ.get("MXFP8_PERF_WARMUP", "20"))
PERF_TRIALS        = int(os.environ.get("MXFP8_PERF_TRIALS", "8"))
PERF_BATCH_ITERS   = int(os.environ.get("MXFP8_PERF_BATCH_ITERS", "30"))

# Determinism stress: how many fwd+bwd iters and what threshold counts as BAD.
# Default 100 iters: at the current ~2-6% race rate one BAD-iter swing equals
# one penalty unit, so 100 trials gives an SNR roughly 4x better than 50
# trials for the same wall.  Target for the loop is <= 2/100 (i.e. <= 2%).
STRESS_ITERS  = int(os.environ.get("MXFP8_STRESS_ITERS", "100"))
STRESS_THRESH = float(os.environ.get("MXFP8_STRESS_THRESH", "1.0"))

# rocm-smi KFD VRAM column: above this byte count, the GPU counts as busy.
KFD_BUSY_VRAM_BYTES = 100 * 1024 * 1024


# Allowed GPU pool.  We are sharing the host with another tenant on GPUs
# 0-3, so this script (and anything it auto-picks for) is restricted to
# GPUs 4-7.  Override via MXFP8_GPU_POOL=2,3,5 (comma-separated) if needed.
GPU_POOL = sorted({
    int(g) for g in os.environ.get("MXFP8_GPU_POOL", "4,5,6,7").split(",") if g.strip()
})


def _pick_idle_gpu() -> str | None:
    """Smallest idle GPU id in ``GPU_POOL`` (busy if KFD lists a PID with
    VRAM ``> KFD_BUSY_VRAM_BYTES``).  If rocm-smi fails, return the first pool id.
    """
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showpids"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except Exception:
        return str(GPU_POOL[0]) if GPU_POOL else None
    busy: set[int] = set()
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
        if vram <= KFD_BUSY_VRAM_BYTES:
            continue
        for gid in re.findall(r"\d+", cols[2]):
            busy.add(int(gid))
    idle = [g for g in GPU_POOL if g not in busy]
    if idle:
        return str(idle[0])
    return str(GPU_POOL[0]) if GPU_POOL else None


# Pick an idle GPU before importing torch so primus_turbo binds to it.
if "HIP_VISIBLE_DEVICES" not in os.environ:
    pick = _pick_idle_gpu()
    if pick is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = pick
        print(f"[metric_mxfp8] auto-picked HIP_VISIBLE_DEVICES={pick}", file=sys.stderr)


import torch  # noqa: E402

import primus_turbo  # noqa: E402, F401
import primus_turbo.pytorch  # noqa: E402, F401
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8  # noqa: E402


def compute_snr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    ref = ref.detach().float()
    actual = actual.detach().float()
    diff = ref - actual
    sig = (ref * ref).mean().clamp_min(1e-30)
    noise = (diff * diff).mean().clamp_min(1e-30)
    return 10.0 * float(torch.log10(sig / noise).item())


def _grouped_gemm_ref(a: torch.Tensor, b: torch.Tensor,
                      group_lens: torch.Tensor, trans_b: bool) -> torch.Tensor:
    """Reference grouped GEMM (per-group bf16 matmul)."""
    group_lens_cpu = group_lens.detach().cpu().tolist()
    out_chunks = []
    start = 0
    for gi, sz in enumerate(group_lens_cpu):
        rhs = b[gi].t() if trans_b else b[gi]
        out_chunks.append(a[start:start + sz] @ rhs)
        start += sz
    return torch.cat(out_chunks, dim=0)


@contextmanager
def _no_grad_for_fwd(a: torch.Tensor, b: torch.Tensor):
    a_was = a.requires_grad
    b_was = b.requires_grad
    a.requires_grad_(False)
    b.requires_grad_(False)
    try:
        yield
    finally:
        a.requires_grad_(a_was)
        b.requires_grad_(b_was)


def _best_of_n_seconds(fn) -> float:
    """Return the minimum batch-average per-call latency across PERF_TRIALS
    batches of PERF_BATCH_ITERS calls each, timed via CUDA events.  Min-of-N
    is robust to one-off CPU / clock jitter that otherwise inflates variance
    to ~50% across runs.
    """
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    best_ms = float("inf")
    for _ in range(PERF_TRIALS):
        torch.cuda.synchronize()
        start.record()
        for _ in range(PERF_BATCH_ITERS):
            fn()
        stop.record()
        stop.synchronize()
        avg_ms = start.elapsed_time(stop) / PERF_BATCH_ITERS
        if avg_ms < best_ms:
            best_ms = avg_ms
    return best_ms / 1000.0


def _bench_fwd(a, b, group_lens, config) -> float:
    """Return forward-only TFLOPS (kernel + wrapper, best of N trials)."""
    G, N, K = b.shape
    total_m = a.size(0)
    fn = lambda: grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    with _no_grad_for_fwd(a, b):
        for _ in range(PERF_WARMUP):
            fn()
        torch.cuda.synchronize()
        elapsed = _best_of_n_seconds(fn)
    flops = 2.0 * total_m * N * K
    return flops / elapsed / 1e12


def _bench_fwd_bwd(a, b, group_lens, config, grad_out) -> tuple[float, float]:
    """Return (fwd, bwd) TFLOPS measured separately, best of N trials."""
    G, N, K = b.shape
    total_m = a.size(0)
    fwd_flops = 2.0 * total_m * N * K
    bwd_flops = 2.0 * fwd_flops
    a.requires_grad_(True)
    b.requires_grad_(True)
    fwd_fn = lambda: grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    out = fwd_fn()
    bwd_fn = lambda: out.backward(grad_out, retain_graph=True)
    for _ in range(PERF_WARMUP):
        fwd_fn()
        bwd_fn()
    torch.cuda.synchronize()
    fwd_elapsed = _best_of_n_seconds(fwd_fn)
    bwd_elapsed = _best_of_n_seconds(bwd_fn)
    a.grad = None
    b.grad = None
    return fwd_flops / fwd_elapsed / 1e12, bwd_flops / bwd_elapsed / 1e12


def _correctness_probe(a, b, group_lens, config, snr_thresh) -> tuple[bool, str]:
    """Run forward + backward once, compare against bf16 reference, return SNR results."""
    a = a.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = _grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)
    out_snr = compute_snr(out_ref, out)
    da_snr  = compute_snr(a_ref.grad, a.grad)
    db_snr  = compute_snr(b_ref.grad, b.grad)
    ok = (out_snr >= snr_thresh and da_snr >= snr_thresh and db_snr >= snr_thresh)
    return ok, f"out={out_snr:.1f} dA={da_snr:.1f} dB={db_snr:.1f} (>={snr_thresh})"


def _stress_probe(label, G, M, N, K, fmt, n_iters) -> tuple[int, str]:
    """Run forward + backward repeatedly with same input; return BAD count vs first run."""
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    total_m = G * M
    group_lens = torch.full((G,), M, dtype=torch.int64, device=DEVICE)
    a_base = torch.randn(total_m, K, dtype=DTYPE, device=DEVICE)
    b_base = torch.randn(G, N, K, dtype=DTYPE, device=DEVICE)
    grad_out = torch.randn(total_m, N, dtype=DTYPE, device=DEVICE)
    config = Float8QuantConfig(
        format=fmt, granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32, scale_dtype=ScaleDtype.E8M0,
    )

    def run_once():
        a = a_base.detach().clone().requires_grad_(True)
        b = b_base.detach().clone().requires_grad_(True)
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
        out.backward(grad_out)
        torch.cuda.synchronize()
        return out.detach().clone(), a.grad.detach().clone(), b.grad.detach().clone()

    ref_out, ref_da, ref_db = run_once()
    bad = 0
    for _ in range(n_iters):
        out, da, db = run_once()
        d = max(
            (ref_out.float() - out.float()).abs().max().item(),
            (ref_da.float() - da.float()).abs().max().item(),
            (ref_db.float() - db.float()).abs().max().item(),
        )
        if d > STRESS_THRESH:
            bad += 1
    return bad, f"{label} {bad}/{n_iters} BAD"


# ----------------------------------------------------------------------------
# Probe set: representative shapes from DeepSeek-V3 + gpt_oss_20B.
# ----------------------------------------------------------------------------
#
# Each entry is (label, G, M, N, K, fmt, snr_thresh, do_bwd).  M-large and
# odd-N shapes are both important, so we include one of each.

SHAPES = [
    # DeepSeek-V3 GateUP B=16 -- largest typical shape, dominates the score.
    ("DSv3-GateUP-B16", 16, 2048, 4096, 7168, Format.E4M3, SNR_E4M3, True),
    # DeepSeek-V3 Down B=16 -- complementary K layout.
    ("DSv3-Down-B16",   16, 2048, 7168, 2048, Format.E4M3, SNR_E4M3, True),
    # gpt_oss_20B Down -- odd N=2880 to exercise the boundary tile path.
    ("gpt-oss-Down-B4",  4, 2048, 2880, 2880, Format.E4M3, SNR_E4M3, True),
    # E5M2 sanity probe -- forward only, smaller shape to keep wall short.
    ("DSv3-GateUP-B4-E5", 4, 2048, 4096, 7168, Format.E5M2, SNR_E5M2, False),
]

# The "worst-known" shape from earlier stress sweeps -- exercises the (G=4,
# M=1024, N=2048, K=2048) FWD `out` race plus the dB wgrad race at relatively
# high frequency.  Keeps the loop sensitive to determinism regressions.
STRESS_SHAPE = ("stress-G4-M1024-N2048-K2048-E4M3", 4, 1024, 2048, 2048, Format.E4M3)


def _build_inputs(G, M, N, K, fmt):
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
    total_m = G * M
    group_lens = torch.full((G,), M, dtype=torch.int64, device=DEVICE)
    a = torch.randn(total_m, K, dtype=DTYPE, device=DEVICE)
    b = torch.randn(G, N, K, dtype=DTYPE, device=DEVICE)
    grad_out = torch.randn(total_m, N, dtype=DTYPE, device=DEVICE)
    config = Float8QuantConfig(
        format=fmt, granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32, scale_dtype=ScaleDtype.E8M0,
    )
    return a, b, group_lens, grad_out, config


def _global_warmup() -> None:
    """Run a few large GEMMs to drive the SOC clock to its boost state.

    Ensures the first measured shape doesn't get an artificially-good number
    from being timed while clock was ramping, which otherwise inflates the
    score noise across runs.
    """
    DEVICE = torch.device("cuda")
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=DEVICE)
    for _ in range(8):
        torch.matmul(a, b)
    torch.cuda.synchronize()


def main() -> int:
    if not torch.cuda.is_available():
        print(0)
        print("[metric_mxfp8] no CUDA available", file=sys.stderr)
        return 0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _global_warmup()

    sum_tflops = 0.0
    snr_fails = 0
    exceptions = 0
    notes: list[str] = []

    t0 = time.monotonic()
    for label, G, M, N, K, fmt, snr_thresh, do_bwd in SHAPES:
        try:
            a, b, group_lens, grad_out, config = _build_inputs(G, M, N, K, fmt)
            ok, snr_msg = _correctness_probe(a, b, group_lens, config, snr_thresh)
            if not ok:
                snr_fails += 1
                notes.append(f"  FAIL {label} snr {snr_msg}")
            else:
                notes.append(f"  OK   {label} snr {snr_msg}")
            if do_bwd:
                fwd_tf, bwd_tf = _bench_fwd_bwd(a, b, group_lens, config, grad_out)
                sum_tflops += fwd_tf + bwd_tf
                notes.append(f"  PERF {label} fwd={fwd_tf:.1f} bwd={bwd_tf:.1f} TFLOPS")
            else:
                fwd_tf = _bench_fwd(a, b, group_lens, config)
                sum_tflops += fwd_tf
                notes.append(f"  PERF {label} fwd-only={fwd_tf:.1f} TFLOPS")
            del a, b, group_lens, grad_out
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            exceptions += 1
            notes.append(f"  ERR  {label}: {exc!r}")
            traceback.print_exc(file=sys.stderr)

    # Determinism stress on the known-worst shape.
    stress_bad = 0
    try:
        s_label, sG, sM, sN, sK, s_fmt = STRESS_SHAPE
        stress_bad, stress_msg = _stress_probe(s_label, sG, sM, sN, sK, s_fmt, STRESS_ITERS)
        notes.append(f"  STR  {stress_msg}")
    except Exception as exc:  # noqa: BLE001
        exceptions += 1
        notes.append(f"  ERR  stress: {exc!r}")
        traceback.print_exc(file=sys.stderr)

    dt = time.monotonic() - t0
    score = (
        int(round(sum_tflops * 10))
        - SNR_FAIL_PENALTY * snr_fails
        - STRESS_BAD_PENALTY * stress_bad
        - EXCEPTION_PENALTY * exceptions
    )
    print(score)
    print(
        f"[metric_mxfp8] sum_tflops={sum_tflops:.1f} "
        f"snr_fail={snr_fails} stress_bad={stress_bad}/{STRESS_ITERS} "
        f"exc={exceptions} score={score} elapsed={dt:.1f}s",
        file=sys.stderr,
    )
    show_all = "--verbose" in sys.argv
    for n in notes:
        if show_all or n.lstrip().startswith(("FAIL", "ERR", "STR")):
            print(n, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
