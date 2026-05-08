#!/usr/bin/env python3
"""R15: Fresh retune of GateUP-B32-M4096 fwd RCR cell (lowest ratio 1.087).

Current rule (R70): gm=8, xcds=4 picked from a (gm ∈ {1..24}) × (xcds ∈
{1,2,4,8,16,32}) sweep with +1.39pp on M=4096. That was a few binding
rebuilds ago; multiple notes in the codebase document "kernel-rebuild
drift" (R30/R31/R32/R45/R50). Re-test with R14-class methodology.

Cell key: tiles_n=22 + tiles_m=16 + k=2880 + m_total>=65536. Hits both
GateUP-B32-M2048 (m_total=65536, tiles_m=8 — DIFFERENT key, R69 gate)
and GateUP-B32-M4096 (m_total=131072, tiles_m=16). The R70 rule covers
tiles_m ∈ {8, 16}, so a re-tune affects BOTH B32 GateUP shapes — must
verify both.
"""
import os
import sys
import statistics
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module
import _metric_hk_ratio as hk_ratio

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _bench_p20(fn, warmup=20, iters=1500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


def _patch_rcr(group_m=None, num_xcds=None, num_slots=None, chunk_size=None):
    """Patch hk.grouped_rcr_dscale to override scheduling args.

    group_m is passed as positional arg #7; intercept via *args replacement.
    The other knobs are kwargs.
    """
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale

    def wrapped(*args, **kwargs):
        if group_m is not None:
            args = list(args)
            if len(args) >= 7:
                args[6] = group_m
            else:
                kwargs["group_m"] = group_m
        if num_xcds is not None:
            kwargs["num_xcds"] = num_xcds
        if num_slots is not None:
            kwargs["num_slots"] = num_slots
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore_rcr(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_fwd(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_impl(
            a_fp8, b_fp8, a_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call)
    return flops / (ms * 1e9), ms


def run_grid(label, B, M, N, K, candidates, seeds=(42, 137, 2024)):
    """candidates: list of (gm, xcds) tuples."""
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'gm':>3} {'xcds':>4}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for (gm, xc) in candidates:
        orig = _patch_rcr(group_m=gm, num_xcds=xc)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_fwd(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[(gm, xc)] = (med, min(seed_meds), max(seed_meds))
        _restore_rcr(orig)

    base_key = (8, 4)  # R70 baseline
    base_ms = results[base_key][0] if base_key in results else min(r[0] for r in results.values())
    sorted_cells = sorted(candidates, key=lambda c: results[c][0])
    for (gm, xc) in sorted_cells:
        med, lo, hi = results[(gm, xc)]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *R70" if (gm, xc) == base_key else ""
        print(f"  {gm:>3} {xc:>4}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    best = sorted_cells[0]
    best_med = results[best][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: cell={best}  ({lift:+.2f}% over R70 (8,4))")
    return results, best, lift


if __name__ == "__main__":
    print(f"[probe] R15 GateUP-B32-M{{2048,4096}} fwd RCR re-tune")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    # Wide candidate grid centered on R70's (8, 4)
    candidates = [
        (1, 2), (1, 4), (1, 8),
        (2, 2), (2, 4), (2, 8),
        (4, 2), (4, 4), (4, 8),
        (6, 4), (8, 2), (8, 4), (8, 8),
        (10, 4), (12, 4), (14, 4), (16, 4),
        (16, 2), (16, 8),
        (24, 4), (32, 4),
    ]

    t0 = time.monotonic()
    res_4096, best_4096, lift_4096 = run_grid(
        "GateUP-B32-M4096 fwd",
        B=32, M=4096, N=5760, K=2880,
        candidates=candidates,
    )
    res_2048, best_2048, lift_2048 = run_grid(
        "GateUP-B32-M2048 fwd",
        B=32, M=2048, N=5760, K=2880,
        candidates=candidates,
    )
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")

    print(f"\n[probe] SUMMARY:")
    print(f"  GateUP-B32-M4096 fwd: best={best_4096}, lift={lift_4096:+.2f}% over R70 (8,4)")
    print(f"  GateUP-B32-M2048 fwd: best={best_2048}, lift={lift_2048:+.2f}% over R70 (8,4)")
