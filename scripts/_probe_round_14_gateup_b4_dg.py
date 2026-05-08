#!/usr/bin/env python3
"""R14 chunk_size sweep for GateUP-B4-M2048 dgrad-via-H4 cell.

Cell: tiles_m=8, tiles_n=11 (post-H4-reroute), k=5760, m_total=8192
Config: gm=8, num_xcds=None (default 8), num_slots=200

With xcds=8 + slots=200:
  cs=64 → block=512, limit=0     → swizzle NO-OP
  cs=32 → block=256, limit=0     → swizzle NO-OP (still > slots)
  cs=25 → block=200=slots, limit=200 → 1 clean chunk (25 PIDs/XCD)
  cs=12 → block=96, limit=192    → 192 chunked + 8 unchanged
  cs=16 → block=128, limit=128   → 128 chunked + 72 unchanged
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


def _bench_p20(fn, warmup=20, iters=2000):
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


def _patch_hk_dscale(value):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale

    def wrapped(*args, **kwargs):
        kwargs.setdefault("chunk_size", value)
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore_hk_dscale(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_dgrad(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_impl(
            g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call)
    return flops / (ms * 1e9), ms


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


def run(label, time_fn, B, M, N, K, chunk_sizes, seeds=(42, 137, 2024, 99, 100)):
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'cs':>4}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for cs in chunk_sizes:
        orig = _patch_hk_dscale(cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_fn(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cs] = (med, min(seed_meds), max(seed_meds))
        _restore_hk_dscale(orig)

    base_ms = results[64][0]
    for cs in chunk_sizes:
        med, lo, hi = results[cs]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *base" if cs == 64 else ""
        print(f"  {cs:>4}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    best_cs = min(chunk_sizes, key=lambda cs: results[cs][0])
    best_med = results[best_cs][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: cs={best_cs}  ({lift:+.2f}% over default 64)")
    return results, best_cs, lift


if __name__ == "__main__":
    print(f"[probe] R14 GateUP-B4-M2048 dgrad-via-H4 chunk_size sweep")
    print(f"        Cell: gm=8, xcds=None(=8), slots=200 [R10 lever]")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    chunk_sizes = [12, 16, 24, 25, 32, 48, 50, 64]

    t0 = time.monotonic()
    res_dg, best_dg, lift_dg = run(
        "GateUP-B4-M2048 dgrad-via-H4", time_dgrad,
        B=4, M=2048, N=5760, K=2880,
        chunk_sizes=chunk_sizes,
    )

    # Counter check: same cell handles GateUP-B4-M2048 fwd? Let me also
    # probe fwd for sanity (different cell — tiles_n=22, tiles_m=8).
    res_fwd, best_fwd, lift_fwd = run(
        "GateUP-B4-M2048 fwd (sanity)", time_fwd,
        B=4, M=2048, N=5760, K=2880,
        chunk_sizes=chunk_sizes,
    )

    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")
    print(f"\n[probe] SUMMARY:")
    print(f"  dgrad-via-H4: best cs={best_dg}, lift={lift_dg:+.2f}%")
    print(f"  fwd:          best cs={best_fwd}, lift={lift_fwd:+.2f}%")
