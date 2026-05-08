#!/usr/bin/env python3
"""R15 tight verify of R14 marginal finding: GateUP-B4-M2048 dgrad-via-H4
chunk_size lever (cell xcds=8, slots=200; R10 num_slots cell).

R14 quick probe found cs=24 wins +0.72% over default cs=64. Within
±0.5pp of R14's noise floor — needs tighter verify before shipping.
Methodology: 2500-iter × p20 × 7 seeds.
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


def _bench_p20(fn, warmup=30, iters=2500):
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


def _patch_rcr(chunk_size):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale

    def wrapped(*args, **kwargs):
        kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore_rcr(orig):
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


if __name__ == "__main__":
    print(f"[probe] R15 tight verify GateUP-B4-M2048 dgrad-via-H4 chunk_size")
    print(f"        (cell xcds=8, slots=200; R10 num_slots lever)")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    cs_list = [None, 16, 24, 25, 32, 50]
    seeds = (42, 137, 2024, 99, 100, 7, 365)
    flops = 2.0 * 4 * 2048 * 5760 * 2880

    print(f"\n  {'cs':>4}  {'med ms':>9}  {'min ms':>9}  {'max ms':>9}  {'spread%':>8}  {'TFLOPS':>8}  {'Δ%':>6}  {'pos/n':>5}")
    results = {}
    for cs in cs_list:
        if cs is None:
            orig = _patch_rcr(0)  # 0 → kernel default 64
        else:
            orig = _patch_rcr(cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_dgrad(4, 2048, 5760, 2880)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cs] = (med, min(seed_meds), max(seed_meds), seed_meds)
        _restore_rcr(orig)

    base_ms = results[None][0]
    base_per_seed = results[None][3]
    for cs in cs_list:
        med, lo, hi, per_seed = results[cs]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        spread = (hi - lo) / med * 100
        # count seeds where this cs beats baseline
        if cs is None:
            pos = 0
            n = 0
        else:
            pos = sum(1 for c, b in zip(per_seed, base_per_seed) if c < b)
            n = len(per_seed)
        cs_str = "def(64)" if cs is None else str(cs)
        marker = " *base" if cs is None else ""
        print(f"  {cs_str:>4}  {med:>9.5f}  {lo:>9.5f}  {hi:>9.5f}  {spread:>7.3f}  {tflops:>8.1f}  {delta_pp:+6.2f}  {pos}/{n}{marker}")

    print(f"\n  per-seed times for default (cs=64): {[f'{t:.5f}' for t in base_per_seed]}")
    for cs in cs_list[1:]:
        print(f"  per-seed times for cs={cs}: {[f'{t:.5f}' for t in results[cs][3]]}")
