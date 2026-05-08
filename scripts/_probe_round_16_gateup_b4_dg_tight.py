#!/usr/bin/env python3
"""R16 tight verify: gm=1 vs gm=8 at slots=200 cs=24 on GateUP-B4-M2048 dA."""
import os, sys, statistics, time
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

_FP8 = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _bench(fn, warmup=30, iters=3000):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(); se.record(); fn(); ee.record(); torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


def _patch(group_m, num_slots, chunk_size):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale
    def wrapped(*args, **kwargs):
        args = list(args)
        if len(args) >= 7: args[6] = group_m
        else: kwargs["group_m"] = group_m
        kwargs["num_slots"] = num_slots
        kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)
    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_dgrad(B, M, N, K, gm, sl, cs):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B*M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8, _GRAN)
    orig = _patch(gm, sl, cs)
    def _call():
        return grouped_gemm_fp8_impl(
            g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )
    flops = 2.0 * (B*M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench(_call)
    _restore(orig)
    return flops / (ms * 1e9), ms


if __name__ == "__main__":
    print(f"[probe] R16 tight: gm=1 vs gm=8 @ slots=200 cs=24")
    seeds = (42, 137, 2024, 99, 100, 7, 365)
    flops = 2.0 * 4 * 2048 * 5760 * 2880

    print(f"\n  cell: (gm, slots, cs)")
    print(f"  {'gm':>3}  {'med ms':>9}  {'min ms':>9}  {'max ms':>9}  {'spread%':>8}  {'TFLOPS':>8}  {'Δ%':>6}  {'wmin/lmax':>10}")
    results = {}
    for gm in (1, 8):
        per_seed = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_dgrad(4, 2048, 5760, 2880, gm, 200, 24)
            per_seed.append(ms)
        results[gm] = per_seed

    base_ms = statistics.median(results[8])
    base_max = max(results[8])
    base_min = min(results[8])
    for gm in (8, 1):
        per_seed = results[gm]
        med = statistics.median(per_seed)
        lo = min(per_seed); hi = max(per_seed)
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        spread = (hi - lo) / med * 100
        if gm == 1:
            wmin_beats_lmax = "YES" if hi < base_min else f"no (max={hi:.5f} > base_min={base_min:.5f})"
        else:
            wmin_beats_lmax = "(base)"
        marker = " *R15" if gm == 8 else ""
        print(f"  {gm:>3}  {med:>9.5f}  {lo:>9.5f}  {hi:>9.5f}  {spread:>7.3f}  {tflops:>8.1f}  {delta_pp:+6.2f}  {wmin_beats_lmax}{marker}")

    # bit-equivalence check
    print(f"\n  bit-equivalence verify (gm=1 vs gm=8 @ slots=200 cs=24):")
    for seed in (42, 137, 2024):
        torch.manual_seed(seed)
        g_lens = torch.full((4,), 2048, dtype=torch.int64, device="cuda")
        g_offs = grouped_gemm_compute_offs(g_lens)
        grad_out = torch.randn((8192, 5760), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((4, 5760, 2880), dtype=torch.bfloat16, device="cuda")
        g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8, _GRAN)
        b_fp8, b_s = quantize_fp8(b, _FP8, _GRAN)
        with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
            orig8 = _patch(8, 200, 24)
            out8 = grouped_gemm_fp8_impl(
                g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
                trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
                granularity=_GRAN.value, num_cu=None,
                default_backend=BackendType.HIPKITTEN.value,
            ).clone()
            _restore(orig8)
            orig1 = _patch(1, 200, 24)
            out1 = grouped_gemm_fp8_impl(
                g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
                trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
                granularity=_GRAN.value, num_cu=None,
                default_backend=BackendType.HIPKITTEN.value,
            ).clone()
            _restore(orig1)
        diff = (out8 - out1).abs().max().item()
        eq = torch.equal(out8, out1)
        print(f"    seed={seed}: max_abs_diff={diff:.6e}, bit_eq={eq}")
