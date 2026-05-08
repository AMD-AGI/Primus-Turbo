#!/usr/bin/env python3
"""R16: After R15 chunk_size=24 ship, re-tune (gm) on GateUP-B4-M2048 dA cell.

Cell now has: gm=8, xcds=None=8, slots=200, chunk_size=24. Question: does
the chunk_size=24 partition (1 clean chunk over 192 of 200 PIDs) shift
the optimal gm? R10 picked gm=8 (matching Down-B4) but that was tested
WITHOUT chunk_size override.
"""
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


def _bench(fn, warmup=20, iters=2000):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(); se.record(); fn(); ee.record(); torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


def _patch(group_m=None, num_xcds=None, num_slots=None, chunk_size=None):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale
    def wrapped(*args, **kwargs):
        if group_m is not None:
            args = list(args)
            if len(args) >= 7: args[6] = group_m
            else: kwargs["group_m"] = group_m
        if num_xcds is not None: kwargs["num_xcds"] = num_xcds
        if num_slots is not None: kwargs["num_slots"] = num_slots
        if chunk_size is not None: kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)
    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_dgrad(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B*M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8, _GRAN)
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
    return flops / (ms * 1e9), ms


if __name__ == "__main__":
    print(f"[probe] R16 GateUP-B4-M2048 dA (gm) re-tune at slots=200 cs=24")
    candidates = [
        # (gm, xcds, slots, cs) — keep slots=200 + cs=24 fixed (R10+R15)
        (8,  None, 200, 24),  # R15 ship
        (1,  None, 200, 24),
        (2,  None, 200, 24),
        (4,  None, 200, 24),
        (12, None, 200, 24),
        (16, None, 200, 24),
        (24, None, 200, 24),
        # also probe with num_xcds explicit (R10 has num_xcds=None → kernel default 8)
        (8,  8,    200, 24),
        # control: R15 ship without chunk_size (= R10 baseline)
        (8,  None, 200, None),
    ]
    seeds = (42, 137, 2024, 99, 100)
    flops = 2.0 * 4 * 2048 * 5760 * 2880
    print(f"\n  {'gm':>3} {'xcds':>4} {'slots':>5} {'cs':>4}  {'med ms':>9}  {'TFLOPS':>8}  {'Δ%':>6}  {'pos/n':>5}")
    results = {}
    for cfg in candidates:
        gm, xc, sl, cs = cfg
        orig = _patch(group_m=gm, num_xcds=xc, num_slots=sl, chunk_size=cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_dgrad(4, 2048, 5760, 2880)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cfg] = (med, seed_meds)
        _restore(orig)
    base_key = (8, None, 200, 24)
    base_ms = results[base_key][0]
    base_per_seed = results[base_key][1]
    sorted_k = sorted(candidates, key=lambda c: results[c][0])
    for cfg in sorted_k:
        gm, xc, sl, cs = cfg
        med, per_seed = results[cfg]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        pos = sum(1 for c, b in zip(per_seed, base_per_seed) if c < b) if cfg != base_key else 0
        cs_str = "def" if cs is None else str(cs)
        xc_str = "def" if xc is None else str(xc)
        marker = " *R15" if cfg == base_key else ""
        print(f"  {gm:>3} {xc_str:>4} {sl:>5} {cs_str:>4}  {med:>9.5f}  {tflops:>8.1f}  {delta_pp:+6.2f}  {pos}/{len(seeds)}{marker}")
