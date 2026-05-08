#!/usr/bin/env python3
"""R15: var-K chunk_size audit on the 6 unaudited wgrad cells.

R13 only shipped chunk_size=96 for Down-B4 wgrad family (xcds=2 cell).
The other 6 metric wgrad cells use xcds=4 + slots=256 (clean cs=64
partition) — but finer chunk_size (32, 16) interleaves the 256 PIDs
across XCDs differently, which may have subtle L2 effects.

Cells to probe:
- GateUP-B4-M2048 wgrad: vk_group_m=1, vk_num_xcds=4, slots=256
- GateUP-B4-M4096 wgrad: same cell key (m_total=16384 → R9-A: gm=4, xcds=4)
- Down-B32-M2048 wgrad: vk_group_m=8, vk_num_xcds=4, slots=256
- Down-B32-M4096 wgrad: vk_group_m=4, vk_num_xcds=4, slots=256
- GateUP-B32-M2048 wgrad: vk_group_m=1, vk_num_xcds=4, slots=256
- GateUP-B32-M4096 wgrad: same cell

Sweep chunk_size ∈ {16, 32, 48, 64, 96, 128}.
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
    grouped_gemm_fp8_variable_k_impl,
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


def _patch_var_k(chunk_size):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_variable_k_crr_dscale

    def wrapped(*args, **kwargs):
        kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_variable_k_crr_dscale", wrapped)
    return orig


def _restore_var_k(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_variable_k_crr_dscale", orig)


def time_wgrad_var_k(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)
    x_fp8, x_s = quantize_fp8(x, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_variable_k_impl(
            x_fp8, g_out_fp8, x_s, g_out_s, g_lens, g_offs,
            trans_a=True, trans_b=False, trans_c=False,
            out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call)
    return flops / (ms * 1e9), ms


def run_cs_sweep(label, B, M, N, K, cs_list, seeds=(42, 137, 2024)):
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'cs':>4}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for cs in cs_list:
        orig = _patch_var_k(cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_wgrad_var_k(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cs] = (med, min(seed_meds), max(seed_meds))
        _restore_var_k(orig)

    base_ms = results[64][0]  # default
    sorted_cs = sorted(cs_list, key=lambda c: results[c][0])
    for cs in cs_list:
        med, lo, hi = results[cs]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *base" if cs == 64 else ""
        print(f"  {cs:>4}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    best = sorted_cs[0]
    best_med = results[best][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: cs={best}  ({lift:+.2f}% over default 64)")
    return results, best, lift


if __name__ == "__main__":
    print(f"[probe] R15 var-K chunk_size audit on 6 unaudited wgrad cells")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    cs_list = [16, 32, 48, 64, 96, 128]

    cells = [
        ("GateUP-B4-M2048 wgrad",  4, 2048, 5760, 2880),
        ("GateUP-B4-M4096 wgrad",  4, 4096, 5760, 2880),
        ("Down-B32-M2048 wgrad",   32, 2048, 2880, 2880),
        ("Down-B32-M4096 wgrad",   32, 4096, 2880, 2880),
        ("GateUP-B32-M2048 wgrad", 32, 2048, 5760, 2880),
        ("GateUP-B32-M4096 wgrad", 32, 4096, 5760, 2880),
    ]

    summary = []
    t0 = time.monotonic()
    for label, B, M, N, K in cells:
        _, best, lift = run_cs_sweep(label, B, M, N, K, cs_list)
        summary.append((label, best, lift))
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")

    print(f"\n[probe] SUMMARY:")
    for label, best, lift in summary:
        marker = " *WIN" if lift >= 0.5 else " (marginal)" if lift > 0.1 else " *NO WIN"
        print(f"  {label:30s}  best cs={best:>3}, lift={lift:+.2f}%{marker}")
