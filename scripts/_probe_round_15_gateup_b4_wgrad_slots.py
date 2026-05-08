#!/usr/bin/env python3
"""R15: GateUP-B4 wgrad var-K num_slots audit (+ chunk_size if num_slots wins).

Cell: vk_group_m=1, vk_num_xcds=4, vk_num_slots=0 → kernel slots=NUM_CUS=256
Tile-step density: 22*11*4 = 968 / 256 = 3.78 ws/CU (borderline).

R3 had counter-evidence at saturated B=32 (slots=192 → -17%) but never
tested borderline B=4 GateUP. Probe slots ∈ {160, 176, 192, 200, 208,
220, 256} mirror of R10/R11 methodology.

If a num_slots reduction WINS, also probe chunk_size at the new cell —
if reduced slots leaves swizzle NO-OP at cs=64, an aligned chunk_size
might double-dip (R14 RCR pattern).
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


def _patch_var_k(num_slots=None, chunk_size=None):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_variable_k_crr_dscale

    def wrapped(*args, **kwargs):
        if num_slots is not None:
            kwargs["num_slots"] = num_slots
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_variable_k_crr_dscale", wrapped)
    return orig


def _restore_var_k(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_variable_k_crr_dscale", orig)


def time_wgrad_var_k(B, M, N, K):
    """Time wgrad var-K via the impl path. M is M_per_g, K is K_fwd."""
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


def run_slots_sweep(label, B, M, N, K, slots_list, seeds=(42, 137, 2024)):
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  Cell tile-steps = (N/256)*(K/256)*B = {(N//256)*(K//256)*B}, slots/CU={(N//256)*(K//256)*B/256:.2f}")
    print(f"\n  {'slots':>5}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for slots in slots_list:
        if slots == 0:
            orig = _patch_var_k(num_slots=None)  # default
        else:
            orig = _patch_var_k(num_slots=slots)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_wgrad_var_k(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[slots] = (med, min(seed_meds), max(seed_meds))
        _restore_var_k(orig)

    base_ms = results[0][0]  # slots=0 (default = 256)
    for slots in slots_list:
        med, lo, hi = results[slots]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *base(=256)" if slots == 0 else ""
        print(f"  {slots:>5}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    best_slots = min(slots_list, key=lambda s: results[s][0])
    best_med = results[best_slots][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: slots={best_slots}  ({lift:+.2f}% over default 256)")
    return results, best_slots, lift


if __name__ == "__main__":
    print(f"[probe] R15 GateUP-B4 wgrad var-K num_slots audit")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    # Cell: vk_group_m=1, vk_num_xcds=4, slots=NUM_CUS=256
    # Tile-step density 968/256 = 3.78 ws/CU (borderline; R3 said
    # "already moderately amortised" but never directly tested).
    slots_list = [0, 160, 192, 200, 208, 220, 240, 256]

    t0 = time.monotonic()
    res_2048, best_2048, lift_2048 = run_slots_sweep(
        "GateUP-B4-M2048 wgrad", B=4, M=2048, N=5760, K=2880,
        slots_list=slots_list,
    )
    res_4096, best_4096, lift_4096 = run_slots_sweep(
        "GateUP-B4-M4096 wgrad", B=4, M=4096, N=5760, K=2880,
        slots_list=slots_list,
    )
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")

    print(f"\n[probe] SUMMARY:")
    print(f"  GateUP-B4-M2048: best slots={best_2048}, lift={lift_2048:+.2f}%")
    print(f"  GateUP-B4-M4096: best slots={best_4096}, lift={lift_4096:+.2f}%")
