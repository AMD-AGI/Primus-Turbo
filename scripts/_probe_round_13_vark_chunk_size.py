#!/usr/bin/env python3
"""Round 13 (gpt_oss FP8 kernel-only ceiling) — var-K chunk_size sweep.

R12 closed the FP8 RCR num_slots audit (3.0 ws/CU tier FALSIFIED). The R13
plan from that note: pivot to **chunk_size lever** in
``chiplet_transform_chunked``. Default is hardcoded 64 in the kernel call
(line ~7827 of kernel_fp8_layouts.cpp).

Hypothesis: at the current Down-B4-M2048 wgrad shipped cell
(gm=1, xcds=2, slots=192), the swizzle math is:

    block = num_xcds * chunk_size = 2 * 64 = 128
    limit = (slots / block) * block = (192 / 128) * 128 = 128

So **only the first 128 of 192 workgroups get chunked** — the trailing 64
fall through to round-robin (the ``if (workgroup_id > limit) return
workgroup_id`` early exit). chunk_size=96 with xcds=2 gives block=192 =
exactly slots → all 192 chunked in one clean chiplet-pair partition.

We probe chunk_size ∈ {16, 32, 48, 64, 96, 128, 192, 256} on the worst
metric shape (Down-B4-M2048 wgrad, m_total=8192, k=2880) and a sibling
(Down-B4-M4096 wgrad, m_total=16384). If any chunk_size beats the
shipped chunk_size=64 by >= 0.5pp on tight-verify median across 5 seeds
AND the spread is < median lift, ship a per-call rule. Otherwise
falsification.

Methodology mirror of R12: 1500-iter × 7-trial × 5-seed p20 with per-iter
cudaDeviceSynchronize. ~5 minutes wall.
"""
import os
import sys
import statistics
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _bench(fn, warmup, iters):
    """Per-iter sync p20 timer, mirror of R12 methodology."""
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
    return times[len(times) // 5]  # p20


def _bit_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a, b)


def make_wgrad_call(B, M, N, K, slots, gm, xcds, chunk_size):
    """Build a kernel-only var-K wgrad call with the given knobs.

    NOTE: Pass quantized FP8 tensors (not bf16) so the kernel reads valid
    FP8 bytes — required for the bit-eq correctness gate. (R15 probe
    was passing bf16 → output was garbage NaN but timings still valid
    because kernel does identical MFMA ops on whatever bytes it reads.)
    """
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
        grouped_gemm_compute_offs,
    )
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    a_col, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN, axis=-2)
    g_col, g_s = quantize_fp8(grad, _FP8_DTYPE, _GRAN, axis=-2)

    hk = hipkit_module.load_fp8()
    out = torch.empty((B, N, K), dtype=torch.bfloat16, device="cuda")

    def _call():
        hk.grouped_variable_k_crr_dscale(
            g_col, a_col, out, g_s, a_s, g_offs,
            group_m=gm, num_xcds=xcds, num_slots=slots,
            chunk_size=chunk_size,
        )
    return _call, B, M, N, K, out


def verify_bit_eq(B, M, N, K, slots, gm, xcds, chunk_sizes):
    """Verify bit-equivalence across chunk_size values for the same input."""
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
        grouped_gemm_compute_offs,
    )
    g_offs = grouped_gemm_compute_offs(g_lens)
    torch.manual_seed(7)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    a_col, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN, axis=-2)
    g_col, g_s = quantize_fp8(grad, _FP8_DTYPE, _GRAN, axis=-2)
    hk = hipkit_module.load_fp8()

    outs = {}
    for cs in chunk_sizes:
        out = torch.empty((B, N, K), dtype=torch.bfloat16, device="cuda")
        hk.grouped_variable_k_crr_dscale(
            g_col, a_col, out, g_s, a_s, g_offs,
            group_m=gm, num_xcds=xcds, num_slots=slots,
            chunk_size=cs,
        )
        outs[cs] = out

    base = outs[chunk_sizes[0]]
    print(f"  bit-eq vs chunk_size={chunk_sizes[0]} (base):")
    all_eq = True
    for cs, out in outs.items():
        eq = _bit_eq(base, out)
        max_abs = (base - out).abs().max().item()
        print(f"    chunk_size={cs:>3}: bit_eq={eq}, max_abs={max_abs:.6f}")
        if not eq:
            all_eq = False
    return all_eq


def run_shape(B, M, N, K, gm, xcds, slots, chunk_sizes,
              warmup=20, iters=1500, repeats=7, seeds=(42, 137, 2024, 100, 99)):
    SHAPE = f"B{B}_M{M}_N{N}_K{K}"
    flops = 2.0 * (B * M) * N * K  # kernel FLOPs

    print(f"\n=== Down-{SHAPE} wgrad var-K @ (gm={gm}, xcds={xcds}, slots={slots}) ===")
    print(f"  iters={iters} × repeats={repeats} × seeds={len(seeds)} (p20)")

    print("  Bit-equivalence probe (chunk_size only changes pid mapping, NOT math):")
    bit_ok = verify_bit_eq(B, M, N, K, slots, gm, xcds, chunk_sizes)
    print(f"  bit-eq across all chunk_sizes: {bit_ok}")
    assert bit_ok, "chunk_size MUST be bit-equivalent — output mismatch is a kernel bug"

    print(f"\n  {'cs':>4}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    results = {}
    for cs in chunk_sizes:
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            call, _, _, _, _, _ = make_wgrad_call(B, M, N, K, slots, gm, xcds, cs)
            ms_per_iter = []
            for _ in range(repeats):
                ms_per_iter.append(_bench(call, warmup=warmup, iters=iters))
            seed_meds.append(statistics.median(ms_per_iter))
        med = statistics.median(seed_meds)
        results[cs] = (med, min(seed_meds), max(seed_meds))

    # Print sorted by chunk_size
    base_ms = results[64][0]  # chunk_size=64 is the existing baseline
    base_tflops = flops / (base_ms * 1e9)
    for cs in chunk_sizes:
        med, lo, hi = results[cs]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100  # +% means faster
        marker = " *base" if cs == 64 else ""
        print(f"  {cs:>4}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")

    # Identify best
    best_cs = min(chunk_sizes, key=lambda cs: results[cs][0])
    best_med, _, _ = results[best_cs]
    best_lift_pp = (base_ms - best_med) / base_ms * 100
    print(f"\n  BEST: chunk_size={best_cs}  ({best_lift_pp:+.2f}% over chunk_size=64 baseline)")
    return results, best_cs, best_lift_pp


if __name__ == "__main__":
    print(f"[probe] Round 13 chunk_size sweep on Down-B4 var-K wgrad")
    print(f"[probe] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    # Anchor: worst gpt_oss wgrad shape per metric (Down-B4-M2048)
    # Current rule cell (R3, R11): gm=1, xcds=2, slots=192
    chunk_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
    t0 = time.monotonic()
    res1, best1, lift1 = run_shape(
        B=4, M=2048, N=2880, K=2880,
        gm=1, xcds=2, slots=192,
        chunk_sizes=chunk_sizes,
    )
    print(f"[probe] anchor wall {time.monotonic()-t0:.1f}s")

    # Sibling: Down-B4-M4096 wgrad (m_total=16384) — same R3 cell
    t1 = time.monotonic()
    res2, best2, lift2 = run_shape(
        B=4, M=4096, N=2880, K=2880,
        gm=1, xcds=2, slots=192,
        chunk_sizes=chunk_sizes,
    )
    print(f"[probe] sibling wall {time.monotonic()-t1:.1f}s")

    print(f"\n[probe] SUMMARY:")
    print(f"  Down-B4-M2048 wgrad: best chunk_size={best1}, lift={lift1:+.2f}%")
    print(f"  Down-B4-M4096 wgrad: best chunk_size={best2}, lift={lift2:+.2f}%")
    print(f"[probe] total wall {time.monotonic()-t0:.1f}s")

    # Decision rule mirror of R15: lift >= 0.5pp AND best != 64 → ship.
    if best1 != 64 and lift1 >= 0.5 and best2 != 64 and lift2 >= 0.3:
        print(f"  → SHIP per-call chunk_size lever (both shapes win)")
    elif best1 != 64 and lift1 >= 0.5:
        print(f"  → SHIP for Down-B4-M2048 only (sibling did not transfer)")
    else:
        print(f"  → FALSIFIED: no chunk_size beats default 64 by margin")
