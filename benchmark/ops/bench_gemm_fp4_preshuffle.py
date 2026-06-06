###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the AITER MXFP4 GEMM preshuffle fast path.

Reports per-shape ms/iter for two paired measurements that together
approximate the end-to-end Flux 12B MXFP4 step-time regression introduced
by Primus-Turbo PR #335 (removal of the conditional preshuffle path in
``GEMMFP4AITERBackend.execute``):

  1. GEMM:     ``gemm_fp4_impl(preshuffled=False)``  vs
               ``gemm_fp4_impl(preshuffled=True)``
               Pre-shuffles inputs once outside the timed loop in the
               True case. Reports the per-call shuffle overhead.

  2. Quantize: ``quantize_mxfp4_impl(shuffle_scale=False, shuffle_out=False)`` vs
               ``quantize_mxfp4_impl(shuffle_scale=True,  shuffle_out=True)``
               Reports the extra cost the fast path moves into the
               fused quantize kernel.

  Net = quantize delta + GEMM delta = realistic per-linear regression.

Requires MI355X / GFX950 (MXFP4 is GFX950-only) and an installed AITER
runtime.

Usage:
    cd benchmark/ops
    python bench_gemm_fp4_preshuffle.py [--warmup 20] [--repeat 200] [--profile]
"""

import argparse

import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
    enable_preshuffle,
    gemm_fp4_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quantize_mxfp4_impl,
)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
FP4_DTYPE = torch.float4_e2m1fn_x2
GRANULARITY = ScalingGranularity.MX_BLOCKWISE
BLOCK_SIZE = 32

PT_OPS = torch.ops.primus_turbo_cpp_extension

# Flux 12B MXFP4 GEMM shapes. Mirrors what the AITER FP4 path runs in
# Flux 12B linears (M=batch*sequence collapsed to 16384 typical;
# K/N pairs span fwd activation x weight and bwd activation x weight).
FLUX_12B_GEMM_SHAPES = [
    (16384, 12288, 3072),
    (16384, 3072, 12288),
    (16384, 3072, 9216),
    (16384, 9216, 3072),
    (16384, 3072, 3072),
    (4096, 3072, 3072),
]


def _time_ms(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def _quantize(x_hp, axis, use_2d_block, shuffle_scale, shuffle_out):
    return quantize_mxfp4_impl(
        x_hp,
        FP4_DTYPE,
        axis=axis,
        block_size=BLOCK_SIZE,
        scaling_recipe=ScalingRecipe(
            use_2d_block=use_2d_block,
            use_sr=False,
            use_rht=False,
            shuffle_scale=shuffle_scale,
            shuffle_out=shuffle_out,
        ),
    )


def bench_gemm(m, n, k, warmup, repeat):
    """Pair: AITER FP4 GEMM with preshuffled False vs True."""
    a_hp = torch.randn((m, k), dtype=DTYPE, device=DEVICE)
    b_hp = torch.randn((n, k), dtype=DTYPE, device=DEVICE)

    # Use the "False" recipes (canonical MX layout) so the preshuffled=False
    # GEMM path can shuffle internally; pre-shuffle outside for the True
    # path.
    qa_data, qa_scale = _quantize(a_hp, axis=1, use_2d_block=False, shuffle_scale=False, shuffle_out=False)
    qb_data, qb_scale = _quantize(b_hp, axis=1, use_2d_block=True, shuffle_scale=False, shuffle_out=False)
    qa_scale_pre = PT_OPS.shuffle_scale(qa_scale, [16, 16])
    qb_scale_pre = PT_OPS.shuffle_scale(qb_scale, [16, 16])
    qb_data_pre = PT_OPS.shuffle_weight(qb_data, [16, 16])

    def call_no_pre():
        return gemm_fp4_impl(
            qa_data, qa_scale, False,
            qb_data, qb_scale, True,
            DTYPE, False,
            granularity=GRANULARITY.value,
            default_backend=BackendType.AITER.value,
            preshuffled=False,
        )

    def call_pre():
        return gemm_fp4_impl(
            qa_data, qa_scale_pre, False,
            qb_data_pre, qb_scale_pre, True,
            DTYPE, False,
            granularity=GRANULARITY.value,
            default_backend=BackendType.AITER.value,
            preshuffled=True,
        )

    t_no_pre = _time_ms(call_no_pre, warmup, repeat)
    t_pre = _time_ms(call_pre, warmup, repeat)
    return t_no_pre, t_pre


def bench_quantize(m, n, k, warmup, repeat):
    """Pair: quantize_mxfp4_impl with shuffle off vs on, for both a (LHS,
    axis=1) and b (RHS, axis=1, use_2d_block=True)."""
    a_hp = torch.randn((m, k), dtype=DTYPE, device=DEVICE)
    b_hp = torch.randn((n, k), dtype=DTYPE, device=DEVICE)

    def quant_a_off():
        return _quantize(a_hp, axis=1, use_2d_block=False, shuffle_scale=False, shuffle_out=False)

    def quant_a_on():
        # A operand: shuffle_scale only (A-data never shuffled in AITER).
        return _quantize(a_hp, axis=1, use_2d_block=False, shuffle_scale=True, shuffle_out=False)

    def quant_b_off():
        return _quantize(b_hp, axis=1, use_2d_block=True, shuffle_scale=False, shuffle_out=False)

    def quant_b_on():
        # B operand: both scale and data shuffled.
        return _quantize(b_hp, axis=1, use_2d_block=True, shuffle_scale=True, shuffle_out=True)

    t_a_off = _time_ms(quant_a_off, warmup, repeat)
    t_a_on = _time_ms(quant_a_on, warmup, repeat)
    t_b_off = _time_ms(quant_b_off, warmup, repeat)
    t_b_on = _time_ms(quant_b_on, warmup, repeat)
    return t_a_off + t_b_off, t_a_on + t_b_on


def profile_one_iter(m, n, k):
    """Capture a torch.profiler trace of a single preshuffled=True call
    and print kernel launches; the 3 ``shuffle_*`` kernels must disappear.
    """
    a_hp = torch.randn((m, k), dtype=DTYPE, device=DEVICE)
    b_hp = torch.randn((n, k), dtype=DTYPE, device=DEVICE)

    qa_data, qa_scale = _quantize(a_hp, axis=1, use_2d_block=False, shuffle_scale=True, shuffle_out=False)
    qb_data, qb_scale = _quantize(b_hp, axis=1, use_2d_block=True, shuffle_scale=True, shuffle_out=True)

    # Warm up
    for _ in range(5):
        gemm_fp4_impl(
            qa_data, qa_scale, False,
            qb_data, qb_scale, True,
            DTYPE, False,
            granularity=GRANULARITY.value,
            default_backend=BackendType.AITER.value,
            preshuffled=True,
        )
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        for _ in range(5):
            gemm_fp4_impl(
                qa_data, qa_scale, False,
                qb_data, qb_scale, True,
                DTYPE, False,
                granularity=GRANULARITY.value,
                default_backend=BackendType.AITER.value,
                preshuffled=True,
            )
        torch.cuda.synchronize()

    print()
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

    shuffle_kernels = [
        e for e in prof.key_averages()
        if "shuffle" in e.key.lower()
    ]
    if shuffle_kernels:
        print(
            f"WARNING: preshuffled=True path saw {len(shuffle_kernels)} shuffle "
            f"kernel(s) — fast path is broken:"
        )
        for e in shuffle_kernels:
            print(f"  {e.key}")
    else:
        print("OK: no shuffle kernels in preshuffled=True path (fast path active)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture a torch.profiler trace of one preshuffled=True GEMM "
        "and verify the 3 shuffle kernels are absent.",
    )
    args = parser.parse_args()

    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        raise SystemExit(f"MXFP4 not supported: {reason}")

    GlobalBackendManager.set_gemm_backend(BackendType.AITER)
    GlobalBackendManager.set_auto_tune(False)
    assert enable_preshuffle() is True, "Test setup: AITER pin + autotune-off should enable preshuffle"

    try:
        header = (
            f"{'M':>6} {'N':>6} {'K':>6} | "
            f"{'GEMM no-pre':>11} {'GEMM pre':>10} {'dGEMM':>9} | "
            f"{'Qz off':>9} {'Qz on':>9} {'dQz':>9} | {'Net dms':>8}"
        )
        print(header)
        print("-" * len(header))

        totals = {"gemm_delta": 0.0, "qz_delta": 0.0}
        for m, n, k in FLUX_12B_GEMM_SHAPES:
            try:
                import aiter

                if aiter.get_GEMM_config(m, n, k) is None:
                    print(f"{m:>6} {n:>6} {k:>6} | (no AITER GEMM config; skipped)")
                    continue
                t_gemm_off, t_gemm_on = bench_gemm(m, n, k, args.warmup, args.repeat)
                t_qz_off, t_qz_on = bench_quantize(m, n, k, args.warmup, args.repeat)
                d_gemm = t_gemm_off - t_gemm_on
                d_qz = t_qz_on - t_qz_off
                net = d_gemm - d_qz
                totals["gemm_delta"] += d_gemm
                totals["qz_delta"] += d_qz
                print(
                    f"{m:>6} {n:>6} {k:>6} | "
                    f"{t_gemm_off:>11.3f} {t_gemm_on:>10.3f} {d_gemm:>+9.3f} | "
                    f"{t_qz_off:>9.3f} {t_qz_on:>9.3f} {d_qz:>+9.3f} | {net:>+8.3f}"
                )
            except Exception as e:
                print(f"{m:>6} {n:>6} {k:>6} | ERROR: {e}")

        print("-" * len(header))
        print(
            f"Totals: GEMM saved = {totals['gemm_delta']:+.3f} ms, "
            f"Quantize moved = {totals['qz_delta']:+.3f} ms, "
            f"Net saved = {totals['gemm_delta'] - totals['qz_delta']:+.3f} ms"
        )

        if args.profile:
            profile_one_iter(*FLUX_12B_GEMM_SHAPES[0])
    finally:
        GlobalBackendManager.reset()


if __name__ == "__main__":
    main()
