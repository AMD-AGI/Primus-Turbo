###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark for the MXFP6 packer's fused epilogue.

Two questions, both of which the fusion has to answer before it is worth shipping:

1. Did adding the prologue template parameter cost the plain (Identity) pack anything?
   The prologue is compiled away for Identity, but the kernel gained three pointer
   arguments and an LDS pass, so occupancy is worth checking rather than asserting.

2. What does the fusion actually save at the shapes Flux 12B uses? The comparison is
   against the path it replaces -- an eager epilogue writing bf16 to HBM followed by a
   pack that reads it back -- not against the pack alone.

Run with: python benchmarks/bench_mxfp6_fused_pack.py
"""

import argparse

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP6_PROLOGUE_BIAS_GELU,
    MXFP6_PROLOGUE_BIAS_GELU_BACKWARD,
)
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
    check_mxfp6_support,
    mxfp6_apply_prologue,
    quantize_mxfp6_dual,
    quantize_mxfp6_fused_dual,
)

# The MLP shapes of a Flux 12B step at seq_length 512 / micro_batch_size 64: 32768 tokens
# through the double blocks and 16384 through the single blocks, against ffn_hidden_size
# 12288. These are the tensors the epilogue round-trip is paid on.
FLUX_SHAPES = [(32768, 12288), (16384, 12288)]


def _bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    supported, reason = check_mxfp6_support()
    if not supported:
        raise SystemExit(reason)

    dev = "cuda:0"
    dtype = torch.bfloat16

    print(f"{'shape':>16} {'mode':>10} {'unfused ms':>11} {'fused ms':>10} {'speedup':>8}")
    for rows, cols in FLUX_SHAPES:
        x = torch.randn((rows, cols), dtype=dtype, device=dev)
        aux = torch.randn((rows, cols), dtype=dtype, device=dev)
        bias = torch.randn((cols,), dtype=dtype, device=dev)

        plain = _bench(lambda x=x: quantize_mxfp6_dual(x), args.iters)
        print(f"{f'{rows}x{cols}':>16} {'identity':>10} {'-':>11} {plain:>10.3f} {'-':>8}")

        for label, mode, a in (
            ("fwd", MXFP6_PROLOGUE_BIAS_GELU, None),
            ("bwd", MXFP6_PROLOGUE_BIAS_GELU_BACKWARD, aux),
        ):
            want_sum = mode == MXFP6_PROLOGUE_BIAS_GELU_BACKWARD

            def unfused(mode=mode, a=a, want_sum=want_sum, x=x, bias=bias):
                staged = mxfp6_apply_prologue(x, a, bias, mode)
                quantize_mxfp6_dual(staged)
                if want_sum:
                    staged.float().sum(0)

            def fused(mode=mode, a=a, want_sum=want_sum, x=x, bias=bias):
                *_, partial = quantize_mxfp6_fused_dual(x, a, bias, mode, want_sum)
                if want_sum:
                    partial.sum(0)

            slow = _bench(unfused, args.iters)
            fast = _bench(fused, args.iters)
            print(f"{f'{rows}x{cols}':>16} {label:>10} {slow:>11.3f} {fast:>10.3f} {slow / fast:>7.2f}x")


if __name__ == "__main__":
    main()
