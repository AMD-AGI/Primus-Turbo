###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""What the block-sparse forward gains from splitting full tiles from partial ones.

A block mask's plan lists, per query block, the key blocks to visit in two groups:
fully kept tiles, which the forward runs with ``mask_mod`` compiled out, and partially
kept ones, which evaluate ``mask_mod`` per element. This runs the same mask with the same
tile size twice:

  split        the plan from ``create_block_mask`` (full and partial lists)
  all-partial  ``plan.combined()``: every visited tile in the partial list, as the
               backward always runs it

and reports the forward time of each. Both use the same inputs, the same visited tiles
and the same kernel knobs (picked once per shape), so the difference is the cost of
evaluating ``mask_mod`` on the full tiles.

Usage: python bench_flex_attention_block_split.py [--cases doc] [--block-sizes 64,128]
"""

import argparse
import statistics

import torch
import triton
import triton.language as tl
from tabulate import tabulate

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.kernels.flex_attention.flex_attention_impl import (
    flex_attention_forward_impl,
)

DTYPE = torch.bfloat16
B, HQ, HKV, S, D = 2, 16, 2, 8192, 128
WARMUP, ROUNDS = 10, 50


@triton.jit
def band_mask(b, h, q_idx, kv_idx):
    d = q_idx - kv_idx
    return (d < 1024) & (d > -1024)


@triton.jit
def prefix_lm_mask(b, h, q_idx, kv_idx):
    return (kv_idx < 1024) | (q_idx >= kv_idx)


@triton.jit
def document_causal_mask(b, h, q_idx, kv_idx):
    # 1000-token documents: boundaries fall inside tiles, so some tiles are partial.
    return (q_idx // 1000 == kv_idx // 1000) & (q_idx >= kv_idx)


# (name, mask_mod). causal_mask normally takes the kernel's causal path; here it runs on
# the block-sparse path like any custom mask.
CASES = [
    ("causal", turbo.ops.causal_mask),
    ("band1k", band_mask),
    ("prefix_lm1k", prefix_lm_mask),
    ("doc1000_causal", document_causal_mask),
]


def median_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ROUNDS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def bench_case(name, mask_mod, block_size, q, k, v):
    split = turbo.ops.create_block_mask(mask_mod, None, None, S, S, BLOCK_SIZE=block_size).plan_at(block_size)
    all_partial = split.combined()
    full = 0 if split.full is None else int(split.full.count.sum())
    visited = full + int(split.masked.count.sum())

    out = torch.empty(B, S, HQ, D, dtype=DTYPE, device="cuda")
    lse = torch.empty(B, HQ, S, dtype=torch.float32, device="cuda")

    def forward(plan):
        flex_attention_forward_impl(
            q, k, v, out, lse,
            scale=D**-0.5, causal=False, window=(-1, -1), mask_mod=mask_mod, block_plan=plan,
        )
        return out

    ref = forward(split).clone()
    diff = (forward(all_partial).float() - ref.float()).abs().max().item()
    split_ms = median_ms(lambda: forward(split))
    partial_ms = median_ms(lambda: forward(all_partial))
    return [
        name,
        block_size,
        f"{100.0 * full / visited:.1f}",
        visited,
        f"{split_ms:.3f}",
        f"{partial_ms:.3f}",
        f"{partial_ms / split_ms:.2f}x",
        f"{diff:.3g}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="", help="substring filter on case names")
    parser.add_argument("--block-sizes", default="64,128", help="comma-separated tile sizes")
    args = parser.parse_args()
    block_sizes = [int(s) for s in args.block_sizes.split(",")]

    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, S, h, D, generator=gen).to("cuda", DTYPE) for h in (HQ, HKV, HKV))

    rows = []
    for name, mask_mod in CASES:
        if args.cases not in name:
            continue
        for block_size in block_sizes:
            rows.append(bench_case(name, mask_mod, block_size, q, k, v))
            print(rows[-1], flush=True)
    print(f"\n{torch.cuda.get_device_name(0)}, torch {torch.__version__}, B={B} H={HQ}/{HKV} S={S} D={D}")
    headers = [
        "mask",
        "block",
        "full tiles %",
        "visited tiles/head",
        "split ms",
        "all-partial ms",
        "split speedup",
        "max |diff|",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
