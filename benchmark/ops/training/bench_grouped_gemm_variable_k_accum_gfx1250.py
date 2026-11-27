###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark beta = 1 on the gfx1250 FlyDSL variable-K grouped wgrad (MI455X): the expert dW written straight
into an fp32 main_grad, against today's bf16 dW plus ``main_grad.add_``.

Defaults are a Muse Spark expert layer as one GPU of the EP4 run sees it: 64 local experts, top-8 routing of a
4096-token micro-batch (softmax top-k of N(0, 1) logits), fc1 dW ``[64, 8192, 4096]`` and fc2 dW
``[64, 4096, 4096]``, called the way ``grouped_gemm``'s backward calls it (dy and x over the permuted rows,
``out[g] = dy[rows_g].T @ x[rows_g]``). Per weight it times:

  overwrite   the shipped wgrad, bf16 dW
  add         main_grad.add_(dW), fp32 += bf16 (Megatron's DDP hook)
  tdm / reg   accumulate=1, the old main_grad brought in through LDS (TDM) or registers

and the per-step total for ``--layers`` x ``--micro-batches`` calls of each weight.

Usage::

    python benchmark/ops/training/bench_grouped_gemm_variable_k_accum_gfx1250.py
    python benchmark/ops/training/bench_grouped_gemm_variable_k_accum_gfx1250.py --layers 4 --micro-batches 2 -o out.csv
"""

import argparse

import pandas as pd
import torch
from tabulate import tabulate

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    grouped_gemm_bf16_variable_k_accum_supported,
    grouped_gemm_bf16_variable_k_flydsl_kernel,
)


def routing(num_experts, tokens, topk, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, num_experts, device="cuda", generator=g)
    experts = logits.softmax(-1).topk(topk, dim=-1).indices
    lens = torch.bincount(experts.flatten(), minlength=num_experts).to(torch.int64)
    offs = torch.zeros(num_experts + 1, dtype=torch.int64, device="cuda")
    offs[1:] = lens.cumsum(0)
    return lens, offs


def median_ms(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def bench_weight(name, out_dim, in_dim, lens, offs, args):
    m = int(offs[-1])
    g = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    dy = torch.randn(m, out_dim, device="cuda", generator=g).to(torch.bfloat16)
    x = torch.randn(m, in_dim, device="cuda", generator=g).to(torch.bfloat16)
    main_grad = torch.zeros(lens.numel(), out_dim, in_dim, device="cuda", dtype=torch.float32)
    vk = grouped_gemm_bf16_variable_k_flydsl_kernel

    t_ow = median_ms(lambda: vk(dy, x, offs, masked_k=lens, out_dtype=torch.bfloat16), args.iters)
    dw = vk(dy, x, offs, masked_k=lens, out_dtype=torch.bfloat16)
    t_add = median_ms(lambda: main_grad.add_(dw), args.iters)
    row = {
        "weight": name,
        "dW": f"[{lens.numel()}, {out_dim}, {in_dim}]",
        "rows": m,
        "overwrite_ms": t_ow,
        "add_ms": t_add,
        "today_ms": t_ow + t_add,
        "overwrite_TFLOPS": 2.0 * m * out_dim * in_dim / t_ow / 1e9,
    }
    for impl in args.impl:
        row[f"{impl}_ms"] = median_ms(
            lambda impl=impl: vk(dy, x, offs, masked_k=lens, out=main_grad, accumulate=1, accum_impl=impl),
            args.iters,
        )
        row[f"{impl}_saves_ms"] = row["today_ms"] - row[f"{impl}_ms"]
    return row


def main(args):
    lens, offs = routing(args.num_experts, args.tokens, args.topk, args.seed)
    print(
        f"{torch.cuda.get_device_name(0)}: {args.num_experts} experts, top-{args.topk} of {args.tokens} tokens, "
        f"rows per expert min / mean / max = {int(lens.min())} / {lens.float().mean():.0f} / {int(lens.max())}"
    )
    avg_m = max(1, args.tokens * args.topk // args.num_experts)
    weights = [("fc1", 2 * args.ffn, args.latent), ("fc2", args.latent, args.ffn)]
    rows = []
    for name, out_dim, in_dim in weights:
        if not grouped_gemm_bf16_variable_k_accum_supported(out_dim, in_dim, avg_m):
            print(f"[skip] {name}: [{out_dim}, {in_dim}] is not whole output tiles")
            continue
        rows.append(bench_weight(name, out_dim, in_dim, lens, offs, args))
    df = pd.DataFrame(rows)
    calls = args.layers * args.micro_batches
    step = {"weight": f"per step ({calls} calls each)", "dW": "", "rows": ""}
    for col in [c for c in df.columns if c.endswith("_ms")]:
        step[col] = df[col].sum() * calls
    df = pd.concat([df, pd.DataFrame([step])], ignore_index=True)
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".3f"))
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the gfx1250 FlyDSL variable-K wgrad with accumulate=1"
    )
    parser.add_argument("--num-experts", type=int, default=64, help="local experts")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096, help="tokens per micro-batch")
    parser.add_argument("--latent", type=int, default=4096, help="expert input width")
    parser.add_argument(
        "--ffn", type=int, default=4096, help="expert FFN width (fc1 dW is [E, 2 * ffn, latent])"
    )
    parser.add_argument("--layers", type=int, default=4, help="MoE layers per step, for the per-step row")
    parser.add_argument("--micro-batches", type=int, default=2, help="micro-batches per step")
    parser.add_argument("--impl", nargs="+", default=["tdm", "reg"], choices=["tdm", "reg"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, default=None)
    main(parser.parse_args())
