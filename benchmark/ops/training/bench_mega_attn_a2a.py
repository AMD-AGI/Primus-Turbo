###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses attention core (a2a + causal attention + a2a), batch 1: the unfused path against the fused
kernel (``primus_turbo.flydsl.mega_attn.a2a_attn_kernel``).

* ``unfused``: ``flash_attn_usp_func``'s sequence -- pack, RCCL all_to_all, unpack, FlyDSL attention,
  pack, all_to_all, unpack (contiguous sequence split);
* ``fused cs=X``: the fused kernel with sequence chunk X (0 = contiguous split);
* ``attn``: the attention alone on this rank's heads over the full sequence, and ``a2a``: the two
  all-to-alls alone (q/k/v and O bytes), ranks aligned first -- the lower bound is max(attn, a2a).

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_mega_attn_a2a.py --model qwen3-30b-a3b --seq 32768
"""

import argparse
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist

from primus_turbo.flydsl.mega_attn.a2a_attn_kernel import (
    a2a_attn_fwd_bank_bytes,
    trace_layout,
    ulysses_a2a_attn_fwd,
)
from primus_turbo.flydsl.ulysses.workspace import get_ulysses_workspace
from primus_turbo.pytorch.ops import flash_attn_func
from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
    get_attention_cp_a2a_helper,
)

MODELS = {
    "qwen3-30b-a3b": dict(hq=32, hkv=4, d=128),
    "qwen3-235b-a22b": dict(hq=64, hkv=4, d=128),
    "llama3.1-8b": dict(hq=32, hkv=8, d=128),
    "llama3.1-70b": dict(hq=64, hkv=8, d=128),
    "glm-4.7": dict(hq=96, hkv=8, d=128),
}


def _barrier():
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()


def time_us(fn, warmup, iters, reps):
    """Median over ``reps`` of the mean of ``iters`` back-to-back calls, ranks aligned before each rep;
    the max over ranks."""
    for _ in range(warmup):
        fn()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        _barrier()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    t = torch.tensor([statistics.median(times)], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def bench(model, seq, chunks, warmup, iters, reps, trace_dir):
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    cfg = MODELS[model]
    hq, hkv, d = cfg["hq"], cfg["hkv"], cfg["d"]
    M, hq_r, hkv_r = seq // n, hq // n, hkv // n
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(rank)
    q_in = torch.randn(M, hq, d, device=dev, dtype=torch.bfloat16, generator=gen)
    k_in = torch.randn(M, hkv, d, device=dev, dtype=torch.bfloat16, generator=gen)
    v_in = torch.randn(M, hkv, d, device=dev, dtype=torch.bfloat16, generator=gen)
    results = {}

    helper = get_attention_cp_a2a_helper(1, seq, hq, hkv, d, d, 1, n)

    def unfused():
        qkv = helper.combine_qkv_before_a2a(q_in[None], k_in[None], v_in[None])
        recv = torch.empty_like(qkv)
        dist.all_to_all_single(recv, qkv, group=group)
        q, k, v = helper.splits_qkv_after_a2a(recv)
        o = flash_attn_func(q, k, v, causal=True)
        o = helper.reshape_o_before_a2a(o)
        recv_o = torch.empty_like(o)
        dist.all_to_all_single(recv_o, o, group=group)
        return helper.reshape_o_after_a2a(recv_o)

    results["unfused"] = time_us(unfused, warmup, iters, reps)

    qg = torch.randn(seq, 1, hq_r, d, device=dev, dtype=torch.bfloat16, generator=gen)
    kg = torch.randn(seq, 1, hkv_r, d, device=dev, dtype=torch.bfloat16, generator=gen)
    vg = torch.randn(seq, 1, hkv_r, d, device=dev, dtype=torch.bfloat16, generator=gen)
    results["attn"] = time_us(
        lambda: flash_attn_func(qg.transpose(0, 1), kg.transpose(0, 1), vg.transpose(0, 1), causal=True),
        warmup,
        iters,
        reps,
    )

    qkv_bytes = torch.empty(M * (hq + 2 * hkv) * d, device=dev, dtype=torch.bfloat16)
    o_bytes = torch.empty(M * hq * d, device=dev, dtype=torch.bfloat16)
    qkv_recv, o_recv = torch.empty_like(qkv_bytes), torch.empty_like(o_bytes)

    def a2a_only():
        dist.all_to_all_single(qkv_recv, qkv_bytes, group=group)
        dist.all_to_all_single(o_recv, o_bytes, group=group)

    results["a2a"] = time_us(a2a_only, warmup, iters, reps)

    ws = get_ulysses_workspace(group, a2a_attn_fwd_bank_bytes(M, n, hq_r, hkv_r, d), name="mega_attn")
    q = torch.empty(seq, hq_r, d, device=dev, dtype=torch.bfloat16)
    k = torch.empty(seq, hkv_r, d, device=dev, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    lse = torch.empty(seq, hq_r, device=dev, dtype=torch.float32)
    out = torch.empty_like(q_in)
    for cs in chunks:
        if cs and (M % cs or cs % 128):
            continue
        results[f"fused cs={cs}"] = time_us(
            lambda cs=cs: ulysses_a2a_attn_fwd(q_in, k_in, v_in, q, k, v, o, lse, out, ws, seq_chunk=cs),
            warmup,
            iters,
            reps,
        )
        if trace_dir:
            total = trace_layout(M, n, hq_r, hkv_r, d)[2]
            trace = torch.zeros(total, device=dev, dtype=torch.int64)
            _barrier()
            ulysses_a2a_attn_fwd(q_in, k_in, v_in, q, k, v, o, lse, out, ws, seq_chunk=cs, trace=trace)
            torch.cuda.synchronize()
            os.makedirs(trace_dir, exist_ok=True)
            torch.save(
                dict(trace=trace.cpu(), layout=trace_layout(M, n, hq_r, hkv_r, d), M=M, n=n, cs=cs),
                os.path.join(trace_dir, f"{model}_s{seq}_n{n}_cs{cs}_r{rank}.pt"),
            )

    if rank == 0:
        lower = max(results["attn"], results["a2a"])
        print(f"\n== {model} seq={seq} n={n} (hq/n={hq_r}, hkv/n={hkv_r}, d={d})", flush=True)
        print(
            f"  attn alone {results['attn']:9.1f} us   a2a alone {results['a2a']:9.1f} us   "
            f"attn+a2a {results['attn'] + results['a2a']:9.1f} us   lower bound {lower:9.1f} us"
        )
        for name, t in results.items():
            if name in ("attn", "a2a"):
                continue
            print(
                f"  {name:16s} {t:9.1f} us   vs unfused {results['unfused'] / t:5.3f}x   "
                f"over attn {t - results['attn']:8.1f} us",
                flush=True,
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen3-30b-a3b"], choices=sorted(MODELS))
    parser.add_argument("--seq", nargs="+", type=int, default=[32768])
    parser.add_argument("--chunks", nargs="+", type=int, default=[0, 256, 512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default=None)
    args = parser.parse_args()
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    for model in args.model:
        for seq in args.seq:
            bench(model, seq, args.chunks, args.warmup, args.iters, args.reps, args.trace_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
