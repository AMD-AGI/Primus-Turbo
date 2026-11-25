###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""All-gather-KV context-parallel attention (``flash_attn_agkv_func``, fused) on one shape.

Timed the same way as ``bench_ulysses_overlap.py``: ranks aligned by a barrier, the launch queued behind
a GPU sleep, median of iterations, max over ranks.

Parts, timed alone:

    ag        K and V all-gather of the local tokens (RCCL)
    rs        dK and dV reduce-scatter (RCCL)
    attn      the two causal FlyDSL calls of this rank (forward / backward), no exchange
    floor     attn fwd + bwd: what the op would cost if the exchange were free and fully hidden
    exposed   floor + ag + rs: the exchange not overlapped at all

The op: forward, backward, forward + backward, and the attention block (fused QKV GEMM, the op, o_proj;
forward + backward with weight gradients). ``hidden`` is (exposed - op) / (exposed - floor) of forward +
backward, over 100 % where the op beats the floor.

    torchrun --nproc_per_node 8 benchmark/ops/training/bench_agkv_cp.py --model qwen3-30b-a3b --seq 16384 32768
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)
from primus_turbo.pytorch.ops import flash_attn_agkv_func

MODELS = {
    "llama3.1-8b": dict(hidden=4096, hq=32, hkv=8, d=128),
    "llama3.1-70b": dict(hidden=8192, hq=64, hkv=8, d=128),
    "gpt-oss-120b": dict(hidden=2880, hq=64, hkv=8, d=64),
    "muse-glimmer-30b": dict(hidden=6656, hq=32, hkv=2, d=128),
    "muse-spark-open": dict(hidden=8192, hq=128, hkv=16, d=64),
    "qwen3-4b": dict(hidden=2560, hq=32, hkv=8, d=128),
    "qwen3-8b": dict(hidden=4096, hq=32, hkv=8, d=128),
    "qwen3-30b-a3b": dict(hidden=2048, hq=32, hkv=4, d=128),
    "qwen3-235b-a22b": dict(hidden=4096, hq=64, hkv=4, d=128),
}


def _fwd(q, k, v):
    return flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, return_lse=True)


def _bwd(dout, q, k, v, out, lse):
    Sq, B, Hq = q.shape[:3]
    return flash_attn_sbhd_flydsl_backward_impl(
        dout, q, k, v, out, lse.view(B, Sq, Hq).permute(0, 2, 1), causal=True
    )


def timeit(fn, group, warmup=5, iters=30):
    times = []
    for it in range(warmup + iters):
        torch.cuda.synchronize()
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(2_000_000)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if it >= warmup:
            times.append(start.elapsed_time(end) * 1e3)
    t = torch.tensor([statistics.median(times)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.item()


def bench_shape(cfg, seq, b, group, warmup, iters):
    n, r = group.size(), group.rank()
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    s_local = seq // n
    c = s_local // 2
    rnd = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    q, k, v, dout = (
        rnd(s_local, b, hq, d),
        rnd(s_local, b, hkv, d),
        rnd(s_local, b, hkv, d),
        rnd(s_local, b, hq, d),
    )
    T = lambda fn: timeit(fn, group, warmup, iters)
    t = {}

    def bench_collectives():
        kg = torch.empty((n * s_local,) + k.shape[1:], device="cuda", dtype=k.dtype)
        vg = torch.empty_like(kg)
        dk, dv = torch.empty_like(k), torch.empty_like(v)

        def ag():
            dist.all_gather_into_tensor(kg, k, group=group)
            dist.all_gather_into_tensor(vg, v, group=group)

        def rs():
            dist.reduce_scatter_tensor(dk, kg, group=group)
            dist.reduce_scatter_tensor(dv, vg, group=group)

        t["ag"], t["rs"] = T(ag), T(rs)

    def bench_attention():
        # q0 (chunk r) sees (r + 1) c keys, q1 (chunk 2n-1-r) (2n - r) c
        kb, vb = rnd(2 * n * c, b, hkv, d), rnd(2 * n * c, b, hkv, d)
        calls = [(slice(0, c), (r + 1) * c), (slice(c, 2 * c), (2 * n - r) * c)]
        fwd_out = [_fwd(q[qs].contiguous(), kb[:nk], vb[:nk]) for qs, nk in calls]
        t["attn_fwd"] = T(lambda: [_fwd(q[qs], kb[:nk], vb[:nk]) for qs, nk in calls])
        t["attn_bwd"] = T(
            lambda: [
                _bwd(dout[qs], q[qs], kb[:nk], vb[:nk], o, lse) for (qs, nk), (o, lse) in zip(calls, fwd_out)
            ]
        )

    bench_collectives()
    bench_attention()
    t["floor"] = t["attn_fwd"] + t["attn_bwd"]
    t["exposed"] = t["floor"] + t["ag"] + t["rs"]

    ql, kl, vl = (z.clone().requires_grad_() for z in (q, k, v))
    op = lambda: flash_attn_agkv_func(ql, kl, vl, group)
    out = op()
    t["fwd"] = T(op)
    t["bwd"] = T(lambda: torch.autograd.grad(out, (ql, kl, vl), dout, retain_graph=True))
    t["fb"] = T(lambda: torch.autograd.grad(op(), (ql, kl, vl), dout))

    x = rnd(s_local, b, hidden).requires_grad_()
    w_qkv = (rnd((hq + 2 * hkv) * d, hidden) * hidden**-0.5).requires_grad_()
    w_o = (rnd(hidden, hq * d) * (hq * d) ** -0.5).requires_grad_()
    dy = rnd(s_local, b, hidden)

    def block():
        qkv = F.linear(x, w_qkv)
        bq, bk, bv = torch.split(qkv, [hq * d, hkv * d, hkv * d], dim=-1)
        o = flash_attn_agkv_func(
            bq.view(s_local, b, hq, d), bk.view(s_local, b, hkv, d), bv.view(s_local, b, hkv, d), group
        )
        y = F.linear(o.view(s_local, b, hq * d), w_o)
        torch.autograd.grad(y, (x, w_qkv, w_o), dy)

    t["block"] = T(block)
    return t


def report(name, seq, b, n, t):
    print(f"\n== {name} seq={seq} cp={n} mbs={b}  (us, max over ranks)")
    print(
        f"  parts: ag {t['ag']:.0f}  rs {t['rs']:.0f}  attn fwd {t['attn_fwd']:.0f}  attn bwd {t['attn_bwd']:.0f}"
        f"  floor {t['floor']:.0f}  exposed {t['exposed']:.0f}"
    )
    hidden = (t["exposed"] - t["fb"]) / max(t["exposed"] - t["floor"], 1e-9)
    print(
        f"  fused: fwd {t['fwd']:.0f}  bwd {t['bwd']:.0f}  fwd+bwd {t['fb']:.0f}  block {t['block']:.0f}"
        f"  fwd+bwd - floor {t['fb'] - t['floor']:.0f}  exposed/fused {t['exposed'] / t['fb']:.3f}x  hidden {hidden:.0%}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", nargs="+", default=["qwen3-30b-a3b"], choices=sorted(MODELS))
    parser.add_argument("--seq", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    for name in args.model:
        for seq in args.seq:
            t = bench_shape(MODELS[name], seq, args.mbs, group, args.warmup, args.iters)
            if rank == 0:
                report(name, seq, args.mbs, n, t)
            torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
