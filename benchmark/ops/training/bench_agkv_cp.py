###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""All-gather-KV context-parallel attention: the modes of ``flash_attn_agkv_func`` on one shape.

Modes (see ``usp/attention_agkv.py``): ``naive`` (RCCL gather, reorder copy, scatter copy, RCCL
reduce-scatter, all serial), ``serial`` (RCCL, copy-free layout, no overlap), ``overlap`` (copy-free
layout, exchange on the SDMA engines under the attention kernels), ``mega`` (compute-unit push / reduce
kernels, local keys attended while the remote ones arrive; run with ``FLUX_ATTN_FLYDSL=1``). All timed the same way as
``bench_ulysses_overlap.py``: ranks aligned by a barrier, the launch queued behind a GPU sleep, median
of iterations, max over ranks.

Parts, timed alone:

    ag        K and V all-gather of the local tokens (RCCL)
    rs        dK and dV reduce-scatter (RCCL)
    attn      the two causal FlyDSL calls of this rank (forward / backward), no exchange
    floor     attn fwd + bwd: what the op would cost if the exchange were free and fully hidden

Per mode: op forward, op backward, op forward + backward, and the attention block (fused QKV GEMM,
the op, o_proj; forward + backward with weight gradients). ``saved`` is against ``naive``;
``hidden`` is (naive - mode) / (naive - floor) of op forward + backward.

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_agkv_cp.py --model qwen3-30b-a3b --seq 16384 32768
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

from primus_turbo.common.flux import enabled as flux_enabled
from primus_turbo.pytorch.ops import flash_attn_agkv_func
from primus_turbo.pytorch.ops.attention.usp.attention_agkv import (
    AGKV_MODES,
    _bwd,
    _fwd,
    _Layout,
)

MODELS = {
    "llama3.1-8b": dict(hidden=4096, hq=32, hkv=8, d=128),
    "llama3.1-70b": dict(hidden=8192, hq=64, hkv=8, d=128),
    "muse-glimmer-30b": dict(hidden=6656, hq=32, hkv=2, d=128),
    "muse-spark-open": dict(hidden=8192, hq=128, hkv=16, d=64),
    "qwen3-4b": dict(hidden=2560, hq=32, hkv=8, d=128),
    "qwen3-8b": dict(hidden=4096, hq=32, hkv=8, d=128),
    "qwen3-30b-a3b": dict(hidden=2048, hq=32, hkv=4, d=128),
    "qwen3-235b-a22b": dict(hidden=4096, hq=64, hkv=4, d=128),
}


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


def bench_shape(cfg, seq, b, group, warmup, iters, modes):
    n = group.size()
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    s_local = seq // n
    lay = _Layout(s_local, group)
    c = lay.c
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
        kb = rnd((2 * n + 1) * c, b, hkv, d)
        vb = rnd((2 * n + 1) * c, b, hkv, d)
        calls = [(slice(0, c), lay.keys0), (slice(c, 2 * c), lay.keys1)]
        fwd_out = [_fwd(q[qs].contiguous(), kb[ks], vb[ks], None) for qs, ks in calls]
        t["attn_fwd"] = T(lambda: [_fwd(q[qs], kb[ks], vb[ks], None) for qs, ks in calls])
        t["attn_bwd"] = T(
            lambda: [
                _bwd(dout[qs], q[qs], kb[ks], vb[ks], o, lse, None, False)
                for (qs, ks), (o, lse) in zip(calls, fwd_out)
            ]
        )
        t["floor"] = t["attn_fwd"] + t["attn_bwd"]

    bench_collectives()
    bench_attention()

    x = rnd(s_local, b, hidden).requires_grad_()
    w_qkv = (rnd((hq + 2 * hkv) * d, hidden) * hidden**-0.5).requires_grad_()
    w_o = (rnd(hidden, hq * d) * (hq * d) ** -0.5).requires_grad_()
    dy = rnd(s_local, b, hidden)
    ql, kl, vl = (z.clone().requires_grad_() for z in (q, k, v))

    def block(mode):
        qkv = F.linear(x, w_qkv)
        bq, bk, bv = torch.split(qkv, [hq * d, hkv * d, hkv * d], dim=-1)
        o = flash_attn_agkv_func(
            bq.view(s_local, b, hq, d),
            bk.view(s_local, b, hkv, d),
            bv.view(s_local, b, hkv, d),
            group,
            mode=mode,
        )
        y = F.linear(o.view(s_local, b, hq * d), w_o)
        torch.autograd.grad(y, (x, w_qkv, w_o), dy)

    def bench_mode(mode):
        op = lambda: flash_attn_agkv_func(ql, kl, vl, group, mode=mode)
        out = op()
        t[f"{mode}_fwd"] = T(op)
        t[f"{mode}_bwd"] = T(lambda: torch.autograd.grad(out, (ql, kl, vl), dout, retain_graph=True))
        t[f"{mode}_fb"] = T(lambda: torch.autograd.grad(op(), (ql, kl, vl), dout))
        t[f"{mode}_block"] = T(lambda: block(mode))

    for mode in modes:
        bench_mode(mode)
    return t


def report(name, seq, b, n, t, modes):
    print(f"\n== {name} seq={seq} cp={n} mbs={b}  (us, max over ranks)")
    print(
        f"  parts: ag {t['ag']:.0f}  rs {t['rs']:.0f}  attn fwd {t['attn_fwd']:.0f}  attn bwd {t['attn_bwd']:.0f}"
        f"  floor {t['floor']:.0f}"
    )
    cols = ("fwd", "bwd", "fb", "block")
    print(
        f"  {'mode':<9}"
        + "".join(f"{c:>9}" for c in cols)
        + f"{'saved fb':>10}{'saved blk':>11}{'hidden':>8}"
    )
    base_fb, base_blk = t["naive_fb"], t["naive_block"]
    for mode in modes:
        fb, blk = t[f"{mode}_fb"], t[f"{mode}_block"]
        hidden = (base_fb - fb) / max(base_fb - t["floor"], 1e-9)
        print(
            f"  {mode:<9}"
            + "".join(f"{t[f'{mode}_{c}']:>9.0f}" for c in cols)
            + f"{base_fb - fb:>10.0f}{base_blk - blk:>11.0f}{hidden:>7.0%}"
        )
    rccl, serial = t["ag"] + t["rs"], t["serial_fb"]
    print(f"  fb: serial {serial:.0f} / attn {t['floor']:.0f} / rccl {rccl:.0f}")
    for mode in modes[2:]:
        fb = t[f"{mode}_fb"]
        print(
            f"  fb: {mode:<8} {fb:6.0f}   serial/{mode} {serial / fb:.3f}x   "
            f"{mode} - max(attn, rccl) {fb - max(t['floor'], rccl):.0f}"
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
    parser.add_argument("--modes", nargs="+", default=list(AGKV_MODES[2:]), choices=AGKV_MODES[2:])
    args = parser.parse_args()
    if "mega" in args.modes and not flux_enabled("FLUX_ATTN_FLYDSL"):
        parser.error("mode mega needs FLUX_ATTN_FLYDSL=1 (run every mode under it to compare like with like)")
    modes = list(AGKV_MODES[:2]) + args.modes
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    for name in args.model:
        for seq in args.seq:
            t = bench_shape(MODELS[name], seq, args.mbs, group, args.warmup, args.iters, modes)
            if rank == 0:
                report(name, seq, args.mbs, n, t, modes)
            torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
