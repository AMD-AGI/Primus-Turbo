###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused GEMM + reduce-scatter (``fused_collective_gemm.gemm_rs``) against its parts, on the MLA
latent backward under sequence parallelism (Ulysses): every rank computes the latent gradients of the
whole sequence from its local heads' q / k / v gradients and reduce-scatters them, optionally with the
up-projection weight gradients in the same launch.

Ops:

    bwd dgrad          d_q_latent / d_kv_latent
    bwd dgrad + wgrad  also d_w_q and d_w_kv (the op's configuration)

Columns (us, ranks aligned by a barrier, the launch queued behind a GPU sleep so host time is hidden,
median of iterations, max over ranks):

    gemm      the op's GEMMs (hipBLASLt): the latent-gradient GEMMs, plus the weight-gradient GEMMs
    rs        RCCL ``reduce_scatter_tensor`` of the partial latent gradients alone
    floor     max(gemm, rs): perfect overlap of the parts
    unfused   ``[dk_nope | dv]`` packing, the GEMMs and the ``k_pe`` head sum, then RCCL reduce-scatter
    fused     ``gemm_rs`` with its default launch knobs

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_gemm_rs.py --seq 32768
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from primus_turbo.flydsl.fused_collective_gemm.gemm_rs_kernel import (
    gemm_rs,
    gemm_rs_bank_bytes,
)
from primus_turbo.flydsl.fused_collective_gemm.workspace import (
    get_collective_gemm_workspace,
)

DSV3 = dict(heads=128, cq=1536, ckv=512, dn=128, dr=64, dv=128)
MODELS = {"deepseek-v3": DSV3, "kimi-k2": dict(DSV3, heads=64)}
OPS = ("bwd dgrad", "bwd dgrad + wgrad")
COLS = ("gemm", "rs", "floor", "unfused", "fused")


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


def rnd(*shape):
    return (torch.randn(*shape, device="cuda") * 0.02).bfloat16()


def build(cfg, seq, mbs, group, wgrad):
    """(gemm, rs, unfused, fused) callables of one op."""
    n = group.size()
    hl, cq, ckv, dn, dr, dv = cfg["heads"] // n, cfg["cq"], cfg["ckv"], cfg["dn"], cfg["dr"], cfg["dv"]
    T = seq // n * mbs
    R, nk = n * T, hl * (dn + dr)
    gq, gk, gv = rnd(R, nk), rnd(R, hl, dn + dr), rnd(R, hl, dv)
    lq, lc = rnd(R, cq), rnd(R, ckv)
    # this rank's weight rows (head-major, [k_nope | v] per head for kv) and the GemmRS operands
    wq, wkv = rnd(nk, cq), rnd(hl * (dn + dv), ckv)
    wq_t, wk_t, wv_t = rnd(cq, nk), rnd(ckv + dr, nk), rnd(ckv, hl * dv)
    d_q_latent = torch.empty(T, cq, dtype=torch.bfloat16, device="cuda")
    d_kv_latent = torch.empty(T, ckv + dr, dtype=torch.bfloat16, device="cuda")
    d_w_q, d_w_kv = torch.empty_like(wq), torch.empty_like(wkv)
    gkv = torch.cat((gk[..., :dn], gv), dim=-1).view(R, -1)
    partial = torch.empty(R, cq + ckv + dr, dtype=torch.bfloat16, device="cuda")
    local = torch.empty(T, cq + ckv + dr, dtype=torch.bfloat16, device="cuda")

    def wgrads(gkv_):
        torch.matmul(gq.t(), lq, out=d_w_q)
        torch.matmul(gkv_.t(), lc, out=d_w_kv)

    def gemm():
        torch.matmul(gq, wq, out=partial[:, :cq])
        torch.matmul(gkv, wkv, out=partial[:, cq : cq + ckv])
        if wgrad:
            wgrads(gkv)

    def rs():
        dist.reduce_scatter_tensor(local, partial, group=group)

    def unfused():
        gkv_ = torch.cat((gk[..., :dn], gv), dim=-1).view(R, -1)
        torch.matmul(gq, wq, out=partial[:, :cq])
        torch.matmul(gkv_, wkv, out=partial[:, cq : cq + ckv])
        partial[:, cq + ckv :] = gk[..., dn:].float().sum(dim=1)
        dist.reduce_scatter_tensor(local, partial, group=group)
        if wgrad:
            wgrads(gkv_)

    def fused():
        ws = get_collective_gemm_workspace(group, gemm_rs_bank_bytes(R, cq, ckv, dr))
        extra = {}
        if wgrad:
            extra = dict(
                lead_wgrad=(gq, lq, d_w_q),
                wgrad=((gk[..., :dn], gv), lc, d_w_kv.view(hl, dn + dv, ckv)),
            )
        gemm_rs(
            gq,
            gk.view(R, nk),
            gv.view(R, hl * dv),
            wq_t,
            wk_t,
            wv_t,
            d_q_latent,
            d_kv_latent,
            ws,
            dn=dn,
            dr=dr,
            dv=dv,
            flag_region=5,
            **extra,
        )

    return gemm, rs, unfused, fused


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=sorted(MODELS), choices=sorted(MODELS))
    parser.add_argument("--seq", nargs="+", type=int, default=[32768])
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="also print one JSON line per op (rank 0)")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    if rank == 0:
        print(f"{'model':<14}{'seq':>7}  {'op':<19}" + "".join(f"{c:>9}" for c in COLS))
    for model in args.models:
        cfg = MODELS[model]
        for seq in args.seq:
            if seq % n or cfg["heads"] % n:
                if rank == 0:
                    print(f"{model:<14}{seq:>7}  skipped: heads or seq do not divide by n={n}")
                continue
            for op in OPS:
                gemm, rs, unfused, fused = build(cfg, seq, args.mbs, group, wgrad=op != OPS[0])
                t = dict(gemm=timeit(gemm, group), rs=timeit(rs, group))
                t["floor"] = max(t["gemm"], t["rs"])
                t["unfused"] = timeit(unfused, group)
                t["fused"] = timeit(fused, group)
                if rank == 0:
                    print(f"{model:<14}{seq:>7}  {op:<19}" + "".join(f"{t[c]:>9.1f}" for c in COLS))
                    if args.json:
                        print("JSON " + json.dumps(dict(model=model, seq=seq, n=n, op=op, **t)))
                del gemm, rs, unfused, fused
                torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
