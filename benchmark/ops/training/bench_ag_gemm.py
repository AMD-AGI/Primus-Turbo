###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused all-gather + GEMM (``fused_collective_gemm.ag_gemm``) against its parts, on the MLA latent
up-projection under sequence parallelism (Ulysses): every rank all-gathers ``[q_latent | c_kv | k_pe]``
of its local tokens and up-projects the whole sequence for its local heads, keeping the gathered
latents for the weight gradients.

Columns (us, ranks aligned by a barrier, the launch queued behind a GPU sleep so host time is hidden,
median of iterations, max over ranks):

    gemm      the q and kv up-projection GEMMs over the gathered rows (hipBLASLt)
    ag        RCCL ``all_gather_into_tensor`` of the latent rows alone
    floor     max(gemm, ag): perfect overlap of the parts
    unfused   RCCL all-gather, then the GEMMs and the ``k = [k_nope | k_pe]`` / ``v`` layout copies
    fused     ``ag_gemm`` with its default launch knobs

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_ag_gemm.py --seq 32768
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from primus_turbo.flydsl.fused_collective_gemm.ag_gemm_kernel import (
    ag_gemm,
    ag_gemm_bank_bytes,
)
from primus_turbo.flydsl.fused_collective_gemm.workspace import (
    get_collective_gemm_workspace,
)

DSV3 = dict(heads=128, cq=1536, ckv=512, dn=128, dr=64, dv=128)
MODELS = {"deepseek-v3": DSV3, "kimi-k2": dict(DSV3, heads=64)}


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


def build(cfg, seq, mbs, group):
    """(gemm, ag, unfused, fused) callables for the latent up-projection forward."""
    n = group.size()
    hl, cq, ckv, dn, dr, dv = cfg["heads"] // n, cfg["cq"], cfg["ckv"], cfg["dn"], cfg["dr"], cfg["dv"]
    T = seq // n * mbs
    R = n * T
    xq, xkv = rnd(T, cq), rnd(T, ckv + dr)
    # this rank's up-projection rows; kv rows: k_nope of every local head, then v
    wq, wkv = rnd(hl * (dn + dr), cq), rnd(hl * (dn + dv), ckv)
    q, k = (torch.empty(R, hl, dn + dr, dtype=torch.bfloat16, device="cuda") for _ in range(2))
    v = torch.empty(R, hl, dv, dtype=torch.bfloat16, device="cuda")
    # the unfused path's kv weight rows in the natural layout: k_nope then v of each head in turn
    wkv_nat = rnd(hl * (dn + dv), ckv)
    lq, lc = rnd(R, cq), rnd(R, ckv)
    packed = torch.cat((xq, xkv), dim=1)
    gathered = torch.empty(R, cq + ckv + dr, dtype=torch.bfloat16, device="cuda")

    def gemm():
        return lq @ wq.t(), lc @ wkv.t()

    def ag():
        dist.all_gather_into_tensor(gathered, packed, group=group)

    def unfused():
        dist.all_gather_into_tensor(gathered, torch.cat((xq, xkv), dim=1), group=group)
        g_lq, g_lkv = gathered[:, :cq], gathered[:, cq:]
        q_ = g_lq @ wq.t()
        kv = (g_lkv[:, :ckv] @ wkv_nat.t()).view(R, hl, dn + dv)
        k_ = torch.empty(R, hl, dn + dr, dtype=torch.bfloat16, device="cuda")
        k_[..., :dn] = kv[..., :dn]
        k_[..., dn:] = g_lkv[:, None, ckv:]
        return q_, k_, kv[..., dn:].contiguous()

    def fused():
        ws = get_collective_gemm_workspace(group, ag_gemm_bank_bytes(R, cq, ckv, dr))
        ag_gemm(xq, xkv, wq, wkv, q, k, v, lq, lc, ws, dn=dn, dr=dr, dv=dv, flag_region=4)

    return gemm, ag, unfused, fused


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
        print(f"{'model':<18}{'seq':>7}" + "".join(f"{c:>9}" for c in COLS))
    for model in args.models:
        cfg = MODELS[model]
        for seq in args.seq:
            if seq % n or cfg["heads"] % n:
                if rank == 0:
                    print(f"{model:<18}{seq:>7} skipped: heads or seq do not divide by n={n}")
                continue
            gemm, ag, unfused, fused = build(cfg, seq, args.mbs, group)
            t = dict(gemm=timeit(gemm, group), ag=timeit(ag, group))
            t["floor"] = max(t["gemm"], t["ag"])
            t["unfused"] = timeit(unfused, group)
            t["fused"] = timeit(fused, group)
            if rank == 0:
                print(f"{model:<18}{seq:>7}" + "".join(f"{t[c]:>9.1f}" for c in COLS))
                if args.json:
                    print("JSON " + json.dumps(dict(model=model, seq=seq, n=n, op="fwd latent up-proj", **t)))
            del gemm, ag, unfused, fused
            torch.cuda.empty_cache()
    dist.destroy_process_group()


COLS = ("gemm", "ag", "floor", "unfused", "fused")

if __name__ == "__main__":
    main()
