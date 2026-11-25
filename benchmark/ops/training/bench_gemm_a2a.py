###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused GEMM + all-to-all (``fused_collective_gemm.gemm_a2a``) against its parts, on the two
sequence-parallel attention projections it fuses (Ulysses context parallelism):

    fwd QKV     x [s/n, hidden] @ W_qkv^T, head-sharded columns scattered to the head owners
    bwd o_proj  dY [s/n, hidden] @ W_o scattered back to the head owners, plus dW_o = dY^T @ O
                in the same launch

Columns (us, ranks aligned by a barrier, the launch queued behind a GPU sleep so host time is hidden,
median of iterations, max over ranks):

    gemm      the GEMM coupled to the exchange (hipBLASLt)
    wgrad     the weight-grad GEMM (hipBLASLt)
    a2a       RCCL ``all_to_all_single`` of the exchanged bytes alone
    floor     max(gemm + wgrad, a2a): perfect overlap of the parts
    fused     ``gemm_a2a``, launch knobs as below

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_gemm_a2a.py --seq 32768
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

from primus_turbo.flydsl.fused_collective_gemm.gemm_a2a_kernel import (
    gemm_a2a,
    gemm_a2a_bank_bytes,
)
from primus_turbo.flydsl.fused_collective_gemm.workspace import (
    get_collective_gemm_workspace,
)

MODELS = {
    "llama3.1-8b": dict(hidden=4096, hq=32, hkv=8, d=128),
    "llama3.1-70b": dict(hidden=8192, hq=64, hkv=8, d=128),
    "qwen3-30b-a3b": dict(hidden=2048, hq=32, hkv=4, d=128),
    "qwen3-235b-a22b": dict(hidden=4096, hq=64, hkv=4, d=128),
    "glm-4.7": dict(hidden=5120, hq=96, hkv=8, d=128),
    "minimax-m2.5": dict(hidden=3072, hq=48, hkv=8, d=128),
    # MLA: the QKV side is the latent all-gather, only the output projection is a GEMM + all-to-all
    "deepseek-v3": dict(hidden=7168, hq=128, hkv=0, d=128),
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


def rnd(*shape):
    return (torch.randn(*shape, device="cuda") * 0.02).bfloat16()


def a2a_fn(numel, group):
    inp = torch.empty(numel, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    return lambda: dist.all_to_all_single(out, inp, group=group)


def _fwd_qkv(cfg, seq, mbs, group, num_cus):
    """Interleaved QKV rows: each rank's column block is hkv / n groups of (hq / hkv q, 1 k, 1 v) heads."""
    n = group.size()
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    T, N = seq // n * mbs, (hq + 2 * hkv) * d
    x, w = rnd(T, hidden), rnd(N, hidden)
    outs = tuple(
        torch.empty(seq * mbs, h // n, d, dtype=torch.bfloat16, device="cuda") for h in (hq, hkv, hkv)
    )
    # 256-row tiles pay from ~3 waves of them on
    BM = 256 if -(-T // 256) * -(-N // 256) >= 3 * num_cus else 128
    kw = dict(head_dim=d, groups=hkv // n, nq=hq // hkv, nk=1, nv=1, seq_local=seq // n, qkv_format="sbhd")

    def fused():
        ws = get_collective_gemm_workspace(group, gemm_a2a_bank_bytes(T, N, 256))
        gemm_a2a(x, w, outs, ws, layout="nt", flag_region=0, BM=BM, num_pull_cu=24, num_tail_pull=8, **kw)

    return lambda: F.linear(x, w), None, T * N, fused, BM


def _bwd_o_proj(cfg, seq, mbs, group):
    """dO = dY @ W_o back to the head owners, dW_o = dY^T @ O in the same launch."""
    n = group.size()
    hidden, hq, d = cfg["hidden"], cfg["hq"], cfg["d"]
    T, K = seq // n * mbs, hq * d
    dy, w_o, o = rnd(T, hidden), rnd(hidden, K), rnd(T, K)
    do, dw = torch.empty(seq * mbs, hq // n, d, dtype=torch.bfloat16, device="cuda"), torch.empty_like(w_o)
    kw = dict(head_dim=d, groups=1, nq=hq // n, nk=0, nv=0, seq_local=seq // n, qkv_format="sbhd")

    def fused():
        ws = get_collective_gemm_workspace(group, gemm_a2a_bank_bytes(T, K, 256))
        gemm_a2a(
            dy,
            w_o,
            (do, do, do),
            ws,
            layout="nn",
            flag_region=1,
            BM=256,
            num_pull_cu=8 if n == 2 else 16,
            wgrad=(dy, o, dw),
            **kw,
        )

    return lambda: dy @ w_o, lambda: dy.t() @ o, T * K, fused, 256


def build_ops(cfg, seq, mbs, group):
    """name -> (gemm, wgrad or None, exchanged elements, fused, BM)."""
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    ops = {"fwd QKV": _fwd_qkv(cfg, seq, mbs, group, num_cus)} if cfg["hkv"] else {}
    ops["bwd o_proj (+dW)"] = _bwd_o_proj(cfg, seq, mbs, group)
    return ops


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
        print(f"{'model':<18}{'seq':>7} {'op':<18}{'BM':>4}" + "".join(f"{c:>9}" for c in COLS))
    for model in args.models:
        cfg = MODELS[model]
        for seq in args.seq:
            if seq % n or cfg["hq"] % n or (cfg["hkv"] and cfg["hkv"] % n):
                if rank == 0:
                    print(f"{model:<18}{seq:>7} skipped: heads or seq do not divide by n={n}")
                continue
            for name, (gemm, wgrad, numel, fused, BM) in build_ops(cfg, seq, args.mbs, group).items():
                t = dict(gemm=timeit(gemm, group), wgrad=timeit(wgrad, group) if wgrad else 0.0)
                t["a2a"] = timeit(a2a_fn(numel, group), group)
                t["floor"] = max(t["gemm"] + t["wgrad"], t["a2a"])
                t["fused"] = timeit(fused, group)
                if rank == 0:
                    print(f"{model:<18}{seq:>7} {name:<18}{BM:>4}" + "".join(f"{t[c]:>9.1f}" for c in COLS))
                    if args.json:
                        print("JSON " + json.dumps(dict(model=model, seq=seq, n=n, op=name, BM=BM, **t)))
            torch.cuda.empty_cache()
    dist.destroy_process_group()


COLS = ("gemm", "wgrad", "a2a", "floor", "fused")

if __name__ == "__main__":
    main()
