###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused all-to-all + GEMM (``fused_collective_gemm.a2a_gemm``) against its parts, on the two
sequence-parallel attention projections it fuses (Ulysses context parallelism):

    fwd o_proj  O [s, h/n * d] back to the sequence owners, @ W_o^T + residual; the gathered O is
                kept for the weight gradient
    bwd QKV     [dQ dK dV] [s, h/n * d] back to the sequence owners, @ W_qkv, plus
                dW_qkv = [dQ dK dV]^T @ x in the same launch

Columns (us, ranks aligned by a barrier, the launch queued behind a GPU sleep so host time is hidden,
median of iterations, max over ranks):

    gemm      the GEMM coupled to the exchange (hipBLASLt; fwd with the residual as its C input)
    wgrad     the weight-grad GEMM (hipBLASLt)
    a2a       RCCL ``all_to_all_single`` of the exchanged bytes alone
    floor     max(gemm + wgrad, a2a): perfect overlap of the parts
    fused     ``a2a_gemm`` with its default (shape-based) launch knobs; where the shape does not tile
              the in-kernel wgrad, the gathered rows are copied out and the wgrad GEMM runs after it

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_a2a_gemm.py --seq 32768
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from primus_turbo.flydsl.fused_collective_gemm.a2a_gemm_kernel import (
    a2a_gemm,
    a2a_gemm_bank_bytes,
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
    # MLA: the QKV side is the latent all-gather, only the output projection is an all-to-all + GEMM
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


def _fwd_o_proj(cfg, seq, mbs, group):
    """y = O_gathered @ W_o^T + residual, O_gathered kept (as when W_o needs its gradient)."""
    n = group.size()
    hidden, hq, d = cfg["hidden"], cfg["hq"], cfg["d"]
    T, K = seq // n * mbs, hq * d
    o, w_o, res = rnd(seq * mbs, K // n), rnd(hidden, K), rnd(T, hidden)
    y, og = torch.empty(T, hidden, dtype=torch.bfloat16, device="cuda"), rnd(T, K)
    kw = dict(seq_local=seq // n, qkv_format="sbhd", head_dim=d)

    def fused():
        ws = get_collective_gemm_workspace(group, a2a_gemm_bank_bytes(T, K))
        a2a_gemm(o, w_o, res, y, og, ws, flag_region=2, **kw)

    return lambda: torch.addmm(res, og, w_o.t()), None, T * K, fused


def _bwd_qkv(cfg, seq, mbs, group):
    """dx = [dQ dK dV] @ W_qkv (Megatron's interleaved rows: hkv / n groups of hq / hkv q, 1 k, 1 v
    heads per rank), dW_qkv = [dQ dK dV]^T @ x in the same launch when the shape tiles it."""
    n = group.size()
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    T, K = seq // n * mbs, (hq + 2 * hkv) * d
    parts = tuple(rnd(seq * mbs, h // n, d) for h in (hq, hkv, hkv))
    w, x = rnd(K, hidden), rnd(T, hidden)
    dx, dw, dqkv = torch.empty_like(x), torch.empty_like(w), rnd(T, K)
    fused_wgrad = T % 256 == 0 and (K // n) % 256 == 0 and hidden % 256 == 0
    kw = dict(seq_local=seq // n, qkv_format="sbhd", layout="nn", groups=hkv // n)

    def fused():
        ws = get_collective_gemm_workspace(group, a2a_gemm_bank_bytes(T, K))
        if fused_wgrad:
            a2a_gemm(parts, w, None, dx, None, ws, flag_region=3, wgrad=(x, dw), **kw)
        else:
            a2a_gemm(parts, w, None, dx, dqkv, ws, flag_region=3, **kw)
            torch.mm(dqkv.t(), x, out=dw)

    return lambda: dqkv @ w, lambda: dqkv.t() @ x, T * K, fused


def build_ops(cfg, seq, mbs, group):
    """name -> (gemm, wgrad or None, exchanged elements, fused)."""
    ops = {"fwd o_proj (+res)": _fwd_o_proj(cfg, seq, mbs, group)}
    if cfg["hkv"]:
        ops["bwd QKV (+dW)"] = _bwd_qkv(cfg, seq, mbs, group)
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
        print(f"{'model':<18}{'seq':>7} {'op':<18}" + "".join(f"{c:>9}" for c in COLS))
    for model in args.models:
        cfg = MODELS[model]
        for seq in args.seq:
            if seq % n or cfg["hq"] % n or (cfg["hkv"] and cfg["hkv"] % n):
                if rank == 0:
                    print(f"{model:<18}{seq:>7} skipped: heads or seq do not divide by n={n}")
                continue
            for name, (gemm, wgrad, numel, fused) in build_ops(cfg, seq, args.mbs, group).items():
                t = dict(gemm=timeit(gemm, group), wgrad=timeit(wgrad, group) if wgrad else 0.0)
                t["a2a"] = timeit(a2a_fn(numel, group), group)
                t["floor"] = max(t["gemm"] + t["wgrad"], t["a2a"])
                t["fused"] = timeit(fused, group)
                if rank == 0:
                    print(f"{model:<18}{seq:>7} {name:<18}" + "".join(f"{t[c]:>9.1f}" for c in COLS))
                    if args.json:
                        print("JSON " + json.dumps(dict(model=model, seq=seq, n=n, op=name, **t)))
            torch.cuda.empty_cache()
    dist.destroy_process_group()


COLS = ("gemm", "wgrad", "a2a", "floor", "fused")

if __name__ == "__main__":
    main()
