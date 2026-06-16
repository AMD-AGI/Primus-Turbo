###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V3 per-rank FP8 perf table: dispatch_grouped_gemm (nt/nn/tn) and
grouped_gemm_combine (nt/nn/tn) — gemm-only vs fused (TFLOPS + us).

Single-process self-loopback (world=1): the comm push goes to the local pool /
combine buffer, so the full fused kernel runs without IPC. Dims = DSv3 per-rank:
M = G*R = 32*2048 = 65536 pool rows, H=7168, I=2048. Per-tensor fp8 scale.

  dispatch (contract K=H=7168): nt N=2I=4096, nn N=I=2048, tn N=I=2048
  combine  (contract K=I=2048): nt/nn/tn N=H=7168
  (dispatch+tn fused is N/A: the pool is [M,K] but a TN GEMM needs A=[K,M].)
"""

import argparse

import torch

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_fp8 import (
    dispatch_grouped_gemm_fp8, grouped_gemm_fp8_only,
)
from primus_turbo.flydsl.mega.grouped_gemm_fp8_combine import grouped_gemm_fp8_combine


class _Comm:
    def __init__(self, **k):
        self.__dict__.update(k)


def _fp8(t):
    return t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


_L2_FLUSH = None


def _l2_flush():
    """Evict L2 by zeroing a 256MB scratch buffer (> L2), so the next timed launch
    reads weights/inputs cold from HBM instead of from an L2 warmed by the prior
    iteration. Enqueued BEFORE e0.record() on the same stream, so it is not counted
    in the measured kernel time."""
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    _L2_FLUSH.zero_()


def _bench(fn, reset=None, warmup=4, iters=30):
    for _ in range(warmup):
        if reset is not None:
            reset()
        fn()
        
    _l2_flush()                       # cold-L2 each timed iter (not timed)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    us = 0.0
    for _ in range(iters):
        if reset is not None:
            reset()
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        us += e0.elapsed_time(e1) * 1000.0
    return us / iters


def _mk_meta(M, BM, device):
    """Self-loopback dispatch metadata: one comm task per BM pool block (dest=self)."""
    n_mblk = M // BM
    dest = torch.zeros(n_mblk, dtype=torch.int32, device=device)
    start = torch.arange(n_mblk, dtype=torch.int32, device=device) * BM
    cnt = torch.full((n_mblk,), BM, dtype=torch.int32, device=device)
    src = torch.arange(M, dtype=torch.int32, device=device)
    comm = _Comm(dest=dest, start=start, cnt=cnt, srcoff=start.clone(), src_tokens=src, num_comm=n_mblk)
    return comm, n_mblk


def bench_dispatch(layout, M, N, H, G, R, BM, BN, iters):
    """A=pool[M,H] fp8 (tn: [H,M]); W per-expert; gemm-only + fused (self-loopback)."""
    dev = "cuda"
    n_mblk = M // BM
    a = _fp8(torch.randn(H, M, device=dev)) if layout == "tn" else _fp8(torch.randn(M, H, device=dev))
    W = _fp8(torch.randn(G, N, H, device=dev) * 0.03) if layout == "nt" \
        else _fp8(torch.randn(G, H, N, device=dev) * 0.03)
    sa = torch.tensor([0.5], device=dev); sb = torch.tensor([1.0], device=dev)
    ttg = torch.arange(n_mblk, dtype=torch.int32, device=dev) // (R // BM)
    mblk = torch.tensor([n_mblk], dtype=torch.int32, device=dev)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    flops = 2.0 * M * N * H

    # gemm-only (pool pre-filled)
    pool = a if layout == "tn" else a.clone()
    grouped_gemm_fp8_only(pool, W, out, ttg, mblk, a_scale=sa, b_scale=sb, layout=layout, BM=BM, BN=BN)
    torch.cuda.synchronize()
    t_gg = _bench(lambda: grouped_gemm_fp8_only(pool, W, out, ttg, mblk, a_scale=sa, b_scale=sb,
                                                layout=layout, BM=BM, BN=BN), iters=iters)

    if layout == "tn":          # dispatch+tn fusion is N/A (pool [M,K] vs A [K,M])
        return t_gg, flops / (t_gg * 1e-6) / 1e12, None, None

    poolf = torch.zeros(M, H, dtype=torch.float8_e4m3fn, device=dev)
    comm, _ = _mk_meta(M, BM, dev)
    pool_ptrs = torch.tensor([poolf.data_ptr()], dtype=torch.int64, device=dev)
    sb_ = torch.zeros(n_mblk, dtype=torch.int32, device=dev)
    sb_ptrs = torch.tensor([sb_.data_ptr()], dtype=torch.int64, device=dev)
    exp = torch.ones(n_mblk, dtype=torch.int32, device=dev)

    def _fused():
        dispatch_grouped_gemm_fp8(a, comm, pool_ptrs, sb_ptrs, poolf, W, out, ttg, sb_, exp, mblk,
                                  a_scale=sa, b_scale=sb, layout=layout, BM=BM, BN=BN,
                                  autotune=True, autotune_reset=sb_.zero_)
    sb_.zero_(); _fused(); torch.cuda.synchronize()   # trigger autotune
    t_f = _bench(_fused, reset=sb_.zero_, iters=iters)
    return t_gg, flops / (t_gg * 1e-6) / 1e12, t_f, flops / (t_f * 1e-6) / 1e12


def bench_combine(layout, M, N, K, G, R, BM, BN, comb_blocks, iters):
    """A=act[M,K] fp8 (tn: [K,M]); GEMM -> l2y[M,N]; gemm-only + fused combine push."""
    dev = "cuda"
    n_mblk = M // BM
    act = _fp8(torch.randn(K, M, device=dev)) if layout == "tn" else _fp8(torch.randn(M, K, device=dev))
    W = _fp8(torch.randn(G, N, K, device=dev) * 0.03) if layout == "nt" \
        else _fp8(torch.randn(G, K, N, device=dev) * 0.03)
    sa = torch.tensor([0.5], device=dev); sb = torch.tensor([1.0], device=dev)
    ttg = torch.arange(n_mblk, dtype=torch.int32, device=dev) // (R // BM)
    mblk = torch.tensor([n_mblk], dtype=torch.int32, device=dev)
    flops = 2.0 * M * N * K

    o = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    grouped_gemm_fp8_only(act, W, o, ttg, mblk, a_scale=sa, b_scale=sb, layout=layout, BM=BM, BN=BN)
    torch.cuda.synchronize()
    t_gg = _bench(lambda: grouped_gemm_fp8_only(act, W, o, ttg, mblk, a_scale=sa, b_scale=sb,
                                                layout=layout, BM=BM, BN=BN), iters=iters)

    l2y = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    comb = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    comb_addrs = torch.tensor([comb.data_ptr()], dtype=torch.int64, device=dev)
    orank = torch.zeros(M, dtype=torch.int32, device=dev)
    oslot = torch.arange(M, dtype=torch.int32, device=dev)
    sbl2 = torch.zeros(n_mblk, dtype=torch.int32, device=dev)

    def _fused():
        grouped_gemm_fp8_combine(act, W, l2y, ttg, sbl2, orank, oslot, comb_addrs, M, mblk,
                                 a_scale=sa, b_scale=sb, layout=layout, BM=BM, BN=BN, comb_blocks=comb_blocks)
    sbl2.zero_(); _fused(); torch.cuda.synchronize()
    t_f = _bench(_fused, reset=sbl2.zero_, iters=iters)
    return t_gg, flops / (t_gg * 1e-6) / 1e12, t_f, flops / (t_f * 1e-6) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--experts-per-rank", type=int, default=32)
    ap.add_argument("--rows-per-expert", type=int, default=2048)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--comb-blocks", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.cuda.set_device(0); torch.manual_seed(7)
    H, I, G, R, BM, BN = args.hidden, args.inter, args.experts_per_rank, args.rows_per_expert, args.bm, args.bn
    M = G * R

    rows = []
    # dispatch: contract K=H; nt N=2I, nn/tn N=I
    for lay, N in [("nt", 2 * I), ("nn", I), ("tn", I)]:
        gg_us, gg_tf, f_us, f_tf = bench_dispatch(lay, M, N, H, G, R, BM, BN, args.iters)
        rows.append(("dispatch", lay, M, N, H, gg_us, gg_tf, f_us, f_tf))
    # combine: contract K=I; N=H
    for lay in ("nt", "nn", "tn"):
        gg_us, gg_tf, f_us, f_tf = bench_combine(lay, M, H, I, G, R, BM, BN, args.comb_blocks, args.iters)
        rows.append(("combine", lay, M, H, I, gg_us, gg_tf, f_us, f_tf))

    print(f"\n==== DeepSeek-V3 per-rank FP8 grouped GEMM (M={M} G={G} H={H} I={I}, BM={BM} BN={BN}) ====")
    print(f"{'op':9} {'lay':3} {'M':>6} {'N':>5} {'K':>5} | {'gemm us':>9} {'gemm TF':>8} | {'fused us':>9} {'fused TF':>8} | {'overlap':>7}")
    print("-" * 86)
    for op, lay, m, n, k, gus, gtf, fus, ftf in rows:
        if fus is None:
            print(f"{op:9} {lay:3} {m:6d} {n:5d} {k:5d} | {gus:9.1f} {gtf:8.0f} | {'N/A':>9} {'N/A':>8} | {'N/A':>7}")
        else:
            print(f"{op:9} {lay:3} {m:6d} {n:5d} {k:5d} | {gus:9.1f} {gtf:8.0f} | {fus:9.1f} {ftf:8.0f} | {fus/gus:6.2f}x")
    print("-" * 86)
    print("[note] single-rank self-loopback: comm push = LOCAL HBM (contends w/ GEMM); real multi-rank")
    print("       comm rides separate XGMI -> fused overlap is better than shown. gemm-only TF = compute peak.")


if __name__ == "__main__":
    main()
