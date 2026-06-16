###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-process self-loopback correctness + perf test for the fused FP8
grouped GEMM + combine PUSH (the K2 mirror of the dispatch test).

world=1 self-loopback: every finished L2Y row is pushed back to the LOCAL combine
buffer at slot=row, so the full fused kernel (grouped FP8 GEMM + scoreboard
handshake + combine push) runs in one process with no IPC. The workload mirrors
the DeepSeek-V3 per-rank K2: contract K, produce N output features per row.

  layout nt: A=act[M,K], W=[G,N,K]        (fwd L2: K=I, N=H)
  layout nn: A=act[M,K], W=[G,K,N]
  layout tn: A=act[K,M], W=[G,K,N]

Reports per layout: correctness (combine_buf vs grouped fp8 reference, cosine),
fused TFLOPS, pure grouped-GEMM TFLOPS (the compute peak), and the overlap ratio.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python -m primus_turbo.flydsl.mega.test_grouped_gemm_fp8_combine --layout nt
"""

import argparse
import time

import torch

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_fp8 import grouped_gemm_fp8_only
from primus_turbo.flydsl.mega.grouped_gemm_fp8_combine import grouped_gemm_fp8_combine


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _fp8(t):
    return t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _bench(fn, reset=None, warmup=4, iters=30):
    for _ in range(warmup):
        if reset is not None:
            reset()
        fn()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="nt", choices=["nt", "nn", "tn"])
    ap.add_argument("--kdim", type=int, default=2048)     # K contraction (DSv3 L2: I=2048)
    ap.add_argument("--ndim", type=int, default=7168)     # N output features (DSv3 L2: H=7168)
    ap.add_argument("--experts-per-rank", type=int, default=32)
    ap.add_argument("--rows-per-expert", type=int, default=2048)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--comb-blocks", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(7)
    LAYOUT = args.layout
    K, N = args.kdim, args.ndim
    G, R = args.experts_per_rank, args.rows_per_expert
    BM, BN = args.bm, args.bn
    assert R % BM == 0 and N % BN == 0
    M = G * R
    n_mblk = M // BM
    blocks_per_expert = R // BM

    # ---- inputs (fp8) ----
    act = _fp8(torch.randn(K, M, device="cuda")) if LAYOUT == "tn" \
        else _fp8(torch.randn(M, K, device="cuda"))
    W = _fp8(torch.randn(G, N, K, device="cuda") * 0.03) if LAYOUT == "nt" \
        else _fp8(torch.randn(G, K, N, device="cuda") * 0.03)
    a_scale = torch.tensor([0.5], device="cuda")
    b_scale = torch.tensor([1.0], device="cuda")

    l2y = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")          # local GEMM output
    comb = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")         # combine buffer (slots=M)
    comb_addrs = torch.tensor([comb.data_ptr()], dtype=torch.int64, device="cuda")
    origin_rank = torch.zeros(M, dtype=torch.int32, device="cuda")        # self (rank 0)
    origin_slot = torch.arange(M, dtype=torch.int32, device="cuda")       # identity slot
    sb_l2 = torch.zeros(n_mblk, dtype=torch.int32, device="cuda")
    tile_to_group = (torch.arange(n_mblk, dtype=torch.int32, device="cuda") // blocks_per_expert)
    mblk_dev = torch.tensor([n_mblk], dtype=torch.int32, device="cuda")

    print(f"[cfg] layout={LAYOUT} M={M} N={N} K={K} G={G} R={R} BM={BM} BN={BN} comb_blocks={args.comb_blocks}")

    # ===== correctness =====
    sb_l2.zero_()
    grouped_gemm_fp8_combine(act, W, l2y, tile_to_group, sb_l2, origin_rank, origin_slot,
                             comb_addrs, M, mblk_dev, a_scale=a_scale, b_scale=b_scale,
                             layout=LAYOUT, BM=BM, BN=BN, comb_blocks=args.comb_blocks)
    torch.cuda.synchronize()

    sa = float(a_scale.item()); sb = float(b_scale.item())

    def ref(g):
        wb = W[g].float() * sb
        if LAYOUT == "nt":
            return (act[g * R:(g + 1) * R].float() * sa) @ wb.t()
        if LAYOUT == "nn":
            return (act[g * R:(g + 1) * R].float() * sa) @ wb
        return (act[:, g * R:(g + 1) * R].t().float() * sa) @ wb   # tn

    cos = sum(_cos(comb[g * R:(g + 1) * R], ref(g)) for g in range(G)) / G
    print(f"[correct] combine_buf mean cosine over {G} experts = {cos:.6f}")

    flops = 2.0 * M * N * K

    # ===== pure grouped GEMM (compute peak) =====
    o = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    grouped_gemm_fp8_only(act, W, o, tile_to_group, mblk_dev, a_scale=a_scale, b_scale=b_scale,
                          layout=LAYOUT, BM=BM, BN=BN)
    torch.cuda.synchronize()
    t_gg = _bench(lambda: grouped_gemm_fp8_only(act, W, o, tile_to_group, mblk_dev,
                                                a_scale=a_scale, b_scale=b_scale, layout=LAYOUT,
                                                BM=BM, BN=BN), iters=args.iters)
    gg_tf = flops / (t_gg * 1e-6) / 1e12
    print(f"[gemm-only] {t_gg:8.1f} us  |  {gg_tf:6.0f} TFLOPS")

    # ===== fused GEMM + combine push =====
    def _fused():
        grouped_gemm_fp8_combine(act, W, l2y, tile_to_group, sb_l2, origin_rank, origin_slot,
                                 comb_addrs, M, mblk_dev, a_scale=a_scale, b_scale=b_scale,
                                 layout=LAYOUT, BM=BM, BN=BN, comb_blocks=args.comb_blocks)

    t_fused = _bench(_fused, reset=sb_l2.zero_, iters=args.iters)
    fused_tf = flops / (t_fused * 1e-6) / 1e12
    print(f"[fused   ] {t_fused:8.1f} us  |  {fused_tf:6.0f} TFLOPS")
    print(f"[overlap ] fused/gemm-only = {t_fused/t_gg:.3f}x "
          f"(combine push {'HIDDEN' if t_fused <= 1.05*t_gg else 'EXPOSED'} under the GEMM)")
    print("[note] single-rank self-loopback: the combine push is LOCAL HBM and contends with the "
          "GEMM; real multi-rank combine rides the separate XGMI fabric, so 'exposed' overstates it.")


if __name__ == "__main__":
    main()
