###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-process self-dispatch correctness + perf test for the fused FP8
dispatch + grouped GEMM (NT).

world=1 self-loopback: every token is pushed to the local pool, so the full
fused kernel (XGMI-style push to self + grouped FP8 GEMM + scoreboard handshake)
runs in one process with no IPC. The GEMM workload reproduces the DeepSeek-V3
per-rank forward L1: G=32 local experts, M~65536 pool rows, N=2I=4096, K=H=7168.

Reports:
  * correctness vs an fp8 grouped-matmul reference (cosine sim),
  * fused TFLOPS (dispatch hidden under GEMM),
  * pure grouped-FP8-GEMM TFLOPS (compute peak baseline),
  * dense FP8 NT GEMM roofline (achievable MFMA fp8 TFLOPS), and the % attained.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python -m primus_turbo.flydsl.mega.test_dispatch_grouped_gemm_fp8
"""

import argparse
import time

import torch

import flydsl.compiler as flyc

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_fp8 import (
    dispatch_grouped_gemm_fp8,
    grouped_gemm_fp8_only,
)
from primus_turbo.flydsl.gemm.gemm_fp8_kernel_v2 import compile_dense_nt_tiled


class _Comm:
    """Minimal CommTasks stand-in (the kernel reads only these fields)."""

    def __init__(self, dest, start, cnt, srcoff, src_tokens, num_comm):
        self.dest = dest
        self.start = start
        self.cnt = cnt
        self.srcoff = srcoff
        self.src_tokens = src_tokens
        self.num_comm = num_comm


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


def _dense_fp8_roofline(n=8192, K=8192):
    """Achievable dense FP8 NT MFMA TFLOPS on this GPU (the compute peak)."""
    a = _fp8(torch.randn(n, K, device="cuda"))
    b = _fp8(torch.randn(n, K, device="cuda"))  # B^T storage [N,K]
    sa = torch.tensor([1.0], device="cuda")
    sb = torch.tensor([1.0], device="cuda")
    out = torch.empty(n, n, dtype=torch.bfloat16, device="cuda")
    launch = compile_dense_nt_tiled(K=K, BLOCK_M=256, BLOCK_N=256, GROUP_M=4)
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), out.view(-1),
            sa, sb, n, n, torch.cuda.current_stream())
    compiled = flyc.compile(launch, *args)   # compile once; bench the hot launch
    t = _bench(lambda: compiled(*args))
    return 2.0 * n * n * K / (t * 1e-6) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=7168)         # K = H
    ap.add_argument("--inter", type=int, default=2048)          # I; out_features N = 2I
    ap.add_argument("--experts-per-rank", type=int, default=32) # G (E/world at E=256,world=8)
    ap.add_argument("--rows-per-expert", type=int, default=2048)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--layout", default="nt", choices=["nt", "nn"])
    ap.add_argument("--no-autotune", action="store_true")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(7)
    LAYOUT = args.layout
    H = args.hidden
    # NT (fwd L1): out_features N = 2I. NN (bwd dgrad): N = I.
    N = (2 * args.inter) if LAYOUT == "nt" else args.inter
    G = args.experts_per_rank
    R = args.rows_per_expert
    BM, BN = args.bm, args.bn
    assert R % BM == 0, "rows_per_expert must be a multiple of BM"
    M = G * R
    n_mblk = M // BM
    blocks_per_expert = R // BM

    # ---- inputs (fp8) ----
    x = _fp8(torch.randn(M, H, device="cuda"))                       # source tokens [M,H=K]
    # NT weight [G,N,K] (B^T storage); NN weight [G,K,N]
    W = _fp8(torch.randn(G, N, H, device="cuda") * 0.05) if LAYOUT == "nt" \
        else _fp8(torch.randn(G, H, N, device="cuda") * 0.05)
    a_scale = torch.tensor([0.5], device="cuda")
    b_scale = torch.tensor([1.0], device="cuda")

    pool = torch.zeros(M, H, dtype=torch.float8_e4m3fn, device="cuda")
    output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    # ---- dispatch metadata: one comm task per BM pool block (self -> local pool) ----
    dest = torch.zeros(n_mblk, dtype=torch.int32, device="cuda")
    start = (torch.arange(n_mblk, dtype=torch.int32, device="cuda") * BM)
    cnt = torch.full((n_mblk,), BM, dtype=torch.int32, device="cuda")
    srcoff = start.clone()
    src_tokens = torch.arange(M, dtype=torch.int32, device="cuda")
    comm = _Comm(dest, start, cnt, srcoff, src_tokens, n_mblk)

    pool_ptrs = torch.tensor([pool.data_ptr()], dtype=torch.int64, device="cuda")
    scoreboard = torch.zeros(n_mblk, dtype=torch.int32, device="cuda")
    scoreboard_ptrs = torch.tensor([scoreboard.data_ptr()], dtype=torch.int64, device="cuda")
    # tile_to_group: pool block b -> expert b // blocks_per_expert
    tile_to_group = (torch.arange(n_mblk, dtype=torch.int32, device="cuda") // blocks_per_expert)
    expected = torch.ones(n_mblk, dtype=torch.int32, device="cuda")   # 1 task / block
    mblk_dev = torch.tensor([n_mblk], dtype=torch.int32, device="cuda")

    print(f"[cfg] layout={LAYOUT} M={M} N={N} K={H} G={G} R={R} BM={BM} BN={BN} n_mblk={n_mblk}")

    # ===== correctness =====
    scoreboard.zero_()
    dispatch_grouped_gemm_fp8(
        x, comm, pool_ptrs, scoreboard_ptrs, pool, W, output,
        tile_to_group, scoreboard, expected, mblk_dev,
        a_scale=a_scale, b_scale=b_scale, layout=LAYOUT, BM=BM, BN=BN, comm_blocks=32, nt_vmcnt=3)
    torch.cuda.synchronize()

    # fp8 grouped reference (exact fp8 values -> float -> matmul, f32 accum)
    sa = float(a_scale.item()); sb = float(b_scale.item())
    cos_acc = 0.0
    for g in range(G):
        xb = x[g * R:(g + 1) * R].float() * sa
        wb = W[g].float() * sb
        ref = (xb @ wb.t()) if LAYOUT == "nt" else (xb @ wb)   # NT: B^T storage; NN: B [K,N]
        cos_acc += _cos(output[g * R:(g + 1) * R], ref)
    cos = cos_acc / G
    pool_ok = bool((pool.float().abs().sum(1) > 0).all().item())
    print(f"[correct] mean cosine over {G} experts = {cos:.6f}  pool_filled={pool_ok}")

    # ===== perf: dense fp8 roofline =====
    peak_tf = _dense_fp8_roofline()
    print(f"[roofline] dense FP8 NT MFMA peak = {peak_tf:.0f} TFLOPS")

    flops = 2.0 * M * N * H

    # ===== perf: pure grouped-fp8-GEMM (pool pre-filled = peak compute) =====
    pool.copy_(x)  # pre-fill so the gemm-only kernel reads real data
    grouped_gemm_fp8_only(pool, W, output, tile_to_group, mblk_dev,
                          a_scale=a_scale, b_scale=b_scale, layout=LAYOUT, BM=BM, BN=BN)
    torch.cuda.synchronize()
    t_gg = _bench(lambda: grouped_gemm_fp8_only(pool, W, output, tile_to_group, mblk_dev,
                                                a_scale=a_scale, b_scale=b_scale, layout=LAYOUT, BM=BM, BN=BN),
                  iters=args.iters)
    gg_tf = flops / (t_gg * 1e-6) / 1e12
    print(f"[gemm-only] {t_gg:8.1f} us  |  {gg_tf:6.0f} TFLOPS  => {100*gg_tf/peak_tf:4.0f}% of dense peak")

    # ===== perf: fused dispatch + grouped GEMM (autotuned) =====
    autotune = not args.no_autotune

    def _fused():
        dispatch_grouped_gemm_fp8(
            x, comm, pool_ptrs, scoreboard_ptrs, pool, W, output,
            tile_to_group, scoreboard, expected, mblk_dev,
            a_scale=a_scale, b_scale=b_scale, layout=LAYOUT, BM=BM, BN=BN, comm_blocks=32, nt_vmcnt=3,
            autotune=autotune, autotune_reset=scoreboard.zero_)

    scoreboard.zero_(); _fused(); torch.cuda.synchronize()   # triggers autotune (cached after)
    if autotune:
        from primus_turbo.flydsl.mega.dispatch_grouped_gemm_fp8 import _DISPATCH_AUTOTUNE_CACHE
        cfg = next(iter(_DISPATCH_AUTOTUNE_CACHE.values()))[1]
        print(f"[autotune] best (comm_blocks, nt_vmcnt, agpr, waves) = {cfg}")
    t_fused = _bench(_fused, reset=scoreboard.zero_, iters=args.iters)
    fused_tf = flops / (t_fused * 1e-6) / 1e12
    print(f"[fused   ] {t_fused:8.1f} us  |  {fused_tf:6.0f} TFLOPS  => {100*fused_tf/peak_tf:4.0f}% of dense peak")
    print(f"[overlap ] fused/gemm-only = {t_fused/t_gg:.3f}x "
          f"(dispatch {'HIDDEN' if t_fused <= 1.05*t_gg else 'EXPOSED'} under the GEMM)")
    print("[note] single-rank self-loopback: the dispatch push is LOCAL HBM and "
          "contends with the GEMM's weight reads, so 'exposed' here overstates the\n"
          "       real multi-rank cost (cross-rank dispatch rides the separate XGMI "
          "fabric). The grouped-GEMM peak (gemm-only) is the headline number.")


if __name__ == "__main__":
    main()
