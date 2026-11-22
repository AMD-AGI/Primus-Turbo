###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Microbench of the EP>1 FP8-block MegaMoE kernels (quant_blk_ep_gfx1250 + the torch layout glue) at the real EP8
per-rank shape: one rank of an EP8 group, T = 4096 tokens per rank, E = 256 (E_l = 32), K = 6, H = 4096, I = 2048,
uniform routing. The group is emulated (EmulatedEPComm(8)) only to produce rank 0's received buffers; every timed
call is rank 0's. Median of N reps (CUDA events, one call per event pair, inputs warm in cache where they fit).
HBM floor = 2.2 us + compulsory bytes / 18.3 TB/s (ROOFLINE_METHOD.md).

  python bench_fused_mega_moe_fp8blk_ep.py [--reps 50] [--R 8] [--T 4096]
"""

import argparse

import torch

import primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk as FM
import primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk_ep as EP
from primus_turbo.triton.quantization import quant_blk_ep_gfx1250 as QE
from primus_turbo.triton.quantization.quant_blk_gfx1250 import combine_rows, quant_dual_blk

dev = "cuda"
HBM = 18.3e12


def bench(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def row(name, us, nbytes, note=""):
    med, lo, hi = us
    floor = 2.2 + nbytes / HBM * 1e6 if nbytes else None
    eff = f"{floor / med * 100:5.1f}%" if floor else "   - "
    fl = f"{floor:8.1f}" if floor else "       -"
    print(f"| {name:44s} | {med:8.1f} | {lo:7.1f} | {hi:7.1f} | {nbytes / 1e6:8.1f} | {fl} | {eff} | {note}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--El", type=int, default=32)
    ap.add_argument("--sweep", action="store_true", help="tile-size variants of the new kernels")
    a = ap.parse_args()
    R, T, El, K, H, I = a.R, a.T, a.El, 6, 4096, 2048
    E = R * El
    g = torch.Generator(device=dev).manual_seed(0)
    xs = [torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16) for _ in range(R)]
    idxs, tws = [], []
    for _ in range(R):
        top, idx = torch.randn(T, E, device=dev, generator=g).topk(K, dim=1)
        idxs.append(idx)
        tws.append(torch.softmax(top, 1))
    comm = EP.EmulatedEPComm(R)
    st = EP._EPState(comm, El, K, H, I, T, "fp8")
    st.idxs = idxs
    for rk in st.ranks:
        rk.T = T
    # the whole group's dispatch (emulated), keeping rank 0's received buffers
    metas, cnts, sq, ss, sbf = [], [], [], [], []
    for rk, x, idx, tw in zip(st.ranks, xs, idxs, tws):
        m, c = EP._src_layout(st, rk, idx, tw)
        q, s = QE.quant_rows_send(x, rk.slot, st.cap)
        qb, _ = QE.quant_rows_send(x, rk.slot, st.cap, quant=False)
        metas.append(m), cnts.append(c), sq.append(q), ss.append(s), sbf.append(qb)
    rm, rc, rq, rs, rb = (comm.all_to_all(v)[0] for v in (metas, cnts, sq, ss, sbf))
    del metas, cnts, sq, ss, sbf
    r0 = st.ranks[0]
    EP._recv_tables(st, r0, rm, rc)
    torch.cuda.synchronize()
    N, P, ct = st.N, st.P, st.ct
    preal = int(r0.preal.item())
    sent0 = int((r0.slot >= 0).sum().item())
    live_routes = int((r0.inv >= 0).sum().item())
    print(f"EP{R} rank 0: T={T} E={E} E_l={El} K={K} H={H} I={I}; rows sent by rank 0 {sent0} "
          f"({sent0 / T:.2f} ranks/token); received rows N={N} (fixed), live routes {live_routes}, "
          f"pool P={P} (fixed), live pool rows {preal}; wgrad={st.wk} compact_t={ct}", flush=True)
    print("| kernel (rank 0, one call) | median us | min | max | MB moved | floor us | floor/med | note |")
    print("|---|---|---|---|---|---|---|---|")
    reps = a.reps
    x0, idx0, tw0 = xs[0], idxs[0], tws[0]
    # --- source side
    qbuf = torch.empty((R, st.cap, H), device=dev, dtype=torch.float8_e4m3fn)
    sbuf = torch.empty((R, st.cap, H // 128), device=dev, dtype=torch.uint8)
    row("quant_rows_send (x 1x128 -> send slots)", bench(lambda: QE.quant_rows_send(x0, r0.slot, st.cap, q=qbuf, s=sbuf),
                                                          reps), T * H * 2 + sent0 * (H + H // 128))
    row("_src_layout = src_layout_ep (slots, cslots, meta)", bench(lambda: EP._src_layout(st, r0, idx0, tw0), reps), 0,
        "3 Triton kernels")
    # --- expert side
    row("_recv_tables (masked routing + combine map)", bench(lambda: EP._recv_tables(st, r0, rm, rc), reps), 0,
        "3 Triton + 3 torch kernels")
    li = torch.where((torch.arange(st.cap, device=dev)[None, :] < rc)[:, :, None], rm[:, :, :K], -1).reshape(N, K)
    m2 = rm.view(N, 3 * K)
    lw, cs = m2[:, K:2 * K].view(torch.float32), m2[:, 2 * K:]
    row("  build_routing_blk_masked (+ pool->slot map)",
        bench(lambda: QE.build_routing_blk_masked(li, lw, El, P, cslot=cs, cap=st.cap, ccap=st.ccap), reps), 0)
    rqv, rsv, rbv = rq.view(N, H), rs.view(N, H // 128), rb.view(N, H)
    row("gather_dual_fp8 (pool 1x128 copy + 128x1 requant)",
        bench(lambda: QE.gather_dual_fp8(rqv, rsv, P, r0.rowmap, preal=r0.preal, compact_t=ct), reps),
        preal * (3 * H + H // 128 + H // 32))
    row("  ref: quant_dual_blk from bf16 rows (bf16 dispatch)",
        bench(lambda: quant_dual_blk(rbv, P, r0.rowmap, preal=r0.preal, compact_t=ct), reps),
        preal * (4 * H + H // 32 + H // 128))
    l2 = torch.randn(P, H, device=dev).to(torch.bfloat16)
    row("pack_pool_rows (combine pack, pool -> slots)",
        bench(lambda: QE.pack_pool_rows(l2, r0.p2s, r0.preal, R * st.ccap), reps), live_routes * H * 4,
        f"{live_routes} live routes of {preal} live pool rows")
    # --- source side combine: what rank 0 gets back is [R, ccap, H]; combine_rows over T*K routes
    recv = torch.randn(R * st.ccap, H, device=dev).to(torch.bfloat16)
    row("combine_rows (y[t] = sum_k row, source side)", bench(lambda: combine_rows(recv, r0.cinv, T, K), reps),
        T * K * H * 2 + T * H * 2)
    # --- EP=1 counterparts at one rank's nominal load (T=4096 tokens, 32 experts)
    from primus_turbo.triton.quantization.quant_blk_gfx1250 import build_routing_blk

    i1 = torch.randn(T, El, device=dev, generator=g).topk(K, dim=1)[1]
    P1 = T * K + El * FM._ALIGN
    rt = build_routing_blk(i1, tws[0], El, P1)
    row("EP=1 ref: build_routing_blk (T=4096, E=32)", bench(lambda: build_routing_blk(i1, tws[0], El, P1), reps), 0)
    row("EP=1 ref: quant_dual_blk(x, rowmap) (dispatch quant)",
        bench(lambda: quant_dual_blk(x0, P1, rt[0], preal=rt[4], compact_t=ct), reps),
        int(rt[4].item()) * (4 * H + H // 32 + H // 128))
    if a.sweep:
        print("| sweep variant | median us | min | max | MB moved | floor us | floor/med | note |")
        for bt in (16, 32, 64):
            row(f"quant_rows_send bt={bt}",
                bench(lambda: QE.quant_rows_send(x0, r0.slot, st.cap, q=qbuf, s=sbuf, bt=bt), reps),
                T * H * 2 + sent0 * (H + H // 128))
        for brr in (16, 32, 64):
            row(f"pack_pool_rows br={brr}", bench(lambda: QE.pack_pool_rows(l2, r0.p2s, r0.preal, R * st.ccap, br=brr),
                                                  reps), live_routes * H * 4)
        for blk in (128, 256, 512, 1024, 2048):
            row(f"build_routing_blk_masked block={blk}",
                bench(lambda: QE.build_routing_blk_masked(li, lw, El, P, block=blk, cslot=cs, cap=st.cap,
                                                          ccap=st.ccap), reps), 0)
    print("note: floor counts live rows only; the fixed-shape grids also launch programs for the dead (worst-case) "
          "rows, which exit at once.", flush=True)


if __name__ == "__main__":
    main()
