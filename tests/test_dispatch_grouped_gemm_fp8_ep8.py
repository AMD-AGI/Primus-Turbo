"""EP8 cross-rank (XGMI) unit test for the Primus-Turbo fp8 fused dispatch + grouped
GEMM (NT, forward L1).

Spawns ``world`` processes (default 8 = EP8) with a real ``SymmetricMemory`` pool, so
``dispatch_grouped_gemm_fp8`` performs ACTUAL cross-rank XGMI peer writes -- not the
single-process self-loopback (whose dispatch contends with the GEMM on local HBM and
overstates the comm cost). Uses a simple deterministic all-to-all so the routing is
collision-free and verifiable WITHOUT the workspace dispatch planner:

  rank r sends ``per_peer`` tokens to EACH peer p; peer p's pool reserves one segment
  per sender (rows [s*per_peer, (s+1)*per_peer) for sender s), so writes never collide.

Validates:
  * dispatch content cross-rank (peer pool == the exact fp8 rows each sender pushed),
  * fused output == torch grouped GEMM on the dispatched pool (cosine), and
  * comm-overlap perf (fused vs pure grouped-GEMM on the same pool) on real XGMI.

Run inside dev_primus from the Primus-Turbo repo root (source flydsl + the built
``_C`` on LD_LIBRARY_PATH; ``sys.path`` adds the repo root so ``primus_turbo`` resolves):
  PYTHONPATH=<Primus-Turbo> LD_LIBRARY_PATH=<site-packages> \
    python tests/test_dispatch_grouped_gemm_fp8_ep8.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_fp8 import (  # noqa: E402
    dispatch_grouped_gemm_fp8,
    dispatch_only,
    grouped_gemm_fp8_only,
)
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory  # noqa: E402


class _Comm:
    """Minimal CommTasks stand-in (the kernel reads only these fields)."""

    def __init__(self, dest, start, cnt, srcoff, src_tokens, num_comm):
        self.dest = dest
        self.start = start
        self.cnt = cnt
        self.srcoff = srcoff
        self.src_tokens = src_tokens
        self.num_comm = num_comm


def _fp8(t):
    return t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _align_up(x, a=256):
    return (x + a - 1) // a * a


_FLUSH_BUF = None
_L2_FLUSH_BYTES = 256 * 1024 * 1024   # set from --l2-flush-mb in _run (0 = off)


def _l2_flush():
    """Evict the L2 / last-level cache so each timed iter runs cold (weights + pool
    otherwise stay cached across iters and understate the real HBM/XGMI cost)."""
    global _FLUSH_BUF
    if _L2_FLUSH_BYTES <= 0:
        return
    if _FLUSH_BUF is None:
        _FLUSH_BUF = torch.empty(_L2_FLUSH_BYTES, dtype=torch.int8, device="cuda")
    _FLUSH_BUF.zero_()


def _reset_sync(reset, group):
    # Cross-rank reset MUST be GPU-complete on ALL ranks before any rank runs fn:
    # zero the scoreboard, sync so the zero actually landed, THEN barrier. Otherwise a
    # fast peer's comm signal arrives before this rank's zero_ executes and the zero
    # wipes it -> the GEMM spins on a lost signal forever (hang). (The single-shot
    # correctness path already does sync-before-barrier; the timing loop must too.)
    reset()
    torch.cuda.synchronize()
    group.barrier()


def _timed(fn, n, group, reset=None, log=None):
    # reset != None -> cross-rank fused: re-zero the scoreboard each iter with a
    # sync-before-barrier (see _reset_sync). Local gemm-only (reset=None) has no
    # handshake. An L2 flush before each timed fn makes every iter cold.
    dlog = log or (lambda m: None)
    sync_reset = reset is not None
    for w in range(3):
        if sync_reset:
            _reset_sync(reset, group)
        dlog(f"warmup {w}: reset+barrier done, launching fn")
        _l2_flush()
        fn()
        dlog(f"warmup {w}: fn launched")
    dlog("warmup done; syncing")
    torch.cuda.synchronize(); group.barrier()
    dlog("warmup synced")
    t = 0.0
    for it in range(n):
        if sync_reset:
            _reset_sync(reset, group)
        dlog(f"iter {it}: reset+barrier done")
        _l2_flush()                          # cold-cache each iter (not timed)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        dlog(f"iter {it}: fn done")
        t += time.perf_counter() - t0
        if sync_reset:
            group.barrier()
        dlog(f"iter {it}: post-barrier done")
    return t / n


def _run(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8421"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}",
                            world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()

    def log(m):
        if rank == 0:
            print(f"[stage] {m}", flush=True)
    log("pg ready")

    global _L2_FLUSH_BYTES
    _L2_FLUSH_BYTES = args.l2_flush_mb * 1024 * 1024   # 0 disables the flush

    H, I, E = args.hidden, args.inter, args.num_experts
    N = 2 * I                       # NT fwd L1 out_features
    BM, BN = 256, 256
    per_peer = args.per_peer        # tokens each rank sends to each peer (multiple of BM)
    assert per_peer % BM == 0, "per_peer must be a multiple of BM"
    epr = E // world
    T = world * per_peer            # this rank's source tokens
    pool_rows = world * per_peer    # one per-sender segment of per_peer rows
    n_mblk = pool_rows // BM
    torch.manual_seed(7 + rank)

    # fp8 source tokens + this rank's local-expert weights [epr, N, H]
    x = _fp8(torch.randn((T, H), device="cuda"))
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    W1 = _fp8(torch.randn((epr, N, H), generator=g, device="cuda") * 0.03)
    a_scale = torch.tensor([0.5], device="cuda")
    b_scale = torch.tensor([1.0], device="cuda")

    # deterministic all-to-all comm tasks: one task per peer p ->
    #   src x[p*per_peer : (p+1)*per_peer]  lands in  peer p's pool[r*per_peer : ...]
    # FINE tasks: one per (peer, BM-block) -- like the real dispatch planner. With
    # round-robin (one block per task) this fills all comm CUs; one coarse task per
    # peer would leave most comm CUs idle (and starve the fused GEMM of CUs).
    bpp = per_peer // BM                                               # BM-blocks per peer
    peers = torch.arange(world, device="cuda").repeat_interleave(bpp)  # [world*bpp]
    blks = torch.arange(bpp, device="cuda").repeat(world)
    dest = peers.to(torch.int32)
    start = (rank * per_peer + blks * BM).to(torch.int32)             # my segment, block b
    cnt = torch.full((world * bpp,), BM, dtype=torch.int32, device="cuda")
    srcoff = (peers * per_peer + blks * BM).to(torch.int32)
    src_tokens = torch.arange(T, dtype=torch.int32, device="cuda")     # identity gather
    comm = _Comm(dest, start, cnt, srcoff, src_tokens, world * bpp)

    # tile_to_group: each BM-block -> a local expert; each block filled by ONE sender -> expected=1
    ttg = (torch.arange(n_mblk, dtype=torch.int32, device="cuda") % epr)
    expected = torch.ones(n_mblk, dtype=torch.int32, device="cuda")
    mblk_dev = torch.tensor([n_mblk], dtype=torch.int32, device="cuda")

    # symmetric pool (fp8 bytes) + scoreboard, peer base ptrs for the XGMI push
    off_pool = 0
    off_sb = _align_up(off_pool + pool_rows * H)        # fp8 = 1 byte
    symm = SymmetricMemory(group, alloc_size=_align_up(off_sb + n_mblk * 4))

    def region(peer, off, sizes, dtype):
        return symm.get_buffer(rank=peer, sizes=list(sizes), dtype=dtype,
                               storage_offset=off // dtype.itemsize)

    keep, pool_col, sb_col = [], [], []
    for peer in range(world):
        pv = region(peer, off_pool, (pool_rows * H,), torch.int8)
        sv = region(peer, off_sb, (n_mblk,), torch.int32)
        keep += [pv, sv]
        pool_col.append(pv.data_ptr()); sb_col.append(sv.data_ptr())
    pool_ptrs = torch.tensor(pool_col, dtype=torch.int64, device="cuda")
    sb_ptrs = torch.tensor(sb_col, dtype=torch.int64, device="cuda")
    local_pool = region(rank, off_pool, (pool_rows * H,), torch.int8).view(
        torch.float8_e4m3fn).view(pool_rows, H)
    scoreboard = region(rank, off_sb, (n_mblk,), torch.int32)
    output = torch.empty((pool_rows, N), dtype=torch.bfloat16, device="cuda")

    def _fused():
        dispatch_grouped_gemm_fp8(
            x, comm, pool_ptrs, sb_ptrs, local_pool, W1, output,
            ttg, scoreboard, expected, mblk_dev,
            a_scale=a_scale, b_scale=b_scale, layout="nt", BM=BM, BN=BN,
            comm_blocks=args.comm_blocks)

    log("symm + buffers ready; running correctness _fused()")
    # ---- run once for correctness ----
    local_pool.view(torch.int8).zero_(); scoreboard.zero_()
    torch.cuda.synchronize(); symm.barrier()
    _fused()
    torch.cuda.synchronize(); symm.barrier()
    log("correctness _fused() done; all_gather")

    # (1) dispatch content: peer pool segment s == sender s's rows destined for me
    x_all = [torch.empty_like(x.view(torch.int8)) for _ in range(world)]
    dist.all_gather(x_all, x.view(torch.int8), group=group)
    disp_ok = True
    for s in range(world):
        sent = x_all[s].view(torch.float8_e4m3fn).view(T, H)[rank * per_peer:(rank + 1) * per_peer]
        got = local_pool[s * per_peer:(s + 1) * per_peer]
        if not torch.equal(got.view(torch.int8), sent.view(torch.int8)):
            disp_ok = False

    # (2) GEMM output == torch grouped GEMM on the dispatched pool
    sa, sb = float(a_scale.item()), float(b_scale.item())
    ca = 0.0
    for blk in range(n_mblk):
        gexp = int(ttg[blk].item())
        rows = slice(blk * BM, (blk + 1) * BM)
        ref = (local_pool[rows].float() * sa) @ (W1[gexp].float() * sb).t()
        ca += _cos(output[rows], ref)
    cos = ca / n_mblk
    log(f"correctness done (cos={cos:.4f}); timing gemm-only")

    # ---- perf: fused (dispatch+GEMM over XGMI) vs pure grouped GEMM on the pool ----
    o2 = torch.empty((pool_rows, N), dtype=torch.bfloat16, device="cuda")
    grouped_gemm_fp8_only(local_pool, W1, o2, ttg, mblk_dev,
                          a_scale=a_scale, b_scale=b_scale, layout="nt", BM=BM, BN=BN)
    torch.cuda.synchronize()
    t_gg = _timed(lambda: grouped_gemm_fp8_only(local_pool, W1, o2, ttg, mblk_dev,
                                                a_scale=a_scale, b_scale=b_scale,
                                                layout="nt", BM=BM, BN=BN),
                  args.iters, group)
    log(f"gemm-only timed ({t_gg*1e6:.0f}us); timing fused")
    t_fused = _timed(_fused, args.iters, group, reset=scoreboard.zero_)
    log(f"fused timed ({t_fused*1e6:.0f}us); timing dispatch-only")
    flops = 2.0 * pool_rows * N * H

    # ---- dispatch-only bandwidth: push T tokens (T*H fp8 bytes) to peer pools ----
    # reset=lambda:None enables _timed's per-iter barrier so all ranks push together
    # (realistic XGMI contention); push_bytes excludes the self-segment (local HBM).
    dispatch_only(x, comm, pool_ptrs, local_pool, comm_blocks=args.comm_blocks)
    torch.cuda.synchronize()
    t_disp = _timed(lambda: dispatch_only(x, comm, pool_ptrs, local_pool, comm_blocks=args.comm_blocks),
                    args.iters, group, reset=lambda: None)
    log(f"dispatch-only timed ({t_disp*1e6:.0f}us); gathering results")
    push_bytes = T * H                              # fp8 = 1 byte; tokens this rank pushes
    xgmi_bytes = push_bytes * (world - 1) / world   # cross-rank portion (self-seg is local)

    rec = (rank, pool_rows, disp_ok, cos, t_gg, t_fused, flops, t_disp, push_bytes, xgmi_bytes)
    allg = [None] * world
    dist.all_gather_object(allg, rec, group=group)
    if rank == 0:
        print(f"\n==== EP{world} fp8 dispatch+grouped-GEMM NT over XGMI ====")
        print(f"  H={H} N(2I)={N} E={E} epr={epr} per_peer={per_peer} "
              f"T={T} pool_rows={pool_rows} BM={BM} "
              f"l2_flush={'off' if args.l2_flush_mb <= 0 else str(args.l2_flush_mb) + 'MB'}")
        for r, m, ok, c, tg, tf, fl, td, pb, xb in allg:
            ggtf = fl / tg / 1e12 if tg > 0 else 0
            futf = fl / tf / 1e12 if tf > 0 else 0
            print(f"  rank{r}: pool={m:6d} disp_ok={ok} cos={c:.6f} | "
                  f"gemm-only {tg*1e6:7.0f}us {ggtf:5.0f}TF | "
                  f"fused {tf*1e6:7.0f}us {futf:5.0f}TF | overlap {tf/tg:.2f}x")
        print("  -- dispatch-only (push) bandwidth --")
        for r, m, ok, c, tg, tf, fl, td, pb, xb in allg:
            push_gbps = pb / td / 1e9 if td > 0 else 0
            xgmi_gbps = xb / td / 1e9 if td > 0 else 0
            print(f"  rank{r}: {td*1e6:7.0f}us | push {pb/1e6:6.0f}MB {push_gbps:6.0f} GB/s "
                  f"| XGMI {xgmi_gbps:6.0f} GB/s")
        agg = sum(p[8] for p in allg) / max(p[7] for p in allg) / 1e9  # total push / slowest time
        print(f"  aggregate push {sum(p[8] for p in allg)/1e6:.0f}MB -> {agg:.0f} GB/s (all {world} ranks)")
        okall = all(p[2] for p in allg)
        cosmin = min(p[3] for p in allg)
        print(f"  [gate] all disp_ok={okall}  min cos={cosmin:.6f}  -> "
              f"{'PASS' if okall and cosmin > 0.99 else 'FAIL'}")

    torch.cuda.synchronize(); group.barrier()
    symm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=8)   # EP8
    ap.add_argument("--hidden", type=int, default=7168)       # K = H
    ap.add_argument("--inter", type=int, default=2048)        # N = 2I
    ap.add_argument("--num-experts", type=int, default=256)
    # default kept modest: per_peer=8192 -> pool 65536 rows -> ~4096 spinning GEMM
    # blocks >> CUs, which can hit a producer-consumer forward-progress deadlock
    # (the comm/producer blocks get starved of CU slots). 2048 -> grid ~1024, safe.
    ap.add_argument("--per-peer", type=int, default=8192)      # tokens/rank/peer (multiple of BM)
    ap.add_argument("--comm-blocks", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--l2-flush-mb", type=int, default=256,
                    help="cache-flush buffer zeroed before each timed iter (0 = off)")
    args = ap.parse_args()
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)
