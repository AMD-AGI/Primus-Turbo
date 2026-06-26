###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 cross-rank correctness + perf test for the fused MoE dispatch-prologue
FlyDSL kernel (``primus_turbo.flydsl.mega.dispatch_prologue_kernel``).

Each of the ``world`` ranks routes its own ``[T, K]`` topk, then the prologue
builds the EP dispatch plan over a single symmetric allocation (``SymmBuffer``
from ``mega_moe_fused``, reused here): the cross-rank counts all-gather, pool
layout, comm tasks, and the cross-rank ``origin_rank`` / ``origin_slot`` push all
exercise real inter-rank traffic. We validate every deterministic table against a
torch reference built from the all-gathered routing, validate the local scatter,
and check the cross-rank origin push by source-rank counts.

Run inside dev_primus (>=2 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python \
      tests/pytorch/ops/test_mega_moe_prolugue.py --num-processes 8
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch
import torch.distributed as dist

try:
    from primus_turbo.flydsl.mega.dispatch_prologue_kernel import (
        dispatch_prologue,
        get_dispatch_prologue_workspace,
    )
    from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

    _HAVE_FLYDSL = True
except Exception as _e:  # flydsl only present inside the dev container
    _HAVE_FLYDSL = False
    _IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def _make_routing(num_tokens, num_topk, num_experts, seed, drop_frac=0.0):
    assert num_topk <= num_experts, "top-k cannot exceed the expert count"
    g = torch.Generator(device="cuda").manual_seed(seed)
    # distinct experts per token = prefix of a per-row random permutation
    idx = torch.rand(num_tokens, num_experts, generator=g, device="cuda").argsort(dim=1)[:, :num_topk]
    idx = idx.contiguous().to(torch.int64)
    if drop_frac > 0:  # mark some pairs as padding (-1)
        drop = torch.rand(num_tokens, num_topk, generator=g, device="cuda") < drop_frac
        idx[drop] = -1
    w = torch.rand(num_tokens, num_topk, dtype=torch.float32, generator=g, device="cuda")
    return idx, w


# ---------------------------------------------------------------------------
# EP reference: mirror the kernel's Phase-C table build for one rank, from the
# all-gathered per-rank routing. C[s, e] = #pairs from source rank s to expert e.
# ---------------------------------------------------------------------------
def _ep_reference(all_idx, rank, num_topk, num_experts, world, block_m, pool_capacity):
    epr = num_experts // world
    all_idx[0].shape[0]
    bm = block_m

    counts_src = torch.zeros(world, num_experts, dtype=torch.int64)  # C[s, e]
    for s in range(world):
        flat = all_idx[s].reshape(-1)
        counts_src[s] = torch.bincount(flat[flat >= 0], minlength=num_experts).to(torch.int64)
    recv = counts_src.sum(0)  # total received per expert (all sources)

    def ceil_bm(x):
        return ((x + bm - 1) // bm) * bm

    # pool_base[e]: local offset of expert e within its destination rank's pool
    pool_base = torch.zeros(num_experts, dtype=torch.int64)
    for dest in range(world):
        running = 0
        for le in range(epr):
            e = dest * epr + le
            pool_base[e] = running
            running += int(ceil_bm(recv[e]))
    # start_per_expert[e]: this rank's slice start = pool_base + rows from preceding ranks
    preceding = counts_src[:rank].sum(0) if rank > 0 else torch.zeros(num_experts, dtype=torch.int64)
    start_per_expert = pool_base + preceding

    # comm tasks: ct index has dest = ct % world, le = ct // world
    destination = torch.zeros(num_experts, dtype=torch.int64)
    start = torch.zeros(num_experts, dtype=torch.int64)
    count = torch.zeros(num_experts, dtype=torch.int64)
    for ct in range(num_experts):
        dest, le = ct % world, ct // world
        e = dest * epr + le
        destination[ct] = dest
        start[ct] = start_per_expert[e]
        count[ct] = counts_src[rank, e]
    # source offsets: exclusive cumsum of this rank's counts in le-major / dest-minor order
    source_off_out = torch.zeros(num_experts, dtype=torch.int64)  # by comm-task index
    source_off_pe = torch.zeros(num_experts, dtype=torch.int64)  # by expert id
    acc = 0
    for le in range(epr):
        for dest in range(world):
            ct = le * world + dest
            e = dest * epr + le
            source_off_out[ct] = acc
            source_off_pe[e] = acc
            acc += int(count[ct])

    # tile_to_expert (local expert id, sentinel = epr) + expected (sources per block) for own experts
    n_mblk = pool_capacity // bm
    ttg = torch.full((n_mblk,), epr, dtype=torch.int64)
    expected = torch.zeros(n_mblk, dtype=torch.int64)
    total_rows = 0
    for le in range(epr):
        e = rank * epr + le
        epb = int(pool_base[e])
        padded = int(ceil_bm(recv[e]))
        b0, nb = epb // bm, padded // bm
        ttg[b0 : b0 + nb] = le
        within = 0
        for s in range(world):
            c = int(counts_src[s, e])
            if c > 0:
                fb = (epb + within) // bm
                lb = (epb + within + c - 1) // bm
                expected[fb : lb + 1] += 1
                within += c
        if le == epr - 1:
            total_rows = epb + padded

    return dict(
        counts_src=counts_src,
        recv=recv,
        pool_base=pool_base,
        start_per_expert=start_per_expert,
        destination=destination,
        start=start,
        count=count,
        source_off_out=source_off_out,
        source_off_pe=source_off_pe,
        ttg=ttg,
        expected=expected,
        total_rows=total_rows,
    )


def _check_tables(
    symm,
    ref,
    plan,
    ret_ttg,
    ret_expected,
    topk_idx,
    topk_w,
    num_topk,
    num_experts,
    world,
    rank,
    block_m,
    pool_capacity,
):
    epr = num_experts // world
    # the 5 per-expert scratch tables live packed in the prologue's cached internal
    # workspace ([5 * E]); the plan output tables come from the prologue's RETURN
    # (allocated internally); origin_rank/origin_slot/meta_scalars stay symmetric
    ws = get_dispatch_prologue_workspace(num_experts, device=topk_idx.device).cpu().to(torch.int64)
    E = num_experts
    ws_views = {
        "send_local": ws[0:E],
        "within_expert_counter": ws[E : 2 * E],
        "start_per_expert": ws[2 * E : 3 * E],
        "source_offset_per_expert": ws[3 * E : 4 * E],
        "pool_base": ws[4 * E : 5 * E],
    }
    # plan = (send_task_table[E,4], src_token_table[pool,2], src_token_weight[pool])
    send_task_table, src_token_table = plan[0], plan[1]
    outs = {
        "destination": send_task_table[:, 0],  # dst_rank
        "start": send_task_table[:, 1],  # dst_offset
        "count": send_task_table[:, 2],  # count
        "source_offset_out": send_task_table[:, 3],  # src_offset
        "source_tokens": src_token_table[:, 0],  # token id
        "source_topk_slot": src_token_table[:, 1],  # top-k slot
        "tile_to_expert": ret_ttg,
        "expected": ret_expected,
    }
    g = lambda name: (
        ws_views[name]
        if name in ws_views
        else outs[name].cpu().to(torch.int64) if name in outs else getattr(symm, name).cpu().to(torch.int64)
    )
    msgs = []

    def eq(name, got, want):
        got, want = got.cpu(), want.cpu()
        ok = torch.equal(got, want)
        msgs.append(f"  {name:26s} {'OK' if ok else 'MISMATCH'}")
        assert ok, f"{name} mismatch\n got ={got.tolist()[:32]}\n want={want.tolist()[:32]}"

    # counters self-reset to 0 after the launch
    eq("send_local(reset=0)", g("send_local"), torch.zeros(num_experts, dtype=torch.int64))
    eq("within_counter(reset=0)", g("within_expert_counter"), torch.zeros(num_experts, dtype=torch.int64))
    # deterministic dispatch tables
    eq("count", g("count"), ref["count"])
    eq("destination", g("destination"), ref["destination"])
    eq("pool_base", g("pool_base"), ref["pool_base"])
    eq("start_per_expert", g("start_per_expert"), ref["start_per_expert"])
    eq("start", g("start"), ref["start"])
    eq("source_offset_out", g("source_offset_out"), ref["source_off_out"])
    eq("source_offset_per_expert", g("source_offset_per_expert"), ref["source_off_pe"])
    eq("tile_to_expert", g("tile_to_expert"), ref["ttg"])
    eq("expected", g("expected"), ref["expected"])
    meta = g("meta_scalars")
    eq(
        "meta_scalars[:3]",
        meta[:3],
        torch.tensor([ref["total_rows"], ref["total_rows"] // block_m, num_experts], dtype=torch.int64),
    )

    # ---- local scatter: source region holds this rank's valid (expert, token, slot) pairs ----
    idx = topk_idx.cpu()
    T, K = idx.shape
    flat = idx.reshape(-1)
    tok = torch.arange(T).view(T, 1).expand(T, K).reshape(-1)
    slot = torch.arange(K).view(1, K).expand(T, K).reshape(-1)
    valid = flat >= 0
    e_v, tok_v, slot_v = flat[valid], tok[valid], slot[valid]
    counts_r = ref["counts_src"][rank]
    used = int(counts_r.sum())
    # source region is ordered by source_offset_per_expert: le-major, dest-minor
    order = torch.tensor([dest * epr + le for le in range(epr) for dest in range(world)])
    pos_expert = torch.repeat_interleave(order, counts_r[order])
    src_tok = g("source_tokens")[:used]
    src_slot = g("source_topk_slot")[:used]
    src_w = plan[2].cpu()[:used]  # src_token_weight (f32) from the returned plan handle
    got = set(zip(pos_expert.tolist(), src_tok.tolist(), src_slot.tolist()))
    want = set(zip(e_v.tolist(), tok_v.tolist(), slot_v.tolist()))
    assert len(want) == used, "reference has duplicate (expert, token, slot) pairs"
    assert got == want, "scatter (expert, token, slot) set mismatch"
    assert torch.allclose(src_w, topk_w.cpu()[src_tok, src_slot], atol=1e-6), "source_weight mismatch"
    msgs.append(f"  {'scatter(token/slot/weight)':26s} OK")

    # ---- cross-rank origin push: this rank received `recv` rows; verify per-source counts ----
    recv_per_source = ref["counts_src"][:, rank * epr : (rank + 1) * epr].sum(1)  # [world]
    orank = g("origin_rank")
    for s in range(world):
        got_s = int((orank == s).sum())
        assert got_s == int(
            recv_per_source[s]
        ), f"origin_rank from src {s}: {got_s} != {int(recv_per_source[s])}"
    occupied = int((orank >= 0).sum())
    assert occupied == int(recv_per_source.sum()), "origin_rank occupied count mismatch"
    oslot = g("origin_slot")
    occ = orank >= 0
    assert (
        int(((oslot[occ] >= 0) & (oslot[occ] < num_topk * symm.num_tokens)).sum()) == occupied
    ), "origin_slot range"
    msgs.append(f"  {'origin_rank/slot(cross-rank)':26s} OK")

    # ---- token dedup map1 (source side): exactly one primary per (token, dst_rank) ----
    # source_dedup[src_slot]: 1 = secondary (skips XGMI push), 0 = primary (pushes).
    (e_v // epr)
    sdedup = plan[3].cpu().to(torch.int64)[:used]
    # src slots are ordered le-major dest-minor (same `pos_expert` order as the scatter);
    # rebuild each slot's (token, dst_rank) and require exactly one primary per group.
    src_dst_rank = (pos_expert // epr).cpu()
    keys = src_tok * world + src_dst_rank  # (token, dst_rank) group key
    n_src_primaries = int(torch.unique(keys).numel())
    assert (
        int((sdedup == 0).sum()) == n_src_primaries
    ), f"source primaries {int((sdedup == 0).sum())} != unique (token, dst_rank) {n_src_primaries}"
    # per-group exactly one primary (sum of primaries per key == 1)
    order_k = torch.argsort(keys)
    ks, ps = keys[order_k], (sdedup[order_k] == 0).to(torch.int64)
    grp_primary = torch.zeros(n_src_primaries, dtype=torch.int64, device="cpu").index_add_(
        0, torch.unique(ks, return_inverse=True)[1], ps
    )
    assert torch.equal(
        grp_primary, torch.ones_like(grp_primary)
    ), "a (token,dst_rank) group lacks exactly 1 primary"
    msgs.append(f"  {'dedup source_dedup(map1)':26s} OK")

    # ---- token dedup map2 (dest side): dedup_src_row links secondaries to their primary ----
    dsr = g("dedup_src_row")
    occ_rows = torch.nonzero(occ, as_tuple=True)[0]
    tok_of = oslot // num_topk  # source token per occupied dest row
    for r in occ_rows.tolist():
        p = int(dsr[r])
        if p < 0:
            continue  # primary
        assert bool(occ[p]), f"dedup_src_row[{r}]={p} points at an empty row"
        assert int(dsr[p]) == -1, f"secondary {r} -> {p} but {p} is itself a secondary"
        assert int(orank[r]) == int(orank[p]) and int(tok_of[r]) == int(
            tok_of[p]
        ), f"secondary {r} -> primary {p} mismatch (origin rank/token)"
    # each (origin_rank, token) group received on this rank has exactly one primary
    grp = orank[occ] * (num_topk * symm.num_tokens) + tok_of[occ]
    inv = torch.unique(grp, return_inverse=True)[1]
    prim = (dsr[occ] == -1).to(torch.int64)
    per_grp = torch.zeros(int(inv.max()) + 1, dtype=torch.int64, device="cpu").index_add_(0, inv, prim)
    assert torch.equal(
        per_grp, torch.ones_like(per_grp)
    ), "an (origin_rank,token) group lacks exactly 1 primary"
    msgs.append(f"  {'dedup dedup_src_row(map2)':26s} OK")

    print("\n".join(msgs))


# ---------------------------------------------------------------------------
# Perf
# ---------------------------------------------------------------------------
def _bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / iters  # us / launch


# ---------------------------------------------------------------------------
# Per-process entry
# ---------------------------------------------------------------------------
def _prologue_kwargs(symm):
    """Kwargs for dispatch_prologue: a single SymLayout struct names every symmetric
    sub-buffer; the plan output tables are allocated + returned by the primitive."""
    return dict(
        sym_layout=symm.make_sym_layout(),
        num_tokens=symm.num_tokens,
        num_topk=symm.num_topk,
        num_experts=symm.num_experts,
        world_size=symm.world,
        rank=symm.rank,
        experts_per_rank=symm.num_experts // symm.world,
        block_m=symm.block_m,
        pool_capacity=symm.pool_capacity,
    )


def _run(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8411"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()

    T, K, E, BM = args.num_tokens, args.num_topk, args.num_experts, args.block_m
    assert E % world == 0, "num_experts must be divisible by world_size"

    # one symmetric allocation owns every prologue buffer (small hidden: prologue
    # ignores pool/act/comb, sized only to keep the shared SymmBuffer allocation cheap)
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=256,
        intermediate_hidden=256,
        block_m=BM,
    )

    topk_idx, topk_w = _make_routing(T, K, E, seed=100 + rank, drop_frac=args.drop_frac)
    if getattr(args, "idx_dtype", "int64") == "int32":
        topk_idx = topk_idx.to(torch.int32)
    kwargs = _prologue_kwargs(symm)

    if rank == 0:
        print(
            f"\n[cfg] world={world} T={T} K={K} E={E} BM={BM} "
            f"pool_cap={symm.pool_capacity} drop={args.drop_frac}"
        )

    # ---- correctness ----
    symm.group.barrier()
    plan, tile_to_expert, tile_expected, _, _, num_pool_blocks, _ = dispatch_prologue(
        topk_idx, topk_w, **kwargs
    )
    torch.cuda.synchronize()
    symm.group.barrier()
    assert num_pool_blocks == symm.pool_capacity // BM
    assert plan[0].shape == (E, 4)  # send_task_table: one row per comm task (== num_experts)

    all_idx = [torch.empty_like(topk_idx) for _ in range(world)]
    dist.all_gather(all_idx, topk_idx, group=group)
    all_idx = [t.cpu() for t in all_idx]
    ref = _ep_reference(all_idx, rank, K, E, world, BM, symm.pool_capacity)
    _check_tables(
        symm,
        ref,
        plan,
        tile_to_expert,
        tile_expected,
        topk_idx,
        topk_w,
        K,
        E,
        world,
        rank,
        BM,
        symm.pool_capacity,
    )
    if rank == 0:
        print("[dispatch_prologue] correctness PASS (all ranks)")

    # ---- perf: latency on rank 0 (cross-rank barriers synchronize the launches) ----
    if args.iters > 0:

        def _fn():
            dispatch_prologue(topk_idx, topk_w, **kwargs)

        symm.group.barrier()
        t_us = _bench(_fn, iters=args.iters)
        if rank == 0:
            print(f"[perf] latency = {t_us:8.2f} us/launch (EP{world}, latency-bound)")
    dist.barrier(group)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
_WORLD = 8


def _build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=_WORLD)
    ap.add_argument("--num-tokens", type=int, default=4096)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--block-m", type=int, default=256)
    ap.add_argument("--drop-frac", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--idx-dtype", choices=["int32", "int64"], default="int64")
    return ap


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs for EP{_WORLD}")
@pytest.mark.parametrize("drop_frac", [0.0, 0.1])
def test_mega_moe_prologue(drop_frac):
    args = _build_argparser().parse_args(
        ["--num-tokens", "4096", "--num-topk", "8", "--num-experts", "32", "--drop-frac", str(drop_frac)]
    )
    torch.multiprocessing.spawn(_run, args=(_WORLD, args), nprocs=_WORLD)


def main():
    args = _build_argparser().parse_args()
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)
    print("[dispatch_prologue] PASS")


if __name__ == "__main__":
    main()
