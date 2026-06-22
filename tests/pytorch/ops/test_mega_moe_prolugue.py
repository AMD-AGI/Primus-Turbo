###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-process (world=1) self-loopback correctness + perf test for the fused
MoE dispatch-prologue FlyDSL kernel (``primus_turbo.flydsl.mega.mega_moe_prologue``).

Driven through the ``MegaMoePrologue`` workspace facade (allocate once, run per
step). world=1: every cross-rank barrier / all-gather / origin push targets this
rank's own symmetric buffers, so the whole persistent prologue runs in one
process with no IPC. We validate the deterministic dispatch tables exactly
against a torch reference, validate the (atomic-ordered) scatter outputs, and
report latency + an approximate effective bandwidth.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python -m pytest -s \
      tests/pytorch/ops/test_mega_moe_prolugue.py
or standalone:
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_moe_prolugue.py
"""

import argparse

import pytest
import torch

try:
    from primus_turbo.flydsl.mega.mega_moe_prologue import MegaMoePrologue

    _HAVE_FLYDSL = True
except Exception as _e:  # flydsl only present inside the dev container
    _HAVE_FLYDSL = False
    _IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Torch reference (world=1, rank=0)
# ---------------------------------------------------------------------------
def _reference(topk_idx, topk_w, num_tokens, num_topk, num_experts, block_m):
    """Deterministic dispatch tables + flat valid (expert, token, slot) arrays."""
    idx = topk_idx.to(torch.int64).cpu()  # [T, K]
    twf = topk_w.to(torch.float32).cpu()  # [T, K]
    T, K = num_tokens, num_topk
    flat = idx.reshape(-1)
    tok = torch.arange(T).view(T, 1).expand(T, K).reshape(-1)
    slot = torch.arange(K).view(1, K).expand(T, K).reshape(-1)
    valid = flat >= 0  # drop padding (-1) pairs
    e_v, tok_v, slot_v = flat[valid], tok[valid], slot[valid]

    counts = torch.bincount(e_v, minlength=num_experts).to(torch.int64)
    padded = ((counts + block_m - 1) // block_m) * block_m
    pool_base = torch.zeros(num_experts, dtype=torch.int64)
    pool_base[1:] = torch.cumsum(padded, 0)[:-1]
    start_per_expert = pool_base.clone()  # rank=0 => no preceding ranks
    source_off = torch.zeros(num_experts, dtype=torch.int64)
    source_off[1:] = torch.cumsum(counts, 0)[:-1]  # k-order == expert order for world=1
    total_rows = int(pool_base[-1] + padded[-1])

    return dict(
        idx=idx,
        twf=twf,
        counts=counts,
        padded=padded,
        pool_base=pool_base,
        start_per_expert=start_per_expert,
        source_off=source_off,
        total_rows=total_rows,
        e_v=e_v,
        tok_v=tok_v,
        slot_v=slot_v,
    )


def _check_correctness(wp, ref, num_tokens, num_experts, block_m, pool_capacity):
    """Assert every deterministic table matches; validate the scatter outputs."""
    g = lambda name: wp.buffers[name].cpu().to(torch.int64)
    msgs = []

    def eq(name, got, want):
        ok = torch.equal(got, want)
        msgs.append(f"  {name:24s} {'OK' if ok else 'MISMATCH'}")
        assert ok, f"{name} mismatch\n got ={got.tolist()}\n want={want.tolist()}"

    # reset buffers
    eq("send_local(reset=0)", g("send_local"), torch.zeros(num_experts, dtype=torch.int64))
    eq("within_counter(reset=0)", g("within_expert_counter"), torch.zeros(num_experts, dtype=torch.int64))
    # deterministic dispatch tables
    eq("count", g("count"), ref["counts"])
    eq("pool_base", g("pool_base"), ref["pool_base"])
    eq("start_per_expert", g("start_per_expert"), ref["start_per_expert"])
    eq("start", g("start"), ref["start_per_expert"])
    eq("destination(all 0)", g("destination"), torch.zeros(num_experts, dtype=torch.int64))
    eq("source_offset_out", g("source_offset_out"), ref["source_off"])
    eq("source_offset_per_expert", g("source_offset_per_expert"), ref["source_off"])

    # tile_to_group: expert id over occupied blocks, sentinel num_experts elsewhere.
    # expected: per pool block, the number of source ranks contributing rows.
    n_mblk = pool_capacity // block_m
    ttg_want = torch.full((n_mblk,), num_experts, dtype=torch.int64)
    expected_want = torch.zeros(n_mblk, dtype=torch.int64)
    for e in range(num_experts):  # E is small (32); a per-expert loop is fine here
        b0 = int(ref["pool_base"][e]) // block_m
        nb = int(ref["padded"][e]) // block_m
        ttg_want[b0 : b0 + nb] = e
        c = int(ref["counts"][e])
        if c > 0:
            lb = (int(ref["pool_base"][e]) + c - 1) // block_m
            expected_want[b0 : lb + 1] += 1
    eq("tile_to_group", g("tile_to_group"), ttg_want)
    eq("expected", g("expected"), expected_want)

    # meta_scalars[0]=total_rows, [1]=n_mblk_rows, [2]=num_experts
    meta = g("meta_scalars")
    eq(
        "meta_scalars[:3]",
        meta[:3],
        torch.tensor([ref["total_rows"], ref["total_rows"] // block_m, num_experts], dtype=torch.int64),
    )

    # ---- scatter: source region holds one (expert, token, slot) per valid pair ----
    # source slots are packed densely in expert order; expert of slot s = pos_expert[s]
    used = int(ref["counts"].sum())
    pos_expert = torch.repeat_interleave(torch.arange(num_experts), ref["counts"])  # [used]
    src_tok = g("source_tokens")[:used]
    src_slot = g("source_topk_slot")[:used]
    src_w = wp.buffers["source_weight"].cpu()[:used]
    # within-expert order is atomic-nondeterministic -> compare the triple SET
    got = set(zip(pos_expert.tolist(), src_tok.tolist(), src_slot.tolist()))
    want = set(zip(ref["e_v"].tolist(), ref["tok_v"].tolist(), ref["slot_v"].tolist()))
    assert len(want) == used, "reference has duplicate (expert, token, slot) pairs"
    assert got == want, "scatter (expert, token, slot) set mismatch"
    # routing weight matches the (token, slot) each slot was scattered with
    assert torch.allclose(src_w, ref["twf"][src_tok, src_slot], atol=1e-6), "source_weight mismatch"
    msgs.append(f"  {'scatter(token/slot/weight)':24s} OK")

    # ---- origin_rank / origin_slot: exact per-row mapping (same within-pos as source) ----
    within = torch.arange(used) - ref["source_off"][pos_expert]  # within-expert position
    occ_rows = ref["start_per_expert"][pos_expert] + within  # pool row of each source slot
    orank = g("origin_rank")
    oslot = g("origin_slot")
    assert torch.equal(orank[occ_rows], torch.zeros(used, dtype=torch.int64)), "origin_rank occupied != 0"
    assert torch.equal(oslot[occ_rows], src_slot * num_tokens + src_tok), "origin_slot mismatch"
    pad_mask = torch.ones(pool_capacity, dtype=torch.bool)
    pad_mask[occ_rows] = False
    assert torch.equal(
        orank[pad_mask], torch.full((int(pad_mask.sum()),), -1, dtype=torch.int64)
    ), "origin_rank padding != -1"
    msgs.append(f"  {'origin_rank/slot(per-row)':24s} OK")

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


def _approx_bytes(num_tokens, num_topk, num_experts, block_m, pool_capacity):
    """Approximate global traffic moved by one prologue launch."""
    total_pairs = num_tokens * num_topk
    n_mblk = pool_capacity // block_m
    init = pool_capacity * 4 + 2 * n_mblk * 4  # origin_rank + expected + tile_to_group
    phase_a = total_pairs * 4  # topk read (low word)
    phase_d_read = total_pairs * 4 + total_pairs * 4  # topk re-read + topk_w
    phase_d_write = total_pairs * (4 + 4 + 4 + 4 + 4)  # src_tok/slot/w + origin_rank/slot
    return init + phase_a + phase_d_read + phase_d_write


# ---------------------------------------------------------------------------
# Test entry
# ---------------------------------------------------------------------------
def _make_routing(num_tokens, num_topk, num_experts, seed=7, drop_frac=0.0):
    assert num_topk <= num_experts, "top-k cannot exceed the expert count"
    torch.manual_seed(seed)
    # distinct experts per token = prefix of a per-row random permutation (vectorized)
    idx = torch.rand(num_tokens, num_experts, device="cuda").argsort(dim=1)[:, :num_topk].contiguous()
    if drop_frac > 0:  # mark some pairs as padding (-1)
        drop = torch.rand(num_tokens, num_topk, device="cuda") < drop_frac
        idx[drop] = -1
    w = torch.rand(num_tokens, num_topk, dtype=torch.float32, device="cuda")
    return idx, w


def _run(num_tokens, num_topk, num_experts, block_m, drop_frac, iters):
    total_pairs = num_tokens * num_topk
    # generous pool: every pair fits even fully imbalanced, + per-expert pad
    pool_capacity = total_pairs + num_experts * block_m
    pool_capacity = ((pool_capacity + block_m - 1) // block_m) * block_m

    topk_idx, topk_w = _make_routing(num_tokens, num_topk, num_experts, drop_frac=drop_frac)
    # world=1 workspace: one factory call owns the whole buffer-shape contract
    wp = MegaMoePrologue.allocate(
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        pool_capacity=pool_capacity,
        world_size=1,
        rank=0,
        block_m=block_m,
    )

    print(
        f"\n[cfg] T={num_tokens} K={num_topk} E={num_experts} BM={block_m} "
        f"pool_cap={pool_capacity} pairs={total_pairs} drop={drop_frac}"
    )

    # ---- correctness ----
    result = wp.run(topk_idx, topk_w)
    torch.cuda.synchronize()
    assert result.num_pool_blocks == pool_capacity // block_m
    assert result.comm_tasks.num_comm == num_experts
    ref = _reference(topk_idx, topk_w, num_tokens, num_topk, num_experts, block_m)
    _check_correctness(wp, ref, num_tokens, num_experts, block_m, pool_capacity)

    # ---- perf: latency + effective bandwidth (workspace reuses its launch cache) ----
    def _fn():
        wp.run(topk_idx, topk_w)

    t_us = _bench(_fn, iters=iters)
    bytes_moved = _approx_bytes(num_tokens, num_topk, num_experts, block_m, pool_capacity)
    bw = bytes_moved / (t_us * 1e-6) / 1e9
    # the prologue is metadata-heavy / latency-bound (serial block-0 table build +
    # grid/cross-rank barriers), so eff BW is informational, not a roofline target
    print(
        f"[perf] latency = {t_us:8.2f} us/launch  |  "
        f"approx traffic = {bytes_moved/1e6:6.2f} MB  |  eff BW = {bw:6.1f} GB/s (latency-bound)"
    )
    return t_us, bw


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.parametrize("num_topk", [6, 8])
@pytest.mark.parametrize("drop_frac", [0.0, 0.1])
def test_mega_moe_prologue(num_topk, drop_frac):
    torch.cuda.set_device(0)
    _run(num_tokens=4096, num_topk=num_topk, num_experts=32, block_m=256, drop_frac=drop_frac, iters=50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=4096)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--block-m", type=int, default=256)
    ap.add_argument("--drop-frac", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    torch.cuda.set_device(0)
    _run(args.num_tokens, args.num_topk, args.num_experts, args.block_m, args.drop_frac, args.iters)
    print("[mega_moe_prologue] PASS")


if __name__ == "__main__":
    main()
