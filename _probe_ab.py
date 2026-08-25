#!/usr/bin/env python3
"""Paired A/B of one backward knob, the two arms INTERLEAVED inside a single process.

The usual instrument is one process per arm and a min of 40 (see _probe_bwdcfg.py), and it
needs the card to itself: with another tenant computing on the same GPU a whole process
reads 18-34% high with a 13% spread BETWEEN processes, so the two arms land in different
parts of that drift and a 3% effect is unreadable. Here the arms alternate iteration by
iteration on the same allocator state, the same clocks and the same neighbour, so the
neighbour is common mode and the PAIRED difference is what survives. Read
``paired_BmA_mean`` together with ``A_wins_pairs`` (out of ``pairs``); a real effect shows
up in both, drift shows up in neither.

Host-plan knobs alternate by swapping a module attribute. BUILD-TIME kwargs alternate by
swapping the whole launcher cache: both sets are built once up front, then a dict swap picks
one, so no JIT ever runs inside the timed loop.

usage: _probe_ab.py <B> <Hq> <Hkv> <S> <D> <knob> [pairs] [W]
knobs:
  null      NOISE CONTROL: both arms are the deployed path, so the reported difference and
            win count ARE this session's paired noise channel. Run it beside every verdict.
  tailcut   the last chunk's batch cut (see _tail_bat_cut) against one whole-batch dispatch
  fills     the band-scaled pipeline gate (see _dq_pipe_fills) against the flat one
  hbr       PRICING ONLY: the fold READS half the bands
  halfband  PRICING ONLY: the whole dQ partial stream at HALF its bytes, both ends -- the
            upper bound on any scheme that gives one work-group two kv bands (64 -> 32)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as bwd
from _probe_bwdcfg import _bat_cut
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

DEV, DT = "cuda", torch.bfloat16
HALF = {"on": False}
CACHE = {}


def _read_half_hook():
    """Route the fold through a doubled band unit while HALF is on (dQ wrong, pricing only)."""
    red = bwd._reduce_dq_partials
    bwd._reduce_dq_partials = lambda ws, dq, bkv, *a, **k: red(
        ws, dq, bkv * 2 if HALF["on"] else bkv, *a, **k
    )


def _build_sets(over, run):
    """Populate CACHE['A'/'B'] with a whole launcher set each; ``over`` is arm B's kwargs."""
    build = bwd.build_flash_attn_bwd_dkdv_module
    run()
    CACHE["A"] = dict(bwd._BWD_CACHE)
    bwd._BWD_CACHE.clear()
    bwd.build_flash_attn_bwd_dkdv_module = lambda **kw: build(**{**kw, **over})
    run()
    CACHE["B"] = dict(bwd._BWD_CACHE)
    bwd.build_flash_attn_bwd_dkdv_module = build

    def enter(tag):
        HALF["on"] = tag == "B"
        bwd._BWD_CACHE.clear()
        bwd._BWD_CACHE.update(CACHE[tag])

    return enter


def arm_switch(name, run):
    """(enter, exact) -- enter(tag) selects the arm; exact = both arms must agree bitwise."""
    if name == "null":
        return (lambda tag: None), True
    if name == "tailcut":
        deployed = bwd._tail_bat_cut
        one = lambda B, *a, **k: [(0, B)]
        return (lambda tag: setattr(bwd, "_tail_bat_cut", deployed if tag == "A" else one)), True
    if name == "fills":
        deployed = bwd._dq_pipe_fills
        flat = lambda block_kv: 4
        return (lambda tag: setattr(bwd, "_dq_pipe_fills", deployed if tag == "A" else flat)), True
    if name == "gridcut":
        # A is the deployed uniform batch cut, B falls back to the tail-only cut it replaced,
        # so a POSITIVE paired_BmA_mean is the uniform plan winning.
        deployed = bwd._dq_grid_cut
        off = lambda *a, **k: 1
        return (lambda tag: setattr(bwd, "_dq_grid_cut", deployed if tag == "A" else off)), True
    if name.startswith("acut"):
        # Arm B makes the plan a UNIFORM grid: every chunk cut into N batch pieces, so no
        # hidden fold is ever wider than the body window it has to run under.
        n = int(name[4:])
        deployed = bwd._pipe_chunks
        cut = lambda B, qs, bkv, sq, hd=64, sbhd=False, **kw: _bat_cut(
            deployed(B, qs, bkv, sq, hd, sbhd, **kw), B, n, sbhd, 0
        )
        return (lambda tag: setattr(bwd, "_pipe_chunks", deployed if tag == "A" else cut)), True
    if name == "hbr":
        _read_half_hook()
        return (lambda tag: HALF.__setitem__("on", tag == "B")), False
    if name == "halfband":
        _read_half_hook()
        return _build_sets({"wsq_nrec": 2}, run), False
    if name.startswith("kw:"):
        # Arm B is the deployed builder with these kwargs overridden, so a POSITIVE
        # paired_BmA_mean means the override LOSES. exact=True holds every arm that only
        # moves a cache policy or a dispatch plane honest about it.
        over = {}
        for part in name[3:].split(","):
            key, _, val = part.partition("=")
            over[key] = int(val) if val.lstrip("-").isdigit() else val
        return _build_sets(over, run), True
    raise SystemExit(f"unknown knob {name}")


def main():
    a = sys.argv[1:]
    B, Hq, Hkv, S, D = (int(x) for x in a[:5])
    name = a[5]
    pairs = int(a[6]) if len(a) > 6 else 60
    W = int(a[7]) if len(a) > 7 else -1
    ws = (W, 0) if W >= 0 else (-1, -1)
    sh = lambda H: torch.randn(S, B, H, D, device=DEV, dtype=DT)
    q, k, v = sh(Hq), sh(Hkv), sh(Hkv)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=ws, return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    f = lambda: flash_attn_sbhd_flydsl_backward_impl(do, q, k, v, o, lse_h, causal=True, window_size=ws)

    enter, exact = arm_switch(name, f)
    chk, ref = {}, None
    for tag in ("A", "B"):
        enter(tag)
        out = f()[:3]
        chk["det_" + tag] = all(torch.equal(x, y) for x, y in zip(f()[:3], f()[:3]))
        if ref is None:
            ref = [x.clone() for x in out]
        elif exact:
            # A dispatch-plan knob may only move WHEN work runs, never what it computes.
            chk["equal_AB"] = all(torch.equal(x, y) for x, y in zip(ref, out))
        for _ in range(4):
            f()
    torch.cuda.synchronize()

    t = {"A": [], "B": []}
    for _ in range(pairs):
        for tag in ("A", "B"):
            enter(tag)
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            f()
            e.record()
            torch.cuda.synchronize()
            t[tag].append(s.elapsed_time(e))
    enter("A")

    def stat(x):
        x = sorted(x)
        n = len(x)
        return dict(
            min=round(x[0], 4),
            p10=round(x[max(0, n // 10)], 4),
            med=round(x[n // 2], 4),
            mean=round(sum(x) / n, 4),
        )

    d = [y - x for x, y in zip(t["A"], t["B"])]
    print(
        json.dumps(
            {
                "knob": name,
                "B": B,
                "Hq": Hq,
                "S": S,
                "D": D,
                "W": W,
                "pairs": pairs,
                "A": stat(t["A"]),
                "B_arm": stat(t["B"]),
                "paired_BmA_mean": round(sum(d) / len(d), 4),
                "A_wins_pairs": sum(1 for x in d if x > 0),
                **chk,
            }
        )
    )


if __name__ == "__main__":
    main()
