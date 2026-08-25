#!/usr/bin/env python3
"""Reproduce the in-training TE/AITER a16 per-layer numbers, and pin which shape they are.

Two settings decide WHICH kernel runs, and both defaults are wrong for this comparison:
  * ``deterministic`` -- flash_attn_func defaults it True, and the gfx950 v3 gate is
    ``not deterministic or seqlen_k <= 256``, so the default sends the call down CK.
  * ``is_v3_atomic_fp32`` -- defaults True (a32, fp32 atomics on dQ). The reported table is
    a16, the bf16 packed-atomic asm kernel, so it must be passed False.

Timed through aiter's own public autograd API (what TE calls), fwd and bwd separately, min
of N in this process, so the rows add up the way the reported table's do.

usage: _aiter_repro.py [--a32] [reps]
"""
import json, sys, torch, aiter
from aiter.ops.mha import FlashAttnFunc

DEV, DT = "cuda", torch.bfloat16
A32 = "--a32" in sys.argv
REPS = next((int(a) for a in sys.argv[1:] if a.isdigit()), 30)
WARMS = 5
D = 128
# ms alone cannot separate a batch split from a seqlen split at equal flops, so every split
# that lands on the reported flop count is timed.
CASES = [
    ("70b B1 S8192", 1, 64, 8, 8192),
    ("70b B2 S8192", 2, 64, 8, 8192),
    ("70b B1 S16384", 1, 64, 8, 16384),
    ("70b B8 S8192 tp8", 8, 8, 1, 8192),
    ("8b  B1 S16384", 1, 32, 8, 16384),
    ("8b  B4 S8192", 4, 32, 8, 8192),
    ("8b  B1 S8192", 1, 32, 8, 8192),
]


def timed(fn):
    for _ in range(WARMS):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(REPS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


for label, B, Hq, Hkv, S in CASES:
    torch.manual_seed(11)
    mk = lambda H, g: torch.randn(B, S, H, D, device=DEV, dtype=DT, requires_grad=g)
    #      q k v  dropout scale causal window     bias alibi det   lse    softmax grad  a32
    run = lambda a, b, c, g: FlashAttnFunc.apply(
        a, b, c, 0.0, None, True, (-1, -1, 0), None, None, False, True, False, g, A32)[0]
    qn, kn, vn = mk(Hq, False), mk(Hkv, False), mk(Hkv, False)
    with torch.no_grad():
        f_ms = timed(lambda: run(qn, kn, vn, False))
    del qn, kn, vn

    q, k, v = mk(Hq, True), mk(Hkv, True), mk(Hkv, True)
    out = run(q, k, v, True)
    go = torch.randn_like(out)

    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(go, retain_graph=True)

    b_ms = timed(bwd)
    ffl, bfl = 2 * B * Hq * S * S * D, 5 * B * Hq * S * S * D
    print(json.dumps({
        "case": label, "atomic": "a32" if A32 else "a16",
        "fwd_ms": round(f_ms, 4), "fwd_tfs": round(ffl / f_ms / 1e9, 1),
        "bwd_ms": round(b_ms, 4), "bwd_tfs": round(bfl / b_ms / 1e9, 1),
        "pair_ms": round(f_ms + b_ms, 4)}), flush=True)
    del q, k, v, out, go
    torch.cuda.empty_cache()
