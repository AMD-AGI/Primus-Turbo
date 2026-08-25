#!/usr/bin/env python3
"""aiter a16 / aiter a32 / FlyDSL on ONE ruler, at the two shapes the reported table pins.

Same process, same clock, min of N each, fwd and bwd timed separately. aiter wants bshd and
FlyDSL wants sbhd, so each is given the layout it wants -- this compares implementations,
not transposes.

a16 is only reachable through FlashAttnFunc.apply: flash_attn_func hardcodes
is_v3_atomic_fp32=True. deterministic must be False or the gfx950 v3 gate falls back to CK.

usage: _repro_3way.py [reps]
"""
import json, sys, torch, aiter
from aiter.ops.mha import FlashAttnFunc

DEV, DT = "cuda", torch.bfloat16
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WARMS, D = 5, 128
CASES = [("70b", 1, 64, 8, 8192), ("8b", 4, 32, 8, 8192), ("70b_b2", 2, 64, 8, 8192)]


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


def aiter_arm(B, Hq, Hkv, S, a32):
    mk = lambda H, g: torch.randn(B, S, H, D, device=DEV, dtype=DT, requires_grad=g)
    run = lambda a, b, c, g: FlashAttnFunc.apply(
        a, b, c, 0.0, None, True, (-1, -1, 0), None, None, False, True, False, g, a32)[0]
    torch.manual_seed(11)
    qn, kn, vn = mk(Hq, False), mk(Hkv, False), mk(Hkv, False)
    with torch.no_grad():
        f = timed(lambda: run(qn, kn, vn, False))
    del qn, kn, vn
    q, k, v = mk(Hq, True), mk(Hkv, True), mk(Hkv, True)
    out = run(q, k, v, True)
    go = torch.randn_like(out)

    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(go, retain_graph=True)

    b = timed(bwd)
    del q, k, v, out, go
    torch.cuda.empty_cache()
    return f, b


def flydsl_arm(B, Hq, Hkv, S):
    from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
        flash_attn_sbhd_flydsl_backward_impl as fb, flash_attn_sbhd_flydsl_forward_impl as ff)
    torch.manual_seed(11)
    mk = lambda H: torch.randn(S, B, H, D, device=DEV, dtype=DT)
    q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
    do = torch.randn_like(q)
    f = timed(lambda: ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True))
    o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
    lh = lse.view(B, S, Hq).permute(0, 2, 1)
    b = timed(lambda: fb(do, q, k, v, o, lh, causal=True, window_size=(-1, -1)))
    del q, k, v, do, o, lse
    torch.cuda.empty_cache()
    return f, b


for label, B, Hq, Hkv, S in CASES:
    ffl, bfl = 2 * B * Hq * S * S * D, 5 * B * Hq * S * S * D
    row = {"case": f"{label} B{B} Hq{Hq} S{S}"}
    for name, (f, b) in (("aiter_a16", aiter_arm(B, Hq, Hkv, S, False)),
                         ("aiter_a32", aiter_arm(B, Hq, Hkv, S, True)),
                         ("flydsl", flydsl_arm(B, Hq, Hkv, S))):
        row[name] = {"fwd": round(f, 4), "bwd": round(b, 4),
                     "fwd_tfs": round(ffl / f / 1e9, 1), "bwd_tfs": round(bfl / b / 1e9, 1)}
    row["bwd_flydsl_over_a16"] = round(row["flydsl"]["bwd"] / row["aiter_a16"]["bwd"], 4)
    print(json.dumps(row), flush=True)
