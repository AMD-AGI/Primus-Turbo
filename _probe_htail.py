#!/usr/bin/env python3
"""The kv-head tail cut, end to end: is it bitwise the same dQ/dK/dV, and what does it buy?

The deployed tail cut runs on the BATCH axis, which does not exist at B=1 -- the shape
production trains at -- so l70b leaves 81% of its fold exposed (base 3.1798, notailred
2.8462, nored 2.7682). This cuts the same tail on the kv-head axis instead: a GQA group's q
heads read only their own kv head's bands, so a head piece folds on its own, and the boundary
on that axis measured 0.004 ms (see _probe_hcut.py).

Emulated at the call site rather than inside _fused_pipelined, so the deployed dispatcher is
untouched until this reads positive: the body goes out as K head slices and the fold follows
each slice for its own q-head range.

usage: _probe_htail.py <B> <Hq> <Hkv> <S> <D> <k> [reps]
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import primus_turbo.flydsl.attention.flash_attn_bwd as M
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb,
    flash_attn_sbhd_flydsl_forward_impl as ff,
)

DEV, DT = "cuda", torch.bfloat16
B, HQ, HKV, S, D, K = (int(x) for x in sys.argv[1:7])
REPS = int(sys.argv[7]) if len(sys.argv) > 7 else 30
assert HKV % K == 0
NH, G = HKV // K, HQ // HKV


def timed(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(REPS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b))
    return best


torch.manual_seed(11)
mk = lambda h: torch.randn(S, B, h, D, device=DEV, dtype=DT)
q, k, v = mk(HQ), mk(HKV), mk(HKV)
do = torch.randn_like(q)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lh = lse.view(B, S, HQ).permute(0, 2, 1)
run = lambda: fb(do, q, k, v, o, lh, causal=True, window_size=(-1, -1))
ref = [t.clone() for t in run()[:3]]
one = timed(run)

_orig_red = M._reduce_dq_partials
_orig_get = M._get_bwd
state = {"piece": None}


def _red(*a, **kw):
    if state["piece"] is not None and kw.get("qh") is None:
        kw["qh"] = state["piece"]
    return _orig_red(*a, **kw)


def _get(*a, **kw):
    dkdv_l, odo_l = _orig_get(*a, **kw)
    if getattr(dkdv_l, "_ht", False):
        return dkdv_l, odo_l
    base_chunk = dkdv_l.chunk

    class _S:
        _ht = True

        def __call__(self, *aa, **kk):
            for h in range(0, HKV, NH):
                base_chunk(0, None, 0, h, NH)(*aa, **kk)

    sl = _S()
    sl.chunk = base_chunk
    return sl, odo_l


M._reduce_dq_partials = _red
M._get_bwd = _get
M._BWD_CACHE.clear()
got = [t.clone() for t in run()[:3]]
cut = timed(run)
print(json.dumps({
    "case": f"B{B} Hq{HQ} Hkv{HKV} S{S} D{D}", "k": K,
    "base_ms": round(one, 4), "head_sliced_ms": round(cut, 4),
    "delta_pct": round((cut - one) / one * 100, 2),
    "bitwise_same": all(torch.equal(x, y) for x, y in zip(ref, got))}))
