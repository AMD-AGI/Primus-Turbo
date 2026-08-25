#!/usr/bin/env python3
"""Price the kv-head axis and prove a head slice is bitwise the same work.

The tail-cut prize is arithmetic: at l70b the exposed tail is 0.334 ms (base 3.1798 against
notailred 2.8462) and cutting it into k pieces leaves 1/k exposed. The only unknown is what a
dispatch boundary on THIS axis costs, so run the whole body as k head slices back to back and
compare with the single launch. Correctness rides along: the pieces own disjoint kv heads and
every address is formed from the ABSOLUTE head index, so dK/dV/dQ must match bitwise.

usage: _probe_hcut.py <B> <Hq> <Hkv> <S> <D> <k> [reps]
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
NH = HKV // K


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

# Force the body out as K head slices: wrap the launcher the dispatcher pulls from cache.
_orig_get = M._get_bwd


def _get(*a, **kw):
    dkdv_l, odo_l = _orig_get(*a, **kw)
    if getattr(dkdv_l, "_hcut", False):
        return dkdv_l, odo_l
    base_chunk = dkdv_l.chunk
    plain = dkdv_l

    class _Sliced:
        _hcut = True
        chunk = staticmethod(base_chunk)

        def __call__(self, *aa, **kk):
            for h in range(0, HKV, NH):
                base_chunk(0, None, 0, h, NH)(*aa, **kk)

    sl = _Sliced()
    sl.chunk = lambda qsp_lo, n_qsp, bat_lo=0, head_lo=0, n_head=None: base_chunk(
        qsp_lo, n_qsp, bat_lo, head_lo, n_head)
    return sl, odo_l


M._get_bwd = _get
M._BWD_CACHE.clear()
got = [t.clone() for t in run()[:3]]
cut = timed(run)
same = all(torch.equal(a, b) for a, b in zip(ref, got))
print(json.dumps({
    "case": f"B{B} Hq{HQ} Hkv{HKV} S{S} D{D}", "k": K,
    "one_launch_ms": round(one, 4), "k_slices_ms": round(cut, 4),
    "boundary_cost_ms": round((cut - one) / max(K - 1, 1), 4),
    "bitwise_same": same}))
