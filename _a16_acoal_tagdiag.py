#!/usr/bin/env python3
"""Tagged touch-count diagnostic for the WSQ_ACOAL address pattern.

Uses the PT_A16_TAG payload switch (see flash_attn_bwd.py): the atomic's data operand
becomes a constant 1.0 (bf16 pair) when traced from the apply_mask=True q-loop copy and
16.0 from the apply_mask=False copy, so a clean fb() -> zero_() -> fb() run's readback is
n_masked + 16*n_unmasked per element -- exact up to bf16 saturation (keep < 256).

For a CORRECT scheme this must be a perfect causal partition, identical in shape to the
already-confirmed non-ACOAL a16 histogram (round 1 of this campaign): row-block r sees
value 16*r + 1 (r unmasked kv-blocks fully attended, plus its own masked diagonal block).
Any other value at a row that SHOULD hold one of those constants proves the coalesced
address pattern is writing to the wrong physical slot for at least one of the two
contributing writes (row/head aliasing), not merely a permutation to invert.

usage: PT_A16_TAG=1 python3 _a16_acoal_tagdiag.py <B> <Hq> <Hkv> <S> <D> [ACOAL=0|1]
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as M

B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
ACOAL = int(sys.argv[6]) if len(sys.argv) > 6 else 1
DEV, DT = "cuda", torch.bfloat16
assert os.environ.get("PT_A16_TAG") == "1", "run with PT_A16_TAG=1"
assert S <= 2048, "bf16 tag accumulator saturates above 256 -- keep S small"

_build = M.build_flash_attn_bwd_dkdv_module
def _spy_build(**kw):
    kw2 = {**kw, "wsq_a16": 1, "wsq_ilv": 1, "wsq_ring": 0, "wsq_acoal": ACOAL}
    return _build(**kw2)
M.build_flash_attn_bwd_dkdv_module = _spy_build

_ws = M._dq_partial_ws
_held = {}
def _ws1(nb, Bn, Sq, hd, device, dtype, pad_bytes=0, ilv=1, carry=False):
    t, c = _ws(nb, Bn, Sq, hd, device, dtype, pad_bytes, 1, carry)
    t.zero_()
    _held["ws"] = t
    return t, c
M._dq_partial_ws = _ws1

_red = M._reduce_dq_partials
_calls = []
M._reduce_dq_partials = lambda *a, **k: _calls.append((a, k))

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb,
    flash_attn_sbhd_flydsl_forward_impl as ff,
)

g = torch.Generator().manual_seed(11)
mk = lambda h: torch.randn(S, B, h, D, generator=g, dtype=DT).to(DEV)
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
do = torch.randn(S, B, Hq, D, generator=g, dtype=DT).to(DEV)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lse_h = lse.view(B, S, Hq).permute(0, 2, 1)

def run():
    if "ws" in _held:
        _held["ws"].zero_()
    _calls.clear()
    dq, dk, dv = fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))
    return dq, dk, dv

# throwaway compile-time launch (see pitfalls/05 sec "实现级坑"), then the clean measurement
run()
if "ws" in _held:
    _held["ws"].zero_()
dq, dk, dv = run()

ws = _held["ws"][0].float().cpu()  # single band, [B, Sq, Hq*D]
ws = ws.view(B, S, Hq, D)
print("ws shape", tuple(ws.shape), "nonzero frac %.4f" % float((ws != 0).float().mean()))

BLOCK = 128  # BLOCK_Q == BLOCK_KV == 128 by default in this file
nblk = (S + BLOCK - 1) // BLOCK
for rb in range(nblk):
    r0, r1 = rb * BLOCK, min((rb + 1) * BLOCK, S)
    n_unmasked = rb  # causal: rb full kv-blocks before the diagonal
    expect = 1.0 + 16.0 * n_unmasked
    blk = ws[:, r0:r1, :, :]
    vals, counts = torch.unique(blk, return_counts=True)
    top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:6]
    ok = (blk == expect).float().mean().item()
    print("row-block %2d [%4d:%4d) expect %6.1f  frac==expect %.4f  top(count,val)=%s"
          % (rb, r0, r1, expect, ok, top))
