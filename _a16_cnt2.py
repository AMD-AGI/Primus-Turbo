#!/usr/bin/env python3
"""Tagged touch-count probe for the a16 atomic dQ emission.

PT_A16_TAG=1 makes every `buffer_atomic_pk_add_bf16` add a CONSTANT instead of the value:
1.0 from the traced apply_mask=True q-loop copy, 16.0 from the apply_mask=False copy. The
image then reads back as `n_masked + 16*n_unmasked` per element, so one run says both how
many times an element is touched AND which traced copy did it.

usage: _a16_cnt.py <B> <Hq> <Hkv> <S> <D>
"""
import collections, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as M

B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
DEV, DT = "cuda", torch.bfloat16

if int(os.environ.get("QSP1", "0")):
    M._qsplit_for = lambda *a, **k: 1

_build = M.build_flash_attn_bwd_dkdv_module
_seen = {}
_launches = []


def _spy_build(**kw):
    ov = {"wsq_a16": 1, "wsq_ilv": 1, "wsq_ring": 0}
    for _kv in os.environ.get("OV", "").split(","):
        if _kv:
            _k, _v = _kv.split("=")
            ov[_k] = int(_v)
    for k in ("q_split", "g3_defer", "g3_st_at", "g3_st_n", "g3_dbat", "kv_halves", "block_kv"):
        _seen[k] = kw.get(k)
    print("   builder kw:", _seen, "ov:", ov)
    _l = _build(**{**kw, **ov})

    def _counting(*a, **k):
        _launches.append((tuple(x for x in a if isinstance(x, int)), tuple(sorted(
            (kk, vv) for kk, vv in k.items() if isinstance(vv, int)))))
        return _l(*a, **k)

    return _counting


M.build_flash_attn_bwd_dkdv_module = _spy_build

_ws = M._dq_partial_ws
_held = {}


def _ws1(nb, Bn, Sq, hd, device, dtype, pad_bytes=0, ilv=1, carry=False):
    t, c = _ws(nb, Bn, Sq, hd, device, dtype, pad_bytes, 1, carry)
    t.zero_()
    _held["ws"], _held["nb"] = t, nb
    return t, c


M._dq_partial_ws = _ws1
M._reduce_dq_partials = lambda *a, **k: None  # the fold would overwrite the counts

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (  # noqa: E402
    flash_attn_sbhd_flydsl_backward_impl as fb,
    flash_attn_sbhd_flydsl_forward_impl as ff,
)

g = torch.Generator().manual_seed(11)
mk = lambda h: torch.randn(S, B, h, D, generator=g, dtype=DT).to(DEV)
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
do = torch.randn(S, B, Hq, D, generator=g, dtype=DT).to(DEV)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lse_h = lse.view(B, S, Hq).permute(0, 2, 1)

fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))
if "ws" in _held and not int(os.environ.get("SINGLE", "0")):
    _held["ws"].zero_()
    fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))

import struct as _st
_e = _held["ws"].reshape(-1)[:2].cpu().view(torch.int16).tolist()
_bits = ((_e[1] & 0xFFFF) << 16) | (_e[0] & 0xFFFF)
print("wsq_cnt trip counter:", _st.unpack("<f", _st.pack("<I", _bits))[0])
print("dkdv launches:", len(_launches))
for _i, _L in enumerate(_launches):
    print("   launch", _i, _L)
t = _held["ws"].float().cpu()
print("ws shape", tuple(t.shape), "bands", _held["nb"])
img = t.reshape(-1)[: B * S * Hq * D].reshape(B, S, Hq, D)
BQ = int(os.environ.get("BQP","64"))
print("q-block -> (site0_deferhook, site1_undeferred, site2_defertail, site3_shadow) counts")
for qb in range(0, S, BQ):
    vals = img[0, qb : qb + BQ, 0, :].reshape(-1)
    u = collections.Counter(int(x) for x in vals.tolist())
    dec = {(c & 3, (c >> 2) & 3, (c >> 4) & 3, (c >> 6) & 3): n for c, n in sorted(u.items())}
    print("  q[%5d:%5d] %s" % (qb, qb + BQ, dec))
_f = _held["ws"].reshape(-1).cpu().view(torch.float32)
print("as-f32 nonzero:", int((_f != 0).sum()), " values:", sorted(set(float(x) for x in _f[_f != 0].tolist()))[:6])
print("nonzero elements:", int((img != 0).sum()), " values:", sorted(set(float(x) for x in img[img != 0].reshape(-1).tolist()))[:8])
print("global value histogram:", dict(collections.Counter(int(x) for x in img.reshape(-1).tolist()).most_common(12)))
