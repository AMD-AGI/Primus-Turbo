#!/usr/bin/env python3
"""End-to-end a16 on the NATIVE COALESCED image (wsq_a16=64), with the un-permute done as a
pure torch bit-permutation.

The body adds dQ into a band-less bf16 image whose flat index is
    blk*512 + w*128 + lane*2 + half,   blk = ((qtile_g*Hq + qh)*NPAIR + dpair)
so one instruction's 64 lanes cover 256 B (4 cache lines) instead of the store layout's 16.
Every axis is a power of two and the q/d maps are bit permutations, so recovering dQ is a
`view -> permute -> reshape`, which is exact by construction and lets the scheme be measured
end to end before a fused kernel exists for it.

  image axes: qblk(Sq/64) t1 t0 qh dp1 dp0 p s kg1 kg0 b3 b2 b1 b0 half
  q = [t1, b2, t0, b3, b1, b0]        (the inverse of the dS qp permutation, _g3_qrow)
  d = [dp1, dp0, p, kg1, kg0, s, half]

usage: _a16_nat.py <B> <Hq> <Hkv> <S> <D> [reps]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as M

B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 20
BQ = 64
NPAIR = D // 32
PERM = (0, 1, 2, 12, 3, 11, 13, 14, 4, 5, 6, 7, 9, 10, 8, 15)

_build = M.build_flash_attn_bwd_dkdv_module
_orig_ws = M._dq_partial_ws
_orig_red = M._reduce_dq_partials
_held = {}


def _spy_build(**kw):
    return _build(**{**kw, "wsq_a16": 64, "wsq_ilv": 1, "wsq_ring": 0})


def _ws_flat(nb, Bn, Sq, hd, device, dtype, pad_bytes=0, ilv=1, carry=False):
    """The deployed allocation (so every band SRD the body builds stays inside it), zeroed.

    a16 only ever touches the first B*Sq*Hq*D elements -- the band-less image -- but the body
    still BUILDS the per-band descriptors, so shrinking the buffer is not safe.
    """
    t, c = _orig_ws(nb, Bn, Sq, hd, device, dtype, pad_bytes, 1, carry)
    t.zero_()
    _held["ws"] = t
    return t, c


def _unpermute(ws, dq, block_kv, num_heads, head_dim, scale, stream, **kw):
    Bn = dq.shape[1] if kw.get("sbhd") else dq.shape[0]
    Sq = dq.shape[0] if kw.get("sbhd") else dq.shape[1]
    n = Bn * Sq * num_heads * head_dim
    v = ws.reshape(-1)[:n].view(
        Bn, Sq // BQ, 2, 2, num_heads, NPAIR // 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    )
    out = v.permute(*PERM).reshape(Bn, Sq, num_heads, head_dim)
    if kw.get("sbhd"):
        out = out.permute(1, 0, 2, 3)
    _held["scale"] = scale
    dq.copy_(out)
    dq.mul_(scale)


def _install():
    M.build_flash_attn_bwd_dkdv_module = _spy_build
    M._dq_partial_ws = _ws_flat
    M._reduce_dq_partials = _unpermute


def _restore():
    M.build_flash_attn_bwd_dkdv_module = _build
    M._dq_partial_ws = _orig_ws
    M._reduce_dq_partials = _orig_red


from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (  # noqa: E402
    flash_attn_sbhd_flydsl_backward_impl as fb,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (  # noqa: E402
    flash_attn_sbhd_flydsl_forward_impl as ff,
)

g = torch.Generator().manual_seed(7)
mk = lambda h: torch.randn(S, B, h, D, generator=g, dtype=torch.bfloat16).cuda()
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
do = torch.randn(S, B, Hq, D, generator=g, dtype=torch.bfloat16).cuda()
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lse_h = lse.view(B, S, Hq).permute(0, 2, 1)


def _run():
    return fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))


def _snr(ref, got):
    ref, got = ref.float(), got.float()
    n = (ref - got).pow(2).mean()
    return float("inf") if n == 0 else 10 * torch.log10(ref.pow(2).mean() / n).item()


def _bench(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


# ★ The impl CACHES the built launcher, so the builder patch has to be in place before the
# very first call -- installing it after a reference run silently reuses the deployed module.
if os.environ.get("AXIS"):
    _install()
    _run()
    _run()
    ref = [None, None, None]
    got = ref
    t_base = t_a16 = float("nan")
else:
    # --- deployed reference (warm) ---------------------------------------------------
    _run()
    ref = [x.clone() for x in _run()[:3]]
    t_base = _bench(_run, REPS)
    _install()
    _run()  # cold call dispatches the grid twice; never read or time it
    got = [x.clone() for x in _run()[:3]]
    t_a16 = _bench(_run, REPS)
    _restore()

if os.environ.get("AXIS"):
    img = _held["ws"].reshape(-1)[: B * S * Hq * D]
    print("  M._A16_TAG =", M._A16_TAG, " env:", os.environ.get("PT_A16_TAG"))
    print("  image nonzero:", int((img != 0).sum()), "of", img.numel(),
          " sample bits:", [hex(int(x) & 0xFFFF) for x in img[:4].view(torch.int16).tolist()])
    v = img.view(B, S // BQ, 2, 2, Hq, NPAIR // 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    dec = ((v.float() / 2.0) - 1.0) * 128.0
    out = dec.permute(*PERM).reshape(B, S, Hq, D)
    if os.environ["AXIS"] == "13":
        want = torch.arange(S, device=out.device).view(1, S, 1, 1).expand_as(out).float()
    else:
        d = torch.arange(D, device=out.device)
        want = (d - d % 4).view(1, 1, 1, D).expand_as(out).float()
    bad = (out - want).abs() > 0.4
    print("  axis %s: mismatch frac %.4f" % (os.environ["AXIS"], float(bad.float().mean())))
    print("  out[0,:4,0,0] =", out[0, :4, 0, 0].tolist(), " want", want[0, :4, 0, 0].tolist())
    print("  out[0,0,0,:8] =", out[0, 0, 0, :8].tolist(), " want", want[0, 0, 0, :8].tolist())
    sys.exit(0)

if os.environ.get("SOLVE"):
    # Solve the image -> dQ bit permutation from the data. Both index spaces are the same
    # size and the map is a pure bit permutation, so flipping one image-index bit flips
    # exactly one dQ-index bit: read it off.
    img = _held["ws"].reshape(-1)[: B * S * Hq * D].float().cpu()
    tgt = (ref[0].float() / _held["scale"]).permute(1, 0, 2, 3).reshape(-1).cpu()
    tgt = tgt.bfloat16().float()  # the image holds bf16 partials
    # value -> index (bf16 values of random data are near-unique)
    pos = {}
    for i, x in enumerate(tgt.tolist()):
        pos.setdefault(x, []).append(i)
    nbits = (B * S * Hq * D).bit_length() - 1
    import random

    random.seed(0)
    votes = {}
    tries = 0
    for _ in range(4000):
        i0 = random.randrange(1 << nbits)
        v0 = float(img[i0])
        if v0 == 0.0 or len(pos.get(v0, [])) != 1:
            continue
        j0 = pos[v0][0]
        for b in range(nbits):
            i1 = i0 ^ (1 << b)
            v1 = float(img[i1])
            if v1 == 0.0 or len(pos.get(v1, [])) != 1:
                continue
            d = j0 ^ pos[v1][0]
            if d and (d & (d - 1)) == 0:
                votes.setdefault(b, {}).setdefault(d.bit_length() - 1, 0)
                votes[b][d.bit_length() - 1] += 1
        tries += 1
        if tries > 200:
            break
    print("  image bit -> dQ bit (dQ index = ((b*S + q)*Hq + h)*D + d):")
    for b in range(nbits):
        vv = votes.get(b, {})
        top = sorted(vv.items(), key=lambda kv: -kv[1])[:2]
        print("    img bit %2d -> %s" % (b, top))

for nm, r, gt in zip(("dq", "dk", "dv"), ref, got):
    print("  %s SNR %6.1f dB" % (nm, _snr(r, gt)))
print("  base %.4f ms   a16-native %.4f ms   delta %+.2f%%" % (t_base, t_a16, 100 * (t_a16 / t_base - 1)))
