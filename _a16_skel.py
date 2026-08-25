#!/usr/bin/env python3
"""End-to-end skeleton for the bf16 packed-atomic dQ scheme, validated before it is wired
into the dispatcher.

The scheme replaces the per-band dQ partial store + multi-band fold with:
  1. one band-less bf16 dQ image, zeroed;
  2. the body adding into it with `buffer_atomic_pk_add_bf16` (`wsq_a16`);
  3. ONE fold pass over a single band.

Step 3 needs no new kernel: the existing fold already reads the PERMUTED partial layout,
applies the scale and un-permutes on the store side
(`(o - (o&31)) + ((o&24)>>1) + (((o>>2)&1)<<4)`), so folding one band IS aiter's
`dq_shuffle` + scale. Calling it with block_kv = Skv makes the band count 1.

dQ is non-deterministic under this scheme by construction (atomic completion order) and its
accumulator is bf16, so the gate here is SNR against a chunked fp32 reference, not bitwise.

usage: _a16_skel.py <B> <Hq> <Hkv> <S> <D> [reps]
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as M

B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 30
DEV, DT = "cuda", torch.bfloat16

# --- 1. body emits atomics into a band-less image -------------------------------------
# A16=0 is the bisect arm: normal partial stores, EVERYTHING else identical (all slots
# zeroed, one fold at the end with qsp dropped). If dQ is right there the fold invocation is
# sound and the fault is in the atomics; if it is wrong there too the fault is mine.
A16 = int(os.environ.get("A16", "1"))
# ACOAL=1 selects the coalesced address pattern (`WSQ_ACOAL`, see `_g3_a16`): 64 lanes cover
# one contiguous 256 B run (4 lines) instead of 16 B apart across 16 rows (16 lines).
ACOAL = int(os.environ.get("ACOAL", "0"))
if int(os.environ.get("QSP1", "0")):
    M._qsplit_for = lambda *a, **k: 1     # host AND builder move together (see _probe_bwdcfg)
_seen_kw = {}
_launches = []
_build = M.build_flash_attn_bwd_dkdv_module
def _spy_build(**kw):
    ov = {"wsq_a16": A16, "wsq_ilv": 1, "wsq_ring": 0, "wsq_acoal": ACOAL}
    if not int(os.environ.get("DEFER", "1")):
        # A store is idempotent, so a path that emits the same dQ closure twice is invisible
        # on the deployed tree; an atomic doubles. Count mode showed exactly 2 touches per
        # element and wave-0-only attribution showed it is the SAME wave twice, so the
        # deferred emission is the remaining suspect. This turns it off.
        ov.update(g3_defer=False, g3_st_at=-1, g3_st_n=None, g3_dbat=None)
    kw2 = {**kw, **ov}
    _seen_kw.update({k: kw2.get(k) for k in ("q_split", "g3_defer", "g3_st_at", "g3_st_n",
                                             "g3_dbat", "kv_halves", "wsq_a16", "wsq_ilv")})
    print("   builder kw:", _seen_kw)
    _l = _build(**kw2)

    def _counting(*a, **k):
        # Same WG, same wave, same static instruction, twice -> the remaining explanation is
        # that the host DISPATCHES the body twice over the same (band, q) range. A store is
        # idempotent so that would be invisible on the deployed path.
        _launches.append(tuple(x for x in a if isinstance(x, int)))
        return _l(*a, **k)

    return _counting


M.build_flash_attn_bwd_dkdv_module = _spy_build

# --- 2. one zeroed band instead of n_bands --------------------------------------------
_ws = M._dq_partial_ws
_held = {}


def _ws1(nb, Bn, Sq, hd, device, dtype, pad_bytes=0, ilv=1, carry=False):
    """Keep the DEPLOYED band count and zero every slot.

    Correctness first: the atomics all land in slot 0 (the band-less image), and with slots
    1..n-1 held at zero the deployed fold's `sum over bands 0..g` is identically slot 0. That
    decouples "are the atomics + the un-permute right" from "how do we make the fold read one
    band", which is a pure performance question. It reads n_bands times the bytes meanwhile.
    """
    t, c = _ws(nb, Bn, Sq, hd, device, dtype, pad_bytes, 1, carry)
    t.zero_()
    _held["ws"], _held["nb"] = t, nb
    return t, c


M._dq_partial_ws = _ws1

# --- 3. the fold runs ONCE, over one band, after the body ------------------------------
_red = M._reduce_dq_partials
_calls = []
M._reduce_dq_partials = lambda *a, **k: _calls.append((a, k))

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


def run(verbose=True):
    # the workspace is allocated inside the first backward, so zeroing lives in _ws1 too
    if "ws" in _held:
        _held["ws"].zero_()
    _calls.clear()
    dq, dk, dv = fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))
    # one fold over the single band: block_kv = Skv, everything else as the host asked for
    if verbose:
        print("dkdv launches:", len(_launches), " int-args:", _launches)
        print("fold calls:", len(_calls))
        for _a, _k in _calls:
            print("   args[2:7]=", _a[2:7], " kw=", {kk: vv for kk, vv in _k.items() if kk != "ph"})
        print("   ws nonzero frac %.4f  absmean %.4g" % (
            float((_held["ws"] != 0).float().mean()), float(_held["ws"].float().abs().mean())))
    # Replay the host's fold sequence VERBATIM. Collapsing it into one call with `qsp`
    # dropped does not work: the fold's row grouping depends on QSP (`SQ % (BQ*QSP)`), so a
    # QSP=1 pass reads a different row->slot map than the QSP=2 body wrote. The bisect that
    # found this: with the atomics OFF (plain partial stores) the collapsed call was also
    # wrong (2.0 dB), which put the fault in the invocation rather than in the atomics.
    for _a, _k in _calls:
        _red(_held["ws"], dq, *_a[2:], **_k)
    return dq, dk, dv


# ★ flyc.compile(raw, *args) executes the kernel ONCE into the real buffers during
# compilation (see pitfalls/05 §实现级坑). A store is idempotent so that pass is invisible
# on the deployed tree; an atomic accumulator is not -- the FIRST call below is the one
# that triggers (or hits a cached) JIT compile and its buffer pollution, so it is thrown
# away. Only the SECOND call, after an explicit zero_(), measures a clean single-pass
# atomic accumulation. This is the fb() -> zero_() -> fb() sequence _a16_cnt.py uses.
_throwaway_dq, _throwaway_dk, _throwaway_dv = run(verbose=False)
if "ws" in _held:
    _held["ws"].zero_()
dq, dk, dv = run()
if os.environ.get("DUMP_WS"):
    # The atomics accumulate every band into slot 0, so the a16 image must equal the
    # BAND-SUM of the plain-store workspace. Comparing those two isolates "the body wrote
    # the wrong place" from "the fold read the wrong place" -- reasoning about the two
    # addressings from source has now failed twice.
    import numpy as _np
    _t = _held["ws"].detach().float().cpu()
    _np.save(os.environ["DUMP_WS"], _t.numpy())
    print("  ws dumped", tuple(_t.shape), "band0 absmean %.5g  all absmean %.5g"
          % (float(_t[0].abs().mean()), float(_t.abs().mean())))
print("ok, dq finite:", bool(torch.isfinite(dq).all()), " nonzero:", float(dq.abs().mean()))

# reference
bh = lambda t: t.permute(1, 2, 0, 3).float().requires_grad_()
qf, kf, vf = bh(q), bh(k), bh(v)
gg = Hq // Hkv
s = (qf @ kf.repeat_interleave(gg, 1).transpose(-1, -2)) * D**-0.5
m = torch.ones(S, S, device=DEV, dtype=torch.bool).tril()
(s.masked_fill(~m, float("-inf")).softmax(-1) @ vf.repeat_interleave(gg, 1)).backward(
    do.permute(1, 2, 0, 3).float())


def snr(ref, got):
    ref = ref.float()
    e = ref - got.float()
    return 10 * math.log10(float(ref.pow(2).sum()) / max(float(e.pow(2).sum()), 1e-30))


for nm, r, x in (("dq", qf, dq), ("dk", kf, dk), ("dv", vf, dv)):
    print("  %s SNR %.1f dB" % (nm, snr(r.grad, x.permute(1, 2, 0, 3))))

# where does dQ land? compare one row against the reference, and look for the reference
# values elsewhere in the same row (a pure permutation) vs nowhere (a row/head mis-map).
got = dq.permute(1, 2, 0, 3)[0, 0, 0].float()      # [D] for b0 h0 q0
ref = qf.grad[0, 0, 0].float()
print("  ref[:8] ", [round(x, 4) for x in ref[:8].tolist()])
print("  got[:8] ", [round(x, 4) for x in got[:8].tolist()])
import torch as _t
d = (got.view(-1, 1) - ref.view(1, -1)).abs()
hit = (d < 1e-2 * ref.abs().clamp(min=1e-3).view(1, -1)).any(1).float().mean()
print("  frac of got[D] found somewhere in ref[D] (same row): %.3f" % float(hit))
allref = qf.grad[0, :, :, :].float().reshape(-1)
print("  |got| mean %.4g   |ref| mean %.4g   ratio %.3f"
      % (float(got.abs().mean()), float(ref.abs().mean()),
         float(got.abs().mean() / max(float(ref.abs().mean()), 1e-9))))
