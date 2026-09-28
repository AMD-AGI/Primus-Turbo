"""Correctness and determinism checks shared by ut/test_correctness.py and validation.py.

Kept OUTSIDE ut/common.py and eager/impl.py so editing it never invalidates the refcache.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ut"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from common import SHAPES, make_inputs, sqnr_db  # noqa: E402
from poison_util import poison_allocator  # noqa: E402
from refcache_util import reference  # noqa: E402

GATE_DB = 49.0   # B0 2026-09-27 (as A0 r10): baseline itself reads 49.82-49.99 dB on edge cases; op.precision_sqnr_db; validation re-reads it from the spec and passes it in


def check_correctness(fn, shape, causal, gate_db=GATE_DB):
    """o and lse SEPARATELY: NaN-prefill, full isfinite coverage, then SQNR >= gate_db."""
    q, k, v = make_inputs(shape, seed=0)
    ref_o, ref_lse, src = reference(shape, q, k, v, causal=causal)
    torch.cuda.synchronize()
    b, sq, skv, hq, hkv, d = SHAPES[shape]
    poison_allocator(max_bytes=b * sq * hq * d * 2)      # largest output: o in bf16
    got_o, got_lse = fn(q, k, v, causal=causal)
    torch.cuda.synchronize()
    res = {"shape": shape, "causal": causal, "ref": src, "ok": True}
    for name, r, g in (("o", ref_o, got_o), ("lse", ref_lse, got_lse)):
        assert tuple(g.shape) == tuple(r.shape), f"{name}: {tuple(g.shape)} != {tuple(r.shape)}"
        fin = int(torch.isfinite(g).sum())
        covered = fin == g.numel()
        db = sqnr_db(r, g) if covered else float("-inf")   # never SQNR over an uncovered buffer
        res[name] = {"finite": fin, "numel": g.numel(), "db": db}
        res["ok"] &= covered and db >= gate_db
    return res


def fmt_correctness(r):
    tag = "causal" if r["causal"] else "full  "
    return (f"{r['shape']:<16} {tag} " + "  ".join(
        f"{n} {r[n]['finite']}/{r[n]['numel']} {r[n]['db']:7.2f} dB" for n in ("o", "lse"))
        + f"  ref={r['ref']}  " + ("PASS" if r["ok"] else "FAIL"))


def _digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]


def check_determinism(fn, shape="fast", runs=200, causal=True):
    """o and lse bitwise identical across `runs` consecutive calls. Returns (ok, detail)."""
    q, k, v = make_inputs(shape, seed=0)
    o0, l0 = fn(q, k, v, causal=causal)
    o0, l0 = o0.clone(), l0.clone()
    torch.cuda.synchronize()
    bad = []
    for i in range(1, runs):
        o, l = fn(q, k, v, causal=causal)
        if not (torch.equal(o.view(torch.int16), o0.view(torch.int16))
                and torch.equal(l.view(torch.int32), l0.view(torch.int32))):
            bad.append(i)
    torch.cuda.synchronize()
    return (not bad), (f"{runs} runs at {shape}: {runs - len(bad)}/{runs} bitwise identical, "
                       f"o={_digest(o0)} lse={_digest(l0)}"
                       + (f", first mismatch at run {bad[0]}" if bad else ""))
