#!/usr/bin/env python3
"""Correctness of an implementation against op/eager/, per tensor, at >= 50 dB.

    python3 test_correctness.py [--impl DIR] [--shapes a,b,c] [--list]

Gate, from op.precision_gate -- all three must hold for every shape:
  1. every element of dq, dk and dv is finite (asserted BEFORE any SQNR is computed,
     with the allocator NaN-poisoned so an unwritten element is visibly not a zero)
  2. SQNR of dq, dk and dv SEPARATELY >= 50 dB against the fp32 eager reference
  3. shapes are compared in both causal and non-causal mode where the config allows it

`--impl` is a PATH and stays a path. Exit 0 only if every shape passes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from common import (  # noqa: E402
    SHAPES, SPEC_SHAPES, causal_modes, forward_reference, load_impl, make_inputs,
    poison_allocator, sqnr_db,
)

# Loaded by path, not by `sys.path` + `from impl import ...`: that binds
# sys.modules["impl"], and every implementation in this job has a file called impl.py.
eager_attn_bwd = load_impl(HERE.parent / "eager")

GATE_DB = 50.0
DEFAULT = ["toy", "gqa4_small", "mha", "unequal_seqlen", "unequal_seqlen_2", "sq_gt_skv",
           "fast", "proxy", "prod"]


def check(fn, shape, causal, verbose=True):
    q, k, v, do = make_inputs(shape, seed=hash(shape) & 0xFFFF)
    o, lse = forward_reference(q, k, v, causal=causal)
    ref = eager_attn_bwd(do, q, k, v, o, lse, causal=causal)
    torch.cuda.synchronize()

    poison_allocator()
    t0 = time.perf_counter()
    got = fn(do, q, k, v, o, lse, causal=causal)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ok = True
    rows = []
    for name, r, g in zip(("dq", "dk", "dv"), ref, got):
        assert g.shape == r.shape, f"{name}: shape {tuple(g.shape)} != {tuple(r.shape)}"
        fin = int(torch.isfinite(g).sum())
        covered = fin == g.numel()
        db = sqnr_db(r, g) if covered else float("-inf")
        rows.append((name, fin, g.numel(), db))
        ok &= covered and db >= GATE_DB
    if verbose:
        tag = "causal" if causal else "full  "
        print(f"  {shape:<18} {tag}  {dt * 1e3:8.1f} ms  " + "  ".join(
            f"{n} {f}/{t} {d:7.2f} dB" for n, f, t, d in rows)
            + ("   PASS" if ok else "   FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=str(HERE.parent / "current"),
                    help="directory holding impl.py; a PATH, never a name")
    ap.add_argument("--shapes", default=",".join(DEFAULT))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n, s in SHAPES.items():
            print(f"{n:<18} {s}  {'[op.shape]' if n in SPEC_SHAPES else ''}")
        return 0

    impl = Path(args.impl).resolve()
    print(f"impl {impl}")
    fn = load_impl(impl)
    print(f"arch {torch.cuda.get_device_properties(0).gcnArchName}   gate {GATE_DB} dB")

    allok = True
    for shape in [s for s in args.shapes.split(",") if s]:
        for causal in causal_modes(shape):
            allok &= check(fn, shape, causal)
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())
