#!/usr/bin/env python3
"""Correctness of an implementation against op/eager/ (the only precision reference).

    python3 test_correctness.py [--impl DIR] [--shapes a,b,c] [--determinism N] [--list]

For every shape and allowed causal mode: output buffers NaN-prefilled, full isfinite
coverage of o AND lse asserted, then SQNR of o and lse SEPARATELY >= 50 dB. `--impl` is a
PATH and stays a path. Exit 0 only if everything passes.
"""
from __future__ import annotations

import os

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from common import SHAPES, SPEC_SHAPES, causal_modes, load_impl  # noqa: E402
from gates import GATE_DB, check_correctness, check_determinism, fmt_correctness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=str(HERE.parent / "current"),
                    help="directory holding impl.py; a PATH, never a name")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--determinism", type=int, default=0, help="runs at `fast`; 0 = skip")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, s in SHAPES.items():
            print(f"{n:<16} {s}  {'[op.shape]' if n in SPEC_SHAPES else ''} causal={causal_modes(n)}")
        return 0
    impl = Path(a.impl).resolve()
    fn = load_impl(impl)
    print(f"impl {impl}\narch {torch.cuda.get_device_properties(0).gcnArchName}  gate {GATE_DB} dB")
    print(f"ENV TORCH_BLAS_PREFER_HIPBLASLT={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT')} "
          f"HIPBLASLT_TENSILE_LIBPATH={os.environ.get('HIPBLASLT_TENSILE_LIBPATH')}")
    ok = True
    for shape in [s for s in a.shapes.split(",") if s]:
        for causal in causal_modes(shape):
            r = check_correctness(fn, shape, causal)
            print("  " + fmt_correctness(r), flush=True)
            ok &= r["ok"]
            torch.cuda.empty_cache()
    if a.determinism:
        dok, detail = check_determinism(fn, "fast", a.determinism)
        print(f"  determinism {detail}  {'PASS' if dok else 'FAIL'}")
        ok &= dok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
