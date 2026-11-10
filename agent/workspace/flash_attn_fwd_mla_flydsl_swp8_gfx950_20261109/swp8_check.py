"""Correctness gate for the eight-stage forward against the round-7 dualwave kernel.

Default: bitwise vs round 7 (same softmax arithmetic, so any difference is a pipeline bug).
--approx: for variants whose fp32 rounding order differs from round 7; every rep must be
bitwise equal to the first (a race shows up as run-to-run noise) and within
max|dO| <= O_TOL, max|dLSE| <= LSE_TOL of round 7. The tolerances are a coarse bug screen,
not the accuracy gate (that is quick_test_bench vs fp32): S perturbations of ~1e-5 flip
bf16 roundings of P, which the bf16 row sum turns into ~1e-4..1e-3 LSE moves on short rows.
Usage:
    python swp8_check.py [--shapes b1_s4k,...] [--reps N] [--approx] [--no-stagger] [--no-hints]
"""

import argparse
import os
import sys

import torch

# SWP8_PKG_ROOT points at a directory holding another flydsl_mla copy (e.g. an older round).
sys.path.insert(0, os.environ.get("SWP8_PKG_ROOT", "/perf_apps/xiaoming/megaattention_overlap_bench"))

from flydsl_mla import mla_fwd_sbhd, mla_fwd_sbhd_swp8  # noqa: E402

SHAPES = {
    "b1_s1k": (1, 1024),
    "b1_s4k": (1, 4096),
    "b1_s16k": (1, 16384),
    "b2_s4k": (2, 4096),
    "b1_s64k": (1, 65536),
}
O_TOL = 8e-3
LSE_TOL = 2e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="b1_s1k,b1_s4k,b1_s16k,b2_s4k,b1_s64k")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--approx", action="store_true")
    ap.add_argument("--no-stagger", action="store_true")
    ap.add_argument("--no-hints", action="store_true")
    args = ap.parse_args()
    kw = {"stagger": not args.no_stagger, "sched_hints": not args.no_hints}
    ok = True
    for label in args.shapes.split(","):
        b, s = SHAPES[label]
        g = torch.Generator(device="cuda").manual_seed(1234 + s + b)
        q = torch.randn((s, b, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn((s, b, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
        o_ref, lse_ref = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
        bad = 0
        first = None
        for _ in range(args.reps):
            o, lse = mla_fwd_sbhd_swp8(q, k, v, causal=True, return_lse=True, **kw)
            torch.cuda.synchronize()
            d = (o.float() - o_ref.float()).abs()
            dl = (lse - lse_ref).abs()
            if args.approx:
                if first is None:
                    first = (o.clone(), lse.clone())
                same = torch.equal(o, first[0]) and torch.equal(lse, first[1])
                good = same and d.max().item() <= O_TOL and dl.max().item() <= LSE_TOL
            else:
                good = torch.equal(o, o_ref) and torch.equal(lse, lse_ref)
            if not good:
                bad += 1
                rows = (d.amax((1, 2, 3)) > 0).nonzero().flatten()
                print(
                    f"  {label}: mismatch max|dO| {d.max().item():.3e} max|dLSE| {dl.max().item():.3e}"
                    f" rows {rows.numel()} first {rows[:8].tolist()} finite {bool(torch.isfinite(o).all())}",
                    flush=True,
                )
        tag = "deterministic and within tolerance of" if args.approx else "bitwise equal to"
        print(
            f"{label}: {args.reps - bad}/{args.reps} {tag} round 7"
            f" (max|dO| {d.max().item():.3e} max|dLSE| {dl.max().item():.3e})",
            flush=True,
        )
        ok &= bad == 0
        del q, k, v, o_ref, lse_ref
        torch.cuda.empty_cache()
    print(("ALL WITHIN TOLERANCE" if args.approx else "ALL BITWISE EQUAL") if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
