"""Elementwise bitwise diff of an arm against the incumbent. Stronger than SQNR.

THE CANONICAL COPY. Every round from 3 to 9 carried a byte-identical
`_scratch/work/bitwise.py` (md5 bf5538332deb), inherited unread from the round before, and
every one of them called `forward_reference` at prod -- about 2000 dispatches of the fp32
Tensile GEMM that wedged round 8 *inside this very script*, at armB. Copy THIS file into a
round's `_scratch/work/`, not the previous round's.

What changed from bf5538332deb: `forward_reference` -> `refcache_util.cached_forward`.
Nothing else. The comparison, the dtypes and the output format are untouched, so a round's
output is directly comparable with rounds 3-9.

Why it matters: o and lse are only INPUTS here -- this script compares two implementations
against each other, never against the fp32 reference -- so there is no reason at all for it
to be computing that reference on the GPU. `cached_forward` reads them from
`op/refcache/<shape>.pt` and falls back to computing only when the cache does not apply
(missing entry, provenance mismatch, non-causal), so it is never worse than the call it
replaces.

    python3 bitwise.py <incumbent_dir> <candidate_dir> fast,proxy,prod
"""
import sys
from pathlib import Path

JOB = Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/"
           "gfx1250-flydsl-attn-bwd-20260917-115934")
sys.path.insert(0, str(JOB / "job_context/op/ut"))
sys.path.insert(0, str(JOB / "job_context/op"))      # refcache_util lives beside validation.py

import torch
from common import SHAPES, make_inputs, load_impl   # noqa: F401  -- SHAPES kept for parity
from refcache_util import cached_forward

ref = load_impl(Path(sys.argv[1]))
cand = load_impl(Path(sys.argv[2]))
for shape in sys.argv[3].split(","):
    q, k, v, do = make_inputs(shape, seed=0)
    o, lse = cached_forward(shape, q, k, v, causal=True)
    a = ref(do, q, k, v, o, lse, causal=True)
    b = cand(do, q, k, v, o, lse, causal=True)
    for n, x, y in zip(("dq", "dk", "dv"), a, b):
        eq = torch.equal(x.to(torch.float32), y.to(torch.float32))
        nd = int((x.to(torch.float32) != y.to(torch.float32)).sum())
        fin = f"finite {int(torch.isfinite(y.float()).sum())}/{y.numel()}"
        print(f"  {shape:6s} {n}: bitwise_identical={eq}  differing={nd}/{x.numel()}  "
              f"{fin}  dtype={y.dtype}", flush=True)
