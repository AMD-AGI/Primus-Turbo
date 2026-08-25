#!/usr/bin/env python3
"""Barrier-delimited segment profile of one loop arm, plus per-segment LDS coverage.

pitfalls/13 §2026-08-19: LDS issue->retire coverage distance is the only static quantity
that tracks the wall on this body. This prints it PER SEGMENT so a candidate can be aimed
at the segment that is actually short-covered instead of at the loop's mean.

usage: _isa_seg2.py <isa.s> <label> [kernel]
"""
import collections
import re
import sys

PATH, LABEL = sys.argv[1], sys.argv[2]
KERNEL = sys.argv[3] if len(sys.argv) > 3 else "flash_attn_bwd_dkdv_kernel_0"


def family(op):
    if op.startswith("v_mfma"):
        return "MFMA"
    if op.startswith("ds_read"):
        return "dsr"
    if op.startswith("ds_write"):
        return "dsw"
    if op == "s_waitcnt":
        return "wait"
    if op.startswith(("buffer_", "global_", "flat_")):
        return "vmem"
    if op == "s_nop":
        return "nop"
    if op == "s_barrier":
        return "bar"
    if op.startswith("s_"):
        return "SALU"
    return "VALU"


lines = open(PATH, errors="ignore").read().split("\n")
start = end = None
for i, ln in enumerate(lines):
    if ln.startswith(KERNEL + ":"):
        start = i
    if start is not None and end is None and i > start and ".Lfunc_end" in ln:
        end = i
body = [ln.strip() for ln in lines[start:end]]
head = next(i for i, ln in enumerate(body) if ln.strip().startswith(LABEL + ":"))
tail = next(
    i for i, ln in enumerate(body) if i > head and re.search(rf"s_c?branch\w*\s+{re.escape(LABEL)}\b", ln)
)
ins = [s for s in body[head : tail + 1] if s and not s.startswith((";", ".")) and not s.endswith(":")]
print(f"{LABEL}: {len(ins)} instructions")

# coverage distance per LDS op, keyed by the instruction index so it can be bucketed
pend, cover = [], {}
for i, s in enumerate(ins):
    if s.startswith(("ds_read", "ds_write")):
        pend.append(i)
    elif s.startswith("s_waitcnt") and "lgkmcnt" in s:
        n = int(re.search(r"lgkmcnt\((\d+)\)", s).group(1))
        while len(pend) > n:
            j = pend.pop(0)
            cover[j] = i - j

if len(sys.argv) > 5:
    lo, hi = int(sys.argv[4]), int(sys.argv[5])
    for i in range(lo, min(hi, len(ins))):
        s = ins[i]
        tag = f" cover={cover[i]}" if i in cover else ""
        print(f"{i:5d} {s}{tag}")
    raise SystemExit

bounds = [i for i, s in enumerate(ins) if s.startswith("s_barrier")]
segs, prev = [], 0
for b in bounds + [len(ins) - 1]:
    segs.append((prev, b))
    prev = b + 1
for k, (a, b) in enumerate(segs):
    fam = collections.Counter()
    drains = 0
    for i in range(a, b + 1):
        fam[family(ins[i].split()[0])] += 1
        if ins[i].startswith("s_waitcnt") and "lgkmcnt(0)" in ins[i]:
            drains += 1
    ds = [cover[i] for i in range(a, b + 1) if i in cover]
    mc = sum(ds) / max(len(ds), 1)
    short = sum(1 for d in ds if d <= 8)
    print(
        f"  seg{k:2d} [{a:5d},{b:5d}] n={b - a + 1:5d} drain0={drains:2d} "
        f"cover n={len(ds):4d} mean={mc:6.1f} <=8:{short:3d}  {dict(fam.most_common())}"
    )
