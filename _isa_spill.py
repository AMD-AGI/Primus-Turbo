#!/usr/bin/env python3
"""Classify a dkdv ISA dump's scratch traffic by loop region and by spilled register.

_isa_body.py says HOW MUCH spills; this says WHAT spills and WHERE, which is what picking a
donor to move into LDS needs. Regions are the ISA's own basic blocks, folded into the two
innermost loops the body spends its time in (the causal-boundary phase and the interior
phase) plus prologue/epilogue, so a spill that only lives in the boundary phase can be told
from one on the hot path.

usage: _isa_spill.py <dump-dir> [kernel-substring]
"""
import glob
import os
import re
import sys
from collections import Counter

DUMP = sys.argv[1]
WANT = sys.argv[2] if len(sys.argv) > 2 else "dkdv"

for path in sorted(glob.glob(os.path.join(DUMP, "**", "21_final_isa.s"), recursive=True)):
    kn = os.path.basename(os.path.dirname(path))
    if WANT not in kn:
        continue
    lines = open(path).read().splitlines()
    # Loop bodies: a block is a loop head if some later branch targets it.
    tgt = Counter(m.group(1) for l in lines for m in [re.search(r"s_(?:cbranch\w*|branch) (\.LBB\S+)", l)] if m)
    blocks, cur = [], None
    for i, l in enumerate(lines):
        m = re.match(r"^(\.LBB\S+):", l)
        if m:
            cur = [m.group(1), i, i]
            blocks.append(cur)
        elif cur:
            cur[2] = i
    size = {b[0]: b[2] - b[1] for b in blocks}
    per_block, per_reg, per_op = Counter(), Counter(), Counter()
    cur = "<prologue>"
    for l in lines:
        m = re.match(r"^(\.LBB\S+):", l)
        if m:
            cur = m.group(1)
        if "scratch_" in l:
            op = l.split()[0]
            per_op[op] += 1
            per_block[cur] += 1
            r = re.findall(r"\b[va]\[?\d+", l)
            if r:
                per_reg[r[0]] += 1
    print(f"=== {kn}  scratch ops {sum(per_op.values())}")
    for op, n in per_op.most_common():
        print(f"    {op:28s} {n}")
    print("    by block (block size in lines, loop-target count):")
    for b, n in per_block.most_common(12):
        print(f"      {b:16s} ops={n:5d} lines={size.get(b, 0):6d} targeted={tgt.get(b, 0)}")
    print("    by first register operand:")
    for r, n in per_reg.most_common(16):
        print(f"      {r:10s} {n}")
