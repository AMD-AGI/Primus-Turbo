#!/usr/bin/env python3
"""MFMA accumulator-chain distance in the hot block.

The body is MFMA-idle roughly half the time while its non-MFMA issue is only ~7% of the
trip, so the binding quantity is how far apart two MFMAs that share an accumulator sit --
i.e. how many INDEPENDENT chains the emission interleaves. This reports, for every MFMA,
the number of MFMAs since the last write of its C operand (the chain distance), plus the
run-length histogram of consecutive MFMAs.

usage: _isa_dep.py <dump-dir>
"""
import collections
import glob
import re
import sys

d = sys.argv[1]
p = [x for x in glob.glob(d + "/**/21_final_isa.s", recursive=True) if "dkdv" in x][0]
b = open(p).read()
parts = re.split(r"^(\.LBB\S+):", b, flags=re.M)
best = None
for i in range(1, len(parts), 2):
    n = len(re.findall(r"^\s+[a-z]", parts[i + 1], re.M))
    if best is None or n > best[1]:
        best = (parts[i], n, parts[i + 1])
lines = [l.strip() for l in best[2].splitlines() if re.match(r"^\s+[a-z]", l)]

MF = re.compile(r"v_mfma_f32_16x16x32_bf16\s+([av])\[(\d+):\d+\],\s*\S+\[[^]]*\],\s*\S+\[[^]]*\],\s*([av])\[(\d+):\d+\]")
last = {}
dist = []
idx = 0
run = 0
runs = []
inplace = 0
for l in lines:
    m = MF.match(l)
    if not m:
        if run:
            runs.append(run)
            run = 0
        continue
    run += 1
    dcls, dreg, ccls, creg = m.groups()
    if (dcls, dreg) == (ccls, creg):
        inplace += 1
    key = (ccls, creg)
    if key in last:
        dist.append(idx - last[key])
    last[(dcls, dreg)] = idx
    idx += 1
if run:
    runs.append(run)
print("hot block", best[0], "instr", best[1], "mfma", idx, "in-place", inplace)
h = collections.Counter(dist)
print("chain distance (MFMAs between two writes of the same accumulator):")
for k in sorted(h)[:14]:
    print("   d=%-4d %5d" % (k, h[k]))
print("   median", sorted(dist)[len(dist) // 2] if dist else None, " mean %.2f" % (sum(dist) / max(1, len(dist))))
print("MFMA run lengths:", collections.Counter(runs).most_common(10))
