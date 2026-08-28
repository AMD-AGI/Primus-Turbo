#!/usr/bin/env python3
"""Hot-block opcode census + issue budget for a dumped dkdv ISA.

usage: _isa_hot.py <dump-dir>
"""
import collections
import glob
import re
import sys

COST = {"v_mfma": 16, "ds_": 1, "buffer_": 1, "global_": 1, "s_": 1}


def cost(op):
    if op.startswith("v_mfma"):
        return 16
    if op.startswith("s_nop"):
        return 1
    return 1


d = sys.argv[1]
p = [x for x in glob.glob(d + "/**/21_final_isa.s", recursive=True) if "dkdv" in x][0]
b = open(p).read()
parts = re.split(r"^(\.LBB\S+):", b, flags=re.M)
best = None
for i in range(1, len(parts), 2):
    n = len(re.findall(r"^\s+[a-z]", parts[i + 1], re.M))
    if best is None or n > best[1]:
        best = (parts[i], n, parts[i + 1])
print("ISA", p)
print("hot block %s  instr=%d" % (best[0], best[1]))
c = collections.Counter(re.findall(r"^\s+([a-z_0-9]+)", best[2], re.M))
tot = sum(cost(k) * v for k, v in c.items())
mf = sum(v for k, v in c.items() if k.startswith("v_mfma"))
print("issue-cycle estimate %d, MFMA %d (%.1f%%)" % (tot, mf * 16, 100.0 * mf * 16 / tot))
for k, v in c.most_common(24):
    print("  %-34s %5d" % (k, v))
print("waits:", collections.Counter(re.findall(r"s_waitcnt\s+(\S+.*)", best[2])).most_common(8))
print("s_nop:", collections.Counter(re.findall(r"s_nop\s+(\d+)", best[2])).most_common(6))
