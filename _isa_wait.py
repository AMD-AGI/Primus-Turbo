#!/usr/bin/env python3
"""Break an ISA dump's s_waitcnt down by what it actually waits on.

At one wave per SIMD nothing hides behind a sibling, so a wait with a nonzero count is a
scheduling hint and a wait with count 0 is a full drain -- the two cost very different
things, and the ratio says whether the waits are a symptom or the disease.
"""
import collections
import re
import sys

path = sys.argv[1]
c = collections.Counter()
nop = collections.Counter()
for ln in open(path):
    t = ln.strip()
    if t.startswith("s_waitcnt"):
        parts = re.findall(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)", t)
        if not parts:
            c[t.split(None, 1)[1] if " " in t else "?"] += 1
        else:
            c[" ".join("%s(%s)" % p for p in parts)] += 1
    elif t.startswith("s_nop"):
        nop[t] += 1
print("s_waitcnt by argument:")
for k, v in c.most_common(20):
    print("  %-40s %5d" % (k, v))
print("total", sum(c.values()))
print("s_nop:", dict(nop.most_common(6)), "total", sum(nop.values()))
