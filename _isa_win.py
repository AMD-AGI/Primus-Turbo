#!/usr/bin/env python3
"""Windows around the full lgkmcnt(0) drains that immediately precede an MFMA.

Those are the coverage-zero points: the wave issued LDS reads and had nothing left to run
before the consumer. Printing the window is the only way to tell WHICH producer/consumer
pair is uncovered, which is what decides whether the cure is a deeper ring, an earlier
issue point, or a different fragment layout.

usage: _isa_win.py <isa.s> [n_windows] [radius]
"""
import collections
import sys

N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
R = int(sys.argv[3]) if len(sys.argv) > 3 else 14
L = [ln.strip() for ln in open(sys.argv[1])
     if ln.strip() and not ln.strip().startswith((";", "//", ".", "/*"))]
ops = [ln.split()[0] for ln in L]

hits = [i for i, ln in enumerate(L)
        if ln.startswith("s_waitcnt") and "lgkmcnt(0)" in ln
        and i + 1 < len(L) and ops[i + 1].startswith("v_mfma")]
print("coverage-zero drains before an MFMA: %d" % len(hits))

# distance back to the nearest ds_read, and how many reads are in the group
dist = collections.Counter()
grp = collections.Counter()
for i in hits:
    d, n = 0, 0
    for j in range(i - 1, max(-1, i - 80), -1):
        if ops[j].startswith("ds_read"):
            if d == 0:
                d = i - j
            n += 1
        elif ops[j].startswith(("v_mfma", "s_barrier")):
            break
    dist[min(d, 20)] += 1
    grp[min(n, 12)] += 1
print("instr distance issue->wait:", sorted(dist.items()))
print("ds_reads in the covered group:", sorted(grp.items()))

step = max(1, len(hits) // max(1, N))
for i in hits[::step][:N]:
    print("\n--- window @%d ---" % i)
    for j in range(max(0, i - R), min(len(L), i + 4)):
        print(("  >> " if j == i else "     ") + L[j][:110])
