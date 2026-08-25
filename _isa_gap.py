#!/usr/bin/env python3
"""How much of the kernel's issue time has the MFMA pipe idle?

At one wave per SIMD there is no sibling to cover a gap, so every maximal run of
instructions that contains no MFMA is MFMA-pipe idle time. Costs are issue slots:
wave64 VALU/TRANS occupy 4 cycles, SALU/branch 1, memory issue 1 (the latency is
covered by the waitcnt, which is counted where it stalls, not here).

usage: _isa_gap.py <isa.s>
"""
import collections
import re
import sys

COST = {"VALU": 4, "TRANS": 4, "SALU": 1, "MEM": 1, "WAIT": 1, "BAR": 1, "NOP": 1, "MFMA": 4}


def kind(op):
    if op.startswith("v_mfma"):
        return "MFMA"
    if op.startswith(("v_exp", "v_log", "v_rcp", "v_rsq", "v_sqrt")):
        return "TRANS"
    if op.startswith("v_"):
        return "VALU"
    if op.startswith(("ds_", "buffer_", "global_", "flat_", "scratch_")):
        return "MEM"
    if op.startswith("s_waitcnt"):
        return "WAIT"
    if op.startswith("s_barrier"):
        return "BAR"
    if op.startswith("s_nop"):
        return "NOP"
    if op.startswith("s_"):
        return "SALU"
    return "SALU"


ops = []
for ln in open(sys.argv[1]):
    t = ln.strip()
    if not t or t.startswith((".", ";", "/")) or t.endswith(":"):
        continue
    o = t.split()[0]
    if re.match(r"^[a-z]", o):
        ops.append((kind(o), o))

total = sum(COST[k] for k, _ in ops)
mfma_cycles = sum(COST[k] for k, _ in ops if k == "MFMA")

# maximal runs with no MFMA
runs, cur = [], []
for k, o in ops:
    if k == "MFMA":
        if cur:
            runs.append(cur)
            cur = []
    else:
        cur.append((k, o))
if cur:
    runs.append(cur)

gap_cycles = sum(sum(COST[k] for k, _ in r) for r in runs)
print("issue cycles: total %d, mfma %d (%.1f%%), non-mfma %d (%.1f%%)"
      % (total, mfma_cycles, 100 * mfma_cycles / total, gap_cycles, 100 * gap_cycles / total))

big = sorted(runs, key=lambda r: -sum(COST[k] for k, _ in r))
print("\nthe 12 largest MFMA-idle runs (cycles, composition, first ops):")
for r in big[:12]:
    c = collections.Counter(k for k, _ in r)
    cyc = sum(COST[k] for k, _ in r)
    print("  %5d cyc  %-46s %s" % (cyc, dict(c), " ".join(o for _, o in r[:5])))

# how much of the idle time sits in runs of each size class
buckets = collections.Counter()
for r in runs:
    cyc = sum(COST[k] for k, _ in r)
    b = "1-8" if cyc <= 8 else "9-32" if cyc <= 32 else "33-128" if cyc <= 128 else ">128"
    buckets[b] += cyc
print("\nidle cycles by run size:", dict(buckets))
print("runs:", len(runs), " idle cycles per mfma:", round(gap_cycles / max(1, mfma_cycles / 4), 2))
