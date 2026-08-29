#!/usr/bin/env python3
"""Classify the full `s_waitcnt lgkmcnt(0)` drains of one kernel.

At one wave per SIMD a full drain blocks for the whole LDS return, so what MATTERS is why
the compiler could not use a counted wait. SMEM shares lgkmcnt and returns out of order, so
any s_load in flight forces every following LDS wait down to 0 -- that is a different cure
(hoist the scalar load) from a barrier drain (nothing to cure) or a genuine consumer stall.

usage: _isa_drain.py <isa.s>
"""
import collections
import sys

L = [ln.strip() for ln in open(sys.argv[1])
     if ln.strip() and not ln.strip().startswith((";", "//", ".", "/*"))]
ops = [ln.split()[0] for ln in L]
print("counts:", dict(collections.Counter(
    o for o in ops if o.startswith(("s_load", "s_buffer", "ds_read", "ds_write", "s_barrier",
                                    "v_mfma", "buffer_", "global_", "s_waitcnt")))))

cls = collections.Counter()
smem_open = collections.Counter()
for i, ln in enumerate(L):
    if not (ln.startswith("s_waitcnt") and "lgkmcnt(0)" in ln):
        continue
    nxt = ops[i + 1] if i + 1 < len(L) else "?"
    kind, smem = "none", 0
    for j in range(i - 1, max(-1, i - 60), -1):
        if ops[j].startswith(("s_load", "s_buffer")):
            smem = 1
        if ops[j].startswith("s_waitcnt") and "lgkmcnt(0)" in L[j]:
            break
        if kind == "none" and ops[j].startswith(("ds_read", "ds_write")):
            kind = ops[j][:8]
    cls[(kind, "smem" if smem else "-", nxt)] += 1
    smem_open[smem] += 1
print("drains with an SMEM op since the previous drain:", dict(smem_open))
for k, v in cls.most_common(20):
    print("  %-40s %d" % (str(k), v))
