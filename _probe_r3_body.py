"""Round-3: full instruction budget of the REAL main-loop body (pitfalls/09 span rule).

Same body-selection rule as _probe_r1_loop.py (largest back-edge span holding v_mfma and no
buffer_store), but the histogram is bucketed into issue classes so the per-K-iter numbers can be
compared against the matrix-pipe shadow: one K-step is 32 v_mfma/wave x 32 cycles = 1024 cycles of
matrix pipe per wave, and with 2 waves/SIMD every non-MFMA instruction of BOTH waves has to issue
inside that same window at >=4 cycles each.
"""

import glob
import re
from collections import Counter


def bucket(l):
    for p, k in (
        ("v_mfma", "mfma"),
        ("ds_read", "ds_read"),
        ("ds_write", "ds_write"),
        ("buffer_load", "vmem_load"),
        ("buffer_store", "vmem_store"),
        ("global_", "vmem_other"),
        ("s_barrier", "barrier"),
        ("s_setprio", "setprio"),
        ("s_waitcnt", "waitcnt"),
        ("s_nop", "nop"),
        ("s_load", "sload"),
        ("v_", "valu"),
        ("s_", "salu"),
    ):
        if l.startswith(p):
            return k
    return None


for f in sorted(glob.glob("/root/.flydsl/debug/*/21_final_isa.s")):
    lines = [l.strip() for l in open(f)]
    if sum(1 for l in lines if l.startswith("v_mfma")) < 500:
        continue
    labels = {}
    for i, l in enumerate(lines):
        m = re.match(r"^(\.?[A-Za-z_][\w.$]*):$", l)
        if m:
            labels[m.group(1)] = i
    best = None
    for i, l in enumerate(lines):
        m = re.match(r"^s_(cbranch\w*|branch)\s+(\S+)", l)
        if not (m and m.group(2) in labels and labels[m.group(2)] < i):
            continue
        a, b = labels[m.group(2)], i
        seg = lines[a : b + 1]
        nm = sum(1 for x in seg if x.startswith("v_mfma"))
        if nm and not any(x.startswith(("buffer_store", "global_store")) for x in seg):
            if best is None or nm > best[2]:
                best = (a, b, nm)
    print(f"=== {f.split('/')[-2]}")
    if best is None:
        print("   no mfma-only back-edge span found")
        continue
    a, b, nm = best
    c = Counter()
    wc = Counter()
    for l in lines[a : b + 1]:
        k = bucket(l)
        if k:
            c[k] += 1
        if l.startswith("s_waitcnt"):
            wc[l.split(None, 1)[1] if " " in l else "?"] += 1
    ki = nm / 32.0
    print(f"   body {a}..{b} ({b - a + 1} lines) v_mfma={nm} => K-steps={ki:.2f}")
    tot4 = 0
    for k in sorted(c, key=lambda x: -c[x]):
        if k != "mfma":
            tot4 += c[k]
        print(f"   {k:11s} {c[k]:5d}  per-K-iter {c[k] / ki:7.2f}")
    print(f"   NON-MFMA per-K-iter/wave = {tot4 / ki:.1f}  (budget 256 before it leaves the shadow)")
    print("   waitcnt modes:", dict(sorted(wc.items(), key=lambda x: -x[1])[:8]))
