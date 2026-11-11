"""MFMA / non-MFMA run structure of the dkdv hot loop: python hot_runs.py <22_final_isa.s> [--seq].

Prints the hot loop (last loop with >= 50 MFMAs) as runs: 'M<n>' = n MFMAs with nothing between
them, otherwise the classes of the instructions between two MFMAs. --seq prints every gap."""
import re
import sys
from collections import Counter

path = sys.argv[1]
L = open(path).read().split("\n")
lab = {m.group(1): i for i, l in enumerate(L) if (m := re.match(r"^(\.LBB0_\d+):", l))}
loops = {}
for i, l in enumerate(L):
    m = re.search(r"s_cbranch_\w+\s+(\.LBB0_\d+)|s_branch\s+(\.LBB0_\d+)", l)
    t = m and (m.group(1) or m.group(2))
    if t in lab and lab[t] < i:
        loops[lab[t]] = max(loops.get(lab[t], 0), i)
hot = [(a, b) for a, b in sorted(loops.items()) if sum("v_mfma" in s for s in L[a : b + 1]) >= 50][-1]
ins = [s.strip() for s in L[hot[0] : hot[1] + 1] if s.strip() and not s.strip().startswith((";", ".", "//"))]


def cls(s):
    op = s.split()[0]
    if op.startswith("v_mfma"):
        return "M"
    if op.startswith("v_exp"):
        return "E"
    if op.startswith("ds_read") or op.startswith("ds_load"):
        return "R"
    if op.startswith("ds_write") or op.startswith("ds_store"):
        return "W"
    if op.startswith("scratch_load"):
        return "L"
    if op.startswith("scratch_store"):
        return "S"
    if op.startswith("buffer_atomic") or op.startswith("global_atomic"):
        return "A"
    if op.startswith("buffer_load") or op.startswith("global_load"):
        return "G"
    if op.startswith("s_waitcnt"):
        return "w"
    if op.startswith("s_barrier"):
        return "b"
    if op.startswith("s_nop"):
        return "n"
    if op.startswith("v_accvgpr"):
        return "a"
    if op.startswith("v_"):
        return "V"
    if op.startswith("s_"):
        return "s"
    return "?"


c = [cls(s) for s in ins]
print("hot loop lines %d-%d, %d instructions:" % (hot[0], hot[1], len(c)), dict(Counter(c)))
# gaps between consecutive MFMAs
gaps, cur, pos = [], [], []
for i, k in enumerate(c):
    if k == "M":
        gaps.append("".join(cur))
        pos.append(i)
        cur = []
    else:
        cur.append(k)
tail = "".join(cur)
n_m = len(pos)
zero = sum(1 for g in gaps[1:] if g == "")
print("MFMAs %d; gaps of 0 instructions between consecutive MFMAs: %d; <=1: %d; <=2: %d" % (
    n_m, zero, sum(1 for g in gaps[1:] if len(g) <= 1), sum(1 for g in gaps[1:] if len(g) <= 2)))
# segments: runs of MFMAs whose gaps are all short (< 2) vs long windows
seq = []
run = 0
for i, g in enumerate(gaps):
    if i > 0 and len(g) < 2:
        run += 1
        if g:
            seq.append(g)
        continue
    if run:
        seq.append("M%d" % run)
    if i > 0 or g:
        seq.append("[%s]" % g if len(g) <= 24 else "[%d: %s]" % (len(g), "".join("%s%d" % (k, v) for k, v in Counter(g).most_common())))
    run = 1
seq.append("M%d" % run)
seq.append("[tail %d: %s]" % (len(tail), "".join("%s%d" % (k, v) for k, v in Counter(tail).most_common())))
print(" ".join(seq))
if "--seq" in sys.argv:
    for i, (g, p) in enumerate(zip(gaps, pos)):
        print(i, p, g)
