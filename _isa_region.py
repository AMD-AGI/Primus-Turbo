"""Segment one kernel's ISA into prologue / MFMA span / epilogue and histogram each."""
import re
import sys
from collections import Counter

p = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "kernel_grouped_mxfp8_nt_1"
lines = open(p).read().splitlines()

cur, body, start = None, [], 0
for i, ln in enumerate(lines):
    m = re.match(r"^([A-Za-z_][\w$.]*):\s*(;.*)?$", ln)
    if m and not m.group(1).startswith(".L"):
        cur = m.group(1)
        if cur == want:
            start = i
    elif cur == want:
        body.append((i - start, ln.strip()))


def op(s):
    return s.split()[0] if s and not s.startswith((".", ";", "//")) and not s.endswith(":") else None


mf = [i for i, s in body if s.startswith("v_mfma")]
st = [i for i, s in body if s.startswith("buffer_store_short")]
print(f"{want}: {len(body)} lines | mfma [{mf[0]}..{mf[-1]}] n={len(mf)} | store [{st[0]}..{st[-1]}] n={len(st)}")

regions = {
    "prologue  [0,mfma0)": (0, mf[0]),
    "mainloop  [mfma0,mfmaN]": (mf[0], mf[-1] + 1),
    "epilogue  (mfmaN,end]": (mf[-1] + 1, 10**9),
}
for name, (lo, hi) in regions.items():
    c = Counter()
    for i, s in body:
        if lo <= i < hi:
            o = op(s)
            if o:
                c[o] += 1
    print(f"\n== {name}  total={sum(c.values())}")
    print("   ", c.most_common(22))

# spill-instruction placement
for pat in ("v_writelane_b32", "v_readlane_b32", "v_readfirstlane_b32"):
    idx = [i for i, s in body if s.startswith(pat)]
    if idx:
        pre = sum(1 for i in idx if i < mf[0])
        mid = sum(1 for i in idx if mf[0] <= i <= mf[-1])
        post = sum(1 for i in idx if i > mf[-1])
        print(f"{pat:22s} n={len(idx):4d}  prologue={pre} mainloop={mid} epilogue={post}")

# per-K-iter slice: window between the 1st and 33rd mfma of the full body
if len(mf) > 40:
    lo, hi = mf[8], mf[40]
    c = Counter()
    for i, s in body:
        if lo <= i <= hi:
            o = op(s)
            if o:
                c[o] += 1
    print(f"\n== steady-state window mfma[8..40] lines {lo}..{hi} total={sum(c.values())}")
    print("   ", c.most_common(22))
