"""Print one steady-state slice of a FLYDSL ISA dump plus a per-K-iteration instruction budget."""
import re
import sys
from collections import Counter

p = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
count = int(sys.argv[3]) if len(sys.argv) > 3 else 130
lines = open(p).read().splitlines()

bar = [i for i, l in enumerate(lines) if l.strip().startswith("s_barrier")]
mfma = [i for i, l in enumerate(lines) if "v_mfma" in l]
print(f"lines={len(lines)} barriers={len(bar)} mfma={len(mfma)}")
if start == 0 and mfma:
    start = mfma[len(mfma) // 2] - 20

# instruction budget between consecutive barriers, in the middle of the unrolled loop
mid = [b for b in bar if abs(b - mfma[len(mfma) // 2]) < 4000]
if len(mid) > 20:
    segs = []
    for a, b in zip(mid[len(mid) // 2 : len(mid) // 2 + 9], mid[len(mid) // 2 + 1 : len(mid) // 2 + 10]):
        c = Counter()
        for l in lines[a + 1 : b]:
            s = l.strip()
            if not s or s.startswith((".", ";", "//")) or s.endswith(":"):
                continue
            c[s.split()[0]] += 1
        segs.append((b - a - 1, c.most_common(8)))
    print("\nper-barrier-segment budget (9 consecutive segments mid-loop):")
    for n, c in segs:
        print(f"  {n:4d} instrs  {c}")

print(f"\n--- ISA lines {start}..{start+count} ---")
for i in range(max(0, start), min(len(lines), start + count)):
    s = lines[i]
    if s.strip().startswith((".", ";")) and "amdhsa" not in s:
        continue
    print(f"{i:6d} {s}")
