"""timeline.py <q22_tl log>: per-10-WMMA segment cycles of the r10w step from the r10w_t1..t4 probe rungs.
Each rung's 4 phase sums are the step's segments [0, p1), [p1, p2), [p2, p3), [p3, end) (probe points after WMMA
#p); t1..t4 put them at 10/20/30, 40/50/60, 70/80/90, 100/110/119, so [30, 40) = t2's [0, 40) - t1's three, etc."""
import re
import sys

PTS = {"r10w_t1": (10, 20, 30), "r10w_t2": (40, 50, 60), "r10w_t3": (70, 80, 90), "r10w_t4": (100, 110, 119)}
txt = open(sys.argv[1]).read()
ph = {}
for b in re.split(r"^== ", txt, flags=re.M)[1:]:
    name = b.split("\n", 1)[0].strip()
    m = re.search(r"wave 0: (\d+) steps, (\d+) cycles per iteration\n((?:[^\n]*\n){4})", b)
    if m:
        ph[name] = (float(m.group(2)), [float(x.split()[-1]) for x in m.group(3).strip().split("\n")])
for k, (tot, v) in ph.items():
    if k not in PTS:
        print(f"{k}: {tot:.0f} cycles per step; phases {v}")
t1, t2, t3, t4 = (ph[k][1] for k in PTS)
segs = [(0, 10, t1[0]), (10, 20, t1[1]), (20, 30, t1[2]), (30, 40, t2[0] - sum(t1[:3])), (40, 50, t2[1]),
        (50, 60, t2[2]), (60, 70, t3[0] - t2[0] - t2[1] - t2[2]), (70, 80, t3[1]), (80, 90, t3[2]),
        (90, 100, t4[0] - t3[0] - t3[1] - t3[2]), (100, 110, t4[1]), (110, 119, t4[2]), (119, 120, t4[3])]
for lo, hi, v in segs:
    n = hi - lo
    print(f"W{lo:3d}..W{hi - 1:3d}: {v:7.1f} cycles  {v / n:5.1f} per WMMA  (floor {8 * n:4d}, excess {v - 8 * n:6.1f})")
print("per-rung step totals:", {k: round(v[0]) for k, v in ph.items()})
