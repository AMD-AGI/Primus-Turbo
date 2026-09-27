"""Resource + instruction-histogram diff of two FlyDSL ISA dumps (22_final_isa.s)."""
import collections
import re
import sys

KEYS = ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
        "group_segment_fixed_size", "private_segment_fixed_size")


def stats(path):
    res, hist = {}, collections.Counter()
    for line in open(path):
        m = re.match(r"\s+\.(\w+):\s+(\d+)", line)
        if m and m.group(1) in KEYS:
            res[m.group(1)] = int(m.group(2))
        m = re.match(r"\s+([a-z][a-z0-9_]+)(\s|$)", line)
        if m and not line.lstrip().startswith("."):
            hist[m.group(1)] += 1
    return res, hist


a, b = (stats(p) for p in sys.argv[1:3])
for k in KEYS:
    print(f"{k:28s} {a[0].get(k)!s:>8} {b[0].get(k)!s:>8}")
print(f"{'total instructions':28s} {sum(a[1].values()):>8} {sum(b[1].values()):>8}")
diff = {op for op in a[1].keys() | b[1].keys() if a[1][op] != b[1][op]}
print("histogram ops differing:", len(diff), sorted(diff)[:20])
if len(sys.argv) > 3:
    for op, n in a[1].most_common(int(sys.argv[3])):
        print(f"  {op:32s} {n:6d} {b[1][op]:6d}")
