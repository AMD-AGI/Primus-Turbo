import csv
import glob
import re
import sys
from collections import defaultdict

_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof_bwd"
f = sorted(glob.glob(f"{_dir}/**/*kernel_trace*csv", recursive=True))[-1]
rows = list(csv.DictReader(open(f)))
cols = list(rows[0].keys())
name_k = [c for c in cols if c.lower() in ("kernel_name", "name")][0]


def dur(r):
    return int(r["End_Timestamp"]) - int(r["Start_Timestamp"])


def short(n):
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*kernel[A-Za-z0-9_]*)", n)
    base = m.group(1) if m else n[:40]
    for key in ("dkdv", "odo", "delta", "fused_dq", "bwd_kernel"):
        if key in n:
            return key
    return base[:32]


agg = defaultdict(list)
for r in rows:
    agg[short(r[name_k])].append(dur(r))


def stat(name, xs):
    xs = sorted(xs)
    n = len(xs)
    print(f"{name:16s} calls={n:4d} avg={sum(xs)/n/1000:8.1f}us min={xs[0]/1000:8.1f} max={xs[-1]/1000:8.1f}")


for name in sorted(agg, key=lambda k: -sum(agg[k])):
    stat(name, agg[name])
print("total_kernels", sum(len(v) for v in agg.values()))
