#!/usr/bin/env python3
"""Segment a rocprofv3 kernel-trace CSV into iterations and print the last one.

Autotune and warm-up dispatches pollute a median over the whole file (memory's
round-1 pitfall), so segment on the unit's first kernel and report the LAST full
iteration in dispatch order.
"""
import csv
import glob
import sys
from collections import OrderedDict

MARK = sys.argv[2] if len(sys.argv) > 2 else "rmsnorm_fwd_residual"


def main(d):
    files = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)
    rows = []
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    key = "Start_Timestamp" if "Start_Timestamp" in rows[0] else "Start Timestamp"
    end = "End_Timestamp" if "End_Timestamp" in rows[0] else "End Timestamp"
    name = "Kernel_Name" if "Kernel_Name" in rows[0] else "Kernel Name"
    rows.sort(key=lambda r: int(r[key]))
    marks = [i for i, r in enumerate(rows) if MARK in r[name]]
    if len(marks) < 2:
        print(f"only {len(marks)} marks; total rows {len(rows)}")
        for r in rows[-40:]:
            print(f"{(int(r[end]) - int(r[key])) / 1000.0:9.2f} us  {r[name][:90]}")
        return
    lo, hi = marks[-2], marks[-1]
    seg = rows[lo:hi]
    tot = 0.0
    agg = OrderedDict()
    print(f"--- iteration: {len(seg)} dispatches (of {len(rows)} total, {len(marks)} marks)")
    for r in seg:
        us = (int(r[end]) - int(r[key])) / 1000.0
        tot += us
        n = r[name].split("(")[0][:72]
        a = agg.setdefault(n, [0, 0.0])
        a[0] += 1
        a[1] += us
        print(f"{us:9.2f} us  {r[name][:100]}")
    print(f"--- total {tot:.1f} us")
    print("--- aggregated")
    for n, (c, us) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{us:9.2f} us  x{c:<3d} {n}")


if __name__ == "__main__":
    main(sys.argv[1])
