#!/usr/bin/env python3
"""Count kernel dispatches in a rocprofv3 kernel-trace directory."""
import collections
import csv
import glob
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rp"
c = collections.Counter()
files = glob.glob(d + "/**/*.csv", recursive=True)
print("csv files:", files)
for f in files:
    if "kernel" not in f.lower():
        continue
    for r in csv.DictReader(open(f)):
        c[(r.get("Kernel_Name") or r.get("Name") or "?")[:70]] += 1
for k, v in c.most_common(15):
    print(v, k)
