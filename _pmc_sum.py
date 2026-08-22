#!/usr/bin/env python3
"""Sum a rocprofv3 counter-collection CSV tree into per-counter totals.

usage: _pmc_sum.py <dir> [tag]

Prints one line per counter: tag counter total dispatches mean. The byte units are
the ones calibrated against a known-size copy kernel (FetchSize x 2048, WriteSize x
1024), so the total column converts straight to absolute DRAM bytes.
"""
import collections
import csv
import glob
import sys


def main():
    root = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "-"
    tot = collections.defaultdict(float)
    disp = collections.defaultdict(set)
    for fn in glob.glob(root + "/**/*.csv", recursive=True):
        with open(fn) as fh:
            for row in csv.DictReader(fh):
                name = row.get("Counter_Name")
                if name is None:
                    continue
                tot[name] += float(row.get("Counter_Value", 0) or 0)
                disp[name].add(row.get("Dispatch_Id"))
    for name in sorted(tot):
        n = max(1, len(disp[name]))
        print(f"{tag} {name} total={tot[name]:.6g} disp={n} mean={tot[name] / n:.6g}")


if __name__ == "__main__":
    main()
