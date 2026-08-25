#!/usr/bin/env python3
"""Print ONE backward's dispatch timeline from a rocprofv3 kernel_trace.csv.

usage: _probe_trace.py <kernel_trace.csv> [iter_from_end]
"""
import csv
import sys


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    back = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ev = [
        (int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"][:34], r.get("Queue_Id", "?"))
        for r in rows
    ]
    ev.sort()
    starts = [i for i, e in enumerate(ev) if "odo" in e[2]]
    print("dispatches=%d odo_marks=%d" % (len(ev), len(starts)))
    a, b = starts[-back], starts[-back + 1]
    t0 = ev[a][0]
    prev_end = t0
    print("---- one backward: %d dispatches, span %.1f us" % (b - a, (ev[b - 1][1] - t0) / 1e3))
    tot = {}
    for s, e, n, q in ev[a:b]:
        gap = (s - prev_end) / 1e3
        prev_end = max(prev_end, e)
        key = n.split("_kernel")[0]
        tot[key] = tot.get(key, 0.0) + (e - s) / 1e3
        print(
            "  q%-4s %8.1f -> %8.1f  dur %7.1f  gap %+7.1f  %s"
            % (q, (s - t0) / 1e3, (e - t0) / 1e3, (e - s) / 1e3, gap, n)
        )
    print("---- GPU time per kernel (us)")
    for k, v in sorted(tot.items(), key=lambda x: -x[1]):
        print("  %-34s %9.1f" % (k, v))


if __name__ == "__main__":
    main()
