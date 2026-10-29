###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Kernel-only times and counters of the DeepEP kernels from rocprofv3 CSV output; no GPU, standard library only.
#
#   python3 rocprof_summary.py trace <dir with *_kernel_trace.csv> [--match REGEX]
#   python3 rocprof_summary.py pmc <dir with *_counter_collection.csv> [--match REGEX]
#
# trace: per kernel and workgroup count, the median duration of each process (rank), and the slowest rank's
#        median; the DeepEP V2 tests time kernels the same way (PyTorch profiler, kernel only).
# pmc:   per kernel, workgroup count and counter, the median over dispatches of each rank (rocprofv3 sums
#        the counter over the GPU's instances); FETCH_SIZE and WRITE_SIZE are in KiB.
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict


def rows(directory, suffix):
    for path in sorted(glob.glob(os.path.join(directory, "**", f"*{suffix}"), recursive=True)):
        rank = os.path.basename(path).split("_")[0]
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                yield rank, row


def short(name):
    return re.sub(r"\(.*", "", name).strip()


def workgroups(row):
    return (
        int(row["Grid_Size_X"]) // max(int(row["Workgroup_Size_X"]), 1)
        if "Grid_Size_X" in row
        else int(row.get("Grid_Size", 0)) // max(int(row.get("Workgroup_Size", 1)), 1)
    )


def trace(directory, match):
    durations = defaultdict(lambda: defaultdict(list))
    for rank, row in rows(directory, "kernel_trace.csv"):
        name = short(row["Kernel_Name"])
        if not re.search(match, name):
            continue
        us = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e3
        durations[(name, workgroups(row))][rank].append(us)
    print("| kernel | workgroups | launches per rank | median µs per rank | slowest rank µs | p10–p90 µs |")
    print("|---|---:|---:|---|---:|---|")
    for (name, wg), per_rank in sorted(durations.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        medians = {r: statistics.median(v) for r, v in per_rank.items()}
        allv = sorted(x for v in per_rank.values() for x in v)
        p10, p90 = allv[len(allv) // 10], allv[(len(allv) * 9) // 10]
        launches = min(len(v) for v in per_rank.values())
        print(
            f"| {name} | {wg} | {launches} | {', '.join(f'{m:.1f}' for m in medians.values())} | "
            f"{max(medians.values()):.1f} | {p10:.1f}–{p90:.1f} |"
        )


def pmc(directory, match):
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for rank, row in rows(directory, "counter_collection.csv"):
        name = short(row["Kernel_Name"])
        if not re.search(match, name):
            continue
        key = (name, workgroups(row), row["Counter_Name"])
        values[key][rank][row.get("Dispatch_Id") or row.get("Correlation_Id")] += float(row["Counter_Value"])
    print("| kernel | workgroups | counter | median per dispatch, each rank | dispatches |")
    print("|---|---:|---|---|---:|")
    for (name, wg, counter), per_rank in sorted(values.items()):
        meds = [statistics.median(d.values()) for d in per_rank.values()]
        n = min(len(d) for d in per_rank.values())
        print(f"| {name} | {wg} | {counter} | {', '.join(f'{m:,.0f}' for m in meds)} | {n} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("trace", "pmc"))
    p.add_argument("directory")
    p.add_argument("--match", default="_v2_kernel|notify|layout", help="regex on the kernel name")
    a = p.parse_args()
    (trace if a.mode == "trace" else pmc)(a.directory, a.match)


if __name__ == "__main__":
    main()
