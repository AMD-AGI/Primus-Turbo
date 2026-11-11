"""Summarize an rocprofv3 ATT stats CSV for one traced CU, normalized per q-block.

Usage: python att_summary.py <stats_ui_output_*.csv> <label> <q_blocks_per_wave> [waves_per_wg=8]
  q_blocks_per_wave: 1 for FlyDSL (one q-block per WG), 2 for opus head/tail merge.
Instructions hit once per wave (or less) are per-q-block fixed cost ("fixed"); instructions hit
more than once per wave belong to a KV loop ("loop").
"""

import csv
import re
import sys
from collections import defaultdict

path, label = sys.argv[1], sys.argv[2]
qb_per_wave = int(sys.argv[3]) if len(sys.argv) > 3 else 1
waves_per_wg = int(sys.argv[4]) if len(sys.argv) > 4 else 8

rows = []
for r in csv.DictReader(open(path)):
    ins = r["Instruction"].strip()
    if not ins or ins.startswith(";"):
        continue
    rows.append(
        dict(va=int(r["Vaddr"]), ins=ins, hit=int(r["Hitcount"]), lat=int(r["Latency"]),
             stall=int(r["Stall"]), idle=int(r["Idle"]))
    )
rows.sort(key=lambda r: r["va"])
waves = next(r["hit"] for r in rows if r["hit"] > 0)
qblocks = waves // waves_per_wg * qb_per_wave


def klass(ins):
    op = ins.split()[0]
    if op.startswith("v_mfma"):
        return "MFMA"
    if op.startswith("ds_"):
        return "LDS"
    if op.startswith(("buffer_", "global_")):
        return "VMEM"
    if op.startswith("scratch_"):
        return "SCRATCH"
    if op.startswith("s_waitcnt"):
        return "WAITCNT"
    if op.startswith("s_barrier"):
        return "BARRIER"
    if op.startswith("s_nop"):
        return "NOP"
    if op.startswith(("s_cbranch", "s_branch")):
        return "BRANCH"
    if op.startswith(("s_load", "s_buffer_load")):
        return "SMEM"
    if op.startswith(("s_setprio", "s_sleep")):
        return "MISC"
    if op.startswith("s_"):
        return "SALU"
    if re.match(r"v_(exp|log|rcp|rsq|sqrt|sin|cos)_", op):
        return "TRANS"
    if op.startswith("v_"):
        return "VALU"
    return "OTHER"


def region(r):
    return "loop" if r["hit"] > 1.2 * waves else "fixed"


cls = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # region -> class -> [insts, lat, stall]
for r in rows:
    for g in (region(r), "all"):
        t = cls[g][klass(r["ins"])]
        t[0] += r["hit"]
        t[1] += r["lat"]
        t[2] += r["stall"]

mfma = defaultdict(int)
for r in rows:
    if r["ins"].startswith("v_mfma"):
        mfma[r["ins"].split()[0]] += r["hit"]


def tot(g, i):
    return sum(t[i] for t in cls[g].values())


print(f"=== {label}: traced CU waves {waves}, q-blocks {qblocks} ===")
print("MFMA mix per q-block:", {k: round(v / qblocks) for k, v in mfma.items()})
print(f"{'':8s} {'cycles/qb':>10s} {'stall/qb':>10s} {'stall%':>7s}")
for g in ("fixed", "loop", "all"):
    print(f"{g:8s} {tot(g, 1) / qblocks:10,.0f} {tot(g, 2) / qblocks:10,.0f} {100 * tot(g, 2) / max(1, tot(g, 1)):7.1f}")
for g in ("fixed", "loop"):
    print(f"-- {g}: per q-block (sum over its 8 waves) --")
    print(f"   {'class':8s} {'insts':>8s} {'cycles':>10s} {'%':>5s} {'stall':>10s} {'%':>5s}")
    L, S = tot(g, 1), tot(g, 2)
    for k, t in sorted(cls[g].items(), key=lambda kv: -kv[1][1]):
        if t[1] < 0.005 * L:
            continue
        print(f"   {k:8s} {t[0] / qblocks:8,.0f} {t[1] / qblocks:10,.0f} {100 * t[1] / L:5.1f} "
              f"{t[2] / qblocks:10,.0f} {100 * t[2] / max(1, S):5.1f}")
print("top 12 stall instructions (stall cycles per q-block):")
for r in sorted(rows, key=lambda r: -r["stall"])[:12]:
    print(f"  {r['va']:#8x} {region(r):5s} hit/wave {r['hit'] / waves:5.1f} stall/qb {r['stall'] / qblocks:8,.0f}  "
          f"{r['ins'][:64]}")
