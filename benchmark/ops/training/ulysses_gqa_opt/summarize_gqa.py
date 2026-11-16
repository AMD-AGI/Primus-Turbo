#!/usr/bin/env python3
"""Per-run summary of the GQA end-to-end matrix: step time (median of iterations 3+), TFLOP/s/GPU,
last-iteration loss and peak allocated memory, plus the change of MegaUlysses vs unfused Ulysses
and vs stock Megatron CP.

    python3 summarize_gqa.py OUT_DIR   (reads OUT_DIR/logs/g_*.log written by run_e2e_gqa.sh)
"""
import glob
import os
import re
import statistics
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
IT = re.compile(
    r"iteration\s+(\d+)/\s*\d+ .*?elapsed time per iteration \(ms\): ([0-9.]+)\S* \| "
    r"(?:throughput|compute) per GPU \(TFLOP/s/GPU\): ([0-9.]+).*?lm loss: ([0-9.E+-]+)"
)
MAXALLOC = re.compile(r"max allocated: ([0-9.]+)")
NAME = re.compile(r"g_(.+)_cp(\d)_s(\d+)_u(-?\d)(_\w+)?\.log")

runs = {}
for f in sorted(glob.glob(f"{R}/logs/g_*.log")):
    m = NAME.match(os.path.basename(f))
    if not m:
        continue
    text = ANSI.sub("", open(f, errors="ignore").read())
    its = {}
    for x in IT.finditer(text):
        its.setdefault(int(x[1]), (float(x[2]), float(x[3]), float(x[4])))
    ks = sorted(its)
    key = (m[1], int(m[2]), int(m[3]), m[5] or "")
    if len(ks) < 4:
        err = "OutOfMemory" if "out of memory" in text.lower() else ("Error" if "Error" in text else "incomplete")
        runs.setdefault(key, {})[m[4]] = err
        continue
    rest = ks[2:]
    alloc = [float(a) for a in MAXALLOC.findall(text)]
    runs.setdefault(key, {})[m[4]] = (
        statistics.median(its[k][0] for k in rest),
        statistics.median(its[k][1] for k in rest),
        its[ks[-1]][2],
        max(alloc) / 1024 if alloc else float("nan"),
    )

def cell(v):
    return f"{v[0]:8.0f}" if isinstance(v, tuple) else f"{v:>8}"

print(f"{'model':30} {'cp':>2} {'seq':>5} {'stock':>8} {'unfused':>8} {'fused':>8} {'vs unf':>7} {'vs stock':>8}"
      f" {'TFLOPs unf->fused':>18} {'loss unf/fused':>16} {'mem unf/fused GB':>17}")
for (model, cp, seq, var), modes in sorted(runs.items()):
    s, u, f = modes.get("0"), modes.get("-1"), modes.get("1")
    vu = f"{(f[0] / u[0] - 1) * 100:+6.1f}%" if isinstance(u, tuple) and isinstance(f, tuple) else "      -"
    vs = f"{(f[0] / s[0] - 1) * 100:+7.1f}%" if isinstance(s, tuple) and isinstance(f, tuple) else "       -"
    tf = f"{u[1]:.0f}->{f[1]:.0f}" if isinstance(u, tuple) and isinstance(f, tuple) else "-"
    lo = f"{u[2]:.4f}/{f[2]:.4f}" if isinstance(u, tuple) and isinstance(f, tuple) else "-"
    me = f"{u[3]:.1f}/{f[3]:.1f}" if isinstance(u, tuple) and isinstance(f, tuple) else "-"
    print(f"{model + var:30} {cp:>2} {seq // 1024:>4}K {cell(s) if s else '       -'} {cell(u) if u else '       -'}"
          f" {cell(f) if f else '       -'} {vu:>7} {vs:>8} {tf:>18} {lo:>16} {me:>17}")
