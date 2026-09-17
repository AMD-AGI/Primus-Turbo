#!/usr/bin/env python3
"""Steady-state tps/mfu from an e2e log, same convention as E2E-AB.md: step >= 8, median.

Reports dispersion as (max-min)/median so a run whose noise floor moved can be told apart
from one whose mean moved -- the 0915 A/B called 6.78% significant against a 0.43% floor,
and that comparison is only meaningful if the floor is re-derived per run, not assumed.
"""
import re, sys, statistics

STEP = re.compile(r"step:\s*(\d+)\b")
NUM  = lambda key, s: (lambda m: float(m.group(1).replace(",", "")) if m else None)(
    re.search(key + r":\s*([\d,]+\.?\d*)", s))

def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP.search(line)
        if not m:
            continue
        rows.append(dict(step=int(m.group(1)), loss=NUM("loss", line), tps=NUM("tps", line),
                         mfu=NUM("mfu", line), mem=NUM("memory", line)))
    return rows

for path in sys.argv[1:]:
    rows = parse(path)
    if not rows:
        print(f"{path}: no step lines"); continue
    st = [r for r in rows if r["step"] >= 8 and r["tps"]]
    print(f"\n=== {path}")
    print(f"steps parsed      {len(rows)}  (last step {rows[-1]['step']})")
    print(f"loss              {rows[0]['loss']} -> {rows[-1]['loss']}")
    mem = [r["mem"] for r in rows if r["mem"]]
    if mem: print(f"peak memory GiB   {max(mem)}")
    if not st:
        print("no steady-state steps (>=8)"); continue
    t = sorted(r["tps"] for r in st)
    f = sorted(r["mfu"] for r in st if r["mfu"])
    med = statistics.median(t)
    print(f"steady n          {len(st)}  (steps 8..{rows[-1]['step']})")
    print(f"steady tps median {med:.0f}   range {t[0]:.0f}-{t[-1]:.0f}"
          f"   dispersion {(t[-1]-t[0])/med*100:.2f}%")
    if f: print(f"steady mfu median {statistics.median(f):.2f}%")
