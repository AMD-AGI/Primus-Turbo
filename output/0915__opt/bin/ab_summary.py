#!/usr/bin/env python3
"""Summarise the replicated A/B: BETWEEN-RUN variance, which is the floor the effect must clear.

The first 8-layer A/B quoted within-run step jitter (1.20% / 0.77%) as its noise floor and
called a 3.42% separation significant against it. That is the wrong variance: it describes how
steady one process is, not how reproducible two processes are. This reports both, and the
per-replicate values, so a monotonic thermal drift across replicates stays visible instead of
being averaged into the mean.
"""
import glob, os, re, statistics, sys

STEP = re.compile(r"step:\s*(\d+)\b")

def steady(path):
    tps, loss1, mem = [], None, []
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP.search(line)
        if not m:
            continue
        step = int(m.group(1))
        t = re.search(r"tps:\s*([\d,]+)", line)
        l = re.search(r"loss:\s*([\d.]+)", line)
        g = re.search(r"memory:\s*([\d.]+)GiB", line)
        if step == 1 and l: loss1 = float(l.group(1))
        if g: mem.append(float(g.group(1)))
        if step >= 8 and t: tps.append(float(t.group(1).replace(",", "")))
    return tps, loss1, (max(mem) if mem else None)

base = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0915__opt/logs"
arms = {"on": [], "off": []}
print(f"{'run':10} {'steady n':>8} {'median tps':>11} {'within%':>8} {'step1 loss':>11} {'peak GiB':>9}")
for r in range(1, 10):
    for arm in ("on", "off"):
        p = f"{base}/e2e.rep{r}-{arm}.log"
        if not os.path.exists(p): continue
        tps, l1, mem = steady(p)
        if not tps: 
            print(f"rep{r}-{arm:4}   (no steady steps -- run incomplete or failed)"); continue
        med = statistics.median(tps)
        arms[arm].append(med)
        print(f"rep{r}-{arm:<6} {len(tps):>8} {med:>11.0f} "
              f"{(max(tps)-min(tps))/med*100:>7.2f}% {l1 if l1 else float('nan'):>11.5f} {mem:>9.2f}")

print()
for arm in ("on", "off"):
    v = arms[arm]
    if not v: continue
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    print(f"{arm.upper():4} n={len(v)}  mean {statistics.mean(v):.1f}  "
          f"between-run sd {sd:.1f} ({sd/statistics.mean(v)*100:.2f}%)  values {[round(x) for x in v]}")

if len(arms["on"]) > 1 and len(arms["off"]) > 1:
    mon, moff = statistics.mean(arms["on"]), statistics.mean(arms["off"])
    son, soff = statistics.stdev(arms["on"]), statistics.stdev(arms["off"])
    n1, n2 = len(arms["on"]), len(arms["off"])
    eff = (mon - moff) / moff * 100
    # pooled sd and Welch-style sem on the difference
    sem = (son**2 / n1 + soff**2 / n2) ** 0.5
    print(f"\nEFFECT  ON vs OFF: {eff:+.2f}%   "
          f"({mon:.0f} vs {moff:.0f}, diff {mon-moff:+.0f} tps, sem {sem:.0f} tps)")
    if sem > 0:
        print(f"        separation = {abs(mon-moff)/sem:.1f} x sem of the difference")
    print(f"        between-run sd is {max(son/mon, soff/moff)*100:.2f}% of throughput; "
          f"effect is {abs(eff)/ (max(son/mon, soff/moff)*100):.1f}x that")
    print("\n  Prior (unreplicated, unseeded, order-confounded) single pair said ON +3.42%.")
