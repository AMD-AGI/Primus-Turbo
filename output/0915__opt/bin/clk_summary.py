"""Summarise a run's clock/power/temperature trace.

Added after a day in which every discussion of "is this box throttling?" had no evidence on
either side. The first sampler wrote empty columns because its field names were guessed; this
reads what rocm-smi on this build actually emits. Idle here is 1100 MHz and loaded is lower,
so the sustained clock is a variable of every experiment run on this card.
"""
import csv, sys, statistics as st

rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        try:
            rows.append((int(r["t"]), int(r["sclk_mhz"]), float(r["power_w"]), float(r["tjunction_c"])))
        except (ValueError, TypeError):
            continue          # a poll that raced the driver writes blanks; drop, do not zero
if not rows:
    print("no usable samples"); raise SystemExit(1)

# Idle samples sit at the nominal clock with low power; treat the loaded population separately,
# because averaging the two produces a number no part of the run ever ran at.
loaded = [r for r in rows if r[2] > 1200]
idle = [r for r in rows if r[2] <= 1200]
print(f"{len(rows)} 个样本，跨度 {rows[-1][0]-rows[0][0]} s")
for name, pop in (("空载", idle), ("负载", loaded)):
    if not pop:
        continue
    c = [r[1] for r in pop]; p = [r[2] for r in pop]; t = [r[3] for r in pop]
    sd = st.pstdev(c)/st.mean(c)*100 if len(c) > 1 else 0.0
    print(f"  {name:<4} n={len(pop):<4} sclk {st.mean(c):7.1f} MHz (min {min(c)}, max {max(c)}, sd {sd:.2f}%)"
          f"  power {st.mean(p):7.1f} W  Tj {st.mean(t):5.1f}→{max(t):.0f}°C")
if loaded:
    c = [r[1] for r in loaded]
    print(f"\n  负载下相对 1100 MHz 标称: {st.mean(c)/1100*100:.1f}%"
          f"   末段 vs 首段: {st.mean(c[-5:])/st.mean(c[:5])*100:.1f}%  (<100% 表示随温度继续下滑)")
