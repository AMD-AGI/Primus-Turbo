"""ab_summary.py TAG [BASE_NAME]: per variant and shape, the fwd+bwd ms of every rep of ab.sh TAG, their mean, the
step TFLOPS geomean over shapes (of the per-shape mean ms), and each variant's change against BASE_NAME (default:
the first variant found)."""
import collections, glob, os, re, statistics, sys

E = "/home/xiaompen/mega_scratch/l2c/logs"
tag = sys.argv[1]
pat = re.compile(rf"ab_{re.escape(tag)}_(.+)_(dsv3_\w+?)_r(\d+)\.log$")
ms = collections.defaultdict(lambda: collections.defaultdict(dict))
flops = {}
order = []
for f in sorted(glob.glob(f"{E}/ab_{tag}_*_r*.log"), key=os.path.getmtime):
    m = pat.search(os.path.basename(f))
    if not m:
        continue
    name, shape, rep = m.group(1), m.group(2), int(m.group(3))
    txt = open(f).read()
    r = re.search(r"^(dsv3\S+)\s+(PASS|FAIL).*?fwd ([\d.]+) ms.*?fwd\+bwd ([\d.]+) ms.*?step ([\d.]+) TF", txt, re.M)
    if not r:
        ms[name][shape][rep] = None
        continue
    if r.group(2) != "PASS":
        print(f"!! {name} {shape} r{rep} {r.group(2)}")
    t, st = float(r.group(4)), float(r.group(5))
    ms[name][shape][rep] = t
    flops[shape] = st * t  # step TF * ms is the shape's constant
    if name not in order:
        order.append(name)
base = sys.argv[2] if len(sys.argv) > 2 else order[0]
shapes = sorted({s for n in ms for s in ms[n]})
agg = {}
for n in order:
    cells, means = [], {}
    for s in shapes:
        v = [x for _, x in sorted(ms[n][s].items()) if x is not None]
        miss = sum(x is None for x in ms[n][s].values())
        if v:
            means[s] = statistics.mean(v)
        cells.append(f"{s}: " + " / ".join(f"{x:.3f}" for x in v) + (f" (+{miss} failed)" if miss else "")
                     + (f" -> {means[s]:.3f}" if v else ""))
    if len(means) == len(shapes):
        agg[n] = (statistics.geometric_mean(flops[s] / means[s] for s in shapes), means)
    print(f"{n:14s} " + " | ".join(cells))
for n in order:
    if n not in agg:
        continue
    tf, means = agg[n]
    line = f"{n:14s} step geomean {tf:.3f} TF"
    if n != base and base in agg:
        b = agg[base]
        line += f"  vs {base}: {100 * (tf / b[0] - 1):+.3f}%  (" + ", ".join(
            f"{s.replace('dsv3_', '')} {100 * (b[1][s] / means[s] - 1):+.2f}%" for s in shapes) + ")"
    print(line)
