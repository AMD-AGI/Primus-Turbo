"""kt_compare.py TAG BASE OTHER...: per-kernel mean us (all ranks, all reps, last 8 iterations) of the kt.sh traces
mega_scratch/l2c/kt/TAG_<variant>_r*, in launch order, with each variant's delta against BASE."""
import collections, csv, glob, re, sys

ROOT = "/home/xiaompen/mega_scratch/l2c/kt"
ITERS = 8


def short(n):
    return re.sub(r"\(.*", "", n)[:60]


def per_kernel(path):
    rows = [r for r in csv.DictReader(open(path)) if r["Kind"] == "KERNEL_DISPATCH"]
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    starts = [i for i, r in enumerate(rows) if "prologue" in r["Kernel_Name"].lower()]
    fwd = starts[::2] if len(starts) >= 2 * (ITERS + 3) else starts
    bounds = fwd[-ITERS:] + [len(rows)]
    per = collections.OrderedDict()
    busy = []
    for a, b in zip(bounds, bounds[1:]):
        seg = rows[a:b]
        busy.append(sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in seg) / 1e3)
        seen = collections.Counter()
        for r in seg:
            k = short(r["Kernel_Name"])
            seen[k] += 1
            per.setdefault((k, seen[k]), []).append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
    per[("GPU busy / iteration", 1)] = busy
    return per


tag, names = sys.argv[1], sys.argv[2:]
data = {}
for n in names:
    acc = collections.OrderedDict()
    files = sorted(glob.glob(f"{ROOT}/{tag}_{n}_r*/**/*kernel_trace.csv", recursive=True))
    for f in files:
        for k, v in per_kernel(f).items():
            acc.setdefault(k, []).extend(v)
    data[n] = (acc, len(files))
base = data[names[0]][0]
print("traces: " + ", ".join(f"{n} {data[n][1]}" for n in names))
print(f"{'kernel':<64}" + "".join(f"{n:>18}" for n in names))
for k, v in base.items():
    if len(v) < ITERS:
        continue
    b = sum(v) / len(v)
    line = f"{k[0] + ' #' + str(k[1]):<64}{b:18.1f}"
    for n in names[1:]:
        o = data[n][0].get(k)
        if not o:
            line += f"{'-':>18}"
            continue
        m = sum(o) / len(o)
        line += f"{m:10.1f} {100 * (m / b - 1):+6.2f}%"
    print(line)
