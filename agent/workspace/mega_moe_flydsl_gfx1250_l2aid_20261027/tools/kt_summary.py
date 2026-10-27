"""kt_summary.py TRACE.csv [ITERS=8]: per-kernel mean duration (us) over the last ITERS fwd+bwd iterations of a
prof_ss.py kernel trace (one agent), in launch order, plus the per-iteration GPU-busy and span totals."""
import csv, collections, re, sys

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["Kind"] == "KERNEL_DISPATCH"]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
# an iteration starts at the forward dispatch prologue
starts = [i for i, r in enumerate(rows) if "prologue" in r["Kernel_Name"].lower()]
fwd_starts = starts[::2] if len(starts) >= 2 * (iters + 3) else starts
last = fwd_starts[-iters:]
bounds = last + [len(rows)]


def short(n):
    n = re.sub(r"\(.*", "", n)
    return n[:90]


per = collections.OrderedDict()
busy, span = [], []
for a, b in zip(bounds, bounds[1:]):
    seg = rows[a:b]
    busy.append(sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in seg) / 1e3)
    span.append((int(seg[-1]["End_Timestamp"]) - int(seg[0]["Start_Timestamp"])) / 1e3)
    seen = collections.Counter()
    for r in seg:
        k = short(r["Kernel_Name"])
        seen[k] += 1
        key = (k, seen[k], r["Grid_Size_X"])
        per.setdefault(key, []).append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
print(f"{len(starts)} prologue launches, using the last {len(last)} iterations")
print(f"{'us mean':>9} {'min':>8} {'max':>8}  grid      kernel (#nth in iteration)")
for (k, n, g), v in per.items():
    if len(v) < len(last):
        continue
    print(f"{sum(v) / len(v):9.1f} {min(v):8.1f} {max(v):8.1f}  {g:>8}  {k} #{n}")
print(f"iteration GPU busy {sum(busy) / len(busy):.1f} us, span {sum(span) / len(span):.1f} us")
