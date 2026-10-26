# python trace_summary.py trace.json.gz : GPU kernel time per profiled step, by kernel and category
import collections
import gzip
import json
import re
import sys

with gzip.open(sys.argv[1]) as f:
    events = json.load(f)["traceEvents"]
steps = [e for e in events if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep")]
kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
num_steps = max(1, len(steps))

CATS = [
    ("deepep", r"dispatch_kernel|combine_kernel|notify_dispatch|notify_combine|dispatch_layout"),
    ("rccl", r"nccl|rccl"),
    ("grouped_gemm", r"grouped|gmm"),
    ("attention", r"attn|mla|fmha|flash"),
    ("gemm", r"Cijk|gemm|matmul|hipblaslt"),
]


def category(name):
    for cat, pattern in CATS:
        if re.search(pattern, name, re.I):
            return cat
    return "other"


by_name = collections.Counter()
by_cat = collections.Counter()
count = collections.Counter()
for k in kernels:
    by_name[k["name"][:90]] += k["dur"]
    by_cat[category(k["name"])] += k["dur"]
    count[k["name"][:90]] += 1

# busy time: union of kernel intervals over all streams
intervals = sorted((k["ts"], k["ts"] + k["dur"]) for k in kernels)
busy, end = 0.0, -1e30
for s, e in intervals:
    if s > end:
        busy += e - s
        end = e
    elif e > end:
        busy += e - end
        end = e
span = (intervals[-1][1] - intervals[0][0]) if intervals else 0
print(f"steps={num_steps} span/step={span / num_steps / 1e3:.1f} ms busy/step={busy / num_steps / 1e3:.1f} ms")
print("by category (ms/step, kernel time summed over streams):")
for cat, t in by_cat.most_common():
    print(f"  {cat:14s} {t / num_steps / 1e3:8.1f}")
print("top kernels (ms/step, calls/step):")
for name, t in by_name.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 15):
    print(f"  {t / num_steps / 1e3:8.1f} {count[name] / num_steps:6.0f}  {name}")
