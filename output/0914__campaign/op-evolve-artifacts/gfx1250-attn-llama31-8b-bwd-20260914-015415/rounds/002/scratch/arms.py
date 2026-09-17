# Multi-arm timing, using ONLY the job's sanctioned paths:
#   precision  -> ut/correctness.check_shape (what validation.py gates on)
#   speed      -> benchmark.measure          (the only timing loop in this job)
# Palindromic order so no arm gets a systematically better slot.
import os, sys, json, statistics
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0, os.path.join(JC, "op")); sys.path.insert(0, os.path.join(JC, "op", "ut"))
import benchmark, correctness
from shapes import ALL_SHAPES

shape = [s for s in ALL_SHAPES if s["name"] == "b4_s8192_hq32_hkv8_d128"][0]
dirs = json.loads(os.environ["ARMS"])      # [[label, path], ...]
order = json.loads(os.environ["ORDER"])    # palindromic label list
path = dict(dirs)

for lab, p in dirs:
    ok, s = correctness.check_shape(p, shape)
    print(f"SQNR {lab:<6} out={s['out']:.1f} dq={s['dq']:.1f} dk={s['dk']:.1f} dv={s['dv']:.1f} "
          f"{'pass' if ok else 'FAIL'}", flush=True)

res = {}
for lab in order:
    r = benchmark.measure(path[lab], shape)
    res.setdefault(lab, []).append(r["bwd_ms"])
    print(f"TIME {lab:<6} bwd_ms={r['bwd_ms']:.4f} fwd_ms={r['fwd_ms']:.4f} "
          f"tf={r['bwd_tflops']:.1f} spread={r['bwd_spread_pct']:.2f}%", flush=True)

print("--- mean of the two slots per arm ---", flush=True)
base = None
for lab, _ in dirs:
    m = sum(res[lab]) / len(res[lab])
    if base is None: base = m
    print(f"{lab:<6} {m:.4f} ms  x_vs_{dirs[0][0]}={base/m:.4f}  slots={['%.4f' % x for x in res[lab]]}")
