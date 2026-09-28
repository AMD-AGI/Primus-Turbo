#!/usr/bin/env python3
"""Ruler probe 3: in ONE process, (a) the harness loop with each sample labelled by its
predecessor arm, (b) a BLOCKED loop: per round each arm runs `lead` untimed calls then `blk`
timed calls back to back (so every timed sample is preceded by the same arm), rounds palindromic."""
import os, sys, json, time, argparse
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"
OP = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op"
sys.path.insert(0, OP)
import benchmark as B
from common import load_impl, make_inputs
import torch

def med(ts):
    ts = sorted(ts); n = len(ts)
    return ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])

ap = argparse.ArgumentParser()
ap.add_argument("--arm-path", action="append", default=[])
ap.add_argument("--shape", default="prod")
ap.add_argument("--iters", type=int, default=101)
ap.add_argument("--warmup-seconds", type=float, default=8.0)
ap.add_argument("--rounds", type=int, default=12)
ap.add_argument("--lead", type=int, default=4)
ap.add_argument("--blk", type=int, default=9)
ap.add_argument("--flush-mb", type=int, default=256)
ap.add_argument("--json")
a = ap.parse_args()
B.L2_FLUSH_MB = a.flush_mb
arms = [(s.partition("=")[0], s.partition("=")[2]) for s in a.arm_path]
fns = {l: load_impl(p) for l, p in arms}
labels = [l for l, _ in arms]
print(f"# PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} arms={labels}", flush=True)
q, k, v = make_inputs(a.shape, seed=0)
flush = torch.empty(B.L2_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
call = lambda l: fns[l](q, k, v, causal=True)
for l in labels: call(l)
torch.cuda.synchronize()
for l in labels:
    t_end = time.perf_counter() + a.warmup_seconds
    while time.perf_counter() < t_end: call(l)
    torch.cuda.synchronize()
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
def timed(l):
    flush.zero_(); ev0.record(); call(l); ev1.record(); ev1.synchronize()
    return ev0.elapsed_time(ev1)
rows = []
# (a) harness loop, predecessor-labelled
s0 = B.sclk_mhz(); prev = None
std = {l: [] for l in labels}; bypred = {}
for i in range(a.iters):
    for l in (labels if i % 2 == 0 else labels[::-1]):
        t = timed(l); std[l].append(t); bypred.setdefault((l, prev), []).append(t); prev = l
s1 = B.sclk_mhz()
for l in labels:
    r = {"mode": "std", "arm": l, "ms": round(med(std[l]), 5), "sclk": [s0, s1]}
    r["by_pred"] = {str(p): [round(med(ts), 5), len(ts)] for (x, p), ts in bypred.items() if x == l}
    rows.append(r)
    print("RESULT mode=std arm=%s ms=%.5f sclk=%s/%s by_pred=%s" % (l, r["ms"], s0, s1, r["by_pred"]), flush=True)
# (b) blocked
s0 = B.sclk_mhz(); blk = {l: [] for l in labels}; pos = {l: [[] for _ in range(a.lead + a.blk)] for l in labels}
strm = {l: [] for l in labels}
for rd in range(a.rounds):
    for l in (labels if rd % 2 == 0 else labels[::-1]):
        for j in range(a.lead):
            pos[l][j].append(timed(l))
        for j in range(a.blk):
            t = timed(l); blk[l].append(t); pos[l][a.lead + j].append(t)
        # (c) stream: the same arm back to back, no host sync between calls, one event pair
        flush.zero_(); ev0.record()
        for _ in range(a.blk):
            call(l)
        ev1.record(); ev1.synchronize()
        strm[l].append(ev0.elapsed_time(ev1) / a.blk)
s1 = B.sclk_mhz()
for l in labels:
    r = {"mode": "blocked", "arm": l, "ms": round(med(blk[l]), 5), "n": len(blk[l]), "sclk": [s0, s1],
         "by_pos": [round(med(p), 5) for p in pos[l]]}
    rows.append(r)
    print("RESULT mode=blocked arm=%s ms=%.5f n=%d sclk=%s/%s by_pos=%s" % (l, r["ms"], r["n"], s0, s1, r["by_pos"]), flush=True)
for l in labels:
    r = {"mode": "stream", "arm": l, "ms": round(med(strm[l]), 5), "n": len(strm[l])}
    rows.append(r)
    print("RESULT mode=stream arm=%s ms=%.5f n=%d" % (l, r["ms"], r["n"]), flush=True)
if a.json: open(a.json, "w").write(json.dumps(rows, indent=2))
