#!/usr/bin/env python3
"""Ruler probe 4 (GPU0 session, 2026-09-28). One process, one shape, modes in the order given:
  std      the job harness loop verbatim (palindromic call-by-call interleave), every sample
           labelled by its predecessor arm;
  blocked  per round each arm runs `lead` untimed + `blk` timed calls back to back, rounds
           palindromic (= fix/benchmark.py --block/--lead).
Imports the job harness read-only for sclk / flush size / make_inputs / load_impl."""
import os, sys, json, time, argparse, hashlib
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
ap.add_argument("--modes", default="std,blocked")
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
for l, p in arms:
    mod = sys.modules[fns[l].__module__]
    kf = getattr(getattr(mod, "_kern", None), "__file__", None)
    kmd5 = hashlib.md5(open(kf, "rb").read()).hexdigest()[:8] if kf else "-"
    print(f"# arm {l} -> {p} kern={kf} md5={kmd5}", flush=True)
print(f"# PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} arms={labels} modes={a.modes}", flush=True)
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
for mode in a.modes.split(","):
    s0 = B.sclk_mhz()
    if mode == "std":
        prev = None; ts = {l: [] for l in labels}; bypred = {}
        for i in range(a.iters):
            for l in (labels if i % 2 == 0 else labels[::-1]):
                t = timed(l); ts[l].append(t); bypred.setdefault((l, prev), []).append(t); prev = l
    elif mode == "stream":   # production-like: lead untimed, then blk calls back to back, one event pair, no host sync
        ts = {l: [] for l in labels}; bypred = {}
        for rd in range(a.rounds):
            for l in (labels if rd % 2 == 0 else labels[::-1]):
                for _ in range(a.lead): call(l)
                flush.zero_(); ev0.record()
                for _ in range(a.blk): call(l)
                ev1.record(); ev1.synchronize(); ts[l].append(ev0.elapsed_time(ev1) / a.blk)
    else:
        ts = {l: [] for l in labels}; bypred = {}
        for rd in range(a.rounds):
            for l in (labels if rd % 2 == 0 else labels[::-1]):
                for _ in range(a.lead): timed(l)
                for _ in range(a.blk): ts[l].append(timed(l))
    s1 = B.sclk_mhz()
    for pos, l in enumerate(labels):
        r = {"mode": mode, "arm": l, "pos": pos, "ms": round(med(ts[l]), 5), "n": len(ts[l]), "sclk": [s0, s1],
             "by_pred": {str(p): [round(med(v), 5), len(v)] for (x, p), v in bypred.items() if x == l}}
        rows.append(r)
        print("RESULT mode=%s arm=%s pos=%d ms=%.5f sclk=%s/%s by_pred=%s" % (mode, l, pos, r["ms"], s0, s1, r["by_pred"]), flush=True)
if a.json: open(a.json, "w").write(json.dumps(rows, indent=2))
