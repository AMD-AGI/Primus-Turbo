#!/usr/bin/env python3
"""Ruler probe: harness-identical timing (std) + host-latency-shielded timing (shield) + host
dispatch time per arm, all in one process. Imports the job harness read-only."""
import os, sys, json, time, argparse
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"
OP = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op"
sys.path.insert(0, OP)
import benchmark as B          # sclk_mhz, measure, counts, witness
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
ap.add_argument("--shield", type=int, default=1)
ap.add_argument("--json")
a = ap.parse_args()
arms = [(s.partition("=")[0], s.partition("=")[2]) for s in a.arm_path]
fns = {l: load_impl(p) for l, p in arms}
labels = [l for l, _ in arms]
info = {}
for l, p in arms:
    mod = sys.modules[fns[l].__module__]
    info[l] = {"path": p, "impl_mod": mod.__name__, "kern_mod": mod._kern.__name__}
    print(f"# arm {l} -> {p} impl_mod={mod.__name__} kern_mod={mod._kern.__name__}", flush=True)
print(f"# PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} FLYDSL_RUNTIME_CACHE_DIR={os.environ.get('FLYDSL_RUNTIME_CACHE_DIR')}", flush=True)

rows = B.measure(a.shape, labels, fns, a.iters, a.warmup_seconds, causal=True)
for r in rows:
    r["mode"] = "std"; r.update(info[r["arm"]])
    print("RESULT mode=std arm=%s ms=%.5f sclk=%s/%s" % (r["arm"], r["latency_ms"], r["sclk_start"], r["sclk_end"]), flush=True)

if a.shield:
    q, k, v = make_inputs(a.shape, seed=0)
    flush = torch.empty(B.L2_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    # calibrate a spin long enough to hide host dispatch (target ~1 ms)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    cyc = 1_000_000
    e0.record(); torch.cuda._sleep(cyc); e1.record(); e1.synchronize()
    sl = e0.elapsed_time(e1)
    cyc = int(cyc * 1.0 / max(sl, 1e-3))
    times = {l: [] for l in labels}; host = {l: [] for l in labels}
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    s0 = B.sclk_mhz()
    for i in range(a.iters):
        for l in (labels if i % 2 == 0 else labels[::-1]):
            flush.zero_()
            torch.cuda._sleep(cyc)
            ev0.record()
            t0 = time.perf_counter()
            fns[l](q, k, v, causal=True)
            host[l].append((time.perf_counter() - t0) * 1e6)
            ev1.record(); ev1.synchronize()
            times[l].append(ev0.elapsed_time(ev1))
    s1 = B.sclk_mhz()
    for l in labels:
        r = {"mode": "shield", "arm": l, "latency_ms": round(med(times[l]), 5), "host_us": round(med(host[l]), 1),
             "min_ms": round(min(times[l]), 5), "sclk_start": s0, "sclk_end": s1, "sleep_cycles": cyc}
        r.update(info[l]); rows.append(r)
        print("RESULT mode=shield arm=%s ms=%.5f host_us=%.1f sclk=%s/%s" % (l, r["latency_ms"], r["host_us"], s0, s1), flush=True)
if a.json:
    open(a.json, "w").write(json.dumps(rows, indent=2))
