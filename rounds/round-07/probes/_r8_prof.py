#!/usr/bin/env python3
"""Attribute every CUDA kernel of the scored proj unit to the aten op that launched it.

methodology/17's approach: read the profiler's trace rather than instrumenting the code.
Prints one line per (cpu_op, kernel) pair with its total device time over one steady-state
iteration, so the unit's non-GEMM, non-norm time stops being anonymous.
"""
import collections
import gzip
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _r8_probe import make_step  # noqa: E402

OUT = "/tmp/_r8_proj_trace.json"


def main():
    step = make_step()
    for _ in range(8):
        step()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=False, record_shapes=True
    ) as p:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    p.export_chrome_trace(OUT)

    ev = json.load(gzip.open(OUT) if OUT.endswith(".gz") else open(OUT))["traceEvents"]
    ker = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ac = [e for e in ev if e.get("cat") == "ac2g" or e.get("cat") == "async_gpu"]
    flow = {}
    for e in ev:
        if e.get("ph") == "s" and e.get("cat") in ("ac2g", "async_gpu", "async"):
            flow.setdefault(e["id"], [None, None])[0] = e
        if e.get("ph") == "f" and e.get("cat") in ("ac2g", "async_gpu", "async"):
            flow.setdefault(e["id"], [None, None])[1] = e
    _ = ac
    runtime = [e for e in ev if e.get("cat") in ("cuda_runtime", "runtime")]
    cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation")]
    cpu.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))

    # correlation -> launching runtime call -> innermost enclosing cpu_op
    rt_by_corr = {e["args"]["correlation"]: e for e in runtime if "correlation" in e.get("args", {})}

    def owner(rt):
        ts, tid = rt["ts"], rt["tid"]
        best = None
        for c in cpu:
            if c["ts"] > ts:
                break
            if c.get("tid") != tid:
                continue
            if c["ts"] <= ts <= c["ts"] + c.get("dur", 0):
                if best is None or c["ts"] >= best["ts"]:
                    best = c
        if not best:
            return "?"
        dims = best.get("args", {}).get("Input Dims") or best.get("args", {}).get("Input dims")
        return f"{best['name']} {dims}"[:110]

    agg = collections.defaultdict(lambda: [0.0, 0])
    for k in ker:
        corr = k.get("args", {}).get("correlation")
        rt = rt_by_corr.get(corr)
        nm = owner(rt) if rt else "?"
        agg[(nm, k["name"][:58])][0] += k.get("dur", 0)
        agg[(nm, k["name"][:58])][1] += 1

    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    tot = sum(v[0] for v in agg.values())
    print(f"total device us over 3 iters = {tot:.0f}  ({tot / 3:.1f} per unit)")
    print(f"{'us/unit':>9} {'n':>4}  cpu_op / kernel")
    for (nm, kn), (d, n) in rows:
        if d / 3 < 1.0:
            continue
        print(f"{d / 3:9.2f} {n // 3:4d}  {nm}\n{'':14s}{kn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
