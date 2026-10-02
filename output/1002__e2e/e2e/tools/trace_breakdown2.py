#!/usr/bin/env python3
"""Per-step GPU time breakdown of kineto traces (torchtitan profile_traces/iteration_N/*.json[.gz]).

  trace_breakdown2.py [--json out.json] [--min-kernels 1500] <trace> [<trace> ...]

Adapted from output/0927__b0/e2e/tools/trace_breakdown.py (same buckets, same attribution by CPU
ancestor range through the launch's correlation id). Additions for the 2026-10-02 A0 runs:
  * STREAM OVERLAP: bwd s6 runs k_dqg on a side stream concurrently with k_dkdv, so the sum of
    kernel durations over-counts wall time. Every bucket now also reports `wall` = the union of its
    kernel intervals, and every e2e::attn_* call its `span` = last kernel end - first kernel start
    (summed over the step's calls). Compare arms on wall/span, not on sum, whenever one arm uses
    two streams. Per-stream sums of the attention buckets are printed too.
  * VALIDITY: on A0 (09-28) kineto kept ~4 GPU kernels per profiled step. A trace with fewer than
    --min-kernels GPU kernels, or whose attention ranges own no kernels, is marked INVALID and
    must not be used (use the CUDA-event numbers from attn_events.py instead).
  * the profiled step's arm(s) are read from the e2e::attn_* range names.
Buckets (first match wins): attn_gqa_sum, attn_copy, attn_fwd, attn_bwd, optimizer, gemm (BLAS
kernel names), gemm_aux (other kernels under aten::mm = nkfix copies/checks), memcpy, memset,
elementwise. idle = GPU span - union of all kernel intervals. json/gzip only: host python is fine.
"""
import bisect
import gzip
import json
import re
import sys
from collections import defaultdict

GEMM_RE = re.compile(r"(cijk_|gemm|hipblaslt|_mt\d+x\d+x\d+|wvsplitk|matmul)", re.I)
ATTN = ("attn_fwd", "attn_bwd", "attn_gqa_sum", "attn_copy")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


def union_ms(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot / 1000.0


def analyse(path, min_kernels, top=12):
    ev = load(path)
    gpu, rt, ranges = [], {}, defaultdict(list)
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(e)
        elif cat in ("cuda_runtime", "cuda_driver"):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                rt[c] = (e["pid"], e["tid"], e["ts"])
        elif cat in ("user_annotation", "cpu_op"):
            n = e["name"]
            if n.startswith("e2e::") or n == "aten::mm" or "Optimizer" in n or \
                    ("optim" in n.lower() and "step" in n.lower()) or n.startswith("ProfilerStep"):
                ranges[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e.get("dur", 0), n))
    idx = {}
    for k, rs in ranges.items():
        rs.sort()
        idx[k] = ([r[0] for r in rs], rs)

    def ancestors(pid, tid, ts):
        if (pid, tid) not in idx:
            return []
        starts, rs = idx[(pid, tid)]
        i = bisect.bisect_right(starts, ts)
        out = []
        j = i - 1
        while j >= 0 and i - j < 4000:
            s, en, n = rs[j]
            if en >= ts:
                out.append((s, n))
            j -= 1
        return out

    buckets = defaultdict(float)
    biv = defaultdict(list)                     # bucket -> kernel intervals (for wall)
    names = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    arms = defaultdict(float)
    calls = defaultdict(lambda: [None, None])   # (range start, range name) -> [first kernel start, last end]
    streams = defaultdict(lambda: defaultdict(float))
    intervals = []
    nkern = 0
    for e in gpu:
        d = e.get("dur", 0) / 1000.0
        iv = (e["ts"], e["ts"] + e.get("dur", 0))
        intervals.append(iv)
        cat = e.get("cat")
        if cat in ("kernel", "Kernel"):
            nkern += 1
        c = e.get("args", {}).get("correlation")
        anc = ancestors(*rt[c]) if c in rt else []
        b, owner = None, None
        for s, n in anc:
            if n == "e2e::asm_gqa_sum":
                b = "attn_gqa_sum"
                break
        if b is None:
            for s, n in anc:
                if n.startswith("e2e::adapter_contiguous"):
                    b = "attn_copy"
                    break
        for s, n in anc:                         # the enclosing attention call (for span / arm)
            if n.startswith("e2e::attn_fwd") or n.startswith("e2e::attn_bwd"):
                owner = (s, n)
                if b is None:
                    b = "attn_fwd" if "attn_fwd" in n else "attn_bwd"
                arms[n] += d
                break
        if b is None and any(("Optimizer" in n or "optim" in n.lower()) for _, n in anc):
            b = "optimizer"
        if b is None:
            if cat == "gpu_memcpy":
                b = "memcpy"
            elif cat == "gpu_memset":
                b = "memset"
            elif GEMM_RE.search(e["name"]):
                b = "gemm"
            elif any(n == "aten::mm" for _, n in anc):
                b = "gemm_aux"
            else:
                b = "elementwise"
        buckets[b] += d
        biv[b].append(iv)
        if owner is not None:
            cw = calls[owner]
            cw[0] = iv[0] if cw[0] is None else min(cw[0], iv[0])
            cw[1] = iv[1] if cw[1] is None else max(cw[1], iv[1])
        if b in ATTN:
            streams[b][str(e.get("tid", e.get("args", {}).get("stream", "?")))] += d
        nm = names[b][e["name"][:110]]
        nm[0] += d
        nm[1] += 1
    span = (max(i[1] for i in intervals) - min(i[0] for i in intervals)) / 1000.0 if intervals else 0.0
    busy = union_ms(intervals)
    step_ranges = [(s, en, n) for rs in ranges.values() for (s, en, n) in rs if n.startswith("ProfilerStep")]
    call_span = defaultdict(float)
    call_n = defaultdict(int)
    for (s, n), (a, z) in calls.items():
        k = "attn_fwd_span" if "attn_fwd" in n else "attn_bwd_span"
        call_span[k] += (z - a) / 1000.0
        call_n[k] += 1
    res = dict(trace=path, steps=sorted({n for (_, _, n) in step_ranges}),
               cpu_step_ms=[(en - s) / 1000.0 for (s, en, _) in step_ranges],
               gpu_kernels=nkern, gpu_span_ms=span, gpu_busy_ms=busy, idle_ms=span - busy,
               kernel_sum_ms=sum(buckets.values()), buckets=dict(buckets),
               wall={b: union_ms(v) for b, v in biv.items()},
               attn_call_span_ms=dict(call_span), attn_calls=dict(call_n),
               arm_ranges=dict(arms), attn_streams={b: dict(v) for b, v in streams.items()},
               top={b: sorted(([k, round(v[0], 3), v[1]] for k, v in d.items()), key=lambda x: -x[1])[:top]
                    for b, d in names.items()})
    res["fa_path_ms"] = sum(buckets.get(k, 0) for k in ATTN)
    res["fa_path_wall_ms"] = union_ms([iv for k in ATTN for iv in biv.get(k, [])])
    arms_seen = sorted({re.sub(r"^e2e::attn_(fwd|bwd)\[(.*)\]$", r"\2", n) for n in arms})
    res["arms"] = arms_seen
    why = []
    if nkern < min_kernels:
        why.append(f"{nkern} GPU kernels < {min_kernels}")
    if res["fa_path_ms"] <= 0:
        why.append("no kernel attributed to an e2e::attn_* range")
    if call_n.get("attn_fwd_span", 0) not in (0, 32) or call_n.get("attn_bwd_span", 0) not in (0, 32):
        why.append(f"attention calls with kernels: fwd {call_n.get('attn_fwd_span', 0)} "
                   f"bwd {call_n.get('attn_bwd_span', 0)} (expected 32 each)")
    res["valid"] = not why
    res["invalid_reason"] = "; ".join(why)
    return res


def show(r):
    print(f"\n=== {r['trace']}")
    print(f"  {'VALID' if r['valid'] else 'INVALID: ' + r['invalid_reason']}  arms {r['arms']}  "
          f"steps {r['steps']} cpu step ms {[round(x, 1) for x in r['cpu_step_ms']]}")
    print(f"  GPU kernels {r['gpu_kernels']}  span {r['gpu_span_ms']:.1f} ms  busy {r['gpu_busy_ms']:.1f}  "
          f"idle {r['idle_ms']:.1f}  kernel-sum {r['kernel_sum_ms']:.1f}")
    for b, v in sorted(r["buckets"].items(), key=lambda x: -x[1]):
        print(f"  {b:14s} sum {v:9.2f} ms  wall {r['wall'].get(b, 0):9.2f} ms")
    print(f"  FA path sum {r['fa_path_ms']:9.2f} ms  wall {r['fa_path_wall_ms']:9.2f} ms  call spans " +
          ", ".join(f"{k}={v:.1f} (n={r['attn_calls'].get(k, 0)})" for k, v in r["attn_call_span_ms"].items()))
    print("  arm ranges: " + ", ".join(f"{k}={v:.1f}" for k, v in r["arm_ranges"].items()))
    for b, st_ in r["attn_streams"].items():
        if len(st_) > 1:
            print(f"  {b} per stream: " + ", ".join(f"{k}={v:.1f}" for k, v in st_.items()))
    for b in ("attn_fwd", "attn_bwd", "attn_gqa_sum", "attn_copy", "gemm", "gemm_aux", "elementwise", "optimizer"):
        if b in r["top"]:
            print(f"  -- {b}")
            for n, ms, c in r["top"][b][:8]:
                print(f"      {ms:9.2f} ms n={c:<5d} {n}")


if __name__ == "__main__":
    out, mk, args = None, 1500, sys.argv[1:]
    while args and args[0].startswith("--"):
        if args[0] == "--json":
            out, args = args[1], args[2:]
        elif args[0] == "--min-kernels":
            mk, args = int(args[1]), args[2:]
        else:
            sys.exit(f"unknown option {args[0]}")
    rs = [analyse(p, mk) for p in args]
    for r in rs:
        show(r)
    nv = sum(1 for r in rs if not r["valid"])
    if nv:
        print(f"\n!! {nv}/{len(rs)} traces INVALID (kineto dropped GPU records?) -- use attn_events.py numbers")
    if out:
        json.dump(rs, open(out, "w"), indent=1)
