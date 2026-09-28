#!/usr/bin/env python3
"""Per-step GPU time breakdown of one kineto trace (torchtitan profile_traces/iteration_N/*.json[.gz]).

Every GPU kernel is attributed by its CPU ancestor ranges (via the launch's correlation id), not by
kernel name, so side-effect kernels launched from the attention adapter (casts, zero_, GQA sum,
preprocess/delta kernels) land in the FA path, like JIRA's "FA path = flex + elem + softmax".

Buckets (first match wins):
  attn_gqa_sum     ancestor e2e::asm_gqa_sum
  attn_copy        ancestor e2e::adapter_contiguous[..]
  attn_fwd         ancestor e2e::attn_fwd[..]
  attn_bwd         ancestor e2e::attn_bwd[..]
  optimizer        ancestor Optimizer.step* / *optimizer*step*
  gemm             kernel name looks like a BLAS GEMM (Cijk_/gemm/hipblaslt/MT..x..)
  gemm_aux         any other kernel whose CPU ancestor is aten::mm (nkfix operand copies + checks)
  memcpy/memset    gpu_memcpy / gpu_memset events
  elementwise      everything else (norms, rope, silu, loss, casts, grad-accum adds, ...)
idle = (last kernel end - first kernel start) - union(kernel busy intervals).
Runs with only json/gzip (no torch): run it inside the container, no GPU, no flock needed.
"""
import gzip, json, re, sys, bisect, heapq
from collections import defaultdict

GEMM_RE = re.compile(r"(cijk_|gemm|hipblaslt|_mt\d+x\d+x\d+|wvsplitk|matmul)", re.I)


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


def analyse(path, top=12):
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
            if n.startswith("e2e::") or n == "aten::mm" or "Optimizer" in n or ("optim" in n.lower() and "step" in n.lower()) \
                    or n.startswith("ProfilerStep"):
                ranges[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e.get("dur", 0), n))
    # per-thread: sorted starts for containment lookups (ranges of interest are few, nesting shallow)
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
        # walk back; ranges of interest are sparse so a bounded scan is fine
        j = i - 1
        while j >= 0 and i - j < 4000:
            s, en, n = rs[j]
            if en >= ts:
                out.append(n)
            j -= 1
        return out

    buckets = defaultdict(float)
    names = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    arms = defaultdict(float)
    intervals = []
    for e in gpu:
        d = e.get("dur", 0) / 1000.0
        intervals.append((e["ts"], e["ts"] + e.get("dur", 0)))
        cat = e.get("cat")
        c = e.get("args", {}).get("correlation")
        anc = ancestors(*rt[c]) if c in rt else []
        b = None
        for n in anc:
            if n == "e2e::asm_gqa_sum":
                b = "attn_gqa_sum"; break
        if b is None:
            for n in anc:
                if n.startswith("e2e::adapter_contiguous"):
                    b = "attn_copy"; break
        if b is None:
            for n in anc:
                if n.startswith("e2e::attn_fwd") or n.startswith("e2e::attn_bwd"):
                    b = "attn_fwd" if "attn_fwd" in n else "attn_bwd"
                    arms[n] += d
                    break
        if b is None and any(("Optimizer" in n or "optim" in n.lower()) for n in anc):
            b = "optimizer"
        if b is None:
            if cat == "gpu_memcpy":
                b = "memcpy"
            elif cat == "gpu_memset":
                b = "memset"
            elif GEMM_RE.search(e["name"]):
                b = "gemm"
            elif "aten::mm" in anc:
                b = "gemm_aux"      # non-GEMM kernels inside aten::mm = the nkfix copies / checks
            else:
                b = "elementwise"
        buckets[b] += d
        nm = names[b][e["name"][:110]]
        nm[0] += d; nm[1] += 1
    intervals.sort()
    busy, cur_s, cur_e = 0, None, None
    for s, en in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, en
        else:
            cur_e = max(cur_e, en)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (intervals[-1][1] - intervals[0][0]) / 1000.0 if intervals else 0
    busy /= 1000.0
    steps = sorted({n for rs in ranges.values() for (_, _, n) in rs if n.startswith("ProfilerStep")})
    stepdur = [en - s for rs in ranges.values() for (s, en, n) in rs if n.startswith("ProfilerStep")]
    res = dict(trace=path, steps=steps, cpu_step_ms=[x / 1000.0 for x in stepdur], gpu_span_ms=span,
               gpu_busy_ms=busy, idle_ms=span - busy, kernel_sum_ms=sum(buckets.values()),
               buckets=dict(buckets), arm_ranges=dict(arms),
               top={b: sorted(([k, round(v[0], 3), v[1]] for k, v in d.items()), key=lambda x: -x[1])[:top]
                    for b, d in names.items()})
    res["fa_path_ms"] = sum(buckets.get(k, 0) for k in ("attn_fwd", "attn_bwd", "attn_gqa_sum", "attn_copy"))
    return res


def show(r):
    print(f"\n=== {r['trace']}")
    print(f"  steps {r['steps']} cpu step ms {[round(x,1) for x in r['cpu_step_ms']]}")
    print(f"  GPU span {r['gpu_span_ms']:.1f} ms  busy {r['gpu_busy_ms']:.1f}  idle {r['idle_ms']:.1f}  "
          f"kernel-sum {r['kernel_sum_ms']:.1f}")
    for b, v in sorted(r["buckets"].items(), key=lambda x: -x[1]):
        print(f"  {b:14s} {v:9.2f} ms")
    print(f"  FA path total  {r['fa_path_ms']:9.2f} ms   arm ranges: " +
          ", ".join(f"{k}={v:.1f}" for k, v in r["arm_ranges"].items()))
    for b in ("attn_fwd", "attn_bwd", "attn_gqa_sum", "attn_copy", "gemm", "gemm_aux", "elementwise", "optimizer"):
        if b in r["top"]:
            print(f"  -- {b}")
            for n, ms, c in r["top"][b][:8]:
                print(f"      {ms:9.2f} ms n={c:<5d} {n}")


if __name__ == "__main__":
    out = None
    args = sys.argv[1:]
    if args and args[0] == "--json":
        out, args = args[1], args[2:]
    rs = [analyse(p) for p in args]
    for r in rs:
        show(r)
    if out:
        json.dump(rs, open(out, "w"), indent=1)
