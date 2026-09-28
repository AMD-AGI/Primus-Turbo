#!/usr/bin/env python3
"""Per-layer attention-kernel timeline of one kineto trace (no torch, no GPU).

usage: timeline.py <rank0_trace.json> [clk_samples.tsv] [--json out.json]
For every attention kernel instance (fwd + bwd, both arms) prints: layer index, start (ms from
step start), duration, the previous/next kernel on the GPU (short name) and the idle gap to it,
overlap with any other kernel (concurrent streams), and -- if a clock-sample file is given
(host sampler, columns: epoch_s sclk_hz power_uw [busy]) -- the mean / min sclk and mean power over
the kernel's window. Kernel absolute time = baseTimeNanoseconds + ts(us)*1000.
"""
import json, sys, bisect, re, statistics as st
from collections import defaultdict

ATTN = [("fwd", re.compile(r"kn_fmha_fwd|fmha_bf16_pertoken")),
        ("bwd", re.compile(r"k_delta|k_dkdv|k_dq|fmha_bwd_hd128"))]


def short(n):
    n = n.replace("aiter::", "")
    if n.startswith("Cijk_"):
        m = re.search(r"Cijk_(\w+?)_.*?(MT\d+x\d+x\d+)", n)
        return "GEMM:" + (m.group(2) if m else "?")
    m = re.search(r"at::native::(?:\(anonymous namespace\)::)?(\w+)", n)
    if n.startswith("void") and m:
        inner = re.findall(r"native::(?:\(anonymous namespace\)::)?(\w+)", n)
        return "ew:" + "/".join(inner[:3])
    return n[:48]


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out_json in a:
        a.remove(out_json)
    path = a[0]
    clk = None
    if len(a) > 1:
        clk = []
        for line in open(a[1]):
            p = line.split()
            if len(p) >= 3 and p[0][0].isdigit():
                clk.append((float(p[0]), float(p[1]) / 1e6, float(p[2]) / 1e6))
        clk.sort()
        ct = [c[0] for c in clk]
    d = json.load(open(path))
    base = d.get("baseTimeNanoseconds", 0) / 1e9
    ev = d["traceEvents"]
    ks = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_memset", "gpu_memcpy")],
                key=lambda e: e["ts"])
    t0 = ks[0]["ts"]
    ends = []
    rows = []
    mx_end = -1
    for i, e in enumerate(ks):
        cls = None
        for c, rx in ATTN:
            if rx.search(e["name"]):
                cls = c
        if cls is None:
            continue
        s, en = e["ts"], e["ts"] + e["dur"]
        # prev kernel = the one that ended last before s (by end time among earlier-started)
        prev = max((k for k in ks[max(0, i - 50):i]), key=lambda k: k["ts"] + k["dur"], default=None)
        nxt = ks[i + 1] if i + 1 < len(ks) else None
        ov = 0.0
        for k in ks[max(0, i - 50):i + 50]:
            if k is e:
                continue
            o = min(en, k["ts"] + k["dur"]) - max(s, k["ts"])
            if o > 0:
                ov += o
        r = dict(cls=cls, name=short(e["name"]), start_ms=(s - t0) / 1e3, dur_ms=e["dur"] / 1e3,
                 prev=short(prev["name"]) if prev else None,
                 gap_prev_us=(s - (prev["ts"] + prev["dur"])) if prev else None,
                 prev_dur_ms=prev["dur"] / 1e3 if prev else None,
                 next=short(nxt["name"]) if nxt else None,
                 gap_next_us=(nxt["ts"] - en) if nxt else None, overlap_us=ov)
        if clk:
            ta, tb = base + s / 1e6, base + en / 1e6
            i0, i1 = bisect.bisect_left(ct, ta), bisect.bisect_right(ct, tb)
            win = clk[max(0, i0 - 1):max(i1, i0)]  # value held at ta plus changes inside
            if win:
                r["sclk_mean"] = st.mean(c[1] for c in win)
                r["sclk_min"] = min(c[1] for c in win)
                r["pwr_mean_w"] = st.mean(c[2] for c in win)
                r["n_samp"] = len(win)
            # 5 ms before
            j0 = bisect.bisect_left(ct, ta - 0.005)
            pre = clk[j0:i0]
            if pre:
                r["sclk_pre5ms"] = st.mean(c[1] for c in pre)
        rows.append(r)
    # layer index per kernel name
    cnt = defaultdict(int)
    for r in rows:
        r["layer"] = cnt[r["name"]]
        cnt[r["name"]] += 1
    step_ms = (max(k["ts"] + k["dur"] for k in ks) - t0) / 1e3
    print(f"# {path}  step span {step_ms:.1f} ms")
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    for n, rs in by.items():
        du = [r["dur_ms"] for r in rs]
        gp = [r["gap_prev_us"] for r in rs if r["gap_prev_us"] is not None]
        pv = defaultdict(int)
        for r in rs:
            pv[r["prev"]] += 1
        line = (f"{n:44s} n={len(rs):2d} sum={sum(du):7.2f} med={st.median(du):6.3f} min={min(du):6.3f} "
                f"max={max(du):6.3f} gap_prev_med={st.median(gp):6.1f}us ovl_sum={sum(r['overlap_us'] for r in rs):.0f}us")
        if clk and all("sclk_mean" in r for r in rs):
            line += f" sclk={st.mean(r['sclk_mean'] for r in rs):.0f}MHz(min {min(r['sclk_min'] for r in rs):.0f}) P={st.mean(r['pwr_mean_w'] for r in rs):.0f}W"
        print(line)
        print("    prev:", dict(sorted(pv.items(), key=lambda x: -x[1])[:4]))
    if out_json:
        json.dump(dict(trace=path, step_ms=step_ms, rows=rows), open(out_json, "w"), indent=0)


main()
