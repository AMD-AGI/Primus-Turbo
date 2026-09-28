#!/usr/bin/env python3
"""Attribute every GPU kernel of an oppoint.py kineto trace to its P::<cond>::<arm>::<rep> range and
K::/L:: sub-range, join the host clock samples, and print per (cond, arm, kernel) statistics.
usage: opana.py <trace.json> <clk file> [--rows out.json]
Rules: blk drops the first 4 calls of each kernel in each block (warm-up); scale drops the first 2.
"""
import json, sys, bisect, re, statistics as st
from collections import defaultdict

ATT = re.compile(r"kn_fmha_fwd|fmha_bf16_pertoken|k_delta|k_dkdv|k_dq|fmha_bwd_hd128|FillFunctor|reduce_kernel")


def sname(n):
    n = n.replace("aiter::", "")
    if n.startswith("Cijk_"):
        m = re.search(r"(MT\d+x\d+x\d+)", n); return "GEMM_" + (m.group(1) if m else "?")
    if "FillFunctor" in n: return "fill"
    if "reduce_kernel" in n: return "gqa_sum"
    return re.sub(r"_0$", "", n)[:40]


def load_clk(p):
    c = []
    for line in open(p):
        x = line.split()
        if len(x) >= 3:
            try: c.append((float(x[0]), float(x[1]) / 1e6, float(x[2]) / 1e6))
            except ValueError: pass
    c.sort(); return c


def clk_win(clk, ct, ta, tb):
    i0, i1 = bisect.bisect_left(ct, ta), bisect.bisect_right(ct, tb)
    w = clk[max(0, i0 - 1):max(i1, i0)]
    if not w: return None
    # time-weighted mean over [ta, tb]
    tot, acc_f, acc_p = 0.0, 0.0, 0.0
    for j, (t, f, p) in enumerate(w):
        a = max(t, ta); b = min(w[j + 1][0], tb) if j + 1 < len(w) else tb
        if b > a: tot += b - a; acc_f += f * (b - a); acc_p += p * (b - a)
    if tot <= 0: return (w[-1][1], w[-1][2], min(x[1] for x in w))
    return (acc_f / tot, acc_p / tot, min(x[1] for x in w))


def main():
    tr, ck = sys.argv[1], sys.argv[2]
    rows_out = sys.argv[sys.argv.index("--rows") + 1] if "--rows" in sys.argv else None
    d = json.load(open(tr)); base = d.get("baseTimeNanoseconds", 0) / 1e9
    ev = d["traceEvents"]
    clk = load_clk(ck); ct = [c[0] for c in clk]
    launch, rng = {}, defaultdict(list)
    kern = []
    for e in ev:
        if e.get("ph") != "X": continue
        c = e.get("cat", "")
        if c in ("kernel", "Kernel"): kern.append(e)
        elif c in ("cuda_runtime", "cuda_driver"):
            cor = e.get("args", {}).get("correlation")
            if cor is not None: launch[cor] = (e["pid"], e["tid"], e["ts"])
        elif c == "user_annotation" and e["name"][:3] in ("P::", "K::", "L::"):
            rng[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    for k in rng: rng[k].sort()

    def enclosing(pid, tid, ts):
        out = {}
        for s, en, n in rng.get((pid, tid), []):
            if s <= ts <= en: out[n[:1]] = n
        return out
    kern.sort(key=lambda e: e["ts"])
    rows = []
    for i, e in enumerate(kern):
        cor = e.get("args", {}).get("correlation")
        if cor not in launch: continue
        enc = enclosing(*launch[cor])
        if "P" not in enc: continue
        _, cond, arm, rep = enc["P"].split("::")
        sub = enc.get("K", enc.get("L", ""))
        prev = kern[i - 1] if i else None
        r = dict(cond=cond, arm=arm, rep=int(rep), sub=sub, k=sname(e["name"]), ts=e["ts"], dur=e["dur"] / 1e3,
                 gap_prev_us=(e["ts"] - prev["ts"] - prev["dur"]) if prev else None,
                 prev=sname(prev["name"]) if prev else None)
        cw = clk_win(clk, ct, base + e["ts"] / 1e6, base + (e["ts"] + e["dur"]) / 1e6)
        if cw: r["sclk"], r["pw"], r["sclk_min"] = cw
        pre = clk_win(clk, ct, base + e["ts"] / 1e6 - 0.003, base + e["ts"] / 1e6)
        if pre: r["sclk_pre"] = pre[0]
        rows.append(r)
    # drop warm-up in blk/scale blocks
    seen = defaultdict(int); keep = []
    for r in rows:
        key = (r["cond"], r["arm"], r["rep"], r["sub"], r["k"])
        seen[key] += 1
        if r["cond"].startswith("blk") and seen[key] <= 4: continue
        if r["cond"].startswith("bblk") and seen[key] <= 2: continue
        if r["cond"] == "scale" and seen[key] <= 2: continue
        keep.append(r)
    g = defaultdict(list)
    for r in keep:
        if ATT.search(r["k"]) or r["k"] in ("fill", "gqa_sum"):
            if r["cond"] in ("layer", "layerbad") or r["k"] not in ("fill",):
                g[(r["cond"], r["sub"] if r["cond"] == "scale" else "", r["k"], r["arm"])].append(r)
    print(f"{'cond':9s} {'sub':10s} {'kernel':38s} {'arm':6s} {'n':>4s} {'med_ms':>7s} {'min':>7s} {'sclk':>6s} {'sclk_pre':>8s} {'W':>5s}")
    for key in sorted(g):
        rs = g[key]
        du = [r["dur"] for r in rs]
        sc = [r["sclk"] for r in rs if "sclk" in r]; sp = [r["sclk_pre"] for r in rs if "sclk_pre" in r]
        pw = [r["pw"] for r in rs if "pw" in r]
        print(f"{key[0]:9s} {key[1][:10]:10s} {key[2][:38]:38s} {key[3]:6s} {len(rs):4d} {st.median(du):7.3f} {min(du):7.3f} "
              f"{st.median(sc) if sc else 0:6.0f} {st.median(sp) if sp else 0:8.0f} {st.median(pw) if pw else 0:5.0f}")
    # GEMM tiles seen in layer conditions (layout check)
    gt = defaultdict(lambda: [0, 0.0])
    for r in keep:
        if r["k"].startswith("GEMM") and r["cond"].startswith("layer"):
            gt[(r["cond"], r["k"])][0] += 1; gt[(r["cond"], r["k"])][1] += r["dur"]
    for k, v in sorted(gt.items()): print("gemm", k, v[0], round(v[1], 1), "ms")
    if rows_out: json.dump(keep, open(rows_out, "w"))


main()
