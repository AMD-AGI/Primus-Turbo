#!/usr/bin/env python3
"""Aggregate opana rows of several oppoint processes: per-process medians and ratios, and a
clock-sensitivity fit dur = a + b * (1000/sclk) per kernel (conds blk/iso/gburst/eburst[/layer]).
usage: opsum.py rows1.json rows2.json ...
"""
import json, sys, statistics as st
from collections import defaultdict

FWD = {"asm": "fmha_bf16_pertokenBf16_hd128_128x256_mas", "fly": "kn_fmha_fwd_prefill_a16w16_m32x8_bshd",
       "fly11": "kn_fmha_fwd_prefill_a16w16_m32x8_bshd"}
BWD = {"asm": ["fmha_bwd_hd128_odo_bf16", "fmha_bwd_hd128_bf16_causal_br_a32_pssk", "fmha_bwd_hd128_dq_convert_bf16"],
       "fly": ["k_delta_bshd", "k_dkdv", "k_dq"]}
BWD["fly11"] = BWD["fly"]
files = sys.argv[1:]
allrows = []
print("== per process medians (ms); bwd = sum of the arm's kernels (ASM without fill/gqa_sum)")
print(f"{'proc':6s} {'cond':9s} {'asm_f':>7s} {'fly_f':>7s} {'f11_f':>7s} {'fly/asm':>7s} {'f11/asm':>7s} | {'asm_b':>7s} {'fly_b':>7s} {'fly/asm':>7s} | sclk asm/fly fwd, bwd")
for fi, f in enumerate(files):
    R = json.load(open(f)); allrows += R
    g = defaultdict(list); c = defaultdict(list)
    for r in R:
        g[(r["cond"], r["arm"], r["k"])].append(r["dur"])
        if "sclk" in r: c[(r["cond"], r["arm"], r["k"])].append(r["sclk"])
    for cond in ("iso", "blk", "eburst", "gburst", "layer", "layerbad"):
        m = lambda a, k: st.median(g[(cond, a, k)]) if g[(cond, a, k)] else float("nan")
        cs = lambda a, k: st.median(c[(cond, a, k)]) if c[(cond, a, k)] else 0
        fa, ff, f11 = m("asm", FWD["asm"]), m("fly", FWD["fly"]), m("fly11", FWD["fly11"])
        ba = sum(m("asm", k) for k in BWD["asm"]); bf = sum(m("fly", k) for k in BWD["fly"])
        print(f"p{fi+1:<5d} {cond:9s} {fa:7.3f} {ff:7.3f} {f11:7.3f} {ff/fa:7.3f} {f11/fa:7.3f} | {ba:7.3f} {bf:7.3f} {bf/ba:7.3f} | "
              f"{cs('asm', FWD['asm']):.0f}/{cs('fly', FWD['fly']):.0f}, {cs('asm', BWD['asm'][1]):.0f}/{cs('fly', 'k_dkdv'):.0f}")
    # scale
    for sub in ("K::fwd_x1", "K::fwd_x2", "K::fwd_x4", "K::fwd_x8", "K::fwd_x16"):
        v = {}
        for a in ("asm", "fly", "fly11"):
            xs = [r["dur"] for r in R if r["cond"] == "scale" and r["sub"] == sub and r["arm"] == a and r["k"] == FWD[a]]
            v[a] = st.median(xs) if xs else float("nan")
        print(f"p{fi+1:<5d} scale {sub[6:]:4s} asm {v['asm']:.3f} fly {v['fly']:.3f} fly11 {v['fly11']:.3f} fly/asm {v['fly']/v['asm']:.3f}")

print("\n== clock sensitivity: least squares dur = a + b*(1000/sclk_MHz), all processes")
print(f"{'arm':6s} {'kernel':40s} {'n':>5s} {'@2350':>7s} {'@1800':>7s} {'@1350':>7s} {'1350/2350':>9s} {'clk-share@1350':>14s} r2")
for a, ks in (("asm", [FWD["asm"]] + BWD["asm"]), ("fly", [FWD["fly"]] + BWD["fly"]), ("fly11", [FWD["fly11"]])):
    for k in ks:
        conds = ("blk", "iso", "gburst", "eburst") if k in FWD.values() else ("blk", "iso", "gburst", "eburst", "layer", "layerbad")
        pts = [(1000.0 / r["sclk"], r["dur"]) for r in allrows if r["arm"] == a and r["k"] == k and r["cond"] in conds and r.get("sclk")]
        if len(pts) < 5: continue
        xs, ys = zip(*pts); mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in pts)
        bb = sxy / sxx; aa = my - bb * mx
        ssr = sum((y - (aa + bb * x)) ** 2 for x, y in pts); sst = sum((y - my) ** 2 for y in ys)
        p = lambda mhz: aa + bb * 1000 / mhz
        print(f"{a:6s} {k[:40]:40s} {len(pts):5d} {p(2350):7.3f} {p(1800):7.3f} {p(1350):7.3f} {p(1350)/p(2350):9.3f} "
              f"{bb*1000/1350/p(1350):14.2f} {1-ssr/sst:.2f}")
