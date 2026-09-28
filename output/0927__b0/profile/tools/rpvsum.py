"""Variant replay summary: per set and condition, fwd ms of asm / fly(r6) / nospec / nodefer and ratios vs asm."""
import json, sys, statistics as st
from collections import defaultdict
FA, FF = "fmha_bf16_pertokenBf16_hd128_128x256_mas", "kn_fmha_fwd_prefill_a16w16_m32x8_bshd"
for fi, f in enumerate(sys.argv[1:]):
    R = json.load(open(f)); g = defaultdict(list)
    for r in R: g[(r["cond"], r["arm"], r["k"])].append(r["dur"])
    m = lambda *k: st.median(g[k]) if g[k] else float("nan")
    sets = sorted({r["cond"].split("_", 1)[1] for r in R})
    print(f"== p{fi+1}: cond set | asm r6 nospec nodefer | r6/asm nospec/asm nodefer/asm")
    for cond in ("blk", "iso", "gb"):
        for s in sets:
            a = m(f"{cond}_{s}", "asm", FA); x = [m(f"{cond}_{s}", v, FF) for v in ("fly", "nospec", "nodefer")]
            print(f"p{fi+1} {cond:3s} {s:6s} | {a:.3f} " + " ".join(f"{y:.3f}" for y in x) + " | " + " ".join(f"{y/a:.3f}" for y in x))
