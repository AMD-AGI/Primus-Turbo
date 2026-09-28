"""Summarise replay rows: per input set, fwd fly/asm and fly11/asm for blk/iso/gb, bwd fly/asm (bblk)."""
import json, sys, statistics as st
from collections import defaultdict
FA, FF = "fmha_bf16_pertokenBf16_hd128_128x256_mas", "kn_fmha_fwd_prefill_a16w16_m32x8_bshd"
BA = ["fmha_bwd_hd128_odo_bf16", "fmha_bwd_hd128_bf16_causal_br_a32_pssk", "fmha_bwd_hd128_dq_convert_bf16"]
BF = ["k_delta_bshd", "k_dkdv", "k_dq"]
for fi, f in enumerate(sys.argv[1:]):
    R = json.load(open(f)); g = defaultdict(list); c = defaultdict(list)
    for r in R:
        g[(r["cond"], r["arm"], r["k"])].append(r["dur"])
        if "sclk" in r: c[(r["cond"], r["arm"], r["k"])].append(r["sclk"])
    m = lambda *k: st.median(g[k]) if g[k] else float("nan")
    ms = lambda *k: st.median(c[k]) if c[k] else 0
    sets = sorted({r["cond"].split("_", 1)[1] for r in R})
    print(f"== p{fi+1}  {'set':6s} " + " ".join(f"{x:>26s}" for x in ("blk asm/fly/f11 (f/a)", "iso asm/fly/f11 (f/a)", "gb asm/fly/f11 (f/a)")) + "   bwd asm/fly (f/a) | sclk blk a/f")
    for sname in sets:
        line = f"   p{fi+1}  {sname:6s} "
        for cond in ("blk", "iso", "gb"):
            a, fl, f11 = m(f"{cond}_{sname}", "asm", FA), m(f"{cond}_{sname}", "fly", FF), m(f"{cond}_{sname}", "fly11", FF)
            line += f" {a:.3f}/{fl:.3f}/{f11:.3f} ({fl/a:.2f}) "
        ba = sum(m(f"bblk_{sname}", "asm", k) for k in BA); bfl = sum(m(f"bblk_{sname}", "fly", k) for k in BF)
        line += f"  {ba:.3f}/{bfl:.3f} ({bfl/ba:.3f}) | {ms(f'blk_{sname}', 'asm', FA):.0f}/{ms(f'blk_{sname}', 'fly', FF):.0f}"
        print(line)
