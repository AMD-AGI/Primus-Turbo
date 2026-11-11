"""Walk the hot loop of an ATT stats CSV in address order, one char per instruction class,
with per-segment cycle totals: python wip/att_walk.py CSV WAVES_PER_WG [min_hit_per_wave]"""
import csv, re, sys
path, wpg = sys.argv[1], int(sys.argv[2])
minh = float(sys.argv[3]) if len(sys.argv) > 3 else 10
rows = []
for r in csv.DictReader(open(path)):
    ins = r["Instruction"].strip()
    if not ins or ins.startswith(";"):
        continue
    rows.append((int(r["Vaddr"]), ins, int(r["Hitcount"]), int(r["Latency"]), int(r["Stall"])))
rows.sort()
waves = next(h for _, _, h, _, _ in rows if h > 0)
def cls(op):
    for p, c in (("v_mfma", "M"), ("ds_", "d"), ("buffer_", "L"), ("s_waitcnt", "W"), ("s_barrier", "B"), ("s_nop", "n"),
                 ("v_exp", "e"), ("v_accvgpr_read", "r"), ("v_accvgpr_write", "w"), ("v_permlane", "p"), ("s_cbranch", "|"), ("s_branch", "|"), ("v_", "v"), ("s_", "o")):
        if op.startswith(p):
            return c
    return "?"
loop = [r for r in rows if r[2] / waves >= minh]
ITERS = loop[0][2]
tot = sum(r[3] for r in loop)
print(f"waves {waves}; hot instructions {len(loop)}; hot cycles/wave {tot / waves:,.0f}")
# segments: split at branches; print pattern and per-hit cycles per class
seg, segs = [], []
for r in loop:
    seg.append(r)
    if cls(r[1].split()[0]) == "|":
        segs.append(seg); seg = []
if seg:
    segs.append(seg)
for s in segs:
    hit = max(r[2] for r in s)
    cyc = sum(r[3] for r in s) / ITERS
    by = {}
    for r in s:
        c = cls(r[1].split()[0]); by[c] = by.get(c, 0) + r[3] / ITERS
    top = " ".join(f"{k}:{v:,.0f}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:7])
    print(f"\n[{s[0][0]:#x} hit/wave {hit / waves:5.1f} cyc/iter {cyc:8,.0f}] {top}")
    line = ""
    for r in s:
        c = cls(r[1].split()[0]); per = r[3] / max(1, r[2])
        line += c.upper() if per >= 24 and c not in "M" else c
    for i in range(0, len(line), 180):
        print("  " + line[i:i + 180])
