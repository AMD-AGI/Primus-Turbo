"""One-line-per-loop summary for a final ISA: every back-edge loop (header label .. last back-branch),
its WMMA count (64 WMMA per 64 KV at R=2, d=128), barriers,
exp coverage, and the signal->wait gap on the fall-through path. usage: report.py file.s"""
import re, sys, collections
f = sys.argv[1]
L = open(f).read().split("\n")
lab = {}
for i, l in enumerate(L):
    m = re.match(r"^(\.LBB\w+):", l)
    if m: lab[m.group(1)] = i + 1
loops = collections.OrderedDict()
for i, l in enumerate(L):
    m = re.match(r"^\s+(s_cbranch\w*|s_branch)\s+(\.LBB\w+)", l)
    if m and m.group(2) in lab and lab[m.group(2)] < i + 1:
        h = lab[m.group(2)]
        loops[h] = max(loops.get(h, 0), i + 1)
def ops(lo, hi):
    o = []
    for i in range(lo - 1, hi):
        m = re.match(r"^\s+([a-z][a-z0-9_]+)", L[i])
        if m and not L[i].strip().startswith("."): o.append((i + 1, m.group(1)))
    return o
for h, e in loops.items():
    o = ops(h, e)
    w = sum(op.startswith("v_wmma") for _, op in o)
    if w < 64: continue
    names = [op for _, op in o if op not in ("s_set_vgpr_msb", "s_wait_alu")]
    last = -10**9; ne = nec = 0
    for k, op in enumerate(names):
        if op.startswith("v_wmma"): last = k
        elif op.startswith("v_exp"): ne += 1; nec += (k - last <= 8)
    sig = [ln for ln, op in o if op == "s_barrier_signal"]; wt = [ln for ln, op in o if op == "s_barrier_wait"]
    gap = ""
    if sig and wt and wt[0] > sig[0]:
        g = [op for ln, op in o if sig[0] < ln < wt[0] and op not in ("s_set_vgpr_msb", "s_wait_alu")]
        gap = f" gap(signal->wait, all blocks)={len(g)} instr ({sum(x.startswith('v_wmma') for x in g)} WMMA)"
    print(f"loop L{h}-{e}: {len(o)} instr, WMMA={w} (={w} KV at R=2, d=128), v_exp={ne} ({nec} within 8 after a WMMA), "
          f"signal={len(sig)} wait={len(wt)} msb={sum(op=='s_set_vgpr_msb' for _,op in o)}{gap}")
