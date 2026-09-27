"""Execution-order view of each main loop (layout rotated to start at the barrier block; rescale
blocks dropped = steady-state no-rescale path; branch-free variants have none). Reports the
QK->PV gap: instructions between the 32nd WMMA (last QK) and the 33rd (first PV), i.e. the
serial softmax segment the matrix pipe idles through, and what sits inside the PV WMMA span."""
import re, sys, collections
f = sys.argv[1]
lines = open(f).read().splitlines()
lab = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"^(\.LBB[0-9_]+):", l))}
loops = collections.OrderedDict()
for i, l in enumerate(lines):
    m = re.match(r"\s+(s_cbranch_\w+|s_branch)\s+(\.LBB[0-9_]+)", l)
    if m and m.group(2) in lab and lab[m.group(2)] < i:
        loops[lab[m.group(2)]] = max(loops.get(lab[m.group(2)], 0), i)
names = sys.argv[2].split(",") if len(sys.argv) > 2 else []
k = 0
def V(o): return o.startswith("v_") and not o.startswith("v_wmma") and o != "v_nop"
print("loop       QK->PV gap: tot valu exp nop perm pkadd salu+wait | in PV span: valu nop | after PV: valu | total")
for a, b in loops.items():
    blocks, cur = [], []
    for i in range(a + 1, b + 1):
        s = lines[i].strip()
        if re.match(r"^\.LBB[0-9_]+:", s):
            if cur: blocks.append(cur)
            cur = []; continue
        if not s or s.startswith((".", ";", "//")): continue
        op = s.split()[0]
        if op == "s_set_vgpr_msb": continue
        cur.append(op)
        if op.startswith("s_cbranch") or op == "s_branch":
            blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    if sum(o.startswith("v_wmma") for bl in blocks for o in bl) < 64: continue
    blocks = [bl for bl in blocks if not (sum(o == "v_pk_mul_f32" for o in bl) >= 32 and not any(o.startswith("v_wmma") for o in bl))]
    bi = next(i for i, bl in enumerate(blocks) if any(o.startswith("s_barrier") for o in bl))
    seq = [o for bl in blocks[bi:] + blocks[:bi] for o in bl]
    w = [i for i, o in enumerate(seq) if o.startswith("v_wmma")]
    gap = seq[w[31] + 1:w[32]]
    span = seq[w[32] + 1:w[63]]
    after = seq[w[63] + 1:]
    g = lambda L, fn: sum(1 for o in L if fn(o))
    nm = names[k] if k < len(names) else f"L{k}"
    print(f"{nm:10s} {len(gap):4d} {g(gap,V):4d} {g(gap,lambda o:o.startswith('v_exp')):3d} {g(gap,lambda o:o=='v_nop'):3d} "
          f"{g(gap,lambda o:o.startswith('v_permlane')):2d} {g(gap,lambda o:o=='v_pk_add_f32' or 'add_f32' in o):3d} "
          f"{g(gap,lambda o:o.startswith('s_')):3d}       | {g(span,V):4d} {g(span,lambda o:o=='v_nop'):3d}        | {g(after,V):4d}          | {len(seq)}")
    k += 1
