"""Per-iteration dynamic counts of each main loop (steady state: prefetch taken).
no-resc = skip blocks that are pure rescale (v_pk_mul_f32 >= 32 and no WMMA); all-resc = every block."""
import re, subprocess, sys, collections
f = sys.argv[1]
lines = open(f).read().splitlines()
lab = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"^(\.LBB[0-9_]+):", l))}
loops = collections.OrderedDict()
for i, l in enumerate(lines):
    m = re.match(r"\s+(s_cbranch_\w+|s_branch)\s+(\.LBB[0-9_]+)", l)
    if m and m.group(2) in lab and lab[m.group(2)] < i:
        a = lab[m.group(2)]
        loops[a] = max(loops.get(a, 0), i)
KEYS = ["tot", "wmma", "exp", "valu", "nop", "pkfma", "pkadd", "pkmul", "perm", "ds", "tdm", "bar", "wait", "delay", "msb", "salu", "br"]
def cnt(ins):
    g = lambda fn: sum(1 for o in ins if fn(o))
    return dict(tot=len(ins), wmma=g(lambda o: o.startswith("v_wmma")), exp=g(lambda o: o.startswith("v_exp")),
        valu=g(lambda o: o.startswith("v_") and not o.startswith("v_wmma") and o != "v_nop"), nop=g(lambda o: o == "v_nop"),
        pkfma=g(lambda o: o == "v_pk_fma_f32"), pkadd=g(lambda o: o == "v_pk_add_f32"), pkmul=g(lambda o: o == "v_pk_mul_f32"),
        perm=g(lambda o: o.startswith("v_permlane")), ds=g(lambda o: o.startswith("ds_")), tdm=g(lambda o: o.startswith("tensor_")),
        bar=g(lambda o: o.startswith("s_barrier")), wait=g(lambda o: o.startswith("s_wait")), delay=g(lambda o: o == "s_delay_alu"),
        msb=g(lambda o: o == "s_set_vgpr_msb"),
        salu=g(lambda o: o.startswith("s_") and not o.startswith(("s_wait", "s_barrier", "s_delay", "s_set_vgpr", "s_cbranch", "s_branch"))),
        br=g(lambda o: o.startswith("s_cbranch") or o == "s_branch"))
names = sys.argv[2].split(",") if len(sys.argv) > 2 else None
k = 0
print("loop".ljust(22) + "".join(x.rjust(6) for x in KEYS) + "  blocks")
for a, b in loops.items():
    blocks, cur = [], []
    for i in range(a + 1, b + 1):
        s = lines[i].strip()
        if re.match(r"^\.LBB[0-9_]+:", s):
            if cur: blocks.append(cur)
            cur = []; continue
        if not s or s.startswith((".", ";", "//")): continue
        op = s.split()[0]; cur.append(op)
        if op.startswith("s_cbranch") or op == "s_branch":
            blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    allins = [o for bl in blocks for o in bl]
    if sum(o.startswith("v_wmma") for o in allins) < 32: continue
    resc = [bl for bl in blocks if sum(o == "v_pk_mul_f32" for o in bl) >= 32 and not any(o.startswith("v_wmma") for o in bl)]
    nores = [o for bl in blocks if bl not in resc for o in bl]
    nm = names[k] if names and k < len(names) else f"L{k}@{a+1}"
    for tag, ins in (("no-resc", nores), ("all-resc", allins)):
        c = cnt(ins)
        print(f"{nm+' '+tag:22s}" + "".join(str(c[x]).rjust(6) for x in KEYS) + f"  {len(blocks)}")
    k += 1
