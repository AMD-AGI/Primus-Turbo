"""List backward-branch loops in a final ISA and classify their instructions."""
import re, sys, collections
lines = open(sys.argv[1]).read().split("\n")
lab = {}
ins = []
for i, l in enumerate(lines):
    m = re.match(r"^(\.LBB\w+):", l)
    if m: lab[m.group(1)] = len(ins)
    m = re.match(r"^\s+([a-z][a-z0-9_]+)\b(.*)", l)
    if m and not l.strip().startswith("."):
        ins.append((m.group(1), m.group(2)))
def cls(op):
    if op.startswith("v_wmma"): return "wmma"
    if op.startswith("v_exp") or op.startswith("v_log") or op.startswith("v_rcp"): return "trans"
    if op.startswith("ds_"): return "lds"
    if op == "s_set_vgpr_msb": return "msb"
    if op == "v_nop" or op == "s_nop": return "nop"
    if op.startswith("s_wait") or op.startswith("s_barrier"): return "wait/bar"
    if op.startswith("v_"): return "valu"
    if op.startswith("s_"): return "salu"
    return "mem"
for j, (op, rest) in enumerate(ins):
    if op.startswith("s_cbranch") or op == "s_branch":
        t = rest.strip().split()[0] if rest.strip() else ""
        if t in lab and lab[t] <= j and j - lab[t] > 200:
            body = ins[lab[t]:j+1]
            c = collections.Counter(cls(o) for o, _ in body)
            w = collections.Counter(o for o, _ in body if o.startswith("s_wait") or o.startswith("s_barrier"))
            print(f"loop {t} len={len(body)} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
            print("   waits:", dict(w))
