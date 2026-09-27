"""Per-loop-body census of a FlyDSL 22_final_isa.s: find backward branches, count ops."""
import collections, re, sys
L = open(sys.argv[1]).read().split("\n")
lab = {}
for i, l in enumerate(L):
    m = re.match(r"^(\.LBB\w+):", l)
    if m: lab[m.group(1)] = i
loops = []
for i, l in enumerate(L):
    m = re.match(r"\s+s_cbranch_\w+\s+(\.LBB\w+)|\s+s_branch\s+(\.LBB\w+)", l)
    if m:
        t = m.group(1) or m.group(2)
        if t in lab and lab[t] < i: loops.append((t, lab[t], i))
KEYS = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "v_wmma", "global_atomic_add_f32", "buffer_load_b128", "buffer_load_b32", "ds_load_tr16_b128",
    "ds_store_b128", "s_wait_loadcnt", "s_wait_dscnt", "s_wait_storecnt", "s_wait_xcnt", "s_set_vgpr_msb", "v_nop", "s_nop"]
for t, a, b in loops:
    h = collections.Counter(); n = 0
    for l in L[a:b + 1]:
        m = re.match(r"\s+([a-z][a-z0-9_]+)", l)
        if m and not l.lstrip().startswith("."):
            h[m.group(1)] += 1; n += 1
    ks = {k: sum(v for op, v in h.items() if op.startswith(k)) for k in KEYS}
    print(f"loop {t} lines {a}-{b} instr {n} " + " ".join(f"{k}={v}" for k, v in ks.items()))
