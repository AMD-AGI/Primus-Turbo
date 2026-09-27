"""Find loops (backward branches) in a 22_final_isa.s and print a per-loop instruction-class histogram.
usage: loops.py <isa.s> [min_len]"""
import collections, re, sys

lines = open(sys.argv[1]).read().split("\n")
minlen = int(sys.argv[2]) if len(sys.argv) > 2 else 200
lab = {}
for i, l in enumerate(lines):
    m = re.match(r"^(\.LBB\w+):", l)
    if m:
        lab[m.group(1)] = i

def cls(op, l):
    if op.startswith("v_wmma"): return "WMMA"
    if op.startswith("v_exp"): return "v_exp"
    if op == "v_nop": return "v_nop"
    if op.startswith("v_permlane"): return "v_permlane"
    if op.startswith("v_"): return "VALU(other)"
    if op.startswith("ds_load_tr"): return "ds_load_tr*"
    if op.startswith("ds_load"): return "ds_load*"
    if op.startswith("ds_"): return "ds_other"
    if op.startswith("tensor_"): return "TDM"
    if op.startswith("s_barrier"): return op
    if op.startswith("s_wait_"): return op
    if op in ("s_nop", "s_delay_alu", "s_set_vgpr_msb"): return op
    if op.startswith("s_cbranch") or op == "s_branch": return "branch"
    if op.startswith("s_"): return "SALU(other)"
    if op.startswith("global_") or op.startswith("buffer_"): return "VMEM"
    return "other:" + op

ORDER = ["WMMA", "v_exp", "VALU(other)", "v_permlane", "v_nop", "ds_load*", "ds_load_tr*", "ds_other",
         "TDM", "s_barrier_signal", "s_barrier_wait", "s_wait_tensorcnt", "s_wait_dscnt", "s_wait_loadcnt",
         "s_wait_asynccnt", "s_wait_alu", "s_nop", "s_delay_alu", "s_set_vgpr_msb", "SALU(other)", "branch"]
loops = []
for i, l in enumerate(lines):
    m = re.match(r"\s+(s_cbranch_\w+|s_branch)\s+(\.LBB\w+)", l)
    if m and m.group(2) in lab and lab[m.group(2)] < i:
        loops.append((lab[m.group(2)], i, m.group(2)))
for s, e, name in loops:
    h = collections.Counter(); ops = collections.Counter(); n = 0
    for l in lines[s:e + 1]:
        m = re.match(r"\s+([a-z][a-z0-9_]+)(\s|$)", l)
        if not m or l.lstrip().startswith("."): continue
        op = m.group(1); n += 1; h[cls(op, l)] += 1
    if n < minlen: continue
    print(f"loop {name} lines {s+1}-{e+1}: {n} instr")
    print("   " + "  ".join(f"{k}={h[k]}" for k in ORDER if h[k]) +
          "  " + "  ".join(f"{k}={v}" for k, v in h.items() if k not in ORDER))
