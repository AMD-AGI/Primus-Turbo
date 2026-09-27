"""Print a line range of a final ISA as a class string (one char per instruction, msb/alu-waits hidden).
W wmma  E trans(v_exp/v_log/v_rcp)  V other valu  P v_permlane*  L ds_load*  s ds_store  T tensor_load
B s_barrier*  t s_wait_tensorcnt  d s_wait_dscnt  S salu  n nop  b branch  | label
usage: seq.py file.s lo hi [width]"""
import re, sys
f, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); w = int(sys.argv[4]) if len(sys.argv) > 4 else 100
L = open(f).read().split("\n")
out = []
for i in range(lo - 1, hi):
    l = L[i]
    if re.match(r"^\.LBB\w+:", l): out.append("|"); continue
    m = re.match(r"^\s+([a-z][a-z0-9_]+)", l)
    if not m or l.strip().startswith("."): continue
    op = m.group(1)
    if op in ("s_set_vgpr_msb", "s_wait_alu", "s_delay_alu"): continue
    c = ("W" if op.startswith("v_wmma") else "E" if re.match(r"v_(exp|log|rcp)", op) else "P" if op.startswith("v_permlane")
         else "L" if op.startswith("ds_load") else "s" if op.startswith("ds_store") else "T" if op.startswith("tensor_")
         else "B" if op.startswith("s_barrier") else "t" if op == "s_wait_tensorcnt" else "d" if op == "s_wait_dscnt"
         else "n" if op in ("v_nop", "s_nop") else "b" if op.startswith("s_cbranch") or op == "s_branch" else "V" if op.startswith("v_") else "S")
    out.append(c)
s = "".join(out)
for k in range(0, len(s), w): print(s[k:k + w])
