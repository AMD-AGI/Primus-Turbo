"""Count v_exp that sit within W real instructions after a v_wmma inside a line range.
usage: overlap.py <isa.s> <a> <b> [W=8]"""
import re, sys
lines = open(sys.argv[1]).read().split("\n")[int(sys.argv[2]) - 1:int(sys.argv[3])]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 8
ops = []
for l in lines:
    m = re.match(r"\s+([a-z][a-z0-9_]+)", l)
    if m and m.group(1) not in ("s_set_vgpr_msb", "s_delay_alu", "s_wait_alu"):
        ops.append(m.group(1))
last_w, n_exp, n_in = -10**9, 0, 0
for i, o in enumerate(ops):
    if o.startswith("v_wmma"):
        last_w = i
    elif o.startswith("v_exp"):
        n_exp += 1
        n_in += (i - last_w) <= W
print(f"v_exp {n_exp}, within {W} instr after a WMMA: {n_in}")
