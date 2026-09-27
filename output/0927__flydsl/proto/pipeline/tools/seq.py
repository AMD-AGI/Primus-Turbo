"""Run-length op sequence of a line range of an ISA file: seq.py <isa.s> <a> <b>"""
import re, sys
lines = open(sys.argv[1]).read().split("\n")[int(sys.argv[2]) - 1:int(sys.argv[3])]
out, prev, n = [], None, 0
for l in lines:
    m = re.match(r"^(\.LBB\d+_\d+):|\s+([a-z][a-z0-9_]+)", l)
    if not m or l.lstrip().startswith(".") and not m.group(1):
        continue
    op = m.group(1) or m.group(2)
    if op in ("s_set_vgpr_msb", "s_delay_alu", "s_wait_alu"):
        continue
    if op == prev:
        n += 1
    else:
        if prev:
            out.append(f"{prev}x{n}" if n > 1 else prev)
        prev, n = op, 1
out.append(f"{prev}x{n}" if n > 1 else prev)
print(" | ".join(out))
