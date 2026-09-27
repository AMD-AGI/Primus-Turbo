"""Split an ISA line range into basic blocks; per-block class counts. Usage: blocks.py isa.s start end"""
import re, sys
lines = open(sys.argv[1]).read().splitlines()
a, b = int(sys.argv[2]) - 1, int(sys.argv[3]) - 1
blocks, cur, start = [], [], a
def flush(end):
    global cur, start
    if cur:
        blocks.append((start + 1, end + 1, cur))
    cur = []
for i in range(a, b + 1):
    s = lines[i].strip()
    if re.match(r"^\.LBB[0-9_]+:", s):
        flush(i - 1); start = i; continue
    if not s or s.startswith((".", ";", "//")):
        continue
    op = s.split()[0]
    cur.append(op)
    if op.startswith("s_cbranch") or op == "s_branch":
        flush(i); start = i + 1
flush(b)
def c(ins):
    g = lambda f: sum(1 for o in ins if f(o))
    return dict(tot=len(ins), wmma=g(lambda o: o.startswith("v_wmma")), exp=g(lambda o: o.startswith("v_exp")),
                valu=g(lambda o: o.startswith("v_") and not o.startswith("v_wmma") and o != "v_nop"),
                nop=g(lambda o: o == "v_nop"), pkmul=g(lambda o: o == "v_pk_mul_f32"), perm=g(lambda o: o.startswith("v_permlane")),
                ds=g(lambda o: o.startswith("ds_")), tdm=g(lambda o: o.startswith("tensor_")),
                bar=g(lambda o: o.startswith("s_barrier")), wait=g(lambda o: o.startswith("s_wait")),
                delay=g(lambda o: o == "s_delay_alu"), msb=g(lambda o: o == "s_set_vgpr_msb"),
                salu=g(lambda o: o.startswith("s_") and not o.startswith(("s_wait", "s_barrier", "s_delay", "s_set_vgpr"))),
                last=ins[-1])
for s, e, ins in blocks:
    print(f"[{s}-{e}]", c(ins))
