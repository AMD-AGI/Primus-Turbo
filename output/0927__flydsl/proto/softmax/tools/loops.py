"""Find the main loops (backward branches) in a FlyDSL 22_final_isa.s and histogram their bodies.
Usage: python3 loops.py <isa.s> [--dump N]   (dump N = print loop N body to stdout)"""
import collections, re, sys

lines = open(sys.argv[1]).read().splitlines()
lab = {}
for i, l in enumerate(lines):
    m = re.match(r"^(\.LBB[0-9_]+):", l)
    if m:
        lab[m.group(1)] = i
loops = []
for i, l in enumerate(lines):
    m = re.match(r"\s+(s_cbranch_\w+|s_branch)\s+(\.LBB[0-9_]+)", l)
    if m and m.group(2) in lab and lab[m.group(2)] < i:
        loops.append((lab[m.group(2)], i))
# keep outermost-ish large loops (main loops contain WMMA)
def insts(a, b):
    out = []
    for l in lines[a:b + 1]:
        s = l.strip()
        if not s or s.startswith((".", ";", "//")) or s.endswith(":"):
            continue
        out.append(s.split()[0])
    return out

CLS = [
    ("WMMA", lambda o: o.startswith("v_wmma")),
    ("v_exp", lambda o: o.startswith("v_exp")),
    ("v_nop", lambda o: o == "v_nop"),
    ("v_permlane", lambda o: o.startswith("v_permlane")),
    ("v_pk_fma_f32", lambda o: o == "v_pk_fma_f32"),
    ("v_pk_add_f32", lambda o: o == "v_pk_add_f32"),
    ("v_pk_mul_f32", lambda o: o == "v_pk_mul_f32"),
    ("v_fmamk/fma scalar", lambda o: "fmamk" in o or o in ("v_fma_f32", "v_fmac_f32_e32", "v_dual_fmac_f32")),
    ("v_add_f32 (incl dual)", lambda o: o in ("v_add_f32_e32", "v_dual_add_f32")),
    ("v_mov (incl dual/b64)", lambda o: o.startswith("v_mov") or o == "v_dual_mov_b32"),
    ("VALU total (v_*, excl WMMA/nop)", lambda o: o.startswith("v_") and not o.startswith("v_wmma") and o != "v_nop"),
    ("ds_load*", lambda o: o.startswith("ds_load")),
    ("ds_* other", lambda o: o.startswith("ds_") and not o.startswith("ds_load")),
    ("TDM (tensor_*)", lambda o: o.startswith("tensor_")),
    ("s_barrier*", lambda o: o.startswith("s_barrier")),
    ("s_wait_tensorcnt", lambda o: o == "s_wait_tensorcnt"),
    ("s_wait_dscnt", lambda o: o == "s_wait_dscnt"),
    ("s_wait_* other", lambda o: o.startswith("s_wait") and o not in ("s_wait_tensorcnt", "s_wait_dscnt")),
    ("s_delay_alu", lambda o: o == "s_delay_alu"),
    ("s_set_vgpr_msb", lambda o: o == "s_set_vgpr_msb"),
    ("s_cbranch*/s_branch", lambda o: o.startswith("s_cbranch") or o == "s_branch"),
    ("SALU other", lambda o: o.startswith("s_") and not (o.startswith(("s_wait", "s_barrier", "s_cbranch", "s_branch")) or o in ("s_delay_alu", "s_set_vgpr_msb"))),
    ("total", lambda o: True),
]
main = []
for a, b in loops:
    ins = insts(a, b)
    if sum(o.startswith("v_wmma") for o in ins) >= 32:
        main.append((a, b, ins))
print(f"{len(main)} WMMA loops (label line, branch line, insts):", [(a + 1, b + 1, len(i)) for a, b, i in main])
hdr = "class".ljust(34) + "".join(f"L{k}".rjust(7) for k in range(len(main)))
print(hdr)
for name, f in CLS:
    print(name.ljust(34) + "".join(str(sum(1 for o in ins if f(o))).rjust(7) for _, _, ins in main))
if "--dump" in sys.argv:
    k = int(sys.argv[sys.argv.index("--dump") + 1])
    a, b, _ = main[k]
    print("\n".join(f"{n + 1}: {lines[n]}" for n in range(a, b + 1)))
