"""Instruction histogram of an ISA line range (one loop body), scaled to 256 KV.
usage: hist.py file.s lo hi kv_per_iter
Rescale if-bodies (blocks jumped over by s_cbranch_vcc*, >=75% v_pk_mul/v_mul) are reported separately
(they run only when the deferred-rescale ballot fires). TDM issue blocks are counted in the main path."""
import collections, re, sys
f, lo, hi, kv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
L = open(f).read().split("\n")
ins = []  # (line, op, text)
for i in range(lo - 1, hi):
    l = L[i]
    m = re.match(r"^(\.LBB\w+):", l)
    if m: ins.append((i + 1, "LABEL", m.group(1))); continue
    m = re.match(r"^\s+([a-z][a-z0-9_]+)(.*)", l)
    if m and not l.strip().startswith("."): ins.append((i + 1, m.group(1), m.group(2)))
# conditional rescale bodies
skip = set()
for k, (ln, op, rest) in enumerate(ins):
    if op.startswith("s_cbranch"):
        tgt = rest.strip().split()[0]
        body = []
        for j in range(k + 1, len(ins)):
            if ins[j][1] == "LABEL" and ins[j][2] == tgt: break
            if ins[j][1].startswith("s_cbranch") or ins[j][1] == "s_branch": body = []; break
            body.append(j)
        ops = [ins[j][1] for j in body if ins[j][1] not in ("s_set_vgpr_msb", "LABEL")]
        if ops and sum(o in ("v_pk_mul_f32", "v_mul_f32_e32", "v_mov_b32_e32", "v_nop") for o in ops) >= 0.75 * len(ops):
            skip.update(body)
def cls(op):
    if op.startswith("v_wmma"): return "WMMA"
    if re.match(r"v_exp", op): return "v_exp"
    if re.match(r"v_(log|rcp|sqrt|rsq)", op): return "trans_other"
    if op.startswith("v_permlane"): return "v_permlane"
    if op == "s_set_vgpr_msb": return "s_set_vgpr_msb"
    if op in ("v_nop", "s_nop"): return "nop"
    if op.startswith("ds_load"): return "ds_load"
    if op.startswith("ds_store"): return "ds_store"
    if op.startswith("ds_"): return "ds_other"
    if op.startswith("tensor_"): return "TDM"
    if op.startswith("s_barrier"): return op
    if op.startswith("s_wait_"): return op
    if op.startswith("v_"): return "VALU"
    if op.startswith("s_cbranch") or op == "s_branch": return "branch"
    if op.startswith("s_"): return "SALU"
    return "other:" + op
main, cond = collections.Counter(), collections.Counter()
for k, (ln, op, rest) in enumerate(ins):
    if op == "LABEL": continue
    (cond if k in skip else main)[cls(op)] += 1
sc = 256 / kv
order = ["WMMA", "v_exp", "trans_other", "VALU", "v_permlane", "ds_load", "ds_store", "ds_other", "TDM", "SALU", "branch",
         "s_barrier_signal", "s_barrier_wait", "s_barrier", "s_wait_tensorcnt", "s_wait_dscnt", "s_wait_alu", "s_wait_loadcnt",
         "s_wait_kmcnt", "nop", "s_set_vgpr_msb"]
keys = order + sorted(k for k in set(main) | set(cond) if k not in order)
tot_m = sum(main.values()); tot_c = sum(cond.values())
print(f"range {lo}-{hi}  kv/iter={kv}  scale x{sc:g}")
print(f"{'class':20s} {'main/256KV':>11s} {'rescale-if/256KV':>17s}")
for k in keys:
    if main[k] or cond[k]: print(f"{k:20s} {main[k]*sc:11g} {cond[k]*sc:17g}")
print(f"{'TOTAL':20s} {tot_m*sc:11g} {tot_c*sc:17g}")
