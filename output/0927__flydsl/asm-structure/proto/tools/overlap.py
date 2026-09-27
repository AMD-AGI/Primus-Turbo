"""Overlap metric for a loop line range: how many v_exp / VALU issue within K instructions after a WMMA
(msb/s_wait_alu not counted), and the longest WMMA-free run. usage: overlap.py file.s lo hi [K]"""
import re, sys
f, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
L = open(f).read().split("\n"); ops = []
for i in range(lo - 1, hi):
    m = re.match(r"^\s+([a-z][a-z0-9_]+)", L[i])
    if m and not L[i].strip().startswith(".") and m.group(1) not in ("s_set_vgpr_msb", "s_wait_alu", "s_delay_alu"):
        ops.append(m.group(1))
last = -10**9; ne = nec = nv = nvc = 0; run = best = 0; wm = [k for k, o in enumerate(ops) if o.startswith("v_wmma")]
for k, o in enumerate(ops):
    if o.startswith("v_wmma"): last = k; run = 0; continue
    run += 1; best = max(best, run)
    if o.startswith("v_exp"): ne += 1; nec += (k - last <= K)
    elif o.startswith("v_") and o not in ("v_nop",): nv += 1; nvc += (k - last <= K)
print(f"v_exp within {K} after a WMMA: {nec}/{ne}   other VALU: {nvc}/{nv}   longest WMMA-free run: {best}   WMMA span: {wm[0] if wm else '-'}..{wm[-1] if wm else '-'} of {len(ops)}")
