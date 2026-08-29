#!/usr/bin/env python3
"""Instruction family mix and issue-cycle budget of one kernel dump.

The MFMA pipe's occupancy is what this body is short of, so what matters is how many
NON-MFMA issue cycles sit between the atoms and whether they can co-execute.

usage: _isa_fam.py <isa.s>
"""
import collections
import sys

W = {"MFMA": 16, "TRANS": 8, "VALU": 4, "ACCMOV": 4, "LDS": 4, "VMEM": 4,
     "SALU": 1, "WAIT": 0, "NOP": 2, "other": 1}


def fam(o):
    if o.startswith("v_mfma"):
        return "MFMA"
    if o.startswith("v_accvgpr"):
        return "ACCMOV"
    if o.startswith(("v_exp", "v_rcp", "v_log", "v_sqrt")):
        return "TRANS"
    if o.startswith("v_"):
        return "VALU"
    if o.startswith("ds_"):
        return "LDS"
    if o.startswith(("buffer_", "global_", "flat_")):
        return "VMEM"
    if o.startswith("s_waitcnt"):
        return "WAIT"
    if o.startswith("s_nop"):
        return "NOP"
    if o.startswith("s_"):
        return "SALU"
    return "other"


L = [ln.strip() for ln in open(sys.argv[1])
     if ln.strip() and not ln.strip().startswith((";", "//", ".", "/*"))]
ops = [ln.split()[0] for ln in L]
c = collections.Counter(fam(o) for o in ops)
tot = sum(c.values())
cyc = {k: v * W[k] for k, v in c.items()}
ctot = sum(cyc.values())
for k, v in c.most_common():
    print("%-8s n=%6d %5.1f%%   cyc=%7d %5.1f%%" % (k, v, 100 * v / tot, cyc[k], 100 * cyc[k] / ctot))
print("total n=%d  issue-cycles=%d  MFMA share=%.1f%%" % (tot, ctot, 100 * cyc["MFMA"] / ctot))
print("top VALU:", collections.Counter(o for o in ops if fam(o) == "VALU").most_common(12))
