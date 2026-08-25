#!/usr/bin/env python3
"""Instruction MIX of a dumped kernel, whole-kernel and inside each backward-branch span.

At one wave per SIMD the wave itself is the only issue source, so what the body costs is
every instruction it issues, not just its MFMAs. This groups the ISA by what the group
would cost to remove, so a candidate can be aimed at the fattest removable class.

usage: _isa_mix.py <path to 21_final_isa.s> [top_n_loops]
"""
import re
import sys
from collections import Counter

INS = re.compile(r"^\s+(s_|v_|ds_|buffer_|global_|flat_|scratch_)")
LAB = re.compile(r"^(\.?[A-Za-z_$][\w$.]*):")
BR = re.compile(r"\s+s_(?:cbranch\w*|branch)\s+(\S+)")


def klass(op):
    if op.startswith("v_mfma"):
        return "MFMA"
    if op.startswith("ds_"):
        return "LDS"
    if op.startswith(("buffer_", "global_", "flat_", "scratch_")):
        return "VMEM"
    if op.startswith("v_exp"):
        return "EXP"
    if op.startswith(("v_cvt_pk", "v_pack", "v_perm", "v_lshl_or", "v_and_or")):
        return "PACK"
    if op.startswith(("v_cmp", "v_cndmask")):
        return "MASK"
    if op.startswith(("v_mov", "v_accvgpr")):
        return "MOV"
    if op.startswith(("v_add", "v_sub", "v_mul", "v_fma", "v_max", "v_min", "v_lshl", "v_lshr", "v_and", "v_or", "v_xor", "v_bfe", "v_mad", "v_dot", "v_rcp", "v_ldexp", "v_cvt")):
        return "VALU"
    if op.startswith("s_waitcnt") or op.startswith("s_barrier"):
        return "WAIT"
    if op.startswith("s_"):
        return "SALU"
    return "OTHER" + ":" + op


def main():
    lines = open(sys.argv[1]).read().splitlines()
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    labels, idx, ops = {}, {}, []
    for i, l in enumerate(lines):
        m = LAB.match(l)
        if m:
            labels[m.group(1)] = len(ops)
        if INS.match(l):
            idx[i] = len(ops)
            ops.append(l.split()[0])
    tot = Counter(klass(o) for o in ops)
    n = len(ops)
    print(f"== whole kernel {n} instructions")
    for k, c in tot.most_common():
        print(f"   {k:8s} {c:6d}  {100.0*c/n:5.1f}%")
    loops = []
    for i, l in enumerate(lines):
        m = BR.match(l)
        if m and m.group(1) in labels and i in idx and labels[m.group(1)] < idx[i]:
            loops.append((idx[i] - labels[m.group(1)], labels[m.group(1)], idx[i], m.group(1)))
    for span, a, b, name in sorted(loops, reverse=True)[:topn]:
        c = Counter(klass(o) for o in ops[a:b])
        print(f"== loop {name} span {span} instructions")
        for k, v in c.most_common():
            print(f"   {k:8s} {v:6d}  {100.0*v/span:5.1f}%")
        sub = Counter(ops[a:b])
        print("   top ops:", ", ".join(f"{o}={v}" for o, v in sub.most_common(12)))


if __name__ == "__main__":
    main()
