#!/usr/bin/env python3
"""Loop extents and code SIZE of a dumped kernel, for judging instruction-fetch pressure.

The fused dkdv body unrolls the whole GQA head group at trace time, so its q loop carries
GQA_GROUP_SIZE copies of every GEMM. gfx950 has a 32 KB L1 instruction cache per CU pair;
a loop whose body exceeds it re-streams its own code from L2 on every trip, and at one wave
per SIMD there is no sibling wave to hide that fetch behind.

usage: _isa_loops.py <path to 21_final_isa.s>
"""
import re
import sys

INS = re.compile(r"^\s+(s_|v_|ds_|buffer_|global_|flat_|scratch_)")
LAB = re.compile(r"^(\.?[A-Za-z_$][\w$.]*):")
BR = re.compile(r"\s+s_(?:cbranch\w*|branch)\s+(\S+)")


def isize(line):
    """gfx9 instruction bytes: 8 with a 32-bit literal or a VOP3/VOP3P encoding, else 4."""
    op = line.split()[0]
    if op.startswith(("v_mfma", "ds_", "buffer_", "global_", "flat_", "scratch_")):
        return 8
    if "_e64" in op or "0x" in line:
        return 8
    return 4


def main():
    lines = open(sys.argv[1]).read().splitlines()
    labels, num, byte, tot = {}, {}, {}, 0
    n = 0
    for i, l in enumerate(lines):
        m = LAB.match(l)
        if m:
            labels[m.group(1)] = (n, tot)
        if INS.match(l):
            num[i], byte[i] = n, tot
            n += 1
            tot += isize(l)
    print(f"instructions {n}  code_bytes {tot}  ({tot/1024:.1f} KB, L1I is 32 KB)")
    loops = []
    for i, l in enumerate(lines):
        m = BR.match(l)
        if m and m.group(1) in labels and i in num and labels[m.group(1)][0] < num[i]:
            a, ab = labels[m.group(1)]
            loops.append((num[i] - a, byte[i] - ab, m.group(1)))
    print("backward branches, innermost first (span_instr, span_bytes, label):")
    for span, sb, name in sorted(loops):
        print(f"  {span:7d} {sb:9d}  ({sb/1024:6.1f} KB)  {name}")


if __name__ == "__main__":
    main()
