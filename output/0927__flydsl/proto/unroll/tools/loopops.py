"""Opcode histogram of line range [a,b] of two ISA files, per tile (divide by n). usage: loopops.py f1 a1 b1 n1 f2 a2 b2 n2"""
import collections, re, sys
def h(f, a, b):
    c = collections.Counter()
    for l in open(f).read().split("\n")[int(a)-1:int(b)]:
        m = re.match(r"\s+([a-z][a-z0-9_]+)(\s|$)", l)
        if m and not l.lstrip().startswith("."): c[m.group(1)] += 1
    return c
A = h(*sys.argv[1:4]); na = int(sys.argv[4]); B = h(*sys.argv[5:8]); nb = int(sys.argv[8])
for op in sorted(A.keys() | B.keys(), key=lambda o: -abs(B[o]/nb - A[o]/na)):
    if A[op]/na != B[op]/nb: print(f"  {op:28s} {A[op]/na:7.1f} {B[op]/nb:7.1f}")
