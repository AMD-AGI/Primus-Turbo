#!/usr/bin/env python3
"""Group the dkdv ISA's dQ emissions by basic block, to tell a STATIC duplicate apart
from an instruction that simply runs twice.

usage: _isa_blocks.py <isa.s> [mnemonic]
"""
import re
import sys

path = sys.argv[1]
mnem = sys.argv[2] if len(sys.argv) > 2 else "buffer_store_dwordx4"
lines = open(path).read().splitlines()

blocks, lbl, cur = [], "<entry>", []
for i, ln in enumerate(lines):
    s = ln.strip()
    if re.match(r"^(\.?[A-Za-z_][\w.$]*):\s*$", s):
        blocks.append((lbl, cur))
        lbl, cur = s[:-1], []
    cur.append((i, s))
blocks.append((lbl, cur))

tot = 0
for lb, ls in blocks:
    hits = [(i, s) for i, s in ls if s.startswith(mnem)]
    if not hits:
        continue
    nm = sum(1 for _, s in ls if s.startswith("v_mfma"))
    br = [s for _, s in ls if s.startswith(("s_branch", "s_cbranch"))]
    print(
        "block %-28s %s=%d mfma=%d lines %d-%d br=%s"
        % (lb, mnem, len(hits), nm, ls[0][0], ls[-1][0], br[:2])
    )
    for i, s in hits:
        print("      %6d  %s" % (i, s))
    tot += len(hits)
print("total", mnem, tot)
