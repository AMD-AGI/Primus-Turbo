#!/usr/bin/env python3
"""Instruction histogram of an ISA dump, grouped by what the instruction COSTS.

At 462 unified registers the dkdv body runs one wave per SIMD, so nothing hides behind a
sibling wave and every non-MFMA instruction is issue time the MFMA pipe does not get. This
says which category to attack.

usage: _isa_hist.py <isa.s> [top]
"""
import collections
import re
import sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 24

CATS = (
    ("mfma", r"^v_mfma"),
    ("ds_read", r"^ds_read"),
    ("ds_write", r"^ds_write"),
    ("buffer_load", r"^buffer_load"),
    ("buffer_store", r"^buffer_store"),
    ("global/flat", r"^(global|flat)_"),
    ("accvgpr", r"^v_accvgpr"),
    ("permlane/dpp", r"^(v_permlane|v_mov_b32_dpp|ds_bpermute|ds_swizzle)"),
    ("exp/trans", r"^v_(exp|log|rcp|rsq|sqrt)"),
    ("v_cvt/pack", r"^v_(cvt|pack|pk_)"),
    ("valu", r"^v_"),
    ("salu", r"^s_(add|sub|mul|and|or|xor|lshl|lshr|ashr|mov|cmp|cselect|min|max|bfe|not|nand|abs|sext|bitcmp|cbranch_scc|getpc|setpc|pack)"),
    ("s_wait/barrier", r"^s_(waitcnt|barrier|setprio|sleep|nop|sched|delay)"),
    ("branch", r"^s_(branch|cbranch|endpgm|swappc)"),
    ("s_load", r"^s_load"),
)

hist = collections.Counter()
mnem = collections.Counter()
for ln in open(path):
    t = ln.strip()
    if not t or t.startswith((".", ";", "/", "//")) or t.endswith(":"):
        continue
    op = t.split()[0]
    if not re.match(r"^[a-z]", op):
        continue
    mnem[op] += 1
    for name, pat in CATS:
        if re.match(pat, op):
            hist[name] += 1
            break
    else:
        hist["other"] += 1

tot = sum(hist.values())
m = hist.get("mfma", 1)
print("total %d instructions, %d mfma -> %.2f non-mfma per mfma" % (tot, m, (tot - m) / m))
for k, v in hist.most_common():
    print("  %-16s %6d  %5.1f%%   %5.2f / mfma" % (k, v, 100 * v / tot, v / m))
print("top mnemonics:")
for k, v in mnem.most_common(top):
    print("  %-32s %6d" % (k, v))
