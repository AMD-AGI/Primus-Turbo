###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.verify_obj -- the code object holds what the edited ``.s`` intends.

Compares the kernel function of the intended ``.s`` with the ``llvm-objdump -d`` listing of the assembled code
object, instruction by instruction: the mnemonic (``_e32`` / ``_e64`` suffixes ignored), every operand's true VGPR
numbers (the disassembler derives them from the ``s_set_vgpr_msb`` state, so an MSB mistake shows up as a register
mismatch), the scalar registers, every ``s_set_vgpr_msb`` immediate and every branch target (as an instruction
ordinal)."""

import re

from .isa import first_tok, function_range, load_any, regs_of, sregs, strip_comment

_MN = re.compile(r"^[a-z][a-z0-9_]*(::[a-z][a-z0-9_]*)?$")


def _seq(path):
    _, ins, _ = load_any(path)
    f, l = function_range(ins)
    # label -> ordinal of the next instruction
    ordinal, label_at, n = {}, {}, 0
    for k in range(f, l + 1):
        x = ins[k]
        if x.label:
            label_at[x.label] = n
        if x.mn and x.cls not in ("LABEL", "DIR", "CLAUSE") and _MN.match(x.mn) and x.mn != "s_code_end":
            ordinal[k] = n
            n += 1
    out = []
    for k in range(f, l + 1):
        x = ins[k]
        if k not in ordinal:
            continue
        mn = x.mn.replace("_e32", "").replace("_e64", "")
        if x.cls == "MSB":
            out.append((mn, (int(strip_comment(x.text).split()[1], 0),)))
            continue
        if mn.startswith(("s_cbranch", "s_branch")):
            t = strip_comment(x.ops[0]).strip() if x.ops else ""
            out.append((mn, ("->", label_at.get(t, "outside"))))
            continue
        regs = tuple((tuple(regs_of(first_tok(o))), tuple(sregs(first_tok(o)))) for o in x.ops if o != "::")
        out.append((mn, regs))
    return out


def compare(intended, objdump, log=print):
    """[] when equal, else the mismatching (index, intended, objdump) triples (first 8), plus ("length", n, m) if
    the instruction counts differ."""
    a, b = _seq(intended), _seq(objdump)
    bad = [i for i, (p, q) in enumerate(zip(a, b)) if p != q]
    log(f"  verify_obj: intended {len(a)} instructions, code object {len(b)}; mismatches {len(bad)}")
    out = [(i, a[i], b[i]) for i in bad[:8]]
    if len(a) != len(b):
        out.append(("length", len(a), len(b)))
    return out
