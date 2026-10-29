###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""shellbase -- the VGPRs the shell's code touches while generated state is live (between INIT and FINAL).

The generated regions keep their state (accumulators, K / V fragments, operands) in VGPRs above `base` from the INIT
marker to the FINAL marker. The shell may use any register elsewhere (the 64-row tail body, the epilogue after FINAL),
but every shell instruction on a path from INIT to FINAL must stay below `base`, and so must every value live across a
marker. shell_span() walks the CFG forward from the instruction after INIT, stops at FINAL, and returns the highest
VGPR referenced on the way (marker bodies excluded) plus the line numbers it visited."""

import re


def shell_span(lines, ins, bl, init_i, final_i, marker_lines):
    """(highest VGPR, visited instruction indices) of the shell code on every path INIT -> FINAL."""
    block_of = {}
    for b, (s, e, _) in enumerate(bl):
        for i in range(s, e + 1):
            block_of[i] = b
    hi, seen_i, seen_b = -1, set(), set()
    # start: the rest of INIT's block, then its successors
    work = [(block_of[init_i], init_i + 1)]
    while work:
        b, start = work.pop()
        s, e, succ = bl[b]
        stop = False
        for i in range(start, e + 1):
            if i == final_i:
                stop = True
                break
            if i in marker_lines:
                continue
            x = ins[i]
            if not x.mn or x.cls in ("LABEL", "DIR"):
                continue
            seen_i.add(i)
            for r in list(x.defs) + list(x.uses):
                hi = max(hi, r)
        if stop:
            continue
        for t in succ:
            if t not in seen_b:
                seen_b.add(t)
                work.append((t, bl[t][0]))
    return hi, seen_i
