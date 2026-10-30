###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""shellbase -- the VGPRs the shell's code references while the generated state is live (from INIT to FINAL).

The generated regions keep their state (accumulators, K / V fragments, operands) in VGPRs at or above `base` from the
INIT marker to the FINAL marker. The shell may use any register elsewhere (e.g. the epilogue after FINAL), but every
shell instruction on a path from INIT to FINAL must stay below `base`."""


def shell_span(ins, bl, init_i, final_i):
    """(highest VGPR, visited instruction indices) of the shell's code on every CFG path from the instruction after
    INIT (line init_i) to FINAL (line final_i); bl: the blocks of asmkit.expert.blocks."""
    block_of = {}
    for b, (s, e, _) in enumerate(bl):
        for i in range(s, e + 1):
            block_of[i] = b
    hi, seen_i, seen_b = -1, set(), set()
    work = [(block_of[init_i], init_i + 1)]  # the rest of INIT's block, then its successors
    while work:
        b, start = work.pop()
        _, e, succ = bl[b]
        for i in range(start, e + 1):
            if i == final_i:
                break
            x = ins[i]
            if x.mn:
                seen_i.add(i)
                hi = max([hi] + x.defs + x.uses)
        else:
            for t in succ:
                if t not in seen_b:
                    seen_b.add(t)
                    work.append((t, bl[t][0]))
    return hi, seen_i
