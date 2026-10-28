###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Static check of WMMA operand-reuse hints in a compiler .s (CPU only, run before a GPU sees the kernel).

    python3 -m hkl.reusecheck build/mla_bwd/<rung>/k_dqds_c.s [...]

matrix_a_reuse / matrix_b_reuse tell the hardware that the A / B operand is the previous WMMA's: the
instruction must follow another v_wmma in straight-line code (no label, branch or s_setpc in between: the
dynamic predecessor must be the static one) whose A / B operand is the same full register range (the
/*v[..]*/ comment when s_set_vgpr_msb applies), and nothing in between may write that range. Prints every
violation; exit 1 if any.
"""

import re
import sys

WMMA = re.compile(r"^\s*v_wmma_\S+\s+(.*)$")
FULL = re.compile(r"(v\[\d+:\d+\]|v\d+)(?:\s*/\*(v\[\d+:\d+\]|v\d+)\*/)?")


def regs(r):
    m = re.match(r"v\[(\d+):(\d+)\]", r)
    if m:
        return set(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.match(r"v(\d+)", r)
    return {int(m.group(1))} if m else set()


def operands(text):
    """Full-number register ranges of the comma-separated operands (comment wins over the encoded name)."""
    out = []
    for part in text.split(","):
        m = FULL.search(part)
        out.append(regs(m.group(2) or m.group(1)) if m else set())
    return out


def dst_full(line):
    """Registers an instruction writes: its first operand when it is a vector register (rough, conservative)."""
    t = line.strip().split(None, 1)
    if len(t) < 2 or not t[0].startswith(("v_", "ds_load", "global_load", "buffer_load", "scratch_load")):
        return set()
    m = FULL.search(t[1].split(",")[0])
    return regs(m.group(2) or m.group(1)) if m else set()


def check(path):
    bad, n_hint, prev = [], 0, None  # prev: (line no, [D, A, B, C]) of the last WMMA in straight-line code
    written = set()  # registers written since prev
    for no, line in enumerate(open(path), 1):
        s = line.split(";")[0].rstrip()
        if not s.strip():
            continue
        if re.match(r"^\S.*:$", s) or re.match(r"^\s*(s_cbranch|s_branch|s_setpc|s_endpgm)", s):
            prev, written = None, set()
            continue
        m = WMMA.match(s)
        if m:
            ops = operands(m.group(1).split(" matrix_")[0])
            for k, idx in (("matrix_a_reuse", 1), ("matrix_b_reuse", 2)):
                if k in s:
                    n_hint += 1
                    if prev is None:
                        bad.append(f"{path}:{no}: {k} with no straight-line previous WMMA")
                    elif prev[1][idx] != ops[idx]:
                        bad.append(f"{path}:{no}: {k} but operand differs from line {prev[0]}")
                    elif prev[1][idx] & written:
                        bad.append(f"{path}:{no}: {k} but operand rewritten since line {prev[0]}")
            prev, written = (no, ops), set()
            continue
        written |= dst_full(s)
    return n_hint, bad


def main(argv=None):
    rc = 0
    for p in (argv if argv is not None else sys.argv[1:]):
        n, bad = check(p)
        print(f"{p}: {n} reuse hints, {len(bad)} violations")
        for b in bad[:20]:
            print("  " + b)
        rc |= bool(bad)
    return rc


if __name__ == "__main__":
    sys.exit(main())
