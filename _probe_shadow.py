#!/usr/bin/env python3
"""Which VALU instructions sit inside an MFMA run, per opcode and per exposed block.

`SQ_VALU_MFMA_COEXEC_CYCLES` says how much VALU co-executes but not WHICH; this reads the
same question off the ISA. A VALU instruction is "in shadow" when it lies between two MFMAs
that are at most `gap` non-MFMA instructions apart, i.e. inside a run the MFMA pipe is busy
through. Everything else is exposed, and the exposed VALU is reported as contiguous BLOCKS so
a named source block can be recognised by its opcode mix and attacked by name.

The stage dump concatenates every kernel of a module with no separator (pitfalls/09), so the
stream is split on `.amdhsa_kernel` before anything is counted.

usage: _probe_shadow.py <dump_dir> [kernel_substr] [gap] [min_block]
"""
import collections
import glob
import os
import re
import sys

MFMA = "v_mfma"
VALU = re.compile(r"^v_")


def _kernels(path):
    """[(name, [opcode...])] for one dump file, split at the .amdhsa_kernel directives."""
    body, cuts = [], []
    for line in open(path):
        m = re.match(r"^\s*\.amdhsa_kernel\s+(\S+)", line)
        if m:
            cuts.append((m.group(1), len(body)))
            continue
        m = re.match(r"^\s+([a-z][a-z0-9_]+)\s", line)
        if m and not m.group(1).startswith(("amdhsa", "section", "size", "type", "globl")):
            body.append(m.group(1))
    # Directives follow their kernel's code, so a cut ends the segment that precedes it.
    out, lo = [], 0
    for name, hi in cuts:
        out.append((name, body[lo:hi]))
        lo = hi
    return out


def _shadow_mask(ops, gap):
    """True where the MFMA pipe is busy: between two MFMAs at most `gap` instructions apart."""
    idx = [i for i, o in enumerate(ops) if o.startswith(MFMA)]
    mask = [False] * len(ops)
    for a, b in zip(idx, idx[1:]):
        if b - a - 1 <= gap:
            for i in range(a, b + 1):
                mask[i] = True
    for i in idx:
        mask[i] = True
    return mask


def main():
    root = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else "dkdv"
    gap = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    minblk = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    for path in sorted(glob.glob(os.path.join(root, "**", "*isa*.s"), recursive=True)):
        for name, ops in _kernels(path):
            if filt not in name and filt not in os.path.basename(os.path.dirname(path)):
                continue
            mask = _shadow_mask(ops, gap)
            tot = collections.Counter()
            hid = collections.Counter()
            for o, m in zip(ops, mask):
                if not VALU.match(o):
                    continue
                tot[o] += 1
                hid[o] += bool(m)
            nv = sum(tot.values())
            nh = sum(hid.values())
            print(f"== {name} ({len(ops)} insts, gap={gap}) VALU {nv} hidden {nh} = {100.0 * nh / max(1, nv):.1f}%")
            for o, n in tot.most_common(18):
                print(f"   {o:28s} {n:6d} hidden {hid[o]:6d} = {100.0 * hid[o] / n:5.1f}%")
            # Exposed VALU as contiguous blocks, so a source block is recognisable by its mix.
            blocks, cur = [], []
            for i, (o, m) in enumerate(zip(ops, mask)):
                if m:
                    if cur:
                        blocks.append(cur)
                        cur = []
                    continue
                cur.append((i, o))
            if cur:
                blocks.append(cur)
            big = [b for b in blocks if sum(1 for _, o in b if VALU.match(o)) >= minblk]
            print(f"   -- {len(blocks)} exposed segments, {len(big)} with >= {minblk} VALU")
            sig = collections.Counter()
            for b in big:
                c = collections.Counter(o for _, o in b if VALU.match(o))
                sig["|".join(f"{o}:{n}" for o, n in sorted(c.items()))] += 1
            for s, n in sig.most_common(12):
                nvb = sum(int(x.split(":")[1]) for x in s.split("|"))
                print(f"   x{n:3d} VALU {nvb:4d}  {s}")


if __name__ == "__main__":
    main()
