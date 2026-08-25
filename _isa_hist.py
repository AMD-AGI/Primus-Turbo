#!/usr/bin/env python3
"""Per-basic-block opcode-class histogram of an ISA dump.

The campaign's own r9 finding is that wall follows the HOT BLOCK, not the kernel total, so
size/scheduling candidates have to be judged per `.LBB`. usage: _isa_hist.py <isa.s> [top_n]
"""
import collections
import re
import sys

CLASS = [
    ("mfma", r"^v_mfma"),
    ("ds_read_tr", r"^ds_read\w*_tr"),
    ("ds_read", r"^ds_read"),
    ("ds_write", r"^ds_write"),
    ("buf_load", r"^(buffer_load|global_load)"),
    ("buf_store", r"^(buffer_store|global_store)"),
    ("atomic", r"atomic"),
    ("accvgpr", r"^v_accvgpr"),
    ("v_mov", r"^v_mov"),
    ("v_cvt", r"^v_cvt"),
    ("v_exp", r"^v_exp"),
    ("v_perm", r"^v_perm"),
    ("valu", r"^v_"),
    ("s_wait", r"^s_wait"),
    ("s_barrier", r"^s_barrier"),
    ("sched", r"^(sched_|s_setprio|iglp|s_nop)"),
    ("salu", r"^s_"),
]


def classify(op):
    for name, pat in CLASS:
        if re.match(pat, op):
            return name
    return "other"


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    blocks = collections.OrderedDict()
    cur = "prologue"
    blocks[cur] = collections.Counter()
    size = collections.Counter()
    for ln in open(path, errors="ignore"):
        m = re.match(r"^(\.LBB[\w.]+):", ln.strip())
        if m:
            cur = m.group(1)
            blocks.setdefault(cur, collections.Counter())
            continue
        m = re.match(r"^\s+([a-z][\w.]*)\s", ln)
        if not m or ln.lstrip().startswith("."):
            continue
        op = m.group(1)
        if not re.match(r"^(s_|v_|ds_|buffer_|global_|flat_|scratch_|sched_|iglp)", op):
            continue
        blocks[cur][classify(op)] += 1
        size[cur] += 1
    tot = sum(size.values())
    print(f"total insts {tot} over {len(blocks)} blocks")
    for blk, n in size.most_common(top):
        c = blocks[blk]
        parts = " ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"{blk:>14} n={n:6d} ({100.0*n/tot:5.1f}%)  {parts}")
    agg = collections.Counter()
    for c in blocks.values():
        agg.update(c)
    print("KERNEL  " + " ".join(f"{k}={v}" for k, v in agg.most_common()))


if __name__ == "__main__":
    main()
