"""Per-basic-block instruction mix inside the hottest loop: python wip/isa_blocks.py FILE.s"""

import collections
import re
import sys

lines = open(sys.argv[1]).read().splitlines()
labels = {}
for i, ln in enumerate(lines):
    m = re.match(r"^(\.LBB\d+_\d+):", ln)
    if m:
        labels[m.group(1)] = i
loops = []
for i, ln in enumerate(lines):
    m = re.match(r"^\s+s_cbranch_\w+\s+(\.LBB\d+_\d+)", ln) or re.match(r"^\s+s_branch\s+(\.LBB\d+_\d+)", ln)
    if m and m.group(1) in labels and labels[m.group(1)] < i:
        loops.append((i - labels[m.group(1)], labels[m.group(1)], i))
_, lo, hi = max(loops)

KEYS = ["mfma", "ds_read", "acc_rd", "acc_wr", "acc_mov", "v_mov", "scratch", "v_exp", "valu", "wait", "nop", "bar", "dma"]


def cat(op):
    if op.startswith("v_mfma"):
        return "mfma"
    if op.startswith(("ds_read", "ds_load")):
        return "ds_read"
    if op.startswith("v_accvgpr_read"):
        return "acc_rd"
    if op.startswith("v_accvgpr_write"):
        return "acc_wr"
    if op.startswith("v_accvgpr_mov"):
        return "acc_mov"
    if op.startswith("v_mov"):
        return "v_mov"
    if op.startswith("scratch_"):
        return "scratch"
    if op.startswith("v_exp"):
        return "v_exp"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("s_waitcnt"):
        return "wait"
    if op.startswith("s_nop"):
        return "nop"
    if op.startswith("s_barrier"):
        return "bar"
    if op.startswith("buffer_load"):
        return "dma"
    return None


blocks = []
cur_name, cur = "entry", collections.Counter()
term = ""
for ln in lines[lo:hi + 1]:
    s = ln.strip()
    m = re.match(r"^(\.LBB\d+_\d+):", s)
    if m:
        blocks.append((cur_name, cur, term))
        cur_name, cur, term = m.group(1), collections.Counter(), ""
        continue
    if not s or s.startswith((";", ".")):
        continue
    op = s.split()[0]
    c = cat(op)
    if c:
        cur[c] += 1
    cur["total"] += 1
    if op.startswith(("s_cbranch", "s_branch")):
        blocks.append((cur_name, cur, s))
        cur_name, cur, term = cur_name.split("+")[0] + "+ft", collections.Counter(), ""
blocks.append((cur_name, cur, term))
print(f"{'block':10s} " + " ".join(f"{k:>7s}" for k in KEYS + ["total"]) + "  terminator")
for name, c, t in blocks:
    print(f"{name:10s} " + " ".join(f"{c.get(k, 0):7d}" for k in KEYS + ["total"]) + f"  {t}")
