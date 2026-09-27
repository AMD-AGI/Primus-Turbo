"""Main-loop instruction histogram of a FlyDSL 22_final_isa.s (pure text, CPU only).

Finds every back-edge (branch to an earlier label); the loop body is the text from the target
label to the branch. Prints each loop's span and a category histogram; nested loops are listed
separately (the outer pass loop of the paired kernel contains the KV loops).
Usage: python3 loop_hist.py <isa.s> [<isa.s> ...]
"""
import collections, re, sys

CATS = [
    ("wmma", lambda op: op.startswith("v_wmma")),
    ("v_exp", lambda op: op.startswith("v_exp")),
    ("v_nop", lambda op: op == "v_nop"),
    ("ds_*", lambda op: op.startswith("ds_")),
    ("tdm", lambda op: op.startswith("tensor_")),
    ("async_lds", lambda op: "async" in op),
    ("global/buffer", lambda op: op.startswith(("global_", "buffer_"))),
    ("s_barrier*", lambda op: op.startswith("s_barrier")),
    ("s_wait*", lambda op: op.startswith("s_wait")),
    ("s_set_vgpr_msb", lambda op: op == "s_set_vgpr_msb"),
    ("v_readlane/v_writelane", lambda op: op in ("v_readlane_b32", "v_writelane_b32")),
    ("VALU(other)", lambda op: op.startswith("v_")),
    ("SALU/other", lambda op: True),
]


def parse(path):
    lines = []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith((";", ".", "//")) and not re.match(r"\.LBB\d+_\d+:", s):
            continue
        lines.append(s)
    return lines


def cat(op):
    for name, f in CATS:
        if f(op):
            return name


def main(path):
    L = parse(path)
    label_at = {}
    for i, s in enumerate(L):
        m = re.match(r"(\.LBB\d+_\d+):", s)
        if m:
            label_at[m.group(1)] = i
    loops = []
    for i, s in enumerate(L):
        m = re.match(r"(s_branch|s_cbranch_\w+)\s+(\.LBB\d+_\d+)", s)
        if m and label_at.get(m.group(2), 1 << 30) < i:
            loops.append((label_at[m.group(2)], i, m.group(2)))
    # merge back-edges sharing a header: keep the widest span
    by_head = {}
    for a, b, lab in loops:
        if lab not in by_head or b > by_head[lab][1]:
            by_head[lab] = (a, b, lab)
    print(f"== {path}")
    for a, b, lab in sorted(by_head.values()):
        h = collections.Counter()
        for s in L[a:b + 1]:
            if s.endswith(":"):
                continue
            h[cat(s.split()[0])] += 1
        n = sum(h.values())
        print(f"loop {lab:10s} lines {a:5d}-{b:5d} insts {n:5d} | " +
              " ".join(f"{k}={h[k]}" for k, _ in CATS if h[k]))


for p in sys.argv[1:]:
    main(p)
