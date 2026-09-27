"""Find loops (backward branches) in a 22_final_isa.s and print a per-loop instruction histogram.
usage: loop_stats.py <isa.s>  -> one row per loop: label, line span, counts per class."""
import collections, re, sys

lines = open(sys.argv[1]).read().split("\n")
labels = {}
for i, l in enumerate(lines):
    m = re.match(r"^(\.LBB\d+_\d+):", l)
    if m:
        labels[m.group(1)] = i
loops = {}
for i, l in enumerate(lines):
    m = re.match(r"\s+s_(?:cbranch_\w+|branch)\s+(\.LBB\d+_\d+)", l)
    if m and m.group(1) in labels and labels[m.group(1)] < i:
        lab = m.group(1)
        loops[lab] = max(loops.get(lab, 0), i)

CLASSES = [
    ("wmma", lambda o: o.startswith("v_wmma")),
    ("v_exp", lambda o: o.startswith("v_exp")),
    ("v_nop", lambda o: o == "v_nop" or o.startswith("s_nop")),
    ("permlane", lambda o: "permlane" in o),
    ("v_cvt", lambda o: o.startswith("v_cvt")),
    ("v_pk", lambda o: o.startswith("v_pk")),
    ("VALU(other)", lambda o: o.startswith("v_")),
    ("ds_load", lambda o: o.startswith("ds_load")),
    ("ds_other", lambda o: o.startswith("ds_")),
    ("TDM", lambda o: o.startswith("tensor_")),
    ("s_barrier", lambda o: o.startswith("s_barrier")),
    ("s_wait_dscnt", lambda o: o == "s_wait_dscnt"),
    ("s_wait_tensor", lambda o: o == "s_wait_tensorcnt"),
    ("s_wait(other)", lambda o: o.startswith("s_wait")),
    ("branch", lambda o: o.startswith("s_cbranch") or o == "s_branch"),
    ("SALU", lambda o: o.startswith("s_")),
    ("other", lambda o: True),
]


def classify(op):
    for n, f in CLASSES:
        if f(op):
            return n


def hist(a, b):
    h = collections.Counter()
    for l in lines[a:b + 1]:
        m = re.match(r"\s+([a-z][a-z0-9_]+)(\s|$)", l)
        if m and not l.lstrip().startswith("."):
            h[classify(m.group(1))] += 1
    return h


if __name__ == "__main__":
    hdr = ["loop", "lines", "n"] + [c for c, _ in CLASSES]
    print("\t".join(hdr))
    for lab, end in sorted(loops.items(), key=lambda kv: labels[kv[0]]):
        a = labels[lab]
        h = hist(a, end)
        print("\t".join([lab, f"{a+1}-{end+1}", str(sum(h.values()))] + [str(h[c]) for c, _ in CLASSES]))
