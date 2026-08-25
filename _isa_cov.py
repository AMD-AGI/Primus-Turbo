#!/usr/bin/env python3
"""Cycle-weighted LDS coverage of one loop arm.

pitfalls/13 §2026-08-19 established instruction-count coverage distance as the only static
quantity that tracks the wall on this body. Counting INSTRUCTIONS under-reads the cover of a
window full of MFMAs (16 cycles each) and over-reads one full of SALU, so this weights the
window by issue cycles and reports the residual stall against an assumed LDS latency.

usage: _isa_cov.py <isa.s> <label> [kernel] [lds_latency]
"""
import collections
import re
import sys

PATH, LABEL = sys.argv[1], sys.argv[2]
KERNEL = sys.argv[3] if len(sys.argv) > 3 else "flash_attn_bwd_dkdv_kernel_0"
LAT = int(sys.argv[4]) if len(sys.argv) > 4 else 120

W = {"MFMA": 16, "VALU": 4, "dsr": 4, "dsw": 4, "vmem": 4, "SALU": 1, "wait": 0, "bar": 0}


def family(op):
    if op.startswith("v_mfma"):
        return "MFMA"
    if op.startswith("ds_read"):
        return "dsr"
    if op.startswith("ds_write"):
        return "dsw"
    if op == "s_waitcnt":
        return "wait"
    if op.startswith(("buffer_", "global_", "flat_")):
        return "vmem"
    if op == "s_nop":
        return "nop"
    if op == "s_barrier":
        return "bar"
    if op.startswith("s_"):
        return "SALU"
    return "VALU"


def weight(s):
    op = s.split()[0]
    f = family(op)
    if f == "nop":
        m = re.search(r"s_nop\s+(\d+)", s)
        return int(m.group(1)) + 1 if m else 1
    return W.get(f, 4)


lines = open(PATH, errors="ignore").read().split("\n")
start = end = None
for i, ln in enumerate(lines):
    if ln.startswith(KERNEL + ":"):
        start = i
    if start is not None and end is None and i > start and ".Lfunc_end" in ln:
        end = i
body = [ln.strip() for ln in lines[start:end]]
head = next(i for i, ln in enumerate(body) if ln.strip().startswith(LABEL + ":"))
tail = next(
    i for i, ln in enumerate(body) if i > head and re.search(rf"s_c?branch\w*\s+{re.escape(LABEL)}\b", ln)
)
ins = [s for s in body[head : tail + 1] if s and not s.startswith((";", ".")) and not s.endswith(":")]
cyc = [weight(s) for s in ins]
pre = [0]
for c in cyc:
    pre.append(pre[-1] + c)
print(f"{LABEL}: {len(ins)} instructions, {pre[-1]} issue-cycles (MFMA 16 / VALU 4 / SALU 1)")

pend, cover, cover_i = [], {}, {}
for i, s in enumerate(ins):
    if s.startswith(("ds_read", "ds_write")):
        pend.append(i)
    elif s.startswith("s_waitcnt") and "lgkmcnt" in s:
        n = int(re.search(r"lgkmcnt\((\d+)\)", s).group(1))
        while len(pend) > n:
            j = pend.pop(0)
            cover[j] = pre[i] - pre[j]
            cover_i[j] = i - j


def key(i):
    s = ins[i]
    op = s.split()[0]
    m = re.search(r",\s*(v\d+)", s)
    return f"{op} {m.group(1) if m else '?'}"


tot = sum(max(0, LAT - c) for c in cover.values())
print(f"lds ops with a cover reading: {len(cover)}   est stall @lat={LAT}: {tot} cycles")
agg = collections.defaultdict(lambda: [0, 0, 0])
for i, c in cover.items():
    a = agg[key(i)]
    a[0] += 1
    a[1] += c
    a[2] += max(0, LAT - c)
print(" n  meancov  stall   op base")
for k, (n, cs, st) in sorted(agg.items(), key=lambda kv: -kv[1][2])[:24]:
    print(f"{n:3d} {cs / n:7.1f} {st:6d}   {k}")

print("\nworst individual reads (idx, cover_cycles, cover_instr):")
for i in sorted(cover, key=lambda j: cover[j])[:24]:
    print(f"  {i:5d} cov={cover[i]:4d}c/{cover_i[i]:3d}i  {ins[i]}")
