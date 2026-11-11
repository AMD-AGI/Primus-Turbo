"""Per-basic-block cycle totals of an ATT stats CSV, with block labels from the compiler ISA.

python wip/att_regions.py CSV ISA [top | .LBBx_y]

The CSV's instructions (address order) are aligned with the ISA's by opcode; every block gets its
summed latency / stall per wave and its s_barrier / s_waitcnt share. With a block label, prints
that block's instructions with latency / stall per visit instead."""
import csv
import re
import sys
from collections import OrderedDict

csv_path, isa_path = sys.argv[1], sys.argv[2]
top_block = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].startswith(".") else None
top = int(sys.argv[3]) if len(sys.argv) > 3 and not top_block else 40

rows = []
for r in csv.DictReader(open(csv_path)):
    ins = r["Instruction"].strip()
    if not ins or ins.startswith(";"):
        continue
    rows.append((int(r["Vaddr"]), ins, int(r["Hitcount"]), int(r["Latency"]), int(r["Stall"]), int(r["Idle"])))
rows.sort()
waves = next(h for _, _, h, _, _, _ in rows if h > 0)

isa = []
label = "entry"
in_text = False
for line in open(isa_path):
    s = line.split(";")[0].rstrip()
    if re.match(r"^flash\w*:", s):
        in_text = True
        continue
    if not in_text:
        continue
    if s.startswith(".Lfunc_end"):
        break
    m = re.match(r"^(\.LBB\d+_\d+):", s)
    if m:
        label = m.group(1)
        continue
    t = s.strip()
    if not t or t.startswith(".") or t.endswith(":"):
        continue
    isa.append((label, t.split()[0]))

ops = [r[1].split()[0] for r in rows]
labels, j = [], 0
for i, op in enumerate(ops):
    while j < len(isa) and isa[j][1] != op and isa[j][1] == "s_nop":
        j += 1
    if j < len(isa) and isa[j][1] == op:
        labels.append(isa[j][0])
        j += 1
    elif op == "s_nop":
        labels.append(isa[j - 1][0] if j else "entry")
    else:
        sys.exit(f"alignment failed at {i}: csv {ops[i:i + 3]} isa {isa[j:j + 3]}")

if top_block:
    for (va, ins, hit, lat, stall, idle), lab in zip(rows, labels):
        if lab == top_block:
            h = max(hit, 1)
            print(f"{va:6x} {hit:5d} {lat / h:8,.0f} {stall / h:8,.0f}  {ins[:90]}")
    sys.exit()

blocks = OrderedDict()
for (va, ins, hit, lat, stall, idle), lab in zip(rows, labels):
    b = blocks.setdefault(lab, dict(n=0, hit=0, lat=0, stall=0, idle=0, bar=0, wait=0, mfma=0))
    b["n"] += 1
    b["hit"] = max(b["hit"], hit)
    b["lat"] += lat
    b["stall"] += stall
    b["idle"] += idle
    op = ins.split()[0]
    if op == "s_barrier":
        b["bar"] += lat
    elif op == "s_waitcnt":
        b["wait"] += lat
    elif op.startswith("v_mfma"):
        b["mfma"] += hit

tot = sum(b["lat"] for b in blocks.values()) / waves
print(f"waves {waves}; cycles/wave {tot:,.0f} (sum of latency)")
print(f"{'block':10s} {'n':>5s} {'hit/w':>6s} {'cyc/w':>9s} {'%':>6s} {'stall/w':>8s} {'bar/w':>8s} {'wait/w':>8s} {'mfma/w':>7s}")
order = sorted(blocks.items(), key=lambda kv: -kv[1]["lat"])[:top]
for lab, b in sorted(order, key=lambda kv: list(blocks).index(kv[0])):
    print(
        f"{lab:10s} {b['n']:5d} {b['hit'] / waves:6.1f} {b['lat'] / waves:9,.0f} {100 * b['lat'] / waves / tot:6.2f}"
        f" {b['stall'] / waves:8,.0f} {b['bar'] / waves:8,.0f} {b['wait'] / waves:8,.0f} {b['mfma'] / waves:7,.0f}"
    )
