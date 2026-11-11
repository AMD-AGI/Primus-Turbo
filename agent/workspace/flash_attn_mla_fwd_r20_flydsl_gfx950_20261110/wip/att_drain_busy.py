"""Per-wave busy vs s_barrier time inside each persistent job's drain (ATT wave JSONs).

python wip/att_drain_busy.py WAVE_DIR ISA [JOB_TOP_LABEL] [DRAIN_LABEL]

A job's drain runs from its first DRAIN_LABEL entry to the next JOB_TOP_LABEL entry. An
instruction's time is the gap to the next one; s_barrier time is the wait, the rest is busy.
With perfectly balanced waves the drain would take about the mean busy time."""
import glob
import json
import os
import re
import sys

wave_dir, isa_path = sys.argv[1], sys.argv[2]
top_label = sys.argv[3] if len(sys.argv) > 3 else ".LBB0_3"
drain_label = sys.argv[4] if len(sys.argv) > 4 else ".LBB0_18"

code = json.load(open(os.path.join(wave_dir, "code.json")))["code"]
ops = [(i, c[0].split()[0], c[5]) for i, c in enumerate(code) if c[0] and not c[0].startswith(";")]
isa, label, on = [], "entry", False
for line in open(isa_path):
    s = line.split(";")[0].rstrip()
    if re.match(r"^flash\w*:", s):
        on = True
        continue
    if not on:
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
lab_of, op_of = {}, {}
j, prev = 0, "entry"
for idx, op, va in sorted(ops, key=lambda x: x[2]):
    while j < len(isa) and isa[j][1] != op and isa[j][1] == "s_nop":
        j += 1
    if j < len(isa) and isa[j][1] == op:
        prev = isa[j][0]
        j += 1
    lab_of[idx] = prev
    op_of[idx] = op

res = {}
for wf in sorted(glob.glob(os.path.join(wave_dir, "se*_wv*.json"))):
    insts = json.load(open(wf))["wave"]["instructions"]
    jobs, cur, state, last = [], None, None, None
    for k in range(len(insts) - 1):
        t, idx = insts[k][0], insts[k][-1]
        lab = lab_of.get(idx, last)
        if lab != last:
            if lab == top_label:
                if cur is not None:
                    jobs.append(cur)
                cur, state = [0.0, 0.0], "loop"
            elif lab == drain_label and state == "loop":
                state = "drain"
            last = lab
        if state == "drain":
            dt = insts[k + 1][0] - t
            cur[1 if op_of.get(idx) == "s_barrier" else 0] += dt
    res[os.path.basename(wf).split("_")[1]] = jobs

names = list(res)
nj = min(len(v) for v in res.values())
print("per job: busy / barrier per wave, drain, mean busy")
tot_drain = tot_mean = 0.0
for i in range(nj):
    cells = [res[n][i] for n in names]
    drain = max(b + w for b, w in cells)
    mean_busy = sum(b for b, _ in cells) / len(cells)
    tot_drain += drain
    tot_mean += mean_busy
    print(f"{i:3d} " + " | ".join(f"{b:6.0f}/{w:6.0f}" for b, w in cells) + f" | {drain:6.0f} {mean_busy:6.0f}")
print(f"avg drain {tot_drain / nj:.0f}, avg mean busy {tot_mean / nj:.0f}, max busy avg "
      f"{sum(max(res[n][i][0] for n in names) for i in range(nj)) / nj:.0f}")
