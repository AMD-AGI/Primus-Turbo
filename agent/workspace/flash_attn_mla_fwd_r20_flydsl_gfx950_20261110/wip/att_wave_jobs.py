"""Per-wave, per-job phase times from rocprofv3 ATT wave JSONs (persistent kernel).

python wip/att_wave_jobs.py WAVE_DIR ISA [JOB_TOP_LABEL] [DRAIN_LABEL]

Block labels come from the compiler ISA, aligned with code.json by opcode (unmatched s_nop
skipped). A job starts at each entry into JOB_TOP_LABEL; its drain starts at the first
DRAIN_LABEL entry after that. Prints per-job [top->drain, drain->next top] cycles per wave and
the drain's per-block time for the slowest and fastest wave."""
import glob
import json
import os
import re
import sys
from collections import defaultdict

wave_dir, isa_path = sys.argv[1], sys.argv[2]
top_label = sys.argv[3] if len(sys.argv) > 3 else ".LBB0_3"
drain_label = sys.argv[4] if len(sys.argv) > 4 else ".LBB0_17"

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

lab_of = {}
j = 0
prev = "entry"
for idx, op, va in sorted(ops, key=lambda x: x[2]):
    while j < len(isa) and isa[j][1] != op and isa[j][1] == "s_nop":
        j += 1
    if j < len(isa) and isa[j][1] == op:
        prev = isa[j][0]
        j += 1
    lab_of[idx] = prev

waves = sorted(glob.glob(os.path.join(wave_dir, "se*_wv*.json")))
per_wave = {}
for wf in waves:
    w = json.load(open(wf))["wave"]
    insts = w["instructions"]
    jobs = []
    cur = None
    last_lab = None
    blk_t = defaultdict(float)
    blk_jobs = []
    for k, ins in enumerate(insts):
        t, idx = ins[0], ins[-1]
        lab = lab_of.get(idx, last_lab)
        if lab != last_lab:
            if lab == top_label:
                if cur is not None:
                    cur["end"] = t
                    jobs.append(cur)
                    blk_jobs.append(blk_t)
                cur = dict(start=t, drain=None, end=None)
                blk_t = defaultdict(float)
            elif lab == drain_label and cur is not None and cur["drain"] is None:
                cur["drain"] = t
            last_lab = lab
        if k + 1 < len(insts):
            blk_t[lab] += insts[k + 1][0] - t
    if cur is not None:
        cur["end"] = insts[-1][0]
        jobs.append(cur)
        blk_jobs.append(blk_t)
    per_wave[os.path.basename(wf)] = (jobs, blk_jobs)

names = list(per_wave)
njobs = min(len(per_wave[n][0]) for n in names)
print(f"waves {len(names)}; jobs {njobs}")
print("job | " + " | ".join(f"{n.split('_')[1]} loop/drain" for n in names))
tot = defaultdict(lambda: [0, 0])
for i in range(njobs):
    cells = []
    for n in names:
        jb = per_wave[n][0][i]
        loop = (jb["drain"] or jb["end"]) - jb["start"]
        drain = jb["end"] - (jb["drain"] or jb["end"])
        tot[n][0] += loop
        tot[n][1] += drain
        cells.append(f"{loop:7d}/{drain:6d}")
    print(f"{i:3d} | " + " | ".join(cells))
print("sum | " + " | ".join(f"{tot[n][0]:7d}/{tot[n][1]:6d}" for n in names))

blocks = defaultdict(lambda: defaultdict(float))
for n in names:
    for bj in per_wave[n][1][:njobs]:
        for lab, v in bj.items():
            blocks[lab][n] += v / njobs
print("\nper-job block time (cycles), blocks >= 300 on some wave:")
print(f"{'block':10s} " + " ".join(f"{n.split('_')[1]:>8s}" for n in names))
order = sorted(blocks, key=lambda b: int(b.split("_")[1]) if b.startswith(".LBB") else -1)
for b in order:
    vals = [blocks[b][n] for n in names]
    if max(vals) >= 300:
        print(f"{b:10s} " + " ".join(f"{v:8.0f}" for v in vals))
