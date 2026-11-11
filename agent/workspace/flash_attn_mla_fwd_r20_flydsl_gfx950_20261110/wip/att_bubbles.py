"""Per-MFMA-slot bubble map of the hot loop from an ATT stats CSV.

python wip/att_bubbles.py CSV [min_hit_per_wave] [top]

A slot is one MFMA plus the non-MFMA instructions up to the next MFMA. Its content is the MFMA's issue
cycles plus every following instruction's latency (issue + stall). Content above the MFMA pipe time
(16 cycles for 16x16x32) is a pipe bubble; it is split into issue overflow and non-MFMA stall."""
import csv, sys

path = sys.argv[1]
minh = float(sys.argv[2]) if len(sys.argv) > 2 else 10
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
PIPE = 16
rows = []
for r in csv.DictReader(open(path)):
    ins = r["Instruction"].strip()
    if not ins or ins.startswith(";"):
        continue
    rows.append((int(r["Vaddr"]), ins, int(r["Hitcount"]), int(r["Latency"]), int(r["Stall"])))
rows.sort()
waves = next(h for _, _, h, _, _ in rows if h > 0)
hot = [r for r in rows if r[2] / waves >= minh]
per_iter = hot[0][2]
slots = []
cur = None
for a, ins, h, lat, st in hot:
    op = ins.split()[0]
    scale = h / per_iter
    if op.startswith("v_mfma"):
        cur = dict(addr=a, ins=ins, mstall=st / h, issue=(lat - st) / per_iter, ostall=0.0, oissue=0.0, ops=[], scale=scale)
        slots.append(cur)
    elif cur is not None:
        cur["oissue"] += (lat - st) / per_iter
        cur["ostall"] += st / per_iter
        cur["ops"].append((op, (lat - st) / h, st / h))
tot_bub = tot_iss = tot_st = 0.0
for s in slots:
    content = s["issue"] / s["scale"] + (s["oissue"] + s["ostall"]) / s["scale"]
    over = content - PIPE
    s["over"] = over
    if over > 0:
        iss_over = max(0.0, s["issue"] / s["scale"] + s["oissue"] / s["scale"] - PIPE)
        s["iss_over"] = iss_over
        s["st_over"] = over - iss_over
    else:
        s["iss_over"] = s["st_over"] = 0.0
    tot_bub += max(0.0, over) * s["scale"]
    tot_iss += s["iss_over"] * s["scale"]
    tot_st += s["st_over"] * s["scale"]
print(f"MFMA slots {len(slots)}  ({sum(s['scale'] for s in slots):.0f} per iter)  "
      f"bubble {tot_bub:.0f} cyc/iter = issue overflow {tot_iss:.0f} + stall {tot_st:.0f}")
under = sum(min(0.0, s["over"]) * s["scale"] for s in slots)
print(f"spare issue capacity in under-full slots {-under:.0f} cyc/iter")
print(f"{'addr':>7} {'scale':>5} {'over':>6} {'iss':>6} {'stall':>6}  ops")
for s in sorted(slots, key=lambda s: -s["over"] * s["scale"])[:top]:
    ops = " ".join(f"{o.replace('v_', '').replace('s_', '')}:{i:.0f}+{st:.0f}" for o, i, st in s["ops"])
    print(f"{s['addr']:7x} {s['scale']:5.1f} {s['over']:6.1f} {s['iss_over']:6.1f} {s['st_over']:6.1f}  {ops[:150]}")

if len(sys.argv) > 5:
    lo, hi = int(sys.argv[4], 16), int(sys.argv[5], 16)
    print(f"\nslots in [{lo:x}, {hi:x}) in address order:")
    for s in slots:
        if lo <= s["addr"] < hi:
            ops = " ".join(f"{o.replace('v_', '').replace('s_', '')}:{i:.0f}+{st:.0f}" for o, i, st in s["ops"])
            print(f"{s['addr']:7x} mst {s['mstall']:4.1f} over {s['over']:6.1f}  {ops[:130]}")
