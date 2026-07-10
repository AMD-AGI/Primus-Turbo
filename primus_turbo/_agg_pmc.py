import csv, collections, glob, statistics as st, sys

f = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dkv3/**/*counter*csv", recursive=True))[-1]
rows = list(csv.DictReader(open(f)))
print("file:", f)
print("cols:", list(rows[0].keys()))
agg = collections.defaultdict(list)
name = None
for r in rows:
    kn = r.get("Kernel_Name") or r.get("Name") or ""
    if "dkdv" not in kn:
        continue
    name = kn[:70]
    cn = r.get("Counter_Name") or r.get("counter_name")
    cv = r.get("Counter_Value") or r.get("counter_value")
    agg[cn].append(float(cv))
print("kernel:", name)
for c, v in agg.items():
    print(f"{c:26s} n={len(v):3d} mean={st.mean(v):.4e} min={min(v):.4e} max={max(v):.4e}")
# per-dispatch static resource info (VGPR / scratch / dur)
vg = [r for r in rows if "dkdv" in (r.get("Kernel_Name") or "")]
if vg:
    r0 = vg[0]
    print("VGPR_Count=", r0.get("VGPR_Count"), "Accum_VGPR=", r0.get("Accum_VGPR_Count"),
          "Scratch=", r0.get("Scratch_Size"), "SGPR=", r0.get("SGPR_Count"))
    durs = [int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in vg if r.get("End_Timestamp")]
    if durs:
        print(f"dur_ns n={len(durs)} mean={st.mean(durs)/1000:.1f}us min={min(durs)/1000:.1f} max={max(durs)/1000:.1f}")
