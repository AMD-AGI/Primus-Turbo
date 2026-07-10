"""Aggregate a rocprofv3 counter_collection.csv per backward kernel (mean/dispatch)."""
import collections
import csv
import glob
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("/tmp/pmc/**/*counter_collection*.csv", recursive=True))[-1]
rows = list(csv.DictReader(open(path)))
if not rows:
    print("no rows in", path)
    sys.exit(0)
print("csv:", path)
print("cols:", list(rows[0].keys()))


def classify(kn):
    if "flash_attn_bwd_kernel" in kn:
        return "fused_dq"
    if "dkdv" in kn:
        return "dkdv"
    if "odo" in kn:
        return "odo"
    if "flash_attn_generic" in kn:
        return "fwd"
    return None


agg = collections.defaultdict(lambda: collections.defaultdict(list))
kn_col = "Kernel_Name" if "Kernel_Name" in rows[0] else ("Kernel_Name " if "Kernel_Name " in rows[0] else None)
for r in rows:
    kn = r.get("Kernel_Name", "") or r.get("Kernel_Name ", "")
    cn = r.get("Counter_Name", "")
    try:
        v = float(r.get("Counter_Value", "nan"))
    except (TypeError, ValueError):
        continue
    key = classify(kn)
    if key:
        agg[key][cn].append(v)

regs = {}
for r in rows:
    kn = r.get("Kernel_Name", "")
    key = classify(kn)
    if key and key not in regs:
        regs[key] = (r.get("VGPR_Count"), r.get("Accum_VGPR_Count"), r.get("SGPR_Count"), r.get("Scratch_Size"), r.get("LDS_Block_Size"))

for k in ("fused_dq", "dkdv", "odo", "fwd"):
    if k not in agg:
        continue
    print("====", k)
    if k in regs:
        v = regs[k]
        print(f"  regs: VGPR={v[0]} AGPR={v[1]} SGPR={v[2]} Scratch={v[3]} LDS={v[4]}")
    for cn, vs in sorted(agg[k].items()):
        print(f"  {cn}: mean={sum(vs)/len(vs):.4f} n={len(vs)}")
