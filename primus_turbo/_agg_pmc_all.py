import collections
import csv
import glob
import statistics as st
import sys

f = sorted(glob.glob((sys.argv[1] if len(sys.argv) > 1 else "/tmp/dqp") + "/**/*counter_collection.csv", recursive=True))[-1]
rows = list(csv.DictReader(open(f)))


def bucket(kn):
    if "dkdv" in kn:
        return "dkdv"
    if "bwd_dq_kernel" in kn:
        return "dq16"
    if "bwd_kernel" in kn:
        return "dq_fused"
    if "odo" in kn:
        return "odo"
    return None


agg = collections.defaultdict(lambda: collections.defaultdict(list))
res = {}
dur = collections.defaultdict(list)
for r in rows:
    kn = r.get("Kernel_Name") or r.get("Name") or ""
    b = bucket(kn)
    if not b:
        continue
    cn = r.get("Counter_Name")
    cv = r.get("Counter_Value")
    if cn and cv:
        agg[b][cn].append(float(cv))
    if b not in res:
        res[b] = (r.get("VGPR_Count"), r.get("Accum_VGPR_Count"), r.get("Scratch_Size"), r.get("SGPR_Count"))
    if r.get("End_Timestamp") and r.get("Start_Timestamp"):
        dur[b].append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))

for b in ("dq16", "dq_fused", "dkdv", "odo"):
    if b not in agg:
        continue
    print("=====", b, " VGPR/AccVGPR/Scratch/SGPR=", res.get(b))
    if dur[b]:
        print(f"  dur mean={st.mean(dur[b])/1000:.1f}us")
    for c, v in agg[b].items():
        print(f"  {c:32s} mean={st.mean(v):.3e}")
