"""Parse rocprof CSV across 3 runs, extract bn128 RRR counters."""
import csv, glob, sys

for i in [1, 2, 3]:
    print(f"=== run {i} ===")
    files = glob.glob(f"/tmp/rp_run{i}/chi2811/*counter_collection.csv")
    if not files:
        print("  no file")
        continue
    with open(files[0]) as f:
        r = csv.DictReader(f)
        for row in r:
            kn = row["Kernel_Name"]
            if "grouped_gemm_fp8_kernel" in kn and "Layout)1, 128" in kn:
                print(f"  {row['Counter_Name']:25} = {row['Counter_Value']}")
