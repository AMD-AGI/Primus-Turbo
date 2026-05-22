import csv, glob, sys
mode = sys.argv[1]
print(f"=== {mode.upper()} ===")
seen = set()
for f in sorted(glob.glob(f"/tmp/rp_{mode}/pmc_*/chi2811/*counter*.csv")):
    for r in csv.DictReader(open(f)):
        k = r["Kernel_Name"]
        if mode == "grp":
            ok = ("grouped_gemm_fp8_kernel" in k and "Layout)1, 256, false, false" in k)
        else:
            ok = ("hk_fp8_kernel11gemm_kernelILNS0_6LayoutE1E" in k or
                  "gemm_kernel<((anonymous namespace)::hk_fp8_kernel::Layout)1>" in k)
        if not ok: continue
        cn = r["Counter_Name"]
        if cn in seen: continue
        seen.add(cn)
        v = r["Counter_Value"]
        ts = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        print(f'{cn:30s} = {v}   (Δt={ts} ns)')
