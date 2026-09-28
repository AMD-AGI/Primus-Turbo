"""Per attention kernel: dispatch resources + median counter values from rocprofv3 --pmc CSVs."""
import csv, glob, os, re, statistics as st, sys
from collections import defaultdict
P = sys.argv[1]
K = [("asm_fwd", "fmha_bf16_pertokenBf16_hd128_128x256_mask"), ("asm_odo", "odo_bf16"), ("asm_bwd", "causal_br_a32_pssk"),
     ("asm_dqcvt", "dq_convert_bf16"), ("fly_fwd", "kn_fmha_fwd_prefill"), ("fly_delta", "k_delta_bshd"), ("fly_dkdv", "k_dkdv"), ("fly_dq", "k_dq_0")]
res = {}; val = defaultdict(lambda: defaultdict(list)); dur = defaultdict(list)
for d in sorted(glob.glob(f"{P}/runs/pmc/pmc_*")):
    tag = os.path.basename(d)[4:]; setn = tag.split("_")[1]
    for r in csv.DictReader(open(f"{d}/pmc_counter_collection.csv")):
        for short, pat in K:
            if pat in r["Kernel_Name"]:
                res[short] = {x: r[x] for x in ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count")}
                val[(short, setn)][r["Counter_Name"]].append(float(r["Counter_Value"]))
                dur[(short, setn, tag)].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6)
print("== dispatch resources (rocprofv3)")
for s, _ in K:
    print(f"{s:10s}", res.get(s))
names = sorted({c for v in val.values() for c in v})
print("\n== median counter per dispatch  (set L16 = real training layer-16 inputs, randn = N(0,1))")
for s, _ in K:
    for setn in ("L16", "randn"):
        v = val.get((s, setn))
        if not v: continue
        print(f"{s:10s} {setn:5s} " + "  ".join(f"{c}={st.median(v[c]):.4g}" for c in names if c in v))
print("\n== derived")
for s, _ in K:
    v = val.get((s, "L16"))
    if not v: continue
    g = lambda c: st.median(v[c]) if c in v else float("nan")
    busy = g("SQ_BUSY_CYCLES"); gui = g("GRBM_GUI_ACTIVE")
    print(f"{s:10s} waves={g('SQ_WAVES'):.0f} wave_cycles/busy={g('SQ_WAVE_CYCLES')/max(busy,1):.2f} "
          f"wmma_inst={g('SQ_INSTS_VEC32_VALU_WMMA'):.4g} wmma_cycles={g('SQ_INST_CYCLES_VALU_WMMA'):.4g} "
          f"bf16_flop={g('SQ_VALU_WMMA_FLOP_BF16'):.4g} wmma_cyc/gui={g('SQ_INST_CYCLES_VALU_WMMA')/max(gui,1):.3g} "
          f"icache_req={g('SQC_ICACHE_REQ'):.4g} icache_miss%={100*g('SQC_ICACHE_MISSES')/max(g('SQC_ICACHE_REQ'),1):.2f} "
          f"chc_rd={g('CHC_REQ_READ'):.4g} chc_rd128={g('CHC_REQ_READ_128B'):.4g} cha_busy%={100*g('CHA_BUSY')/max(g('CHA_CYCLE'),1):.1f} "
          f"gl1a_busy%={100*g('GL1A_BUSY')/max(g('GL1A_CYCLE'),1):.1f} gl2_rd_lvl={g('GL1C_GL2_REQ_READ_LEVEL'):.4g} gl2_wr_lvl={g('GL1C_GL2_REQ_WRITE_LEVEL'):.4g} "
          f"spi_noalloc={g('SPI_RA_REQ_NO_ALLOC'):.4g} gui={gui:.4g}")
print("\n== dispatch ms under the profiler (median per run)")
for k in sorted(dur):
    print(k, round(st.median(dur[k]), 3))
