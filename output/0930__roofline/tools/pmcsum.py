# Per-kernel summary of a rocprofv3 --pmc csv: cycles = GRBM_GUI_ACTIVE / 8 (calibrated on mb1: 8 XCDs).
import csv, sys, collections, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(dict); meta = {}
for r in rows:
    d = r['Dispatch_Id']; by[d][r['Counter_Name']] = float(r['Counter_Value'])
    meta[d] = (r['Kernel_Name'], int(r['Start_Timestamp']), int(r['End_Timestamp']), int(r['Grid_Size']), int(r['Workgroup_Size']), r['VGPR_Count'], r['LDS_Block_Size'])
agg = collections.defaultdict(list)
for d, c in by.items():
    k, s, e, g, wg, vg, lds = meta[d]
    if 'GRBM_GUI_ACTIVE' not in c: continue
    agg[(k[:60], g, wg, vg, lds)].append((c['GRBM_GUI_ACTIVE'] / 8, (e - s) / 1e6, c.get('SQ_WAVES', 0)))
for key, v in sorted(agg.items(), key=lambda kv: -st.median(x[0] for x in kv[1]) * len(kv[1])):
    if len(v) < 3 or st.median(x[1] for x in v) < 0.02: continue
    cy = st.median(x[0] for x in v); du = st.median(x[1] for x in v)
    print(f"{key[0]:60s} n={len(v):3d} grid={key[1]} wg={key[2]} vgpr={key[3]} lds={key[4]} waves={st.median(x[2] for x in v):.0f} "
          f"cyc={cy:.4g} (min {min(x[0] for x in v):.4g}) ms={du:.4f} f_eff={cy/du/1e3:.0f}MHz")
