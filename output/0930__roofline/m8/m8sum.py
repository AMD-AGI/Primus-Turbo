# Segment the m8drive PMC csv by marker fills (grid = 16384*(arm+1)); per arm drop LEAD=3, report medians.
import csv, collections, glob, statistics as st, sys
names = ["base", "noexp", "nosm", "nomask", "nobar", "wl", "asm", "base_nc", "asm_nc"]
f = glob.glob(sys.argv[1] + "/**/*counter_collection.csv", recursive=True)[0]
d = collections.OrderedDict()
for r in sorted(csv.DictReader(open(f)), key=lambda r: int(r['Dispatch_Id'])):
    x = d.setdefault(int(r['Dispatch_Id']), {'k': r['Kernel_Name'], 'g': int(r['Grid_Size']), 's': int(r['Start_Timestamp']), 'e': int(r['End_Timestamp'])})
    x[r['Counter_Name']] = float(r['Counter_Value'])
cur = None; seg = collections.defaultdict(list)
for x in d.values():
    if 'vectorized_elementwise' in x['k'] and x['g'] % 16384 == 0 and 1 <= x['g'] // 16384 <= len(names) and x['g'] < 200000:
        cur = names[x['g'] // 16384 - 1]; continue
    if cur and 'fmha' in x['k']: seg[cur].append(x)
base = None
for n in names:
    v = seg[n][3:]
    cy = st.median(x['GRBM_GUI_ACTIVE'] / 8 for x in v); ms = st.median((x['e'] - x['s']) / 1e6 for x in v)
    if n == "base": base = cy
    print(f"{n:8s} n={len(v)} cyc/SIMD={cy:.4g} vs_base={cy/base:.3f} ms={ms:.4f} f_eff={cy/ms/1e3:.0f}MHz kernel={v[0]['k'][:45]}")
