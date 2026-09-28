import json, glob, os, sys
W = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler/runs"
def ms(r): return r.get("ms", r.get("latency_ms"))
out = []
for f in sorted(glob.glob(W + "/*.json")):
    tag = os.path.basename(f)[:-5]
    rows = json.load(open(f))
    if isinstance(rows, list) and rows and "order" in rows[0]:
        for r in rows: r["mode"] = "fix"
    modes = {}
    for r in rows:
        modes.setdefault(r.get("mode", "std"), []).append(r)
    for m, rs in modes.items():
        order = [r["arm"] for r in rs]
        tree = {}
        for r in rs: tree.setdefault(r["arm"].split("_")[0], []).append(ms(r))
        aa = []
        for t, v in tree.items():
            if len(v) >= 2: aa.append(f"{t} A/A {v[0]/v[1]:.4f}")
        mean = {t: sum(v)/len(v) for t, v in tree.items()}
        rat = []
        if "r6" in mean:
            for t in mean:
                if t != "r6": rat.append(f"r6/{t}={mean['r6']/mean[t]:.4f}" if t=="l12" else f"{t}/r6={mean[t]/mean['r6']:.4f}")
        cells = " ".join(f"{a}={ms(r):.5f}" for a, r in zip(order, rs))
        print(f"{tag:5s} {m:8s} {cells} | {' '.join(aa)} | {' '.join(rat)}")
