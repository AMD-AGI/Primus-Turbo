import sqlite3, collections, sys
db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tr/t_results.db"
c = sqlite3.connect(db)
S = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
KD = [t for t in S if "kernel_dispatch" in t][0]
KS = [t for t in S if "kernel_symbol" in t][0]
kcols = [r[1] for r in c.execute(f"pragma table_info({KS})")]
namecol = "display_name" if "display_name" in kcols else ("kernel_name" if "kernel_name" in kcols else kcols[2])
rows = list(c.execute(
    f"select k.{namecol}, d.start, d.end, d.grid_size_x, d.workgroup_size_x "
    f"from {KD} d join {KS} k on k.id=d.kernel_id order by d.start"))
print("dispatches", len(rows))
marks = [i for i, r in enumerate(rows) if "odo" in r[0]]
print("odo marks", len(marks))
a, b = marks[-3], marks[-2]
t0 = rows[a][1]; span = rows[b-1][2] - t0
tot = collections.Counter(); cnt = collections.Counter()
prev = t0; gap = 0
for n, s, e, g, w in rows[a:b]:
    key = n.split("_kernel")[0]
    tot[key] += (e - s) / 1e3; cnt[key] += 1
    gap += max(0, s - prev) / 1e3; prev = max(prev, e)
print(f"one bwd: {b-a} dispatches, span {span/1e3:.1f} us, idle-gap {gap:.1f} us")
for k, v in tot.most_common():
    print(f"  {k:44s} n={cnt[k]:3d}  {v:9.1f} us  {100*v/(span/1e3):5.1f}%")
