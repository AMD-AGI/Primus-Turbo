import sqlite3, collections
c = sqlite3.connect("/tmp/tr/t_results.db")
S = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
KD = [t for t in S if "kernel_dispatch" in t][0]
KS = [t for t in S if "kernel_symbol" in t][0]
kc = [r[1] for r in c.execute(f"pragma table_info({KS})")]
nc = "display_name" if "display_name" in kc else "kernel_name"
rows = list(c.execute(f"select k.{nc}, d.start, d.end, d.grid_size_x, d.workgroup_size_x, d.group_segment_size, d.queue_id from {KD} d join {KS} k on k.id=d.kernel_id order by d.start"))
marks = [i for i, r in enumerate(rows) if "odo" in r[0]]
a, b = marks[-3], marks[-2]
t0 = rows[a][1]
for n, s, e, g, w, lds, q in rows[a:b]:
    print(f"{(s-t0)/1e3:9.1f} +{(e-s)/1e3:8.1f}us  q{q}  grid={g:9d} wg={w:4d} wgs={g//max(w,1):7d} lds={lds:6d}  {n.split('_kernel')[0]}")
