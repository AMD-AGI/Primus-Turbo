import sqlite3, collections, sys
db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tra/a_results.db"
c = sqlite3.connect(db)
S = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
KD = [t for t in S if "kernel_dispatch" in t][0]
KS = [t for t in S if "kernel_symbol" in t][0]
kc = [r[1] for r in c.execute(f"pragma table_info({KS})")]
nc = "display_name" if "display_name" in kc else "kernel_name"
rows = list(c.execute(f"select k.{nc}, d.start, d.end, d.grid_size_x, d.grid_size_y, d.grid_size_z,"
                     f" d.workgroup_size_x, d.group_segment_size from {KD} d join {KS} k on k.id=d.kernel_id order by d.start"))
print("dispatches", len(rows))
seen = collections.Counter()
for n, s, e, gx, gy, gz, w, lds in rows[-14:]:
    wgs = (gx // max(w, 1)) * max(gy, 1) * max(gz, 1)
    print(f"{(e-s)/1e3:8.1f}us  grid=({gx},{gy},{gz}) wg={w} wgs={wgs:7d} lds={lds:6d}  {n[:70]}")
