#!/usr/bin/env python3
"""Summarise a rocprofv3 --kernel-trace sqlite db: per-kernel count / total / mean us."""
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ktr/k1_results.db"
c = sqlite3.connect(db)
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
disp = next((t for t in tabs if "kernel_dispatch" in t), None)
if disp is None:
    print([t for t in tabs])
    raise SystemExit(0)
cols = [r[1] for r in c.execute(f"pragma table_info({disp})")]
print(disp, cols)
sym = next((t for t in tabs if "kernel_symbol" in t), None)
scols = [r[1] for r in c.execute(f"pragma table_info({sym})")] if sym else []
print(sym, scols)
q = (
    f"select s.display_name, count(*), sum(d.end-d.start)/1000.0, avg(d.end-d.start)/1000.0 "
    f"from {disp} d join {sym} s on d.kernel_id=s.id group by 1 order by 3 desc"
)
try:
    rows = list(c.execute(q))
except Exception as e:
    print("q failed", e)
    rows = list(c.execute(f"select kernel_id, count(*), sum(end-start)/1000.0 from {disp} group by 1"))
for r in rows:
    print("  ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r))
