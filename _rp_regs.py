"""Read kernel register/LDS metadata out of a rocprofv3 rocpd .db (schema per KB
connection/common/05: rocpd_info_kernel_symbol_<guid> holds arch_vgpr/accum_vgpr/
group_segment_size/private_segment_size)."""
import glob
import sqlite3
import sys

for db in sys.argv[1:]:
    for f in glob.glob(db):
        con = sqlite3.connect(f)
        tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        sym = [t for t in tabs if "info_kernel_symbol" in t]
        if not sym:
            print(f"== {f}: no kernel_symbol table; tables={tabs[:12]}")
            continue
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({sym[0]})")]
        want = [
            c
            for c in (
                "kernel_name",
                "arch_vgpr_count",
                "accum_vgpr_count",
                "sgpr_count",
                "group_segment_size",
                "private_segment_size",
            )
            if c in cols
        ]
        print(f"== {f}")
        q = f"SELECT {','.join(want)} FROM {sym[0]}"
        for row in con.execute(q):
            nm = str(row[0])
            if "at::native" in nm or "rocclr" in nm or "reduce_kernel" in nm:
                continue
            print("   " + "  ".join(f"{k}={v}" for k, v in zip(want, (nm.split("(")[0][:44],) + row[1:])))
