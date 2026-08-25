#!/usr/bin/env python3
"""Sum rocprofv3 PMC counters per kernel out of the rocpd SQLite result database.

rocprofv3's `--output-format csv -d <dir>` aborts in finalisation on this container (signal 6,
then it hangs until killed); the default rocpd/SQLite output works, and its
``counters_collection`` view already joins kernel names to counter values. Point this at the
newest `*_results.db` rocprofv3 leaves behind.

usage: _pmc_read.py <results.db> [kernel name substring]
"""
import sqlite3
import sys


def main():
    db = sqlite3.connect(sys.argv[1])
    like = f"%{sys.argv[2]}%" if len(sys.argv) > 2 else "%"
    rows = db.execute(
        "select kernel_name, counter_name, sum(value), count(distinct dispatch_id), "
        "sum(duration)/1000.0, max(vgpr_count), max(accum_vgpr_count), max(lds_block_size) "
        "from counters_collection where kernel_name like ? group by 1,2 order by 1,2",
        (like,),
    )
    for name, ctr, val, n, us, v, a, lds in rows:
        print(
            f"{name.split('(')[0][:44]:46s} {ctr:22s} {val:>16.0f}  n={n:<5d} "
            f"us={us/max(n,1):9.1f} vgpr={v} agpr={a} lds={lds}"
        )


if __name__ == "__main__":
    main()
