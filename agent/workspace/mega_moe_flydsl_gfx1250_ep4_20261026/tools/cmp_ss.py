"""cmp_ss.py DIR_A DIR_B: per kernel position in the steady-state iteration, median duration (us) A vs B, per rank
index (traces sorted by pid), plus the iteration span."""
import csv, glob, statistics, sys


def iters(path):
    rows = sorted(csv.DictReader(open(path)), key=lambda r: int(r["Start_Timestamp"]))
    ks = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in rows]
    starts = [i for i, k in enumerate(ks) if "prologue" in k[2]][-8:]
    return [ks[a:b] for a, b in zip(starts, starts[1:])], [ks[b][0] - ks[a][0] for a, b in zip(starts, starts[1:])]


A = sorted(glob.glob(f"{sys.argv[1]}/**/*kernel_trace.csv", recursive=True))
B = sorted(glob.glob(f"{sys.argv[2]}/**/*kernel_trace.csv", recursive=True))
for ra, (pa, pb) in enumerate(zip(A, B)):
    ia, sa = iters(pa)
    ib, sb = iters(pb)
    n = min(len(i) for i in ia + ib)
    print(f"rank-index {ra}: iter {statistics.median(sa)/1e3:.1f} -> {statistics.median(sb)/1e3:.1f} us")
    for j in range(n):
        da = statistics.median((it[j][1] - it[j][0]) / 1e3 for it in ia)
        db = statistics.median((it[j][1] - it[j][0]) / 1e3 for it in ib)
        if abs(db - da) > 8 or da > 300:
            print(f"   {j:3d} {ia[0][j][2][:48]:48s} {da:8.1f} -> {db:8.1f}  ({db - da:+7.1f})")
