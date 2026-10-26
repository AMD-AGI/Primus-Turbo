"""gaps_ss.py DIR: per rank, over the steady-state iterations (each starting at a prologue kernel): median iteration
span, prologue duration, prologue end -> dispatch_gemm start, and total GPU idle per iteration."""
import csv, glob, statistics, sys

for path in sorted(glob.glob(f"{sys.argv[1]}/**/*kernel_trace.csv", recursive=True)):
    rows = sorted(csv.DictReader(open(path)), key=lambda r: int(r["Start_Timestamp"]))
    ks = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in rows]
    starts = [i for i, k in enumerate(ks) if "prologue" in k[2]]
    starts = starts[-8:]  # the timed back-to-back iterations
    spans, pro, gap_dg, idle = [], [], [], []
    for a, b in zip(starts, starts[1:]):
        it = ks[a:b]
        spans.append((ks[b][0] - ks[a][0]) / 1e3)
        pro.append((it[0][1] - it[0][0]) / 1e3)
        dg = next(k for k in it if k[2].startswith("dispatch_gemm"))
        gap_dg.append((dg[0] - it[0][1]) / 1e3)
        busy_end, tot = it[0][0], 0
        for s, e, _ in it:
            if s > busy_end:
                tot += s - busy_end
            busy_end = max(busy_end, e)
        idle.append((tot + max(0, ks[b][0] - busy_end)) / 1e3)
    m = statistics.median
    print(f"{path.split('/')[-2]}: iter {m(spans):8.1f} us  prologue {m(pro):6.1f}  prologue end->dispatch_gemm {m(gap_dg):6.1f}"
          f"  GPU idle/iter {m(idle):6.1f} us  (n={len(spans)})")
