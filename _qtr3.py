import sqlite3, collections, itertools
c = sqlite3.connect("/tmp/tr/t_results.db")
S = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
KD = [t for t in S if "kernel_dispatch" in t][0]
KS = [t for t in S if "kernel_symbol" in t][0]
kc = [r[1] for r in c.execute(f"pragma table_info({KS})")]
nc = "display_name" if "display_name" in kc else "kernel_name"
rows = list(c.execute(f"select k.{nc}, d.start, d.end, d.grid_size_x, d.queue_id from {KD} d join {KS} k on k.id=d.kernel_id order by d.start"))
short = lambda n: n.split("_kernel")[0].replace("flash_attn_bwd_", "").replace("flash_attn_", "")[:9]
seq = [short(r[0]) for r in rows]
print("run-length of last 160:")
print(" ".join(f"{k}x{len(list(g))}" for k, g in itertools.groupby(seq[-160:])))
# true period: use 'lset' as the end marker
m = [i for i, s in enumerate(seq) if s == "lset"]
print("lset marks", len(m), "at", m[-4:])
a, b = m[-3] + 1, m[-2] + 1
t0 = rows[a][1]; span = rows[b-1][2] - t0
tot = collections.Counter(); cnt = collections.Counter(); prev = t0; gap = 0.0; gaps = []
for n, s, e, g, q in rows[a:b]:
    k = short(n); tot[k] += (e - s) / 1e3; cnt[k] += 1
    d = max(0, s - prev) / 1e3
    if d > 5: gaps.append(((prev - t0) / 1e3, d))
    gap += d; prev = max(prev, e)
print(f"one bwd: {b-a} disp, span {span/1e3:.1f} us, idle {gap:.1f} us")
for k, v in tot.most_common(): print(f"  {k:12s} n={cnt[k]:3d} {v:9.1f} us")
print("gaps >5us:", [(round(a2,1), round(d,1)) for a2, d in gaps])
