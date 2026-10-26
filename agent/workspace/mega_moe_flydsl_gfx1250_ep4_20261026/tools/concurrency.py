"""concurrency.py DUMP.pt: resident workgroups over time in one gemm_combine launch (timeline_combine_ep TL_DUMP).

A GEMM tile lives start -> GEMM done; a lead / tail combine workgroup start -> reduce done."""
import sys, torch

z = torch.load(sys.argv[1])
d, t0, live, lead, tail = z["d"], z["t0"], z["live"], z["is_lead"], z["is_tail"]
us = lambda v: (v - t0).double() / 100.0
role = live | lead | tail
start = us(d[:, 0])
end = torch.where(live, us(d[:, 1]), us(d[:, 3]))
bid = torch.arange(len(d))
s, e, b = start[role], end[role], bid[role]
T = float(e.max())
grid = torch.arange(0.0, T, 5.0, dtype=torch.float64)
conc = ((s[None, :] <= grid[:, None]) & (e[None, :] > grid[:, None])).sum(1)
print(f"{sys.argv[1]}: span {T:.0f} us, max resident {int(conc.max())}, median resident over span "
      f"{float(conc.double().median()):.0f}")
for t in range(0, int(T), 100):
    i = min(int(t / 5), len(grid) - 1)
    m = (s <= t) & (e > t)
    per_xcd = torch.bincount(b[m] % 8, minlength=8).tolist()
    print(f"  t={t:5d} us resident {int(conc[i]):4d}  per bid%8 {per_xcd}  (gemm {int((m & live[role]).sum())},"
          f" lead {int((m & lead[role]).sum())}, tail {int((m & tail[role]).sum())})")
