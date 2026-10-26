"""tail_phases.py PREFIX WORLD: per-rank gemm_combine phase edges from the timeline_combine_ep TL_DUMP files.

Stamps: 0 start, 1 GEMM role done, 2 combine push done, 3 reduce done (100 MHz realtime, shared across ranks)."""
import sys, torch

prefix, world = sys.argv[1], int(sys.argv[2])
zs = [torch.load(f"{prefix}_r{r}.pt") for r in range(world)]
g0 = min(int(z["t0"]) for z in zs)
us = lambda v: (v - g0).double() / 100.0
for r, z in enumerate(zs):
    d, live, comb = z["d"], z["live"], z["is_lead"] | z["is_tail"]
    s, e1, e2, e3 = (us(d[:, i]) for i in range(4))
    tail = z["is_tail"]
    q = lambda x: f"{float(x.min()):7.0f}/{float(x.median()):7.0f}/{float(x.max()):7.0f}"
    print(f"rank {r}: start {float(s[live | comb].min()):6.0f}  GEMM done min/med/max {q(e1[live])}  "
          f"tail start {q(s[tail])}  push done {q(e2[comb])}  reduce done {q(e3[comb])}")
    push_ms = (e2[comb] - torch.maximum(s[comb], e1[comb] * 0)).double()
    red = e3[comb] - e2[comb]
    print(f"         per combine WG: push phase med {float((e2[comb]-s[comb]).median()):6.0f} us, "
          f"reduce phase min/med/max {float(red.min()):5.0f}/{float(red.median()):5.0f}/{float(red.max()):5.0f} us; "
          f"tail WGs starting after last GEMM done: {int((s[tail] > e1[live].max()).sum())}/{int(tail.sum())}")
