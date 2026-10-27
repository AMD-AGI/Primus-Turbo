"""gc_trace_wg.py TRACE.pt: per combine workgroup view of a gc_trace.py dump — the wait from entry to the first
chunk's ready (the block-flag polls), chunks per workgroup, push-busy time, and the combine span after the GEMM."""
import sys
import numpy as np
import torch

d = torch.load(sys.argv[1], weights_only=False)
tpu = d["ticks_per_us"]
for m, tr in zip(d["metas"], d["traces"]):
    wg = tr[: m["chunk_base"]].reshape(-1, 4)[: m["grid"]]
    role, ent, t1 = wg[:, 3], wg[:, 0], wg[:, 1]
    t0 = ent[role > 0].min()
    gemm_end = t1[role == 1].max()
    ch = tr[m["chunk_base"] :].reshape(-1, 4)
    cidx = np.nonzero(ch[:, 1] > 0)[0]
    ch = ch[cidx]
    print(f"{'L1 dgrad' if m['grad_gate'] else 'L2 fwd'}: {len(ch)} chunks, GEMM end {(gemm_end - t0) / tpu:.0f} us")
    for r, name in ((2, "lead"), (3, "tail")):
        firsts, nch, busy, span = [], [], [], []
        for b in np.nonzero(role == r)[0]:
            mine = ch[ch[:, 0] == b]
            nch.append(len(mine))
            if not len(mine):
                continue
            mine = mine[np.argsort(mine[:, 1])]
            firsts.append((mine[0, 1] - ent[b]) / tpu)
            busy.append((mine[:, 2] - mine[:, 1]).sum() / tpu)
            span.append((mine[-1, 2] - max(ent[b], gemm_end if r == 3 else ent[b])) / tpu)
        firsts, nch = np.array(firsts), np.array(nch)
        print(
            f"  {name}: first-chunk ready after entry median {np.median(firsts):.0f} p90 {np.quantile(firsts, .9):.0f} "
            f"max {firsts.max():.0f} us; chunks/WG median {np.median(nch):.0f} [{nch.min()}, {nch.max()}]; "
            f"push-busy median {np.median(busy):.0f} us of combine span median {np.median(span):.0f} us"
        )
    # chunks in take order: when were they ready relative to the GEMM end
    ready = (ch[:, 1] - gemm_end) / tpu
    print(f"  chunk ready vs GEMM end: {np.mean(ready < 0) * 100:.0f}% before; after-GEMM ready quantiles 50/90/100%: "
          f"{np.quantile(ready[ready >= 0], .5):.0f} / {np.quantile(ready[ready >= 0], .9):.0f} / {ready.max():.0f} us")
