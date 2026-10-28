"""Per-step shader-cycle stamps of a wave-specialized dK/dV rung built with -DHKL_WS_STAMP=1 (workgroup (0, 0, 0),
128 steps; producers: points 0 start, 1 readback issued, 2 B + A done, 3 arrived, 4 waited; consumers: 0 start,
1 reads issued, 2 dV done, 3 tensor wait done, 4 arrived, 5 dK + export done, 6 waited).

    python3 -m micro.ws_stamps mla/ladder_bwd.json <rung> [--calls 30]

Runs the rung's whole backward at the ladder's problem `--calls` times back to back (clock settles), then reads the
stamps of the last call from the 64 KB guard in front of the dS layout. Prints per-wave medians of the intervals.
"""
import argparse
import statistics
import sys

import torch

from hkl import ladder as L
from hkl import ref as R
from hkl import runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder")
    ap.add_argument("rung")
    ap.add_argument("--calls", type=int, default=30)
    a = ap.parse_args()
    spec = runner.load_ladder(a.ladder)
    sh = L.parse_shape(None, spec["problem"])
    arm = runner.load(a.ladder, a.rung)
    mod = arm.module
    dev = torch.device("cuda")
    inp = R.make_inputs(sh, dev, seed=0)
    o, lse = R.ref_fwd(sh, inp["q"], inp["k"], inp["v"])
    o = o.to(torch.bfloat16)
    sbhd = mod.is_sbhd(inp["q"], lse)
    f = [mod._fold(t, sbhd) for t in (inp["do"], inp["q"], inp["k"], inp["v"], o)]
    lse_k = (lse.reshape(1, -1, lse.shape[-1]) if sbhd else lse).contiguous().float()
    stream = torch.cuda.current_stream()
    table = mod.kernels()
    scale = sh.d ** -0.5
    plan = mod._plan(*f, lse_k, scale, True, stream)
    for _ in range(a.calls):
        for _, fn, args in plan["launches"]:
            if isinstance(fn, str):
                grid, block, values = args
                table[fn].call(grid, block, values, stream)
            else:
                mod._fly._launch(fn, args)
    torch.cuda.synchronize()
    raw = plan["ds_full"][: mod.DS_GUARD_B // 2].view(torch.int64).cpu().tolist()
    names = {"producer": ["readback issued", "B + A done", "arrived", "waited"],
             "consumer": ["reads issued", "dV done", "tensor wait", "arrived", "dK + export", "waited"]}
    for w in range(4):
        role = "producer" if w < 2 else "consumer"
        st = [raw[w * 2048 + (s + 1) * 16: w * 2048 + (s + 2) * 16] for s in range(127)]
        st = [x for x in st if x[0] > 0]
        n = len(names[role])
        rows = []
        for x in st[3:-3]:  # steady steps
            pts = x[: n + 1]
            if all(v > 0 for v in pts):
                rows.append([pts[i + 1] - pts[i] for i in range(n)])
        steps = [st[i + 1][0] - st[i][0] for i in range(3, len(st) - 4)]
        med = [statistics.median(r[i] for r in rows) for i in range(n)] if rows else []
        print(f"wave {w} {role}: steps {len(st)}  step median {statistics.median(steps) if steps else 0:.0f} cyc;  "
              + "  ".join(f"{nm} +{m:.0f}" for nm, m in zip(names[role], med)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
