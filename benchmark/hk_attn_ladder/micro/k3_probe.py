"""Per-phase cycles of the 2-wave dK/dV full loop: a rung built with -DHKL_PROBE=1 (patches/a5_probe.patch) sums
s_get_shader_cycles deltas per phase over the full iterations of workgroup (0, 0, 0) (SALU only inside the loop, the
production asm patch regenerated for its .s), and writes them after the loop to the 64 KB guard in front of the dS
layout: wave w, u32 [w * 16, +6) phase sums, [w * 16 + 6] the full-iteration count.

    python3 -m micro.k3_probe mla/ladder_bwd.json r8a_pr [--calls 30]

Phases of one iteration: readback (hh0 of the next stage, after the ring barrier) + loop back; S/dP + softmax
(sdp3); dO^T tr16 burst issue; dS^T export (DPP regroup + stores); dV run with the Q^T tr16 interleave; dK kh0 +
tensorcnt + split barrier + dK kh1/kh2 window + wait.
"""
import argparse
import sys

import torch

from hkl import ladder as L
from hkl import ref as R
from hkl import runner

PHASES = ("readback + loop", "S/dP + softmax", "dO^T burst issue", "dS^T export", "dV + Q^T tr16",
          "dK + ring barrier")
# --fine (patches/a7_probe8.patch): S/dP + softmax split at the end of hh0's and of hh1's three (hh, kh) pairs
PHASES8 = ("readback + loop", "pack (P / dS)", "dO^T burst issue", "dS^T export", "dV + Q^T tr16",
           "dK + ring barrier", "top + hh0 S/dP + softmax", "hh1 S/dP + softmax")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder")
    ap.add_argument("rung")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--fine", action="store_true", help="8-phase probe (patches/a7_probe8.patch)")
    a = ap.parse_args()
    spec = runner.load_ladder(a.ladder)
    sh = L.parse_shape(None, spec["problem"])
    mod = runner.load(a.ladder, a.rung).module
    dev = torch.device("cuda")
    inp = R.make_inputs(sh, dev, seed=0)
    o, lse = R.ref_fwd(sh, inp["q"], inp["k"], inp["v"])
    o = o.to(torch.bfloat16)
    sbhd = mod.is_sbhd(inp["q"], lse)
    f = [mod._fold(t, sbhd) for t in (inp["do"], inp["q"], inp["k"], inp["v"], o)]
    lse_k = (lse.reshape(1, -1, lse.shape[-1]) if sbhd else lse).contiguous().float()
    stream = torch.cuda.current_stream()
    table = mod.kernels()
    plan = mod._plan(*f, lse_k, sh.d ** -0.5, True, stream)
    for _ in range(a.calls):
        for _, fn, args in plan["launches"]:
            if isinstance(fn, str):
                grid, block, values = args
                table[fn].call(grid, block, values, stream)
            else:
                mod._fly._launch(fn, args)
    torch.cuda.synchronize()
    g = plan["ds_full"][: mod.DS_GUARD_B // 2].view(torch.int32).cpu().tolist()
    names, nph = (PHASES8, 8) if a.fine else (PHASES, 6)
    for w in range(2):
        sums = [x & 0xFFFFFFFF for x in g[w * 16: w * 16 + nph]]
        n = g[w * 16 + nph]
        tot = sum(sums) / max(n, 1)
        print(f"wave {w}: {n} full iterations, {tot:.0f} cycles per iteration")
        order = (6, 7, 1, 2, 3, 4, 5, 0) if a.fine else range(nph)
        for i in order:
            name, s = names[i], sums[i]
            print(f"   {name:20s} {s / max(n, 1):7.1f}  ({100 * s / max(sum(sums), 1):4.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
