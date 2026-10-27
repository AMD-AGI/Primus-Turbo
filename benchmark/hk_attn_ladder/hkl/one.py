###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run one arm (rung or baseline) a few times at one shape: the workload for rocprofv3 (ATT, kernel trace).

    python3 -m hkl.one llama/ladder_fwd.json f03 [-n 3] [--warm 2] [--shape b=1 sq=4096 skv=4096]

Warm calls first (JIT, caches, clocks), then -n calls with a device sync after each, so a profiler that
selects dispatches by index (--att-consecutive-kernels, kernel-include regex + skip) sees whole calls.
"""

import argparse
import sys

import torch

from . import ladder as L
from . import ref as R
from . import runner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladder")
    ap.add_argument("arm", help="rung id or baseline id")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--shape", nargs="*")
    a = ap.parse_args(argv)
    spec = runner.load_ladder(a.ladder)
    direction = spec["direction"]
    sh = L.parse_shape(a.shape, spec["problem"])
    arm = (runner.load_baseline(a.ladder, a.arm) if a.arm in runner.baseline_ids(a.ladder)
           else runner.load(a.ladder, a.arm))  # fmt: skip
    dev = torch.device("cuda")
    inp = R.make_inputs(sh, dev, seed=0)
    if direction == "bwd":
        o, lse = R.ref_fwd(sh, inp["q"], inp["k"], inp["v"])
        inp["o"], inp["lse"] = o.to(torch.bfloat16), lse
    call = L.make_call(arm, direction, inp, sh)
    for _ in range(a.warm):
        call()
    torch.cuda.synchronize()
    for i in range(a.n):
        call()
        torch.cuda.synchronize()
    print(f"{arm} {sh}: {a.warm} warm + {a.n} calls done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
