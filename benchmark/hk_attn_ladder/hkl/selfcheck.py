###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only consistency check of every built rung (no GPU, no kernel launch):

    python3 -m hkl.selfcheck llama/ladder_fwd.json llama/ladder_bwd.json [--rung ID ...]

For each rung: load it through hkl.runner (its own lab module instance), let the lab module open the
code object exactly as it will before the first launch (sha256 vs meta.json, resources, and every
kernel argument's count / size / kind against the code object's own metadata: hiprt.kernel_meta), and
plan the production launch on meta tensors where the module plans on tensors. Run it where torch
imports (e.g. the device-less build container); torch never sees a GPU here.
"""

import argparse
import sys
import traceback

import torch

from . import runner

PROD = {"llama": dict(b=4, sq=8192, skv=8192, hq=32, hkv=8, d=128)}


def check(ladder, rid):
    r = runner.load(ladder, rid)
    mod = r.module
    loader = getattr(mod, "_kernel", None) or getattr(mod, "_kernels")
    k = loader()
    n = len(k) if isinstance(k, dict) else 1
    msg = f"{n} kernel(s) loaded"
    if hasattr(mod, "flash_attn_fwd") and "llama" in str(ladder):
        p = PROD["llama"]
        q = torch.empty((p["b"], p["sq"], p["hq"], p["d"]), dtype=torch.bfloat16, device="meta")
        kv = torch.empty((p["b"], p["skv"], p["hkv"], p["d"]), dtype=torch.bfloat16, device="meta")
        try:
            if mod.__name__.endswith(".fwd_pp"):
                o = torch.empty_like(q)
                lse = torch.empty((p["b"], p["hq"], p["sq"]), dtype=torch.float32, device="meta")
                grid, block, _ = mod.plan(q, kv, kv, o, lse, 0.088388, True)
            else:
                pl = mod.plan(q, kv, kv, None, True)
                grid, block = pl["grid"], pl["block"]
            msg += f", prod plan grid {tuple(grid)} block {tuple(block)}"
        except Exception as e:  # noqa: BLE001 - report every rung
            return False, f"plan failed: {type(e).__name__}: {e}"
    return True, msg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladders", nargs="+")
    ap.add_argument("--rung", nargs="*")
    a = ap.parse_args(argv)
    bad = 0
    for ladder in a.ladders:
        for rid in a.rung or runner.rung_ids(ladder):
            try:
                ok, msg = check(ladder, rid)
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
            bad += not ok
            print(f"  {ladder:<24} {rid:<16} {'ok  ' if ok else 'FAIL'} {msg}")
    print(f"{bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
