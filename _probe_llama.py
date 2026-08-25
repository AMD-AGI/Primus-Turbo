#!/usr/bin/env python3
"""Per-cell probe for the llama-shape campaign. One shape per process, min-of-N.

usage: _probe_llama.py <B> <Hq> <Hkv> <S> <D> [W] [reps] [pass]
       pass in {fwd,bwd,both}; prints one JSON line.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

DEV, DT = "cuda", torch.bfloat16


def _sbhd(B, S, H, D):
    return torch.randn(S, B, H, D, device=DEV, dtype=DT)


def _time(fn, warms, reps):
    for _ in range(warms):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def main():
    a = sys.argv[1:]
    B, Hq, Hkv, S, D = (int(x) for x in a[:5])
    W = int(a[5]) if len(a) > 5 else -1
    reps = int(a[6]) if len(a) > 6 else 40
    which = a[7] if len(a) > 7 else "both"
    ws = (W, 0) if W >= 0 else (-1, -1)
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=ws, return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    out = {"B": B, "Hq": Hq, "Hkv": Hkv, "S": S, "D": D, "W": W}
    if which in ("fwd", "both"):
        out["fwd"] = round(
            _time(
                lambda: flash_attn_sbhd_flydsl_forward_impl(
                    q, k, v, causal=True, window_size=ws, return_lse=True
                ),
                5,
                reps,
            ),
            4,
        )
    if which in ("bwd", "both"):
        out["bwd"] = round(
            _time(
                lambda: flash_attn_sbhd_flydsl_backward_impl(
                    do, q, k, v, o, lse_h, causal=True, window_size=ws
                ),
                5,
                reps,
            ),
            4,
        )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
