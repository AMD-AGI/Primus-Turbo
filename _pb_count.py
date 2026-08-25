#!/usr/bin/env python3
"""Count dQ fold launches and print their (bat_lo, n_bat, qsp, qblk) per backward.

Verifies that a dispatch-plan arm reaches the reduce at all, which is the check the
campaign's four inert-arm families all failed. usage: _pb_count.py <B> <Hq> <Hkv> <S> <D> <arm>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import _probe_bwdcfg as P
import primus_turbo.flydsl.attention.flash_attn_bwd as bwd
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)


def main():
    a = sys.argv[1:]
    B, Hq, Hkv, S, D = (int(x) for x in a[:5])
    P.patch_all(a[5])
    red = bwd._reduce_dq_partials
    calls = []

    def spy(*args, **kw):
        calls.append((kw.get("bat_lo", 0), kw.get("n_bat"), kw.get("qsp"), kw.get("qblk")))
        return red(*args, **kw)

    bwd._reduce_dq_partials = spy
    sh = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
    q, k, v = sh(Hq), sh(Hkv), sh(Hkv)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    f = lambda: flash_attn_sbhd_flydsl_backward_impl(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))
    ref = f()
    calls.clear()
    cur = f()
    print(f"arm={a[5]} fold_launches={len(calls)} bitwise={all(torch.equal(x, y) for x, y in zip(ref[:3], cur[:3]))}")
    for c in calls:
        print("  ", c)


if __name__ == "__main__":
    main()
