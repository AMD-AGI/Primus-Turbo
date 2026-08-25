#!/usr/bin/env python3
"""One shape, one backward call, nothing else -- the smallest thing rocprofv3 can wrap.

_probe_bwdcfg builds a launcher per arm, and under the profiler's dispatch interception each
of those JITs serially (a 25-arm probe did not finish in 28 minutes). Point the profiler at
this instead when collecting PMC counters for the deployed path.

usage: _probe_pmc.py <B> <Hq> <Hkv> <S> <D> [reps] [W]
"""
import sys

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)


def main():
    a = sys.argv[1:]
    B, Hq, Hkv, S, D = (int(x) for x in a[:5])
    reps = int(a[5]) if len(a) > 5 else 1
    W = int(a[6]) if len(a) > 6 else -1
    ws = (W, 0) if W >= 0 else (-1, -1)
    sh = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
    q, k, v = sh(Hq), sh(Hkv), sh(Hkv)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(
        q, k, v, causal=True, window_size=ws, return_lse=True
    )
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    for _ in range(reps):
        flash_attn_sbhd_flydsl_backward_impl(do, q, k, v, o, lse_h, causal=True, window_size=ws)
    torch.cuda.synchronize()
    print("done")


if __name__ == "__main__":
    main()
