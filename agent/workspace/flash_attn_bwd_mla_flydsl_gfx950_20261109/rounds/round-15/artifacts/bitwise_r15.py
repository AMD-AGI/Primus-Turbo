"""dK/dV of qdesc_csub=6 bitwise against the modulo walk (0), dQ by SNR: python bitwise_r15.py"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("MLA_PKG_ROOT", "/perf_apps/xiaoming/megaattention_overlap_bench"))
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla.ops_bwd import mla_bwd_sbhd  # noqa: E402


def snr(a, b):
    a, b = a.float(), b.float()
    return (10 * torch.log10(b.pow(2).sum() / (a - b).pow(2).sum().clamp_min(1e-30))).item()


for b, s in ((1, 1024), (1, 4096), (2, 4096), (1, 8192), (1, 4224), (1, 2112)):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k = (torch.randn((s, b, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
    v = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    do = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    o, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
    r = mla_bwd_sbhd(do, q, k, v, o, lse, qdesc_csub=0)
    n = mla_bwd_sbhd(do, q, k, v, o, lse, qdesc_csub=6)
    torch.cuda.synchronize()
    print(
        "B%d S%d dK bitwise %s dV bitwise %s dQ snr vs modulo %.2f dB"
        % (b, s, torch.equal(r[1], n[1]), torch.equal(r[2], n[2]), snr(n[0], r[0])),
        flush=True,
    )
