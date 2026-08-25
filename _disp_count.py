#!/usr/bin/env python3
"""One backward call, nothing else -- so a kernel-trace row count IS the dispatch count.

Every marker placed anywhere inside the dkdv body (emission loop, _gemm3_tiles entry, _gemm3
entry, head-step entry) reads 2x, yet dK/dV are correct and they are LOOP-CARRIED sums, which
a doubled body would have doubled. The remaining explanation is that the dispatch itself
happens twice -- invisible on every deployed output because they are all STORED, not
accumulated. This counts the dispatches.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_forward_impl as ff,
)

B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
g = torch.Generator().manual_seed(3)
mk = lambda h: torch.randn(S, B, h, D, generator=g, dtype=torch.bfloat16).cuda()
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
do = torch.randn(S, B, Hq, D, generator=g, dtype=torch.bfloat16).cuda()
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))  # warm the JIT + plan
torch.cuda.synchronize()
print("=== TRACE WINDOW OPENS ===", flush=True)
torch.cuda.profiler.start()
fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("=== TRACE WINDOW CLOSES ===", flush=True)
