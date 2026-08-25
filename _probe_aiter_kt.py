#!/usr/bin/env python3
"""aiter's backward through its own autograd node, so the call path is the one it ships.
Host overhead does not matter here: the number that counts is read from rocprofv3
--kernel-trace, which is kernel-only.
usage: rocprofv3 --kernel-trace -d <dir> -- python _probe_aiter_kt.py <B> <Hq> <Hkv> <S> <D>"""
import sys

import torch

import aiter

B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
mk = lambda H: torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
torch.manual_seed(11)
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
out = aiter.flash_attn_func(q, k, v, causal=True)
go = torch.randn_like(out)
for _ in range(12):
    for t in (q, k, v):
        t.grad = None
    out.backward(go, retain_graph=True)
torch.cuda.synchronize()
print("done")
