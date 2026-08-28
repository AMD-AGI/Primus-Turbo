#!/usr/bin/env python3
"""Isolated timing of the a16 dQ un-permute pass, min-of-N, with its TB/s.

The scored bench cannot resolve this leg (its same-binary spread is ~1.4%); this ruler
resolves 3%. Honours PT_A16_LCM / PT_A16_SCM, so the (load, store) cache policy pair can
be swept without editing the tree.

usage: _unp_cm.py <B> <Sq> <Hq> <D> [reps]
"""
import os
import sys
import time

import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as M

B, SQ, HQ, D = (int(x) for x in sys.argv[1:5])
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 60

img = torch.randn(B * SQ * HQ * D, dtype=torch.bfloat16, device="cuda")
dq = torch.empty(SQ, B, HQ, D, dtype=torch.bfloat16, device="cuda")
st = torch.cuda.current_stream()
SC = 1.0 / M._LOG2E
run = lambda: M._unpermute_dq_a16(img, dq, B, SQ, HQ, D, SC, st)

for _ in range(5):
    run()
torch.cuda.synchronize()
best = 1e9
for _ in range(REPS):
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    best = min(best, (time.perf_counter() - t0) * 1e3)
gb = 2 * img.numel() * 2 / 1e9
print(
    "lcm=%d scm=%d  ms %.4f  GB %.3f  TB/s %.2f"
    % (M._A16_LCM, M._A16_SCM, best, gb, gb / (best * 1e-3) / 1e3)
)
