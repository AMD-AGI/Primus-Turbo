#!/usr/bin/env python3
"""Profiling driver: calls ONE impl on ONE shape with SYNTHETIC o/lse.

No torch.matmul anywhere -- the reference forward's rocBLAS/hipBLASLt path faults under
rocprofv3 on this image, and the kernel durations do not care whether o/lse are the true
forward outputs. Numerically meaningless; timing/counters only.
"""
import sys, torch
sys.path.insert(0, "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/ut")
from common import SHAPES, load_impl

path, shape, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
b, sq, skv, hq, hkv, d = SHAPES[shape]
fn = load_impl(path)
g = torch.Generator(device="cuda").manual_seed(0)
r = lambda *s: torch.randn(*s, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
q, k, v, do = r(b, sq, hq, d), r(b, skv, hkv, d), r(b, skv, hkv, d), r(b, sq, hq, d)
o = r(b, sq, hq, d)
lse = torch.randn(b, hq, sq, generator=g, device="cuda", dtype=torch.float32).abs() + 8.0
for _ in range(3):
    fn(do, q, k, v, o, lse, causal=True)
torch.cuda.synchronize()
for _ in range(iters):
    fn(do, q, k, v, o, lse, causal=True)
torch.cuda.synchronize()
print("profdrv done", shape, iters)
