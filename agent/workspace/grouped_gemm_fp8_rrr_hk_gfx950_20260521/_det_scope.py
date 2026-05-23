"""Scope of nondeterminism: is it dense too? Is it grouped G=1?
Narrows bug to (A) HK kernel-wide, (B) grouped-only, or (C) grouped+multi-G only.

A FAIL case shape (qwen-down style): M_g=8192 K=1536 N=4096.
Test 1: dense hk_gemm_fp8 at [M=131072, K, N]  ← full M flattened
Test 2: hk_grp at B=1 [M_g=131072, ...]
Test 3: hk_grp at B=16 [M_g=8192 per group]   ← known nondet
"""
import os, sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
import primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_gemm = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8
hk_grp  = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

def _quant(t):
    return quantize_fp8(t, float8_e4m3, ScalingGranularity.TENSORWISE)

def det_run(label, fn, n=5):
    outs = [fn().detach().clone() for _ in range(n)]
    torch.cuda.synchronize()
    base = outs[0]
    fails = []
    for i in range(1, n):
        diff = outs[i] - base
        nz = (diff != 0).sum().item()
        mx = diff.abs().max().item()
        eq = torch.equal(base, outs[i])
        fails.append((eq, 100.0*nz/base.numel(), mx))
    print(f"{label:<60}")
    for i, (eq, p, mx) in enumerate(fails):
        print(f"    vs run-{i+2}: equal={eq}  diffs={p:.4f}%  max_abs={mx:.6g}")

# Shape from qwen-down-B16-M8192 (worst nondet)
N, K = 4096, 1536
M_g = 8192; B = 16
M_total = B * M_g  # 131072

torch.manual_seed(42)

# Test 1: dense hk_gemm at full M_total [131072, K] x [K, N]
a_d = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_d = (torch.randn((K, N),       dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af_d, asc_d = _quant(a_d); bf_d, bsc_d = _quant(b_d)
det_run(
    f"DENSE hk_gemm rrr [M={M_total}, K={K}] x [K, N={N}]",
    lambda: hk_gemm(af_d, bf_d, asc_d, bsc_d, "rrr", 4, torch.bfloat16))

# Test 2: hk_grp at B=1 [M_g=M_total, K] x [1, K, N]
a_g1 = a_d.contiguous()  # reuse same data
b_g1 = b_d.unsqueeze(0).contiguous()  # [1, K, N]
af_g1, asc_g1 = _quant(a_g1); bf_g1, bsc_g1 = _quant(b_g1)
g_offs_1 = grouped_gemm_compute_offs(torch.tensor([M_total], dtype=torch.int64, device=DEV))
det_run(
    f"GROUPED G=1 bn=128 [M_g={M_total}, K={K}, N={N}]",
    lambda: hk_grp(af_g1, bf_g1, asc_g1, bsc_g1, g_offs_1, 4, M_total, 4, torch.bfloat16, 128))

# Test 3: hk_grp at B=16 (known nondet baseline)
a_g16 = (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_g16 = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af_g16, asc_g16 = _quant(a_g16); bf_g16, bsc_g16 = _quant(b_g16)
g_offs_16 = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
det_run(
    f"GROUPED G=16 bn=128 [B={B}, M_g={M_g}, K={K}, N={N}]",
    lambda: hk_grp(af_g16, bf_g16, asc_g16, bsc_g16, g_offs_16, 4, M_g, 4, torch.bfloat16, 128))

# Test 4: hk_grp at G=2 (intermediate)
g_offs_2 = grouped_gemm_compute_offs(torch.tensor([M_total//2, M_total//2], dtype=torch.int64, device=DEV))
a_g2 = a_d.contiguous()
b_g2_tensor = b_d.unsqueeze(0).expand(2, K, N).contiguous()
af_g2, asc_g2 = _quant(a_g2); bf_g2, bsc_g2 = _quant(b_g2_tensor)
det_run(
    f"GROUPED G=2 bn=128 [B=2, M_g={M_total//2}, K={K}, N={N}]",
    lambda: hk_grp(af_g2, bf_g2, asc_g2, bsc_g2, g_offs_2, 4, M_total//2, 4, torch.bfloat16, 128))
