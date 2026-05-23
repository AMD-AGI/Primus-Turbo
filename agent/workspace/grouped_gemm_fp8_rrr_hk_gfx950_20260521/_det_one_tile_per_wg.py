"""Test: each wg processes ≤1 tile (no persistent loop iter > 1).
If deterministic → race comes from cross-tile state carryover in persistent loop.
If still nondet → race is per-tile (some specific K-iter / store / mfma issue).

Shapes:
- G=1, M_g=2048, N=128, K=1536: M-tiles=8, N-tiles=1, total=8 → 8 wgs do 1 tile each (NUM_CUS=256 idle)
- G=1, M_g=8192, N=128, K=1536: M-tiles=32, N-tiles=1, total=32 → 32 wgs do 1 tile each
- G=1, M_g=131072 K=1536 N=4096 (original baseline nondet shape, 16384 tiles, ~64 tiles/wg)
"""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

def _quant(t): return quantize_fp8(t, float8_e4m3, ScalingGranularity.TENSORWISE)

def det_run(label, fn, n=5):
    outs = [fn().detach().clone() for _ in range(n)]
    torch.cuda.synchronize()
    base = outs[0]
    print(f"{label}")
    for i in range(1, n):
        diff = outs[i] - base
        nz = (diff != 0).sum().item()
        mx = diff.abs().max().item()
        eq = torch.equal(base, outs[i])
        mark = "OK" if eq else "RACE"
        print(f"  run-{i+1}: {mark}  equal={eq} diffs={100.0*nz/base.numel():.4f}% max_abs={mx:.6g}")

torch.manual_seed(42)

# Case A: only 8 tiles total (M-tiles=8, N-tiles=1, G=1)
N, K = 128, 1536
B, M_g = 1, 2048
M_total = B * M_g  # 2048; M-tiles=8
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = _quant(a); bf, bsc = _quant(b)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
det_run(
    f"A. 8 tiles [G=1 M_g={M_g} N={N} K={K}] (8 wgs do 1 tile each, 248 wgs idle)",
    lambda: hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128))

# Case B: 32 tiles total
N, K = 128, 1536
B, M_g = 1, 8192
M_total = B * M_g  # 8192; M-tiles=32
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = _quant(a); bf, bsc = _quant(b)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
det_run(
    f"B. 32 tiles [G=1 M_g={M_g} N={N} K={K}] (32 wgs do 1 tile each)",
    lambda: hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128))

# Case C: ~256 tiles total (each wg ~1 tile)
N, K = 1024, 1536
B, M_g = 1, 16384
M_total = B * M_g  # 16384; M-tiles=64, N-tiles=8, total=512
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = _quant(a); bf, bsc = _quant(b)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
det_run(
    f"C. 512 tiles [G=1 M_g={M_g} N={N} K={K}] (256 wgs do ~2 tiles each)",
    lambda: hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128))

# Baseline nondet: many tiles per wg
N, K = 4096, 1536
B, M_g = 16, 8192
M_total = B * M_g
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = _quant(a); bf, bsc = _quant(b)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
det_run(
    f"BASELINE. 16384 tiles [B={B} M_g={M_g} N={N} K={K}] (64 tiles/wg)",
    lambda: hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128))
