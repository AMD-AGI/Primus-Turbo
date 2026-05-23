"""Geometric diff pattern for RCR bn128 (analogous to RRR _diff_pattern.py).
Identifies which (group, M-tile, N-tile, warp_lane) the corruption clusters in.
Use K=384 N=256 B=16 M_g=8192 — the racing shape."""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_fp8

torch.manual_seed(42)
B, M_g, N, K = 16, 8192, 256, 384
M_total = B * M_g
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

outs = [hk(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone() for _ in range(5)]
torch.cuda.synchronize()
base = outs[0]
print(f"RCR bn128 shape: M_total={M_total} N={N} K={K} (BLK_M=256, BLK_N=128)")
print(f"Output shape: {base.shape}")

mask = torch.zeros_like(base, dtype=torch.bool)
for i in range(1, 5):
    mask |= (outs[i] != base)
total_diff = mask.sum().item()
print(f"Cells diff in >=1 of 4 reruns: {total_diff} ({100*total_diff/base.numel():.4f}%)")
print()

print("Per-group breakdown (M-row → group_idx = row // M_g):")
print(f"{'group':>5} {'diff_rows':>10} {'diff_cells':>11} {'%_of_group':>11}")
for g in range(B):
    g_mask = mask[g*M_g:(g+1)*M_g]
    print(f"{g:>5} {g_mask.any(dim=1).sum().item():>10} {g_mask.sum().item():>11} {100*g_mask.sum().item()/(M_g*N):>10.4f}%")

print()
print("Group 0 per M-tile (256 rows each):")
for t in range(M_g // 256):
    t_mask = mask[t*256:(t+1)*256]
    n = t_mask.sum().item()
    if n > 0:
        print(f"  M-tile {t:2} (rows {t*256:5}-{(t+1)*256-1:5}): {n}")

print()
print("Group 0 per N-tile (128 cols each):")
for c in range(N // 128):
    c_mask = mask[:M_g, c*128:(c+1)*128]
    n = c_mask.sum().item()
    print(f"  N-tile {c} (cols {c*128:4}-{(c+1)*128-1:4}): {n}")
