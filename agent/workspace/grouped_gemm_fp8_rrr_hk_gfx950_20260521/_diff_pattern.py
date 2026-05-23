"""Where do the bit-different cells cluster? Geometric localization of race.
Output is [M_total, N]. Index by M-row → group/tile, by N → tile-col.
"""
import os, sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

torch.manual_seed(42)
# G=16 worst case (qwen-down shape)
B, M_g, N, K = 16, 8192, 4096, 1536
M_total = B * M_g  # 131072

a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

def run():
    return hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone()

outs = [run() for _ in range(5)]
torch.cuda.synchronize()
base = outs[0]
print(f"Output shape: {base.shape}  M_total={M_total} N={N} groups={B} M_g={M_g}")
print(f"BLK_M=256, BLK_N=128 → M-tiles per group = {M_g // 256}, N-tiles = {N // 128}")
print()

# Aggregate diff mask across runs
mask = torch.zeros_like(base, dtype=torch.bool)
for i in range(1, 5):
    mask |= (outs[i] != base)
total_diff = mask.sum().item()
print(f"Cells that differ in >=1 of 4 reruns vs run0: {total_diff} ({100*total_diff/base.numel():.4f}%)")

# Cluster by M-row → which group? which M-tile inside group?
diff_rows = mask.any(dim=1)
diff_row_idx = diff_rows.nonzero(as_tuple=True)[0]
print(f"Distinct rows with diffs: {diff_row_idx.numel()}")

# Per-group distribution
print("\nPer-group breakdown (M-row → group_idx = row // M_g):")
print(f"{'group':>5} {'diff_rows':>10} {'diff_cells':>11} {'%_of_group_cells':>16}")
for g in range(B):
    row_lo, row_hi = g * M_g, (g + 1) * M_g
    g_mask = mask[row_lo:row_hi]
    nrows = g_mask.any(dim=1).sum().item()
    ncells = g_mask.sum().item()
    pct = 100.0 * ncells / (M_g * N)
    print(f"{g:>5} {nrows:>10} {ncells:>11} {pct:>15.4f}%")

# M-tile breakdown for group 0
print("\nGroup 0: per M-tile (256 rows each):")
g_mask = mask[:M_g]
for t in range(M_g // 256):
    t_mask = g_mask[t*256:(t+1)*256]
    ncells = t_mask.sum().item()
    if ncells > 0:
        print(f"  M-tile {t} (rows {t*256}-{(t+1)*256-1}): diffs = {ncells}")

# N-col breakdown for group 0
print("\nGroup 0: per N-tile (128 cols each):")
g_mask = mask[:M_g]
for c in range(N // 128):
    c_mask = g_mask[:, c*128:(c+1)*128]
    ncells = c_mask.sum().item()
    if ncells > 0:
        print(f"  N-tile {c} (cols {c*128}-{(c+1)*128-1}): diffs = {ncells}")

# Are diffs deterministic (always same cells differ) or each run different cells?
print("\nIntersection of diff masks (cells that differ in ALL 4 reruns):")
intersect = mask.clone()
for i in range(1, 5):
    intersect &= (outs[i] != base)
inter_count = intersect.sum().item()
print(f"  Cells differing in ALL 4 reruns: {inter_count} ({100*inter_count/base.numel():.4f}%)")
print(f"  (if ~0, race position varies run-to-run; if = total_diff, same cells always)")
