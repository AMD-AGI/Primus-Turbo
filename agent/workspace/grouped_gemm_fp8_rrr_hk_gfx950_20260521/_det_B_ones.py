"""Force B = constant ones. Output[i,j] = sum_k(A[i,k] * 1) = row_sum(A[i,:]).
Same value across all N cols. Trace race to A reads vs B reads.

If output bit-equal across runs → A reads consistent (race not in A path).
If output rows-but-cols-equal-within-rows AND nondet → race in A.
If output cols differ within rows → race in B (since B=1 const, B can't race).
"""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

torch.manual_seed(42)
B, M_g, N, K = 1, 4096, 128, 384
a = (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
# B = ones — fp8 representable
b = torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

outs = [hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone() for _ in range(5)]
torch.cuda.synchronize()
print("Per-run stats (output should be row-sum(A)):")
for i, o in enumerate(outs):
    # Each row should have same value across N
    row_var = (o.max(dim=1).values - o.min(dim=1).values).abs().max().item()
    print(f"  run-{i+1}: row_max_var (should be 0 if B=1)={row_var:.6g}  max_abs={o.abs().max().item():.6g}")

print("\nBitwise equality vs run-1:")
base = outs[0]
for i in range(1, 5):
    eq = torch.equal(base, outs[i])
    diff_pct = 100.0 * (outs[i] != base).sum().item() / base.numel()
    print(f"  run-{i+1}: bit_eq={eq} differs={diff_pct:.4f}%")

# Locate where diffs cluster: rows? cols?
print("\nDiff distribution (run-1 vs run-2):")
diff_mask = (outs[1] != base)
if diff_mask.any():
    row_diffs = diff_mask.any(dim=1)
    col_diffs = diff_mask.any(dim=0)
    print(f"  diff rows: {row_diffs.sum().item()}/{base.shape[0]}")
    print(f"  diff cols: {col_diffs.sum().item()}/{base.shape[1]}")
    # If diff is row-wise consistent (whole row differs), B=1 wouldn't cause this.
    # If diff is col-wise (only certain cols), but B=1 means cols are identical → cols differ → race involves cell-level.
    print(f"  total diff cells: {diff_mask.sum().item()}")
