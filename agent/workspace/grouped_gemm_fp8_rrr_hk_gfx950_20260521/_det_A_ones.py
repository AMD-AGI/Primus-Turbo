"""A=ones, B=random. output[i,j] = sum_k(B[k,j]) = col_sum(B[:,j]).
If output bit-equal → A=1 deterministic regardless of B → race in A path.
If output nondet → race not in A.
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
a = torch.ones((B*M_g, K), dtype=torch.bfloat16, device=DEV).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

outs = [hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone() for _ in range(5)]
torch.cuda.synchronize()
print("Per-run stats (A=1, B=random; output should be col_sum(B)):")
for i, o in enumerate(outs):
    col_var = (o.max(dim=0).values - o.min(dim=0).values).abs().max().item()
    print(f"  run-{i+1}: col_max_var (should be 0 if A=1)={col_var:.6g}  max_abs={o.abs().max().item():.6g}")

print("\nBitwise equality vs run-1:")
base = outs[0]
for i in range(1, 5):
    eq = torch.equal(base, outs[i])
    diff_pct = 100.0 * (outs[i] != base).sum().item() / base.numel()
    print(f"  run-{i+1}: bit_eq={eq} differs={diff_pct:.4f}%")
