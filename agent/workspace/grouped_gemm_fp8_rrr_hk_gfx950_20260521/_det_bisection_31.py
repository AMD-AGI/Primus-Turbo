"""Bisection #31: hard sched_barrier(0) at every site in bn128 main loop.
If race fixed → compiler scheduler reorder IS the mechanism (and that's a
specific finding: 'compiler emits an instruction sequence that the runtime
HW pipeline cannot tolerate without explicit ordering hints'). 
If race persists → really is HW issue."""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

torch.manual_seed(42)
# Same K=384 minimal nondet shape used throughout
B, M_g, N, K = 16, 8192, 4096, 1536
M_total = B * M_g

a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

def run():
    return hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone()

outs = [run() for _ in range(5)]
torch.cuda.synchronize()

print(f"bisection #31: sched_barrier(0) at EVERY site in main loop")
print(f"Shape: B={B} M_g={M_g} N={N} K={K}")
base = outs[0]
for i in range(1, 5):
    eq = torch.equal(base, outs[i])
    diff_pct = 100.0 * (outs[i] != base).sum().item() / base.numel()
    print(f"  run-1 vs run-{i+1}: bit_eq={eq}  differs={diff_pct:.4f}%")

# Also do K=384 minimal
print()
print("K=384 minimal nondet shape:")
B, M_g, N, K = 1, 4096, 256, 384
M_total = B * M_g
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

def run2():
    return hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone()
outs = [run2() for _ in range(5)]
torch.cuda.synchronize()
base = outs[0]
for i in range(1, 5):
    eq = torch.equal(base, outs[i])
    diff_pct = 100.0 * (outs[i] != base).sum().item() / base.numel()
    print(f"  run-1 vs run-{i+1}: bit_eq={eq}  differs={diff_pct:.4f}%")
