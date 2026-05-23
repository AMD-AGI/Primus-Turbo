"""Run DENSE hk_gemm_rrr with same shape and B variance as grouped K=384.
Dense uses identical v_mfma_f32_16x16x128_f8f6f4 instruction.
If dense det → mfma instruction is fine, race is in grouped's persistent/dispatch overhead.
If dense nondet → mfma itself races for this operand pattern."""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_gemm = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8

torch.manual_seed(42)
# Same K=384 as grouped nondet test, dense single tile shape
# Dense expects [M, K] x [K, N], all aligned to 256 typically
M, K, N = 4096, 384, 256  # Dense needs N>=256; use 256

a = (torch.randn((M, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()  # K x N for "rrr"
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)

outs = [hk_gemm(af, bf, asc, bsc, "rrr", 4, torch.bfloat16).detach().clone() for _ in range(5)]
torch.cuda.synchronize()

print(f"DENSE hk_gemm_rrr K={K} M={M} N={N}, random A and random B:")
for i, o in enumerate(outs):
    print(f"  run-{i+1}: max_abs={o.abs().max().item():.6g}  shape={tuple(o.shape)}")
print()
base = outs[0]
for i in range(1, 5):
    eq = torch.equal(base, outs[i])
    diff_pct = 100.0 * (outs[i] != base).sum().item() / base.numel()
    print(f"  run-1 vs run-{i+1}: bit_eq={eq}  differs={diff_pct:.4f}%")
