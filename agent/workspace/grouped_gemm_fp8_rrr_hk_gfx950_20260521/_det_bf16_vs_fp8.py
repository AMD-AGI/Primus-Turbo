"""Compare bf16 grouped RRR det vs fp8 grouped RRR det.
If bf16 deterministic → bug is fp8-specific (mfma_f8 or fp8 scale path).
If bf16 also nondet → bug is in grouped framework (persistent loop/dispatcher/store/LDS path).
"""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp_fp8  = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8
hk_grp_bf16 = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_bf16

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

# Same shape as fp8 worst nondet: qwen-down B=16 M_g=8192 K=1536 N=4096
B, M_g, N, K = 16, 8192, 4096, 1536
M_total = B * M_g

a_bf = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_bf = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

# bf16 path
det_run(
    f"BF16 grouped RRR [B={B} M_g={M_g} K={K} N={N}]",
    lambda: hk_grp_bf16(a_bf, b_bf, g_offs, 4, M_g, 4))

# fp8 path (control - should be nondet as before)
af, asc = quantize_fp8(a_bf, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b_bf, float8_e4m3, ScalingGranularity.TENSORWISE)
det_run(
    f"FP8 grouped RRR bn128 [same shape] (control)",
    lambda: hk_grp_fp8(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128))
