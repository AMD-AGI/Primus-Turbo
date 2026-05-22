"""rocprof PMC counter capture for dsv3-up worst-case shape.
Compare grouped (bn256 RRR) vs dense (RRR) on K=7168 N=4096 M=4096."""
import sys, os, math
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch
import primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8
hk_gemm = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8

B, M_g, N, K = 4, 4096, 4096, 7168
M_total = B * M_g
torch.manual_seed(42)
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

a_d = (torch.randn((M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_d = (torch.randn((K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af_d, asc_d = quantize_fp8(a_d, float8_e4m3, ScalingGranularity.TENSORWISE)
bf_d, bsc_d = quantize_fp8(b_d, float8_e4m3, ScalingGranularity.TENSORWISE)

import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "both"

if mode in ("grouped", "both"):
    for _ in range(3):
        hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 32, torch.bfloat16, 256)  # (gm=4, xcds=32, bn=256)
torch.cuda.synchronize()

if mode in ("dense", "both"):
    for _ in range(3):
        hk_gemm(af_d, bf_d, asc_d, bsc_d, "rrr", 4, torch.bfloat16)  # gm=4
torch.cuda.synchronize()
