"""Verify hypothesis: RRR_STEADY_VMCNT 4→0 fixes grouped non-det without breaking dense.
- DENSE hk_gemm RRR: should remain deterministic (5/5 bit-equal)
- GROUPED G=1, G=16: should now be deterministic
"""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_gemm = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8
hk_grp  = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

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

# Same shape as earlier scope test (qwen-down style)
N, K = 4096, 1536
M_g_grp16 = 8192; B = 16
M_total = B * M_g_grp16  # 131072

torch.manual_seed(42)
# Dense at M_total (would be N*K-heavy but Mt is most M_total we'll see)
a_d = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_d = (torch.randn((K, N),       dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af_d, asc_d = _quant(a_d); bf_d, bsc_d = _quant(b_d)
det_run(
    f"DENSE hk_gemm rrr [M={M_total}, K={K}] x [K, N={N}]",
    lambda: hk_gemm(af_d, bf_d, asc_d, bsc_d, "rrr", 4, torch.bfloat16))

# Grouped G=1 (single group, persistent loop still active)
b_g1 = b_d.unsqueeze(0).contiguous()
af_g1, asc_g1 = _quant(a_d); bf_g1, bsc_g1 = _quant(b_g1)
g_offs_1 = grouped_gemm_compute_offs(torch.tensor([M_total], dtype=torch.int64, device=DEV))
det_run(
    f"GROUPED G=1 bn=128 [M_g={M_total}, K={K}, N={N}]",
    lambda: hk_grp(af_g1, bf_g1, asc_g1, bsc_g1, g_offs_1, 4, M_total, 4, torch.bfloat16, 128))

# Grouped G=16 (worst nondet case)
a_g16 = (torch.randn((B*M_g_grp16, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_g16 = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af_g16, asc_g16 = _quant(a_g16); bf_g16, bsc_g16 = _quant(b_g16)
g_offs_16 = grouped_gemm_compute_offs(torch.full((B,), M_g_grp16, dtype=torch.int64, device=DEV))
det_run(
    f"GROUPED G=16 bn=128 [B={B}, M_g={M_g_grp16}, K={K}, N={N}]",
    lambda: hk_grp(af_g16, bf_g16, asc_g16, bsc_g16, g_offs_16, 4, M_g_grp16, 4, torch.bfloat16, 128))

# bn256 grouped (dsv3-up-B16-M8192 was earlier deterministic in bitwise, but try bn256 grouped just in case)
N2, K2 = 4096, 7168
M_g2 = 8192
a_g256 = (torch.randn((B*M_g2, K2), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_g256 = (torch.randn((B, K2, N2), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af_256, asc_256 = _quant(a_g256); bf_256, bsc_256 = _quant(b_g256)
g_offs_256 = grouped_gemm_compute_offs(torch.full((B,), M_g2, dtype=torch.int64, device=DEV))
det_run(
    f"GROUPED G=16 bn=256 [B={B}, M_g={M_g2}, K={K2}, N={N2}]  (dsv3-up shape)",
    lambda: hk_grp(af_256, bf_256, asc_256, bsc_256, g_offs_256, 4, M_g2, 4, torch.bfloat16, 256))
