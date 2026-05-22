"""Round 3: Quick sweep of (RRR_STEADY_VMCNT, RRR_PREFETCH_LGKM) for bn256
on dsv3-up worst-case shape. Single shape: dsv3-up-B4-M4096 (N=4096 K=7168).
Records TFLOPS for each combo. Just need to rebuild for each combo."""
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

# dsv3-up-B4-M4096
B, M_g, N, K = 4, 4096, 4096, 7168
M_total = B * M_g
torch.manual_seed(42)
a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))

# Dense ref
a_d = (torch.randn((M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b_d = (torch.randn((K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af_d, asc_d = quantize_fp8(a_d, float8_e4m3, ScalingGranularity.TENSORWISE)
bf_d, bsc_d = quantize_fp8(b_d, float8_e4m3, ScalingGranularity.TENSORWISE)

def time_us(f, w=10, n=100):
    for _ in range(w):
        f(); torch.cuda.synchronize()
    e = [torch.cuda.Event(enable_timing=True) for _ in range(n+1)]
    e[0].record()
    for i in range(n):
        f(); e[i+1].record()
    torch.cuda.synchronize()
    return sum(e[i].elapsed_time(e[i+1]) for i in range(n)) / n * 1000.0

# Dense best
best_d = math.inf
for gm in [1, 2, 4, 8, 16]:
    try:
        us = time_us(lambda: hk_gemm(af_d, bf_d, asc_d, bsc_d, "rrr", gm, torch.bfloat16))
        best_d = min(best_d, us)
    except: pass
T_d = 2.0*M_g*N*K/(best_d*1e6)

# Grouped sweep
GRP_CANDS = [(2,4),(4,4),(4,16),(4,32),(8,32),(8,16),(4,8),(16,4),(1,4),(8,4),(1,0),(4,0),(8,0),(16,0)]
best_us, best_cfg = math.inf, None
for bn in [128, 256]:
    if bn == 256 and N % 256 != 0: continue
    for gm, xcds in GRP_CANDS:
        try:
            us = time_us(lambda: hk_grp(af, bf, asc, bsc, g_offs, gm, M_g, xcds, torch.bfloat16, bn))
            if us < best_us: best_us, best_cfg = us, (gm, xcds, bn)
        except: pass
T_g = 2.0*B*M_g*N*K/(best_us*1e6)

ratio = T_g/T_d
print(f"dsv3-up-B4-M4096 (B={B} M_g={M_g} N={N} K={K}):")
print(f"  dense:  {T_d:.0f} TFLOPS ({best_d:.1f} us)")
print(f"  grp:    {T_g:.0f} TFLOPS ({best_us:.1f} us)  cfg={best_cfg}")
print(f"  ratio:  {ratio:.4f}  gap: {(1-ratio)*100:.2f}%")
