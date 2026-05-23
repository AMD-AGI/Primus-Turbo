"""Single tile test: 1 tile in grid (M=256, N=128).
If deterministic → race is multi-wg related (cache state varies).
If nondet → race is truly within single wg execution.
"""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

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

torch.manual_seed(42)
cases = [
    ("SINGLE TILE K=384 [G=1 M_g=256 N=128]  (1 tile, 1 wg active)", 1, 256, 128, 384),
    ("SINGLE TILE K=256 [G=1 M_g=256 N=128]  (control: should be det)", 1, 256, 128, 256),
    ("2 TILES K=384 [G=1 M_g=512 N=128]      (2 tiles, 2 wgs)", 1, 512, 128, 384),
    ("4 TILES K=384 [G=1 M_g=1024 N=128]     (4 tiles)", 1, 1024, 128, 384),
]
for label, B, M_g, N, K in cases:
    a = (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    af, asc = _quant(a); bf, bsc = _quant(b)
    g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
    det_run(label,
        lambda af=af, bf=bf, asc=asc, bsc=bsc, g_offs=g_offs, M_g=M_g:
            hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128))
