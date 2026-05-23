"""Verify the fp8 inputs (af, bf) passed to hk_grp are bitwise-identical
across the 5 reruns. If they differ → quantize_fp8 is the source of nondet,
not the kernel itself.
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
B, M_g, N, K = 1, 256, 128, 384
a = (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
torch.cuda.synchronize()

# Capture af, bf bytes
af_bytes_0 = af.view(torch.uint8).clone()
bf_bytes_0 = bf.view(torch.uint8).clone()
asc_0 = asc.clone()
bsc_0 = bsc.clone()

# Now call hk_grp 5 times. Between calls verify af/bf bytes unchanged.
print(f"af shape={tuple(af.shape)} dtype={af.dtype} scale={asc.item():.6e}")
print(f"bf shape={tuple(bf.shape)} dtype={bf.dtype} scale={bsc.item():.6e}")
print()
print("Calling hk_grp 5 times. Verify af/bf bytes stable between calls.")

outs = []
for r in range(5):
    out = hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone()
    torch.cuda.synchronize()
    outs.append(out)

    # Verify af, bf bytes unchanged
    af_now = af.view(torch.uint8)
    bf_now = bf.view(torch.uint8)
    af_eq = torch.equal(af_bytes_0, af_now)
    bf_eq = torch.equal(bf_bytes_0, bf_now)
    asc_eq = (asc == asc_0).item()
    bsc_eq = (bsc == bsc_0).item()
    print(f"  After run-{r+1}: af bytes_eq={af_eq}  bf bytes_eq={bf_eq}  scale_eq={asc_eq},{bsc_eq}")

print()
base = outs[0]
for i in range(1, 5):
    diff = outs[i] - base
    nz = (diff != 0).sum().item()
    mx = diff.abs().max().item()
    eq = torch.equal(base, outs[i])
    print(f"out run-1 vs run-{i+1}: equal={eq} diffs={100.0*nz/base.numel():.4f}% max_abs={mx:.6g}")
