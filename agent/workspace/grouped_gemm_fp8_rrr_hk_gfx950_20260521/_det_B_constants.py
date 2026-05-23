"""Distinguish 'mfma data-dependent' vs 'B LDS read race masked by all-same-bytes'.
Test B = various constants AND B = mostly-same-with-few-different to pinpoint.

If B=2.0, 0.5 constants also det → "all-bytes-same" hides LDS race, race in B LDS read
If B=ones-only det → ones is special, mfma data-dependent
If B with one different byte triggers race only at that byte → LDS read race confirmed
"""
import sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch, primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

DEV = "cuda"
hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

def det_run(label, a_factory, b_factory, n=5):
    torch.manual_seed(42)
    a = a_factory()
    b = b_factory()
    af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
    bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
    g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
    outs = [hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone() for _ in range(n)]
    torch.cuda.synchronize()
    base = outs[0]
    n_diff = sum(1 for i in range(1, n) if not torch.equal(base, outs[i]))
    pcts = [100.0*(outs[i]!=base).sum().item()/base.numel() for i in range(1, n)]
    max_abs = outs[0].abs().max().item()
    bf_unique = bf.view(torch.uint8).unique().numel()
    print(f"{label:<55} bf_unique_bytes={bf_unique}  nondet={n_diff}/{n-1}  diffs={pcts}  max_abs={max_abs:.3f}")

B, M_g, N, K = 1, 4096, 128, 384

# Test 1: B = ones (control)
det_run("B=1.0 const", lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        lambda: torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV).contiguous())

# Test 2: B = 2.0 const
det_run("B=2.0 const", lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        lambda: (torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV) * 2.0).contiguous())

# Test 3: B = 0.5 const
det_run("B=0.5 const", lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        lambda: (torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.5).contiguous())

# Test 4: B = 0.05 const (small, like quantized random magnitude)
det_run("B=0.05 const", lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        lambda: (torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous())

# Test 5: B = 2 distinct values block-checker (half 1.0, half 2.0)
def b_block():
    bb = torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV)
    bb[:, K//2:, :] = 2.0  # Second half K = 2.0
    return bb.contiguous()
det_run("B=block(half-1,half-2)", lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        b_block)

# Test 6: B = random in {1.0, 2.0} discrete only 2 fp8 values
def b_2val():
    return (torch.randint(0, 2, (B, K, N), dtype=torch.uint8, device=DEV).to(torch.bfloat16) + 1.0).contiguous()
det_run("B=random in {1.0,2.0}", lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        b_2val)

# Test 7: B = random in {0.0, 1.0} sparse
def b_sparse():
    return (torch.randint(0, 2, (B, K, N), dtype=torch.uint8, device=DEV).to(torch.bfloat16)).contiguous()
det_run("B=random in {0.0,1.0}", lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        b_sparse)
