"""Mirror test: A patterns with B=const.
If A also fails when bytes differ, A path has same race.
If A doesn't trigger race even with varied bytes (B is const), race is B-specific.
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
    af_u = af.view(torch.uint8).unique().numel()
    bf_u = bf.view(torch.uint8).unique().numel()
    print(f"{label:<55} af_u={af_u} bf_u={bf_u}  nondet={n_diff}/{n-1}  diffs[max]={max(pcts):.4f}")

B, M_g, N, K = 1, 4096, 128, 384

# Cross-test matrix:
print("=== A const, B const (control: should all be det) ===")
det_run("A=1.0 B=1.0", lambda: torch.ones((B*M_g, K), dtype=torch.bfloat16, device=DEV),
        lambda: torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV))
print()

print("=== A varied bytes, B const (test A path) ===")
det_run("A=random B=1.0",
        lambda: (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous(),
        lambda: torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV))
det_run("A=block(half-1,half-2) B=1.0",
        lambda: torch.cat([torch.ones((B*M_g, K//2), dtype=torch.bfloat16, device=DEV),
                          torch.ones((B*M_g, K//2), dtype=torch.bfloat16, device=DEV)*2.0], dim=1).contiguous(),
        lambda: torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV))
det_run("A=random {1,2} B=1.0",
        lambda: (torch.randint(0, 2, (B*M_g, K), dtype=torch.uint8, device=DEV).to(torch.bfloat16) + 1.0).contiguous(),
        lambda: torch.ones((B, K, N), dtype=torch.bfloat16, device=DEV))
print()

print("=== A const, B varied bytes (test B path, control) ===")
det_run("A=1.0 B=random",
        lambda: torch.ones((B*M_g, K), dtype=torch.bfloat16, device=DEV),
        lambda: (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous())
det_run("A=1.0 B=block(half-1,half-2)",
        lambda: torch.ones((B*M_g, K), dtype=torch.bfloat16, device=DEV),
        lambda: torch.cat([torch.ones((B, K//2, N), dtype=torch.bfloat16, device=DEV),
                          torch.ones((B, K//2, N), dtype=torch.bfloat16, device=DEV)*2.0], dim=1).contiguous())
