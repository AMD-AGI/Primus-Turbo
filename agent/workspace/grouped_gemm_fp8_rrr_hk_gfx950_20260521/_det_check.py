"""Bitwise determinism: same input, kernel run 5x, check torch.equal."""
import os, sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
import primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.core.backend import GlobalBackendManager, BackendType

DEV = "cuda"
cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
GlobalBackendManager.set_auto_tune(False)

CASES = [
    ("dsv3-down-B16-M2048", 16, 2048, 7168, 2048),
    ("qwen-down-B16-M8192", 16, 8192, 4096, 1536),  # worst delta
]

for label, B, M_g, N, K in CASES:
    print(f"\n=== {label}: B={B} M_g={M_g} N={N} K={K} ===")
    torch.manual_seed(42)
    a = (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    b = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    a.requires_grad_(True); b.requires_grad_(True)
    group_lens = torch.full((B,), M_g, dtype=torch.int64, device=DEV)

    outs = []
    for i in range(5):
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
        outs.append(out.detach().clone())
        torch.cuda.synchronize()

    base = outs[0]
    for i in range(1, 5):
        diff = outs[i] - base
        max_abs = diff.abs().max().item()
        eq = torch.equal(base, outs[i])
        nz = (diff != 0).sum().item()
        print(f"  run-1 vs run-{i+1}: bitwise_equal={eq}  max_abs_diff={max_abs:.6g}  nonzero_diffs={nz}/{base.numel()} ({100.0*nz/base.numel():.3f}%)")
