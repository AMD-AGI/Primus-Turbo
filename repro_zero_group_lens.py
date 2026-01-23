#!/usr/bin/env python3
"""
Minimal reproduction for: block-wise scaling FP8 group GEMM crashed if group_lens contains 0

Issue: When group_lens contains zeros (valid MoE scenario where some experts
receive no tokens), the block-wise scaling FP8 grouped GEMM crashes with
illegal memory access in backward pass.

Root cause: In backward, grouped_gemm_fp8_variable_k_impl (CK kernel) fails
when processing groups with 0 tokens.

Usage: python repro_zero_group_lens.py
"""

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8

torch.manual_seed(42)
device = "cuda:0"
dtype = torch.bfloat16

# MoE scenario: 8 experts, only first 2 have tokens
E = 8
in_features = 2048
out_features = 8192

# BUG TRIGGER: group_lens contains zeros
group_lens = torch.tensor([8192, 8192, 0, 0, 0, 0, 0, 0], dtype=torch.int64, device=device)
total_m = group_lens.sum().item()

print(f"E={E}, in_features={in_features}, out_features={out_features}")
print(f"group_lens = {group_lens.tolist()}")
print(f"total_m = {total_m}")

a = torch.randn((total_m, in_features), dtype=dtype, device=device, requires_grad=True)
b = torch.randn((E, out_features, in_features), dtype=dtype, device=device, requires_grad=True)

config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128)

print("\nForward pass...")
out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
torch.cuda.synchronize()
print(f"Forward OK: out.shape = {out.shape}")

print("\nBackward pass...")
try:
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    print("Backward OK")
except RuntimeError as e:
    print(f"Backward CRASHED!")
    print(f"\n>>> BUG REPRODUCED <<<")
    print(f"Error: {e}")
