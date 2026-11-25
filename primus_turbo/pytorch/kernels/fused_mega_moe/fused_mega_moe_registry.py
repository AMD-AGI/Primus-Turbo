###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel registry of the fused MegaMoE (see ``primus_turbo.registry``). Data only."""

KERNELS = (
    {
        "name": "mega_moe.bf16",
        "summary": "MegaMoE: dispatch + GEMM + combine in fused kernels, forward and backward",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "dtype": ("bf16",),
        "kernel": "primus_turbo.flydsl.mega.gfx1250",
        "impl": "primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_gfx1250_impl",
        "gates": (
            "primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_forward_impl:"
            "FusedMegaMoEForwardFlyDSLBackend",
            "primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_impl:"
            "FusedMegaMoEBackwardFlyDSLBackend",
            "primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_gfx1250_impl:gfx1250_supports",
        ),
        "tests": (
            "tests/pytorch/ops/test_mega_bf16_gfx1250_prims.py",
            "tests/pytorch/ops/test_mega_bf16_gfx1250_gemm.py",
            "tests/pytorch/ops/test_mega_bf16_gfx1250_variable_k_accum.py",
            "tests/pytorch/ops/test_mega_bf16_gfx1250_dispatch_gemm.py",
            "tests/pytorch/ops/test_mega_bf16_gfx1250_gemm_combine.py",
            "tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe.py",
            "tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe_ep.py",
        ),
        "benchmarks": ("benchmark/ops/training/bench_mega_moe_gfx1250.py",),
    },
)
