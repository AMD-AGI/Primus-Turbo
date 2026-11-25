###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel registry of the grouped GEMM (see ``primus_turbo.registry``). Data only."""

KERNELS = (
    {
        "name": "grouped_gemm.flydsl.bf16",
        "summary": "bf16 grouped GEMM on WMMA / TDM: NT, NN and the variable-K weight gradient; "
        "'wgrad_accum' is the variable-K gradient accumulated into an fp32 main_grad",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "dtype": ("bf16", "fp16"),
        "kernel": "primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250",
        "gates": (
            "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl:GroupedGEMMFlyDSLBackend",
            "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl:GroupedGEMMVariableKFlyDSLBackend",
        ),
        "provides": {"wgrad_accum": "grouped_gemm_bf16_variable_k_accum_supported"},
        "tests": (
            "tests/pytorch/ops/test_grouped_gemm.py",
            "tests/pytorch/ops/test_grouped_gemm_flydsl_gfx1250_shapes.py",
            "tests/pytorch/ops/test_grouped_gemm_flydsl_gfx1250_keys.py",
            "tests/pytorch/ops/test_grouped_gemm_flydsl_gfx1250_kwargs_keys.py",
            "tests/pytorch/ops/test_grouped_gemm_flydsl_gfx1250_variable_k_accum.py",
        ),
        "benchmarks": ("benchmark/ops/training/bench_grouped_gemm_variable_k_accum_gfx1250.py",),
    },
)
