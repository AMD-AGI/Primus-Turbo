###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel registry of the attention operations (see ``primus_turbo.registry``). Data only."""

KERNELS = (
    {
        "name": "attention.flydsl.mha",
        "summary": "dense multi-head / grouped-query attention at head dim 128, forward and backward",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "dtype": ("bf16",),
        "kernel": "primus_turbo.flydsl.attention.gfx1250",
        "impl": "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl",
        "gates": (
            "primus_turbo.pytorch.kernels.attention.attention_impl:_flydsl_gfx1250_ok",
            "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl:flydsl_gfx1250_unsupported_reason",
        ),
        "tests": ("tests/pytorch/ops/test_attention_flydsl_gfx1250.py",),
    },
    {
        "name": "attention.flydsl.mla",
        "summary": "multi-head latent attention at the DeepSeek-V3 head dims (qk 192, v 128), "
        "forward and backward",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "dtype": ("bf16",),
        "kernel": "primus_turbo.flydsl.attention.gfx1250_mla",
        "impl": "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_mla_impl",
        "gates": (
            "primus_turbo.pytorch.kernels.attention.attention_impl:_flydsl_gfx1250_ok",
            "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_mla_impl:"
            "flydsl_gfx1250_mla_unsupported_reason",
        ),
        "tests": (
            "tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py",
            "tests/pytorch/ops/test_attention_flydsl_gfx1250_mla_bwd.py",
            "tests/pytorch/ops/test_attention_flydsl_gfx1250_mla_codegen.py",
        ),
    },
    {
        "name": "attention.hipkittens.mla",
        "summary": "HipKittens kernels of the MLA backward at the DeepSeek-V3 head dims; needs the code "
        "objects that only a gfx1250 build produces",
        "arch": ("gfx1250",),
        "backend": "hipkittens",
        "dtype": ("bf16",),
        "impl": "primus_turbo.pytorch.kernels.attention.attention_hipkittens_gfx1250_mla_impl",
        "gates": (
            "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_mla_impl:mla_bwd_impl",
            "primus_turbo.pytorch.kernels.attention.attention_hipkittens_gfx1250_mla_impl:"
            "hk_mla_unsupported_reason",
        ),
        "sources": ("csrc/kernels/attention/hipkittens/gfx1250_mla",),
        "products": ("lib/hk_gfx1250_mla/*.hsaco",),
        "tests": ("tests/pytorch/ops/test_attention_hipkittens_gfx1250_mla.py",),
    },
    {
        "name": "attention.flydsl.sparse_mla",
        "summary": "DeepSeek-V4 sparse MLA, forward and backward (the wave32 kernels of the FlyDSL backend)",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "dtype": ("bf16",),
        "kernel": "primus_turbo.flydsl.attention.gfx1250_sparse_mla",
        "gates": (
            "primus_turbo.pytorch.kernels.attention.sparse_mla_impl:SparseMlaFwdFlydslBackend",
            "primus_turbo.pytorch.kernels.attention.sparse_mla_impl:SparseMlaBwdFlydslBackend",
        ),
        "tests": (
            "tests/pytorch/ops/test_attention_flydsl_gfx1250_sparse_mla.py",
            "tests/pytorch/ops/test_attention.py::test_sparse_mla_op",
        ),
    },
)
