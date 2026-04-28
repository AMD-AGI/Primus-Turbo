###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared HipKittens integration layer for Primus-Turbo backends.

The four HIPKITTEN-typed kernel backends (dense / grouped, BF16 / FP8) all
share the same plumbing: discover and import the prebuilt ``tk_*_layouts``
module, parse its autotune cache, decide which (layout, M, N, K) shapes are
supported, and finally call the ``gemm_{rcr,rrr,crr}`` / ``grouped_*_balanced``
entrypoints with the right ``group_m`` / ``num_xcds`` / ``kernel`` config.

This subpackage centralizes that plumbing so the per-precision backend
implementations are thin wrappers around ``hipkitten.dispatch`` calls.
"""
from primus_turbo.pytorch.kernels.hipkitten.config import (
    HipKittenConfig,
    has_in_cache,
    lookup,
)
from primus_turbo.pytorch.kernels.hipkitten.dispatch import (
    dense_run,
    force_rcr_kernel,
    grouped_run_balanced,
)
from primus_turbo.pytorch.kernels.hipkitten.layout import (
    aligned_for,
    layout_of,
    padded_shape,
    round_up,
)
from primus_turbo.pytorch.kernels.hipkitten.loader import (
    HipKittenModule,
    has_bf16,
    has_fp8,
    load_bf16,
    load_fp8,
)

__all__ = [
    "HipKittenConfig",
    "HipKittenModule",
    "aligned_for",
    "dense_run",
    "force_rcr_kernel",
    "grouped_run_balanced",
    "has_bf16",
    "has_fp8",
    "has_in_cache",
    "layout_of",
    "load_bf16",
    "load_fp8",
    "lookup",
    "padded_shape",
    "round_up",
]
