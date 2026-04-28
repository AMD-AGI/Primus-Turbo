###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared HipKittens integration layer for Primus-Turbo backends.

The four HIPKITTEN-typed kernel backends (dense / grouped, BF16 / FP8) all
share the same plumbing:

  1. Discover and import the prebuilt ``tk_*_layouts`` extension via
     :func:`load_bf16` / :func:`load_fp8`.
  2. Decide whether ``(layout, M, N, K, dtype)`` is supportable in
     ``can_handle`` (alignment + dtype + layout — no shape lookup).
  3. Pick a per-call config via :func:`select_default_config` (pure if/else
     rules, no IO).
  4. Launch the kernel via ``dispatch.dense_run`` /
     ``dispatch.grouped_run_balanced``.

This subpackage centralizes that plumbing so the per-precision backend
implementations are thin wrappers around ``hipkitten.dispatch`` calls.
"""
from primus_turbo.pytorch.kernels.hipkitten.config import (
    HipKittenConfig,
    select_default_config,
)
from primus_turbo.pytorch.kernels.hipkitten.dispatch import (
    dense_run,
    force_rcr_kernel,
    fp8_has_dscale,
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
    "fp8_has_dscale",
    "grouped_run_balanced",
    "has_bf16",
    "has_fp8",
    "layout_of",
    "load_bf16",
    "load_fp8",
    "padded_shape",
    "round_up",
    "select_default_config",
]
