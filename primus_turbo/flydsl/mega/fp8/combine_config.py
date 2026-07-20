###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared block/thread config for the mega fp8 combine (L2) kernels.

These three constants are the combine kernel's 3-role (combine-push / topk-reduce / GEMM) block
layout. Extracted here as the single source of truth so the fp8 combine kernel does not have to
import them from a bf16 kernel module (keeps the fp8 subpackage free of the bf16 GEMM/combine
kernels)."""

_WARP = 64  # gfx950 wavefront
_BLOCK_THREADS = 512  # combine kernel block size
_NUM_WARPS = _BLOCK_THREADS // _WARP  # 8
