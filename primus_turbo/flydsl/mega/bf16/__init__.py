###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""MegaMoE-private BF16 FlyDSL tiles, pinned to the f6d5ab68 snapshot.

``ed8d7af4`` (#486) rebuilt the *shared* FlyDSL BF16 grouped GEMM around
``Mfma16x16x32`` and new S2R loaders. That change is fine for the host-side
grouped-GEMM path, but it silently broke MegaMoE BF16 (it1 grad norm jumped
from ~1.45 to ~1.88e6). Mega MoE therefore keeps its own copies of:

* ``gemm_helper.py`` -- MFMA atoms / S2R loaders / swizzle as of f6d5ab68
* ``gemm_bf16_kernel.py`` -- dense BF16 tile used by dispatch/combine
* ``grouped_gemm_bf16_kernel.py`` -- variable-K wgrad tile used by dispatch

Only ``primus_turbo.flydsl.mega.*`` BF16 kernels import from this package.
The shared ``flydsl/gemm`` and ``flydsl/grouped_gemm`` trees remain free to
evolve for non-Mega callers.
"""
