###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""The whole MegaMoE BF16 stack, tiles pinned to the f6d5ab68 snapshot.

``ed8d7af4`` (#486) rebuilt the *shared* FlyDSL BF16 grouped GEMM around
``Mfma16x16x32`` and new S2R loaders. That change is fine for the host-side
grouped-GEMM path, but it silently broke MegaMoE BF16 (it1 grad norm jumped
from ~1.45 to ~1.88e6). The tiles are therefore vendored here:

* ``gemm_helper.py`` -- MFMA atoms / S2R loaders / swizzle as of f6d5ab68
* ``gemm_bf16_kernel.py`` -- dense BF16 tile used by dispatch/combine
* ``grouped_gemm_bf16_kernel.py`` -- variable-K wgrad tile used by dispatch

The kernels that consume them, and the symmetric-memory foundation they run
on, live here too, so the BF16 path is one directory rather than a mix of
private tiles and modules sitting next to the MXFP8 package:

* ``symm_buffer.py`` -- ``SymBuffer`` + ``Workspace`` heap carving
* ``barrier.py`` -- grid sync / XGMI barrier
* ``ep_intranode.py`` -- dispatch / combine / top-k reduce tiles
* ``dispatch_prologue_kernel.py`` -- routing prologue
* ``dispatch_grouped_gemm_bf16_kernel.py``, ``grouped_gemm_combine_bf16_kernel.py``

MXFP8 has its own equivalents under ``mega/fp8`` and imports nothing from
here; the shared ``flydsl/gemm`` and ``flydsl/grouped_gemm`` trees remain
free to evolve for non-Mega callers.
"""
