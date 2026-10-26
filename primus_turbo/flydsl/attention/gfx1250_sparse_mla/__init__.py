###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""FlyDSL DeepSeek-V4 sparse-MLA attention for gfx1250 (MI455X): forward and backward, bf16.

The entry points are ``fwd_kernel.sparse_mla_fwd_gfx1250`` and
``bwd_kernel.sparse_mla_bwd_gfx1250`` (same contract as the Triton backend); ``helper`` holds
the wave32 WMMA / LDS / buffer primitives they share. The backward reduces dKV with the Triton
kernels in ``primus_turbo.triton.attention.sparse_mla_dkv_gather`` / ``sparse_mla_dsink``.
"""
