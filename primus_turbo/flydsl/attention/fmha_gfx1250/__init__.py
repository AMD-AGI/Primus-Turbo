###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Vendored gfx1250 (RDNA4 / MI4xx) FlyDSL FMHA kernels.

Self-contained FlyDSL flash-attention kernels for the DeepSeek-V3 MLA head
dims (D_qk=192, D_v=128, bf16, varlen THD). The four modules
(``fmha_kernel`` / ``fmha_core_loop`` / ``fmha_prologue`` / ``fmha_schedule``)
import only ``flydsl.*`` and each other -- there is NO runtime dependency on
any external repo. The compiled forward always supports emitting LSE
(``return_lse=True``), which the training backward pass consumes.
"""

from .fmha_kernel import (  # noqa: F401
    HEAD_DIM_QK,
    HEAD_DIM_V,
    flash_attn_varlen_d192_gfx1250,
)
