###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The AdaLN modulation GEMMs at a 32-row micro-batch (see ``kernels/gemm/gemm_adaln_impl.py``). The ops are not
autograd-aware: a caller's own ``autograd.Function`` calls the forward, dgrad and wgrad ops it needs."""

from primus_turbo.pytorch.kernels.gemm.gemm_adaln_impl import (
    ADALN_GEMM_M,
    adaln_gemm_dgrad,
    adaln_gemm_fwd,
    adaln_gemm_table,
    adaln_gemm_wgrad_out,
)

__all__ = ["ADALN_GEMM_M", "adaln_gemm_dgrad", "adaln_gemm_fwd", "adaln_gemm_table", "adaln_gemm_wgrad_out"]
