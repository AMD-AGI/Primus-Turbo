###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_impl import (
    fused_mega_moe_backward_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_forward_impl import (
    fused_mega_moe_forward_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_stage1_fp8_impl import (
    fused_mega_moe_stage1_backward_fp8_impl,
    fused_mega_moe_stage1_forward_fp8_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_stage1_impl import (
    fused_mega_moe_stage1_backward_impl,
    fused_mega_moe_stage1_forward_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_stage2_fp8_impl import (
    fused_mega_moe_stage2_backward_fp8_impl,
    fused_mega_moe_stage2_forward_fp8_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_stage2_impl import (
    fused_mega_moe_stage2_backward_impl,
    fused_mega_moe_stage2_forward_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_fp8_weights import (
    advance_weight_generation,  # per-optimizer-step weight invalidation; part of the fp8 contract
)

__all__ = [
    "fused_mega_moe_backward_impl",
    "fused_mega_moe_forward_impl",
    "fused_mega_moe_stage1_forward_impl",
    "fused_mega_moe_stage1_backward_impl",
    "fused_mega_moe_stage2_forward_impl",
    "fused_mega_moe_stage2_backward_impl",
    "fused_mega_moe_stage1_forward_fp8_impl",
    "fused_mega_moe_stage1_backward_fp8_impl",
    "fused_mega_moe_stage2_forward_fp8_impl",
    "fused_mega_moe_stage2_backward_fp8_impl",
    "advance_weight_generation",
]
