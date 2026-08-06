###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import (
    fused_mega_moe_backward_fp8_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_impl import (
    fused_mega_moe_backward_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_forward_fp8_impl import (
    fused_mega_moe_forward_fp8_impl,
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

# Part of the fp8 path's contract rather than an internal helper: whoever owns the expert module has
# to call it once per optimizer step, or the experts keep training on stale quantized weights.
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_weight_prep_fp8 import (
    advance_weight_generation,
)

__all__ = [
    "fused_mega_moe_backward_impl",
    "fused_mega_moe_forward_impl",
    "fused_mega_moe_stage1_forward_impl",
    "fused_mega_moe_stage1_backward_impl",
    "fused_mega_moe_stage2_forward_impl",
    "fused_mega_moe_stage2_backward_impl",
    "fused_mega_moe_backward_fp8_impl",
    "fused_mega_moe_forward_fp8_impl",
    "fused_mega_moe_stage1_forward_fp8_impl",
    "fused_mega_moe_stage1_backward_fp8_impl",
    "fused_mega_moe_stage2_forward_fp8_impl",
    "fused_mega_moe_stage2_backward_fp8_impl",
    "advance_weight_generation",
]
