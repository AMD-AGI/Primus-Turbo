###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch


def quantize_fp8_tensorwise_impl(
    x: torch.Tensor, dtype: torch.dtype, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Tensor-Wise
    """
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, dtype, scale)
    return x_fp8, scale_inv


def quantize_fp8_rowwise_impl(
    x: torch.Tensor, dtype: torch.dtype, axis: int, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Row-Wise
    """
    if not x.is_contiguous():
        x = x.contiguous()
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_rowwise(x, dtype, axis)
    return x_fp8, scale_inv
