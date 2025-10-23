###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch


def quantize_fp8_tensorwise_impl(
    x: torch.Tensor, out_dtype: torch.dtype, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Tensor-Wise
    """
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, out_dtype, scale)
    return x_fp8, scale_inv


def quantize_fp8_rowwise_impl(
    x: torch.Tensor, out_dtype: torch.dtype, axis: int, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Row-Wise
    """
    if not x.is_contiguous():
        x = x.contiguous()
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_rowwise(x, out_dtype, axis, scale)
    return x_fp8, scale_inv


def dequantize_fp8_tensorwise_impl(x: torch.Tensor, out_dtype: torch.dtype, scale_inv: torch.Tensor):
    """
    DeQuantize FP8 Tensor-Wise
    """
    return torch.ops.primus_turbo_cpp_extension.dequantize_fp8_tensorwise(x, scale_inv, out_dtype)


def dequantize_fp8_rowwise_impl(x: torch.Tensor, out_dtype: torch.dtype, axis: int, scale_inv: torch.Tensor):
    """
    DeQuantize FP8 Row-Wise
    """
    raise NotImplementedError(f"Un-impl")
