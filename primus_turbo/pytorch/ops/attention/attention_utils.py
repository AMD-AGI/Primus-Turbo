###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    get_f8_fwd_dtype,
)
from primus_turbo.triton.attention.attention_kernel import FIXED_BLOCK_M


def _check_and_convert(t, scale, float8_fw):
    finfo = torch.finfo(float8_fw)
    return (t * scale).clamp(min=finfo.min, max=finfo.max).to(dtype=float8_fw) if t.dtype != float8_fw else t


def block_scaling_node(tensor, use_fp8, BLOCK_M=FIXED_BLOCK_M, float8_dtype=get_f8_fwd_dtype()):
    """
    Used to scale tensor in per-block mode

    Inputs:
        tensor(Tensor): bf16 tensor
        BLOCK_M(int): triton block size
        float8_dtype(Tensor.dtype): float8_dtype

    Output:
        fp8tensor(Tensor): tensor after blockwise quant
        unscale_tensor(Tensor): tensor for unscale quanted tensor from fp8 to bf16
    """
    if use_fp8:
        tensor = tensor.permute(0, 2, 1, 3)  # [B, H, L, D]
        B, H, L, D = tensor.shape
        tensor = tensor.reshape(B, H, L // BLOCK_M, BLOCK_M, D).reshape(B, H, L // BLOCK_M, BLOCK_M * D)
        MAX_E4M3 = torch.finfo(float8_dtype).max
        tensor_max = tensor.abs().max(dim=-1)[0]
        tensor_max = torch.where(tensor_max == 0, MAX_E4M3, tensor_max)
        scale = MAX_E4M3 / tensor_max
        tensor = tensor * scale.reshape(scale.shape + (1,))
        tensor = tensor.clamp(-MAX_E4M3, MAX_E4M3)
        tensor = tensor.to(float8_dtype)
        tensor = tensor.reshape(B, H, L, D).permute(0, 2, 1, 3).contiguous()
        # [B, L, H, D]
        return tensor, 1.0 / scale.to(torch.float32).contiguous()
    else:
        scale = torch.tensor([1.0], device=tensor.device)
        return tensor, scale


def get_p_scale(use_fp8: bool):
    """
    Get p_scale for FA internal quantization
    """
    if use_fp8:
        float8_fw = get_f8_fwd_dtype()
        dtype_max = torch.finfo(float8_fw).max
        p_scale = dtype_max
    else:
        p_scale = 1.0

    return p_scale
