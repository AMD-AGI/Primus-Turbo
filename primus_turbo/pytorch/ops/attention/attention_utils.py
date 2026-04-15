###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import torch

from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    get_f8_fwd_dtype,
)
from primus_turbo.triton.attention.attention_kernel import FIXED_BLOCK_M

PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32_ENV_KEY = "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"


def _resolve_is_v3_atomic_fp32_from_env() -> bool:
    val = os.getenv(PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32_ENV_KEY, "1")
    return val == "1" if val in ("0", "1") else True


def _check_and_convert(t, scale, float8_fw):
    finfo = torch.finfo(float8_fw)
    return (t * scale).clamp(min=finfo.min, max=finfo.max).to(dtype=float8_fw) if t.dtype != float8_fw else t


def _infer_qkv_format(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> str:
    """Infer whether the memory layout is ``"bshd"`` or ``"sbhd"``.

    All three tensors are assumed to have **logical** shape ``[b, s, h, d]``.
    """

    def _infer_format(t: torch.Tensor) -> str:
        """Detect the memory layout of a single ``[b, s, h, d]`` tensor.

        - **bshd** contiguous: stride = (s*h*d, h*d, d, 1)
        - **sbhd** (transposed from contiguous [s,b,h,d]): stride = (h*d, b*h*d, d, 1)
        """
        b, s, h, d = t.size()
        hd = h * d
        s0, s1 = t.stride(0), t.stride(1)

        # bshd: stride[0] = dim1 * h * d,  stride[1] = h * d
        if s0 == s * hd and s1 == hd:
            return "bshd"
        # sbhd: stride[0] = h * d,  stride[1] = dim0 * h * d
        if s0 == hd and s1 == b * hd:
            return "sbhd"

        assert False, (
            f"Cannot infer qkv layout from shape {tuple(t.size())} " f"and strides {tuple(t.stride())}"
        )

    assert q.ndim == 4, f"Expected 4-D tensor for q, got {q.ndim}-D"
    q_format = _infer_format(q)

    for name, t in (("k", k), ("v", v)):
        assert t.ndim == 4, f"Expected 4-D tensor for {name}, got {t.ndim}-D"
        other_format = _infer_format(t)
        assert other_format == q_format, (
            f"Layout mismatch: q is {q_format} but {name} is {other_format}. "
            "All of q, k, v must share the same memory layout."
        )

    return q_format


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
