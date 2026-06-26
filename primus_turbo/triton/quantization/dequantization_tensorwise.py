###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Standalone Triton implementation of tensorwise FP8 dequantize."""

import torch
import triton
import triton.language as tl


@triton.jit
def dequantize_tensorwise_kernel(
    x_ptr,
    scale_inv_ptr,
    y_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = offs.to(tl.int64)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale_inv = tl.load(scale_inv_ptr).to(tl.float32)  # single global scalar
    y = x * scale_inv
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def dequantize_fp8_tensorwise_triton(
    x: torch.Tensor,
    scale_inv: torch.Tensor,
    dest_dtype: torch.dtype,
) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    assert scale_inv.numel() == 1, "tensorwise scale_inv must be a scalar tensor"
    assert dest_dtype in (torch.bfloat16, torch.float16, torch.float32)

    n = x.numel()
    out = torch.empty_like(x, dtype=dest_dtype)
    if n == 0:
        return out

    scale_inv_f32 = scale_inv.to(torch.float32).reshape(())
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    dequantize_tensorwise_kernel[grid](x, scale_inv_f32, out, n, BLOCK=BLOCK)
    return out
