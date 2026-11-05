###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tensorwise FP8 quantization with a precomputed scale.

Used by FSDP2 FP8 all-gather: every shard is cast with the same globally
all-reduced scale so the unsharded FP8 weight carries one consistent scale.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _quantize_fp8_prescaled_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr)
    v = x * scale
    v = tl.clamp(v, min=-FP8_MAX, max=FP8_MAX)
    tl.store(output_ptr + offsets, v.to(output_ptr.dtype.element_ty), mask=mask)


@triton_op("primus_turbo::quantize_fp8_prescaled", mutates_args=())
def quantize_fp8_prescaled(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(fp8_dtype).max
    output = torch.empty_like(x, dtype=fp8_dtype)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    wrap_triton(_quantize_fp8_prescaled_kernel)[grid](
        x,
        output,
        scale,
        n_elements,
        fp8_max,
        BLOCK_SIZE=1024,
    )
    return output, scale_inv.clone()


@quantize_fp8_prescaled.register_fake
def _quantize_fp8_prescaled_fake(x, fp8_dtype, scale, scale_inv):
    return torch.empty_like(x, dtype=fp8_dtype), scale_inv.clone()


# ---------------------------------------------------------------------------
# Stochastic rounding variant: adds uniform [-0.5, 0.5) noise before
# truncation to make quantization error unbiased in expectation.
# ---------------------------------------------------------------------------


@triton.jit
def _quantize_fp8_prescaled_stochastic_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    seed,
    n_elements,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr)
    v = x * scale
    v = tl.clamp(v, min=-FP8_MAX, max=FP8_MAX)
    noise = tl.rand(seed, offsets) - 0.5
    v_sr = v + noise
    tl.store(output_ptr + offsets, v_sr.to(output_ptr.dtype.element_ty), mask=mask)


@triton_op("primus_turbo::quantize_fp8_prescaled_stochastic", mutates_args=())
def quantize_fp8_prescaled_stochastic(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(fp8_dtype).max
    output = torch.empty_like(x, dtype=fp8_dtype)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    wrap_triton(_quantize_fp8_prescaled_stochastic_kernel)[grid](
        x,
        output,
        scale,
        seed,
        n_elements,
        fp8_max,
        BLOCK_SIZE=1024,
    )
    return output, scale_inv.clone()


@quantize_fp8_prescaled_stochastic.register_fake
def _quantize_fp8_prescaled_stochastic_fake(x, fp8_dtype, scale, scale_inv, seed):
    return torch.empty_like(x, dtype=fp8_dtype), scale_inv.clone()
