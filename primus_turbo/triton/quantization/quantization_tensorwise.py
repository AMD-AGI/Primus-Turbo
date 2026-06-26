###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Standalone Triton implementation of tensorwise FP8 quantize."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Matches compute_scale_from_amax's default eps (quant_utils.cuh / quantization.h).
_AMAX_EPS = 1e-12

# Number of partial-max slots = number of reduction programs (power of two so the
# quant kernel can reduce them with a single vectorized load).
_NUM_PARTIALS = 1024

# Persistent-kernel occupancy: resident programs per CU for the quant pass.
_PROGRAMS_PER_CU = 8


def _elementwise_configs():
    """Autotune space: per-program tile (drives vectorization width) x warps x stages."""
    configs = []
    for block in (2048, 4096, 8192, 16384):
        for num_warps in (4, 8, 16):
            for num_stages in (1, 2):
                configs.append(triton.Config({"BLOCK": block}, num_warps=num_warps, num_stages=num_stages))
    return configs


@triton.autotune(configs=_elementwise_configs(), key=["n"])
@triton.jit
def compute_amax_partials_kernel(x_ptr, partials_ptr, n, BLOCK: tl.constexpr):
    """Each program grid-strides over the tensor and stores one partial abs-max.

    No atomics: ``partials_ptr`` (zero-initialized, |x| >= 0) gets one plain store
    per program, which is then reduced inside the quant kernel."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_blocks = tl.cdiv(n, BLOCK)

    acc = tl.zeros([BLOCK], tl.float32)
    for blk in range(pid, num_blocks, nprog):
        offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))

    tl.store(partials_ptr + pid, tl.max(acc))


@triton.autotune(configs=_elementwise_configs(), key=["n"])
@triton.jit
def quantize_tensorwise_persistent_kernel(
    x_ptr,
    in_ptr,
    scale_inv_ptr,
    y_ptr,
    n,
    IS_AMAX: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Persistent quantize: derive the global scale inline, then grid-stride cast.

    ``in_ptr`` holds either the ``partials`` buffer (``IS_AMAX``: reduce it to the
    global amax then ``scale = FP8_MAX / max(amax, eps)``) or a single caller-provided
    ``scale``. Program 0 writes the scalar ``scale_inv``. IEEE-correct division matches
    torch / the HIP op (the default AMD reciprocal is only ~1 ULP accurate).
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    if IS_AMAX:
        parts = tl.load(in_ptr + tl.arange(0, NUM_PARTIALS))
        amax = tl.maximum(tl.max(parts), EPS)
        scale = tl.fdiv(FP8_MAX, amax, ieee_rounding=True)
    else:
        scale = tl.load(in_ptr).to(tl.float32)
    if pid == 0:
        tl.store(scale_inv_ptr, tl.fdiv(1.0, scale, ieee_rounding=True))

    num_blocks = tl.cdiv(n, BLOCK)
    for blk in range(pid, num_blocks, nprog):
        offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.clamp(x * scale, FP8_MIN, FP8_MAX)
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def quantize_fp8_tensorwise_triton(
    x: torch.Tensor,
    dest_dtype: torch.dtype,
    scale_opt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

    n = x.numel()
    fp8_max = float(torch.finfo(dest_dtype).max)
    fp8_min = float(torch.finfo(dest_dtype).min)

    out = torch.empty_like(x, dtype=dest_dtype)
    scale_inv = torch.empty((), dtype=torch.float32, device=x.device)

    if n == 0:
        if scale_opt is not None:
            scale_inv = (1.0 / scale_opt.to(torch.float32)).reshape(())
        return out, scale_inv

    # Lazy import: this module is imported during primus_turbo.pytorch package init
    # (via quantization_impl), so a top-level import of core.utils would be circular.
    from primus_turbo.pytorch.core.utils import get_num_cus

    cus = max(1, get_num_cus())
    # Persistent quant grid (each program grid-strides many tiles).
    quant_grid = lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), cus * _PROGRAMS_PER_CU),)

    if scale_opt is not None:
        assert scale_opt.numel() == 1, "tensorwise scale must be a scalar tensor"
        scale_in = scale_opt.to(torch.float32).reshape(1)
        quantize_tensorwise_persistent_kernel[quant_grid](
            x,
            scale_in,
            scale_inv,
            out,
            n,
            IS_AMAX=False,
            NUM_PARTIALS=_NUM_PARTIALS,
            FP8_MIN=fp8_min,
            FP8_MAX=fp8_max,
            EPS=_AMAX_EPS,
        )
    else:
        # Pass 1: contention-free partial abs-max. Pass 2: persistent reduce + quant.
        partials = torch.zeros(_NUM_PARTIALS, dtype=torch.float32, device=x.device)
        reduce_grid = lambda meta: (min(triton.cdiv(n, meta["BLOCK"]), _NUM_PARTIALS),)
        compute_amax_partials_kernel[reduce_grid](x, partials, n)
        quantize_tensorwise_persistent_kernel[quant_grid](
            x,
            partials,
            scale_inv,
            out,
            n,
            IS_AMAX=True,
            NUM_PARTIALS=_NUM_PARTIALS,
            FP8_MIN=fp8_min,
            FP8_MAX=fp8_max,
            EPS=_AMAX_EPS,
        )

    return out, scale_inv
