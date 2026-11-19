###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Custom ops of the DiT adaLN modulation (``adaln_modulate``): the concat of two token segments (text
and image), LayerNorm without affine parameters, and the ``(1 + scale)``/``shift`` modulation.

Both passes are FlyDSL kernels (gfx950); a Triton kernel adds the backward's per-wave partial sums of
the shift and scale gradients per sample.
"""

from typing import Optional, Tuple

import torch
import triton

from primus_turbo.flydsl.normalization.adaln_modulate_kernel import (
    adaln_modulate_supported,
    flydsl_adaln_modulate_bwd,
    flydsl_adaln_modulate_fwd,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.triton.reduce.colsum_kernel import sample_colsum_kernel


def _require_gfx950():
    if not is_gfx950():
        raise RuntimeError("adaln_modulate: its FlyDSL kernels are built for gfx950")


def adaln_modulate_shape_error(x_txt, x_img, scale, shift) -> Optional[str]:
    """Why ``adaln_modulate`` cannot take these operands, or None.

    The limits come from the FlyDSL kernels' tiling and apply on every arch, so a call is accepted or
    rejected the same way everywhere: BF16 CUDA ``[B, S_*, C]`` segments with the same B and C, each
    sequence length a multiple of 128 (a backward block's 128 rows lie in one segment), C a multiple of
    512 (64 lanes x 8 values per vector), and ``[B, C]`` scale and shift with unit column stride and one
    row stride (chunks of one modulation GEMM output). A caller with other shapes keeps its unfused path.
    """
    if x_txt.dim() != 3 or x_img.dim() != 3 or scale.dim() != 2 or shift.dim() != 2:
        return "expected x_txt [B, S_txt, C], x_img [B, S_img, C], scale [B, C] and shift [B, C]"
    batch, c = x_txt.shape[0], x_txt.shape[2]
    if tuple(scale.shape) != (batch, c) or tuple(shift.shape) != (batch, c):
        return f"scale and shift must be [B, C] = [{batch}, {c}], got {tuple(scale.shape)} and {tuple(shift.shape)}"
    if not adaln_modulate_supported(x_txt, x_img, scale, shift):
        return (
            "needs BF16 CUDA tensors with matching batch and channels, channels a multiple of 512, both "
            "sequence lengths multiples of 128, and scale and shift with unit column stride and one row "
            f"stride; got x_txt {x_txt.dtype} {tuple(x_txt.shape)}, x_img {x_img.dtype} {tuple(x_img.shape)}, "
            f"scale strides {scale.stride()}, shift strides {shift.stride()}"
        )
    return None


@torch.library.custom_op("primus_turbo::adaln_modulate_fwd_impl", mutates_args=(), device_types="cuda")
def adaln_modulate_fwd_impl(
    x_txt: torch.Tensor, x_img: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(copy of x_txt, copy of x_img, layer_norm(cat([x_txt, x_img], 1)) * (1 + scale[:, None]) +
    shift[:, None], FP32 [B, S, 2] row mean and rstd)``; the statistics are for the backward."""
    _require_gfx950()
    return flydsl_adaln_modulate_fwd(x_txt, x_img, scale, shift, eps)


@adaln_modulate_fwd_impl.register_fake
def _adaln_modulate_fwd_impl_fake(
    x_txt: torch.Tensor, x_img: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (x_txt.shape[0], x_txt.shape[1] + x_img.shape[1], x_txt.shape[2])
    return (
        x_txt.new_empty(x_txt.shape),
        x_img.new_empty(x_img.shape),
        x_txt.new_empty(shape),
        x_txt.new_empty(shape[:2] + (2,), dtype=torch.float32),
    )


@torch.library.custom_op("primus_turbo::adaln_modulate_bwd_impl", mutates_args=(), device_types="cuda")
def adaln_modulate_bwd_impl(
    g_y: torch.Tensor,
    g_r_txt: torch.Tensor,
    g_r_img: torch.Tensor,
    x_txt: torch.Tensor,
    x_img: torch.Tensor,
    scale: torch.Tensor,
    stats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(gradient of x_txt, gradient of x_img, [B, 2, C] gradients of shift and scale)`` for the
    gradients ``g_y`` of the output and ``g_r_*`` of the copies."""
    _require_gfx950()
    dx_txt, dx_img, part = flydsl_adaln_modulate_bwd(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats)
    batch, parts_per_sample, n_cols = part.shape
    dss = torch.empty((batch, 2, x_txt.shape[2]), device=x_txt.device, dtype=scale.dtype)
    sample_colsum_kernel[(batch, triton.cdiv(n_cols, 256))](
        part, dss, parts_per_sample, n_cols, BLOCK_P=16, BLOCK_C=256, num_warps=4
    )
    return dx_txt, dx_img, dss


@adaln_modulate_bwd_impl.register_fake
def _adaln_modulate_bwd_impl_fake(
    g_y: torch.Tensor,
    g_r_txt: torch.Tensor,
    g_r_img: torch.Tensor,
    x_txt: torch.Tensor,
    x_img: torch.Tensor,
    scale: torch.Tensor,
    stats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        x_txt.new_empty(x_txt.shape),
        x_img.new_empty(x_img.shape),
        scale.new_empty((x_txt.shape[0], 2, x_txt.shape[2])),
    )
