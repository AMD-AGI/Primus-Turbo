###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Custom ops of the FLUX dual-stream block's gated residual + second adaLN (``flux_adaln_dual``):
per stream ``x1 = x + g * a`` and ``y = LN(x1) * (1 + s) + h``, both streams in one launch.

Both passes are FlyDSL kernels (gfx950); a Triton kernel adds the backward's per-workgroup partial
sums of the shift, scale and gate gradients per sample.
"""

from typing import Optional, Tuple

import torch
import triton

from primus_turbo.flydsl.flux.adaln_dual_kernel import (
    BWD_ROWS,
    N_SUMS,
    adaln_dual_supported,
    flydsl_adaln_dual_bwd,
    flydsl_adaln_dual_fwd,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.triton.flux.colsum_kernel import flux_sample_colsum_kernel


def _require_gfx950():
    if not is_gfx950():
        raise RuntimeError("flux_adaln_dual: its FlyDSL kernels are built for gfx950")


def adaln_dual_shape_error(
    x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt
) -> Optional[str]:
    """Why ``flux_adaln_dual`` cannot take these operands, or None."""
    vectors = (g_img, s_img, h_img, g_txt, s_txt, h_txt)
    for name, a, x in (("a_img", a_img, x_img), ("a_txt", a_txt, x_txt)):
        if a.shape != x.shape or a.dtype != x.dtype or a.device != x.device:
            return f"{name} must match its stream's x, got {a.dtype} {tuple(a.shape)} for {x.dtype} {tuple(x.shape)}"
    if not adaln_dual_supported(x_img, x_txt, vectors):
        return (
            "needs BF16 CUDA [B, S, C] streams of equal length, a multiple of 32, with C a multiple of 1536, "
            "and [B, C] modulation vectors with unit column stride and one row stride; got x_img "
            f"{x_img.dtype} {tuple(x_img.shape)}, x_txt {x_txt.dtype} {tuple(x_txt.shape)}, vectors "
            + ", ".join(f"{tuple(v.shape)}/{v.stride()}" for v in vectors)
        )
    return None


@torch.library.custom_op("primus_turbo::flux_adaln_dual_fwd_impl", mutates_args=(), device_types="cuda")
def flux_adaln_dual_fwd_impl(
    x_img: torch.Tensor,
    a_img: torch.Tensor,
    g_img: torch.Tensor,
    s_img: torch.Tensor,
    h_img: torch.Tensor,
    x_txt: torch.Tensor,
    a_txt: torch.Tensor,
    g_txt: torch.Tensor,
    s_txt: torch.Tensor,
    h_txt: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per stream ``x1 = x + g[:, None] * a`` and ``layer_norm(x1) * (1 + s[:, None]) + h[:, None]``, as
    ``(x1_img, y_img, x1_txt, y_txt, FP32 [rows, 2] row mean and rstd)``; the statistics are for the
    backward."""
    _require_gfx950()
    return flydsl_adaln_dual_fwd(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps)


@flux_adaln_dual_fwd_impl.register_fake
def _flux_adaln_dual_fwd_impl_fake(
    x_img: torch.Tensor,
    a_img: torch.Tensor,
    g_img: torch.Tensor,
    s_img: torch.Tensor,
    h_img: torch.Tensor,
    x_txt: torch.Tensor,
    a_txt: torch.Tensor,
    g_txt: torch.Tensor,
    s_txt: torch.Tensor,
    h_txt: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_rows = x_img.shape[0] * (x_img.shape[1] + x_txt.shape[1])
    return (
        x_img.new_empty(x_img.shape),
        x_img.new_empty(x_img.shape),
        x_txt.new_empty(x_txt.shape),
        x_txt.new_empty(x_txt.shape),
        x_img.new_empty((n_rows, 2), dtype=torch.float32),
    )


@torch.library.custom_op("primus_turbo::flux_adaln_dual_bwd_impl", mutates_args=(), device_types="cuda")
def flux_adaln_dual_bwd_impl(
    gy_img: torch.Tensor,
    gx_img: torch.Tensor,
    x_img: torch.Tensor,
    a_img: torch.Tensor,
    g_img: torch.Tensor,
    s_img: torch.Tensor,
    gy_txt: torch.Tensor,
    gx_txt: torch.Tensor,
    x_txt: torch.Tensor,
    a_txt: torch.Tensor,
    g_txt: torch.Tensor,
    s_txt: torch.Tensor,
    stats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(gradients of x_img, a_img, x_txt, a_txt, and [2B, 3, C] shift, scale and gate gradients, image
    samples first)`` for the gradients ``gy`` of the LayerNorm outputs and ``gx`` of ``x1``."""
    _require_gfx950()
    dx_img, da_img, dx_txt, da_txt, part = flydsl_adaln_dual_bwd(
        gy_img, gx_img, x_img, a_img, g_img, s_img, gy_txt, gx_txt, x_txt, a_txt, g_txt, s_txt, stats
    )
    batch, n_img, c = x_img.shape
    # The partials of a sample's rows are consecutive within its stream, image samples first; both
    # streams have the same number of rows per sample.
    parts_per_sample = n_img // BWD_ROWS
    vec = torch.empty((2 * batch, N_SUMS, c), device=x_img.device, dtype=g_img.dtype)
    flux_sample_colsum_kernel[(2 * batch, triton.cdiv(N_SUMS * c, 256))](
        part, vec, parts_per_sample, N_SUMS * c, BLOCK_P=8, BLOCK_C=256, num_warps=4
    )
    return dx_img, da_img, dx_txt, da_txt, vec


@flux_adaln_dual_bwd_impl.register_fake
def _flux_adaln_dual_bwd_impl_fake(
    gy_img: torch.Tensor,
    gx_img: torch.Tensor,
    x_img: torch.Tensor,
    a_img: torch.Tensor,
    g_img: torch.Tensor,
    s_img: torch.Tensor,
    gy_txt: torch.Tensor,
    gx_txt: torch.Tensor,
    x_txt: torch.Tensor,
    a_txt: torch.Tensor,
    g_txt: torch.Tensor,
    s_txt: torch.Tensor,
    stats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        x_img.new_empty(x_img.shape),
        x_img.new_empty(x_img.shape),
        x_txt.new_empty(x_txt.shape),
        x_txt.new_empty(x_txt.shape),
        g_img.new_empty((2 * x_img.shape[0], N_SUMS, x_img.shape[2])),
    )
