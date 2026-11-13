###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused BF16 ops for the FLUX.1 transformer blocks (diffusers' ``FluxTransformerBlock`` and
``FluxSingleTransformerBlock``), each computing what the unfused block computes, with the same
roundings where the docstring says so.

Public API:
    - ``flux_gelu_linear(x, weight, bias, prefix=None)``
    - ``flux_adaln_single(x_txt, x_img, scale, shift, eps)``
    - ``flux_adaln_dual(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps)``

The ``*_shape_error`` functions of ``primus_turbo.pytorch.kernels.flux`` say whether an op takes a
call and can be evaluated under ``torch.compile``; an op takes the same calls on every arch. The
FlyDSL kernels run on gfx950 (``primus_turbo.pytorch.core.utils.is_gfx950``); elsewhere the GELU is
unfused and the adaLN ops compute the same arithmetic in eager PyTorch.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.flux.adaln_dual_impl import (
    adaln_dual_shape_error,
    flux_adaln_dual_bwd_impl,
    flux_adaln_dual_fwd_impl,
)
from primus_turbo.pytorch.kernels.flux.adaln_impl import (
    adaln_single_shape_error,
    flux_adaln_single_bwd_impl,
    flux_adaln_single_fwd_impl,
)
from primus_turbo.pytorch.kernels.flux.gelu_linear_impl import (
    flux_gelu_bwd_impl,
    flux_gelu_linear_fwd_impl,
    gelu_linear_shape_error,
)

__all__ = [
    "flux_adaln_dual",
    "flux_adaln_single",
    "flux_gelu_linear",
]


class _FluxGeluLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, prefix):
        out, aux = flux_gelu_linear_fwd_impl(x, weight, bias, prefix)
        ctx.save_for_backward(x, weight, aux)
        ctx.width = 0 if prefix is None else prefix.shape[1]
        ctx.has_prefix = prefix is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, aux = ctx.saved_tensors
        grad_h, grad_bias = flux_gelu_bwd_impl(grad_out[:, ctx.width :], aux)
        grad_prefix = grad_out[:, : ctx.width] if ctx.has_prefix else None
        return grad_h @ weight, grad_h.t() @ x, grad_bias, grad_prefix


def flux_gelu_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, prefix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``cat([prefix, gelu_tanh(x @ weight.T + bias)], dim=1)`` for BF16 ``x [M, K]``, ``weight [N, K]``,
    ``bias [N]`` and an optional ``prefix [M, P]``: FLUX's MLP up-projection and GELU, with the
    single-stream block's concat of the attention output in front.

    The GELU is evaluated as Inductor's ``gelu(approximate="tanh")``, operation for operation; the
    backward computes Inductor's ``gelu_backward`` and the bias gradient in one pass.
    """
    why = gelu_linear_shape_error(x, weight, bias, prefix)
    if why is not None:
        raise ValueError(f"flux_gelu_linear: unsupported input ({why})")
    return _FluxGeluLinearFunction.apply(x, weight, bias, prefix)


class _FluxAdaLNSingleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_txt, x_img, scale, shift, eps):
        r_txt, r_img, y, stats = flux_adaln_single_fwd_impl(x_txt, x_img, scale, shift, eps)
        ctx.save_for_backward(x_txt, x_img, scale, stats)
        return r_txt, r_img, y

    @staticmethod
    def backward(ctx, g_r_txt, g_r_img, g_y):
        x_txt, x_img, scale, stats = ctx.saved_tensors
        dx_txt, dx_img, dss = flux_adaln_single_bwd_impl(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats)
        return dx_txt, dx_img, dss[:, 1], dss[:, 0], None


def _adaln_single_eager(x_txt, x_img, scale, shift, eps):
    x = torch.cat([x_txt, x_img], dim=1)
    one_plus = (1 + scale.float()).to(x.dtype).float()
    y = F.layer_norm(x.float(), (x.shape[-1],), eps=eps) * one_plus[:, None] + shift.float()[:, None]
    return x_txt.clone(), x_img.clone(), y.to(x.dtype)


def flux_adaln_single(
    x_txt: torch.Tensor, x_img: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The FLUX single-stream block's adaLN: ``(x_txt, x_img, LN(cat([x_txt, x_img], 1)) * (1 +
    scale[:, None]) + shift[:, None])`` for BF16 ``[B, S_*, C]`` streams and ``[B, C]`` modulation
    chunks, LayerNorm without affine parameters. The first two outputs are copies of the inputs for
    the block's residual, so its ``torch.cat`` of the two streams needs no pass of its own.

    Inductor's arithmetic (``1 + scale`` rounded to BF16, FP32 inside, one rounding per output);
    only the order of the row and column sums differs. FlyDSL kernels on gfx950, eager PyTorch
    elsewhere.
    """
    why = adaln_single_shape_error(x_txt, x_img, scale, shift)
    if why is not None:
        raise ValueError(f"flux_adaln_single: unsupported input ({why})")
    if not is_gfx950():
        return _adaln_single_eager(x_txt, x_img, scale, shift, float(eps))
    return _FluxAdaLNSingleFunction.apply(x_txt, x_img, scale, shift, float(eps))


class _FluxAdaLNDualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps):
        x1_img, y_img, x1_txt, y_txt, stats = flux_adaln_dual_fwd_impl(
            x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps
        )
        ctx.save_for_backward(x_img, a_img, g_img, s_img, x_txt, a_txt, g_txt, s_txt, stats)
        return x1_img, y_img, x1_txt, y_txt

    @staticmethod
    def backward(ctx, gx_img, gy_img, gx_txt, gy_txt):
        x_img, a_img, g_img, s_img, x_txt, a_txt, g_txt, s_txt, stats = ctx.saved_tensors
        dx_img, da_img, dx_txt, da_txt, vec = flux_adaln_dual_bwd_impl(
            gy_img, gx_img, x_img, a_img, g_img, s_img, gy_txt, gx_txt, x_txt, a_txt, g_txt, s_txt, stats
        )
        batch = x_img.shape[0]
        vi, vt = vec[:batch], vec[batch:]
        return (
            dx_img,
            da_img,
            vi[:, 2],
            vi[:, 1],
            vi[:, 0],
            dx_txt,
            da_txt,
            vt[:, 2],
            vt[:, 1],
            vt[:, 0],
            None,
        )


def _adaln_dual_eager(x, a, g, s, h, eps):
    x1 = x.float() + g.float()[:, None] * a.float()
    y = F.layer_norm(x1, (x1.shape[-1],), eps=eps) * (1 + s.float()[:, None]) + h.float()[:, None]
    return x1.to(x.dtype), y.to(x.dtype)


def flux_adaln_dual(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The FLUX dual-stream block's gated attention residual and second adaLN, for both streams:
    ``x1 = x + g[:, None] * a`` and ``y = LN(x1) * (1 + s[:, None]) + h[:, None]``, as
    ``(x1_img, y_img, x1_txt, y_txt)``, for BF16 ``[B, S, C]`` streams of equal length, the attention
    outputs ``a`` and ``[B, C]`` gate, scale and shift chunks; LayerNorm without affine parameters.

    Inductor's arithmetic (``x1`` and ``1 + s`` in FP32, one rounding per output), with the row
    statistics combined exactly instead of by Welford's update; ``x1`` is returned in BF16. FlyDSL
    kernels on gfx950, eager PyTorch elsewhere.
    """
    why = adaln_dual_shape_error(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt)
    if why is not None:
        raise ValueError(f"flux_adaln_dual: unsupported input ({why})")
    if not is_gfx950():
        return _adaln_dual_eager(x_img, a_img, g_img, s_img, h_img, float(eps)) + _adaln_dual_eager(
            x_txt, a_txt, g_txt, s_txt, h_txt, float(eps)
        )
    return _FluxAdaLNDualFunction.apply(
        x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, float(eps)
    )
