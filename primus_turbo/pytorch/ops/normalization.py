###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Normalization ops: Triton-backed RMSNorm (standard + fused residual variant) and the DiT adaLN
modulation (FlyDSL on gfx950, eager PyTorch elsewhere).

Public API:
    - ``rmsnorm(x, gamma, eps=1e-6, zero_centered=False) -> y``
    - ``rmsnorm_residual(x, residual, gamma, eps=1e-6) -> (y, x_plus_r)``
    - ``rmsnorm_residual_fp4(...) -> (y, x_plus_r, QuantizedTensorPair)``
    - ``adaln_modulate(x_txt, x_img, scale, shift, eps) -> (r_txt, r_img, y)``
    - ``gated_residual_adaln(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps)
      -> (x1_img, y_img, x1_txt, y_txt)``
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.normalization.adaln_modulate_impl import (
    adaln_modulate_bwd_impl,
    adaln_modulate_fwd_impl,
    adaln_modulate_shape_error,
)
from primus_turbo.pytorch.kernels.normalization.gated_residual_adaln_impl import (
    gated_residual_adaln_bwd_impl,
    gated_residual_adaln_fwd_impl,
    gated_residual_adaln_shape_error,
)
from primus_turbo.pytorch.kernels.normalization.rmsnorm_impl import (
    rmsnorm_bwd_impl,
    rmsnorm_bwd_residual_impl,
    rmsnorm_fwd_impl,
    rmsnorm_fwd_residual_impl,
)

__all__ = [
    "rmsnorm",
    "rmsnorm_residual",
    "rmsnorm_residual_fp4",
    "adaln_modulate",
    "gated_residual_adaln",
]


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False):
        assert x.is_cuda and gamma.is_cuda, "rmsnorm: x and gamma must be CUDA tensors"
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H, (
            f"rmsnorm: last dim of x ({orig_shape[-1]}) must equal gamma.shape[0] ({H})"
        )

        y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_impl(x, gamma, eps, zero_centered)

        ctx.save_for_backward(x2, gamma, rstd)
        ctx.eps = eps
        ctx.zero_centered = zero_centered
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2, gamma, rstd = ctx.saved_tensors
        dx, dg = rmsnorm_bwd_impl(
            grad_out,
            x2,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
            ctx.zero_centered,
        )
        return dx.reshape(ctx.orig_shape), dg, None, None


class _RMSNormResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
        assert x.is_cuda and residual.is_cuda and gamma.is_cuda, (
            "rmsnorm_residual: x, residual and gamma must be CUDA tensors"
        )
        assert x.shape == residual.shape, (
            f"rmsnorm_residual: shape mismatch {tuple(x.shape)} vs {tuple(residual.shape)}"
        )
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H

        y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_residual_impl(
            x, residual, gamma, eps
        )

        ctx.save_for_backward(x_plus_r, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape), x_plus_r.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_xpr: torch.Tensor):
        x_plus_r, gamma, rstd = ctx.saved_tensors
        dx, dg = rmsnorm_bwd_residual_impl(
            grad_y,
            grad_xpr,
            x_plus_r,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
        )
        dx_out = dx.reshape(ctx.orig_shape)
        # Jacobian of add() is [I, I] -> both x and residual get the same grad.
        return dx_out, dx_out, dg, None


class _RMSNormResidualWithRstdFunction(torch.autograd.Function):
    """Same as residual RMSNorm but also returns Triton's ``rstd`` (f32, non-diff).

    R3 reads ``rstd`` for the FlyDSL dual under ``no_grad``. Returning Float4 from
    this Function hits unimplemented ``fill_cuda``; f32 ``rstd`` is safe.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        skip_y_store: bool,
    ):
        assert x.is_cuda and residual.is_cuda and gamma.is_cuda
        assert x.shape == residual.shape
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H

        skip = bool(skip_y_store)
        y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_residual_impl(
            x, residual, gamma, eps, skip_y_store=skip
        )
        ctx.save_for_backward(x_plus_r, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        y_out = y.reshape(orig_shape)
        xpr_out = x_plus_r.reshape(orig_shape)
        # y stays a differentiable STE for a consumer of the quantized pair, so it must
        # not be marked non-differentiable even when its store is skipped.
        ctx.mark_non_differentiable(rstd)
        return y_out, xpr_out, rstd

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_xpr: torch.Tensor, grad_rstd: torch.Tensor):
        del grad_rstd
        x_plus_r, gamma, rstd = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros(ctx.orig_shape, device=x_plus_r.device, dtype=x_plus_r.dtype)
        dx, dg = rmsnorm_bwd_residual_impl(
            grad_y,
            grad_xpr,
            x_plus_r,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
        )
        dx_out = dx.reshape(ctx.orig_shape)
        return dx_out, dx_out, dg, None, None


def rmsnorm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False
) -> torch.Tensor:
    """RMSNorm.

    Args:
        x: input tensor; normalization is over the last dim.
        gamma: learnable gain of shape ``[x.shape[-1]]``.
        eps: variance epsilon.
        zero_centered: if True, the effective gain is ``(1 + gamma)`` (computed in
            fp32). Initialize ``gamma`` to zeros in this mode.
    """
    return _RMSNormFunction.apply(x, gamma, eps, zero_centered)


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _RMSNormResidualFunction.apply(x, residual, gamma, eps)


class _RMSNormFP4STE(torch.autograd.Function):
    """Quantize ``y`` without reading it, and keep it on the gradient path.

    The pair is computed from ``xpr``/``rstd``/``gamma`` in f32 registers, so the
    values never depend on ``y`` -- that is the whole point, and it is what lets the
    caller skip the bf16 ``y`` store. But a consumer of the pair still has to send its
    dgrad back to the norm, and a ``QuantizedTensor`` built outside the graph is a leaf.
    Taking ``y`` as the input and handing its gradient straight through makes the pair a
    graph node whose backward lands on ``y``, i.e. a straight-through estimator.

    The Float4 outputs are safe here because they are consumed: autograd never has to
    synthesise a gradient for them (which is where ``fill_cuda`` is unimplemented).
    """

    @staticmethod
    def forward(ctx, y, xpr2, rstd, gamma, config, row_recipe, col_recipe, dual_quant):
        with torch.no_grad():
            gamma_f32 = gamma.detach().to(dtype=torch.float32).contiguous()
            row, row_scale, col, col_scale = dual_quant(
                xpr2,
                rstd.reshape(-1).contiguous(),
                gamma_f32,
                float4_e2m1fn_x2,
                col_rht=col_recipe.use_rht,
                scale_rounding_mode=config.scale_rounding_mode,
            )

        def _wrap(data, scale_inv, shape, recipe, axis):
            return QuantizedTensor(
                data,
                scale_inv,
                shape=shape,
                orig_dtype=y.dtype,
                dest_dtype=float4_e2m1fn_x2,
                granularity=config.granularity,
                block_size=config.block_size,
                scaling_recipe=recipe,
                scale_rounding_mode=config.scale_rounding_mode,
                quantized_axis=axis,
                requires_grad=y.requires_grad,
            )

        M, K = xpr2.shape
        return (
            _wrap(row, row_scale, torch.Size((M, K)), row_recipe, -1),
            _wrap(col, col_scale, torch.Size((K, M)), col_recipe, -2),
        )

    @staticmethod
    def backward(ctx, grad_row, grad_col):
        # The consumer's dgrad arrives on the row-wise operand; the col-wise one is a
        # wgrad operand, which produces no gradient of its own input.
        del grad_col
        return grad_row, None, None, None, None, None, None, None


def rmsnorm_residual_fp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
    config: Union[Float4QuantConfig, None] = None,
    skip_y_store: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, QuantizedTensorPair]:
    """Residual RMSNorm whose ``y`` comes back already quantized, without a second pass.

    Returns ``(y, x_plus_r, y_fp4)``. ``y_fp4.data`` is the row-wise operand and
    ``y_fp4.data_t`` the col-wise (RHT) one, so it feeds :func:`gemm_fp4` or
    :func:`mlp_fp4` as ``a`` / ``x`` with no further quantization.

    ``y`` and ``x_plus_r`` both carry autograd; ``y`` stays the STE so a consumer
    of ``y_fp4`` still sends its dgrad back through RMSNorm. Triton's ``rstd`` is
    a non-differentiable f32, which is what lets the dual run under ``no_grad``
    (a Float4 tensor must not enter ``run_backward``: ``fill_cuda`` is
    unimplemented).

    ``skip_y_store``: skip the bf16 ``y`` HBM store. Legal only when every consumer
    reads ``y_fp4``; a consumer that quantizes bf16 ``y`` itself (the final LN
    feeding the LM head) must leave this False.

    The scales come out of the norm's own registers rather than from a reload of
    ``y``, so this is not bit-identical to quantizing a materialized bf16 ``y``:
    ``rstd * gamma`` is applied in f32 and the bf16 rounding never happens. It is
    the more accurate of the two.
    """
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import flydsl_rmsnorm_dual_quant

    if config is None:
        config = Float4QuantConfig()
    supported, reason = check_mxfp4_support()
    assert supported, reason
    assert config.granularity == ScalingGranularity.MX_BLOCKWISE, (
        f"rmsnorm_residual_fp4 currently covers MX_BLOCKWISE only, got {config.granularity}"
    )
    assert not config.use_preshuffle, "the fused dual emits raw (non-preshuffled) scales"
    assert x.dtype == torch.bfloat16, f"the fused dual is bf16-only, got {x.dtype}"

    row_recipe = ScalingRecipe()
    col_recipe = ScalingRecipe(use_rht=True)

    y, x_plus_r, rstd = _RMSNormResidualWithRstdFunction.apply(x, residual, gamma, eps, skip_y_store)
    H = gamma.shape[0]
    xpr2 = x_plus_r.reshape(-1, H).contiguous()
    row_qt, col_qt = _RMSNormFP4STE.apply(
        y, xpr2, rstd, gamma, config, row_recipe, col_recipe, flydsl_rmsnorm_dual_quant
    )
    return y, x_plus_r, QuantizedTensorPair(row_qt, col_qt)


class _AdaLNModulateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_txt, x_img, scale, shift, eps):
        r_txt, r_img, y, stats = adaln_modulate_fwd_impl(x_txt, x_img, scale, shift, eps)
        ctx.save_for_backward(x_txt, x_img, scale, stats)
        return r_txt, r_img, y

    @staticmethod
    def backward(ctx, g_r_txt, g_r_img, g_y):
        x_txt, x_img, scale, stats = ctx.saved_tensors
        dx_txt, dx_img, dss = adaln_modulate_bwd_impl(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats)
        return dx_txt, dx_img, dss[:, 1], dss[:, 0], None


def _adaln_modulate_eager(x_txt, x_img, scale, shift, eps):
    x = torch.cat([x_txt, x_img], dim=1)
    one_plus = (1 + scale.float()).to(x.dtype).float()
    y = F.layer_norm(x.float(), (x.shape[-1],), eps=eps) * one_plus[:, None] + shift.float()[:, None]
    return x_txt.clone(), x_img.clone(), y.to(x.dtype)


def adaln_modulate(
    x_txt: torch.Tensor, x_img: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DiT adaLN modulation of two concatenated token segments: ``(x_txt, x_img, LN(cat([x_txt, x_img],
    1)) * (1 + scale[:, None]) + shift[:, None])`` for BF16 ``[B, S_*, C]`` segments and ``[B, C]``
    modulation chunks, LayerNorm without affine parameters. The first two outputs are copies of the
    inputs for the block's residual, so the caller's ``torch.cat`` of the two segments needs no pass of
    its own. FLUX's single-stream block computes this on its text and image tokens.

    Inductor's arithmetic (``1 + scale`` rounded to BF16, FP32 inside, one rounding per output);
    only the order of the row and column sums differs. FlyDSL kernels on gfx950, eager PyTorch
    elsewhere.

    Takes BF16 CUDA tensors with C a multiple of 512, both sequence lengths multiples of 128, and scale
    and shift with unit column stride and one row stride, on every arch; anything else raises
    ``ValueError``. ``adaln_modulate_shape_error`` in
    ``primus_turbo.pytorch.kernels.normalization.adaln_modulate_impl`` lets a caller check first.
    """
    why = adaln_modulate_shape_error(x_txt, x_img, scale, shift)
    if why is not None:
        raise ValueError(f"adaln_modulate: unsupported input ({why})")
    if not is_gfx950():
        return _adaln_modulate_eager(x_txt, x_img, scale, shift, float(eps))
    return _AdaLNModulateFunction.apply(x_txt, x_img, scale, shift, float(eps))


class _GatedResidualAdaLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps):
        x1_img, y_img, x1_txt, y_txt, stats = gated_residual_adaln_fwd_impl(
            x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps
        )
        ctx.save_for_backward(x_img, a_img, g_img, s_img, x_txt, a_txt, g_txt, s_txt, stats)
        return x1_img, y_img, x1_txt, y_txt

    @staticmethod
    def backward(ctx, gx_img, gy_img, gx_txt, gy_txt):
        x_img, a_img, g_img, s_img, x_txt, a_txt, g_txt, s_txt, stats = ctx.saved_tensors
        dx_img, da_img, dx_txt, da_txt, vec = gated_residual_adaln_bwd_impl(
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


def _gated_residual_adaln_eager(x, a, g, s, h, eps):
    x1 = x.float() + g.float()[:, None] * a.float()
    y = F.layer_norm(x1, (x1.shape[-1],), eps=eps) * (1 + s.float()[:, None]) + h.float()[:, None]
    return x1.to(x.dtype), y.to(x.dtype)


def gated_residual_adaln(
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
    """DiT gated residual followed by adaLN modulation, for the two streams of an MMDiT block in one
    call: ``x1 = x + g[:, None] * a`` and ``y = LN(x1) * (1 + s[:, None]) + h[:, None]``, as
    ``(x1_img, y_img, x1_txt, y_txt)``, for BF16 ``[B, S_*, C]`` streams, the attention outputs ``a``
    and ``[B, C]`` gate, scale and shift chunks; LayerNorm without affine parameters. FLUX's dual-stream
    block computes this between its attention and its MLPs.

    Inductor's arithmetic (``x1`` and ``1 + s`` in FP32, one rounding per output), with the row
    statistics combined exactly instead of by Welford's update; ``x1`` is returned in BF16. FlyDSL
    kernels on gfx950, eager PyTorch elsewhere.

    Takes BF16 CUDA tensors with both sequence lengths multiples of 32 (they may differ), C a multiple
    of 1536, and modulation vectors with unit column stride and one row stride, on every arch; anything
    else raises ``ValueError``. ``gated_residual_adaln_shape_error`` in
    ``primus_turbo.pytorch.kernels.normalization.gated_residual_adaln_impl`` lets a caller check first.
    """
    why = gated_residual_adaln_shape_error(
        x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt
    )
    if why is not None:
        raise ValueError(f"gated_residual_adaln: unsupported input ({why})")
    if not is_gfx950():
        return _gated_residual_adaln_eager(
            x_img, a_img, g_img, s_img, h_img, float(eps)
        ) + _gated_residual_adaln_eager(x_txt, a_txt, g_txt, s_txt, h_txt, float(eps))
    return _GatedResidualAdaLNFunction.apply(
        x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, float(eps)
    )
