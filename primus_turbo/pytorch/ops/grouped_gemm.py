###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grouped GEMM op with autograd support.

Provides a ``grouped_gemm`` functional API analogous to
TransformerEngine's ``_GroupedLinear`` autograd function, but adapted
for Primus-Turbo's contiguous weight layout ``[G, out_features, in_features]``
and simplified FP8 path.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_impl
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_impl,
)

__all__ = ["grouped_gemm"]


def _ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


class _GroupedLinear(torch.autograd.Function):
    """Grouped linear forward / backward (autograd function).

    Aligns with TransformerEngine's ``_GroupedLinear`` structure while using
    Primus-Turbo's kernel dispatch layer (CK / hipBLASLt / Triton).

    Weight layout
    -------------
    Primus-Turbo stores grouped weights as a **single contiguous** tensor of
    shape ``[G, out_features, in_features]`` (when ``trans_b=True``).  TE, by
    contrast, keeps one ``[out_features, in_features]`` parameter per GEMM.

    FP8
    ---
    Activation-side FP8 quantisation is delegated to
    ``primus_turbo.pytorch.ops.grouped_gemm_fp8``.
    # TODO: support FP8 weight quantisation (fp8 weight)
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_b: bool,
        num_cu: Optional[int],
    ) -> torch.Tensor:
        in_features = weight.size(-1) if trans_b else weight.size(-2)
        if inp.size(-1) != in_features:
            raise ValueError(
                f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
                f"weight tensor (shape={tuple(weight.size())})"
            )
        inp_2d = inp.reshape(-1, in_features)
        activation_dtype = inp_2d.dtype

        if len(group_lens) == 1:
            assert weight.size(0) == 1, f"Expected first dim to be 1, got {weight.size(0)}"
            w_2d = weight.squeeze(0)
            out = gemm_impl(
                inp_2d,
                False,
                w_2d,
                trans_b,
                activation_dtype,
                False,
                default_backend=BackendType.HIPBLASLT.value,
            )
        else:
            out = grouped_gemm_impl(
                inp_2d,
                weight,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=trans_b,
                num_cu=num_cu,
                default_backend=BackendType.CK.value,
                maybe_pre_sync=True,
            )

        if bias is not None:
            out = out + bias

        ctx.save_for_backward(inp_2d, weight, group_lens, group_offs)
        ctx.trans_b = trans_b
        ctx.num_cu = num_cu
        ctx.use_bias = bias is not None
        ctx.inp_shape = inp.shape
        ctx.activation_dtype = activation_dtype

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = _ensure_contiguous(grad_output)

        inp, weight, group_lens, group_offs = ctx.saved_tensors
        requires_dgrad = ctx.needs_input_grad[0]
        requires_wgrad = ctx.needs_input_grad[1]

        grad_input = None
        grad_weight = None
        grad_bias = None

        if len(group_lens) == 1:
            assert weight.size(0) == 1
            w_2d = weight.squeeze(0)

            if requires_dgrad:
                grad_input = gemm_impl(
                    grad_output,
                    False,
                    w_2d,
                    not ctx.trans_b,
                    ctx.activation_dtype,
                    False,
                    default_backend=BackendType.HIPBLASLT.value,
                )

            if requires_wgrad:
                grad_weight = gemm_impl(
                    inp,
                    True,
                    grad_output,
                    False,
                    weight.dtype,
                    ctx.trans_b,
                    default_backend=BackendType.HIPBLASLT.value,
                ).view(weight.size())
        else:
            if requires_dgrad:
                grad_input = grouped_gemm_impl(
                    grad_output,
                    weight,
                    group_lens,
                    group_offs,
                    trans_a=False,
                    trans_b=not ctx.trans_b,
                    num_cu=ctx.num_cu,
                    default_backend=BackendType.CK.value,
                )

            if requires_wgrad:
                grad_weight = grouped_gemm_variable_k_impl(
                    inp,
                    grad_output,
                    group_lens,
                    group_offs,
                    trans_a=True,
                    trans_b=False,
                    trans_c=ctx.trans_b,
                    num_cu=ctx.num_cu,
                    default_backend=BackendType.CK.value,
                )

        if ctx.use_bias:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None, None


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: Optional[torch.Tensor] = None,
    trans_b: bool = False,
    bias: Optional[torch.Tensor] = None,
    num_cu: Optional[int] = None,
) -> torch.Tensor:
    """Grouped GEMM with optional bias and autograd support.

    Computes ``out[g] = a[g] @ b[g]`` (with optional transpose / bias) for
    each group *g*, where the per-group rows of ``a`` are concatenated along
    dim-0 and ``b`` is a contiguous ``[G, ...]`` tensor.

    This mirrors TransformerEngine's ``_GroupedLinear`` autograd function but
    keeps Primus-Turbo's contiguous-weight convention.

    Args:
        a: Input activations, shape ``[sum(group_lens), K]``.
        b: Weight tensor, shape ``[G, K, N]`` (or ``[G, N, K]`` when
           ``trans_b=True``).
        group_lens: Number of rows per group, shape ``[G]``, dtype int64.
        group_offs: Exclusive prefix-sum of *group_lens*, shape ``[G+1]``.
            Computed automatically when ``None``.
        trans_b: If ``True``, treat each ``b[g]`` as transposed.
        bias: Optional bias of shape ``[N]``.
        num_cu: Limit the number of compute units.  ``None`` = default.

    Returns:
        Output tensor of shape ``[sum(group_lens), N]``.
    """
    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)

    return _GroupedLinear.apply(a, b, bias, group_lens, group_offs, trans_b, num_cu)
