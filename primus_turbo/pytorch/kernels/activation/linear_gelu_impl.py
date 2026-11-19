###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Custom ops of the linear + bias + tanh GELU (``linear_gelu``), optionally written after a prefix.

The forward runs the GEMM as ``torch.addmm`` and writes ``cat([prefix, gelu_tanh(h)])`` with one
FlyDSL pass on gfx950 (unfused elsewhere and for shapes the kernel does not tile). The backward's
GELU gradient is one Triton kernel that also writes per-program column sums of it, reduced to the
bias gradient by a second one; it evaluates Inductor's ``gelu_backward`` expression in the same
order and sums the BF16-rounded gradient, as the unfused graph does.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton

from primus_turbo.flydsl.activation.gelu_kernel import (
    flydsl_gelu_cat_forward,
    gelu_cat_supported,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.triton.activation.gelu_bwd_kernel import gelu_tanh_bwd_bias_kernel
from primus_turbo.triton.reduce.colsum_kernel import colsum_kernel

_BIAS_BLOCK_R = 8
_BIAS_BLOCK_C = 256
_BIAS_PARTS = 128


def linear_gelu_shape_error(x, weight, bias, prefix) -> Optional[str]:
    """Why ``linear_gelu`` cannot take these operands, or None. Any 2-D shapes are accepted; the FlyDSL
    GELU runs on gfx950 when M is a multiple of 2048 and N and the prefix width are multiples of 3072,
    and the unfused GELU otherwise."""
    operands = [("x", x), ("weight", weight), ("bias", bias)] + (
        [] if prefix is None else [("prefix", prefix)]
    )
    for name, t in operands:
        if not isinstance(t, torch.Tensor) or not t.is_cuda or t.dtype != torch.bfloat16:
            return f"{name} must be a BF16 CUDA tensor"
    if x.dim() != 2 or weight.dim() != 2 or bias.dim() != 1:
        return (
            f"expected x [M, K], weight [N, K] and bias [N], got {tuple(x.shape)}, "
            f"{tuple(weight.shape)} and {tuple(bias.shape)}"
        )
    if weight.shape[1] != x.shape[1] or bias.shape[0] != weight.shape[0]:
        return f"x {tuple(x.shape)}, weight {tuple(weight.shape)} and bias {tuple(bias.shape)} do not match"
    if prefix is not None and (prefix.dim() != 2 or prefix.shape[0] != x.shape[0]):
        return f"prefix must be 2-D with x's {x.shape[0]} rows, got {tuple(prefix.shape)}"
    return None


@torch.library.custom_op("primus_turbo::linear_gelu_fwd_impl", mutates_args=(), device_types="cuda")
def linear_gelu_fwd_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, prefix: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cat([prefix, gelu_tanh(x @ weight.T + bias)], dim=1), x @ weight.T + bias)``."""
    x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
    m, n = x.shape[0], weight.shape[0]
    width = 0 if prefix is None else prefix.shape[1]
    aux = torch.addmm(bias, x, weight.t())
    if is_gfx950() and gelu_cat_supported(m, width, n):
        return flydsl_gelu_cat_forward(aux, prefix), aux
    out = torch.empty((m, width + n), device=x.device, dtype=x.dtype)
    if prefix is not None:
        out[:, :width].copy_(prefix)
    out[:, width:].copy_(F.gelu(aux, approximate="tanh"))
    return out, aux


@linear_gelu_fwd_impl.register_fake
def _linear_gelu_fwd_impl_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, prefix: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    width = 0 if prefix is None else prefix.shape[1]
    return x.new_empty((x.shape[0], width + weight.shape[0])), x.new_empty((x.shape[0], weight.shape[0]))


def gelu_bwd_bias_into(grad: torch.Tensor, aux: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    """Writes ``gelu_backward(grad, aux, "tanh")`` into ``grad_h`` and returns its column sums. All three
    are 2-D BF16 with unit column stride; their row strides may differ (column slices of wider buffers)."""
    n_rows, n_cols = aux.shape
    rows_per_prog = triton.cdiv(triton.cdiv(n_rows, _BIAS_PARTS), _BIAS_BLOCK_R) * _BIAS_BLOCK_R
    n_parts = triton.cdiv(n_rows, rows_per_prog)
    parts = torch.empty((n_parts, n_cols), device=aux.device, dtype=torch.float32)
    gelu_tanh_bwd_bias_kernel[(n_parts, triton.cdiv(n_cols, _BIAS_BLOCK_C))](
        grad,
        aux,
        grad_h,
        parts,
        n_rows,
        n_cols,
        grad.stride(0),
        rows_per_prog,
        aux.stride(0),
        grad_h.stride(0),
        BLOCK_R=_BIAS_BLOCK_R,
        BLOCK_C=_BIAS_BLOCK_C,
        num_warps=8,
    )
    grad_bias = torch.empty(n_cols, device=aux.device, dtype=aux.dtype)
    colsum_kernel[(triton.cdiv(n_cols, 64),)](
        parts, grad_bias, n_parts, n_cols, BLOCK_R=32, BLOCK_C=64, num_warps=4
    )
    return grad_bias


@torch.library.custom_op("primus_turbo::linear_gelu_bwd_impl", mutates_args=(), device_types="cuda")
def linear_gelu_bwd_impl(grad: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(gelu_backward(grad, aux, "tanh"), its column sums)`` for a 2-D BF16 ``grad``."""
    if grad.stride(1) != 1:
        grad = grad.contiguous()
    aux = aux.contiguous()
    grad_h = torch.empty_like(aux)
    return grad_h, gelu_bwd_bias_into(grad, aux, grad_h)


@linear_gelu_bwd_impl.register_fake
def _linear_gelu_bwd_impl_fake(grad: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return aux.new_empty(aux.shape), aux.new_empty((aux.shape[1],))
