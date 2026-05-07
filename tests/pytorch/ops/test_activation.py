###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import random

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.activation import (
    bias_geglu_impl,
    bias_gelu_impl,
    bias_swiglu_impl,
    weighted_bias_geglu_impl,
    weighted_bias_quick_geglu_impl,
    weighted_bias_swiglu_impl,
)
from tests.pytorch.test_utils import get_tolerances


def _reset_seed():
    torch.manual_seed(42)
    random.seed(42)


# ── Reference implementations (plain PyTorch, fp32 intermediate) ─────────────


def swiglu_ref(x: torch.Tensor):
    dtype = x.dtype
    x = x.float()
    x_1, x_2 = torch.chunk(x, 2, dim=-1)
    return (F.silu(x_1) * x_2).to(dtype)


def geglu_ref(x: torch.Tensor):
    dtype = x.dtype
    x = x.float()
    x_1, x_2 = torch.chunk(x, 2, dim=-1)
    return (F.gelu(x_1) * x_2).to(dtype)


def weighted_swiglu_ref(x: torch.Tensor, weights: torch.Tensor):
    dtype = x.dtype
    x = x.float()
    x_1, x_2 = torch.chunk(x, 2, dim=-1)
    return (F.silu(x_1) * x_2 * weights.float()).to(dtype)


def weighted_geglu_ref(x: torch.Tensor, weights: torch.Tensor):
    dtype = x.dtype
    x = x.float()
    x_1, x_2 = torch.chunk(x, 2, dim=-1)
    return (F.gelu(x_1) * x_2 * weights.float()).to(dtype)


def weighted_quick_geglu_ref(x: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0):
    dtype = x.dtype
    x = x.float()
    x_1, x_2 = torch.chunk(x, 2, dim=-1)
    return ((x_1 * torch.sigmoid(1.702 * x_1)) * (x_2 + linear_offset) * weights.float()).to(dtype)


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_fwd_tolerances(dtype):
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    return get_tolerances(dtype)


def get_bwd_tolerances(dtype):
    if dtype in (torch.bfloat16, torch.float16):
        return dict(rtol=3.5e-1, atol=3.5e-1)
    return get_tolerances(dtype)


def make_row_mask(num_tokens: int, device: str):
    """Create a row_mask with ~75% active rows."""
    mask = torch.ones(num_tokens, device=device, dtype=torch.int64)
    num_masked = max(1, num_tokens // 4)
    indices = torch.randperm(num_tokens, device=device)[:num_masked]
    mask[indices] = 0
    return mask


# ── Tests: bias_gelu_impl ────────────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 128, 2048])
@pytest.mark.parametrize("hidden_size", [128, 256, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_row_mask", [False, True])
def test_bias_gelu_impl(num_tokens, hidden_size, dtype, has_bias, has_row_mask):
    _reset_seed()
    device = "cuda"

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, dtype=dtype) if has_bias else None
    row_mask = make_row_mask(num_tokens, device) if has_row_mask else None

    x_ref = x.clone().detach().float().requires_grad_(True)

    out = bias_gelu_impl(x, bias, row_mask)

    y_ref = x_ref + bias.float() if has_bias else x_ref
    out_ref = F.gelu(y_ref).to(dtype)
    if has_row_mask:
        out_ref = out_ref * row_mask.unsqueeze(-1).to(out_ref.dtype)

    torch.testing.assert_close(out, out_ref, **get_fwd_tolerances(dtype))

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), **get_bwd_tolerances(dtype))


# ── Tests: bias_geglu_impl / bias_swiglu_impl ────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 128, 2048])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_row_mask", [False, True])
@pytest.mark.parametrize("act_type", ["geglu", "swiglu"])
def test_bias_glu_impl(num_tokens, hidden_size, dtype, has_bias, has_row_mask, act_type):
    _reset_seed()
    device = "cuda"

    x = torch.randn(num_tokens, hidden_size * 2, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size * 2, device=device, dtype=dtype) if has_bias else None
    row_mask = make_row_mask(num_tokens, device) if has_row_mask else None

    x_ref = x.clone().detach().float().requires_grad_(True)

    if act_type == "geglu":
        out = bias_geglu_impl(x, bias, row_mask)
    else:
        out = bias_swiglu_impl(x, bias, row_mask)

    y_ref = x_ref + bias.float() if has_bias else x_ref
    y1, y2 = torch.chunk(y_ref, 2, dim=-1)
    if act_type == "geglu":
        out_ref = (F.gelu(y1) * y2).to(dtype)
    else:
        out_ref = (F.silu(y1) * y2).to(dtype)
    if has_row_mask:
        out_ref = out_ref * row_mask.unsqueeze(-1).to(out_ref.dtype)

    torch.testing.assert_close(out, out_ref, **get_fwd_tolerances(dtype))

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), **get_bwd_tolerances(dtype))


# ── Tests: weighted_bias_swiglu_impl / weighted_bias_geglu_impl ──────────────


@pytest.mark.parametrize("num_tokens", [1, 64, 128])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_row_mask", [False, True])
@pytest.mark.parametrize("act_type", ["geglu", "swiglu"])
def test_weighted_bias_glu_impl(num_tokens, hidden_size, dtype, has_bias, has_row_mask, act_type):
    _reset_seed()
    device = "cuda"

    x = torch.randn(num_tokens, hidden_size * 2, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size * 2, device=device, dtype=dtype) if has_bias else None
    weights = torch.rand(num_tokens, 1, device=device, dtype=torch.float32, requires_grad=True)
    row_mask = make_row_mask(num_tokens, device) if has_row_mask else None

    x_ref = x.clone().detach().float().requires_grad_(True)
    w_ref = weights.clone().detach().requires_grad_(True)

    if act_type == "geglu":
        out = weighted_bias_geglu_impl(x, bias, weights, row_mask)
    else:
        out = weighted_bias_swiglu_impl(x, bias, weights, row_mask)

    y_ref = x_ref + bias.float() if has_bias else x_ref
    y1, y2 = torch.chunk(y_ref, 2, dim=-1)
    if act_type == "geglu":
        act_ref = F.gelu(y1) * y2
    else:
        act_ref = F.silu(y1) * y2
    out_ref = (act_ref * w_ref.float()).to(dtype)
    if has_row_mask:
        out_ref = out_ref * row_mask.unsqueeze(-1).to(out_ref.dtype)

    torch.testing.assert_close(out, out_ref, **get_fwd_tolerances(dtype))

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), **get_bwd_tolerances(dtype))
    torch.testing.assert_close(weights.grad, w_ref.grad, **get_bwd_tolerances(dtype))


# ── Tests: weighted_bias_quick_geglu_impl ────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [16, 64, 128])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_row_mask", [False, True])
@pytest.mark.parametrize("linear_offset", [0.0, 1.0])
def test_weighted_bias_quick_geglu_impl(
    num_tokens, hidden_size, dtype, has_bias, has_row_mask, linear_offset
):
    _reset_seed()
    device = "cuda"

    x = torch.randn(num_tokens, hidden_size * 2, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size * 2, device=device, dtype=dtype) if has_bias else None
    weights = torch.rand(num_tokens, 1, device=device, dtype=torch.float32, requires_grad=True)
    row_mask = make_row_mask(num_tokens, device) if has_row_mask else None

    x_ref = x.clone().detach().float().requires_grad_(True)
    w_ref = weights.clone().detach().requires_grad_(True)

    out = weighted_bias_quick_geglu_impl(x, bias, weights, row_mask, linear_offset)

    y_ref = x_ref + bias.float() if has_bias else x_ref
    y1, y2 = torch.chunk(y_ref, 2, dim=-1)
    out_ref = ((y1 * torch.sigmoid(1.702 * y1)) * (y2 + linear_offset) * w_ref.float()).to(dtype)
    if has_row_mask:
        out_ref = out_ref * row_mask.unsqueeze(-1).to(out_ref.dtype)

    torch.testing.assert_close(out, out_ref, **get_fwd_tolerances(dtype))

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), **get_bwd_tolerances(dtype))
    torch.testing.assert_close(weights.grad, w_ref.grad, **get_bwd_tolerances(dtype))
