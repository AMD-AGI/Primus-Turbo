###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Returned LSE, padded head dimensions and FP8 backward on gfx950."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScalingGranularity
from primus_turbo.pytorch.ops import flash_attn_fp8_func, flash_attn_func, flash_attn_varlen_func
from tests.pytorch.ops.gradient_contract_helpers import (  # noqa: F401
    attention_reference,
    caller_thread_backward,
)
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.mark.parametrize("head_dim", [63, 64])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_aiter_padded_heads_and_lse(head_dim, varlen, deterministic):
    torch.manual_seed(804)
    GlobalBackendManager.set_attn_backend(BackendType.AITER)
    shapes = [(2, 128, h, head_dim) for h in (4, 2, 2)]
    full = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes]
    xs = [(x.flatten(0, 1) if varlen else x).detach().requires_grad_() for x in full]
    refs = [x.float().requires_grad_() for x in full]
    if varlen:
        cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
        out, lse, softmax = flash_attn_varlen_func(
            *xs,
            cu,
            cu,
            128,
            128,
            causal=True,
            deterministic=deterministic,
            return_lse=True,
            return_attn_probs=True,
        )
        expected_shape = (4, 256)
    else:
        out, lse, softmax = flash_attn_func(
            *xs, causal=True, deterministic=deterministic, return_lse=True, return_attn_probs=True
        )
        expected_shape = (2, 4, 128)
    ref = attention_reference(*refs)
    torch.testing.assert_close(out.float().reshape(ref.shape), ref, atol=0.02, rtol=0.02)
    q, k, _ = refs
    scores = (
        q.transpose(1, 2) @ k.repeat_interleave(2, dim=2).transpose(1, 2).transpose(-1, -2) / head_dim**0.5
    )
    scores = scores.masked_fill(torch.ones(128, 128, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    expected_lse = scores.logsumexp(-1)
    if varlen:
        expected_lse = expected_lse.transpose(0, 1).reshape(4, 256)
    assert lse.shape == expected_shape
    torch.testing.assert_close(lse, expected_lse, rtol=0.01, atol=0.01)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.reshape(ref.shape).float())
    for x, r in zip(xs, refs):
        assert compute_snr(r.grad, x.grad.reshape(r.shape).float()) > 35


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("grad_inputs", ["q", "kv", "qkv"])
def test_fp8_attention_gradients(causal, grad_inputs):
    torch.manual_seed(805)
    xs = [
        torch.randn(2, 128, h, 64, device="cuda", dtype=torch.bfloat16).requires_grad_(n in grad_inputs)
        for n, h in zip("qkv", (4, 2, 2))
    ]
    refs = [x.detach().float().requires_grad_(x.requires_grad) for x in xs]
    out, lse = flash_attn_fp8_func(*xs, causal=causal, return_lse=True)
    ref = attention_reference(*refs, causal=causal)
    assert torch.isfinite(lse).all()
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    assert compute_snr(ref, out.float()) > 20
    for x, r in zip(xs, refs):
        if x.requires_grad:
            assert compute_snr(r.grad, x.grad.float()) > 15
        else:
            assert x.grad is None


@pytest.mark.parametrize(
    "config",
    [Float8QuantConfig(), Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128)],
)
def test_fp8_attention_rejects_incompatible_quantization(config):
    x = torch.zeros(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="only supports"):
        flash_attn_fp8_func(x, x, x, fp8_config=config)
