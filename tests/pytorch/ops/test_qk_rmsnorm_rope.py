###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope

_D = 64


def _rmsnorm(x, gamma, eps):
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    # This cast is part of the production contract: standalone RMSNorm writes
    # BF16 before standalone RoPE reads it.
    return (x.float() * rstd * gamma.float()).bfloat16()


def _rope(x, freqs):
    half = x.shape[-1] // 2
    angle = freqs[:, :half].float()[:, None, None, :]
    cosine, sine = angle.cos(), angle.sin()
    lo, hi = x[..., :half].float(), x[..., half:].float()
    return torch.cat((lo * cosine - hi * sine, hi * cosine + lo * sine), dim=-1).bfloat16()


def _reference(qkv, q_gamma, k_gamma, freqs, split, eps):
    q_width, k_width, v_width = split
    q, k, v = qkv.split((q_width, k_width, v_width), dim=-1)
    S, B, NG = qkv.shape[:3]
    q = q.reshape(S, B, -1, _D)
    k = k.reshape(S, B, NG, _D)
    v = v.reshape(S, B, NG, _D).clone()
    q = _rope(_rmsnorm(q, q_gamma, eps), freqs.reshape(S, _D))
    k = _rope(_rmsnorm(k, k_gamma, eps), freqs.reshape(S, _D))
    return q, k, v


def _inputs(S=8, B=4, NG=2, NPG=8, seed=123):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    split = [NPG * _D, _D, _D]
    qkv = torch.randn(S, B, NG, sum(split), device="cuda", generator=gen, dtype=torch.float32).bfloat16()
    q_gamma = torch.randn(_D, device="cuda", generator=gen, dtype=torch.float32).bfloat16()
    k_gamma = torch.randn(_D, device="cuda", generator=gen, dtype=torch.float32).bfloat16()
    # Megatron supplies duplicated rotate-half angles in [S,1,1,D].
    half = torch.randn(S, _D // 2, device="cuda", generator=gen, dtype=torch.float32)
    freqs = torch.cat((half, half), dim=-1).reshape(S, 1, 1, _D).contiguous()
    return qkv, q_gamma, k_gamma, freqs, split


@pytest.mark.parametrize("S,B,NG", [(4, 1, 1), (8, 4, 2)])
def test_fused_qkv_rmsnorm_rope_forward(S, B, NG):
    eps = 1.0e-5
    args = _inputs(S=S, B=B, NG=NG)
    actual = fused_qkv_rmsnorm_rope(*args, eps)
    expected = _reference(*args, eps)
    for got, want in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_fused_qkv_rmsnorm_rope_backward():
    eps = 1.0e-5
    qkv, q_gamma, k_gamma, freqs, split = _inputs(seed=321)
    qkv.requires_grad_()
    q_gamma.requires_grad_()
    k_gamma.requires_grad_()
    qkv_ref = qkv.detach().clone().requires_grad_()
    q_gamma_ref = q_gamma.detach().clone().requires_grad_()
    k_gamma_ref = k_gamma.detach().clone().requires_grad_()

    actual = fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, split, eps)
    expected = _reference(qkv_ref, q_gamma_ref, k_gamma_ref, freqs, split, eps)
    gen = torch.Generator(device="cuda").manual_seed(456)
    grads = tuple(
        torch.randn(x.shape, device="cuda", generator=gen, dtype=torch.float32).bfloat16() for x in actual
    )
    torch.autograd.backward(actual, grads)
    torch.autograd.backward(expected, grads)

    torch.testing.assert_close(qkv.grad, qkv_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(q_gamma.grad, q_gamma_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_gamma.grad, k_gamma_ref.grad, rtol=2e-2, atol=2e-2)


def test_fused_qkv_rmsnorm_rope_rejects_wrong_head_dim():
    qkv, q_gamma, k_gamma, freqs, _ = _inputs()
    with pytest.raises(ValueError, match="unsupported input"):
        fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, [8 * _D, 128, 128], 1.0e-5)
