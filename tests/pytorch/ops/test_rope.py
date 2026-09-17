###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.flydsl.rope.rope_kernel import ROPE_HEAD_DIM as _D
from primus_turbo.pytorch.ops.rope import fused_qkv_rope, fused_qkv_rope_supported
from tests.pytorch.test_utils import get_tolerances


def _rope_ref(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE over ``[S, B, heads, D]`` given per-position angles ``[S, D]``.

    The first half of a head pairs with the second; ``angles[s, p]`` turns the pair
    ``(x[..., p], x[..., p + D // 2])``.
    """
    half = x.shape[-1] // 2
    ang = angles[:, :half].to(torch.float32)
    cos = ang.cos()[:, None, None, :]
    sin = ang.sin()[:, None, None, :]
    lo, hi = x[..., :half].float(), x[..., half:].float()
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1)


def _split_ref(qkv, q_freqs, k_freqs, qkv_split_arg_list):
    """Reference for the whole op: rotate q and k, pass v through.

    ``qkv`` packs one KV group per row as ``[q heads..., k, v]``; the op hands back
    the heads split out, so q is ``[S, B, groups * q_per_group, D]`` and k and v are
    ``[S, B, groups, D]``.
    """
    q_size, k_size, v_size = qkv_split_arg_list
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
    S, B, NG = qkv.shape[:3]
    q = _rope_ref(q.reshape(S, B, -1, _D), q_freqs.reshape(S, _D))
    k = _rope_ref(k.reshape(S, B, NG, _D), k_freqs.reshape(S, _D))
    return q.to(qkv.dtype), k.to(qkv.dtype), v.reshape(S, B, NG, _D).clone()


def _inputs(S, B, n_q_heads, n_kv_heads=1, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    q_size = n_q_heads * _D
    k_size = v_size = n_kv_heads * _D
    qkv = torch.randn(
        S, B, 1, q_size + k_size + v_size, generator=gen, device=device, dtype=torch.float32
    ).bfloat16()
    q_freqs = torch.randn(S, 1, 1, _D, generator=gen, device=device, dtype=torch.float32)
    k_freqs = torch.randn(S, 1, 1, _D, generator=gen, device=device, dtype=torch.float32)
    return qkv, q_freqs, k_freqs, [q_size, k_size, v_size]


@pytest.mark.parametrize("S,B", [(4, 1), (8, 4), (16, 2), (64, 4)])
@pytest.mark.parametrize("n_q_heads", [1, 4, 8])
def test_fused_qkv_rope_forward(S, B, n_q_heads):
    qkv, q_freqs, k_freqs, split = _inputs(S, B, n_q_heads)
    q, k, v = fused_qkv_rope(qkv, q_freqs, k_freqs, split)
    q_ref, k_ref, v_ref = _split_ref(qkv, q_freqs, k_freqs, split)
    tol = get_tolerances(torch.bfloat16)
    torch.testing.assert_close(q, q_ref, **tol)
    torch.testing.assert_close(k, k_ref, **tol)
    # v is a copy, so it has to come back untouched.
    torch.testing.assert_close(v, v_ref, rtol=0, atol=0)


@pytest.mark.parametrize("S,B", [(4, 1), (8, 4)])
@pytest.mark.parametrize("n_q_heads", [1, 4])
def test_fused_qkv_rope_backward(S, B, n_q_heads):
    qkv, q_freqs, k_freqs, split = _inputs(S, B, n_q_heads, seed=1)
    qkv = qkv.requires_grad_()
    qkv_ref = qkv.detach().clone().requires_grad_()

    q, k, v = fused_qkv_rope(qkv, q_freqs, k_freqs, split)
    q_ref, k_ref, v_ref = _split_ref(qkv_ref, q_freqs, k_freqs, split)

    gen = torch.Generator(device="cuda").manual_seed(7)
    gq, gk, gv = (
        torch.randn(t.shape, generator=gen, device="cuda", dtype=torch.float32).bfloat16() for t in (q, k, v)
    )
    torch.autograd.backward([q, k, v], [gq, gk, gv])
    torch.autograd.backward([q_ref, k_ref, v_ref], [gq.float(), gk.float(), gv.float()])

    torch.testing.assert_close(qkv.grad, qkv_ref.grad.to(qkv.dtype), **get_tolerances(torch.bfloat16))


@pytest.mark.parametrize(
    "mutate,reason",
    [
        # S*B must miss the row group, which needs B itself to miss it -- with B=4
        # every S lands on a multiple.
        (lambda a: (a[0][:3], a[1][:3], a[2][:3], a[3]), "S*B not a multiple of the row group"),
        (lambda a: (a[0].float(), *a[1:]), "dtype is not bfloat16"),
        (lambda a: (a[0], a[1], a[2], [a[3][0], 64, 64]), "k/v are not one 128-wide head"),
        (lambda a: (a[0][:, :, :, :-1], *a[1:]), "last dim does not match the split"),
    ],
)
def test_unsupported_inputs_are_rejected(mutate, reason):
    """Shapes the kernels were not built for must raise, not read out of bounds."""
    args = mutate(_inputs(8, 1, 4))
    assert not fused_qkv_rope_supported(*args), reason
    with pytest.raises(ValueError, match="unsupported input"):
        fused_qkv_rope(*args)
