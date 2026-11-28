###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope

_D = 64
_FLYDSL_ARCHS = ("gfx942", "gfx950")


def _arch():
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _require_backend(backend):
    _require_gpu()
    if backend == BackendType.FLYDSL and _arch() not in _FLYDSL_ARCHS:
        pytest.skip(f"the FlyDSL backend requires one of {_FLYDSL_ARCHS}")


_BACKENDS = [BackendType.FLYDSL, BackendType.TRITON]


def _rmsnorm(x, gamma, eps):
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    # This cast is part of the production contract: standalone RMSNorm writes
    # BF16 before standalone RoPE reads it.
    y = x.float() * rstd
    return (y if gamma is None else y * gamma.float()).bfloat16()


def _rope(x, freqs):
    half = x.shape[-1] // 2
    angle = freqs[:, :half].float()[:, None, None, :]
    cosine, sine = angle.cos(), angle.sin()
    lo, hi = x[..., :half].float(), x[..., half:].float()
    return torch.cat((lo * cosine - hi * sine, hi * cosine + lo * sine), dim=-1).bfloat16()


def _reference(qkv, q_gamma, k_gamma, freqs, split, eps):
    D = split[-2]
    gated = len(split) == 4
    parts = qkv.split(tuple(split), dim=-1)
    q, k, v = parts[0], parts[-2], parts[-1]
    S, B, NG = qkv.shape[:3]
    q = q.reshape(S, B, -1, D)
    k = k.reshape(S, B, NG, D)
    v = v.reshape(S, B, NG, D).clone()
    q, k = _rmsnorm(q, q_gamma, eps), _rmsnorm(k, k_gamma, eps)
    if freqs is not None:
        q, k = _rope(q, freqs.reshape(S, D)), _rope(k, freqs.reshape(S, D))
    return (q, k, v, parts[1].clone()) if gated else (q, k, v)


def _inputs(S=8, B=4, NG=2, NPG=8, seed=123, D=_D, gated=False, weights=True, rope=True):
    _require_gpu()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    split = [NPG * D, NPG * D, D, D] if gated else [NPG * D, D, D]
    qkv = torch.randn(S, B, NG, sum(split), device="cuda", generator=gen, dtype=torch.float32).bfloat16()
    q_gamma = torch.randn(D, device="cuda", generator=gen, dtype=torch.float32).bfloat16()
    k_gamma = torch.randn(D, device="cuda", generator=gen, dtype=torch.float32).bfloat16()
    # Megatron supplies duplicated rotate-half angles in [S,1,1,D].
    half = torch.randn(S, D // 2, device="cuda", generator=gen, dtype=torch.float32)
    freqs = torch.cat((half, half), dim=-1).reshape(S, 1, 1, D).contiguous()
    return (
        qkv,
        q_gamma if weights else None,
        k_gamma if weights else None,
        freqs if rope else None,
        split,
    )


def _check_forward(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    for got, want in zip(actual[2:], expected[2:]):  # v and the gate are passed through
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def _check_backward(args, eps, backend):
    qkv, q_gamma, k_gamma, freqs, split = args
    leaves = [t for t in (qkv, q_gamma, k_gamma) if t is not None]
    refs = [t.detach().clone().requires_grad_() for t in leaves]
    for t in leaves:
        t.requires_grad_()
    ref_args = dict(zip(("qkv", "q_gamma", "k_gamma"), refs + [None] * (3 - len(refs))))
    actual = fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, split, eps, backend=backend)
    expected = _reference(ref_args["qkv"], ref_args["q_gamma"], ref_args["k_gamma"], freqs, split, eps)
    gen = torch.Generator(device="cuda").manual_seed(456)
    grads = tuple(
        torch.randn(x.shape, device="cuda", generator=gen, dtype=torch.float32).bfloat16() for x in actual
    )
    torch.autograd.backward(actual, grads)
    torch.autograd.backward(expected, grads)
    for got, want in zip(leaves, refs):
        torch.testing.assert_close(got.grad, want.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "S,B,NG,NPG,D",
    [
        (4, 8, 2, 8, 64),
        (8, 2, 8, 8, 64),
        (129, 4, 4, 8, 64),
        (4, 1, 16, 1, 64),
        # head_dim 128: Qwen3 GQA groups (NG=4, NPG=8), odd S, single-head groups
        (4, 8, 4, 8, 128),
        (129, 2, 4, 8, 128),
        (8, 1, 8, 1, 128),
    ],
)
def test_fused_qkv_rmsnorm_rope_forward(S, B, NG, NPG, D, backend):
    _require_backend(backend)
    eps = 1.0e-5
    args = _inputs(S=S, B=B, NG=NG, NPG=NPG, D=D)
    _check_forward(fused_qkv_rmsnorm_rope(*args, eps, backend=backend), _reference(*args, eps))


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "B,NG,NPG,D",
    [(2, 8, 8, 64), (1, 16, 1, 64), (8, 4, 8, 128), (2, 4, 8, 128), (1, 8, 1, 128)],
)
def test_fused_qkv_rmsnorm_rope_backward(B, NG, NPG, D, backend):
    _require_backend(backend)
    _check_backward(_inputs(S=65, B=B, NG=NG, NPG=NPG, seed=321, D=D), 1.0e-5, backend)


# Triton-only inputs: a gate block (Megatron's attention_output_gate), weightless norms, no RoPE (NoPE layers),
# heads per group that are not a power of 2, groups not divisible by the groups per program, other head dims.
_TRITON_CASES = [
    # S, B, NG, NPG, D, gated, weights, rope
    (64, 2, 16, 8, 64, True, False, True),  # Muse Spark: gated, weightless
    (64, 2, 16, 8, 64, True, False, False),  # Muse Spark NoPE layer
    (64, 2, 4, 8, 64, True, True, True),
    (33, 1, 6, 3, 128, False, True, True),
    (33, 3, 5, 6, 32, True, False, True),
    (16, 2, 3, 1, 64, False, False, False),
]


@pytest.mark.parametrize("S,B,NG,NPG,D,gated,weights,rope", _TRITON_CASES)
def test_fused_qkv_rmsnorm_rope_triton_cases(S, B, NG, NPG, D, gated, weights, rope):
    eps = 1.0e-5
    args = _inputs(S=S, B=B, NG=NG, NPG=NPG, D=D, gated=gated, weights=weights, rope=rope)
    _check_forward(fused_qkv_rmsnorm_rope(*args, eps), _reference(*args, eps))
    _check_backward(
        _inputs(S=S, B=B, NG=NG, NPG=NPG, D=D, gated=gated, weights=weights, rope=rope, seed=7), eps, None
    )


def test_fused_qkv_rmsnorm_rope_views_and_partial_grads():
    eps = 1.0e-5
    S, B, NG, NPG = 32, 2, 4, 8
    qkv, _, _, freqs, split = _inputs(S=S, B=B, NG=NG, NPG=NPG, gated=True, weights=False)
    qkv.requires_grad_()
    q, k, v, gate = fused_qkv_rmsnorm_rope(qkv, None, None, freqs, split, eps)
    assert gate.shape == (S, B, NG, NPG * _D) and gate.data_ptr() == qkv.data_ptr() + NPG * _D * 2
    assert v.data_ptr() == qkv.data_ptr() + (2 * NPG + 1) * _D * 2
    # only q and the gate reach the loss: the k and v columns of d(qkv) are zero
    w = torch.randn(B, S, NG * NPG, _D, device="cuda").bfloat16().transpose(0, 1)  # non-contiguous dq
    ((q.float() * w.float()).sum() + gate.float().sum()).backward()
    assert torch.count_nonzero(qkv.grad[..., 2 * NPG * _D :]) == 0
    assert torch.equal(qkv.grad[..., NPG * _D : 2 * NPG * _D], torch.ones_like(gate))


@pytest.mark.parametrize("B,NG,D", [(4, 2, 64), (2, 2, 128)])
def test_fused_qkv_rmsnorm_rope_untileable_shape(B, NG, D):
    """Shapes the FlyDSL row tiling does not cover: forcing FlyDSL raises, the default takes Triton."""
    eps = 1.0e-5
    args = _inputs(S=8, B=B, NG=NG, D=D)
    if _arch() in _FLYDSL_ARCHS:
        with pytest.raises(ValueError, match="rows_per_wave"):
            fused_qkv_rmsnorm_rope(*args, eps, backend=BackendType.FLYDSL)
    _check_forward(fused_qkv_rmsnorm_rope(*args, eps), _reference(*args, eps))


def test_fused_qkv_rmsnorm_rope_flydsl_rejects_triton_only_inputs():
    _require_backend(BackendType.FLYDSL)
    args = _inputs(B=8, NG=2, gated=True)
    with pytest.raises(ValueError, match="no gate block"):
        fused_qkv_rmsnorm_rope(*args, 1.0e-5, backend=BackendType.FLYDSL)


def test_fused_qkv_rmsnorm_rope_rejects_wrong_head_dim():
    qkv, q_gamma, k_gamma, freqs, split = _inputs(B=8, NG=2, D=96)
    with pytest.raises(ValueError, match="head_dim"):
        fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, split, 1.0e-5)


def test_fused_qkv_rmsnorm_rope_rejects_strided_gamma():
    qkv, _, k_gamma, freqs, split = _inputs(B=8, NG=2)
    q_gamma = torch.randn(2 * _D, device="cuda", dtype=torch.bfloat16)[::2]
    assert not q_gamma.is_contiguous()
    with pytest.raises(ValueError, match="contiguous bfloat16"):
        fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, split, 1.0e-5)


def test_fused_qkv_rmsnorm_rope_rejects_one_gamma():
    qkv, q_gamma, _, freqs, split = _inputs(B=8, NG=2)
    with pytest.raises(ValueError, match="both be given or both be None"):
        fused_qkv_rmsnorm_rope(qkv, q_gamma, None, freqs, split, 1.0e-5)


def test_fused_qkv_rmsnorm_rope_rejects_gate_width():
    qkv, q_gamma, k_gamma, freqs, _ = _inputs(B=8, NG=2, gated=True)
    with pytest.raises(ValueError, match="gate width"):
        fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, [8 * _D, 7 * _D, _D, _D], 1.0e-5)


def test_fused_qkv_rmsnorm_rope_rejects_zero_query_width():
    qkv, q_gamma, k_gamma, freqs, _ = _inputs(B=1, NG=16, NPG=1)
    qkv = qkv[..., -2 * _D :].contiguous()
    with pytest.raises(ValueError, match="q multiple"):
        fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, [0, _D, _D], 1.0e-5)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_fused_qkv_rmsnorm_rope_rejects_zero_dimensions(dim):
    qkv, q_gamma, k_gamma, freqs, split = _inputs(S=8, B=2, NG=8)
    slices = [slice(None)] * qkv.ndim
    slices[dim] = slice(0)
    qkv = qkv[tuple(slices)].contiguous()
    if dim == 0:
        freqs = freqs[:0].contiguous()

    with pytest.raises(ValueError, match="must be positive"):
        fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, split, 1.0e-5)
