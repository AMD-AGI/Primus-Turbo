###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL dense attention on gfx1250 (MI455X): gate, routing and numerics.

The gate and routing tests patch the arch probe and use meta / CPU tensors, so they run on
any machine. The numeric tests need a gfx1250 card; their fp32 reference runs on the CPU.
"""

import contextlib
import math
import os

import pytest
import torch

import primus_turbo.pytorch.ops.attention.flash_attn_interface as flash_attn_interface
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.attention import attention_impl
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    resolve_flash_attn_backend,
)
from primus_turbo.pytorch.ops import flash_attn_func
from tests.pytorch.test_utils import compute_snr, pinned_backend_takes

_ON_GFX1250 = torch.cuda.is_available() and is_gfx1250()
needs_gfx1250 = pytest.mark.skipif(not _ON_GFX1250, reason="gfx1250 FlyDSL attention kernels")

D = 128


def _meta(b, sq, skv, hq, hkv, d=D, dtype=torch.bfloat16):
    q = torch.empty(b, sq, hq, d, dtype=dtype, device="meta")
    k = torch.empty(b, skv, hkv, d, dtype=dtype, device="meta")
    return q, k, torch.empty_like(k)


def _gate(monkeypatch, q, k, v, **kwargs):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    kwargs.setdefault("causal", True)
    return attention_impl.DenseAttnFwdFlydslBackend.can_handle(q, k=k, v=v, **kwargs)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (4, 8192, 8192, 32, 8),
        (1, 64, 64, 1, 1),
        (2, 1024, 2048, 8, 2),
        (1, 256, 256, 64, 64),
        (1, 256, 256, 64, 4),
    ],
    ids=["llama3.1-8b", "smallest", "sq_lt_skv", "mha", "gqa16"],
)
@pytest.mark.parametrize("causal", [True, False])
def test_gate_accepts(monkeypatch, shape, causal):
    q, k, v = _meta(*shape)
    assert _gate(monkeypatch, q, k, v, causal=causal) is True


def test_gate_accepts_causal_spelled_as_a_window(monkeypatch):
    q, k, v = _meta(1, 256, 256, 4, 2)
    assert _gate(monkeypatch, q, k, v, window_size=(-1, 0)) is True


@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((1, 256, 256, 4, 2, 128, torch.float16), {}),
        ((1, 256, 256, 4, 2, 64), {}),
        ((1, 256, 256, 4, 2, 256), {}),
        ((1, 256, 256, 6, 4), {}),
        ((1, 256, 256, 28, 4), {}),
        ((1, 256, 256, 32, 1), {}),
        ((1, 96, 96, 4, 2), {}),
        ((1, 256, 240, 4, 2), {}),
        ((1, 512, 256, 4, 2), {}),
        ((1, 256, 256, 4, 2), {"dropout_p": 0.1}),
        ((1, 256, 256, 4, 2), {"bias": torch.empty(1)}),
        ((1, 256, 256, 4, 2), {"alibi_slopes": torch.empty(4)}),
        ((1, 256, 256, 4, 2), {"sink": torch.empty(4)}),
        ((1, 256, 256, 4, 2), {"window_size": (128, 0)}),
        ((1, 256, 256, 4, 2), {"window_size": (-1, 0), "causal": False}),
        ((1, 256, 256, 4, 2), {"return_softmax": True}),
        ((1, 256, 256, 4, 2), {"softmax_scale": True}),
    ],
    ids=[
        "fp16",
        "d64",
        "d256",
        "hq_not_multiple_of_hkv",
        "gqa_7",
        "gqa_32",
        "sq_not_multiple_of_64",
        "skv_not_multiple_of_32",
        "causal_sq_gt_skv",
        "dropout",
        "bias",
        "alibi",
        "sink",
        "sliding_window",
        "window_without_causal",
        "return_softmax",
        "bool_scale",
    ],
)
def test_gate_refuses(monkeypatch, shape, kwargs):
    q, k, v = _meta(*shape)
    assert _gate(monkeypatch, q, k, v, **kwargs) is False


def test_gate_refuses_missing_kv(monkeypatch):
    q, k, v = _meta(1, 256, 256, 4, 2)
    assert _gate(monkeypatch, q, None, v) is False
    assert _gate(monkeypatch, q, k, None) is False


def test_gfx950_kernels_are_not_offered_to_other_archs(monkeypatch):
    """gfx1250 compares greater than gfx950; the gfx950 gate used to accept it."""
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (12, 5))
    q, k, v = (t.permute(1, 0, 2, 3) for t in _meta(1, 256, 256, 4, 2))  # sbhd bytes
    backend = attention_impl.DenseAttnFwdFlydslBackend
    assert backend.can_handle(q, k=k, v=v, causal=True, qkv_format="sbhd") is False


# ---------------------------------------------------------------------------
# Dispatch and routing
# ---------------------------------------------------------------------------


def _resolve_kwargs(q, k, v, causal=True):
    return dict(q=q, k=k, v=v, causal=causal, window_size=(-1, -1), qkv_format="bshd", needs_backward=True)


def test_unpinned_gfx1250_call_resolves_to_flydsl(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    q, k, v = _meta(4, 8192, 8192, 32, 8)
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v)) == BackendType.FLYDSL


def test_pinned_flydsl_refuses_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    q, k, v = _meta(1, 96, 96, 4, 2)  # sq not a multiple of 64
    assert pinned_backend_takes(BackendType.FLYDSL, **_resolve_kwargs(q, k, v)) is False


def test_forward_and_backward_route_to_the_gfx1250_kernels(monkeypatch):
    """FlashAttnFunc on "gfx1250": the gfx1250 impls are called with contiguous BSHD tensors
    and lse [B, Hq, Sq], and the gfx950 sbhd impls are never reached."""
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    monkeypatch.setattr(flash_attn_interface, "is_gfx1250", lambda: True)
    b, sq, skv, hq, hkv = 2, 128, 256, 4, 2
    calls = []

    def fwd(q, k, v, softmax_scale=None, causal=True):
        calls.append(("fwd", q.is_contiguous(), k.is_contiguous(), softmax_scale, causal))
        return torch.zeros_like(q), torch.zeros(b, hq, sq)

    def bwd(dout, q, k, v, out, lse, softmax_scale=None, causal=True):
        calls.append(("bwd", tuple(lse.shape), tuple(out.shape)))
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v)

    def forbidden(*args, **kwargs):
        raise AssertionError("gfx950 FlyDSL impl reached on gfx1250")

    monkeypatch.setattr(flash_attn_interface, "flash_attn_flydsl_gfx1250_forward_impl", fwd)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_flydsl_gfx1250_backward_impl", bwd)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_sbhd_flydsl_forward_impl", forbidden)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_sbhd_flydsl_backward_impl", forbidden)

    # sbhd bytes on purpose: the adapter must hand the kernels contiguous BSHD.
    q = torch.randn(sq, b, hq, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    k = torch.randn(skv, b, hkv, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    v = torch.randn(skv, b, hkv, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    try:
        out = flash_attn_func(q, k, v, causal=True)
        out.sum().backward()
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)

    assert calls[0] == ("fwd", True, True, None, True)
    assert calls[1] == ("bwd", (b, hq, sq), (b, sq, hq, D))
    assert k.grad.shape == (b, skv, hkv, D) and torch.all(k.grad == 1)


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


def _inputs(b, sq, skv, hq, hkv, layout="bshd", seed=0):
    """[b, s, h, d]-shaped bf16 tensors whose bytes are in ``layout`` order."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def make(s, h):
        if layout == "bshd":
            t = torch.randn(b, s, h, D, generator=g)
        elif layout == "sbhd":
            t = torch.randn(s, b, h, D, generator=g).permute(1, 0, 2, 3)
        else:  # bhsd
            t = torch.randn(b, h, s, D, generator=g).transpose(1, 2)
        return t.to(torch.bfloat16)

    return make(sq, hq), make(skv, hkv), make(skv, hkv), make(sq, hq)


def _reference(q, k, v, dout, causal):
    """fp32 attention and its gradients on the CPU; bottom-right causal; returns lse too."""
    q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
    sq, skv, g = q.shape[1], k.shape[1], q.shape[2] // k.shape[2]
    s = torch.einsum("bqhd,bkhd->bhqk", q, k.repeat_interleave(g, 2)) / math.sqrt(D)
    if causal:
        mask = torch.ones(sq, skv, dtype=torch.bool).triu(skv - sq + 1)
        s = s.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(s, -1)
    out = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.repeat_interleave(g, 2))
    out.backward(dout.float())
    return out.detach(), lse.detach(), q.grad, k.grad, v.grad


@contextlib.contextmanager
def _pinned_flydsl():
    """Pin FLYDSL, so a refusal raises instead of quietly running another backend."""
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    try:
        yield
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)


def _run(q, k, v, dout, causal, **kwargs):
    q, k, v = (t.to("cuda").requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True, **kwargs)
    out.backward(dout.to("cuda"))
    return out, lse, q.grad, k.grad, v.grad


def _check(got, ref, min_db=40.0):
    for name, x, r in zip(("out", "lse", "dq", "dk", "dv"), got, ref):
        assert x.shape == r.shape, (name, x.shape, r.shape)
        x = x.float().cpu()
        assert torch.isfinite(x).all(), f"{name} has non-finite values"
        snr = compute_snr(r, x)
        assert snr > min_db, f"{name}: {snr:.1f} dB"


@needs_gfx1250
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 128, 2, 1),
        (2, 256, 256, 8, 2),
        (1, 512, 512, 4, 4),
        (1, 256, 512, 8, 2),
        (2, 256, 256, 16, 1),
        (1, 1024, 1024, 32, 8),
        (2, 2048, 2048, 32, 8),
        (1, 2048, 2048, 64, 8),
        (1, 1024, 1024, 128, 8),
        (1, 512, 4096, 16, 16),
    ],
    ids=[
        "toy",
        "gqa4",
        "mha",
        "sq_lt_skv",
        "gqa16",
        "llama_heads_small_grid",
        "llama_heads_large_grid",
        "gqa8_large_grid",
        "gqa16_large_grid",
        "dkdv_unsplit",
    ],
)
@pytest.mark.parametrize("causal", [True, False])
def test_matches_reference(shape, causal):
    """Covers both forward variants (grid below / above the CU count) and the backward's
    split-K (small grids), grouped dQ (large grids) and unsplit dK/dV (dkdv_unsplit, whose
    Skv/32 * Hkv * B = 2048 kv tiles fill the machine twice) paths."""
    q, k, v, dout = _inputs(*shape)
    _check(_run(q, k, v, dout, causal), _reference(q, k, v, dout, causal))


@needs_gfx1250
@pytest.mark.parametrize("layout", ["sbhd", "bhsd"])
def test_matches_reference_in_other_byte_orders(layout):
    q, k, v, dout = _inputs(2, 256, 256, 8, 2, layout=layout)
    _check(_run(q, k, v, dout, True), _reference(q, k, v, dout, True))


@needs_gfx1250
def test_explicit_softmax_scale():
    """The scale is a runtime argument, not baked in: scale c on q equals 1/sqrt(D) on
    q * c * sqrt(D), whose dq is the true dq divided by c * sqrt(D)."""
    c = 0.05 * math.sqrt(D)
    q, k, v, dout = _inputs(1, 256, 256, 4, 2)
    got = _run(q, k, v, dout, True, softmax_scale=0.05)
    out, lse, dq, dk, dv = _reference(q.float() * c, k, v, dout, True)
    _check(got, (out, lse, dq * c, dk, dv))


@needs_gfx1250
def test_backward_is_deterministic():
    q, k, v, dout = _inputs(2, 1024, 1024, 32, 8)
    first = _run(q, k, v, dout, True)
    for _ in range(3):
        again = _run(q, k, v, dout, True)
        for a, b in zip(first, again):
            assert torch.equal(a, b)


@needs_gfx1250
def test_runs_on_a_side_stream():
    q, k, v, dout = _inputs(1, 256, 256, 4, 2)
    ref = _reference(q, k, v, dout, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        got = _run(q, k, v, dout, True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    _check(got, ref)


@needs_gfx1250
def test_torch_compile_fullgraph():
    q, k, v, dout = _inputs(1, 256, 256, 4, 2)
    ref = _reference(q, k, v, dout, True)

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, causal=True, return_lse=True)

    q_, k_, v_ = (t.to("cuda").requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out, lse = fn(q_, k_, v_)
    out.backward(dout.to("cuda"))
    _check((out, lse, q_.grad, k_.grad, v_.grad), ref)


@needs_gfx1250
@pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_TEST_LARGE") != "1",
    reason="Llama-3.1-8B training shape; set PRIMUS_TURBO_TEST_LARGE=1 (about a minute of CPU reference)",
)
def test_llama31_8b_training_shape():
    """b4 s8192 hq32 hkv8: every output finite on the card; batch 0, kv head 0 (q heads 0-3,
    whose dk/dv depend on nothing else) against the CPU reference."""
    b, s, hq, hkv = 4, 8192, 32, 8
    q, k, v, dout = _inputs(b, s, s, hq, hkv)
    got = _run(q, k, v, dout, True)
    for x in got:
        assert torch.isfinite(x).all()
    g = hq // hkv
    sub = (q[:1, :, :g], k[:1, :, :1], v[:1, :, :1], dout[:1, :, :g])
    ref = _reference(*sub, True)
    out, lse, dq, dk, dv = got
    _check((out[:1, :, :g], lse[:1, :g], dq[:1, :, :g], dk[:1, :, :1], dv[:1, :, :1]), ref)
