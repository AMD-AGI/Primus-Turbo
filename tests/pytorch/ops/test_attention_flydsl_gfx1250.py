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
import importlib
import math
import os
import pathlib
import sys
import types

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

# The gfx1250 kernel packages; importing primus_turbo (the imports above) must not import any of them.
_GFX1250_KERNEL_PACKAGES = tuple(
    f"primus_turbo.flydsl.attention.{p}" for p in ("gfx1250", "gfx1250_mla_fwd", "gfx1250_mla_bwd")
)


def _gfx1250_kernel_modules():
    return sorted(m for m in sys.modules for p in _GFX1250_KERNEL_PACKAGES if m == p or m.startswith(p + "."))


_GFX1250_KERNEL_MODULES_AT_IMPORT = _gfx1250_kernel_modules()


def _mla_flydsl_reason():
    from primus_turbo.flydsl.attention.gfx1250_mla_version import (
        flydsl_unavailable_reason,
    )

    return flydsl_unavailable_reason()


# MLA tests that need the MLA kernel modules importable (flydsl 0.3.4.x); with another flydsl the
# gate refuses every MLA call, which test_mla_gate_falls_back_without_flydsl_0_3_4 covers.
needs_mla_flydsl = pytest.mark.skipif(_mla_flydsl_reason() is not None, reason=str(_mla_flydsl_reason()))

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


# ---------------------------------------------------------------------------
# DeepSeek-V3 MLA head dims: qk 192 / v 128, MHA (primus_turbo/flydsl/attention/gfx1250_mla_*)
# ---------------------------------------------------------------------------

D_QK, D_V = 192, 128
# DeepSeek-V3's softmax scale (yarn mscale^2 / sqrt(192)); a runtime argument, not 1/sqrt(192).
MLA_SCALE = 0.1352337788608801


def _meta_mla(b, sq, skv, hq, hkv, dtype=torch.bfloat16, dv=D_V):
    q = torch.empty(b, sq, hq, D_QK, dtype=dtype, device="meta")
    k = torch.empty(b, skv, hkv, D_QK, dtype=dtype, device="meta")
    v = torch.empty(b, skv, hkv, dv, dtype=dtype, device="meta")
    return q, k, v


@pytest.mark.parametrize(
    "shape",
    [(2, 4096, 4096, 128, 128), (1, 256, 256, 2, 2), (1, 256, 512, 4, 4), (4, 64, 64, 1, 1)],
    ids=["dsv3_mbs2", "toy", "sq_lt_skv", "smallest"],
)
@needs_mla_flydsl
def test_mla_gate_accepts(monkeypatch, shape):
    q, k, v = _meta_mla(*shape)
    assert _gate(monkeypatch, q, k, v, causal=True, softmax_scale=MLA_SCALE) is True


@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((1, 256, 256, 4, 2), {}),
        ((1, 256, 256, 2, 2, torch.float16), {}),
        ((1, 256, 256, 2, 2, torch.bfloat16, 192), {}),
        ((1, 96, 96, 2, 2), {}),
        ((1, 256, 240, 2, 2), {}),
        ((1, 256, 288, 2, 2), {}),
        ((1, 512, 256, 2, 2), {}),
        ((1, 256, 256, 2, 2), {"dropout_p": 0.1}),
        ((1, 256, 256, 2, 2), {"sink": torch.empty(2)}),
        ((1, 256, 256, 2, 2), {"window_size": (128, 0)}),
        ((1, 256, 256, 2, 2), {"window_size": (-1, 0), "causal": False}),
        ((1, 256, 256, 2, 2), {"return_softmax": True}),
        ((1, 256, 256, 2, 2), {"causal": False}),
    ],
    ids=[
        "gqa",
        "fp16",
        "v_192",
        "sq_not_multiple_of_64",
        "skv_not_multiple_of_32",
        "skv_not_multiple_of_64",
        "causal_sq_gt_skv",
        "dropout",
        "sink",
        "sliding_window",
        "window_without_causal",
        "return_softmax",
        "non_causal_forward_spills",
    ],
)
def test_mla_gate_refuses(monkeypatch, shape, kwargs):
    q, k, v = _meta_mla(*shape)
    assert _gate(monkeypatch, q, k, v, **kwargs) is False


def _mla_reference(q, k, v, dout, causal, scale):
    """fp32 CPU attention + grads for [b, s, h, d] tensors (MHA, bottom-right causal)."""
    with torch.enable_grad():
        q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
        sq, skv = q.shape[1], k.shape[1]
        s = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
        if causal:
            s = s.masked_fill(torch.ones(sq, skv, dtype=torch.bool).triu(skv - sq + 1), float("-inf"))
        lse = torch.logsumexp(s, -1)
        out = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v)
        out.backward(dout.float())
    return out.detach(), lse.detach(), q.grad, k.grad, v.grad


def _mla_inputs(b, sq, skv, h, layout="sbhd", seed=0):
    """[b, s, h, d]-shaped bf16 q/k/v/dout whose bytes are in ``layout`` order."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def make(s, d):
        if layout == "bshd":
            t = torch.randn(b, s, h, d, generator=g)
        elif layout == "sbhd":
            t = torch.randn(s, b, h, d, generator=g).permute(1, 0, 2, 3)
        else:  # bhsd
            t = torch.randn(b, h, s, d, generator=g).transpose(1, 2)
        return t.to(torch.bfloat16)

    return make(sq, D_QK), make(skv, D_QK), make(skv, D_V), make(sq, D_V)


@needs_mla_flydsl
@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
def test_mla_routes_to_the_mla_kernels_without_copies(monkeypatch, layout):
    """FlashAttnFunc on "gfx1250" with MLA head dims: the MLA kernels get contiguous tensors --
    for b > 1 sbhd storage the batch folded into the heads, [1, s, b*h, d], with no copy -- and
    the results come back [b, s, h, d] (views of sbhd storage), equal to an fp32 reference. The
    kernels are replaced by that reference on the kernel layout, so this runs on the CPU."""
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    monkeypatch.setattr(flash_attn_interface, "is_gfx1250", lambda: True)
    from primus_turbo.pytorch.kernels.attention import (
        attention_flydsl_gfx1250_mla_impl as mla_impl,
    )

    b, s, h = 2, 128, 4
    calls = []

    def fwd(q, k, v, softmax_scale, causal):
        calls.append(("fwd", tuple(q.shape), q.is_contiguous(), q.data_ptr(), softmax_scale, causal))
        out, lse, *_ = _mla_reference(q, k, v, torch.zeros(*q.shape[:3], D_V), causal, softmax_scale)
        return out.to(q.dtype).contiguous(), lse.contiguous()  # the kernels' outputs are contiguous

    def bwd(dout, q, k, v, out, lse, softmax_scale, causal):
        calls.append(("bwd", tuple(dout.shape), dout.is_contiguous(), tuple(lse.shape)))
        _, _, dq, dk, dv = _mla_reference(q, k, v, dout, causal, softmax_scale)
        return tuple(g.to(q.dtype).contiguous() for g in (dq, dk, dv))

    def forbidden(*args, **kwargs):
        raise AssertionError("head-dim-128 gfx1250 impl reached for MLA head dims")

    monkeypatch.setattr(mla_impl, "_forward", fwd)
    monkeypatch.setattr(mla_impl, "_backward", bwd)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_flydsl_gfx1250_forward_impl", forbidden)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_flydsl_gfx1250_backward_impl", forbidden)

    q, k, v, dout = _mla_inputs(b, s, s, h, layout=layout)
    qg, kg, vg = (t.requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out, lse = flash_attn_func(qg, kg, vg, softmax_scale=MLA_SCALE, causal=True, return_lse=True)
    out.backward(dout)

    fold = layout == "sbhd"
    kshape = (1, s, b * h, D_QK) if fold else (b, s, h, D_QK)
    assert calls[0] == ("fwd", kshape, True, q.data_ptr(), MLA_SCALE, True)
    assert calls[1] == ("bwd", kshape[:3] + (D_V,), True, (kshape[0], kshape[2], s))
    assert out.shape == (b, s, h, D_V) and lse.shape == (b, h, s)
    for x in (out, qg.grad, kg.grad, vg.grad):
        assert x.transpose(0, 1).is_contiguous() == fold
    ref = _mla_reference(q, k, v, dout, True, MLA_SCALE)
    for name, x, r in zip(("out", "lse", "dq", "dk", "dv"), (out, lse, qg.grad, kg.grad, vg.grad), ref):
        assert compute_snr(r, x.detach().float()) > 40, name


def _run_mla(q, k, v, dout, causal, scale=MLA_SCALE):
    q, k, v = (t.to("cuda").requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out, lse = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)
    out.backward(dout.to("cuda"))
    torch.cuda.synchronize()
    return out, lse, q.grad, k.grad, v.grad


# Toy shapes: the gfx1250 numeric tests are meant to run one test per process (the card rule for
# new kernels); `pytest -k mla_matches` selects them.
@needs_gfx1250
@pytest.mark.parametrize(
    ("shape", "layout", "causal"),
    [
        ((1, 256, 256, 2), "sbhd", True),
        ((2, 256, 256, 4), "sbhd", True),
        ((2, 256, 256, 4), "bshd", True),
        ((2, 256, 256, 2), "bhsd", True),
        ((1, 256, 512, 2), "sbhd", True),
    ],
    ids=["toy", "b2_sbhd_folded", "b2_bshd", "b2_bhsd_copied", "sq_lt_skv"],
)
def test_mla_matches_reference(shape, layout, causal):
    q, k, v, dout = _mla_inputs(*shape, layout=layout)
    _check(_run_mla(q, k, v, dout, causal), _mla_reference(q, k, v, dout, causal, MLA_SCALE), min_db=48.0)


@needs_gfx1250
def test_mla_folded_equals_unfolded_and_is_deterministic():
    """Folding the batch into the heads relabels work, it does not change arithmetic: sbhd and
    bshd storage of the same values give bitwise-equal results, and so does a rerun."""
    q, k, v, dout = _mla_inputs(2, 256, 256, 4, layout="sbhd")
    folded = _run_mla(q, k, v, dout, True)
    again = _run_mla(q, k, v, dout, True)
    plain = _run_mla(*(t.contiguous() for t in (q, k, v, dout)), True)
    for a, b_, c in zip(folded, again, plain):
        assert torch.equal(a, b_) and torch.equal(a, c)


@needs_mla_flydsl
def test_mla_fwd_jit_key_covers_module_knobs(monkeypatch):
    """FlyDSL keys its compile cache by function source and closure scalars, not module globals.
    The MLA forward routes its module-level knobs through the kernels' closures, so a build with
    another value of one (here RESCALE_THRESHOLD) never reuses a cached binary."""
    from flydsl.compiler import jit_function

    from primus_turbo.flydsl.attention.gfx1250_mla_fwd import (
        fmha_fwd_prefill_a16w16_m32x8 as kern,
    )

    def key():
        monkeypatch.setattr(kern, "_launch_fns", {})
        kern.build_fmha_fwd_prefill_a16w16_m32x8.cache_clear()
        kern._ensure_bshd_kernel(False, True, True, False, 1, qk_hdim=D_QK, dtype_str="bf16")
        (fn,) = kern._launch_fns.values()
        return jit_function._jit_function_cache_key(fn.func)

    try:
        base = key()
        monkeypatch.setattr(kern, "RESCALE_THRESHOLD", kern.RESCALE_THRESHOLD + 1.0)
        assert key() != base
    finally:
        monkeypatch.undo()
        kern.build_fmha_fwd_prefill_a16w16_m32x8.cache_clear()


def _mla_bwd_impl():
    """The MLA backward host module, or a skip where its flydsl 0.3.4 kernels cannot be imported."""
    try:
        from primus_turbo.flydsl.attention.gfx1250_mla_bwd import impl
    except ImportError as exc:  # flydsl missing or another version (the package says which)
        pytest.skip(f"gfx1250 MLA backward not importable: {exc}")
    return impl


def test_mla_bwd_jit_key_covers_module_knobs():
    """FlyDSL keys a launch by the sources of the launcher and its kernels, their closure scalars and
    the module globals a static walk of their top-level code finds. The MLA backward's kernels read
    most module constants inside nested helpers, which that walk misses (DQ_BARRIER_FENCE, KH0_SALU,
    TDM_DEPTH, ...), so every kernel reads MODULE_KNOBS, the value of every knob of kernels.py. A copy
    of kernels.py with one knob edited gets a new key for every launcher; an unedited copy the same
    key, so the persistent cache still hits across processes."""
    from flydsl.compiler import jit_function

    kern = _mla_bwd_impl()._k
    assert kern.MODULE_KNOBS == kern._module_knobs()
    names = {n for n, _ in kern.MODULE_KNOBS}
    assert {"TDM_DEPTH", "KH0_SALU", "DQ_BARRIER_FENCE", "DQT_VT_SGB", "HEAD_GROUP", "D_QK"} <= names
    launchers = ("launch_delta", "launch_dkdv", "launch_dkdv64", "launch_dqg", "launch_dqg96")
    src = pathlib.Path(kern.__file__).read_text()

    def keys(mod):
        # the two parts of FlyDSL's key that depend on the module: per-function (source, closure
        # scalars) and the snapshot of the discovered module globals
        out = {}
        for name in launchers:
            fn = getattr(mod, name).func
            refs = jit_function._snapshot_refs(jit_function._discover_global_refs(fn), stable=True)
            out[name] = (jit_function._jit_function_cache_key(fn), tuple(sorted(refs.items())))
        return out

    def copy(old="", new=""):
        assert not old or src.count(old) == 1, old
        mod = types.ModuleType(kern.__name__)
        mod.__file__, mod.__package__ = kern.__file__, kern.__package__
        exec(compile(src.replace(old, new) if old else src, kern.__file__, "exec"), mod.__dict__)
        return mod

    base = keys(kern)
    assert keys(copy()) == base
    for old, new in (
        ("\nDQ_BARRIER_FENCE = True\n", "\nDQ_BARRIER_FENCE = False\n"),
        ("\nKH0_SALU = 2\n", "\nKH0_SALU = 1\n"),
    ):
        edited = keys(copy(old, new))
        assert [n for n in launchers if edited[n] == base[n]] == [], old.strip()


@pytest.mark.parametrize(
    ("dims", "expect"),
    [
        ((1, 256, 256, 2, 2), (True, True)),
        ((1, 1024, 1024, 8, 8), (True, True)),
        ((1, 2048, 2048, 8, 8), (True, True)),
        ((1, 2048, 2048, 16, 16), (False, True)),
        ((1, 2048, 2048, 32, 32), (False, False)),
        ((2, 4096, 4096, 128, 128), (False, False)),
        ((1, 4096, 4096, 256, 256), (False, False)),
    ],
    ids=["toy", "b1s1k_h8", "b1s2k_h8", "b1s2k_h16", "b1s2k_h32", "dsv3_mbs2", "dsv3_mbs2_folded"],
)
def test_mla_bwd_small_grid_launch_set(dims, expect):
    """Small grids launch the one-wave k_dkdv / one k_dqg over [0, Sq) per chain; the DeepSeek-V3
    training shapes keep the two-wave kernels. A fold [1, s, b*h, d] decides like [b, s, h, d]."""
    impl = _mla_bwd_impl()
    b, sq, skv, hq, hkv = dims
    geo = impl._geometry(b, sq, skv, hq, hkv)
    assert geo["small"] == expect
    assert impl._geometry(1, sq, skv, b * hq, b * hkv)["small"] == expect
    nw, nblk = geo["dkdv"]
    assert nw * nblk * impl._k.BLOCK_KV == skv and nw == (1 if expect[0] else impl._k.DKDV_NW)
    assert sum(ntile * 16 * nqw * nwave for nqw, nwave, _, ntile in geo["dq"]) == sq
    if expect[1]:
        assert geo["dq"] == [(impl._k.NQW, 1, 0, sq // impl._k.DQ_BQW)]
    for forced in (True, False):
        assert impl._geometry(b, sq, skv, hq, hkv, small_grid=forced)["small"] == (forced, forced)


# ---------------------------------------------------------------------------
# Packaging: lazy imports, the flydsl version gate, other archs
# ---------------------------------------------------------------------------


def test_importing_primus_turbo_does_not_import_the_gfx1250_kernels():
    """The gfx1250 kernel packages are imported on the first gfx1250 call, never by
    ``import primus_turbo``: they are wave32 code for one arch, and the MLA ones need flydsl 0.3.4.x."""
    assert _GFX1250_KERNEL_MODULES_AT_IMPORT == []


def _fake_flydsl(monkeypatch, version):
    """Make ``import flydsl`` fail (version None) or find a flydsl reporting ``version``."""
    if version is None:
        monkeypatch.setitem(sys.modules, "flydsl", None)
    else:
        fake = types.ModuleType("flydsl")
        fake.__version__ = version
        monkeypatch.setitem(sys.modules, "flydsl", fake)


@pytest.mark.parametrize("packaging", [True, False], ids=["packaging", "no_packaging"])
@pytest.mark.parametrize(
    ("version", "ok"),
    [
        ("0.3.4", True),
        ("0.3.4.1", True),
        ("0.3.4.1+g1234567", True),
        ("0.3.4.post1", True),
        (None, False),
        ("0.2.4", False),
        ("0.3.3", False),
        ("0.3.5", False),
        ("0.4.0", False),
        ("unknown", False),
    ],
)
def test_mla_flydsl_version_requirement(monkeypatch, version, ok, packaging):
    from primus_turbo.flydsl.attention import gfx1250_mla_version as ver

    if not packaging:  # the release-number fallback
        monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
        monkeypatch.setattr(ver, "_version_ok", ver._satisfies)  # uncached
    _fake_flydsl(monkeypatch, version)
    reason = ver.flydsl_unavailable_reason()
    assert (reason is None) == ok, reason
    if not ok:
        assert f"flydsl{ver.FLYDSL_REQUIREMENT}" in reason
        with pytest.raises(ImportError):
            ver.require_flydsl()


@pytest.mark.parametrize("version", ["0.3.4.dev1", "0.3.5rc1"])
def test_mla_flydsl_version_requirement_excludes_prereleases_outside(monkeypatch, version):
    from primus_turbo.flydsl.attention import gfx1250_mla_version as ver

    _fake_flydsl(monkeypatch, version)
    assert ver.flydsl_unavailable_reason() is not None


@pytest.mark.parametrize("version", [None, "0.2.4", "0.3.5"], ids=["missing", "0.2.4", "0.3.5"])
def test_mla_gate_falls_back_without_flydsl_0_3_4(monkeypatch, version):
    """With flydsl missing or another release the MLA gate gives the reason before importing any
    kernel module, FlyDSL declines the call and the dispatcher falls back to another backend."""
    from primus_turbo.pytorch.kernels.attention import (
        attention_flydsl_gfx1250_mla_impl as mla_impl,
    )

    def forbidden():
        raise AssertionError("gfx1250 MLA kernel module imported without flydsl 0.3.4.x")

    monkeypatch.setattr(mla_impl, "_bwd_kernels", forbidden)
    monkeypatch.setattr(mla_impl, "_fwd_interface", forbidden)
    _fake_flydsl(monkeypatch, version)
    q, k, v = _meta_mla(2, 4096, 4096, 128, 128)
    reason = mla_impl.flydsl_gfx1250_mla_unsupported_reason(q, k, v, True)
    assert reason is not None and "flydsl>=0.3.4,<0.3.5" in reason, reason
    assert _gate(monkeypatch, q, k, v, causal=True, softmax_scale=MLA_SCALE) is False
    assert pinned_backend_takes(BackendType.FLYDSL, **_resolve_kwargs(q, k, v)) is False
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v)) != BackendType.FLYDSL


@pytest.mark.parametrize("pkg", ["gfx1250_mla_fwd", "gfx1250_mla_bwd"])
def test_mla_packages_refuse_import_without_flydsl_0_3_4(monkeypatch, pkg):
    """Importing an MLA kernel package with another flydsl raises ImportError naming the requirement
    (no assert, no kernel module loaded), the error the gate and the tests' skips expect."""
    name = f"primus_turbo.flydsl.attention.{pkg}"
    for m in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, m)
    _fake_flydsl(monkeypatch, "0.2.4")
    with pytest.raises(ImportError, match=r"flydsl>=0\.3\.4,<0\.3\.5, found flydsl 0\.2\.4"):
        importlib.import_module(name + (".interface" if pkg.endswith("fwd") else ".impl"))
    assert [m for m in sys.modules if m == name or m.startswith(name + ".")] == []


@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
def test_mla_head_dims_route_as_before_on_gfx950(monkeypatch, layout):
    """On gfx950 the gfx1250 MLA gate, its flydsl check and its kernels are never consulted: MLA head
    dims go to another backend as before (the gfx950 FlyDSL kernels take head dims 64 / 128), and
    head dim 128 in sbhd storage still goes to the gfx950 FlyDSL kernels."""
    from primus_turbo.pytorch.kernels.attention import (
        attention_flydsl_gfx1250_mla_impl as mla_impl,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("gfx1250 MLA code consulted on gfx950")

    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 5))
    monkeypatch.setattr(attention_impl, "flydsl_gfx1250_mla_unsupported_reason", forbidden)
    for fn in ("_flydsl_unavailable_reason", "_bwd_kernels", "_fwd_interface"):
        monkeypatch.setattr(mla_impl, fn, forbidden)

    def tensors(d_qk, d_v):
        def make(d):
            if layout == "sbhd":
                return torch.empty(4096, 2, 128, d, dtype=torch.bfloat16, device="meta").permute(1, 0, 2, 3)
            return torch.empty(2, 4096, 128, d, dtype=torch.bfloat16, device="meta")

        return make(d_qk), make(d_qk), make(d_v)

    kwargs = dict(causal=True, window_size=(-1, -1), qkv_format=layout, needs_backward=True)
    q, k, v = tensors(D_QK, D_V)
    assert attention_impl.DenseAttnFwdFlydslBackend.can_handle(q, k=k, v=v, **kwargs) is False
    assert resolve_flash_attn_backend(False, None, q=q, k=k, v=v, **kwargs) != BackendType.FLYDSL
    q, k, v = tensors(D, D)
    got = resolve_flash_attn_backend(False, None, q=q, k=k, v=v, **kwargs)
    assert (got == BackendType.FLYDSL) == (layout == "sbhd"), got


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
def test_mla_torch_compile_traces_the_custom_ops(monkeypatch, layout, dynamic):
    """torch.compile(fullgraph) of FlashAttnFunc on MLA head dims, with meta tensors: the forward and
    backward AOT graphs each hold the MLA custom op exactly once, and its registered fake impls give
    the eager path's shapes, dtypes and (folded sbhd view) strides. No kernel runs."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    _mla_bwd_impl()  # the gate asks the backward's kernel constants
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    monkeypatch.setattr(flash_attn_interface, "is_gfx1250", lambda: True)
    b, s, h = 2, 256, 4
    seen = {"fw": [], "bw": []}

    def record(kind):
        def compiler(gm, example_inputs):
            seen[kind] += [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
            return make_boxed_func(gm.forward)

        return compiler

    def make(d):
        dims = (s, b, h, d) if layout == "sbhd" else (b, s, h, d)
        t = torch.empty(*dims, dtype=torch.bfloat16, device="meta")
        return (t.permute(1, 0, 2, 3) if layout == "sbhd" else t).requires_grad_()

    q, k, v = make(D_QK), make(D_QK), make(D_V)
    backend = aot_autograd(fw_compiler=record("fw"), bw_compiler=record("bw"))

    @torch.compile(fullgraph=True, dynamic=dynamic, backend=backend)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, softmax_scale=MLA_SCALE, causal=True, return_lse=True)

    torch._dynamo.reset()
    try:
        with _pinned_flydsl():
            out, lse = fn(q, k, v)
            (out.float().sum() + lse.sum()).backward()
    finally:
        torch._dynamo.reset()
    ops = {kind: [t for t in targets if t.startswith("primus_turbo.")] for kind, targets in seen.items()}
    assert ops == {
        "fw": ["primus_turbo.flash_attn_flydsl_gfx1250_mla_forward.default"],
        "bw": ["primus_turbo.flash_attn_flydsl_gfx1250_mla_backward.default"],
    }
    assert out.shape == (b, s, h, D_V) and out.dtype == torch.bfloat16
    assert lse.shape == (b, h, s) and lse.dtype == torch.float32 and lse.is_contiguous()
    for x, ref in ((out, None), (q.grad, q), (k.grad, k), (v.grad, v)):
        assert x.transpose(0, 1).is_contiguous() == (layout == "sbhd")
        if ref is not None:
            assert x.shape == ref.shape and x.dtype == ref.dtype


@needs_gfx1250
def test_mla_torch_compile_fullgraph():
    """The compiled MLA path on the card (b2 sbhd: the folded launch) equals the eager path bitwise
    and the CPU fp32 reference."""
    q, k, v, dout = _mla_inputs(2, 256, 256, 4, layout="sbhd")
    ref = _mla_reference(q, k, v, dout, True, MLA_SCALE)
    eager = _run_mla(q, k, v, dout, True)

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, softmax_scale=MLA_SCALE, causal=True, return_lse=True)

    q_, k_, v_ = (t.to("cuda").requires_grad_() for t in (q, k, v))
    torch._dynamo.reset()
    try:
        with _pinned_flydsl():
            out, lse = fn(q_, k_, v_)
        out.backward(dout.to("cuda"))
        torch.cuda.synchronize()
    finally:
        torch._dynamo.reset()
    got = (out, lse, q_.grad, k_.grad, v_.grad)
    _check(got, ref, min_db=48.0)
    for a, b_ in zip(got, eager):
        assert torch.equal(a, b_)


# Shapes the gate accepts beyond the full 256-row forward tiles: seqlen_q % 256 != 0 leaves whole
# forward waves past seqlen_q (empty Q TDM, skipped O stores, masked LSE lanes), the backward's dQ
# split takes its other forms (sq 64: q_split 64 and no k_dqg96 tile; 128: q_split 32 + one tile;
# 192: no head launch), and sq < skv shifts the causal diagonal. Each runs the backward's default
# launch set (one-wave kernels at these sizes) and the two-wave set.
@needs_gfx1250
@pytest.mark.parametrize("bwd_set", ["default", "two_wave"])
@pytest.mark.parametrize(
    ("shape", "layout"),
    [
        ((1, 64, 64, 2), "sbhd"),
        ((1, 128, 128, 2), "sbhd"),
        ((1, 192, 192, 2), "sbhd"),
        ((2, 64, 64, 2), "sbhd"),
        ((2, 64, 64, 2), "bshd"),
        ((1, 64, 128, 2), "sbhd"),
        ((1, 128, 384, 2), "sbhd"),
        ((2, 192, 448, 2), "sbhd"),
    ],
    ids=[
        "s64",
        "s128",
        "s192",
        "b2_s64_folded",
        "b2_s64_bshd",
        "sq64_skv128",
        "sq128_skv384",
        "b2_sq192_skv448_folded",
    ],
)
def test_mla_partial_tiles_match_reference(monkeypatch, shape, layout, bwd_set):
    if bwd_set == "two_wave":
        monkeypatch.setattr(_mla_bwd_impl(), "SMALL_GRID_WAVES", {"dkdv": 0, "dq": 0})
    q, k, v, dout = _mla_inputs(*shape, layout=layout)
    _check(_run_mla(q, k, v, dout, True), _mla_reference(q, k, v, dout, True, MLA_SCALE), min_db=48.0)
