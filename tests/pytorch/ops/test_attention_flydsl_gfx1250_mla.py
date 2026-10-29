###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL attention on gfx1250 (MI455X) at the DeepSeek-V3 MLA head dims (qk 192, v 128).

The gate, routing, launch and tracing tests patch the arch probe and use meta or CPU tensors, so
they run anywhere. The numeric tests need a gfx1250 card; their fp32 reference runs on the CPU.
"""

import contextlib
import hashlib
import importlib
import inspect
import re
import sys

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.attention import (
    attention_flydsl_gfx1250_mla_impl as mla_impl,
)
from primus_turbo.pytorch.kernels.attention import attention_impl
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    FlashAttnDenseDispatcher,
    resolve_flash_attn_backend,
)
from primus_turbo.pytorch.ops import flash_attn_func
from primus_turbo.pytorch.ops.attention import flash_attn_interface
from tests.pytorch.test_utils import compute_snr, pinned_backend_takes

_KERNELS = ("primus_turbo.flydsl.attention.gfx1250_mla", "primus_turbo.hipkittens.gfx1250_mla_bwd")
# Taken when pytest imports this module, before any test runs.
_KERNEL_MODULES_AT_IMPORT = sorted(
    m for m in sys.modules if any(m == p or m.startswith(p + ".") for p in _KERNELS)
)

needs_gfx1250 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 (MI455X) GPU"
)

D_QK, D_V = 192, 128
MLA_SCALE = 0.1352337788608801  # DeepSeek-V3: yarn mscale**2 / sqrt(192)


def _meta(b=1, sq=256, skv=256, h=2, hkv=None, d_qk=D_QK, d_v=D_V, dtype=torch.bfloat16, layout="bshd"):
    """[b, s, h, d]-shaped meta q, k, v whose bytes are in ``layout`` order."""

    def make(s, heads, d):
        if layout == "sbhd":
            return torch.empty(s, b, heads, d, dtype=dtype, device="meta").transpose(0, 1)
        if layout == "bhsd":
            return torch.empty(b, heads, s, d, dtype=dtype, device="meta").transpose(1, 2)
        return torch.empty(b, s, heads, d, dtype=dtype, device="meta")

    hkv = h if hkv is None else hkv
    return make(sq, h, d_qk), make(skv, hkv, d_qk), make(skv, hkv, d_v)


def _on_gfx1250(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    monkeypatch.setattr(flash_attn_interface, "is_gfx1250", lambda: True)


def _gate(monkeypatch, q, k, v, **kwargs):
    _on_gfx1250(monkeypatch)
    return attention_impl.DenseAttnFwdFlydslBackend.can_handle(q, k=k, v=v, **{"causal": True, **kwargs})


def _resolve_kwargs(q, k, v, qkv_format="bshd"):
    return dict(q=q, k=k, v=v, causal=True, window_size=(-1, -1), qkv_format=qkv_format, needs_backward=True)


@contextlib.contextmanager
def _pinned_flydsl():
    """Pin FLYDSL, so a refusal raises instead of running another backend."""
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    try:
        yield
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)


def _reference(q, k, v, dout, scale):
    """fp32 attention, lse and gradients on the CPU: MHA, bottom-right causal, [b, s, h, d]."""
    with torch.enable_grad():
        q, k, v = (t.detach().cpu().float().requires_grad_() for t in (q, k, v))
        sq, skv = q.shape[1], k.shape[1]
        s = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
        s = s.masked_fill(torch.ones(sq, skv, dtype=torch.bool).triu(skv - sq + 1), float("-inf"))
        lse = torch.logsumexp(s, -1)
        out = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v)
        out.backward(dout.cpu().float())
    return out.detach(), lse.detach(), q.grad, k.grad, v.grad


def _reference_by_heads(q, k, v, dout, scale, heads=4):
    """_reference in chunks of heads: one 4096 x 4096 fp32 score matrix is 64 MiB per head."""
    parts = [
        _reference(*(t[:, :, i : i + heads] for t in (q, k, v, dout)), scale)
        for i in range(0, q.shape[2], heads)
    ]
    return tuple(torch.cat([p[j] for p in parts], dim=1 if j == 1 else 2) for j in range(5))


def _inputs(b, sq, skv, h, layout="sbhd", seed=0):
    """bf16 CPU q, k, v, dout, [b, s, h, d]-shaped with bytes in ``layout`` order."""
    g = torch.Generator().manual_seed(seed)

    def make(s, d):
        if layout == "sbhd":
            return torch.randn(s, b, h, d, generator=g).to(torch.bfloat16).transpose(0, 1)
        if layout == "bhsd":
            return torch.randn(b, h, s, d, generator=g).to(torch.bfloat16).transpose(1, 2)
        return torch.randn(b, s, h, d, generator=g).to(torch.bfloat16)

    return make(sq, D_QK), make(skv, D_QK), make(skv, D_V), make(sq, D_V)


# ---------------------------------------------------------------------------
# Gate and dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["bshd", "sbhd", "bhsd"])
@pytest.mark.parametrize(
    ("dims", "kwargs"),
    [
        pytest.param((2, 4096, 4096, 128), {"softmax_scale": MLA_SCALE}, id="dsv3_b2"),
        pytest.param((1, 4096, 4096, 128), {"softmax_scale": MLA_SCALE}, id="dsv3_b1"),
        pytest.param((1, 256, 512, 2), {}, id="sq_lt_skv"),
        pytest.param((1, 64, 64, 1), {"window_size": (-1, 0)}, id="smallest_causal_as_window"),
    ],
)
def test_gate_accepts(monkeypatch, dims, kwargs, layout):
    q, k, v = _meta(*dims, layout=layout)
    assert _gate(monkeypatch, q, k, v, qkv_format=layout, **kwargs) is True


@pytest.mark.parametrize(
    ("tensors", "kwargs"),
    [
        pytest.param({"dtype": torch.float16}, {}, id="fp16"),
        pytest.param({"d_v": 192}, {}, id="v_192"),
        pytest.param({"d_qk": 128, "d_v": 128}, {}, id="head_dim_128"),
        pytest.param({"hkv": 1}, {}, id="gqa"),
        pytest.param({"sq": 96, "skv": 96}, {}, id="sq_not_multiple_of_64"),
        pytest.param({"skv": 288}, {}, id="skv_not_multiple_of_64"),
        pytest.param({"sq": 512}, {}, id="sq_gt_skv"),
        pytest.param({"sq": 32768, "skv": 32768, "h": 128}, {}, id="too_large"),
        pytest.param({"b": 0}, {}, id="empty_batch"),
        pytest.param({"h": 0}, {}, id="no_heads"),
        pytest.param({}, {"causal": False}, id="non_causal"),
        pytest.param({}, {"causal": False, "window_size": (-1, 0)}, id="window_without_causal"),
        pytest.param({}, {"window_size": (128, 0)}, id="sliding_window"),
        pytest.param({}, {"window_size": (-1, 64)}, id="right_window"),
        pytest.param({}, {"softmax_scale": True}, id="bool_scale"),
        pytest.param({}, {"softmax_scale": torch.tensor(0.1)}, id="tensor_scale"),
        pytest.param({}, {"dropout_p": 0.1}, id="dropout"),
        pytest.param({}, {"bias": torch.empty(1)}, id="bias"),
        pytest.param({}, {"alibi_slopes": torch.empty(2)}, id="alibi"),
        pytest.param({}, {"sink": torch.empty(2)}, id="sink"),
        pytest.param({}, {"return_softmax": True}, id="return_softmax"),
    ],
)
def test_gate_refuses(monkeypatch, tensors, kwargs):
    q, k, v = _meta(**tensors)
    assert _gate(monkeypatch, q, k, v, **kwargs) is False


def test_gate_refuses_mismatched_tensors(monkeypatch):
    q, k, v = _meta()
    _, k2, v2 = _meta(b=2)
    assert _gate(monkeypatch, q, k2, v2) is False  # batch
    assert _gate(monkeypatch, q, k, _meta(skv=512)[2]) is False  # v seqlen != k seqlen
    assert _gate(monkeypatch, q, k.to(torch.float16), v) is False  # mixed dtypes
    assert _gate(monkeypatch, q[0], k[0], v[0]) is False  # 3-D
    assert _gate(monkeypatch, q, torch.empty(k.shape, dtype=k.dtype), v) is False  # device
    assert _gate(monkeypatch, q, None, v) is False and _gate(monkeypatch, q, k, None) is False


@pytest.mark.parametrize("layout", ["bshd", "sbhd"])
def test_unpinned_mla_call_resolves_to_flydsl(monkeypatch, layout):
    _on_gfx1250(monkeypatch)
    q, k, v = _meta(2, 4096, 4096, 128, layout=layout)
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v, layout)) == BackendType.FLYDSL


@pytest.mark.parametrize(
    "tensors",
    [pytest.param({"sq": 96, "skv": 96}, id="sq_96"), pytest.param({"d_qk": 128, "d_v": 128}, id="hd128")],
)
def test_pinned_flydsl_refuses_instead_of_falling_back(monkeypatch, tensors):
    _on_gfx1250(monkeypatch)
    q, k, v = _meta(**tensors)
    assert pinned_backend_takes(BackendType.FLYDSL, **_resolve_kwargs(q, k, v)) is False


@pytest.mark.parametrize(
    ("tensors", "kwargs"),
    [
        pytest.param({}, {"dropout_p": 0.1}, id="dropout"),
        pytest.param({}, {"window_size": (128, 0)}, id="sliding_window"),
        pytest.param({}, {"return_attn_probs": True}, id="return_attn_probs"),
        pytest.param({}, {"softmax_scale": True}, id="bool_scale"),
        pytest.param({"d_qk": 128, "d_v": 128}, {}, id="hd128"),
        pytest.param({"skv": 288}, {}, id="skv_not_multiple_of_64"),
    ],
)
def test_flash_attn_func_refuses_without_the_gate(monkeypatch, tensors, kwargs):
    """FlashAttnFunc.apply is reachable without the gate, so its gfx1250 branch checks again."""
    _on_gfx1250(monkeypatch)
    monkeypatch.setattr(flash_attn_interface, "resolve_flash_attn_backend", lambda **kw: BackendType.FLYDSL)
    q, k, v = _meta(**tensors)
    with pytest.raises(ValueError, match="gfx1250 flydsl attention"):
        flash_attn_func(q, k, v, **{"causal": True, **kwargs})


def test_flash_attn_func_refuses_mixed_devices_without_the_gate(monkeypatch):
    """A launch must never see another device's pointers, so apply() checks the devices too."""
    _on_gfx1250(monkeypatch)
    monkeypatch.setattr(flash_attn_interface, "resolve_flash_attn_backend", lambda **kw: BackendType.FLYDSL)
    q, k, v = _meta()
    with pytest.raises(ValueError, match="gfx1250 flydsl attention: q/k/v must be on one device"):
        flash_attn_func(torch.empty(q.shape, dtype=q.dtype), k, v, causal=True)


@pytest.mark.parametrize(
    ("entry", "change", "match"),
    [
        pytest.param("fwd", {"causal": False}, "causal attention only", id="fwd_non_causal"),
        pytest.param(
            "fwd", {"k": torch.empty(1, 256, 2, D_QK, dtype=torch.bfloat16)}, "one device", id="fwd_devices"
        ),
        pytest.param("fwd", {"b": 0}, "must be positive", id="fwd_empty_batch"),
        pytest.param("bwd", {"causal": False}, "causal attention only", id="bwd_non_causal"),
        pytest.param("bwd", {"lse": torch.empty(1, 2, 256)}, "one device", id="bwd_devices"),
        pytest.param("bwd", {"b": 0}, "must be positive", id="bwd_empty_batch"),
        pytest.param(
            "bwd", {"lse": torch.empty(1, 256, 2, device="meta")}, "do not match", id="bwd_lse_layout"
        ),
        pytest.param(
            "bwd", {"dout": torch.empty(1, 256, 2, D_V, device="meta")}, "dout must be", id="bwd_fp32_dout"
        ),
    ],
)
def test_kernel_entries_raise_value_error(entry, change, match):
    """The kernel package's entries refuse what their kernels cannot run with ValueError, which
    python -O keeps, before anything is allocated or launched."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd, fwd

    change = dict(change)
    b, causal = change.pop("b", 1), change.pop("causal", True)
    q, k, v = _meta(b)
    out = torch.empty(b, 256, 2, D_V, dtype=torch.bfloat16, device="meta")
    args = dict(dout=out, q=q, k=k, v=v, out=out, lse=torch.empty(b, 2, 256, device="meta"))
    args.update(change)
    with pytest.raises(ValueError, match=match):
        if entry == "fwd":
            fwd.flash_attn_mla_fwd(args["q"], args["k"], args["v"], MLA_SCALE, causal=causal)
        else:
            bwd.flash_attn_mla_bwd(**args, softmax_scale=MLA_SCALE, causal=causal)


@pytest.mark.parametrize("layout", ["bshd", "sbhd"])
def test_mla_head_dims_route_as_before_on_gfx950(monkeypatch, layout):
    """On gfx950 the MLA gate and kernels are never consulted: MLA head dims go to aiter, and head
    dim 128 in SBHD storage still goes to the gfx950 FlyDSL kernels."""

    def forbidden(*args, **kwargs):
        raise AssertionError("gfx1250 MLA code consulted on gfx950")

    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 5))
    monkeypatch.setattr(attention_impl, "flydsl_gfx1250_mla_unsupported_reason", forbidden)
    monkeypatch.setattr(mla_impl, "_bwd", forbidden)
    q, k, v = _meta(2, 4096, 4096, 128, layout=layout)
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v, layout)) == BackendType.AITER
    q, k, v = _meta(2, 4096, 4096, 128, d_qk=128, d_v=128, layout=layout)
    got = resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v, layout))
    assert got == (BackendType.FLYDSL if layout == "sbhd" else BackendType.AITER)


def test_gfx950_flydsl_kernels_are_not_offered_to_gfx1250(monkeypatch):
    """gfx1250's capability compares greater than gfx950's; the gfx950 gate tests for equality."""
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (12, 5))
    q, k, _ = (t.reshape(-1, 2, 128) for t in _meta(2, 256, 256, 2, d_qk=128, d_v=128))
    cu = torch.tensor([0, 256, 512], dtype=torch.int32)
    varlen = attention_impl.VarlenAttnFwdFlydslBackend
    assert varlen.can_handle(q, k=k, cu_seqlens_q=cu, cu_seqlens_k=cu, causal=True) is False


def test_autotune_key_tells_eligibility_apart():
    """A tune-cache hit replays a backend without can_handle, so what FlyDSL's gate reads is keyed."""
    make_key = FlashAttnDenseDispatcher.make_key
    q, k, v = _meta(2, 4096, 4096, 128)
    base = dict(q=q, k=k, v=v, causal=True, window_size=(-1, -1), qkv_format="bshd")
    _, k_long, v_long = _meta(2, 4096, 8192, 128)
    variants = [
        dict(base, v=_meta(2, 4096, 4096, 128, d_v=192)[2]),
        dict(base, k=k_long, v=v_long),
        dict(base, dropout_p=0.1),
        dict(base, bias=torch.empty(1)),
        dict(base, alibi_slopes=torch.empty(128)),
    ]
    assert make_key(**base) == make_key(**dict(base, **dict(zip("qkv", _meta(2, 4096, 4096, 128)))))
    assert all(make_key(**kwargs) != make_key(**base) for kwargs in variants)


def test_execute_runs_the_mla_forward(monkeypatch):
    """DenseAttnFwdFlydslBackend.execute, the autotune path, runs the MLA forward on gfx1250."""
    _on_gfx1250(monkeypatch)
    monkeypatch.setattr(mla_impl, "_forward_op", mla_impl._forward_op_fake)
    q, k, v = _meta(2, 256, 256, 4, layout="sbhd")
    backend = attention_impl.DenseAttnFwdFlydslBackend
    out, lse = backend.execute(q, k, v, MLA_SCALE, True, (-1, -1))
    assert out.shape == (2, 256, 4, D_V) and lse.shape == (2, 4, 256)
    assert backend.execute(q, k, v, MLA_SCALE, True, (-1, -1), return_lse=False).shape == out.shape


@pytest.mark.parametrize("scale", [None, MLA_SCALE])
@pytest.mark.parametrize("layout", ["sbhd", "bshd", "bhsd"])
def test_flash_attn_func_runs_the_mla_kernels(monkeypatch, layout, scale):
    """b > 1 SBHD storage reaches the kernels folded to [1, s, b*h, d] and BSHD storage as it is,
    both without a copy; other byte orders are copied. Results come back [b, s, h, d] (SBHD views
    when folded), the default scale is 192**-0.5, and the gfx950 kernels are never reached. The
    kernels are replaced by the CPU reference on the kernel layout."""
    _on_gfx1250(monkeypatch)
    calls = []

    def fwd(q, k, v, softmax_scale, causal):
        calls.append(("fwd", tuple(q.shape), q.is_contiguous(), q.data_ptr(), softmax_scale, causal))
        out, lse, *_ = _reference(q, k, v, torch.zeros(*q.shape[:3], D_V), softmax_scale)
        return out.to(q.dtype).contiguous(), lse.contiguous()  # as the kernels return them

    def bwd(dout, q, k, v, out, lse, softmax_scale, causal):
        calls.append(
            ("bwd", tuple(dout.shape), dout.is_contiguous(), dout.data_ptr(), tuple(lse.shape), softmax_scale)
        )
        return tuple(g.to(q.dtype).contiguous() for g in _reference(q, k, v, dout, softmax_scale)[2:])

    def forbidden(*args, **kwargs):
        raise AssertionError("gfx950 FlyDSL kernels reached on gfx1250")

    monkeypatch.setattr(mla_impl, "_forward_op", fwd)
    monkeypatch.setattr(mla_impl, "_backward_op", bwd)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_sbhd_flydsl_forward_impl", forbidden)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_sbhd_flydsl_backward_impl", forbidden)

    b, s, h = 2, 128, 4
    q, k, v, dout = _inputs(b, s, s, h, layout=layout)
    for t in (q, k, v):
        t.requires_grad_()
    with _pinned_flydsl():
        out, lse = flash_attn_func(q, k, v, softmax_scale=scale, causal=True, return_lse=True)
    out.backward(dout)

    used = D_QK**-0.5 if scale is None else scale
    fold = layout == "sbhd"
    kshape = (1, s, b * h) if fold else (b, s, h)
    q_ptr, dout_ptr = calls[0][3], calls[1][3]
    assert calls == [
        ("fwd", kshape + (D_QK,), True, q_ptr, used, True),
        ("bwd", kshape + (D_V,), True, dout_ptr, (kshape[0], kshape[2], s), used),
    ]
    copied = layout == "bhsd"
    assert (q_ptr == q.data_ptr(), dout_ptr == dout.data_ptr()) == (not copied, not copied)
    for x in (out, q.grad, k.grad, v.grad):
        assert x.transpose(0, 1).is_contiguous() == fold
    ref = _reference(q, k, v, dout, used)
    for name, x, r in zip(("out", "lse", "dq", "dk", "dv"), (out, lse, q.grad, k.grad, v.grad), ref):
        assert x.shape == r.shape and compute_snr(r, x.detach()) > 40, name


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
def test_torch_compile_traces_the_custom_ops(monkeypatch, layout, dynamic):
    """torch.compile(fullgraph=True) on meta tensors: the forward and backward graphs each hold one
    MLA custom op, and its fake impls give the eager shapes, dtypes and strides."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    _on_gfx1250(monkeypatch)
    seen = {"fw": [], "bw": []}

    def record(kind):
        def compiler(gm, example_inputs):
            seen[kind] += [str(n.target) for n in gm.graph.nodes if str(n.target).startswith("primus_turbo.")]
            return make_boxed_func(gm.forward)

        return compiler

    b, s, h = 2, 256, 4
    q, k, v = (t.requires_grad_() for t in _meta(b, s, s, h, layout=layout))
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
    assert seen == {
        "fw": ["primus_turbo.flash_attn_flydsl_gfx1250_mla_forward.default"],
        "bw": ["primus_turbo.flash_attn_flydsl_gfx1250_mla_backward.default"],
    }
    assert out.shape == (b, s, h, D_V) and out.dtype == torch.bfloat16
    assert lse.shape == (b, h, s) and lse.dtype == torch.float32
    for x in (out, q.grad, k.grad, v.grad):
        assert x.transpose(0, 1).is_contiguous() == (layout == "sbhd")


@pytest.mark.parametrize(
    ("dims", "waves", "launchers"),
    [
        pytest.param((1, 256, 256, 2), None, ["delta", "dkdv", "dqg"], id="b1_s256_h2"),
        pytest.param(
            (1, 256, 256, 2),
            {"dkdv": 0, "dq": 0},
            ["delta", "dkdv64", "dqg96", "dqg"],
            id="b1_s256_h2_two_wave",
        ),
        pytest.param((1, 2048, 2048, 16), None, ["delta", "dkdv64", "dqg"], id="b1_s2048_h16"),
        pytest.param((1, 2048, 2048, 32), None, ["delta", "dkdv64", "dqg96", "dqg"], id="b1_s2048_h32"),
        pytest.param((2, 4096, 4096, 128), None, ["delta", "dkdv64", "dqg96", "dqg"], id="dsv3_b2"),
        pytest.param((1, 4096, 4096, 256), None, ["delta", "dkdv64", "dqg96", "dqg"], id="dsv3_b2_folded"),
    ],
)
def test_backward_launch_sets(monkeypatch, dims, waves, launchers):
    """Small grids launch the one-wave dK/dV kernel and one dQ kernel over all queries, the
    DeepSeek-V3 shapes the two-wave kernels; the numeric tests force the latter as here."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd, bwd_kernels

    if waves is not None:
        monkeypatch.setattr(bwd, "SMALL_GRID_WAVES", waves)
    b, sq, skv, h = dims
    q, k, v = _meta(b, sq, skv, h)
    out = torch.empty(b, sq, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, sq, device="meta")
    launches, _ = bwd._launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
    assert [fn for fn, _ in launches] == [getattr(bwd_kernels, f"launch_{n}") for n in launchers]


@pytest.mark.parametrize("waves", [None, {"dkdv": 0, "dq": 0}], ids=["default", "two_wave"])
@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((1, 64, 64, 1), id="b1_s64_h1"),
        pytest.param((1, 192, 448, 2), id="b1_s192_skv448_h2"),
        pytest.param((1, 2048, 2048, 32), id="b1_s2048_h32"),
        pytest.param((2, 4096, 4096, 128), id="dsv3_b2"),
        pytest.param((1, 4096, 4096, 256), id="dsv3_b2_folded"),
        pytest.param((1, 512, 512, 96), id="b1_s512_h96"),
    ],
)
def test_backward_launches_cover_every_row_and_head(monkeypatch, dims, waves):
    """The dK/dV grids cover every kv row and head, and the dQ tiles partition the queries."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_kernels as K

    if waves is not None:
        monkeypatch.setattr(bwd, "SMALL_GRID_WAVES", waves)
    b, sq, skv, h = dims
    q, k, v = _meta(b, sq, skv, h)
    out = torch.empty(b, sq, h, D_V, dtype=torch.bfloat16, device="meta")
    launches, _ = bwd._launches(
        out, q, k, v, out, torch.empty(b, h, sq, device="meta"), MLA_SCALE, True, None
    )
    tiles = []
    for fn, args in launches[1:]:
        a = inspect.signature(fn.func).bind(*args).arguments
        if fn is K.launch_dkdv:
            assert (a["nblk"] * K.BLOCK_KV, a["nhkv"] * a["nb"]) == (skv, b * h)
        elif fn is K.launch_dkdv64:
            assert (a["nblk"] * K.BLOCK_KV * K.DKDV_NW, a["hg"] * a["ngz"]) == (skv, b * h)
        else:
            assert a["hg"] * a["ngz"] == b * h
            tiles.append(
                (a["q_off"], a["q_off"] + a["ntile"] * (K.DQ_BQW96 if fn is K.launch_dqg96 else K.DQ_BQW))
            )
    tiles.sort()
    assert tiles[0][0] == 0 and tiles[-1][1] == sq and all(x[1] == y[0] for x, y in zip(tiles, tiles[1:]))


def test_importing_primus_turbo_does_not_import_the_kernels():
    assert _KERNEL_MODULES_AT_IMPORT == []


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _nan_filled_outputs():
    """Buffers the kernels' host code allocates start as NaN, so an element no launch writes fails
    the isfinite check instead of passing on what an earlier test left in the same block."""

    def nan(alloc):
        def fn(*args, **kwargs):
            t = alloc(*args, **kwargs)
            return t.fill_(float("nan")) if t.is_floating_point() else t

        return fn

    with pytest.MonkeyPatch.context() as mp:
        for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
            mp.setattr(owner, name, nan(getattr(owner, name)))
        yield


def _run(q, k, v, dout, scale=MLA_SCALE, attn=flash_attn_func):
    """Pinned FLYDSL forward and backward on the card; returns out, lse, dq, dk, dv."""
    q, k, v = (t.cuda().requires_grad_() for t in (q, k, v))
    dout = dout.cuda()
    with _pinned_flydsl(), _nan_filled_outputs():
        out, lse = attn(q, k, v, softmax_scale=scale, causal=True, return_lse=True)
        out.backward(dout)
    return out, lse, q.grad, k.grad, v.grad


def _check(got, ref):
    for name, x, r in zip(("out", "lse", "dq", "dk", "dv"), got, ref):
        x = x.detach().float().cpu()
        assert x.shape == r.shape and torch.isfinite(x).all(), name
        snr, floor = compute_snr(r, x), (100.0 if name == "lse" else 48.0)
        assert snr >= floor, f"{name}: {snr:.1f} dB < {floor} dB"


@needs_gfx1250
@pytest.mark.parametrize("bwd_launches", ["default", "two_wave"])
@pytest.mark.parametrize(
    ("dims", "layout", "scale"),
    [
        pytest.param((2, 128, 128, 128), "bshd", MLA_SCALE, id="head_groups"),
        pytest.param((2, 256, 256, 4), "sbhd", MLA_SCALE, id="b2_sbhd_folded"),
        pytest.param((2, 192, 448, 2), "sbhd", MLA_SCALE, id="b2_sbhd_folded_sq_lt_skv"),
        pytest.param((2, 128, 384, 2), "bshd", MLA_SCALE, id="b2_bshd_sq_lt_skv"),
        pytest.param((2, 256, 256, 2), "bhsd", MLA_SCALE, id="b2_bhsd_copied"),
        pytest.param((1, 64, 128, 2), "sbhd", MLA_SCALE, id="sq64_skv128"),
        pytest.param((1, 256, 256, 2), "bshd", None, id="default_scale"),
    ],
)
def test_matches_reference(monkeypatch, dims, layout, scale, bwd_launches):
    if bwd_launches == "two_wave":
        from primus_turbo.flydsl.attention.gfx1250_mla import bwd

        monkeypatch.setattr(bwd, "SMALL_GRID_WAVES", {"dkdv": 0, "dq": 0})
    q, k, v, dout = _inputs(*dims, layout=layout)
    ref = _reference(q, k, v, dout, D_QK**-0.5 if scale is None else scale)
    _check(_run(q, k, v, dout, scale), ref)


@needs_gfx1250
def test_production_dispatch_matches_reference():
    """(1, 4096, 4096, 16) takes the DeepSeek-V3 dispatch: grouped longest-first forward blocks,
    k_dkdv64, k_dqg96 from q_off 64 and k_dqg with the XCD head remap. A rerun is bitwise identical."""
    q, k, v, dout = _inputs(1, 4096, 4096, 16, layout="bshd")
    got = _run(q, k, v, dout)
    _check(got, _reference_by_heads(q, k, v, dout, MLA_SCALE))
    assert all(torch.equal(a, b) for a, b in zip(got, _run(q, k, v, dout)))


@needs_gfx1250
def test_folded_equals_unfolded_and_is_deterministic():
    """Folding the batch into the heads relabels work without changing arithmetic, and a repeated
    call is bitwise identical."""
    q, k, v, dout = _inputs(2, 256, 256, 4, layout="sbhd")
    folded = _run(q, k, v, dout)
    again = _run(q, k, v, dout)
    unfolded = _run(*(t.contiguous() for t in (q, k, v, dout)))
    for a, b, c in zip(folded, again, unfolded):
        assert torch.equal(a, b) and torch.equal(a, c)


@needs_gfx1250
def test_torch_compile_fullgraph():
    """The compiled path on the card traces the custom ops and equals eager bitwise."""
    q, k, v, dout = _inputs(2, 256, 256, 4, layout="sbhd")
    eager = _run(q, k, v, dout)
    torch._dynamo.reset()
    try:
        got = _run(q, k, v, dout, attn=torch.compile(flash_attn_func, fullgraph=True, backend="aot_eager"))
    finally:
        torch._dynamo.reset()
    _check(got, _reference(q, k, v, dout, MLA_SCALE))
    for a, b in zip(got, eager):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Backward on the HIP kernels (PRIMUS_TURBO_MLA_BWD_IMPL=hk, primus_turbo/hipkittens/gfx1250_mla_bwd)
# ---------------------------------------------------------------------------


def _hk_package(name):
    return importlib.import_module(f"primus_turbo.hipkittens.gfx1250_mla_bwd.{name}")


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("", "flydsl"), ("flydsl", "flydsl"), ("hk", "hk"), (" HK ", "hk"), ("hipkittens", None)],
    ids=["unset", "flydsl", "hk", "hk_spelled_loosely", "unknown"],
)
def test_mla_bwd_impl_setting(monkeypatch, setting, expected):
    monkeypatch.setattr(mla_impl, "MLA_BWD_IMPL_SETTING", setting)
    if expected is None:
        with pytest.raises(ValueError, match="PRIMUS_TURBO_MLA_BWD_IMPL"):
            mla_impl.mla_bwd_impl()
    else:
        assert mla_impl.mla_bwd_impl() == expected


@pytest.mark.parametrize(
    "dims",
    [(2, 4096, 4096, 128), (1, 4096, 4096, 256), (1, 256, 256, 2), (1, 256, 384, 2)],
    ids=["dsv3_mbs2", "dsv3_mbs2_folded", "toy", "rect"],
)
def test_hk_gate_accepts(monkeypatch, dims):
    monkeypatch.setattr(_hk_package("hsaco"), "unavailable_reason", lambda *a: None)
    assert mla_impl.hk_mla_bwd_unsupported_reason(*_meta(*dims), True) is None


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        ((1, 512, 512, 2), "seqlen_q % 384"),
        ((1, 128, 128, 2), "seqlen_q % 384"),
        ((1, 256, 16384, 256), "proven extent"),
        ((1, 256, 256, 4, 2), "MHA only"),
        ((1, 512, 256, 2), "seqlen_q <= seqlen_kv"),
    ],
    ids=["sq512", "sq128", "beyond_proven_extent", "gqa", "sq_gt_skv"],
)
def test_hk_gate_refuses(monkeypatch, dims, match):
    monkeypatch.setattr(_hk_package("hsaco"), "unavailable_reason", lambda *a: None)
    reason = mla_impl.hk_mla_bwd_unsupported_reason(*_meta(*dims), True)
    assert reason is not None and match in reason


def test_hk_gate_refuses_without_the_code_objects(monkeypatch):
    monkeypatch.setattr(
        _hk_package("hsaco"), "unavailable_reason", lambda *a: "k_dkdv96ds.hsaco is not built"
    )
    assert (
        mla_impl.hk_mla_bwd_unsupported_reason(*_meta(1, 256, 256, 2), True)
        == "k_dkdv96ds.hsaco is not built"
    )


def test_hk_code_object_checks(tmp_path):
    """unavailable_reason: a missing code object, then one with foreign bytes, are reasons."""
    hsaco = _hk_package("hsaco")
    assert "not built" in hsaco.unavailable_reason(tmp_path / "missing")
    bad = tmp_path / "bad"
    bad.mkdir()
    for name in hsaco.load_manifest()["kernels"]:
        (bad / f"{name}.hsaco").write_bytes(b"not a code object")
    assert "sha256" in hsaco.unavailable_reason(bad)


def test_hk_optional_code_object(monkeypatch, tmp_path):
    """k_delta is optional: with its code object missing or foreign the backward stays available
    (unavailable_reason None) and hiprt loads the other kernels; a foreign required one is a reason.
    A stand-in manifest pins placeholder bytes (no compiler, no GPU)."""
    hsaco, hiprt = _hk_package("hsaco"), _hk_package("hiprt")
    man = hsaco.load_manifest()
    blobs = {n: f"code object {n}".encode() for n in man["kernels"]}
    pins = {n: dict(k, hsaco_sha256=hashlib.sha256(blobs[n]).hexdigest()) for n, k in man["kernels"].items()}
    monkeypatch.setattr(hsaco, "_MANIFEST", dict(man, kernels=pins))
    monkeypatch.setattr(hsaco, "_REASONS", {})

    def bin_dir(name, k_delta=None, k_dqds_c=None):
        d = tmp_path / name
        d.mkdir()
        for n, data in blobs.items():
            data = {"k_delta": k_delta, "k_dqds_c": k_dqds_c}.get(n) or data
            if data != b"missing":
                (d / f"{n}.hsaco").write_bytes(data)
        return d

    full = bin_dir("full")
    assert hsaco.unavailable_reason(full) is None and hsaco.kernel_unavailable_reason("k_delta", full) is None
    assert set(hiprt.load_kernels(full)) == set(blobs)
    for name, data, why in (("foreign", b"other bytes", "sha256"), ("missing", b"missing", "not built")):
        d = bin_dir(name, k_delta=data)
        assert hsaco.unavailable_reason(d) is None and why in hsaco.kernel_unavailable_reason("k_delta", d)
        assert set(hiprt.load_kernels(d)) == {"k_dkdv96ds", "k_dqds_c"}
    d = bin_dir("required_foreign", k_dqds_c=b"other bytes")
    assert "k_dqds_c.hsaco sha256" in hsaco.unavailable_reason(d)
    with pytest.raises(hiprt.HipError, match="unavailable"):
        hiprt.load_kernels(d)


def test_hk_manifest():
    """Every kernel: its source is shipped, its arguments tile its kernarg segment in order, and the
    flags are the device-only gfx1250 build (no KITTENS_CDNA4)."""
    hsaco, hiprt = _hk_package("hsaco"), _hk_package("hiprt")
    m = hsaco.load_manifest()
    assert (
        m["arch"] == "gfx1250"
        and "--offload-arch=gfx1250" in m["flags"]
        and "--cuda-device-only" in m["flags"]
    )
    assert not any("KITTENS_CDNA4" in f for f in m["flags"])
    assert set(m["kernels"]) == {"k_dkdv96ds", "k_dqds_c", "k_delta"}
    assert [n for n, k in m["kernels"].items() if k.get("optional")] == ["k_delta"]
    for name, k in m["kernels"].items():
        assert (hsaco.HIP_DIR / k["src"]).is_file(), name
        assert len(k["hsaco_sha256"]) == 64 and k["block"] in (64, 128, 256)
        last = k["args"][-1]
        assert last["offset"] + last["size"] <= k["kernarg_size"], name
        values = [1 if a["kind"] == "ptr" else 0 for a in k["args"]]
        assert len(hiprt.pack_args(k["args"], values)) == last["offset"] + last["size"], name


_ASM_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$")
# A VGPR operand; above v255 the compiler adds the true register as a comment: v[26:29] /*v[794:797]*/
_ASM_VGPR = re.compile(r"v(\[\d+:\d+\]|\d+)(?:\s*/\*\s*v(\[\d+:\d+\]|\d+)\s*\*/)?")


def _asm_patch_hunks(patch):
    """[(old start, new start, [(tag, line)])] of a one-file unified diff, each hunk's counts checked."""
    hunks = []
    for block in patch.split("\n@@ ")[1:]:
        header, *body = ("@@ " + block).rstrip("\n").split("\n")
        m = _ASM_HUNK.match(header)
        assert m and all(line[:1] in (" ", "-", "+") for line in body), header
        ops = [(line[0], line[1:]) for line in body]
        assert sum(t != "+" for t, _ in ops) == int(m.group(2) or 1), header
        assert sum(t != "-" for t, _ in ops) == int(m.group(4) or 1), header
        hunks.append((int(m.group(1)), int(m.group(3)), ops))
    return hunks


def _asm_vgpr_shape(line):
    """The line with every VGPR operand replaced by its tuple size and 256-VGPR block."""

    def shape(m):
        lo, *hi = (int(n) for n in re.findall(r"\d+", m.group(2) or m.group(1)))
        return f"v<{(hi[0] if hi else lo) - lo + 1}@{lo // 256}>"

    return _ASM_VGPR.sub(shape, line)


def test_hk_asm_patches(tmp_path):
    """Every manifest asm_patch (tools/regen_asm_patch.py) is shipped and only renumbers VGPRs (same
    tuple sizes, same 256-VGPR blocks, so every s_set_vgpr_msb stays valid) and changes s_wait_alu lines.
    hsaco.py applies it to its own context only: it applies to an input rebuilt from its hunks, a
    changed context line is rejected, and a compiler .s with another sha256 is refused before anything
    is patched or assembled (no compiler needed)."""
    hsaco = _hk_package("hsaco")
    kernels = {n: k for n, k in hsaco.load_manifest()["kernels"].items() if "asm_patch" in k}
    assert set(kernels) == {"k_dkdv96ds"}
    for name, k in kernels.items():
        assert re.fullmatch(r"[0-9a-f]{64}", k["asm_input_sha256"]), name
        patch = (hsaco.HIP_DIR / k["asm_patch"]).read_text()
        src, want, shift, renumbered = [], [], 0, 0
        for old_start, new_start, ops in _asm_patch_hunks(patch):
            assert old_start > len(src) and new_start == old_start + shift, (name, old_start)
            fill = [f"\t; line {n + 1}" for n in range(len(src), old_start - 1)]
            src += fill + [t for tag, t in ops if tag != "+"]
            want += fill + [t for tag, t in ops if tag != "-"]
            shift += sum(t == "+" for t, _ in ops) - sum(t == "-" for t, _ in ops)
            old = [t for tag, t in ops if tag == "-" and "s_wait_alu" not in t]
            new = [t for tag, t in ops if tag == "+" and "s_wait_alu" not in t]
            assert len(old) == len(new), (name, old_start)
            for a, b in zip(old, new):
                assert a != b and _asm_vgpr_shape(a) == _asm_vgpr_shape(b), (name, a, b)
            renumbered += len(new)
        assert renumbered > 0, name
        text = "\n".join(src + ["\ts_endpgm"]) + "\n"
        assert hsaco._apply_unified_diff(text, patch) == "\n".join(want + ["\ts_endpgm"]) + "\n"
        first = _asm_patch_hunks(patch)[0][0] - 1  # the first hunk's first (context) line
        bad = "\n".join(src[:first] + [src[first] + " "] + src[first + 1 :] + ["\ts_endpgm"]) + "\n"
        with pytest.raises(ValueError, match="does not apply"):
            hsaco._apply_unified_diff(bad, patch)
        saved = tmp_path / f"{name}-hip-amdgpu12.50-amd-amdhsa-gfx1250.s"  # as -save-temps=obj names it
        saved.write_text(text)
        with pytest.raises(RuntimeError, match="asm_input_sha256"):
            hsaco._patch_and_reassemble(name, k, [], tmp_path)
        saved.write_text(bad)
        k_bad = dict(k, asm_input_sha256=hashlib.sha256(bad.encode()).hexdigest())
        with pytest.raises(RuntimeError, match="does not apply"):
            hsaco._patch_and_reassemble(name, k_bad, [], tmp_path)
        assert saved.read_text() == bad


def test_hk_plan_on_meta_tensors():
    """The launch plan at DeepSeek-V3's shape, on meta tensors (no GPU): three launches, the grids,
    the dS scratch size, and argument tuples that match the manifest's kernel signatures."""
    impl = _hk_package("impl")
    m = _hk_package("hsaco").load_manifest()
    b, s, h = 2, 4096, 128
    q, k, v = _meta(b, s, s, h)
    o = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, s, dtype=torch.float32, device="meta")
    plan = impl._plan(o, q, k, v, o, lse, MLA_SCALE, True, None)
    assert [x[0] for x in plan["launches"]] == ["delta", "dkdv", "dq"]
    assert plan["grids"]["dkdv"] == (32, 43, 8)  # 42 blocks of 96 kv rows + one 64-row tail
    assert plan["grids"]["dq"] == (32, 11, 8)  # 10 tiles of 384 queries + the 256-query head
    assert plan["ds_layout_bytes"] == 4_328_521_728
    assert plan["ds_full"].numel() * 2 == 4_328_521_728 + 2 * impl.DS_GUARD_B
    for _, name, (grid, block, values) in plan["launches"][1:]:
        assert block == m["kernels"][name]["block"] and len(values) == len(m["kernels"][name]["args"])
    with pytest.raises(ValueError, match="not proven"):
        impl._plan(o, q, k, v, o, lse, MLA_SCALE, True, None, head_group=48)


def test_hk_plan_delta_launch(monkeypatch):
    """The "delta" launch at DeepSeek-V3's shape: the HIP k_delta (64-query workgroups, the manifest's
    argument list) or the FlyDSL k_delta (32-query workgroups); by default the HIP one exactly when its
    code object is the pinned one. Both write the plan's delta / nl / nd (nl / nd feed dkdv)."""
    impl, hsaco = _hk_package("impl"), _hk_package("hsaco")
    args = hsaco.load_manifest()["kernels"]["k_delta"]["args"]
    b, s, h = 2, 4096, 128
    q, k, v = _meta(b, s, s, h)
    o = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, s, dtype=torch.float32, device="meta")
    hip = impl._plan(o, q, k, v, o, lse, MLA_SCALE, True, None, hk_delta=True)
    name, fn, (grid, block, values) = hip["launches"][0]
    assert (name, fn, grid, block) == ("delta", "k_delta", (64, 256, 1), 256) and hip["grids"][
        "delta"
    ] == grid
    assert [a["name"] for a in args] == ["DO", "O", "DEL", "LSE", "NL", "ND", "scale", "S", "H"]
    assert len(values) == len(args) and values[6:] == (MLA_SCALE, s, h)
    assert (values[0], values[1], values[3]) == (o, o, lse)
    assert (values[2], values[4], values[5]) == (hip["delta"], hip["nl"], hip["nd"])
    assert hip["launches"][1][2][2][4:6] == (hip["nl"], hip["nd"])  # what dkdv reads
    fly = impl._plan(o, q, k, v, o, lse, MLA_SCALE, True, None, hk_delta=False)
    assert fly["launches"][0][1] is impl._k.launch_delta and fly["grids"]["delta"] == (128, 256, 1)
    assert fly["launches"][0][2][4:6] == (fly["nl"], fly["nd"])
    for reason, want in ((None, "k_delta"), ("k_delta.hsaco is not built", impl._k.launch_delta)):
        monkeypatch.setattr(hsaco, "kernel_unavailable_reason", lambda name, *a, r=reason: r)
        assert impl._plan(o, q, k, v, o, lse, MLA_SCALE, True, None)["launches"][0][1] == want


@pytest.mark.parametrize(("s", "uses_hk"), [(256, True), (128, False)], ids=["hk", "falls_back"])
def test_hk_routes_and_falls_back(monkeypatch, s, uses_hk):
    """PRIMUS_TURBO_MLA_BWD_IMPL=hk replaces only the backward kernels (same kernel-layout tensors);
    a shape the HIP kernels refuse (seqlen 128) runs the FlyDSL backward. The kernels are replaced
    by the CPU reference."""
    _on_gfx1250(monkeypatch)
    monkeypatch.setattr(mla_impl, "MLA_BWD_IMPL_SETTING", "hk")
    monkeypatch.setattr(_hk_package("hsaco"), "unavailable_reason", lambda *a: None)
    calls = []

    def fwd(q, k, v, softmax_scale, causal):
        out, lse, *_ = _reference(q, k, v, torch.zeros(*q.shape[:3], D_V), softmax_scale)
        return out.to(q.dtype).contiguous(), lse.contiguous()

    def bwd(tag):
        def run(dout, q, k, v, out, lse, softmax_scale, causal):
            calls.append((tag, tuple(q.shape)))
            return tuple(g.to(q.dtype).contiguous() for g in _reference(q, k, v, dout, softmax_scale)[2:])

        return run

    monkeypatch.setattr(mla_impl, "_forward_op", fwd)
    monkeypatch.setattr(mla_impl, "_backward_op", bwd("flydsl"))
    monkeypatch.setattr(mla_impl, "_backward_hk_op", bwd("hk"))
    b, h = 2, 2
    q, k, v, dout = _inputs(b, s, s, h, layout="sbhd")
    qg, kg, vg = (t.clone().requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out = flash_attn_func(qg, kg, vg, softmax_scale=MLA_SCALE, causal=True)
    out.backward(dout)
    assert calls == [("hk" if uses_hk else "flydsl", (1, s, b * h, D_QK))]
    ref = _reference(q, k, v, dout, MLA_SCALE)
    for name, x, r in zip(("dq", "dk", "dv"), (qg.grad, kg.grad, vg.grad), ref[2:]):
        assert compute_snr(r, x.detach().float()) > 40, name


def _hk_code_objects_reason():
    return _hk_package("hsaco").unavailable_reason()


needs_hk = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()) or _hk_code_objects_reason() is not None,
    reason="needs a gfx1250 (MI455X) GPU and the HIP MLA backward code objects",
)


def _run_hk(monkeypatch, q, k, v, dout):
    monkeypatch.setattr(mla_impl, "MLA_BWD_IMPL_SETTING", "hk")
    calls = []
    real = mla_impl._backward_hk_op

    def spy(*args):
        calls.append(tuple(args[1].shape))
        return real(*args)

    monkeypatch.setattr(mla_impl, "_backward_hk_op", spy)
    got = _run(q, k, v, dout)
    assert len(calls) == 1, "the HIP backward was not used"
    return got


@needs_hk
@pytest.mark.parametrize(
    ("dims", "layout"),
    [
        ((1, 256, 256, 2), "sbhd"),
        ((2, 256, 256, 2), "sbhd"),
        ((1, 256, 384, 2), "sbhd"),
        ((1, 640, 640, 4), "bshd"),
    ],
    ids=["toy", "toy_folded", "rect", "sq640"],
)
def test_hk_matches_reference_and_flydsl(monkeypatch, dims, layout):
    """The HIP backward matches the fp32 reference, is bitwise deterministic and bitwise equal to
    the FlyDSL backward (same dS rounding and accumulation order)."""
    q, k, v, dout = _inputs(*dims, layout=layout)
    got = _run_hk(monkeypatch, q, k, v, dout)
    _check(got, _reference(q, k, v, dout, MLA_SCALE))
    again = _run_hk(monkeypatch, q, k, v, dout)
    monkeypatch.setattr(mla_impl, "MLA_BWD_IMPL_SETTING", "flydsl")
    fly = _run(q, k, v, dout)
    for a, b, c in zip(got, again, fly):
        assert torch.equal(a, b) and torch.equal(a, c)


@needs_hk
def test_hk_production_shape_matches_reference():
    """(1, 4096, 4096, 16): 42 96-row kv blocks + the 64-row tail, 10 384-query dQ tiles + the
    256-query head; with the dS scratch NaN-filled, so a block dQ reads but dK/dV never wrote fails."""
    impl = _hk_package("impl")
    q, k, v, dout = _inputs(1, 4096, 4096, 16, layout="bshd")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mla_impl, "MLA_BWD_IMPL_SETTING", "hk")
        mp.setattr(impl, "POISON", True)
        got = _run(q, k, v, dout)
    _check(got, _reference_by_heads(q, k, v, dout, MLA_SCALE))


@needs_hk
def test_hk_direct_rect32_matches_reference():
    """hk_attn_bwd called directly at (Skv - Sq) % 64 == 32 (the adapter's FlyDSL gate refuses it, the HIP
    kernels' own shape rules admit it): the dQ kernel's last steady unit reads the first masked kv32 row, whose
    stored pairs start mid-row there (5c20de7e .. 71bd0329 read the wrong half-block: dq ~18 dB)."""
    impl = _hk_package("impl")
    q, k, v, dout = _inputs(1, 4096, 4128, 4, layout="bshd")
    ref = _reference_by_heads(q, k, v, dout, MLA_SCALE)
    out, lse = ref[0].to(torch.bfloat16).cuda(), ref[1].float().cuda()
    args = [t.cuda().contiguous() for t in (dout, q, k, v)]
    got = impl.hk_attn_bwd(args[0], args[1], args[2], args[3], out, lse, MLA_SCALE, True)
    again = impl.hk_attn_bwd(args[0], args[1], args[2], args[3], out, lse, MLA_SCALE, True)
    for name, x, y, r in zip(("dq", "dk", "dv"), got, again, ref[2:]):
        assert torch.equal(x, y), name
        snr = compute_snr(r, x.float().cpu())
        assert snr >= 48.0, f"{name}: {snr:.1f} dB < 48 dB"


needs_hk_delta = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250())
    or _hk_code_objects_reason() is not None
    or _hk_package("hsaco").kernel_unavailable_reason("k_delta") is not None,
    reason="needs a gfx1250 (MI455X) GPU and the HIP MLA backward code objects, k_delta included",
)


def _delta_inputs(b, s, h, special, seed=0):
    """Kernel-layout dO / O [b, s, h, 128] bf16 and lse [b, h, s] fp32 on the card. special: every
    (row, head) of dO and of O scaled by 2^-k, k in [0, 80), so the bf16 x bf16 products range from
    normal to below the fp32 denormal range (where a fused multiply-add rounds differently from
    FlyDSL's multiply then add); 5% bf16 denormals, 2% signed zeros; lse with 5% fp32 denormals,
    1% signed zeros and 1% +-3e38 (NL overflows to +-inf)."""
    g = torch.Generator().manual_seed(seed)
    do, o = (torch.randn(b, s, h, D_V, generator=g) for _ in range(2))
    lse = torch.randn(b, h, s, generator=g) * 4 + 6
    if special:
        do, o = (t * torch.exp2(-torch.randint(0, 80, (b, s, h, 1), generator=g).float()) for t in (do, o))
    do, o = do.to(torch.bfloat16), o.to(torch.bfloat16)
    if special:

        def sign(shape, dtype, bit):
            return torch.randint(0, 2, shape, generator=g, dtype=dtype) * torch.tensor(bit, dtype=dtype)

        for t in (do, o):
            bits = t.view(torch.int16)
            den = torch.randint(1, 128, bits.shape, generator=g, dtype=torch.int16) | sign(
                bits.shape, torch.int16, -32768
            )
            pick = torch.rand(bits.shape, generator=g)
            bits[pick < 0.05] = den[pick < 0.05]
            zero = (pick >= 0.05) & (pick < 0.07)
            bits[zero] = sign(bits.shape, torch.int16, -32768)[zero]
        bits = lse.view(torch.int32)
        den = torch.randint(1, 1 << 23, bits.shape, generator=g, dtype=torch.int32) | sign(
            bits.shape, torch.int32, -(2**31)
        )
        pick = torch.rand(bits.shape, generator=g)
        bits[pick < 0.05] = den[pick < 0.05]
        zero = (pick >= 0.05) & (pick < 0.06)
        bits[zero] = sign(bits.shape, torch.int32, -(2**31))[zero]
        huge = pick >= 0.99
        lse[huge] = 3e38 * (1 - 2 * torch.randint(0, 2, lse.shape, generator=g).float())[huge]
    return do.cuda(), o.cuda(), lse.cuda()


def _delta_launches(do, o, lse, scale=MLA_SCALE):
    """{"hk": ..., "flydsl": ...}: (delta, NL, ND) of the plan's "delta" launch on the HIP and on the
    FlyDSL k_delta, the same inputs, outputs NaN-filled first."""
    impl = _hk_package("impl")
    b, s, h, _ = do.shape
    q = torch.empty(b, s, h, D_QK, dtype=torch.bfloat16, device="cuda")
    v = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()
    got = {}
    for key, hk in (("hk", True), ("flydsl", False)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(impl, "POISON", True)
            plan = impl._plan(do, q, q, v, o, lse, scale, True, stream, hk_delta=hk)
        _, fn, args = plan["launches"][0]
        if hk:
            grid, block, values = args
            impl.kernels()[fn].call(grid, block, values, stream)
        else:
            impl._fly._launch(fn, args)
        got[key] = (plan["delta"], plan["nl"], plan["nd"])
    torch.cuda.synchronize()
    return got


@needs_hk_delta
@pytest.mark.parametrize(
    ("dims", "special"),
    [
        ((1, 256, 2), False),
        ((2, 256, 4), False),
        ((1, 640, 4), False),
        ((1, 4096, 16), False),
        ((2, 256, 4), True),
        ((1, 640, 4), True),
    ],
    ids=["toy", "b2", "sq640", "sq4096", "b2_denormals", "sq640_denormals"],
)
def test_hk_delta_matches_flydsl(dims, special):
    """The HIP k_delta writes delta, NL and ND bitwise equal to the FlyDSL k_delta, also for denormal
    inputs, products that underflow fp32, signed zeros and an lse whose NL overflows."""
    got = _delta_launches(*_delta_inputs(*dims, special))
    for name, x, y in zip(("delta", "NL", "ND"), got["hk"], got["flydsl"]):
        assert not torch.isnan(x).any(), name
        assert torch.equal(x.view(torch.int32), y.view(torch.int32)), name
    if special:
        delta = got["hk"][0]
        tiny = (delta != 0) & (delta.abs() < torch.finfo(torch.float32).tiny)
        assert tiny.any(), "no denormal delta: the inputs do not reach the fp32 denormal range"


@needs_hk_delta
def test_hk_delta_falls_back_to_flydsl(monkeypatch):
    """Without the k_delta code object hk_attn_bwd runs the FlyDSL k_delta, and the gradients are
    bitwise the same."""
    impl = _hk_package("impl")
    q, k, v, dout = (t.cuda() for t in _inputs(1, 256, 256, 4, layout="bshd"))
    out, lse = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    hip = impl.hk_attn_bwd(dout, q, k, v, out, lse, MLA_SCALE, True)
    monkeypatch.setattr(impl, "_KERNELS", {n: x for n, x in impl.kernels().items() if n != "k_delta"})
    fly = impl.hk_attn_bwd(dout, q, k, v, out, lse, MLA_SCALE, True)
    for a, b in zip(hip, fly):
        assert torch.isfinite(a).all() and torch.equal(a, b)


def test_hk_plan_head_chunks_under_a_scratch_budget():
    """A dS scratch budget below the whole call's splits the (dkdv, dq) pair into head chunks of one
    batch that share one scratch; no head group fitting raises ScratchBudgetError."""
    impl = _hk_package("impl")
    b, s, h = 2, 4096, 128
    q, k, v = _meta(b, s, s, h)
    o = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, s, dtype=torch.float32, device="meta")
    full = impl._plan(o, q, k, v, o, lse, MLA_SCALE, True, None)
    assert full["chunks"] is None and len(full["launches"]) == 3
    half = impl._scratch_bytes(1, s, s, 64, True)
    plan = impl._plan(o, q, k, v, o, lse, MLA_SCALE, True, None, max_scratch_bytes=half)
    assert plan["chunks"] == [(0, 0, 64), (0, 64, 64), (1, 0, 64), (1, 64, 64)]
    assert [x[0] for x in plan["launches"]] == ["delta"] + ["dkdv", "dq"] * 4
    assert plan["ds_full"].numel() * 2 == half
    assert plan["launches"][1][2][0] == (32, 43, 2) and plan["launches"][2][2][0] == (32, 11, 2)
    with pytest.raises(impl.ScratchBudgetError):
        impl._plan(
            o,
            q,
            k,
            v,
            o,
            lse,
            MLA_SCALE,
            True,
            None,
            max_scratch_bytes=impl._scratch_bytes(1, s, s, 32, True) - 1,
        )


@needs_hk
def test_hk_head_chunks_are_bitwise_equal():
    """Head-chunked calls (two chunks of 32 heads) return exactly the unchunked gradients."""
    impl = _hk_package("impl")
    q, k, v, dout = (t.cuda() for t in _inputs(1, 256, 256, 64, layout="bshd"))
    out, lse = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    full = impl.hk_attn_bwd(dout, q, k, v, out, lse, MLA_SCALE, True)
    budget = impl._scratch_bytes(1, 256, 256, 32, True)
    chunked = impl.hk_attn_bwd(dout, q, k, v, out, lse, MLA_SCALE, True, max_scratch_bytes=budget)
    for a, b in zip(full, chunked):
        assert torch.isfinite(a).all() and torch.equal(a, b)
