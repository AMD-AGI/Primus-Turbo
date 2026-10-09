###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The AITER backend of the bf16 GEMM dispatcher (aiter's ``adaln_gemm`` kernels): at every shape it lists, bitwise
against the torch GEMMs (``x @ w.T + bias``: ``torch.addmm``; ``g @ w``: ``torch.matmul``; ``g.T @ x``: ``torch.mm``)
and against the direct aiter call, eager and under ``torch.compile(fullgraph=True)``; and the dispatcher rules of a
strict default backend."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
)
from primus_turbo.pytorch.kernels.gemm.gemm_impl import (
    GEMMAiterBackend,
    gemm_accum_impl,
    gemm_impl,
)

SHAPES = [(18432, 3072), (9216, 3072)]
M = 32
AITER = BackendType.AITER.value
BF16 = torch.bfloat16

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _need(*keys):
    missing = [key for key in keys if key not in GEMMAiterBackend.shapes()]
    if missing:
        pytest.skip(f"no aiter adaln_gemm kernel for {missing}")


def _bits(t):
    return t.view(torch.int16)


def _inputs(n, k):
    g = torch.Generator(device="cuda").manual_seed(n + k)
    x = torch.randn(M, k, device="cuda", dtype=BF16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=BF16, generator=g) * 0.02
    b = torch.randn(n, device="cuda", dtype=BF16, generator=g)
    go = torch.randn(M, n, device="cuda", dtype=BF16, generator=g)
    return x, w, b, go


def _maybe_compile(f, compiled):
    if not compiled:
        return f
    torch._dynamo.reset()
    return torch.compile(f, fullgraph=True)


@pytest.fixture(autouse=True)
def _reset_backends():
    yield
    GlobalBackendManager.reset()


def test_shapes_are_plain_data():
    s = GEMMAiterBackend.shapes()
    assert isinstance(s, frozenset) and s is GEMMAiterBackend.shapes()
    for p, n, k in s:
        assert p in ("fwd", "dgrad", "wgrad") and isinstance(n, int) and isinstance(k, int)


@gpu
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_fwd_bias(n, k, compiled):
    _need(("fwd", n, k))
    from aiter.ops.adaln_gemm import adaln_fwd

    x, w, b, _ = _inputs(n, k)
    ref = torch.addmm(b, x, w.t())
    f = _maybe_compile(lambda x_, w_, b_: gemm_impl(x_, False, w_, True, BF16, False, AITER, b_), compiled)
    y = f(x, w, b)
    assert y.shape == (M, n) and y.dtype == BF16
    assert torch.equal(_bits(y), _bits(ref))
    assert torch.equal(_bits(y), _bits(adaln_fwd(x, w, b, torch.empty_like(y))))
    # the same product spelled with trans_c: (w @ x.T).T
    assert torch.equal(_bits(gemm_impl(w, False, x, True, BF16, True, AITER, b)), _bits(ref))


@gpu
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_dgrad(n, k, compiled):
    _need(("dgrad", n, k))
    from aiter.ops.adaln_gemm import adaln_dgrad

    _, w, _, go = _inputs(n, k)
    f = _maybe_compile(lambda go_, w_: gemm_impl(go_, False, w_, False, BF16, False, AITER), compiled)
    dx = f(go, w)
    assert dx.shape == (M, k) and dx.dtype == BF16
    assert torch.equal(_bits(dx), _bits(torch.matmul(go, w)))
    assert torch.equal(_bits(dx), _bits(adaln_dgrad(go, w, torch.empty_like(dx))))
    # repeated calls reuse aiter's workspace and counters: same bits
    assert torch.equal(_bits(f(go, w)), _bits(dx))


@gpu
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_wgrad_store(n, k, compiled):
    _need(("wgrad", n, k))
    from aiter.ops.adaln_gemm import adaln_wgrad

    x, _, _, go = _inputs(n, k)
    ref = torch.mm(go.t(), x)

    def f(go_, x_, out_):
        gemm_accum_impl(go_, True, x_, False, BF16, False, out_, AITER, accumulate=False)

    f = _maybe_compile(f, compiled)
    out = torch.full((n, k), float("nan"), device="cuda", dtype=BF16)  # overwritten, not read
    f(go, x, out)
    assert torch.equal(_bits(out), _bits(ref))
    assert torch.equal(_bits(out), _bits(adaln_wgrad(go, x, torch.empty_like(out))))
    # returned, and spelled as ops.gemm's wgrad: (x.T @ g).T
    assert torch.equal(_bits(gemm_impl(go, True, x, False, BF16, False, AITER)), _bits(ref))
    assert torch.equal(_bits(gemm_impl(x, True, go, False, BF16, True, AITER)), _bits(ref))


@gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_can_handle_in_compiled_region(compiled):
    n, k = SHAPES[0]
    _need(("fwd", n, k))
    x, w, b, _ = _inputs(n, k)

    def f(x_, w_, b_):
        if GEMMAiterBackend.can_handle(
            a=x_, trans_a=False, b=w_, trans_b=True, out_dtype=BF16, trans_c=False, bias=b_
        ):
            return gemm_impl(x_, False, w_, True, BF16, False, AITER, b_)
        return torch.addmm(b_, x_, w_.t()) * 0

    y = _maybe_compile(f, compiled)(x, w, b)
    assert torch.equal(_bits(y), _bits(torch.addmm(b, x, w.t())))


@gpu
def test_declines_what_aiter_has_no_kernel_for():
    n, k = SHAPES[0]
    _need(("fwd", n, k), ("dgrad", n, k), ("wgrad", n, k))
    x, w, b, go = _inputs(n, k)
    can = GEMMAiterBackend.can_handle
    kw = dict(out_dtype=BF16, trans_c=False)
    assert can(a=x, trans_a=False, b=w, trans_b=True, bias=b, **kw)
    assert not can(a=x, trans_a=False, b=w, trans_b=True, **kw)  # the forward kernel has a bias epilogue
    assert not can(a=go, trans_a=False, b=w, trans_b=False, bias=b, **kw)
    assert not can(a=torch.cat([x, x]), trans_a=False, b=w, trans_b=True, bias=b, **kw)  # M = 64
    # (N, K) not built
    assert not can(a=x[:, :1024], trans_a=False, b=w[:, :1024], trans_b=True, bias=b, **kw)
    assert not can(
        a=x.half(),
        trans_a=False,
        b=w.half(),
        trans_b=True,
        bias=b.half(),
        out_dtype=torch.half,
        trans_c=False,
    )
    assert not can(a=x, trans_a=False, b=w, trans_b=True, bias=b, out_dtype=torch.float32, trans_c=False)
    out = torch.empty(n, k, device="cuda", dtype=BF16)
    assert can(a=go, trans_a=True, b=x, trans_b=False, out=out, **kw)
    # the kernel overwrites
    assert not can(a=go, trans_a=True, b=x, trans_b=False, out=out, inplace_add_to_out=True, **kw)
    assert not can(a=go, trans_a=True, b=x, trans_b=False, out=out.float(), **kw)


@gpu
def test_pinned_aiter_raises_where_it_cannot_handle():
    n, k = SHAPES[0]
    _need(("fwd", n, k), ("wgrad", n, k))
    x, w, b, go = _inputs(n, k)
    with pytest.raises(ValueError, match="strict"):
        gemm_impl(torch.cat([x, x]), False, w, True, BF16, False, AITER, b)
    with pytest.raises(ValueError, match="strict"):
        gemm_accum_impl(go, True, x, False, BF16, False, torch.zeros(n, k, device="cuda", dtype=BF16), AITER)
    # no backend but AITER has a bias epilogue: elsewhere a bias is an error, not dropped
    with pytest.raises(ValueError, match="No compatible backend"):
        gemm_impl(x[:, :1024], False, w[:, :1024], True, BF16, False, BackendType.HIPBLASLT.value, b)


@gpu
@pytest.mark.parametrize("backend", [BackendType.HIPBLASLT, BackendType.TRITON])
def test_store_into_out_on_other_backends(backend):
    x, _, _, go = _inputs(1024, 512)
    out = torch.full((1024, 512), float("nan"), device="cuda", dtype=BF16)
    gemm_accum_impl(go, True, x, False, BF16, False, out, backend.value, accumulate=False)
    torch.testing.assert_close(out, torch.mm(go.t(), x), rtol=1e-2, atol=1e-2)
    acc = torch.ones_like(out)
    gemm_accum_impl(go, True, x, False, BF16, False, acc, backend.value)
    torch.testing.assert_close(acc, torch.mm(go.t(), x) + 1, rtol=1e-2, atol=1e-2)


# --- dispatcher rules of a strict default backend (no GPU) ---------------------------------------------------------


def _fake(name, handles):
    def can_handle(**kwargs):
        return handles(**kwargs)

    def execute(**kwargs):
        return name

    return type(
        name, (KernelBackend,), {"can_handle": staticmethod(can_handle), "execute": staticmethod(execute)}
    )


class _Dispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.HIPBLASLT: BackendEntry(_fake("general", lambda **kw: True)),
        BackendType.AITER: BackendEntry(_fake("exact", lambda x, **kw: x == 1), strict=True),
    }

    @classmethod
    def make_key(cls, x, **kwargs):
        return x


def test_strict_default_is_pinned():
    pinned = BackendChoice(backend=BackendType.AITER)
    assert _Dispatcher.dispatch(pinned, None, x=1) == "exact"
    with pytest.raises(ValueError, match="strict"):
        _Dispatcher.dispatch(pinned, None, x=2)  # no fallback to "general"
    with pytest.raises(ValueError, match="strict"):
        _Dispatcher.resolve(BackendType.AITER, None, x=2)
    assert _Dispatcher.resolve(BackendType.AITER, None, x=1) == BackendType.AITER
    # a non-strict default still falls back
    assert _Dispatcher.dispatch(BackendChoice(backend=BackendType.TRITON), None, x=2) == "general"


def test_strict_default_ahead_of_auto_tune_behind_user_choice():
    pinned = BackendChoice(backend=BackendType.AITER)
    assert _Dispatcher.dispatch(pinned, BackendChoice(auto_tune=True), x=1) == "exact"
    assert _Dispatcher.dispatch(pinned, BackendChoice(backend=BackendType.HIPBLASLT), x=1) == "general"
