###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``rmsnorm_residual`` with a second summand, one alias of ``y`` per consumer and exact rounding.

With ``exact=True`` the fused op must match, bit for bit, the graph it replaces: bf16 adds, the plain
``rmsnorm``, autograd summing the consumers' gradients of ``y`` and adding the residual gradient.
The grid-stride backward kernels are autotuned independently, and a different ``num_warps`` changes
the row reduction tree, so the bitwise tests pin both autotuners to the same single config.
"""

import contextlib

import pytest
import torch
import triton

from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_residual
from primus_turbo.triton.normalization.rmsnorm_kernel import (
    rmsnorm_bwd_kernel_grid_stride,
    rmsnorm_bwd_residual_kernel_grid_stride,
)
from tests.pytorch.test_utils import get_tolerances

SHAPES = [(8192, 7168), (512, 7168), (4096, 2048), (8192, 256), (1000, 1536), (64, 512)]
BWD_CONFIGS = [(4, 1), (8, 1), (8, 2)]


@contextlib.contextmanager
def _pinned_bwd_config(num_warps, num_stages):
    """Both grid-stride backward autotuners on one config, their caches cleared."""
    tuners = (rmsnorm_bwd_kernel_grid_stride, rmsnorm_bwd_residual_kernel_grid_stride)
    saved = [(t.configs, dict(t.cache)) for t in tuners]
    try:
        for t in tuners:
            t.configs = [triton.Config({}, num_warps=num_warps, num_stages=num_stages)]
            t.cache.clear()
        yield
    finally:
        for t, (configs, cache) in zip(tuners, saved):
            t.configs = configs
            t.cache.clear()
            t.cache.update(cache)


def _inputs(shape, n, seed=0):
    """``n`` bf16 tensors of ``shape``, then a gamma for them, from one seeded generator."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ts = [torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=g) for _ in range(n)]
    return [*ts, torch.randn(shape[-1], dtype=torch.bfloat16, device="cuda", generator=g)]


def _uses(y, n):
    """One use of ``y`` per consumer, the newest first. Autograd runs the newest node first and adds the
    gradients in the order they arrive, so they sum as (g0 + g1) + g2."""
    if n == 1:
        return [y]
    return [y * 1 for _ in range(n)][::-1]


def _backward(ys, x_plus_r, gys, gxpr):
    """Backward from the uses of ``y`` and, unless ``gxpr`` is None (a sum with no consumer), from the sum."""
    if gxpr is None:
        torch.autograd.backward(ys, gys)
    else:
        torch.autograd.backward([*ys, x_plus_r], [*gys, gxpr])


def _assert_bitwise(actual, expected, what):
    assert actual.dtype == expected.dtype, what
    diff = (
        (actual.view(torch.int16) != expected.view(torch.int16))
        if actual.dtype == torch.bfloat16
        else (actual != expected)
    )
    assert not diff.any(), f"{what}: {int(diff.sum())} of {diff.numel()} elements differ"


def _assert_all_bitwise(out, ref):
    for name, a, b in zip(("y", "x_plus_r", "dx", "dx_extra", "dresidual", "dgamma"), out, ref):
        if b is None:
            assert a is None, name
            continue
        _assert_bitwise(a, b, name)


def _reference(x, x_extra, r, gamma, gys, gxpr, eps):
    """Separate adds, the plain rmsnorm, autograd's accumulation: what the fused op replaces."""
    xr, rr, gr = (t.detach().clone().requires_grad_() for t in (x, r, gamma))
    er = None if x_extra is None else x_extra.detach().clone().requires_grad_()
    h = (xr + er) + rr if er is not None else xr + rr
    y = rmsnorm(h, gr, eps)
    _backward(_uses(y, len(gys)), h, gys, gxpr)
    return y.detach(), h.detach(), xr.grad, None if er is None else er.grad, rr.grad, gr.grad


def _fused(x, x_extra, r, gamma, gys, gxpr, eps, exact=True):
    xf, rf, gf = (t.detach().clone().requires_grad_() for t in (x, r, gamma))
    ef = None if x_extra is None else x_extra.detach().clone().requires_grad_()
    *ys, xpr = rmsnorm_residual(xf, rf, gf, eps, x_extra=ef, num_outputs=len(gys), exact=exact)
    _backward(ys, xpr, gys, gxpr)
    return ys[0].detach(), xpr.detach(), xf.grad, None if ef is None else ef.grad, rf.grad, gf.grad


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_extra", [False, True])
def test_forward_exact(shape, with_extra):
    x, e, r, gamma = _inputs(shape, 3)
    h = (x + e) + r if with_extra else x + r
    y_ref = rmsnorm(h, gamma, 1e-6)
    y, xpr = rmsnorm_residual(x, r, gamma, 1e-6, x_extra=e if with_extra else None, exact=True)
    _assert_bitwise(xpr, h, "x_plus_r")
    _assert_bitwise(y, y_ref, "y")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("num_outputs", [1, 2, 3])
@pytest.mark.parametrize("with_extra", [False, True])
@pytest.mark.parametrize("config", BWD_CONFIGS)
def test_backward_exact(shape, num_outputs, with_extra, config):
    x, e, r, gxpr, *gys, gamma = _inputs(shape, 4 + num_outputs, seed=1)
    extra = e if with_extra else None
    with _pinned_bwd_config(*config):
        ref = _reference(x, extra, r, gamma, gys, gxpr, 1e-6)
        out = _fused(x, extra, r, gamma, gys, gxpr, 1e-6)
    _assert_all_bitwise(out, ref)


@pytest.mark.parametrize("shape", [(8192, 7168), (64, 512)])
@pytest.mark.parametrize("config", BWD_CONFIGS)
def test_backward_exact_unused_sum(shape, config):
    """``x_plus_r`` with no consumer (a final norm): no residual gradient, no add, signed zeros kept."""
    x, e, r, *gys, gamma = _inputs(shape, 6, seed=6)
    with _pinned_bwd_config(*config):
        ref = _reference(x, e, r, gamma, gys, None, 1e-6)
        out = _fused(x, e, r, gamma, gys, None, 1e-6)
    _assert_all_bitwise(out, ref)


@pytest.mark.parametrize("shape", [(8192, 7168), (4096, 2048)])
@pytest.mark.parametrize("num_outputs", [1, 3])
def test_backward_unpinned_close(shape, num_outputs):
    """Under the default autotuning the reduction tree may differ: equal within bf16 rounding."""
    x, e, r, gxpr, *gys, gamma = _inputs(shape, 4 + num_outputs, seed=2)
    ref = _reference(x, e, r, gamma, gys, gxpr, 1e-6)
    out = _fused(x, e, r, gamma, gys, gxpr, 1e-6)
    for name, a, b in zip(("y", "x_plus_r", "dx", "dx_extra", "dresidual"), out, ref):
        torch.testing.assert_close(a, b, **get_tolerances(torch.bfloat16), msg=name)
    torch.testing.assert_close(out[5], ref[5], rtol=3e-2, atol=3e-2, msg="dgamma")


def test_inexact_close():
    """``exact=False`` keeps the sums in fp32: close to the unfused graph, not bitwise."""
    shape = (4096, 7168)
    x, e, r, gxpr, *gys, gamma = _inputs(shape, 7, seed=3)
    ref = _reference(x, e, r, gamma, gys, gxpr, 1e-6)
    out = _fused(x, e, r, gamma, gys, gxpr, 1e-6, exact=False)
    # one rounding instead of two: the gap is an ulp of the summands (|x| up to ~10 here), which
    # where they cancel exceeds the default atol by far
    for name, a, b in zip(("y", "x_plus_r", "dx", "dx_extra", "dresidual"), out, ref):
        torch.testing.assert_close(a, b, rtol=2e-2, atol=1e-1, msg=name)


def test_aliases_share_storage_and_reject_inplace():
    shape = (128, 7168)
    x, r, gamma = _inputs(shape, 2, seed=4)
    x.requires_grad_()
    y0, y1, y2, _ = rmsnorm_residual(x, r, gamma, 1e-6, num_outputs=3, exact=True)
    assert y0.data_ptr() == y1.data_ptr() == y2.data_ptr()
    with pytest.raises(RuntimeError, match="view"):
        y1.add_(1)


def test_default_call_unchanged():
    """No new keyword: the original Function and its fp32 sum, as before."""
    shape = (256, 7168)
    x, r, gamma = _inputs(shape, 2, seed=5)
    x.requires_grad_()
    y, xpr = rmsnorm_residual(x, r, gamma, 1e-6)
    assert type(y.grad_fn).__name__ == "_RMSNormResidualFunctionBackward"
