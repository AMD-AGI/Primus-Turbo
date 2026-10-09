###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The AdaLN GEMM ops (aiter's ``adaln_gemm`` kernels) at every shape ``adaln_gemm_table`` lists: bitwise against the
torch GEMMs they replace (forward ``torch.addmm``, dgrad ``torch.matmul``, wgrad ``torch.mm``) and against the direct
aiter call, eager and under ``torch.compile(fullgraph=True)``."""

import pytest
import torch

from primus_turbo.pytorch.ops import (
    ADALN_GEMM_M,
    adaln_gemm_dgrad,
    adaln_gemm_fwd,
    adaln_gemm_table,
    adaln_gemm_wgrad_out,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SHAPES = [(18432, 3072), (9216, 3072)]


def _need(pass_, n, k):
    if (pass_, n, k) not in adaln_gemm_table():
        pytest.skip(f"no aiter adaln_gemm kernel for ({pass_}, {n}, {k})")


def _bits(t):
    return t.view(torch.int16)


def _inputs(n, k):
    g = torch.Generator(device="cuda").manual_seed(n + k)
    x = torch.randn(ADALN_GEMM_M, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g)
    go = torch.randn(ADALN_GEMM_M, n, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, w, b, go


def test_table_is_plain_data():
    t = adaln_gemm_table()
    assert isinstance(t, frozenset)
    for p, n, k in t:
        assert p in ("fwd", "dgrad", "wgrad") and isinstance(n, int) and isinstance(k, int)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_adaln_gemm_fwd(n, k, compiled):
    _need("fwd", n, k)
    from aiter.ops.adaln_gemm import adaln_fwd

    x, w, b, _ = _inputs(n, k)
    f = adaln_gemm_fwd
    if compiled:
        torch._dynamo.reset()
        f = torch.compile(lambda x_, w_, b_: adaln_gemm_fwd(x_, w_, b_), fullgraph=True)
    y = f(x, w, b)
    assert y.shape == (ADALN_GEMM_M, n) and y.dtype == torch.bfloat16
    assert torch.equal(_bits(y), _bits(torch.addmm(b, x, w.t())))
    assert torch.equal(_bits(y), _bits(adaln_fwd(x, w, b, torch.empty_like(y))))


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_adaln_gemm_dgrad(n, k, compiled):
    _need("dgrad", n, k)
    from aiter.ops.adaln_gemm import adaln_dgrad

    _, w, _, go = _inputs(n, k)
    f = adaln_gemm_dgrad
    if compiled:
        torch._dynamo.reset()
        f = torch.compile(lambda go_, w_: adaln_gemm_dgrad(go_, w_), fullgraph=True)
    dx = f(go, w)
    assert dx.shape == (ADALN_GEMM_M, k) and dx.dtype == torch.bfloat16
    assert torch.equal(_bits(dx), _bits(torch.matmul(go, w)))
    assert torch.equal(_bits(dx), _bits(adaln_dgrad(go, w, torch.empty_like(dx))))
    # the workspace and counters the dgrad keeps between calls: a second call is identical
    assert torch.equal(_bits(f(go, w)), _bits(dx))


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_adaln_gemm_wgrad_out(n, k, compiled):
    _need("wgrad", n, k)
    from aiter.ops.adaln_gemm import adaln_wgrad

    x, _, _, go = _inputs(n, k)
    out = torch.full((n, k), float("nan"), device="cuda", dtype=torch.bfloat16)
    if compiled:
        torch._dynamo.reset()

        def f(go_, x_, out_):
            adaln_gemm_wgrad_out(go_, x_, out_)
            return out_ * 1  # a consumer of the mutated buffer in the same graph

        y = torch.compile(f, fullgraph=True)(go, x, out)
        assert torch.equal(_bits(y), _bits(out))
    else:
        adaln_gemm_wgrad_out(go, x, out)
    assert torch.equal(_bits(out), _bits(torch.mm(go.t(), x)))
    assert torch.equal(_bits(out), _bits(adaln_wgrad(go, x, torch.empty_like(out))))
