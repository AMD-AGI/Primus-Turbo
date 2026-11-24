###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-step FP8 weight cache of the block-FP8 Linear: verification mode and cache-hit equivalence.

A cache filled with the block quantization of the weight (data, scales, E8M0 bytes, transposed copies) is a hit
for the Linear op: forward and dgrad are bit-identical to the run that quantizes the weight itself. With the
verification switch every hit is re-quantized and compared, and a corrupted cache raises.
"""

import pytest
import torch

import primus_turbo.common.envs as envs
import primus_turbo.pytorch as pt
from primus_turbo.pytorch.core import fp8blk_weight_cache as wcache
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from tests.pytorch.ops.gfx1250_helper import bits_equal, requires_flydsl

pytestmark = requires_flydsl
CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128)
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_KERNEL", "g90")


def _fill_cache(w):
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        quant_fp8_blockwise_for_weight_impl,
    )

    cache = wcache.attach(w, FP8)
    q, s = quant_fp8_blockwise_for_weight_impl(w.detach().contiguous(), FP8, block_size=128)
    e = ((s.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    cache.w.copy_(q)
    cache.scale_inv.copy_(s)
    cache.e8m0.copy_(e)
    cache.wt.copy_(q.t())
    cache.e8m0_t.copy_(e.t())
    cache.ready = True
    return cache


def _run(x, w, gy):
    xx = x.detach().requires_grad_(True)
    w.grad = None
    y = pt.ops.gemm_fp8(xx, w, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=CFG)
    y.backward(gy)
    torch.cuda.synchronize()
    return y.detach().clone(), xx.grad.clone()


def _inputs(t=512, n=1024, k=1024):
    torch.manual_seed(0)
    x = (torch.randn(t, k, device="cuda") * 0.5).bfloat16()
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16().requires_grad_(True)
    gy = (torch.randn(t, n, device="cuda") * 0.1).bfloat16()
    return x, w, gy


def test_cache_hit_equals_uncached_run(monkeypatch):
    x, w, gy = _inputs()
    monkeypatch.delenv("PRIMUS_TURBO_FP8_WCACHE", raising=False)
    y0, dx0 = _run(x, w, gy)
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE_VERIFY", "1")
    cache = _fill_cache(w)
    y1, dx1 = _run(x, w, gy)
    assert cache.matches(w, torch.float8_e4m3fn)
    assert bits_equal(y0, y1) and bits_equal(dx0, dx1)


@pytest.mark.parametrize("field", ["w", "wt", "e8m0", "e8m0_t", "scale_inv"])
def test_verify_detects_corrupted_cache(monkeypatch, field):
    x, w, gy = _inputs()
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE_VERIFY", "1")
    cache = _fill_cache(w)
    assert wcache.lookup(w, FP8) is cache  # intact cache passes the check
    t = getattr(cache, field)
    if t.dtype == torch.float32:
        t.view(torch.int32).view(-1)[0] ^= 0x00800000
    else:
        t.view(torch.uint8).view(-1)[0] ^= 0x01
    with pytest.raises(RuntimeError, match="WCACHE_VERIFY"):
        wcache.lookup(w, FP8)


def test_cache_is_not_used_when_disabled_or_not_ready(monkeypatch):
    x, w, gy = _inputs()
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE", "1")
    cache = _fill_cache(w)
    cache.ready = False
    assert wcache.lookup(w, FP8) is None
    cache.ready = True
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE", "0")
    envs.reset_cache("PRIMUS_TURBO_FP8_WCACHE")  # envs caches a switch at its first read
    assert wcache.lookup(w, FP8) is None
