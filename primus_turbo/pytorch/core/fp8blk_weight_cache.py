###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""Per-optimizer-step cache of 128x128-block FP8 weights (PRIMUS_TURBO_FP8_WCACHE=1).

The optimizer (Primus PRIMUS_FP8BLK_ADAM_QUANT=1, fused into its Adam BF16 writeback) fills one
cache per weight Parameter right after the weight update:
  w          fp8 E4M3 [.., N, K]        == quant_fp8_blockwise_for_weight_impl(weight)[0] (UE8M0)
  scale_inv  fp32     [.., N/128, K/128] == quant_fp8_blockwise_for_weight_impl(weight)[1]
  e8m0       uint8    [.., N/128, K/128] exponent byte of scale_inv (g90 GEMM operand)
  wt         fp8 E4M3 [.., K, N]        == w.transpose(-1, -2) (same tile, same scale)
  e8m0_t     uint8    [.., K/128, N/128] == e8m0.transpose(-1, -2)
The block-FP8 GEMM ops then skip their per-call weight quantization (and the dgrad W^T copy).

A cache is used only while it is ``ready``: the optimizer sets it after the fused kernel has been
queued (stream order makes the data visible to later forward/backward kernels), and clears it
whenever it updates the weight any other way. Weights must not be modified outside the optimizer
while the cache is enabled. PRIMUS_TURBO_FP8_WCACHE_VERIFY=1 re-quantizes every cache hit and
raises on any byte difference (debug only; costs a sync).
"""

import os
from typing import Optional

import torch

BLOCK = 128


def enabled() -> bool:
    # The optimizer emits UE8M0 (power-of-two) scales, i.e. the PRIMUS_TURBO_FP8_BLOCK_UE8M0=1 quant.
    return (
        os.environ.get("PRIMUS_TURBO_FP8_WCACHE", "0") == "1"
        and os.environ.get("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "0") == "1"
    )


def _verify() -> bool:
    return os.environ.get("PRIMUS_TURBO_FP8_WCACHE_VERIFY", "0") == "1"


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class Fp8BlockWeightCache:
    """FP8 W / W^T / scales of one weight Parameter, refreshed by the optimizer each step."""

    def __init__(self, param: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn, w=None, scale_inv=None,
                 e8m0=None):
        assert param.dim() in (2, 3), f"block FP8 weight cache needs a 2D/3D weight, got {tuple(param.shape)}"
        lead = tuple(param.shape[:-2])
        n, k = param.shape[-2:]
        tn, tk = _cdiv(n, BLOCK), _cdiv(k, BLOCK)
        dev = param.device
        self.shape = tuple(param.shape)
        self.dtype = dtype
        self.data_ptr = param.data_ptr()
        # w / scale_inv / e8m0 may be given as (contiguous) views of a joint buffer, see attach_joint
        self.w = torch.empty(lead + (n, k), dtype=dtype, device=dev) if w is None else w
        self.wt = torch.empty(lead + (k, n), dtype=dtype, device=dev)
        self.scale_inv = torch.empty(lead + (tn, tk), dtype=torch.float32, device=dev) if scale_inv is None else scale_inv
        self.e8m0 = torch.empty(lead + (tn, tk), dtype=torch.uint8, device=dev) if e8m0 is None else e8m0
        self.e8m0_t = torch.empty(lead + (tk, tn), dtype=torch.uint8, device=dev)
        self.ready = False
        self.hits = 0

    def matches(self, b: torch.Tensor, dtype: torch.dtype) -> bool:
        return (
            self.ready
            and dtype == self.dtype
            and tuple(b.shape) == self.shape
            and b.data_ptr() == self.data_ptr
        )


def attach(param: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn) -> Fp8BlockWeightCache:
    """Create (once) and return the cache of ``param``; called by the optimizer."""
    cache = getattr(param, "_primus_fp8blk_cache", None)
    if cache is None or cache.shape != tuple(param.shape) or cache.data_ptr != param.data_ptr():
        cache = Fp8BlockWeightCache(param, dtype)
        param._primus_fp8blk_cache = cache
    return cache


class JointFp8BlockWeightCache:
    """Caches of several 2D weights [N_i, K] (same K, N_i % 128 == 0) whose W / scale_inv / E8M0 are row slices of
    one joint [sum N_i, K] buffer, in the given order. The optimizer fills each slice through the per-weight cache
    (contiguous views), so the joint W and its scales are ready with no per-step copy (used by the concat-N GEMM,
    e.g. DSv4 wq_a || wkv)."""

    def __init__(self, params, dtype):
        k = params[0].shape[1]
        ns = [int(p.shape[0]) for p in params]
        ntot = sum(ns)
        dev = params[0].device
        self.key = tuple((p.data_ptr(), tuple(p.shape)) for p in params)
        self.dtype = dtype
        self.w = torch.empty((ntot, k), dtype=dtype, device=dev)
        self.scale_inv = torch.empty((ntot // BLOCK, _cdiv(k, BLOCK)), dtype=torch.float32, device=dev)
        self.e8m0 = torch.empty((ntot // BLOCK, _cdiv(k, BLOCK)), dtype=torch.uint8, device=dev)
        self.caches = []
        off = 0
        for p, n in zip(params, ns):
            c = Fp8BlockWeightCache(p, dtype, w=self.w[off:off + n], scale_inv=self.scale_inv[off // BLOCK:(off + n) // BLOCK],
                                    e8m0=self.e8m0[off // BLOCK:(off + n) // BLOCK])
            p._primus_fp8blk_cache = c
            self.caches.append(c)
            off += n

    def ready(self, params, dtype) -> bool:
        """All per-weight caches are ready and still the joint views (none was re-created by ``attach``)."""
        if dtype != self.dtype:
            return False
        for p, c in zip(params, self.caches):
            if getattr(p, "_primus_fp8blk_cache", None) is not c or not c.matches(p, dtype):
                return False
        return True


def attach_joint(params, dtype: torch.dtype = torch.float8_e4m3fn) -> Optional[JointFp8BlockWeightCache]:
    """Create (once, before the optimizer's first ``attach``) joint caches for ``params``; None if not possible
    (cache disabled, shapes, or a weight already has its own cache). Tags the weights like ``lookup``."""
    if not enabled() or len(params) < 2:
        return None
    jc = getattr(params[0], "_primus_fp8blk_joint", None)
    key = tuple((p.data_ptr(), tuple(p.shape)) for p in params)
    if jc is not None:
        return jc if jc.key == key else None
    k = params[0].shape[-1]
    if any(p.dim() != 2 or p.shape[1] != k or p.shape[0] % BLOCK or k % BLOCK for p in params):
        return None
    if any(getattr(p, "_primus_fp8blk_cache", None) is not None for p in params):
        return None
    jc = JointFp8BlockWeightCache(params, dtype)
    for p in params:
        p._primus_fp8blk_wanted = True
        p._primus_fp8blk_joint = jc
    return jc


def wants_cache(param: torch.Tensor) -> bool:
    """True if a block-FP8 GEMM has quantized this weight (tag set by ``lookup``)."""
    return bool(getattr(param, "_primus_fp8blk_wanted", False))


def _check(b: torch.Tensor, cache: Fp8BlockWeightCache) -> None:
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        quant_fp8_blockwise_for_weight_impl,
    )

    q, s = quant_fp8_blockwise_for_weight_impl(b.contiguous(), cache.dtype, block_size=BLOCK)
    e = ((s.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    bad = []
    if not torch.equal(q.view(torch.uint8), cache.w.view(torch.uint8)):
        bad.append("w")
    if not torch.equal(s.view(torch.int32), cache.scale_inv.view(torch.int32)):
        bad.append("scale_inv")
    if not torch.equal(cache.wt.view(torch.uint8), q.view(torch.uint8).transpose(-1, -2)):
        bad.append("wt")
    if not torch.equal(cache.e8m0, e) or not torch.equal(cache.e8m0_t, e.transpose(-1, -2)):
        bad.append("e8m0")
    if bad:
        raise RuntimeError(f"PRIMUS_TURBO_FP8_WCACHE_VERIFY: cached {bad} differ from a fresh quant of {tuple(b.shape)}")


def lookup(b: torch.Tensor, dtype: torch.dtype) -> Optional[Fp8BlockWeightCache]:
    """Ready cache of weight ``b`` or None. Tags ``b`` so the optimizer starts caching it."""
    if not enabled() or not isinstance(b, torch.Tensor) or b.dim() not in (2, 3):
        return None
    try:
        b._primus_fp8blk_wanted = True
    except (AttributeError, RuntimeError):
        return None
    cache = getattr(b, "_primus_fp8blk_cache", None)
    if cache is None or not cache.matches(b, dtype):
        return None
    if _verify():
        _check(b, cache)
    cache.hits += 1
    return cache
