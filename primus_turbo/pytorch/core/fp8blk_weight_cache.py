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

    def __init__(self, param: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
        assert param.dim() in (2, 3), f"block FP8 weight cache needs a 2D/3D weight, got {tuple(param.shape)}"
        lead = tuple(param.shape[:-2])
        n, k = param.shape[-2:]
        tn, tk = _cdiv(n, BLOCK), _cdiv(k, BLOCK)
        dev = param.device
        self.shape = tuple(param.shape)
        self.dtype = dtype
        self.data_ptr = param.data_ptr()
        self.w = torch.empty(lead + (n, k), dtype=dtype, device=dev)
        self.wt = torch.empty(lead + (k, n), dtype=dtype, device=dev)
        self.scale_inv = torch.empty(lead + (tn, tk), dtype=torch.float32, device=dev)
        self.e8m0 = torch.empty(lead + (tn, tk), dtype=torch.uint8, device=dev)
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
