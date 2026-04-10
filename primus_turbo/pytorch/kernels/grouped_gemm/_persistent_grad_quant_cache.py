###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Process-wide persistent cache for the FP8 quantization of the
``grad_out`` tensor inside the grouped-GEMM tensorwise backward pass.

Round-77 OPTIMIZE rationale
---------------------------
Round-56 introduced a process-singleton cache for the WEIGHT operand
``b`` (`_persistent_b_quant_cache.py`, +19.8 % step). Round-57 extended
the *same* idea to the forward ACTIVATION operand ``a``
(`_persistent_act_quant_cache.py`, +16.5 % additional). The remaining
FP8 surface that has never had a *dedicated* cache is the
``grad_out -> dC_fp8`` tensorwise quantize fired in the backward pass
of ``GroupedGemmFP8TensorFunc``.

Round-76's profile_summary still shows non-trivial residual launches on
that surface:

* ``reduce_row_kernel<AbsMaxOp,bf16->f32>``  ~0.65 ms / 9 launches
* ``compute_scale_from_amax_kernel``        ~0.014 ms / 3 launches
* ``unary_kernel<...,float8_e4m3>``         ~0.84 ms / 3 launches

Within ONE backward call ``grad_out`` is consumed by both the dgrad
GEMM (``grad_in = dC @ B^T``) and the wgrad GEMM
(``grad_b = a^T @ dC``). The autograd engine materialises a single
``grad_out`` Tensor whose identity is stable across both consumers, so
a (data_ptr, shape, dtype, _version, device) key + weakref liveness
guard memoises the FP8 quantize across the two consumers — without
ever serving stale bytes if PyTorch's caching allocator recycles the
``data_ptr`` (the weakref dies and we recompute).

Why a SEPARATE cache (rather than reusing the activation cache from
R57)?
*  Cache-isolation: the activation LRU is sized for forward ``a``
   tensors and may evict ``grad_out`` entries when the bench harness
   sweeps many shapes. A dedicated cache keeps the two surfaces from
   competing for slots.
*  Mechanism attribution: each round in this campaign cleanly maps to
   exactly one persistent cache file, which keeps roll-back and
   profile attribution unambiguous.
*  Lifetime semantics differ: ``grad_out`` is created and destroyed
   inside one autograd backward, whereas ``a``/``b`` survive across
   many iterations. Splitting the caches lets the LRU eviction policy
   drift naturally to whatever each surface needs.

Safety contract — identical to R56/R57:

* Key:  ``(data_ptr, shape, dtype, target_fp8_dtype, _version, device)``
* Anti-collision: each cache entry stores a ``weakref`` to the source
  ``grad_out`` bf16/fp16 tensor. On lookup we require the weakref to
  be alive AND the live tensor's identifying fields to still match.
  PyTorch's caching allocator can recycle a freed device pointer, so
  without this guard a freshly allocated ``grad_out`` could land on a
  recycled ``data_ptr`` with the same ``_version=0`` and silently get
  a stale fp8 buffer.
* On miss we always call the upstream quantize implementation, so
  this is a strict perf optimisation with no numerical drift.
* LRU cap (default 4) bounds peak VRAM. ``grad_out`` buffers can be
  large; a small cap is sufficient since the within-step reuse pattern
  is dgrad+wgrad against the *same* ``grad_out`` Tensor.

Notes
-----
* Cache is process-singleton; CPython GIL serialises dict mutations.
* ``clear()`` is a test/diagnostics hook; it does not free the
  underlying tensors immediately if other references exist (e.g.
  autograd saved tensors), but any future call recomputes from
  scratch.
* Distinct from R23's delayed-scaling history (which altered scale
  semantics across calls and broke correctness because ``grad_out``
  amax drifts): here the cached scale is exactly the per-call
  ``quantize_fp8`` output for the *current* bytes — no temporal
  smoothing. The cache is a memoisation of an idempotent function,
  not a stale-amax reuse policy.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any, Tuple

import torch

__all__ = [
    "get_or_quantize_grad_out_fp8_tensorwise",
    "clear",
    "stats",
]

# `grad_out` reuse pattern is dgrad + wgrad against ONE Tensor identity
# per backward call. A small LRU is enough — we mostly need slot
# stability across the two within-call consumers.
_DEFAULT_CAPACITY: int = 4

# Value layout: (grad_out_fp8, grad_out_scale_inv, weak_ref_to_source_grad_out)
_CACHE: "OrderedDict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor, Any]]" = OrderedDict()

_HITS: int = 0
_MISSES: int = 0


def _make_key(t: torch.Tensor, target_dtype: torch.dtype) -> Tuple[Any, ...]:
    """Build a cache key that uniquely identifies a quantization result.

    ``t._version`` is PyTorch's canonical mutation-version counter
    (bumped by every in-place op). Combined with ``data_ptr``, ``shape``,
    ``dtype`` and the FP8 target dtype, plus the device, this is enough
    to detect any byte-level change in the source tensor — provided the
    weakref liveness probe rules out a recycled ``data_ptr``.
    """
    return (
        t.data_ptr(),
        tuple(t.shape),
        t.dtype,
        target_dtype,
        int(t._version),
        t.device.type,
        t.device.index,
    )


def _entry_matches_live(t: torch.Tensor, ref: Any) -> bool:
    """True only if the weakref'd source tensor is still alive AND its
    identifying fields match the new ``t``.

    Anti-collision guard against PyTorch's caching allocator handing out
    a recycled ``data_ptr``: when the original source tensor has been
    garbage-collected, its ``data_ptr`` is free for reuse and we MUST
    NOT serve a stale fp8 buffer that was quantized from different
    bytes.
    """
    if ref is None:
        return False
    live = ref()
    if live is None:
        return False
    try:
        return (
            live.data_ptr() == t.data_ptr()
            and tuple(live.shape) == tuple(t.shape)
            and live.dtype == t.dtype
            and int(live._version) == int(t._version)
            and live.device == t.device
        )
    except Exception:
        return False


def get_or_quantize_grad_out_fp8_tensorwise(
    t: torch.Tensor,
    target_dtype: torch.dtype,
    quantize_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_out_fp8, grad_out_scale_inv)`` for the ``grad_out``
    tensor ``t``, reusing a previously cached pair when keys match AND
    the original source tensor is still alive (weakref-verified).

    Parameters
    ----------
    t:
        The bf16/fp16 ``grad_out`` tensor that would be passed to
        ``quantize_fp8(..., ScalingGranularity.TENSORWISE)``.
    target_dtype:
        The target FP8 dtype (e.g. ``float8_e4m3fnuz`` /
        ``float8_e4m3``). For HYBRID configs this is typically the
        backward-stage dtype (``float8_e5m2``).
    quantize_fn:
        A no-arg callable that computes ``(grad_out_fp8,
        grad_out_scale_inv)`` from ``t``. Called on a cache miss only.
        We accept a thunk rather than the signature
        ``(t, target_dtype, ...)`` so the call site remains the single
        source of truth for backend-specific kwargs and so this module
        avoids a hard dependency on the ``quantize_fp8`` import graph
        (which would create a circular import with the top-level
        ``primus_turbo.pytorch.ops.quantization`` module).

    Notes
    -----
    Falls through to ``quantize_fn()`` (no caching, no key bookkeeping)
    for tensors we cannot key safely (e.g. non-CUDA tensors or
    non-weakrefable tensors). This keeps the helper a strict perf
    optimisation with no behavioural drift on the edge cases.
    """
    global _HITS, _MISSES

    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        # No safe identity for CPU/meta tensors here; just compute.
        return quantize_fn()

    key = _make_key(t, target_dtype)

    cached = _CACHE.get(key)
    if cached is not None:
        t_fp8, t_scale_inv, ref = cached
        if _entry_matches_live(t, ref):
            # True hit: weakref is alive and identifying fields match.
            _CACHE.move_to_end(key)
            _HITS += 1
            return t_fp8, t_scale_inv
        # Stale entry (data_ptr recycled or weakref died). Evict so we
        # never serve this buffer again, then fall through to recompute.
        try:
            del _CACHE[key]
        except KeyError:  # pragma: no cover -- concurrent access
            pass

    out = quantize_fn()
    # Defensive: only cache if the implementation returned the
    # canonical (fp8_buffer, scale_inv) tuple AND we can take a weakref
    # to the source tensor (required for the liveness guard).
    if (
        isinstance(out, tuple)
        and len(out) == 2
        and isinstance(out[0], torch.Tensor)
        and isinstance(out[1], torch.Tensor)
    ):
        try:
            ref = weakref.ref(t)
        except TypeError:
            # Source tensor is not weak-referenceable; skip caching to
            # preserve correctness.
            _MISSES += 1
            return out
        _CACHE[key] = (out[0], out[1], ref)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _DEFAULT_CAPACITY:
            _CACHE.popitem(last=False)
    _MISSES += 1
    return out


def clear() -> None:
    """Drop all cached entries. Used by tests / diagnostics."""
    global _HITS, _MISSES
    _CACHE.clear()
    _HITS = 0
    _MISSES = 0


def reset() -> None:
    """Alias for ``clear()`` — provided for parity with the
    R56/R57 cache helpers and the test harnesses that import
    ``reset`` rather than ``clear``."""
    clear()


def stats() -> dict:
    """Return current cache hit/miss counters and size."""
    return {"hits": _HITS, "misses": _MISSES, "size": len(_CACHE), "capacity": _DEFAULT_CAPACITY}
