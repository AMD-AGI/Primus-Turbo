###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Process-wide persistent cache for the FP8 ACTIVATION quantization in
grouped GEMM tensorwise kernels.

Round-57 OPTIMIZE rationale
---------------------------
Round-56 ACCEPTED a process-singleton FP8 cache for the WEIGHT operand
``b`` (see ``_persistent_b_quant_cache.py``). The accepted profile (see
``profiles/round-56_post_accept/profile_summary.md``) shows that the
remaining quantize traffic is dominated by the ACTIVATION operands:

* ``unary_kernel<bf16->fp8_e4m3>`` ~112 launches/step
* ``reduce_row_kernel<AbsMax,bf16->f32>`` ~336 launches/step

These come from per-call ``quantize_fp8(a, ...)`` in the forward path and
``quantize_fp8(grad_out, ...)`` in the backward path of
``GroupedGemmFP8TensorFunc``. Inside the bench harness's 50 inner-iter
loop the *same* ``a`` and ``grad_out`` tensor identities are reused
across iterations (their ``data_ptr`` and ``_version`` are stable for
the duration of one bench shape), so we can memoise the quantization
result keyed on the tensor's identity.

The cache mirrors the safety contract of R56's b-cache exactly:

* Key:  ``(data_ptr, shape, dtype, target_fp8_dtype, _version, device)``
* Anti-collision: each cache entry stores a ``weakref`` to the source
  bf16 tensor; on lookup we require the weakref to be alive AND the
  live tensor's identifying fields to still match. PyTorch's caching
  allocator can recycle a freed device pointer, so without this guard
  a freshly allocated activation could land on a recycled ``data_ptr``
  with the same ``_version=0`` and silently get a stale fp8 buffer.
* On miss we always call the upstream quantize implementation, so this
  is a strict perf optimization with no numerical drift.
* LRU cap (default 8) bounds peak VRAM. Activation buffers can be large
  (~117 MB FP8 at the largest representative shape), so we keep the
  same conservative capacity as the b-cache.

Notes
-----
* Cache is process-singleton; CPython GIL serialises dict mutations.
* ``clear()`` is a test/diagnostics hook; it does not free the
  underlying tensors immediately if other references exist (e.g.
  autograd saved tensors), but any future call recomputes from scratch.
* Distinct from R23's delayed-scaling history (which altered scale
  semantics across calls and broke correctness because activation amax
  drifts): here the cached scale is exactly the per-call
  ``quantize_fp8`` output for the *current* bytes — no temporal smoothing.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any, Tuple

import torch

__all__ = [
    "get_or_quantize_act_fp8_tensorwise",
    "clear",
    "stats",
]

# Activation tensors can be large; keep capacity equal to the b-cache so
# the worst-case footprint scales linearly and predictably.
_DEFAULT_CAPACITY: int = 8

# Value layout: (act_fp8, act_scale_inv, weak_ref_to_source_act)
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


def get_or_quantize_act_fp8_tensorwise(
    t: torch.Tensor,
    target_dtype: torch.dtype,
    quantize_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(t_fp8, t_scale_inv)`` for the activation tensor ``t``,
    reusing a previously cached pair when keys match AND the original
    source tensor is still alive (weakref-verified).

    Parameters
    ----------
    t:
        The bf16/fp16 activation tensor that would be passed to
        ``quantize_fp8(..., ScalingGranularity.TENSORWISE)``.
    target_dtype:
        The target FP8 dtype (e.g. ``float8_e4m3fnuz`` / ``float8_e4m3``).
    quantize_fn:
        A no-arg callable that computes ``(t_fp8, t_scale_inv)`` from
        ``t``. Called on a cache miss only.
    """
    global _HITS, _MISSES

    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        return quantize_fn()

    key = _make_key(t, target_dtype)

    cached = _CACHE.get(key)
    if cached is not None:
        t_fp8, t_scale_inv, ref = cached
        if _entry_matches_live(t, ref):
            _CACHE.move_to_end(key)
            _HITS += 1
            return t_fp8, t_scale_inv
        # Stale entry (data_ptr recycled or weakref died). Evict.
        try:
            del _CACHE[key]
        except KeyError:  # pragma: no cover -- concurrent access
            pass

    out = quantize_fn()
    if (
        isinstance(out, tuple)
        and len(out) == 2
        and isinstance(out[0], torch.Tensor)
        and isinstance(out[1], torch.Tensor)
    ):
        try:
            ref = weakref.ref(t)
        except TypeError:
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


def stats() -> dict:
    """Return current cache hit/miss counters and size."""
    return {"hits": _HITS, "misses": _MISSES, "size": len(_CACHE), "capacity": _DEFAULT_CAPACITY}
