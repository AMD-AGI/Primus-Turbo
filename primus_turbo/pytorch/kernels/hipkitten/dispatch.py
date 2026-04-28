###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Layout-dispatched calls into the HipKittens BF16 / FP8 binding.

The binding signatures differ slightly between precisions:

  * BF16 (``tk_bf16_layouts``):
      ``gemm_{rcr,rrr,crr}(a, b, c, group_m, num_xcds)``
      ``grouped_{rcr,rrr,crr}_balanced(a, b, c, group_m, num_xcds)``

  * FP8 (``tk_fp8_layouts``):
      ``gemm_{rcr,rrr,crr}(a, b, c, scale_a, scale_b, group_m)``
      (no grouped entrypoints; FP8 grouped is implemented in Python by
      looping over the dense entry per group)

This module hides the difference behind :func:`dense_run` and
:func:`grouped_run_balanced`, plus a thread-safe context manager for the
``TK_RCR_FORCE_KERNEL`` env-var hack the FP8 binding uses to select among
RCR kernel templates (the binding will eventually take ``kernel`` as a
parameter; once it does we can drop the env-var path entirely).
"""
from __future__ import annotations

import contextlib
import os
import threading
from functools import lru_cache
from typing import Any, Callable

import torch

from primus_turbo.pytorch.kernels.hipkitten.config import HipKittenConfig
from primus_turbo.pytorch.kernels.hipkitten.loader import HipKittenModule

# TK_RCR_FORCE_KERNEL is a process-wide env var read by tk_fp8_layouts at
# kernel launch time. Serialize our writes to it so two FP8 RCR calls on
# different threads cannot race on save/restore. Single-threaded users pay
# only the lock-acquire cost.
_RCR_KERNEL_LOCK = threading.Lock()
_TK_RCR_FORCE_KERNEL_ENV = "TK_RCR_FORCE_KERNEL"


# Cache the resolved binding callables per (module, layout) so the inner
# dispatch loop does not pay an `hasattr` / `getattr` cost on every call.
# In a tight metric loop (50 iters per shape, 8 FP8 shapes) the redundant
# attribute lookups added ~1-2 us per dispatch on top of the kernel; this
# cache reduces it to a single dict probe (`@lru_cache`).
@lru_cache(maxsize=16)
def _fp8_dense_fn(module: Any, layout: str) -> Callable:
    return getattr(module, f"gemm_{layout}")


@lru_cache(maxsize=16)
def _fp8_dscale_fn(module: Any, layout: str) -> Callable | None:
    return getattr(module, f"gemm_{layout}_dscale", None)


@lru_cache(maxsize=16)
def _bf16_dense_fn(module: Any, layout: str) -> Callable:
    return getattr(module, f"gemm_{layout}")


def fp8_has_dscale(hk: HipKittenModule, layout: str) -> bool:
    """Return True if ``hk.module`` exposes a ``gemm_<layout>_dscale`` entry.

    Used by :class:`GEMMFP8HipKittenBackend.execute` to decide whether to
    take the device-pointer scale path. Resolution is cached per
    (module, layout).
    """
    return _fp8_dscale_fn(hk.module, layout) is not None


@contextlib.contextmanager
def force_rcr_kernel(kernel: str | None):
    """Pin ``TK_RCR_FORCE_KERNEL`` for the duration of an FP8 RCR call.

    Yields immediately when ``kernel`` is None (BF16 path or FP8 non-RCR
    layouts). Restores the previous env-var value on exit even on
    exception, so the env never leaks between callers.
    """
    if kernel is None:
        yield
        return
    with _RCR_KERNEL_LOCK:
        prev = os.environ.get(_TK_RCR_FORCE_KERNEL_ENV)
        os.environ[_TK_RCR_FORCE_KERNEL_ENV] = kernel
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop(_TK_RCR_FORCE_KERNEL_ENV, None)
            else:
                os.environ[_TK_RCR_FORCE_KERNEL_ENV] = prev


def dense_run(
    hk: HipKittenModule,
    cfg: HipKittenConfig,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    scale_a: float | None = None,
    scale_b: float | None = None,
    scale_a_dev: torch.Tensor | None = None,
    scale_b_dev: torch.Tensor | None = None,
) -> None:
    """Dispatch ``module.gemm_{layout}(...)`` with ``cfg``-derived parameters.

    BF16 binding takes ``(a, b, c, group_m, num_xcds)`` while FP8 takes
    ``(a, b, c, scale_a, scale_b, group_m)``. ``c`` is written in place;
    ``a`` / ``b`` must already be ``contiguous()`` (callers do this near
    the layout / padding decision).

    For FP8 the caller may pass either host-side scalars (``scale_a`` /
    ``scale_b``) or 0-d device tensors (``scale_a_dev`` / ``scale_b_dev``).
    The device-tensor path uses the binding's ``gemm_<layout>_dscale``
    entry which reads the scales in-kernel and skips the ``.item()`` stream
    sync that the host-scalar path otherwise pays per dispatch (~18us on
    small dense FP8 shapes — comparable to the GEMM kernel itself, so
    this is the main lever to close the gap to hipBLASLt).
    """
    rcr_kernel = cfg.kernel if cfg.layout == "rcr" else None
    with force_rcr_kernel(rcr_kernel):
        if hk.dtype == "bf16":
            _bf16_dense_fn(hk.module, cfg.layout)(a, b, c, cfg.group_m, cfg.num_xcds)
            return
        if scale_a_dev is not None and scale_b_dev is not None:
            fn_dscale = _fp8_dscale_fn(hk.module, cfg.layout)
            if fn_dscale is not None:
                fn_dscale(a, b, c, scale_a_dev, scale_b_dev, cfg.group_m)
                return
        if scale_a is None or scale_b is None:
            raise ValueError("HipKittens FP8 dense_run requires scale_a/scale_b or scale_*_dev")
        _fp8_dense_fn(hk.module, cfg.layout)(a, b, c, scale_a, scale_b, cfg.group_m)


def grouped_run_balanced(
    hk: HipKittenModule,
    cfg: HipKittenConfig,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> None:
    """Dispatch ``module.grouped_{layout}_balanced(...)`` (BF16 only).

    FP8 has no native grouped entrypoint, so the FP8 grouped backend
    implements grouped GEMM by looping :func:`dense_run` over the groups
    in Python instead.
    """
    if hk.dtype != "bf16":
        raise ValueError("HipKittens grouped_run_balanced is BF16-only")
    fn = getattr(hk.module, f"grouped_{cfg.layout}_balanced")
    fn(a, b, c, cfg.group_m, cfg.num_xcds)
