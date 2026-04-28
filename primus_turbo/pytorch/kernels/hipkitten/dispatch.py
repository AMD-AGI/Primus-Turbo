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

NOTE: per the project policy, this module MUST NOT keep any per-call /
per-shape cache (no dict / weakref / data_ptr / _version / lru_cache /
TTL). Resolutions are done with plain ``getattr`` on every call; the
attribute lookup cost is on the order of tens of nanoseconds and is not a
performance lever to chase.
"""
from __future__ import annotations

import contextlib
import os
import threading

import torch

from primus_turbo.pytorch.kernels.hipkitten.config import HipKittenConfig
from primus_turbo.pytorch.kernels.hipkitten.loader import HipKittenModule

# TK_RCR_FORCE_KERNEL is a process-wide env var read by tk_fp8_layouts at
# kernel launch time. Serialize our writes to it so two FP8 RCR calls on
# different threads cannot race on save/restore. Single-threaded users pay
# only the lock-acquire cost.
_RCR_KERNEL_LOCK = threading.Lock()
_TK_RCR_FORCE_KERNEL_ENV = "TK_RCR_FORCE_KERNEL"


def fp8_has_dscale(hk: HipKittenModule, layout: str) -> bool:
    """Return True if ``hk.module`` exposes a ``gemm_<layout>_dscale`` entry.

    Used by :class:`GEMMFP8HipKittenBackend.execute` to decide whether to
    take the device-pointer scale path. The pre-resolved
    ``hk.gemm_*_dscale`` attributes are populated at module-load time
    (loader.py), so the runtime check is a single attribute load.
    """
    return hk.gemm_dscale(layout) is not None


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
    sync that the host-scalar path otherwise pays per dispatch.

    The kernel callables (``hk.gemm_{rcr,rrr,crr}{_dscale}``) are
    pre-resolved at module-load time (loader.py), so this dispatcher
    issues no ``getattr`` per call.
    """
    if hk.dtype == "bf16":
        # Fast path: BF16 has no kernel-template knob, so we can skip
        # the env-var context manager entirely. Pre-resolved attribute
        # load via the dataclass.
        hk.gemm(cfg.layout)(a, b, c, cfg.group_m, cfg.num_xcds)
        return
    # FP8 path. Apply the kernel-template env hack only when we have a
    # non-None ``cfg.kernel`` to set (None on RRR/CRR, and on RCR rules
    # that pick the binding default).
    rcr_kernel = cfg.kernel if cfg.layout == "rcr" else None
    if rcr_kernel is None:
        if scale_a_dev is not None and scale_b_dev is not None:
            fn_dscale = hk.gemm_dscale(cfg.layout)
            if fn_dscale is not None:
                fn_dscale(a, b, c, scale_a_dev, scale_b_dev, cfg.group_m)
                return
        if scale_a is None or scale_b is None:
            raise ValueError("HipKittens FP8 dense_run requires scale_a/scale_b or scale_*_dev")
        hk.gemm(cfg.layout)(a, b, c, scale_a, scale_b, cfg.group_m)
        return
    with force_rcr_kernel(rcr_kernel):
        if scale_a_dev is not None and scale_b_dev is not None:
            fn_dscale = hk.gemm_dscale(cfg.layout)
            if fn_dscale is not None:
                fn_dscale(a, b, c, scale_a_dev, scale_b_dev, cfg.group_m)
                return
        if scale_a is None or scale_b is None:
            raise ValueError("HipKittens FP8 dense_run requires scale_a/scale_b or scale_*_dev")
        hk.gemm(cfg.layout)(a, b, c, scale_a, scale_b, cfg.group_m)


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
    in Python instead. The grouped launchers are pre-resolved at
    module-load time and exposed via :meth:`HipKittenModule.grouped_balanced`.
    """
    if hk.dtype != "bf16":
        raise ValueError("HipKittens grouped_run_balanced is BF16-only")
    fn = hk.grouped_balanced(cfg.layout)
    if fn is None:
        raise AttributeError(
            f"HipKittens BF16 binding does not expose grouped_{cfg.layout}_balanced; "
            "rebuild tk_bf16_layouts.so or use the per-group dense_run fallback."
        )
    fn(a, b, c, cfg.group_m, cfg.num_xcds)
