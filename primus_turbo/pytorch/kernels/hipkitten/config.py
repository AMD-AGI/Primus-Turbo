###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Per-shape HipKittens kernel configuration.

A :class:`HipKittenConfig` describes everything needed to launch a single
GEMM/grouped-GEMM call: the layout (rcr/rrr/crr), the ``group_m`` grouped
tile-scheduling factor, optionally ``num_xcds`` (BF16) and ``kernel`` (FP8),
and a ``cache_hit`` flag that records whether the values came from the
offline autotune cache or from a fallback default.

Backends use :func:`lookup` to populate the config; cache misses still
produce a usable Config (with the binding's documented defaults) so a
backend can choose to either reject the shape in ``can_handle`` or to
fall through with the defaults if the shape happens to be aligned.
"""
from __future__ import annotations

from dataclasses import dataclass

from primus_turbo.pytorch.kernels.hipkitten.layout import Layout
from primus_turbo.pytorch.kernels.hipkitten.loader import HipKittenModule


@dataclass(frozen=True)
class HipKittenConfig:
    """Resolved per-call kernel configuration.

    Attributes:
        layout: One of ``rcr``/``rrr``/``crr`` (already resolved from
            ``trans_a, trans_b`` upstream).
        group_m: HipKittens' grouped tile-scheduling factor.
        num_xcds: XCD assignment knob (BF16 binding only). ``None`` for FP8.
        kernel: FP8 binding's kernel-template id, applied via
            ``TK_RCR_FORCE_KERNEL`` for the RCR layout. ``None`` for BF16
            and for FP8 layouts other than RCR.
        cache_hit: ``True`` when the values came from the autotune cache;
            ``False`` when they fell back to the binding defaults.
    """

    layout: Layout
    group_m: int
    num_xcds: int | None
    kernel: str | None
    cache_hit: bool


def has_in_cache(hk: HipKittenModule, layout: Layout, m: int, n: int, k: int) -> bool:
    """True when the autotune cache has a tuned entry for ``(layout, m, n, k)``."""
    return (layout, m, n, k) in hk.cache


def lookup(hk: HipKittenModule, layout: Layout, m: int, n: int, k: int) -> HipKittenConfig:
    """Resolve the kernel config for a given shape, with fallback defaults.

    The cache is keyed by ``(layout, M_padded, N_padded, K_padded)`` -- pad
    upstream before calling. Fallbacks come from the binding's documented
    defaults (``group_m=4``, ``num_xcds=8`` for BF16; FP8 carries a
    ``DEFAULT_GROUP_M`` constant that we mirror).
    """
    entry = hk.cache.get((layout, m, n, k))
    if entry is None:
        if hk.dtype == "bf16":
            return HipKittenConfig(
                layout=layout,
                group_m=hk.default_group_m,
                num_xcds=hk.default_num_xcds,
                kernel=None,
                cache_hit=False,
            )
        return HipKittenConfig(
            layout=layout,
            group_m=hk.default_group_m,
            num_xcds=None,
            kernel="8",
            cache_hit=False,
        )
    return HipKittenConfig(
        layout=layout,
        group_m=int(entry["group_m"]),
        num_xcds=entry.get("num_xcds"),
        kernel=entry.get("kernel"),
        cache_hit=True,
    )
