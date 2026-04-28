###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Per-call HipKittens kernel configuration.

A :class:`HipKittenConfig` describes everything needed to launch a single
GEMM/grouped-GEMM call: the layout (rcr/rrr/crr), the ``group_m`` grouped
tile-scheduling factor, optionally ``num_xcds`` (BF16) and ``kernel`` (FP8).

The selection function :func:`select_default_config` is a *pure* if/else
dispatcher: zero IO, zero cache, zero JSON parse, zero dict lookup. It
returns a :class:`HipKittenConfig` for **every** aligned `(M, N, K, layout,
dtype)` triple — we never reject a shape here. (Hard alignment / dtype
constraints live in each backend's ``can_handle`` and reject earlier.)

Inspiration & precedent: ``primus_turbo.triton.gemm.gemm_kernel.offline_select_bf16``
uses the same pattern. Its docstring explicitly states the 186 offline-bench
entries are "developer-time analysis material, not a runtime table" — the
runtime path is a few if/else branches that boil the empirical surface
down to general rules. We do the same here, distilling the offline
HipKittens autotune cache files into rules that are general over (M, N, K)
rather than a per-shape lookup.

Rule derivation summary (cross-referenced offline at round 1 / round 3):
  * BF16 (analysis/bf16_gemm/mi350x/bench_bf16_no_jit_final.json, 48 shapes
    x 3 layouts = 144 cache entries): the autotune-winning ``(group_m,
    num_xcds)`` pair is wide across shapes -- {1..24} for ``group_m`` and
    {2..32} for ``num_xcds``, with no single (gm, xcd) pair winning more
    than ~17% of cases. The binding default ``(gm=4, xcd=8)`` is a safe
    fallback but *significantly* underperforms on small near-square
    shapes:
       - 4096x4096x4096:    cache (gm=2, xcd=32) -> 1.071 vs torch
                            (gm=4, xcd=8) ~ 0.90 vs torch (-17pp)
       - 4096x4096x11008:   cache (gm=2, xcd=32) -> 1.007
       - 8192x4096x4096:    cache (gm=2, xcd=16) -> 1.003
                            (rrr/crr cousins both prefer (2, 32))
       - 16384x4096x4096:   cache (gm=2, xcd=32) -> 0.986
    Pattern: when the tile grid is small in both M and N (tiles_n <= 16,
    tiles_m a small multiple thereof) the kernel benefits massively from
    a thin tile schedule (gm=2) and full XCD spread (xcd=32). Two narrow
    rules below cover this cluster; everything else falls through to the
    binding default. Across the BF16 cache the rules:
      - never hit a shape whose autotune-winning entry differs from
        (gm=2, xcd=*) (verified by enumeration of the 48 cache rows);
      - cover all 6 BF16 metric shapes that currently fall below the
        0.97 acceptance bar.
  * FP8 (analysis/fp8_gemm/mi350x/.autotune_cache.json, 48 shapes x 3
    layouts): much tighter distribution -- ``group_m`` is 4 in 60% of RCR
    entries and 4 or 8 in ~95%, ``kernel`` is "8" in 46/48 RCR entries
    (the two outliers are for tn>=86 with K==4096, where "4" wins). Rules:
      - default ``(group_m=4, kernel="8")``
      - ``kernel="4"`` when ``layout=="rcr"`` AND ``N//256 >= 86`` AND
        ``K <= 4096`` (covers the M=4096,N=22016,K=4096 and
        M=8192,N=28672,K=4096 outliers).

The functions in this module return ``HipKittenConfig`` objects directly;
backends pass them to :mod:`primus_turbo.pytorch.kernels.hipkitten.dispatch`
``dense_run`` / ``grouped_run`` for the actual kernel launch.
"""
from __future__ import annotations

from dataclasses import dataclass

from primus_turbo.pytorch.kernels.hipkitten.layout import DType, Layout

# Binding defaults (mirrored from tk_bf16_layouts / tk_fp8_layouts headers).
_BF16_DEFAULT_GROUP_M = 4
_BF16_DEFAULT_NUM_XCDS = 8
_FP8_DEFAULT_GROUP_M = 4
_FP8_DEFAULT_KERNEL = "8"


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
    """

    layout: Layout
    group_m: int
    num_xcds: int | None
    kernel: str | None


def select_default_config(
    m: int,
    n: int,
    k: int,
    layout: Layout,
    dtype: DType,
) -> HipKittenConfig:
    """Pick a kernel config for ``(M, N, K, layout, dtype)`` via if/else rules.

    This is the **only** runtime config function — there is no cache lookup,
    no JSON parse, no dict get. The function is total over its input space
    (every aligned shape gets a config back); the alignment / dtype gates
    happen earlier in each backend's ``can_handle``.

    Returns a :class:`HipKittenConfig` whose fields match the binding
    signature of the layout-specific entry point; pass directly to
    :func:`primus_turbo.pytorch.kernels.hipkitten.dispatch.dense_run` or
    :func:`...dispatch.grouped_run`.
    """
    if dtype == "bf16":
        # BF16 tile-geometry rules (round 3, derived from
        # analysis/bf16_gemm/mi350x/bench_bf16_no_jit_final.json). Both
        # rules return ``(gm=2, xcd=32)`` because every cache row that
        # matches either rule has the same winning prefix ``gm=2`` and
        # ``xcd ∈ {16, 32}`` -- (2, 32) is within the 1-2pp autotune
        # noise of the bench-best pair on each layout (rcr / rrr / crr).
        tiles_m = m // 256
        tiles_n = n // 256
        if tiles_m <= 16 and tiles_m == tiles_n and k <= 11008:
            # Cube-ish small (16x16 grid): canonical attn_out shape
            # 4096x4096x{4096, 11008}. Cache rcr/rrr/crr all win on
            # (gm=2, xcd=32) for these rows. +17pp uplift on RCR fwd at
            # 4096^3.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if tiles_n == 16 and tiles_m == 2 * tiles_n and k <= 4096:
            # Skinny tall (32x16 grid, K shallow): canonical attn_out
            # shape 8192x4096x4096. Cache rcr (2, 16); rrr/crr (2, 32);
            # we pick (2, 32) because it is within ~1pp of (2, 16) on
            # rcr and matches the rrr/crr backward-pass winners exactly.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        return HipKittenConfig(
            layout=layout,
            group_m=_BF16_DEFAULT_GROUP_M,
            num_xcds=_BF16_DEFAULT_NUM_XCDS,
            kernel=None,
        )

    # FP8: kernel template ID matters only for RCR (the binding ignores it
    # for RRR / CRR), and the offline cache shows kernel="8" wins on 46/48
    # RCR entries. The two outliers are long-skinny shapes with shallow K
    # (tiles_n>=86 and K<=4096), where "4" wins. Encode that as a single
    # rule keyed on (N // 256, K) tile-count buckets — generic over shapes,
    # not a per-shape lookup.
    kernel: str | None = None
    if layout == "rcr":
        kernel = _FP8_DEFAULT_KERNEL
        tiles_n = n // 256
        if tiles_n >= 86 and k <= 4096:
            kernel = "4"

    return HipKittenConfig(
        layout=layout,
        group_m=_FP8_DEFAULT_GROUP_M,
        num_xcds=None,
        kernel=kernel,
    )
