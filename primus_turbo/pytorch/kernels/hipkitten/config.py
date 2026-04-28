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

Rule derivation summary (cross-referenced offline at round 1 / round 3,
extended at round 5 from a direct microbench sweep over the 5 BF16_bwd
metric shapes that fell <0.97 in round 4 — see /tmp/bench_bf16_bwd_round2.log
archived in the round-5 commit message: 30 (group_m, num_xcds) candidates
swept on each of 14 (layout, m, n, k) tuples, picking empirical max;
extended again at round 7 with a 30-config sweep over the BF16_bwd /
BF16_fwd shapes that still fell <0.97 / <0.97 — see
/tmp/bench_bwd_sweep.log + /tmp/bench_rule_validation.log archived in
the round-7 commit message):
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
       - 4096x4096x12288:   bench round-2 (gm=2, xcd=32) -> 1555.2 TF
                            (default 1542.9, +0.8pp). Round-5 extension.
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
  * BF16 backward (round 5): ``RRR tiles_m == 32 + tiles_n >= 32`` shapes
    show (gm=24, xcd=4) winning over the round-1 (gm=2, xcd=16) — sweep
    on dA (8192, 14336, 4096) gives 1467.5 / 1443.5 = +1.7pp. The same
    rule subset narrowed away from the small-tiles_m ``(rrr tiles_m=16,
    tiles_n>=32, k<=8192)`` family because the sweep there shows the
    default ``(4, 8)`` already wins (1411.7 TF; round-1 (2, 16) was
    1395.3 TF / -1.2pp).
  * BF16 backward (round 5) CRR additions for shapes that were falling
    to the binding default:
      - ``32 <= tiles_m < 64`` and ``tiles_n == 16`` and ``k <= 4096``:
        (2, 32). Anchor: dB-after-swap (12288, 4096, 4096) — bench
        1414.7 vs default 1393.9 = +1.5pp.
      - ``tiles_m <= 16`` and ``tiles_n >= 32`` and ``k > 4096``:
        (2, 32). Anchor: dB-after-swap (4096, 14336, 8192) — bench
        1330.2 vs default 1313.2 = +1.3pp. The rule is intentionally
        ``crr``-scoped to avoid colliding with ``rrr``-only patterns.
  * Round 7 — BF16 forward RCR rules covering the 4 "long-N" /
    "deep-K skinny" forward shapes that still fell ``<0.97`` after
    round 5 (cube + skinny-tall covered only the 2 small-tile cases):
      - ``rcr`` and ``tiles_n >= 86`` and ``k <= 4096``: (24, 2). Covers
        ``(4096,22016,4096)``, ``(4096,28672,4096)``, ``(8192,22016,4096)``,
        ``(8192,28672,4096)``. 30-config sweep on all 4 shapes shows
        ``(24, 2)`` ties or wins for each (gains: +1.6pp / +1.2pp /
        +2.6pp / +1.0pp over default ``(4, 8)``). Cache configs vary
        across these four shapes — (24,2) / (4,16) / (2,32) / (16,4) —
        but the empirical sweep collapses them to a single winner.
      - ``rcr`` and ``tiles_m == 32`` and ``tiles_n == 16`` and
        ``k >= 11008``: (2, 32). Covers ``(8192,4096,11008)`` and
        ``(8192,4096,14336)``. Same config the existing skinny-tall
        rule picks at ``k <= 4096`` — keeps the rcr ``(32,16,*)``
        family coherent. Bench: +1.7pp on k=11008, +1.9pp on k=14336.
  * Round 7 — BF16 backward RRR additions for the dA shapes that fall
    through to default in BF16_bwd:
      - ``rrr`` and ``tiles_m <= 16`` and ``tiles_m == tiles_n`` and
        ``12288 < k <= 32768``: (24, 2). Anchor: dA RRR
        ``(4096, 4096, 22016)`` for fwd ``(4096, 22016, 4096)`` —
        bench 1450.2 vs default 1433.8 = +1.1pp. Extends the cube
        rule's K bound for the deeper-K tier of square cubes.
      - ``rrr`` and ``tiles_m <= 16`` and ``32 <= tiles_n < 64`` and
        ``k <= 4096``: (2, 32). Anchor: dA RRR ``(4096, 11008, 4096)``
        for fwd ``(4096, 4096, 11008)`` — bench 1350.5 vs default
        1322.6 = +2.1pp. Same config as the cube rule, just relaxes
        the tiles_m == tiles_n constraint.
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
        # BF16 tile-geometry rules. Forward (RCR) rules derived from
        # analysis/bf16_gemm/mi350x/bench_bf16_no_jit_final.json (round
        # 3). Backward (RRR/CRR) rules added in round 4 from an initial
        # 10-candidate microbench, then refined in round 5 with a wider
        # 30-candidate sweep on the 5 BF16_bwd Llama shapes that still
        # fell <0.97 (see /tmp/bench_bf16_bwd_round2.log archived in
        # the round-5 commit message). Round 7 added 4 more rules
        # (rcr long-N, rcr deep-K skinny, rrr cube deep-K, rrr small-M
        # medium-N) to push BF16_bwd above 0.97 — see
        # /tmp/bench_rule_validation.log.
        tiles_m = m // 256
        tiles_n = n // 256
        if tiles_m <= 16 and tiles_m == tiles_n and k <= 12288:
            # Cube-ish small (16x16 grid). Round 1: 4096^3 fwd RCR and
            # 4096x4096x11008 win on (gm=2, xcd=32). Round 5 extends the
            # K bound from 11008 to 12288 to capture dA RRR
            # (4096, 4096, 12288) (= round-2 sweep best 1555.2 TF, +0.8pp
            # over default). 22016-cube falls through (sweep shows
            # default still wins by 0.6pp there) and any LLM K above
            # 12288 is non-cube in the metric suite, so the wider bound
            # has no collateral. Also catches the dB-after-swap RCR-fwd
            # cube cousins (e.g. 8192x4096x4096 -> dB (4096,4096,8192))
            # that already passed in round 4.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if tiles_n == 16 and tiles_m == 2 * tiles_n and k <= 4096:
            # Skinny tall (32x16 grid, K shallow): canonical attn_out
            # shape 8192x4096x4096. Cache rcr (2, 16); rrr/crr (2, 32);
            # we pick (2, 32) because it is within ~1pp of (2, 16) on
            # rcr and matches the rrr/crr backward-pass winners exactly.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if layout == "rcr" and tiles_n >= 86 and k <= 4096:
            # Round-7 rule. Long-N RCR with shallow K — covers
            # Llama-2-7B mlp_gate_up (4096x22016x4096) and gpt_oss /
            # Llama-3 mlp variants (4096x28672x4096, 8192x22016x4096,
            # 8192x28672x4096). 30-config sweep on all 4 shapes shows
            # (24, 2) ties or wins each (gains +1.6 to +2.6pp over the
            # binding default). The cache picks differ shape-by-shape
            # (24,2 / 4,16 / 2,32 / 16,4) but the empirical sweep
            # collapses them to a single robust winner.
            return HipKittenConfig(layout=layout, group_m=24, num_xcds=2, kernel=None)
        if layout == "rcr" and tiles_m == 32 and tiles_n == 16 and k >= 11008:
            # Round-7 rule. Deep-K skinny RCR (32x16 grid). Covers
            # Llama mlp_down forward shapes (8192x4096x11008,
            # 8192x4096x14336). Same config as the skinny-tall rule
            # at shallow K — keeps the (32,16,*) family coherent.
            # Bench: +1.7pp on k=11008, +1.9pp on k=14336 over default.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if layout == "rrr" and tiles_m <= 16 and tiles_m == tiles_n and 12288 < k <= 32768:
            # Round-7 rule. Deep-K cube RRR. Anchor: dA RRR
            # (4096, 4096, 22016) for fwd (4096, 22016, 4096) —
            # +1.1pp over default. Extends the cube rule (which
            # caps at k<=12288) into the deeper-K tier without
            # overlap.
            return HipKittenConfig(layout=layout, group_m=24, num_xcds=2, kernel=None)
        if layout == "rrr" and tiles_m <= 16 and 32 <= tiles_n < 64 and k <= 4096:
            # Round-7 rule. Small-M medium-N shallow-K RRR. Anchor:
            # dA RRR (4096, 11008, 4096) for fwd (4096, 4096, 11008)
            # — +2.1pp over default. Same config as the cube rule;
            # the tiles_n band intentionally excludes the >=64 family
            # which the cache shows benefits from a different config.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if layout == "rrr" and tiles_m == 32 and tiles_n >= 32 and k <= 8192:
            # Tall-N RRR with tiles_m EXACTLY 32. Canonical: dA shape
            # from turbo.ops.gemm autograd backward of fwd
            # 8192xN_largex4096 (RRR (8192, N_large, 4096)).
            # Round-2 30-candidate sweep on (8192, 14336, 4096) gives
            # (24, 4) = 1467.5 TF, beating round-1 winner (2, 16) by
            # +1.7pp and default (4, 8) by +3.3pp. Distinguish from
            # tiles_m==16 RRR (which the same sweep shows prefers the
            # default (4, 8) — e.g. dA RRR (4096, 11008, 4096): default
            # 1411.7, (2, 16) 1395.3, (24, 4) << default), so the bound
            # is exact (==32) rather than the round-1 inclusive (<=32).
            return HipKittenConfig(layout=layout, group_m=24, num_xcds=4, kernel=None)
        if layout == "crr" and tiles_m >= 64 and tiles_n == 16:
            # Long-N backward dB-after-swap (CRR sees logical (N_fwd,
            # K_fwd, M_fwd) post-swap). Canonical Llama-2-7B
            # mlp_gate_up backward dB:
            #   fwd 4096x22016x4096 -> dB CRR (22016,4096,4096) = (86,16,4096)
            # and Llama-3.1-8B
            #   fwd 8192x28672x4096 -> dB CRR (28672,4096,8192) = (112,16,8192)
            # Bench: shallow K (=4096) prefers (gm=24, xcd=2) by +1.4pp;
            # deeper K (=8192) prefers (gm=2, xcd=32) by +0.2pp. The K
            # split mirrors the cache row family for these long-N CRR
            # shapes.
            if k <= 4096:
                return HipKittenConfig(layout=layout, group_m=24, num_xcds=2, kernel=None)
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if layout == "crr" and 32 <= tiles_m < 64 and tiles_n == 16 and k <= 4096:
            # Mid-tiles_m CRR with tiles_n==16, K shallow. Canonical
            # Llama-2-7B attn_qkv backward dB:
            #   fwd 4096x12288x4096 -> dB CRR (12288, 4096, 4096) = (48, 16, 4096)
            # Round-2 sweep: (2, 32) = 1414.7 TF beats default (4, 8) by
            # +1.5pp. (24, 2) — the tiles_m>=64 winner — only reaches
            # 1393.5 here, so the rule is rightfully tile-bucketed
            # rather than just adding shape 48 to the long-N rule.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if layout == "crr" and tiles_m <= 16 and tiles_n >= 32 and k > 4096:
            # Asymmetric tall-N CRR with deep K. Canonical: dB-after-swap
            # for fwd 8192x4096x14336 -> CRR (4096, 14336, 8192) =
            # (16, 56, 8192). Round-2 sweep: (2, 32) = 1330.2 TF, +1.3pp
            # over default (4, 8) and within 0.04pp of the empirical max
            # (6, 8) = 1330.2 (chose (2, 32) because it is the same
            # config the long-N CRR rule above picks for the deep-K
            # tier, keeping the rule family coherent).
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
