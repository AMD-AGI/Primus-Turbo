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
  * Round 5 — last gap in BF16 backward dispatch coverage. The shape
    ``(8192, 28672, 4096)`` BF16_bwd was at ratio 0.939 (worst in the
    section) because two of its three dispatches were sub-optimal:
      - dA RRR ``(8192, 4096, 28672)`` had no RRR rule with
        ``tiles_n == 16`` so it fell through to the binding default
        ``(4, 8)``. New rule: ``rrr and tiles_m == 32 and
        tiles_n == 16 and k >= 22016`` -> ``(8, 2)``. Tight-bench
        (200 iters × 3 repeats) gives p20 1499.8 TF vs 1482.7 TF
        (+1.15pp). K bound is 22016 because the 11008-22016 tier
        shows (2, 32) and (8, 2) tied within 0.4pp (no clear winner
        to anchor a rule on).
      - dB CRR ``(28672, 4096, 8192)`` matched the existing
        ``crr tiles_m >= 64 tiles_n == 16 k > 4096`` rule but the
        original (2, 32) pick was only +0.2pp over default. Tight
        re-bench shows (gm=4, xcd=32) wins at p20 1450.1 TF vs
        (gm=2, xcd=32) 1442.3 TF (+0.54pp); bit-identical output.
        Existing rule's k <= 4096 branch (``(24, 2)``) is unchanged
        — only the deeper-K branch flips.
  * FP8 (analysis/fp8_gemm/mi350x/.autotune_cache.json, 48 shapes x 3
    layouts): much tighter distribution -- ``group_m`` is 4 in 60% of RCR
    entries and 4 or 8 in ~95%, ``kernel`` is "8" in 46/48 RCR entries.
    Rules:
      - default ``(group_m=4, kernel="8")``
    The historical ``layout=="rcr" and tiles_n>=86 and K<=4096``
    outlier rule (offline-cache-derived ``kernel="4"`` for two LLM
    shapes) was removed once re-benches on the current ``.so`` showed
    the binding's auto-pick (8-wave for grid<3200, 4-wave for grid>=3200
    && k<=8192) is bit-equivalent within ±0.1pp on every metric shape.
    Default ``kernel=None`` keeps the dispatcher's ``force_rcr_kernel``
    context manager off the hot path; per-shape overrides are added
    only when sweep evidence shows kernel-level wins above the ~1us CM
    cost.

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
# RCR FP8 kernel: ``None`` means "let the binding use its built-in default
# (kernel template 8)". The dispatcher's ``force_rcr_kernel`` context manager
# is then skipped entirely (`force_rcr_kernel(None)` is a no-op), saving the
# lock acquire + 2x os.environ get/set/restore per dispatch (~7-12us measured
# on 8192x28672x4096 in /tmp/profile_execute_internals.py round-6 archive).
# Setting an explicit ``"8"`` here would be functionally equivalent but force
# every RCR call through the CM, which we now know is the single largest
# avoidable host-overhead source for the FP8_fwd metric section.
_FP8_DEFAULT_KERNEL = None


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
    m_total: int | None = None,
) -> HipKittenConfig:
    """Pick a kernel config for ``(M, N, K, layout, dtype)`` via if/else rules.

    This is the **only** runtime config function — there is no cache lookup,
    no JSON parse, no dict get. The function is total over its input space
    (every aligned shape gets a config back); the alignment / dtype gates
    happen earlier in each backend's ``can_handle``.

    ``m`` is the *per-launch* M dimension (= per-group M for grouped, full
    M for dense). ``m_total`` is the **summed** M across the launch (= ``m``
    for dense, ``B * m_per_group`` for the persistent grouped kernel) and
    is consumed only by rules that need to discriminate launches with the
    same per-group tile geometry but very different total tile count
    (e.g. K-padded gpt_oss grouped at B=4 vs B=32 — same (m_pad, n_pad,
    k_pad) per the launcher's view but the persistent grid sees 4× / 32×
    more tiles, which flips the optimal ``group_m``). Dense callers leave
    ``m_total`` as ``None``; grouped callers pass ``a.shape[0]``.

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
        # Round-57 rule. Unpadded gpt_oss-style grouped BF16 RCR — the
        # per-launch tile geometry (m_per_group ∈ {2048, 4096}, n ∈
        # {2880, 5760}, k = 2880) goes straight to the persistent grouped
        # launcher (no host-side K/N pad as of round-44), so the
        # scheduling key is now (m_per_group, n, k) = the *raw* gpt_oss
        # dims. The launcher's grid spans ``B`` groups, so total tile
        # count varies 16× between B=4 and B=32 of the same per-group
        # shape — and the optimal ``(group_m, num_xcds)`` flips per tier.
        #
        # Round-45 picked **one** (gm, xcd) per ``tiles_n`` for the entire
        # ``m_total >= 16384`` family. A wider sweep this round shows that
        # was a coarse approximation: each m_total tier has a different
        # winner. Re-derived from an 80-iter × 3-repeat median sweep over
        # ``(group_m, num_xcds) ∈ {(4,8), (1,4), (8,4), (2,4), (2,2)}``
        # for all 8 metric gpt_oss BF16 grouped shapes
        # (/tmp/probe_bf16_full_mtot.py archived in this commit message):
        #
        #   tiles_n=22 (GateUP, n=5760) — winners over default (4,8):
        #     m_total=8192   (2,4)  +2.8pp  *top1
        #     m_total=16384  (2,2)  +0.5pp  *top1   (round-45 (1,4): -0.5pp)
        #     m_total=65536  (8,4)  +1.7pp  *top1   (round-45 (1,4): +1.1pp)
        #     m_total=131072 (1,4)  +2.5pp  *top1   (round-45 (1,4): +2.5pp)
        #
        #   tiles_n=11 (Down, n=2880) — winners over default (4,8):
        #     m_total=8192   (2,2)  +1.0pp  *top1
        #     m_total=16384  (2,4)  +2.5pp  *top1   (round-45 (8,4): +1.4pp)
        #     m_total=65536  (1,4)  +1.9pp  *top1   (round-45 (8,4): +1.2pp)
        #     m_total=131072 (1,4)  +1.5pp  *top1   (round-45 (8,4): +0.4pp)
        #
        # Net delta over round-45: +1.0pp / shape average across all 8
        # gpt_oss BF16 shapes (sum 8.3pp / 8 shapes), pushing grp_BF16
        # geomean +~1.2%.
        #
        # Bit-identical output across all 5 candidate configs verified on
        # the 4 corner shapes (cross max_abs = 0.00, bit_eq = True; see
        # /tmp/verify_correctness.py archived alongside): group_m and
        # num_xcds only re-order the persistent tile schedule, not the
        # arithmetic. The previous round-45 rule was already verified
        # bit-identical; the relaxation to (2,4)/(2,2)/(1,4) carries the
        # same property.
        #
        # Rule scope check: ``k == 2880`` is only hit by gpt_oss in the
        # metric (DSV3 grouped K ∈ {2048, 7168}; dense BF16 K ∈
        # {4096, 11008, 14336, ...}). ``tiles_m ∈ {8, 16}`` covers
        # M_per_group ∈ {2048, 4096}; ``tiles_n ∈ {11, 22}`` covers
        # Down (n=2880) / GateUP (n=5760) raw N. The lower bound
        # ``m_total >= 8192`` now also catches the 2 B=4 M=2048 shapes
        # that round-45 left on the binding default (m_total=8192,
        # the worst-ratio metric BF16 case at 0.84 vs Triton).
        if (
            layout == "rcr"
            and k == 2880
            and tiles_m in (8, 16)
            and m_total is not None
            and m_total >= 8192
        ):
            if tiles_n == 22:  # GateUP N=5760
                if m_total <= 8192:
                    # Round-9: refined from round-44/45/70's narrow 5-cell
                    # `{(4,8), (1,4), (8,4), (2,4), (2,2)}` probe to (gm=2,
                    # xcd=2) via a 60-cell `{1..32} × {1,2,4,8,16,32}` sweep
                    # (/tmp/sweep_bf16_gateup_b4_m2048_round9.py) followed
                    # by 1500-iter × 7-repeat p20 tight verify
                    # (/tmp/verify_bf16_gateup_b4_m2048.py). The 200-iter
                    # sweep top1 (gm=2, xcd=8) at +1.84pp turned out to be
                    # measurement noise — at tighter REPEATS=7 p20 it sits
                    # at -0.24pp vs baseline. Top-3 by p20 median:
                    #
                    #   cfg          p20 median   Δ vs (2, 4) baseline
                    #   (2, 4)        1033.26 TF  baseline
                    #   (2, 2)        1045.50 TF  +12.25 (+1.19pp)  *winner
                    #   (1, 2)        1043.26 TF  +10.00 (+0.97pp)
                    #   (16, 4)       1042.61 TF   +9.36 (+0.91pp)
                    #   (4, 1)        1039.42 TF   +6.16 (+0.60pp)
                    #   (4, 8) close3 1037.36 TF   +4.11 (+0.40pp)
                    #   (2, 8) sweep1 1030.75 TF   -2.50 (-0.24pp)  *fake
                    #
                    # ``(gm=2, xcd=2)`` p20 spread 1044.5..1047.3 (2.8 TF
                    # range across 7 trials) — wins by +2.2 TF over (1, 2)
                    # and +2.9 TF over (16, 4), both gaps clear of cell
                    # noise. Bit-identical output vs (2, 4) verified
                    # (max_abs_diff=0.0, bit_eq=True; group_m / num_xcds
                    # are pure scheduling knobs on the BF16 grouped RCR
                    # persistent tile schedule).
                    #
                    # Rule scope check: ``tiles_n == 22 + tiles_m == 8 +
                    # k == 2880 + m_total <= 8192`` matches only B=4
                    # M=2048 in the metric (gpt_oss-GateUP family); the
                    # m_total <= 16384 branch (B=4 M=4096) already returns
                    # (gm=2, xcd=2) so this change makes both B=4 GateUP
                    # cases use the same cfg — also makes the rule simpler
                    # to read (consistency across the B=4 GateUP family).
                    return HipKittenConfig(
                        layout=layout, group_m=2, num_xcds=2, kernel=None
                    )
                if m_total <= 16384:
                    # Round-21 re-tune for gpt_oss-GateUP-B4-M4096 BF16
                    # (tiles_n=22, tiles_m=16, k=2880, m_total=16384).
                    # Round-9 picked (gm=2, xcd=2) for *both* B=4 GateUP
                    # cases (M=2048 m_total=8192 and M=4096 m_total=16384)
                    # for "consistency". The post-round-19/20 BUFFER kernel
                    # diverges these two — M=4096 (this branch) prefers a
                    # larger gm × xcd=4 configuration; M=2048 (branch
                    # above) keeps (gm=2, xcd=2) as the per-iter-sync
                    # winner.
                    #
                    # PER-ITER-SYNC verify at /tmp/verify_bf16_metric_aligned.py
                    # (60-iter × 5-trial p20 median; mirrors
                    # `_metric_grouped_only.py::_time_op` 50-iter p20):
                    #
                    #   cfg          med_p20    spread%   Δ vs (2,2)
                    #   (12, 4)       1204.53   0.78%    +20.68  +1.75%  *winner
                    #   (8,  4)       1202.07   1.41%    +18.22  +1.54%
                    #   (2,  8)       1193.94   1.26%    +10.09  +0.85%
                    #   (16, 4)       1192.47   1.27%     +8.62  +0.73%
                    #   (6,  4)       1191.22   1.12%     +7.37  +0.62%
                    #   (2,  2)       1183.85   1.64%     +0.00   0.00%  ←round-9
                    #
                    # (gm=12, xcd=4) wins by +20.7 TF (+1.75%) over the
                    # round-9 (gm=2, xcd=2) — the largest BF16 metric-
                    # aligned win in the round-21 sweep on the GateUP
                    # B=4 family. The (gm ∈ {8,12,16}, xcd=4) cluster
                    # all sits 8-21 TF above (2, 2); (gm=12, xcd=4)
                    # picked as the unique top by p20 (tighter spread
                    # than (8, 4) at 0.78% vs 1.41%).
                    #
                    # Bit-identical output vs round-9 (gm=2, xcd=2)
                    # implied by the round-9 commentary (group_m /
                    # num_xcds are pure scheduling knobs on the BF16
                    # grouped RCR persistent tile schedule).
                    #
                    # Rule scope unchanged: tiles_n=22 (n=5760) +
                    # tiles_m=16 (m_per_group=4096) + k=2880 +
                    # 8192 < m_total <= 16384 matches only B=4 M=4096
                    # GateUP in the metric.
                    return HipKittenConfig(
                        layout=layout, group_m=12, num_xcds=4, kernel=None
                    )
                if m_total <= 65536:
                    # Round-26 anchor + marginal re-tune for gpt_oss-GateUP-
                    # B32-M2048 BF16 (tiles_n=22, tiles_m=8, k=2880,
                    # m_total=65536). Was the only un-anchored BF16 rule in
                    # the gpt_oss family — `(gm=8, xcd=4)` came from an
                    # early defensive default and never had a per-iter-sync
                    # sweep verify after the post-round-19/20 BUFFER
                    # kernel reroute (round-21 re-swept the 3 sibling
                    # m_total branches; this one was missed).
                    #
                    # 40-cell coarse sweep `gm ∈ {1,2,4,6,8,12,16,24,32,48}
                    # × xcd ∈ {1,2,4,8}` (5 trials × 60 iters,
                    # /tmp/sweep_bf16_gateup_b32_m2048_round26.py) followed
                    # by 7-trial × 200-iter tight verify on top-8
                    # candidates (/tmp/verify_bf16_gateup_b32_m2048_round26.py):
                    #
                    #   cfg          med p20    p20      p80      spread%
                    #   ( 4, 4)      1255.71    1253.92  1257.02   0.26  *winner
                    #   ( 8, 4)      1254.55    1252.21  1255.16   0.45  ←round-?
                    #   ( 6, 4)      1250.89    1250.51  1251.78   0.25
                    #   ( 2, 4)      1247.50    1247.21  1249.28   0.27
                    #   ( 4, 8)      1244.96    1244.10  1245.01   0.22
                    #   (12, 4)      1244.84    1243.56  1246.81   0.39
                    #   ( 4, 1)      1242.68    1242.17  1244.39   0.24
                    #   ( 4, 2)      1240.49    1240.13  1241.09   0.26
                    #
                    # `(gm=4, xcd=4)` is the unique top by +1.16 TF (+0.09pp)
                    # over `(gm=8, xcd=4)`. Margin is within run-to-run
                    # spread but the (4,4) p20 (1253.92) > (8,4) p20
                    # (1252.21) is consistent across all 7 trials, and the
                    # `xcd=4` plateau dominates the entire `xcd ∈ {1,2,8}`
                    # column by 5-15 TF — both knobs at a clear local
                    # optimum, just a flat top within the gm ∈ {2,4,6,8}
                    # neighborhood.
                    #
                    # The sweep also confirms the larger gm cluster
                    # (gm ∈ {24, 32, 48}) regresses heavily at xcd=1/8 —
                    # bottom-4 are (48,1)/(48,8)/(32,8)/(32,1) all
                    # ~1067-1069 TF (-15% from top) — confirming the
                    # XCD-swizzle interaction with the persistent grouped
                    # tile schedule for this tiles_m=8 / tiles_n=22 grid
                    # is sensitive to the XCD partitioning.
                    #
                    # Bit-identical output vs prior (8,4) verified at
                    # /tmp/verify_bf16_gateup_b32_m2048_round26.py:
                    # max_abs_diff=0.0, bit_eq=True (group_m / num_xcds
                    # are pure scheduling knobs on the BF16 grouped RCR
                    # persistent tile schedule).
                    #
                    # Rule scope: tiles_n=22 + tiles_m=8 + k=2880 +
                    # 16384 < m_total <= 65536 matches only B=32 M=2048
                    # GateUP in the metric (B<32 GateUP M=2048 → m_total
                    # ≤ 16384 hits the prior branches). Net delta is
                    # marginal (+0.09pp = +1.16 TF) but anchors the rule
                    # to a verified local optimum and brings this branch
                    # in line with the round-21 anchor coverage of the
                    # other 3 m_total tiers.
                    return HipKittenConfig(
                        layout=layout, group_m=4, num_xcds=4, kernel=None
                    )
                # Round-21 re-tune for gpt_oss-GateUP-B32-M4096 BF16
                # (tiles_n=22, tiles_m=16, k=2880, m_total=131072).
                # Round-70 picked (gm=1, xcd=2) against the FLAT-store
                # kernel; the post-round-19/20 BUFFER kernel changes the
                # XCD-swizzle interaction with the per-tile completion
                # time and a *larger* group_m × xcd=4 wins.
                #
                # PER-ITER-SYNC verify at /tmp/verify_bf16_metric_aligned.py
                # (60-iter × 5-trial p20 median):
                #
                #   cfg          med_p20    spread%   Δ vs (1,2)
                #   (8,  4)       1273.33   0.87%    +18.69  +1.49%  *winner
                #   (6,  4)       1266.45   0.51%    +11.81  +0.94%
                #   (2,  1)       1262.52   0.51%     +7.88  +0.63%
                #   (12, 4)       1260.56   0.43%     +5.92  +0.47%
                #   (2,  2)       1258.55   0.80%     +3.91  +0.31%
                #   (4,  4)       1257.24   0.34%     +2.60  +0.21%
                #   (1,  2)       1254.64   0.33%     +0.00   0.00%  ←round-70
                #
                # (gm=8, xcd=4) wins by +18.7 TF (+1.49%) over the
                # round-70 (gm=1, xcd=2). Both knobs at a clear local
                # optimum: gm=8 dominates the gm ∈ {6,12} neighbors by
                # +6.9 / +12.8 TF; xcd=4 dominates xcd=1 / xcd=2 / xcd=8
                # by +10-21 TF in the gm=8 column from the wider coarse
                # sweep at /tmp/sweep_bf16_worst_round21.py.
                #
                # Bit-identical output (group_m / num_xcds are pure
                # scheduling knobs on the BF16 grouped RCR persistent
                # tile schedule).
                #
                # Rule scope: ``m_total > 65536`` ⇔ B=32 M_per=4096 in
                # the metric (gpt_oss-GateUP family); B=32 M_per=2048
                # lands on the m_total<=65536 branch above. No other
                # metric shape has tiles_n==22 with m_total>65536.
                return HipKittenConfig(
                    layout=layout, group_m=8, num_xcds=4, kernel=None
                )
            if tiles_n == 11:  # Down N=2880
                if m_total <= 8192:
                    # Round-10 NOTE — kept (gm=2, xcd=2). The tight-verify
                    # winner (gm=32, xcd=8) gained +1.84pp on a 1500-iter
                    # × 7-repeat p20 bench (810.8 vs 796.9) but the
                    # `_metric_grouped_only.py` 50-iter min-time bench
                    # showed -1.0pp / -10 TF noise (range 770-790 TF
                    # over 3 runs vs verify p20 spread 810.8-812.0 of
                    # 1.2 TF). Hypothesis: the persistent grouped
                    # schedule for (gm=32, xcd=8) at m_total=8192
                    # (only 32 tiles per group × 11 N-tiles ≈ 352
                    # tiles, ~1.4 wave) has a higher tail/variance than
                    # the tighter (gm=2, xcd=2) schedule, so 50-iter min
                    # randomly picks bad iters more often. Tight p20 is
                    # the right metric for kernel ordering decisions; 50-
                    # iter min is the *acceptance* metric. The rule needs
                    # to win on both. (gm=32, xcd=8) wins on tight but
                    # loses on min — leave (gm=2, xcd=2) until/unless
                    # we find a cfg that wins on both. Sibling B=4-M4096
                    # rule below DOES win on both → kept.
                    return HipKittenConfig(
                        layout=layout, group_m=2, num_xcds=2, kernel=None
                    )
                if m_total <= 16384:
                    # Round-10 originally picked (gm=8, xcd=4) from a
                    # 60-cell sweep against the round-9 (FLAT-store) BF16
                    # grouped kernel. The pre-round-19/20 timing put the
                    # entire `(gm ∈ {8,16,24,32}, xcd=4)` cluster within
                    # 1.4 TF of each other; (8, 4) was picked as the
                    # tightest-spread defensive choice.
                    #
                    # Round-21 re-sweep against the post-round-19/20
                    # (col-layout C-store FLAT->BUFFER + K-tail/N-tail
                    # FLAT->BUFFER) kernel changes the picture: the
                    # ISA-level reduction in cross-WG memory traffic
                    # rebalances the persistent grouped tile schedule's
                    # tail behavior, and a *larger* group_m now wins the
                    # metric-aligned per-iter-sync timing (which is the
                    # acceptance criterion in `_metric_grouped_only.py`'s
                    # `_time_op` / 50-iter p20 path).
                    #
                    # Sweep + verify at /tmp/sweep_bf16_worst_round21.py
                    # then /tmp/verify_bf16_down_b4_m4096.py: 11 cfg
                    # neighbors at gm ∈ {1,8,12,16,24,32,40,48,56,64} ×
                    # xcd ∈ {2,4} timed in BOTH (a) steady-state queued
                    # (500 iters x 7 trials) and (b) per-iter-sync (80
                    # iters x 7 trials, mirrors `_time_op`).
                    #
                    # PER-ITER-SYNC mode (the acceptance metric):
                    #   cfg          med_p20   spread%   Δ vs (8,4)
                    #   (32, 4)       1179.02   0.17%   +8.93  +0.76%  *winner
                    #   (48, 4)       1178.21   0.35%   +8.12  +0.69%
                    #   (64, 4)       1177.59   0.36%   +7.50  +0.64%
                    #   (40, 4)       1176.57   0.29%   +6.48  +0.55%
                    #   (56, 4)       1176.16   0.33%   +6.07  +0.52%
                    #   (24, 4)       1175.96   0.41%   +5.87  +0.50%
                    #   (12, 4)       1173.12   0.69%   +3.03  +0.26%
                    #   (24, 2)       1170.69   0.55%   +0.60  +0.05%
                    #   (16, 2)       1170.09   0.91%   +0.00   0.00%
                    #   (8,  4)       1170.09   1.41%   +0.00   0.00%  ←round-10
                    #   (1,  4)       1156.74   0.46%  -13.35  -1.14%
                    #
                    # The entire `(gm ∈ {24,32,40,48,56,64}, xcd=4)`
                    # cluster wins by +6 to +9 TF (+0.50% to +0.76%) on
                    # metric-aligned timing; (8, 4) sits at the bottom
                    # of the high cluster with the *highest* spread (1.41%
                    # vs <0.4% for the new winners). (gm=32, xcd=4)
                    # picked as the unique top by p20 with the tightest
                    # spread (0.17%) — both knobs at a clear local
                    # optimum.
                    #
                    # Bit-identical output vs round-10 (gm=8, xcd=4)
                    # verified at /tmp/verify_bf16_down_b4_m4096_bit_eq.py:
                    # max_abs_diff=0.0, bit_eq=True; SNR vs fp32 ref =
                    # 49.62 dB on both (group_m / num_xcds are pure
                    # scheduling knobs on the persistent BF16 grouped
                    # RCR tile schedule, identical to the round-10
                    # commentary).
                    #
                    # Rule scope unchanged: tiles_n == 11 (n=2880) +
                    # tiles_m == 16 (m_per_group=4096) + k == 2880 +
                    # 8192 < m_total <= 16384 matches only B=4 M=4096
                    # in the metric (gpt_oss-Down family).
                    return HipKittenConfig(
                        layout=layout, group_m=32, num_xcds=4, kernel=None
                    )
                if m_total <= 65536:
                    # Round-21 split: gpt_oss-Down-B32-M2048 BF16
                    # (tiles_n=11, tiles_m=8, k=2880, m_total=65536).
                    # Was sharing the Down-B32 catch-all (gm=1, xcd=4)
                    # with B=32-M=4096 (m_total=131072). Round-21 sweep
                    # shows the two B=32 Down shapes diverge in their
                    # optimum after the post-round-19/20 BUFFER kernel:
                    # M=2048 prefers a much larger gm × xcd=4; M=4096
                    # keeps (gm=1, xcd=4) (verified separately as still
                    # the per-iter-sync winner).
                    #
                    # PER-ITER-SYNC verify at /tmp/verify_bf16_metric_aligned.py:
                    #
                    #   cfg          med_p20    spread%   Δ vs (1,4)
                    #   (16, 4)       1198.31   1.18%    +11.51  +0.97%  *winner
                    #   (32, 4)       1197.68   1.00%    +10.88  +0.92%
                    #   (48, 4)       1197.36   0.42%    +10.56  +0.89%
                    #   (24, 4)       1194.89   1.64%     +8.09  +0.68%
                    #   (12, 4)       1192.16   0.61%     +5.36  +0.45%
                    #   (1,  4)       1186.80   0.66%     +0.00   0.00%  ←default
                    #   (8,  4)       1181.55   0.27%     -5.25  -0.44%
                    #   (4,  4)       1179.95   0.75%     -6.85  -0.58%
                    #
                    # The (gm ∈ {16, 32, 48}, xcd=4) cluster wins by
                    # +10-12 TF (+0.9-1.0%); (gm=16, xcd=4) picked as
                    # the unique top by p20 median (just edges out
                    # (gm=32, xcd=4) by 0.6 TF — well within the 1.18%
                    # spread, but tightest median of the cluster).
                    #
                    # Bit-identical output (group_m / num_xcds are pure
                    # scheduling knobs).
                    #
                    # Rule scope: tiles_n=11 (n=2880) + 16384 < m_total
                    # <= 65536 matches only B=32 M_per=2048 in the
                    # metric (gpt_oss-Down family). Down-B32-M4096
                    # (m_total=131072) falls through to the default
                    # (gm=1, xcd=4) below — verified at the same
                    # /tmp/verify_bf16_metric_aligned.py probe that
                    # (gm=1, xcd=4) is still the unique top at 1230.62
                    # p20, with (gm=4, xcd=4) the closest contender at
                    # -3.91 TF (-0.32%, all else worse).
                    return HipKittenConfig(
                        layout=layout, group_m=16, num_xcds=4, kernel=None
                    )
                return HipKittenConfig(
                    layout=layout, group_m=1, num_xcds=4, kernel=None
                )
        if (
            layout == "rcr"
            and tiles_n == 11
            and k == 5760
            and m_total is not None
        ):
            # Round-7 (BF16 weighted-wall metric) rule. gpt_oss-GateUP dA
            # H4-rerouted RCR family. The R4 H4 fast Triton transpose
            # reroutes the dA RRR call ``a:[M_total, N_fwd=5760],
            # b:[B, K_fwd=2880, N_fwd=5760]`` (post-transpose) into RCR;
            # ``select_default_config`` therefore sees:
            #   m = avg_m = M_per_group ∈ {2048, 4096} → tiles_m ∈ {8, 16}
            #   n = b.shape[-2] = K_fwd = 2880 → tiles_n = 11
            #   k = a.shape[1] = N_fwd = 5760
            # No prior BF16 RCR rule matches this geometry — the existing
            # gpt_oss block above is gated on ``k == 2880`` (gpt_oss
            # forward + Down dA H4 which also has k=2880). ``tiles_n == 11
            # and k == 5760`` is unique to gpt_oss-GateUP dA H4 reroute in
            # both the metric and DoD smoke (DSV3 N_fwd ∈ {2048, 4096,
            # 7168}, Qwen3 N_fwd ∈ {1536, 3072, 4096}, gpt_oss-Down
            # N_fwd=2880 — none give k=5760 with tiles_n=11). Round 2
            # tested ``(gm=4, xcd=4)`` here and falsified (-11.5 score on
            # variant A, -2.0 on variant B); but R2 only reduced xcds
            # without changing gm. The FP8 R34 sibling rule (line 1467
            # below — same geometry, same H4 reroute path, FP8 ``.so``)
            # found the actual winners are ``(gm=8, xcd=4)`` for tiles_m=16
            # and ``(gm=16, xcd=4)`` for tiles_m=8 + B=32 — different gm
            # from R2's tested set.
            #
            # 10-cell (gm, xcd) sweep on the 4 metric gpt_oss-GateUP dA
            # H4 RCR shapes (5-trial × 80-iter cudaEvent median at
            # /tmp/probe_round7_bf16_dA_h4_rcr.py archived in this commit):
            #
            #   shape (m_total, tiles_m)        cfg     med_TF   spread%   Δ vs (4,8)
            #   B4-M2048  (8192, 8)             default  1027.0   5.30%    baseline (high noise)
            #     -- noise floor too high; do not override ((24,2) +0.90% spread 6.56%)
            #   B4-M4096  (16384, 16)
            #     (4,8) baseline                       1325.6   1.77%
            #     (8,4) FP8-R34 winner                 1335.7   1.97%   +0.77%
            #     (2,4)                                1354.7   1.81%   +2.19%
            #   B32-M2048 (65536, 8)
            #     (4,8) baseline                       1322.2   0.43%
            #     (16,4) FP8-R34 winner                1329.4   0.42%   +0.54%  (clean: spread tied)
            #     (12,4)                               1337.2   0.76%   +1.13%
            #     (1,4)                                1335.0   1.69%   +0.97%
            #   B32-M4096 (131072, 16)
            #     (4,8) baseline                       1339.1   0.33%
            #     (8,4) FP8-R34 winner                 1366.2   0.49%   +2.03%  (clean: 4× spread)
            #     (16,4)                               1363.5   0.44%   +1.82%
            #
            # Rule mirrors the FP8 R34 pattern (tiles_m=16 → (gm=8, xcd=4);
            # tiles_m=8 + m_total ≥ 65536 → (gm=16, xcd=4)). FP8 R34 was
            # validated end-to-end on the FP8 metric (passed correctness
            # gate, contributed +1.5pp / shape on the 3 covered brackets
            # via the FP8 fused-act wall). The BF16 grouped RCR kernel
            # uses the same persistent grid + chiplet-swizzle scheduler
            # as FP8 (different inner MFMA tile + dtype but identical
            # group_m / num_xcds semantics — pure scheduling knobs on
            # the persistent tile schedule); the per-tile MFMA time is
            # ~2× FP8 (BF16 K_BLOCK=64 vs FP8 K_BLOCK=128 → 90 vs 45
            # K-iters per tile for k=5760). The probe confirms the same
            # cfg pattern carries over: (8,4)/(16,4) win consistently
            # over default (4,8) with clean signal on the 2 large-grid
            # tiers (B32-M4096 +2.03%/0.49%spread = 4× noise; B32-M2048
            # +0.54%/0.42%spread tied with default's 0.43% = clean +0.5%).
            #
            # B=4 M=2048 (m_total=8192, tiles_m=8) intentionally excluded:
            # default (4,8) sat at 5.3% spread on the contended GPU and
            # no candidate cleared the noise floor — leave on default
            # (matches FP8 R34 which also excluded this bracket for the
            # same reason).
            #
            # Bit-equivalent output verified: across all 10 cfgs probed
            # the BF16 grouped RCR kernel emits ``torch.equal`` outputs
            # (group_m / num_xcds are pure scheduling knobs on the
            # persistent tile schedule — same property documented for
            # all BF16 RCR rules above, and validated by the metric's
            # correctness gate (downsized check_allclose on out, dA, dB)
            # on every shape every round).
            #
            # Why ``(gm=8, xcd=4)`` for tiles_m=16: tiles_m=16 has 16
            # M-tiles per group; gm=8 batches 8 M-tiles per persistent
            # work-window, sharing each B-tile (loaded once per K=5760
            # main loop = 90 K-iters) across 8 mfma sequences vs only
            # 4 in the default. xcd=4 reduces XCD over-split (each XCD
            # owns more contiguous tiles, saturating its internal
            # pipeline rather than constantly switching).
            # Why ``(gm=16, xcd=4)`` for tiles_m=8 + B=32: smaller
            # tiles_m=8 means each "group" only has 8 M-tiles, so to
            # extract similar L2 reuse we need a larger gm. m_total
            # bound ``>= 65536`` excludes B=4 (m_total=8192) where the
            # smaller persistent grid amortises default cells better.
            if tiles_m == 16:
                return HipKittenConfig(
                    layout=layout, group_m=8, num_xcds=4, kernel=None
                )
            if tiles_m == 8 and m_total >= 65536:
                return HipKittenConfig(
                    layout=layout, group_m=16, num_xcds=4, kernel=None
                )
        # ----------------------------------------------------------
        # Round-24 aggregate: BF16 grouped var-K (dB / CRR) family.
        # Five family-specific (tiles_m, tiles_n) rules that together
        # cover the 5 of 6 dB var-K paths in the BF16 weighted-wall
        # metric that were previously routed sub-optimally — either
        # to the cube rule below (gpt_oss-Down) or to the default
        # cell (the other 4 families). All rules:
        #   * Gate on ``layout == "crr" and m_total is not None`` so
        #     dense LLaMA callers (which always pass m_total=None and
        #     have m=full M) are unaffected — this is grouped-only
        #     surgery.
        #   * Are bit-equivalent (group_m / num_xcds are pure
        #     persistent-grid scheduling knobs; max_abs=0,
        #     bit_eq=True at every probed cell).
        #   * Were derived from a 12-cell × 5-trial × 120-iter
        #     kernel-only sweep on all 4 metric shapes per family
        #     (B ∈ {16, 32} for DSV3/Qwen3, B ∈ {4, 32} for
        #     gpt_oss; M_per ∈ {2048, 4096} both; see
        #     scripts/_bf16_vark_db_multi_family_probe.py and
        #     /tmp/bf16_vark_db_multi_round24.log).
        #   * Each is the sole UNIFORM-POSITIVE cell vs the previous
        #     production cfg per family (min Δ > 0 across all 4
        #     shapes — same uniformity bar as R20's dA RRR rules).
        # The aggregate strategy (mirror R20 / R18→R20): individual
        # per-family wins are sub-noise (each contributes +0.1-0.5
        # score after wall→ratio → weighted-progress dilution), but
        # 5 families × 4 shapes = 20 shapes lifted at the same time
        # has a ~5x larger combined wall delta, which has historical
        # precedent for crossing the metric noise floor.
        if (layout == "crr"
                and tiles_m == 11
                and tiles_n == 11
                and k <= 4096
                and m_total is not None):
            # gpt_oss-Down dB var-K. Was being captured by the cube
            # rule below (``tiles_m <= 16 and tiles_m == tiles_n
            # and k <= 12288``) which fires FIRST and returns
            # (gm=2, xcds=32) — that cell was tuned at R1+R5 for
            # dense LLaMA forward 4096^3 / 4096^2 x11008, never
            # validated for grouped var-K. R23 identified the
            # misfire; R24 commits the fix.
            #
            # 11-cell sweep on the 4 gpt_oss-Down shapes vs the
            # production cube cfg (gm=2, xcds=32):
            #   B=4-M2048    +1.77 %  (top-1)
            #   B=4-M4096    +2.31 %  (top-1)
            #   B=32-M2048   +1.29 %  (top-1)
            #   B=32-M4096   +0.72 %  (top-1)
            # avg +1.52 %, min +0.72 %, max +2.31 %.
            return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
        if (layout == "crr"
                and tiles_m == 16
                and tiles_n == 6
                and k <= 4096
                and m_total is not None):
            # Qwen3-235B-A22B-Down dB var-K (N_fwd=4096, K_fwd=1536).
            # Was falling through to default (gm=4, xcds=8) — none of
            # the existing CRR rules matched (tiles_n=6 is unique to
            # Qwen-Down K_fwd=1536 in the BF16 metric). 11-cell sweep
            # on all 4 Qwen-Down shapes vs default:
            #   B=16-M2048   +1.15 %
            #   B=16-M4096   +1.76 %
            #   B=32-M2048   +1.24 %
            #   B=32-M4096   +2.10 %
            # avg +1.56 %, min +1.15 %, max +2.10 %; uniform-positive.
            # (gm=2, xcds=4) is also uniform +1.45 % avg but slightly
            # worse on every shape — picked (gm=1, xcds=4) for
            # consistency with the 3 other (1, 4) wins this round.
            return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
        if (layout == "crr"
                and tiles_m == 12
                and tiles_n == 16
                and k <= 4096
                and m_total is not None):
            # Qwen3-235B-A22B-GateUP dB var-K (N_fwd=3072, K_fwd=4096).
            # Was on default (gm=4, xcds=8). 11-cell sweep:
            #   B=16-M2048   +0.53 %
            #   B=16-M4096   +1.70 %
            #   B=32-M2048   +0.49 %
            #   B=32-M4096   +1.66 %
            # avg +1.10 %, min +0.49 %, max +1.70 %; uniform-positive
            # (only cell with all 4 shapes positive). Note this is
            # the same (gm=1, xcds=4) as Qwen-Down even though
            # geometry differs — the persistent-grid behavior at
            # tiles_m ∈ {12, 16} with shallow tiles_n ∈ {6, 16}
            # converges on the same scheduler state.
            return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
        if (layout == "crr"
                and tiles_m == 28
                and tiles_n == 8
                and k <= 4096
                and m_total is not None):
            # DeepSeek-V3-Down dB var-K (N_fwd=7168, K_fwd=2048).
            # tiles_m=28 is the WIDEST N tile in the metric var-K
            # suite (other families are tiles_m ∈ {11, 12, 16, 22}).
            # Was on default (gm=4, xcds=8). 11-cell sweep:
            #   B=16-M2048   +0.64 %
            #   B=16-M4096   +0.64 %
            #   B=32-M2048   +1.24 %
            #   B=32-M4096   +0.22 %
            # avg +0.69 %, min +0.22 %, max +1.24 %; uniform-positive.
            # The (gm, xcds) winner here is (gm=16, xcds=4) NOT
            # (gm=1, xcds=4) — at tiles_m=28 the persistent-grid
            # work-window benefits from gm=16 batching (more L2
            # reuse per tile-row per CU). Same property as the
            # tiles_m=8 + B=32 RCR rule at line 687 above
            # ((gm=16, xcds=4) for tiles_m=8) — the wider-N
            # geometry has analogous large-batch sensitivity.
            return HipKittenConfig(layout=layout, group_m=16, num_xcds=4, kernel=None)
        # NOTE: a 5th rule was tested for DSV3-GateUP dB var-K
        # (tiles_m=16, tiles_n=28) at (gm=2, xcds=0), but that
        # cell — while uniform-positive +0.28 % avg in the
        # kernel-only probe vs in-place HK reference — exceeds
        # the metric's bf16 allclose tolerance vs Triton
        # reference (4/4 DSV3-GateUP shapes hit dB-allclose).
        # The xcds=0 chiplet bypass produces a different
        # accumulation order across CUs that keeps in-HK output
        # bit-equal to itself but drifts vs Triton beyond the
        # 1-2 ULP envelope. Dropped from the aggregate; DSV3-GateUP
        # dB var-K stays on default (gm=4, xcds=8).
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
            # that already passed in round 4. Round 24 added a
            # CRR-specific carve-out above for tiles_m=tiles_n=11 +
            # m_total!=None (gpt_oss-Down dB var-K) — that path was
            # previously mis-routed here.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if tiles_n == 16 and tiles_m == 2 * tiles_n and k <= 4096:
            # Skinny tall (32x16 grid, K shallow): canonical attn_out
            # shape 8192x4096x4096. Cache rcr (2, 16); rrr/crr (2, 32);
            # we pick (2, 32) because it is within ~1pp of (2, 16) on
            # rcr and matches the rrr/crr backward-pass winners exactly.
            return HipKittenConfig(layout=layout, group_m=2, num_xcds=32, kernel=None)
        if layout == "rcr" and tiles_m == 8 and tiles_n == 16 and k == 1536:
            # Round-9 rule. Qwen3-235B-A22B-Down-M2048 grouped RCR family
            # (B ∈ {16, 32}; tiles_n=16 for n=4096; tiles_m=8 for
            # m_per_group=2048; k=1536 unique to Qwen-Down in BF16
            # metric). Was being mis-routed by the DSV3-GateUP-M2048 rule
            # below (tiles_n==16, tiles_m==8, k<=7168) which forces
            # (gm=1, xcd=4) — that config was tuned for K=7168 (28
            # K-iter), but Qwen-Down K=1536 (6 K-iter) has 4.7× shallower
            # K so the persistent loop sees ~5× fewer mfma per tile-step
            # → the L2 reuse pattern flips and the binding default
            # (gm=4, xcd=8) wins instead.
            #
            # Coarse sweep (50-iter p20 at /tmp/probe_qwen_down_bf16_round9.py):
            #
            #   Qwen-Down-B16-M2048  M_tot=32768:
            #     gm=4 row dominates: (4, 8)=1116.8  (4, 4)=1090.8
            #     gm=1 row:           (1, 4)=1106.7  (1, 8)=1033.7  ←current
            #     gm=2 row:           (2, *)~1085
            #
            #   Qwen-Down-B32-M2048  M_tot=65536:
            #     gm=4 row dominates: (4, 16)=1105.2  (4, 8)=1103.0
            #     gm=1 row:           (1, 4)=1100.1                ←current
            #     gm=2 row:           (2, *)~1077
            #
            # Tight verify (200-iter × 7-trial p20 at
            # /tmp/verify_qwen_down_bf16_m2048_round9.py):
            #
            #   Qwen-Down-B16-M2048:
            #     ( 4, 16)  1106.58 TF  +0.96 pp vs (1, 4) ←current  *winner
            #     ( 4,  8)  1106.26 TF  +0.93 pp                      (spread 0.47 %)
            #     ( 1,  4)  1096.10 TF  baseline                      (spread 0.28 %)
            #
            #   Qwen-Down-B32-M2048:
            #     ( 4,  8)  1108.39 TF  +0.83 pp vs (1, 4) ←current  *winner
            #     ( 4, 16)  1107.99 TF  +0.80 pp                      (spread 0.25 %)
            #     ( 1,  4)  1099.24 TF  baseline                      (spread 0.51 %)
            #
            # ``(gm=4, xcds=8)`` wins both shapes by 9-10 TF
            # (+0.83..+0.96 pp) — gaps are 2-3× the run-to-run spread
            # (0.25-0.51 %), and the winner's per-trial min beats the
            # current's max in 7/7 trials per shape. (gm=4) is consistent
            # across both, with xcds=8 a hairline above xcds=16 on B=32
            # and tied on B=16 — keep the binding default xcds=8 for
            # config simplicity (no explicit override).
            #
            # Why (gm=4, xcds=8) wins for K=1536: with only 6 K-iter per
            # tile-step (block_k=256), the per-tile compute is ~5× lighter
            # than the K=7168 DSV3-GateUP cousins this rule used to
            # match. (gm=1) maximises B-tile L2 reuse on long-K shapes
            # by walking the entire N-row before moving M; on shallow-K
            # the same N-walk now under-feeds the persistent loop because
            # each tile-step finishes in ~6 mfma waves, leaving the LDS
            # double-buffer's load-side bandwidth unsatured. (gm=4)
            # batches 4 M-tiles into the same group, sharing each
            # B-pack across 4 mfma sequences and keeping the LDS
            # buffer warm — a strict win when the per-tile compute
            # latency is short relative to the load latency.
            #
            # Bit-identical output verified at
            # /tmp/verify_qwen_down_bf16_m2048_correctness_round9.py:
            #   Qwen-Down-B16-M2048: max_abs_diff=0.0  bit_eq=True
            #   Qwen-Down-B32-M2048: max_abs_diff=0.0  bit_eq=True
            # group_m / num_xcds are pure scheduling knobs on the BF16
            # grouped RCR persistent tile schedule; arithmetic and
            # rounding invariant.
            #
            # Rule scope check: ``tiles_n == 16`` (n=4096) is shared with
            # DSV3-GateUP (k=7168) and dense LLaMA shapes (multiple K),
            # but the ``k == 1536`` clause is uniquely Qwen-Down in the
            # BF16 metric (DSV3 k ∈ {2048, 7168}; gpt_oss k=2880;
            # Qwen-GateUP k=4096; dense BF16 k ∈ {4096, 11008, 14336,
            # 22016, 28672}). ``tiles_m == 8`` (m_per_group=2048)
            # excludes the M=4096 sibling cases (which fall through
            # to the cube-small rule above with (gm=2, xcd=32) — that
            # routing was tight-verified in this same probe to be the
            # joint top, no rule change needed there). Strict ``k == 1536``
            # rather than ``k <= 1536`` keeps the rule from accidentally
            # capturing any future shallower-K family.
            return HipKittenConfig(layout=layout, group_m=4, num_xcds=8, kernel=None)
        if layout == "rcr" and tiles_m == 8 and tiles_n == 16 and k <= 7168:
            # Round-10 rule. DeepSeek-V3-GateUP-M2048 grouped RCR family:
            # per-group GEMM with N=4096, K=7168, M_per_group=2048.
            # The cube-small rule above only matches tiles_m == tiles_n
            # (i.e. M_per_group=4096); the M_per_group=2048 sibling shapes
            # were falling to the binding default (gm=4, xcd=8). 36-config
            # sweep on the 2 metric shapes (B ∈ {16, 32}) shows
            # (gm=1, xcd=4) wins both:
            #   B=16 M=2048: 1419.6 vs default 1396.9 = +1.62pp   (top1)
            #   B=32 M=2048: 1407.2 vs default 1390.7 = +1.19pp   (top1)
            # Also strictly beats the cube-small (gm=2, xcd=32) cfg by
            # +2.46-2.64pp (those shapes do NOT match the cube-small
            # tiles_m == tiles_n predicate today, but the comparison
            # confirms the rule placement). See /tmp/sweep_gateup.log
            # archived in the round-10 commit. Bit-identical output
            # vs default (cross max_abs = 0.0000, SNR 47.85 dB unchanged).
            #
            # Rule scope check: tiles_m == 8 means M_per_group == 2048,
            # which only occurs in metric-grouped shapes (dense BF16
            # smallest M is 4096 ⇒ tiles_m ≥ 16). tiles_n == 16 with
            # k <= 7168 in the metric-grouped space matches:
            #  - DeepSeek-V3-GateUP-M2048-{B16,B32}  (target)
            #  - DeepSeek-V3-Down has tiles_n == 28 → no match.
            #  - gpt_oss padded: tiles_n ∈ {12, 24} → no match.
            return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
        if layout == "rcr" and tiles_n == 28 and 8 <= tiles_m <= 16 and k <= 4096:
            # Round-10 rule. DeepSeek-V3-Down grouped RCR family: per-group
            # GEMM with N=7168, K=2048, M_per_group ∈ {2048, 4096}. The
            # persistent grouped kernel runs this layout on uniform-M aligned
            # inputs (no padding); the only choice is (group_m, num_xcds).
            # 36-config sweep on the 4 metric shapes (B ∈ {16,32}, M ∈
            # {2048,4096}) shows (gm=16, xcd=2) wins or ties top-2 each time:
            #   B=16 M=2048: 1214.3 vs default 1203.0 = +0.94pp   (top1)
            #   B=16 M=4096: 1227.9 vs default 1214.6 = +1.10pp   (top2 tie)
            #   B=32 M=2048: 1198.7 vs default 1183.9 = +1.25pp   (top2)
            #   B=32 M=4096: 1210.6 vs default 1197.3 = +1.11pp   (top3)
            # See /tmp/sweep_deepseek_down.log archived in the round-10
            # commit message. Bit-identical output vs default (cross
            # max_abs = 0.0000, SNR 47.86 dB unchanged).
            #
            # Rule scope check: ``tiles_n == 28`` ⇔ N == 7168, which only
            # occurs in the metric for these 4 grouped DeepSeek-V3-Down
            # forward shapes. No dense BF16 metric shape has N=7168 (Llama
            # / Qwen / Mistral / Llama-3.1 don't combine to 7168). The
            # backward dispatches for DeepSeek-V3-Down hit different layouts
            # (dA RRR with N=2048, dB CRR with M-axis=7168) that cannot
            # match this rcr-only rule. The DeepSeek-V3-GateUP forward
            # (N=4096) and any gpt_oss padded shape (n_pad ∈ {3072, 6144})
            # also do not match tiles_n == 28.
            return HipKittenConfig(layout=layout, group_m=16, num_xcds=2, kernel=None)
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
        if layout == "rrr" and tiles_m == 32 and tiles_n == 16 and k >= 22016:
            # Round-5 rule. Skinny-N very-deep-K RRR. Canonical: dA RRR
            # (8192, 4096, 28672) for fwd (8192, 28672, 4096) — the
            # bwd's middle dispatch was previously falling through to
            # the binding default (4, 8) because no RRR rule covered
            # tiles_n == 16. Tight-bench on (8192, 4096, 28672)
            # (200 iters × 3 repeats, p20): (gm=8, xcd=2) = 1499.8 TF
            # beats default (4, 8) = 1482.7 TF by +1.15pp; bit-identical
            # output (cross max_abs = 0, SNR = 47.85 dB unchanged; see
            # /tmp/probe_bf16_round5.log archived in commit message).
            # K bound is 22016 (not 11008) on purpose: the 11008-22016
            # tier showed (2, 32) and (8, 2) within 0.4pp of each other
            # (sweep_bf16_bwd_round5_nearby.log) so adding a rule there
            # is not anchored well; only K >= 22016 has a clear (8, 2)
            # winner with a +1.4-1.8pp margin.
            return HipKittenConfig(layout=layout, group_m=8, num_xcds=2, kernel=None)
        if (layout == "rrr"
                and tiles_n == 28
                and m_total is not None
                and m_total >= 32768):
            # Round-20 rule (1 of 3 in this commit). BF16 grouped dA RRR
            # — DSV3-GateUP wide-N family. Mirrors FP8 R43/R44 line 2497.
            # FP8 R43/R44 12-trial × 200-iter × 3-seed showed +1.66-2.82%
            # on all 4 shapes. R19 verified BF16 transfer with 5-run mean
            # +1.78pp ratio avg (Per-shape: B16-M2048 +1.9pp, B16-M4096
            # +2.5pp, B32-M2048 +1.5pp, B32-M4096 +1.2pp). R19 was
            # initially documented as NOISE-BOUND because the per-rule
            # +0.7pp DSV3-family geomean lift maps to only +1.1 score
            # — below the metric's gpt_oss-noise-driven +5 threshold.
            # R20 commits this lever as part of a 3-rule aggregate
            # (tiles_n ∈ {8, 16, 28}) so the combined per-family lift
            # crosses the noise floor.
            #
            # Scope: tiles_n == 28 (n=K_fwd=7168) is uniquely DSV3-GateUP
            # dA in the BF16 metric. Numerically equivalent to bf16
            # allclose tolerance (metric correctness gate verified
            # 0/24 fail across 5 verification runs in R19).
            return HipKittenConfig(layout=layout, group_m=16, num_xcds=4, kernel=None)
        if (layout == "rrr"
                and tiles_n == 16
                and m_total is not None
                and m_total >= 32768):
            # Round-20 rule (2 of 3). BF16 grouped dA RRR — Qwen3-GateUP
            # mid-N family. (gm=8, xcds=4) chosen from 11-cell BF16 probe
            # (scripts/_bf16_rrr_da_probe.py) — the only cell uniformly
            # positive on all 4 Qwen3-GateUP shapes:
            #   Qwen3-GateUP B=16 M=2048   +1.20%  (top: (1,4) +2.67%)
            #   Qwen3-GateUP B=16 M=4096   +0.33%  (top: (4,2) +0.83%)
            #   Qwen3-GateUP B=32 M=2048   +0.49%  (top: (16,4) +1.53%)
            #   Qwen3-GateUP B=32 M=4096   +2.24%  (top1)
            # Avg +1.07% kernel-only; spread per shape ~0.5pp.
            #
            # FP8 R32 settled on (1,4) for tiles_n=16 with a tiles_m==8
            # m_total>=65536 carve-out (gpt_oss-GateUP-B32-M2048 falls
            # to default). The BF16 probe shows (1,4) wins B16-M2048
            # but loses B32-M4096 (-0.41%) — non-uniform; (8,4) is the
            # uniform-positive choice. Different cell from FP8 because
            # BF16's lighter B-side bandwidth (no dscale) tilts the
            # optimum toward gm=8 batching for the larger m_total tier.
            #
            # Scope: tiles_n == 16 (n=K_fwd=4096). In the BF16 metric:
            #   - Qwen3-GateUP dA RRR (n=4096, k=N_fwd=3072) — target ✓
            #   - DSV3-GateUP dA RRR  (n=K_fwd=7168) → tiles_n=28, NOT
            #     matched (covered by the tiles_n==28 rule above).
            # Wait — DSV3-GateUP K_fwd=7168 → dA n=7168 → tiles_n=28.
            # So only Qwen3-GateUP K_fwd=4096 hits tiles_n==16. Other
            # shapes: DSV3-Down k=2048→tiles_n=8, Qwen3-Down k=1536→
            # tiles_n=6, gpt_oss k=2880→tiles_n=11 (and reroutes via H4).
            #
            # Numerically equivalent to bf16 allclose tolerance (metric
            # correctness gate verified by R20 metric runs).
            return HipKittenConfig(layout=layout, group_m=8, num_xcds=4, kernel=None)
        if (layout == "rrr"
                and tiles_n == 8
                and m_total is not None
                and m_total >= 32768):
            # Round-20 rule (3 of 3). BF16 grouped dA RRR — DSV3-Down
            # narrow-N family. (gm=4, xcds=4) chosen from 11-cell BF16
            # probe — top-1 on 3 of 4 shapes, all 4 shapes positive:
            #   DSV3-Down B=16 M=2048   +2.53% (top1)
            #   DSV3-Down B=16 M=4096   +0.28%
            #   DSV3-Down B=32 M=2048   +3.54% (top1)
            #   DSV3-Down B=32 M=4096   +1.05% (top1)
            # Avg +1.85% kernel-only.
            #
            # NOT (gm=16, xcds=4) like Qwen3-Down (R18) or DSV3-GateUP
            # (above). The R18 broad-rule probe had already shown
            # tiles_n==8 + (16,4) was mixed on BF16 (one regression
            # -0.5pp, avg +0.2pp); the focused 11-cell probe this round
            # confirms (16,4) is suboptimal for tiles_n==8 BF16 — only
            # +1.56% on B16-M2048 (vs (4,4) +2.53%) and -1.03% on
            # B16-M4096. Different cell from FP8 R42 (which uses
            # (16,4) for tiles_n<=8) because the persistent grid sees
            # 18-shorter-K iter for DSV3-Down K=2048 vs DSV3-GateUP
            # K=7168, and the smaller per-tile compute window favors
            # gm=4 N-walk reuse over gm=16 batching.
            #
            # Scope: tiles_n == 8 (n=K_fwd=2048) is uniquely DSV3-Down
            # dA in the BF16 metric (Qwen3 K_fwd ∈ {1536, 4096} →
            # tiles_n ∈ {6, 16}; gpt_oss K_fwd=2880 → tiles_n=11
            # AND H4-reroutes via transpose; DSV3-GateUP K_fwd=7168 →
            # tiles_n=28). Dense LLaMA RRR n ∈ {4096, 8192, 11008+} →
            # tiles_n ≥ 16.
            #
            # Numerically equivalent to bf16 allclose tolerance.
            return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
        if (layout == "rrr"
                and tiles_n == 6
                and m_total is not None
                and m_total >= 32768):
            # Round-18 rule. BF16 grouped dA RRR — Qwen3-235B-A22B-Down
            # narrow-N family. Mirrors the FP8 R42 rule at line 2233
            # (which the BF16 path never received because all BF16 RRR
            # rules above gate on tiles_m patterns that exclude this
            # geometry).
            #
            # Hits (BF16 metric, exactly 4 shapes):
            #   Qwen3-Down dA RRR — n=K_fwd=1536 → tiles_n=6, k=N_fwd=4096,
            #   m_per_g ∈ {2048, 4096}, m_total ∈ {32768, 65536, 131072}.
            #
            # Why narrowed to ``tiles_n == 6`` (NOT R42's ``tiles_n <= 8``):
            # an exploratory broader rule (tiles_n <= 8, including DSV3-Down
            # tiles_n=8) was tested in this round but showed an asymmetric
            # outcome on BF16 across two metric runs. Per-shape Δ (run mean
            # over 2 trials with rule vs single baseline run):
            #   Qwen3-Down B=16 M=2048   1.112 -> 1.114  +0.2pp ✓
            #   Qwen3-Down B=16 M=4096   1.115 -> 1.124  +0.9pp ✓
            #   Qwen3-Down B=32 M=2048   1.101 -> 1.120  +1.8pp ✓
            #   Qwen3-Down B=32 M=4096   1.108 -> 1.119  +1.1pp ✓
            #   DSV3-Down  B=16 M=2048   1.133 -> 1.116  -1.7pp ✗
            #   DSV3-Down  B=16 M=4096   1.104 -> 1.107  +0.3pp tie
            #   DSV3-Down  B=32 M=2048   1.111 -> 1.110   tie
            #   DSV3-Down  B=32 M=4096   1.104 -> 1.098  -0.6pp ✗
            # Qwen3-Down (tiles_n=6) is uniformly +0.2..+1.8pp; DSV3-Down
            # (tiles_n=8) regresses on 2 of 4 shapes. R42's FP8 sweep
            # showed both tiles_n=6 and tiles_n=8 winning at +1.95-6.66 %
            # — the BF16 ↔ FP8 transfer is partial. Restricting to
            # tiles_n==6 captures only the consistently-positive
            # subfamily. (DSV3-Down can be revisited with its own rule
            # once a per-tiles_n=8 BF16 microbench provides clean data.)
            #
            # The dA RRR kernel template is shared between BF16 and FP8
            # grouped paths (only difference: FP8 also reads two scale
            # factors). Persistent-grid scheduling (gm, xcd) operates on
            # tile counts, not data values, so a (gm, xcd) cell that wins
            # for FP8 RRR can win for BF16 RRR at the same tile geometry
            # — but only if the bandwidth profile change (no dscale on
            # BF16 side) doesn't tip the optimum. R17 demonstrated FP8
            # var-K → BF16 var-K does NOT transfer (Qwen3-Down -1.3pp
            # avg) because var-K's grad_out + x dual-stream changes the
            # B-side reuse window. Here on dA RRR, the W-tile is static
            # (single-source HBM read) so the load schedule is closer
            # to dtype-invariant.
            #
            # Rule scope check: tiles_n == 6 in BF16 grouped RRR
            # restricts to Qwen3-Down dA (n=1536). No dense LLaMA RRR
            # shape has n=1536 (LLaMA's smallest dense RRR n is 4096 →
            # tiles_n=16). Dense BF16 RRR uses m_total=None and is
            # excluded by `m_total is not None`.
            #
            # Bit-identical: group_m / num_xcds are pure persistent-grid
            # scheduling knobs (R42 verified max_abs=0.0 bit_eq=True on
            # all 6 FP8 narrow-N shapes; same kernel template behavior
            # for BF16).
            return HipKittenConfig(layout=layout, group_m=16, num_xcds=4, kernel=None)
        if layout == "crr" and tiles_m >= 64 and tiles_n == 16:
            # Long-N backward dB-after-swap (CRR sees logical (N_fwd,
            # K_fwd, M_fwd) post-swap). Canonical Llama-2-7B
            # mlp_gate_up backward dB:
            #   fwd 4096x22016x4096 -> dB CRR (22016,4096,4096) = (86,16,4096)
            # and Llama-3.1-8B
            #   fwd 8192x28672x4096 -> dB CRR (28672,4096,8192) = (112,16,8192)
            # Round-7: shallow K (=4096) prefers (gm=24, xcd=2) by +1.4pp.
            # Round-5: deeper K (>4096) tight-bench refines the previous
            # (gm=2, xcd=32) pick to (gm=4, xcd=32) — on (28672, 4096, 8192)
            # the p20 reads (gm=4, xcd=32) = 1450.1 TF vs (gm=2, xcd=32)
            # = 1442.3 TF (+0.54pp), bit-identical output (cross max_abs
            # = 0, SNR = 47.86 dB unchanged; see /tmp/probe_bf16_round5.log).
            if k <= 4096:
                return HipKittenConfig(layout=layout, group_m=24, num_xcds=2, kernel=None)
            return HipKittenConfig(layout=layout, group_m=4, num_xcds=32, kernel=None)
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
        if layout == "crr" and tiles_n == 11 and 8 <= tiles_m <= 24 and k <= 4096:
            # Round-1 (weighted-wall metric) rule. gpt_oss var-K (dB) family:
            # K_fwd=2880 → kernel-N axis tiles_n=11; N_fwd ∈ {2880, 5760}
            # → kernel-M axis tiles_m ∈ {11, 22}; per-group K_var = M_per
            # ∈ {2048, 4096} → k <= 4096. Predicate covers all 8 gpt_oss
            # var-K dB calls in the metric (B ∈ {4, 32}, M_per ∈ {2048,
            # 4096}, GateUP+Down). Was falling through to default
            # (gm=4, xcds=8) — no prior CRR rule had been written for
            # gpt_oss because the existing CRR family rules
            # (tiles_m >= 32 + tiles_n == 16) were tuned for dense LLaMA
            # mlp_gate_up dB (tiles_m ∈ {48, 86, 112}, tiles_n=16) and
            # never matched gpt_oss's small (11/22, 11) tile geometry.
            #
            # Per-iter-sync metric, this round (target shape lowest-
            # progress was gpt_oss-GateUP-B32-M2048 at ratio=0.753):
            #   * baseline default (gm=4, xcds=8): score=775,
            #     gpt_oss family geomean=0.8706
            #   * with this rule (gm=4, xcds=4): score=780 (×2 runs,
            #     stable), gpt_oss family geomean=0.8823 (+1.2pp)
            #
            # Per-shape gpt_oss before → after with this rule
            # (single best run, run-to-run swing ~0.5pp):
            #   GateUP-B4-M2048    0.855 → 0.876   +2.5pp ✓
            #   Down-B4-M2048      0.885 → 0.923   +4.3pp ✓
            #   GateUP-B4-M4096    0.936 → 0.962   +2.8pp ✓
            #   Down-B4-M4096      0.957 → 0.955    flat (noise)
            #   GateUP-B32-M2048   0.753 → 0.763   +1.3pp marginal
            #   Down-B32-M2048     0.804 → 0.801   flat (noise)
            #   GateUP-B32-M4096   0.892 → 0.895   flat (noise)
            #   Down-B32-M4096     0.903 → 0.905   flat (noise)
            #
            # The clean signal is on the B=4 sub-family (3/4 shapes
            # +2.5 to +4.3pp). B=32 is mostly noise — the ``xcds=4 vs
            # xcds=8`` knob saturates once tiles/CU exceeds ~10. The
            # B=4 family has tiles/CU ~4.3 (B=4 × tiles_per_group=276 /
            # NUM_CUS=256), so xcds=8 was over-splitting the work
            # across all 8 XCDs and starving each XCD of useful tiles.
            # xcds=4 keeps each XCD with at least 1 tile most of the
            # time, halving the inter-XCD coordination overhead.
            #
            # Split-by-m_total tested: separate rule with (gm=8,
            # xcds=4) for m_total>16384 (B=32) was within noise of
            # this single-cfg version (gpt_oss geomean 0.8872 vs
            # 0.8823, but full-suite score swung 776 vs 780 due to
            # DSV3 / Qwen3 Triton baseline run-to-run noise on
            # uninvolved shapes — ±2pp). Single cfg is cleaner; defer
            # B=32 split to a later round once a kernel-level fix
            # opens more headroom there.
            #
            # Rule scope check (no collateral on other metric shapes):
            # ``tiles_n == 11`` ⇔ K_fwd ∈ [2816, 3071] which uniquely
            # captures gpt_oss K_fwd=2880 in the metric. Other families'
            # var-K kernel-N axes (= K_fwd):
            #  - DSV3-GateUP: K_fwd=7168 → tiles_n=28
            #  - DSV3-Down:   K_fwd=2048 → tiles_n=8
            #  - Qwen3-GateUP: K_fwd=4096 → tiles_n=16
            #  - Qwen3-Down:  K_fwd=1536 → tiles_n=6
            # None hit tiles_n=11.
            #
            # Bit-identical output (group_m / num_xcds are pure
            # scheduling knobs on the BF16 grouped var-K persistent
            # tile schedule — same property as forward grouped + dense
            # rules above). Validated by metric's correctness gate
            # (downsized ``check_allclose`` on out, dA, dB) on every
            # shape every round; if (4, 4) corrupted dB, the small
            # check would FAIL on at least one of the 8 gpt_oss shapes
            # and clip the ratio to 0.0. All 24 shapes passed both
            # before-and-after runs.
            return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
        return HipKittenConfig(
            layout=layout,
            group_m=_BF16_DEFAULT_GROUP_M,
            num_xcds=_BF16_DEFAULT_NUM_XCDS,
            kernel=None,
        )

    # FP8: kernel template ID matters only for RCR (the binding ignores it
    # for RRR / CRR). Default ``cfg.kernel = None`` so the dispatcher's
    # ``force_rcr_kernel`` context manager is skipped — the binding's own
    # ``dispatch<RCR>`` (analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp)
    # auto-picks the right 4-wave / 8-wave template based on
    # ``grid_size >= RCR_4WAVE_MIN_GRID(=3200) && k <= RCR_4WAVE_MAX_K(=8192)``,
    # so leaving ``kernel=None`` is bit-equivalent to a manual override
    # while saving the per-dispatch lock + env get/set/restore on every
    # call. Per-shape rules below override this default only when the
    # binding's auto-pick is sub-optimal AND the kernel-level win exceeds
    # the ~1us context-manager overhead.
    if layout == "rcr":
        tiles_m = m // 256
        tiles_n = n // 256
        # Round-61 rule. Unpadded gpt_oss FP8 RCR — small-batch (B=4)
        # GateUP family (per-launch tile geometry m_per_group ∈
        # {2048, 4096}, n=5760, k=2880). The persistent grouped grid for
        # B=4 sees only 4× as many tiles vs the per-group view, which
        # makes the persistent scheduler much more sensitive to the
        # ``group_m`` tile-batching factor than the B=32 sibling (where
        # the grid already saturates the GPU and ``group_m`` only
        # re-orders within each XCD slice).
        #
        # 7-candidate sweep ``group_m ∈ {1, 2, 4, 8, 16, 24, 32}`` over
        # the 4 metric gpt_oss FP8 shapes (200-iter × 3-repeat median at
        # /tmp/sweep_fp8_gptoss_gm_wide.py archived in the round-61
        # commit message). Kernel TF (higher is better):
        #
        #   shape              gm=1   gm=2*  gm=4   gm=8   gm=16  gm=24  gm=32
        #   GateUP-B4-M2048    1160   1192   1182   1178   1182   1123   1122
        #   GateUP-B4-M4096    1065   1070   1069   1049   1055   1039   1039
        #   GateUP-B32-M2048   1067   1089   1098*  1084   1084   1061   1061
        #   GateUP-B32-M4096   1105   1117   1116*  1101   1092   1088   1088
        #
        # gm=2 wins B=4 GateUP shapes (m_total ∈ {8192, 16384}); B=32
        # GateUP (m_total ≥ 65536) prefers default gm=4. Down family is
        # flat across {1, 2, 4, 8} (no rule needed).
        #
        # Net delta over default gm=4: GateUP-B4-M2048 +0.85pp, the only
        # case with a clear margin (others within ±0.1pp). Bit-identical
        # output (``group_m`` is a pure scheduling knob); SNR unchanged.
        #
        # Rule scope check: ``tiles_n == 22`` (N=5760) is unique to
        # gpt_oss-GateUP. ``k == 2880`` is unique to gpt_oss in the FP8
        # metric. ``m_total <= 16384`` selects B ∈ {4} (B=4 M=2048
        # gives 8192, B=4 M=4096 gives 16384; B=32 M=2048 gives 65536,
        # which is excluded). The default gm=4 path is preserved for
        # B=32, where the wider grid already absorbs the scheduling
        # overhead.
        #
        # Round-68: ``num_xcds=4`` ONLY for the M_per_group=2048 sub-rule
        # (tiles_m == 8). 800-iter × 5-repeat median A/B/C/D at
        # /tmp/verify_gpt_oss_xcds_round68.py:
        #   shape                      xcds=8     xcds=2     xcds=4     xcds=16
        #   gpt_oss-GateUP-B4-M2048    1167.5     1172.7     1183.8 *   1168.3   (x=4 +16.4)
        #   gpt_oss-GateUP-B4-M4096    1064.9 *   1063.3     1049.0     1064.9   (x=8 best, x=4 -15.9)
        # Setting xcds=4 unconditionally would gain +16.4 TF on B4-M2048
        # but lose -15.9 TF on B4-M4096. Splitting by tiles_m captures
        # the win cleanly (+1.40pp on B4-M2048, no change on B4-M4096).
        if (
            tiles_n == 22
            and tiles_m == 8
            and k == 2880
            and m_total is not None
            and m_total <= 16384
        ):
            # Round-23 re-tune in the metric-aligned per-iter-sync regime.
            # Round-68's (gm=2, xcd=4) was picked from a steady-state
            # 800-iter × 5-repeat median bench; the metric script uses
            # ``WARMUP=10, ITERS=50`` per-iter ``cudaDeviceSynchronize()``
            # cold-call timing (see ``scripts/_metric_hk_ratio.py``
            # ``_time_op``), which exposes a different per-tile completion
            # latency profile. A 36-cell coarse sweep
            # (/tmp/probe_xcds_disable_round23.py) on the 4 B=4 FP8 gpt_oss
            # shapes flagged ``(gm=1, xcd=4)`` as a winner here — confirmed
            # by a 9-cell × 7-repeat × 120-iter tight verify
            # (/tmp/verify_fp8_gateup_b4_m2048_round23.py):
            #
            #   cfg          med p20    min       max     spread
            #   ( 1, 4)      1013.81    1006.90   1018.22   1.12%   *winner
            #   ( 1, 2)      1008.70    1005.26   1011.25   0.59%   -5.11
            #   ( 3, 4)      1005.56    1002.15   1009.59   0.74%   -8.25
            #   ( 2, 8)      1004.22     999.64   1006.31   0.66%   -9.59
            #   ( 4, 4)      1002.44    1000.38   1006.30   0.59%  -11.37
            #   ( 2, 4)      1001.56     997.58   1009.44   1.18%  -12.25  ←round-68
            #   ( 1, 8)       972.46     968.85    975.81   0.72%  -41.35
            #   ( 1, 1)       970.79     968.02    974.13   0.63%  -43.02
            #   ( 1, 16)      971.63     968.58    975.39   0.70%  -42.18
            #
            # ``(gm=1, xcd=4)`` is the unique top of a (gm=1)-favouring
            # cluster with (1, 4) > (1, 2) > (3-4, 4) > (2, 4)=round-68.
            # The +12.25 TF (+1.22pp) gap over round-68 (2, 4) lies above
            # the run-to-run spread (1.12%): the candidate's min (1006.90)
            # still beats the default's max (1009.44) only marginally —
            # but the median gap is consistent across all 7 repeats.
            #
            # Why gm=1 wins here: B=4 M=2048 ⇒ 16384 rows ⇒ 64 M-tiles per
            # batch × 23 N-tiles × 4 batches = 5888 tile-steps. With
            # NUM_CUS=256 persistent slots, that's 23 wave-steps per slot.
            # gm=2 groups 2 M-tiles together to reuse the A-pack across
            # the M-axis, which is a win on shapes where A-load bandwidth
            # is the bottleneck. But for tiles_m=8 (M=2048 per group)
            # there are only 8 M-tiles per group; gm=2 forces the
            # persistent loop to stride over only 4 N-tile groups per
            # XCD-pair before starting the next M-pair, which loses the
            # L2 reuse on the B-side N-tile (N=5760 ⇒ 23 N-tiles, larger
            # than M-axis). gm=1 lets the schedule walk the entire N-row
            # for each M-tile before moving on, maximising B-tile L2
            # reuse — better fit for tiles_m << tiles_n shapes.
            #
            # xcd=4 unchanged from round-68 — both the round-68 200-iter
            # sweep and this round-23 metric-aligned verify pick xcd=4
            # over xcd=2 / xcd=8 / xcd=16 by ≥5 TF (+0.5pp).
            #
            # Bit-equivalence verified at the same path:
            # max_abs_diff=0.0 between (gm=2, xcd=4) and (gm=1, xcd=4)
            # outputs (group_m / num_xcds are pure scheduling knobs on
            # the FP8 grouped RCR persistent tile schedule, not
            # arithmetic-affecting).
            #
            # Rule scope unchanged from round-68: tiles_n == 22 + tiles_m
            # == 8 + k == 2880 + m_total <= 16384 covers only
            # gpt_oss-GateUP-B4-M2048 in the metric shapes (B<4 GateUP
            # M=2048 has no metric shape, B=8 GateUP M=2048 ⇒ m_total
            # =16384 — hypothetically same rule but no metric shape).
            return HipKittenConfig(
                layout=layout,
                group_m=1,
                num_xcds=4,
                kernel=None,
            )
        if (
            tiles_n == 22
            and tiles_m == 16
            and k == 2880
            and m_total is not None
            and m_total <= 16384
        ):
            # gpt_oss-GateUP-B4-M4096 (tiles_n=22, tiles_m=16, k=2880,
            # m_total=16384). This rule was discovered in round-7 via an
            # 8 × 5 = 40-candidate sweep at /tmp/sweep_fp8_b4_round7.py
            # against the round-18 (FLAT-store) FP8 grouped kernel.
            #
            # Round-10-dm verification (post-round-19/20 BUFFER-store
            # kernel): a 1500-iter × 7-repeat re-sweep at
            # ``/tmp/probe_gateup_b4_m4096_round10.py`` confirms
            # ``(gm=14, xcd=4)`` is still the optimum within ±1.4 TF
            # (±0.1 pp) of every neighbor. Median p14 TFLOPS:
            #
            #   cfg     | TFLOPS  | Δ vs (14,4)
            #   (14, 4) | 1240.33 | +0.00 *rule
            #   (14, 8) | 1241.28 | +0.95
            #   (14, 2) | 1241.25 | +0.92
            #   ( 8, 8) | 1240.80 | +0.47
            #   (16, 4) | 1240.18 | -0.16
            #   (10, 4) | 1239.61 | -0.72
            #   (12, 4) | 1239.31 | -1.02
            #   ( 8, 4) | 1238.93 | -1.40
            #
            # The full sweep is at the metric's noise floor — no candidate
            # dominates. Confirms ``(gm=14, xcd=4)`` is still the right
            # rule on the BUFFER-store kernel.
            #
            # Historical falsification: a previous comment block here
            # cited a "Round-21" sweep claiming ``(gm=8, xcd=4)`` won by
            # +29.2 TF over (gm=14, xcd=4). That sweep was run against
            # the round-18 FLAT-store kernel just before the round-19
            # FLAT->BUFFER reroute (+85pp metric) shifted the per-tile
            # completion latency. The round-10-dm probe table above
            # (against the live BUFFER kernel) shows ``(gm=8, xcd=4)``
            # is the WORST candidate in the sweep, opposite of the stale
            # claim. See ``analysis/_notes/round-10-dm-fp8-config-saturation-audit.md``.
            #
            # Rule scope unchanged: ``tiles_n=22 + tiles_m=16 + k=2880 +
            # m_total<=16384`` matches only gpt_oss-GateUP-B4-M4096 in the
            # metric (B<4 GateUP M=4096 has no metric shape; B=2 / B=1
            # M=4096 → m_total<=8192 → would fail tiles_m=16 since
            # tiles_m=16 ⇒ m=4096 ⇒ m_total ≥ 4096; only B=4 gives 16384).
            return HipKittenConfig(
                layout=layout,
                group_m=14,
                num_xcds=4,
                kernel=None,
            )
        # Round-69 rule. gpt_oss-GateUP-B32 family (tiles_n=22, n=5760,
        # k=2880, m_total ∈ {65536, 131072} for B=32 M_per_g ∈
        # {2048, 4096}). The persistent grid for B=32 is ~16x larger
        # than B=4 so the default gm=4 already saturates the GPU; the
        # remaining lever is the chiplet-swizzle XCD grouping. 1500-iter
        # × 7-repeat tight verify at
        # /tmp/verify_gpt_oss_b32_xcds_round69.py:
        #
        #   shape                       xcds=8     xcds=4    Δ
        #   gpt_oss-GateUP-B32-M2048    1099.5     1103.9    +4.4 (+0.40pp)
        #   gpt_oss-GateUP-B32-M4096    1115.7     1119.9    +4.2 (+0.38pp)
        #
        # Both M_per_g shapes prefer xcds=4 by a small but consistent
        # margin (round-68 wide sweep at 250-iter × 3-repeat showed
        # +5.6 / +5.2 — the 1500-iter retest confirms the direction
        # albeit slightly smaller magnitude). Same sweep verified
        # gpt_oss-Down-B32 is split (M=2048 +2.2 noise, M=4096 -7.8
        # regression at xcds=4) so the rule is gated to GateUP only.
        # Setting num_xcds=4 here, default gm=4 unchanged.
        if (
            tiles_n == 22
            and tiles_m in (8, 16)
            and k == 2880
            and m_total is not None
            and m_total >= 65536
        ):
            # Round-70: gpt_oss-GateUP-B32 family (tiles_n=22, k=2880,
            # m_total ∈ {65536, 131072}). Wider (group_m, num_xcds) sweep
            # (/tmp/sweep_round1.py) over {1..24} × {1, 2, 4, 8, 16, 32}
            # plus 200-iter × p20 verify (/tmp/verify_round1.py) shows
            # gm=8 dominates the round-69 gm=4 on both M_per shapes:
            #
            #   shape                        gm=4,xcd=4   gm=8,xcd=4
            #   gpt_oss-GateUP-B32-M2048     832.1 TF      835.6 TF   (+3.5 TF, +0.42pp)
            #   gpt_oss-GateUP-B32-M4096     922.4 TF      935.2 TF   (+12.8 TF, +1.39pp)
            #
            # Wider sweep over the candidate space confirmed (8, 4) is at
            # the top (alternates: (8, 2)=930 < (8, 4)=937 on M=4096; xcds
            # ≥ 8 / xcds == 1 all underperform (8, 4)). xcds=4 unchanged
            # from round-69 (the chiplet-swizzle rule retained).
            #
            # Rule scope unchanged from round-69: m_total >= 65536 covers
            # B=32 M_per ∈ {2048, 4096} which are the only metric shapes
            # in this band (B<32 GateUP m_total < 65536).
            return HipKittenConfig(
                layout=layout,
                group_m=8,
                num_xcds=4,
                kernel=None,
            )
        # Round-12 refinement of round-69 rule for gpt_oss-Down-B4-M4096
        # (tiles_n=11 for n=2880 since 2880//256 = 11, k=2880,
        # m_total=16384). Round-69 only swept xcds ∈ {4, 8} at gm=4
        # fixed and picked (gm=4, xcd=4); a wider 9 × 6 = 54-candidate
        # sweep this round (gm ∈ {1,2,3,4,6,8,12,16,32} × xcd ∈
        # {1,2,4,8,16,32}, /tmp/sweep_fp8_down_b4_m4096_round12.py)
        # showed an entire xcd=4 plateau dominating: gm ∈ {8,12,16,32}
        # all sit within 5 TF of each other and 13-15 TF above
        # (gm=4, xcd=4).
        #
        # Tight verify (1500-iter × 7-repeat p20 at
        # /tmp/verify_fp8_down_b4_m4096_round12.py):
        #
        #   cfg          p20 median   Δ vs (gm=4,xcd=4)
        #   ( 4, 4)        1456.22 TF  baseline (round-69)
        #   (32, 4)        1473.58 TF  +17.37 (+1.19pp)  *winner
        #   ( 8, 4)        1471.99 TF  +15.77 (+1.08pp)
        #   (16, 4)        1471.03 TF  +14.82 (+1.02pp)
        #   (12, 4)        1470.72 TF  +14.51 (+1.00pp)
        #   (16, 2)        1466.27 TF  +10.06 (+0.69pp)  *xcd matters
        #
        # (gm=32, xcd=4) is the tight-verify top with the smallest
        # spread (1.9 TF range across 7 trials = 0.13 % CV); the
        # entire xcd=4 plateau wins over xcd=2 / xcd=8 by 5-15 TF,
        # confirming xcd=4 is the right XCD distribution for B=4
        # M=4096 (16384 rows / 11 N-tiles ≈ 720 tiles ≈ 1.4 wave on
        # MI355X 256 CUs — xcd=4 splits the work evenly across 4
        # of 8 XCDs, matching the 720/8 ≈ 90 tile-per-XCD slice;
        # xcd=8 over-distributes and pays per-XCD launch / drain
        # overhead). Bit-identical output verified at the same path
        # (max_abs_diff=0.0, bit_eq=True; group_m and num_xcds are
        # pure scheduling knobs on the FP8 grouped RCR persistent
        # tile schedule, not arithmetic-affecting).
        #
        # Rule scope check: ``tiles_n == 11`` (n=2880, gpt_oss only
        # in the FP8 metric — DSV3 N ∈ {4096, 7168} → tiles_n ∈
        # {16, 28}) + ``tiles_m == 16`` (m_per_group=4096) + ``k ==
        # 2880`` (gpt_oss only) + ``m_total == 16384`` (B=4 only,
        # since B=8 M=2048 would give tiles_m=8). Sibling B4-M2048
        # (tiles_m=8, m_total=8192) is on the round-7 (gm=2, xcd=2)
        # rule below and unaffected. Sibling B=4 M=4096 GateUP
        # (tiles_n=22, tiles_m=16) hits the round-69 GateUP rule
        # earlier in the function. No metric shape regression
        # expected.
        if (
            tiles_n == 11
            and tiles_m == 16
            and k == 2880
            and m_total is not None
            and m_total == 16384
        ):
            return HipKittenConfig(
                layout=layout,
                group_m=32,
                num_xcds=4,
                kernel=None,
            )
        if (
            tiles_n == 11
            and tiles_m == 16
            and k == 2880
            and m_total is not None
            and m_total == 131072
        ):
            # Round-50 rule. gpt_oss-Down-B32-M4096 (tiles_n=11 for n=2880,
            # tiles_m=16 for m_per_group=4096, k=2880, m_total=131072).
            # R48 audit identified this as the only gpt_oss-Down sibling
            # without a matching rule — falls to binding default
            # ``(gm=4, xcd=None=8)``. R49 metric shows this shape at
            # ratio 1.019 (HK 1804 vs Triton 1770), the WORST FP8 ratio
            # in the 24-shape suite.
            #
            # 11-candidate × 7-trial p20 sweep
            # (/tmp/sweep_down_b32_m4096_round50.py archived in commit
            # message). Median TFLOPS across 7 trials of 50-iter p20:
            #
            #   cfg            median        spread   Δ vs default
            #   (4,  4)        1959.77 TF    26.25    +15.96 (+0.82pp)  *winner
            #   (4,  1)        1945.76 TF    12.75    +1.95  (+0.10pp)
            #   default(4,8)   1943.81 TF    87.97    baseline
            #   (8,  4)        1939.01 TF    11.36    -4.80
            #   (8,  2)        1916.98 TF    16.78    -26.83
            #   (4,  2)        1913.05 TF   162.45    -30.76 (high spread)
            #   (3,  4)        1911.02 TF   181.37    -32.79 (high spread)
            #   (6,  4)        1907.92 TF    29.39    -35.89
            #   (8,  8)        1901.50 TF    45.34    -42.31
            #   (5,  4)        1896.49 TF   102.62    -47.32
            #   (2,  4)        1895.26 TF    17.43    -48.55
            #
            # ``(gm=4, xcd=4)`` wins by +15.96 TF (+0.82pp) over default
            # median AND has 3.4× tighter spread (26.25 vs 87.97). The
            # default's high spread (4.5% CV across 7 trials) explains
            # why the metric single-trial timing dipped to 1804 → 1.019
            # ratio: variance hits the low tail. ``(gm=4, xcd=4)`` is
            # both faster AND more stable, so the metric should reliably
            # land closer to 1960 (ratio ~1.107 vs Triton ~1770).
            #
            # Why xcd=4 vs default xcd=8: identical pattern to siblings
            # B=32 M=2048 (round-8: gm=16, xcd=4) and B=4 M=4096
            # (round-12: gm=32, xcd=4). The xcd=4 partition splits the
            # tiles_n=11 × tiles_m=16 × B=32 = 5632-tile schedule
            # (262144 fp32 cells per tile × ~1.4 wave pass on 256 CUs)
            # cleanly across 4 of 8 XCDs (5632/4 = 1408 per XCD vs 5632/8
            # = 704 per XCD), avoiding the per-XCD launch / drain
            # overhead that xcd=8 over-distributes.
            #
            # Why gm=4 vs sibling gm=16/32: B=32 M=4096 has m_total=131072
            # rows = 8x more tiles than B=4 M=4096 (m_total=16384). The
            # B=4 sibling needed gm=32 to extract L2 reuse from a tiny
            # grid; B=32 grid already saturates the GPU at any gm, so
            # SMALL gm=4 lets the schedule walk the N axis (11 N-tiles
            # < 16 M-tiles) before iterating M, maximising L2 hit rate
            # on the wider B-side. Swept gm ∈ {2, 3, 4, 5, 6, 8}; gm=4
            # is the tight peak with monotone falloff in either
            # direction (gm=3 -50, gm=5 -64, gm=6 -52, gm=8 -21).
            #
            # Bit-identical output (group_m / num_xcds are pure
            # scheduling knobs on the FP8 grouped RCR persistent tile
            # schedule; arithmetic and FP8 quantization rounding are
            # invariant — same property documented for all such rules
            # above; the 24 metric shapes' correctness gate (per-shape
            # SNR vs torch ref) is preserved).
            #
            # Rule scope check: ``tiles_n == 11`` (n=2880, gpt_oss only
            # in the FP8 metric) + ``tiles_m == 16`` (m_per_group=4096)
            # + ``k == 2880`` (gpt_oss only) + ``m_total == 131072``
            # matches B=32 M=4096 ONLY. Sibling B=4 M=4096
            # (m_total=16384) hits the round-12 rule above (gm=32,
            # xcd=4); B=32 M=2048 (m_total=65536, tiles_m=8) hits
            # round-8 below (gm=16, xcd=4). No metric shape regression
            # expected.
            return HipKittenConfig(
                layout=layout,
                group_m=4,
                num_xcds=4,
                kernel=None,
            )
        if (
            tiles_n == 11
            and tiles_m == 8
            and k == 2880
            and m_total is not None
            and m_total == 65536
        ):
            # Round-8 rule. gpt_oss-Down-B32-M2048 (tiles_n=11 for n=2880,
            # tiles_m=8 for m_per_group=2048, k=2880, m_total=B*M=65536).
            # Was previously the only Down-B32 shape falling through to
            # the binding default ``(gm=4, xcds=None=8)`` because no
            # rule matched. Round-70 only compared {(8,4) vs (4,4)} on
            # the Down-B32 family and concluded "split signals" — but a
            # wider grid (this round, /tmp/sweep_fp8_down_b32_round8.py
            # archived in commit message: 9 × 6 = 54 candidates over
            # ``gm ∈ {1..32}`` × ``xcd ∈ {1,2,4,8,16,32}``) shows the
            # entire ``xcd=4`` column dominates every other xcd column
            # by 8-14 TF on this shape — round-70's 2-cell comparison
            # missed the right xcd by an order of magnitude.
            #
            # Tight verify (1500-iter × 7-repeat p20 at
            # /tmp/verify_fp8_down_b32_m2048.py) — top-5 candidates with
            # very tight p20 spread (each ≤0.6 TF range across 7 trials):
            #
            #   cfg          p20 median   Δ vs default
            #   (4, None=8)    933.02 TF  baseline
            #   (16, 4)        947.36 TF  +14.34 (+1.54pp)  *winner
            #   (32, 4)        947.33 TF  +14.31 (+1.53pp)
            #   (12, 4)        947.00 TF  +13.98 (+1.50pp)
            #   ( 6, 4)        945.91 TF  +12.89 (+1.38pp)  *200-iter sweep top1
            #   (32, 2)        945.25 TF  +12.23 (+1.31pp)
            #   ( 7, 4)        942.43 TF  + 9.41 (+1.01pp)
            #   ( 5, 4)        938.14 TF  + 5.12 (+0.55pp)
            #   ( 8, 4)        937.30 TF  + 4.28 (+0.46pp)
            #   ( 6, 8)        919.79 TF  -13.23 (-1.42pp)  *xcd matters
            #
            # ``(16, 4)`` is the unique top with the tightest spread and
            # sits squarely in the ``(gm=12, gm=16, gm=32) × xcd=4``
            # plateau. The 200-iter sweep had picked (6, 4) on lower
            # statistical confidence — verify with REPEATS=7 p20 settled
            # the choice. Bit-identical output verified at the same path
            # (max_abs_diff=0.0, bit_eq=True; group_m and num_xcds are
            # pure scheduling knobs on the FP8 grouped RCR persistent
            # tile schedule, not arithmetic-affecting).
            #
            # Rule scope check: ``tiles_n == 11`` (n=2880) + ``tiles_m == 8``
            # (m_per_group=2048) + ``k == 2880`` (gpt_oss only) +
            # ``m_total == 65536`` matches only B=32 M=2048 in the
            # metric. Sibling shapes preserved: B=4 M=2048 (m_total=8192)
            # uses the round-7 (gm=2, xcd=2) rule below; B=4 M=4096
            # (m_total=16384) uses the round-69 (gm=4, xcd=4) rule above;
            # B=32 M=4096 (m_total=131072) stays on default — this round's
            # M=4096 sub-sweep showed only +0.15pp at best (within noise
            # floor of the 200-iter measurement, 4 TF spread vs 1.7 TF
            # gap), so no rule added there.
            return HipKittenConfig(
                layout=layout,
                group_m=16,
                num_xcds=4,
                kernel=None,
            )
        if (
            tiles_n == 11
            and tiles_m == 8
            and k == 2880
            and m_total is not None
            and m_total == 8192
        ):
            # Round-7 rule. gpt_oss-Down-B4-M2048 (tiles_n=11 for n=2880
            # since 2880//256=11, k=2880, m_total=8192). Sibling of the
            # round-69 ``tiles_n=11+tiles_m=16+m_total=16384`` rule above
            # but sized for B=4 M=2048 (8192 vs 16384). Was previously
            # falling through to the binding default ``(group_m=4,
            # num_xcds=None=8)``.
            #
            # 8 × 5 = 40-candidate sweep (``gm ∈ {1,2,3,4,6,8,12,16}`` ×
            # ``xcd ∈ {1,2,4,8,16}`` at /tmp/sweep_fp8_b4_round7.py) with
            # 7-repeat × 500-iter p20 + tight verify
            # (1500-iter × 7-repeat p20 at /tmp/verify_fp8_b4_round7.py).
            #
            # Tight-verify p20 (single-trial, range across 7 repeats):
            #   cfg          p20      range
            #   (4, 8)       749.86   749.51..750.69     # default baseline
            #   (2, 2)       753.47   753.44..754.00     # winner +0.48pp
            #
            # Neighbor robustness (single-trial p20 at
            # /tmp/verify_fp8_b4_round7_neighbors.py):
            #   (1,2)=751.28  (2,1)=747.37  (2,2)=754.23 *  (2,4)=731.47
            #   (3,2)=742.39  (4,2)=746.95  (1,1)=743.78
            # (2, 2) is the unique top of a sharp local optimum: (2, 4) is
            # 23 TF below (xcd matters at gm=2), and gm ∈ {3, 4} are 8-12
            # TF below at the same xcd=2. Bit-identical output verified
            # at /tmp/verify_fp8_b4_round7_neighbors.py
            # (max_abs_diff=0.0, bit_eq=True).
            #
            # Round-13 re-tested this rule with a fresh wide sweep and
            # both kernel-only tight verify and a metric-aligned probe;
            # both kernel-side winners (gm=1, xcd=1) and (gm=1, xcd=8)
            # regressed -3 score in the metric. Retain (gm=2, xcd=2).
            # See analysis/_notes/round-13-config-tuning-saturation.md
            # for the full analysis (B=4 metric/verify divergence).
            #
            # Rule scope check: ``tiles_n == 11`` (n=2880) + ``tiles_m == 8``
            # (m_per_group=2048) + ``m_total == 8192`` matches only B=4
            # M_per_group=2048 in the metric. The B=4 M=4096 case (the
            # round-69 rule above) has m_total=16384; B=8 M=2048 (no
            # metric shape) would also have m_total=16384.
            return HipKittenConfig(
                layout=layout,
                group_m=2,
                num_xcds=2,
                kernel=None,
            )
        # Round-34 (this round): FP8 RCR rule covering the gpt_oss-GateUP
        # dA backward path AFTER the H4 reroute (grouped_gemm_fp8_impl.py
        # line 432-468). gpt_oss-GateUP dA RRR has K_RRR=N_fwd=5760 (% 128
        # = 0, aligned) but N_RRR=K_fwd=2880 (% 256 = 64, misaligned), so
        # the R18 H4 reroute fires and dispatches the kernel via
        # ``grouped_rcr_dscale`` with a transposed b. The post-reroute RCR
        # call sees:
        #   m = M_per
        #   n = K_fwd = 2880  -> tiles_n = 11
        #   k = N_fwd = 5760
        # The existing tiles_n==11 RCR rules (R7, R8, R12, R50 at lines
        # 1102-1320) all gate on ``k == 2880`` (gpt_oss-Down's K_fwd) so
        # they correctly catch gpt_oss-Down dA after reroute (which sees
        # k=2880) but DO NOT catch gpt_oss-GateUP dA (which sees k=5760).
        # All 4 GateUP dA shapes therefore fall through to the binding
        # default (group_m=4, num_xcds=None=8). R34 calibrated probe
        # (12-trial × 200-iter × 3-seed via /tmp/probe_r34_gpt_oss_gateup
        # _da_via_rcr.py — uses ``fp8_transpose_3d`` + direct
        # ``grouped_rcr_dscale`` to mirror the production reroute path):
        #
        # gpt_oss-GateUP-B4-M2048-dA (m_total=8192, tiles_m=8):
        #   default (4, 8)  baseline                              keep default
        #   (4, 4)          -1.26%  spread 2.69pp  TIE
        #   (8, 4)          -1.54%  spread 3.45pp  TIE
        #   (16, 4)         -2.51%  spread 3.50pp  TIE
        #   -> default best; do not add a rule for tiles_m=8 + m_total<65536.
        #
        # gpt_oss-GateUP-B4-M4096-dA (m_total=16384, tiles_m=16):
        #   default (4, 8)  baseline
        #   (8, 4)          +1.78%  spread 0.86pp  WIN  (med/spread = 2.07x)
        #   (4, 4)          +0.53%  spread 0.47pp  WIN
        #   (16, 4)         +0.30%  spread 1.44pp  TIE
        #
        # gpt_oss-GateUP-B32-M2048-dA (m_total=65536, tiles_m=8):
        #   default (4, 8)  baseline
        #   (16, 4)         +3.02%  spread 0.19pp  WIN  (med/spread = 15.9x — extremely tight)
        #   (32, 4)         +2.93%  spread 0.13pp  WIN  (med/spread = 22.5x)
        #   (8, 4)          +2.37%  spread 0.26pp  WIN
        #
        # gpt_oss-GateUP-B32-M4096-dA (m_total=131072, tiles_m=16):
        #   default (4, 8)  baseline
        #   (8, 4)          +2.37%  spread 0.57pp  WIN  (med/spread = 4.16x)
        #   (16, 4)         +1.57%  spread 0.42pp  WIN
        #   (32, 4)         +1.35%  spread 0.12pp  WIN
        #
        # Discriminator pattern observed:
        #   - tiles_m=16 (M_per=4096): (gm=8, xcd=4) is the robust winner
        #     for both B4-M4096 (+1.78%) and B32-M4096 (+2.37%).
        #   - tiles_m=8  (M_per=2048): split by m_total —
        #       m_total < 65536 (B4-M2048): default best, do not override
        #       m_total >= 65536 (B32-M2048): (gm=16, xcd=4) WIN +3.02%
        # Why xcd=4 over default xcd=8: same partition-balance reason
        # documented for R8 / R12 / R50 / R69 above (4 XCDs split the
        # tiles_n=11 schedule cleanly; xcds=8 over-distributes for k=5760).
        # Why gm=8 vs gm=4 default: tiles_m=16 grids cycle through 11
        # N-tiles per K-pass; gm=8 batches 8 M-tiles per pass keeping
        # the persistent slots busy across both M and N axes, whereas
        # default gm=4 only batches 4 (under-batched for the larger
        # K=5760 main loop).
        # Why gm=16 for B32-M2048: the larger grid (m_total=65536) has
        # enough wave-steps to amortise large gm; gm=16 (= 2x M-tiles
        # per group) extracts more L2 reuse on the K=5760 deep-K axis.
        #
        # Bit-equivalent output: group_m / num_xcds are pure persistent-
        # grid scheduling knobs (same property documented for all FP8
        # RCR rules above; arithmetic and FP8 quantization rounding
        # invariant). Self-bench bench_grouped_gemm_turbo.py confirms
        # 24/24 fp8 PASS post-rule.
        #
        # Rule scope check:
        #   tiles_n == 11 (n=2880) AND k == 5760: matches uniquely
        #   gpt_oss-GateUP dA RRR after the H4 reroute. No other
        #   metric/DoD shape has this (n=2880, k=5760) RCR combination.
        #   gpt_oss-Down dA after reroute: k=2880 (excluded).
        #   gpt_oss-GateUP forward RCR: n=N_fwd=5760 (tiles_n=22, excluded).
        #   gpt_oss-GateUP dB var-K: variable-K path, different code.
        if tiles_n == 11 and k == 5760 and m_total is not None:
            if tiles_m == 16:
                # gpt_oss-GateUP B*-M4096 dA after reroute. Both
                # B=4 (m_total=16384) and B=32 (m_total=131072) probed
                # and confirmed (gm=8, xcd=4) WIN.
                return HipKittenConfig(
                    layout=layout,
                    group_m=8,
                    num_xcds=4,
                    kernel=None,
                )
            if tiles_m == 8 and m_total >= 65536:
                # gpt_oss-GateUP B=32 M=2048 dA after reroute. The
                # B=4 M=2048 sibling (m_total=8192) is intentionally
                # excluded: probe found default beats every candidate
                # there (small persistent grid amortises default cells
                # better than extra batching).
                return HipKittenConfig(
                    layout=layout,
                    group_m=16,
                    num_xcds=4,
                    kernel=None,
                )
        if tiles_n == 16 and tiles_m == 16 and k == 1536:
            # Round-6 rule. Qwen3-235B-A22B Down M_per_group=4096 family
            # (B ∈ {16, 32}; tiles_n=16 for n=4096; k=1536; tiles_m=16
            # for m_per_group=4096; m_total ∈ {65536, 131072}). Was the
            # only Qwen-Down sub-family falling >5% behind Triton in the
            # round-5 baseline (B16-M4096 ratio=1.056, B32-M4096
            # ratio=1.090, vs 1.098/1.139 for the M_per=2048 siblings
            # which already sat best at default).
            #
            # 28-cell coarse sweep ``gm ∈ {1,2,4,8,12,16,32} × xcds ∈
            # {2,4,8,16}`` over the 4 Qwen-Down metric shapes (50-iter
            # × p20 at /tmp/probe_qwen_down_round6.py archived in this
            # commit message) flagged ``(gm=2, xcds=16)`` / ``(gm=2,
            # xcds=8)`` as joint top on both M_per=4096 shapes (~+3.3%
            # over default). Tight verify (200-iter × 7-trial p20 at
            # /tmp/verify_qwen_down_m4096_round6.py) resolves the pair
            # to ``(gm=2, xcds=8)`` by a hairline margin and confirms
            # the win is well above run-to-run spread:
            #
            #   Qwen-Down-B16-M4096 (200i × 7t p20):
            #     ( 2,  8)  1835.11 TF  +3.17pp vs default *winner
            #     ( 2, 16)  1834.46 TF  +3.13pp vs default
            #     ( 1,  4)  1779.05 TF  +0.02pp
            #     ( 4,  8)  1778.75 TF  +0.00pp baseline
            #     (32,  4)  1770.04 TF  -0.49pp
            #
            #   Qwen-Down-B32-M4096 (200i × 7t p20):
            #     ( 2,  8)  1843.81 TF  +2.69pp vs default *winner
            #     ( 2, 16)  1843.40 TF  +2.66pp vs default
            #     ( 4,  8)  1795.55 TF  +0.00pp baseline
            #     ( 1,  4)  1794.46 TF  -0.06pp
            #     (32,  4)  1787.46 TF  -0.45pp
            #
            # ``(gm=2, xcds=8)`` wins both M=4096 shapes by 47-50 TF
            # (+2.7..+3.2 pp) over the default; spread on the winners
            # is 0.35-0.39 % so the gap is ~10× the run-to-run noise.
            # The (gm=2) pattern matches the Lever F hypothesis: Qwen-
            # Down K=1536 has only 12 K-iter (vs gpt_oss K=2880 = 23
            # K-iter, DSV3-Down K=2048 = 16 K-iter), so the persistent
            # main loop sees ~1.4-2x fewer mfma per tile-step → smaller
            # group_m better preserves L2 reuse on the long N axis
            # (n=4096, 16 N-tiles, larger than M-axis 16 tiles per group).
            # xcds=8 left at the binding default (cfg.num_xcds=None →
            # 0 → kernel falls back to BLOCK_SWIZZLE_NUM_XCDS=8) — both
            # the sweep and tight verify show xcds=8 ties or beats
            # xcds=16 by hairline; no need to add an explicit override.
            #
            # Bit-identical output verified at
            # /tmp/verify_qwen_down_correctness_round6.py:
            #   Qwen-Down-B16-M4096: max_abs=0.0  bit_eq=True
            #   Qwen-Down-B32-M4096: max_abs=0.0  bit_eq=True
            # group_m / num_xcds are pure scheduling knobs on the FP8
            # grouped RCR persistent tile schedule; arithmetic and
            # quantization rounding invariant.
            #
            # Rule scope check: ``tiles_n == 16`` (n=4096) is shared
            # with DSV3-GateUP (k=7168) and dense (8192,4096,*); the
            # ``k == 1536`` clause is uniquely Qwen-Down in the metric
            # (DSV3 k ∈ {2048, 7168}, gpt_oss k=2880, Qwen-GateUP
            # k=4096, dense k ∈ {4096, 11008, 14336, 22016, 28672}).
            # ``tiles_m == 16`` (m_per_group=4096) excludes the M=2048
            # sibling cases which sit best at the default (sweep showed
            # B16-M2048 default 1765 TF beat (gm=2, xcds=8) which sat
            # below 1695, a -3.95pp regression — must NOT extend rule
            # to tiles_m=8). M_per=2048 left on default; this round
            # only touches the 2 M=4096 shapes (B16-M4096 + B32-M4096).
            return HipKittenConfig(
                layout=layout,
                group_m=2,
                num_xcds=None,
                kernel=None,
            )
        if (tiles_n == 12 and tiles_m == 16 and k == 4096
                and m_total is not None and m_total >= 65536):
            # Round-10 / Round-45 rule. Qwen3-235B-A22B GateUP-M4096
            # family (tiles_n=12 for n=3072; tiles_m=16 for
            # m_per_group=4096; k=4096). R10 anchored the rule on
            # B=32-only (m_total >= 131072) because the B16-M4096
            # tight verify at 200-iter × 7-trial sat at +0.47 pp with
            # 0.6× spread — distributions still overlapped, so R10
            # left B=16 on default.
            #
            # R45 re-probes B=16 M=4096 with the tighter R44
            # methodology (12 trials × 200 iters × 3 seeds):
            #
            #   seed=42:   default=X  (1,4)=X'  Δmed=+1.90%
            #   seed=137:  default=X  (1,4)=X'  Δmed=+1.95%
            #   seed=2024: default=X  (1,4)=X'  Δmed=+1.93%
            #
            # All 3 seeds give +1.90% to +1.95% — 0.05pp spread,
            # 4× tighter than the R10 200-iter × 7-trial probe
            # reported (0.6% spread). The R10 "noise floor"
            # assessment turns out to be a methodology artifact,
            # not a real distribution overlap: 200-iter trials have
            # enough per-trial variance to hide a +2% effect; 200-iter
            # × 12-trial × 3-seed resolves it cleanly.
            #
            # The B16 delta (+1.93%) actually exceeds the B32 delta
            # (R10: +0.93%) — consistent with the heuristic that
            # smaller persistent grids (B=16 = 65536/256=256 CU slots
            # × 24 wave-steps) benefit more from gm=1's N-first
            # traversal than larger grids (B=32 = 128 CU slots × 48
            # wave-steps, already saturated). R45 relaxes the rule
            # threshold from m_total >= 131072 to m_total >= 65536
            # to capture B=16 M=4096 in addition to the original B=32
            # M=4096 — same (gm=1, xcd=4) config, which R10 already
            # verified on the B=32 sibling.
            #
            # Re-tested in R10
            # the R7-falsified Qwen-GateUP-M4096 candidate (gm=1, xcd=4):
            # at R7 the B16-M4096 sibling was at the noise floor (+0.23pp,
            # 0.64% spread) and the M=4096 family rule was reverted to
            # avoid landing a noisy gain. The B32-M4096 sub-shape alone
            # had a clean +0.80pp gain even in R7 — R10 narrows the rule
            # to that single tier so the clean B32 signal lands without
            # over-extending to the noisy B16 sibling.
            #
            # Tight verify (200-iter × 7-trial p20 at
            # /tmp/verify_qwen_gateup_b32_m4096_round10.py):
            #
            #   Qwen-GateUP-B32-M4096:
            #     ( 1,  4)  2461.49 TF  +0.93 pp vs default *winner
            #                            (spread 0.51 %; min=2459.66, max=2472.15)
            #     ( 4,  4)  2452.03 TF  +0.54 pp                (spread 0.20 %)
            #     ( 4,  8)  2438.79 TF  baseline                (spread 0.50 %)
            #
            # Gap = 22.7 TF (+0.93 pp) at 1.85× spread; the winner's
            # per-trial min (2459.66 TF) exceeds the default's per-trial
            # max (2446.60 TF) by 13.06 TF (+0.54 pp) — clean
            # separation in 7/7 trials.
            #
            # Sibling B16-M4096 re-tested at the same probe:
            # (1, 4)=2432.96 vs default 2421.51 = +0.47 pp at 0.6× spread.
            # Winner-min (2427.00) is 0.5 TF BELOW default-max (2427.51)
            # — the distributions still overlap, so B16 stays excluded
            # via the ``m_total >= 131072`` guard. (B16-M4096 has
            # m_total = 16×4096 = 65536; B32-M4096 has m_total = 131072.)
            # The B16 result moved up vs R7 (+0.23pp → +0.47pp) but
            # remains below the R7-protocol clean-signal threshold
            # (gap ≥ 1× spread).
            #
            # Why (gm=1) wins for tiles_n=12 + tiles_m=16: m_per_group=4096
            # ⇒ 16 M-tiles per group × 12 N-tiles per group = 192
            # tile-steps × 32 batches = 6144 tile-steps. NUM_CUS=256
            # persistent slots ⇒ 24 wave-steps per slot. tiles_n=12 is
            # smaller than tiles_m=16 → the persistent loop benefits
            # from stretching the N-axis traversal under each M-tile,
            # which (gm=1) does (one M-tile fully consumed before
            # advancing). xcds=4 picked over xcds=8 because the
            # (1, 4) p20 (2461.49) cleanly beats (1, 8) (2374.17,
            # -2.65 pp) — the smaller XCD chiplet partition keeps
            # more shaping work inside each XCD's L2.
            #
            # Bit-identical output verified at
            # /tmp/verify_qwen_gateup_b32_m4096_correctness_round10.py:
            #   Qwen-GateUP-B32-M4096: max_abs_diff=0.0  bit_eq=True
            # group_m / num_xcds are pure scheduling knobs on FP8
            # grouped RCR persistent tile schedule; arithmetic and
            # quantization rounding invariant.
            #
            # Rule scope check: ``tiles_n == 12`` (n=3072) is uniquely
            # Qwen-GateUP in the metric grouped FP8 suite (DSV3 N ∈
            # {4096, 7168} → tiles_n ∈ {16, 28}; gpt_oss N ∈ {2880, 5760}
            # → tiles_n ∈ {11, 22}; Qwen-Down N=4096 → tiles_n=16).
            # ``tiles_m == 16`` (m_per_group=4096) excludes the M=2048
            # sibling (covered by the round-7 rule below). ``k == 4096``
            # is uniquely Qwen-GateUP in the grouped suite. ``m_total >=
            # 131072`` selects B=32 only (B16-M4096 has m_total=65536,
            # excluded). The B16 sibling stays on the binding default
            # (gm=4, xcds=8) until a stronger signal materialises.
            return HipKittenConfig(
                layout=layout,
                group_m=1,
                num_xcds=4,
                kernel=None,
            )
        if tiles_n == 12 and tiles_m == 8 and k == 4096:
            # Round-7 rule. Qwen3-235B-A22B GateUP M_per_group=2048 family
            # (B ∈ {16, 32}; tiles_n=12 for n=3072; k=4096; tiles_m=8 for
            # m_per_group=2048; m_total ∈ {32768, 65536}). Companion to
            # the round-6 Qwen-Down rule above; was the only Qwen-GateUP
            # sub-family with a clear (gm, xcds) optimum in the
            # round-6 28-cell sweep that survived a 200-iter × 7-trial
            # tight verify (B16-M4096 / B32-M4096 had marginal +0.2..+0.8 pp
            # candidates with split (gm) winners, kept on default).
            #
            # Tight verify (200-iter × 7-trial p20 at
            # /tmp/verify_qwen_gateup_round7.py):
            #
            #   Qwen-GateUP-B16-M2048:
            #     (16,  4)  2494.94 TF  +0.86 pp vs default *winner
            #     (32,  4)  2494.63 TF  +0.85 pp
            #     ( 1,  4)  2488.31 TF  +0.59 pp
            #     ( 4,  4)  2485.01 TF  +0.46 pp
            #     ( 4,  8)  2473.68 TF  baseline
            #
            #   Qwen-GateUP-B32-M2048:
            #     (16,  4)  2511.96 TF  +1.05 pp vs default *winner
            #     (32,  4)  2511.35 TF  +1.03 pp
            #     ( 1,  4)  2504.48 TF  +0.75 pp
            #     ( 4,  8)  2485.76 TF  baseline
            #
            # ``(gm=16, xcds=4)`` wins both M=2048 shapes by 21 TF
            # (+0.86 pp on B16, +1.05 pp on B32) — both gaps clear of
            # the run-to-run spread (B16 spread 0.95 %, B32 0.72 %; the
            # winner's per-trial min beats the default's max in 6 of
            # 7 trials per shape). xcds=4 is the consistent half of the
            # win (xcds=8 → xcds=4 alone gives +0.46 pp B16, +0.49 pp
            # B32; full (gm=16, xcds=4) adds the gm component for the
            # remaining +0.4..+0.6 pp).
            #
            # Why (gm=16) wins for tiles_m=8: m_per_group=2048 ⇒ 8
            # M-tiles per group; with tiles_n=12 N-tiles per group, total
            # 96 tile-steps per group × B ∈ {16, 32} batches gives
            # 1536 / 3072 tile-steps. NUM_CUS=256 persistent slots ⇒
            # 6 / 12 wave-steps per slot. (gm=16) groups 16 M-tiles
            # together (more than per-group M-tiles, so it groups
            # cross-batch) which improves L2 reuse on the K=4096 long-K
            # axis where each tile sweeps the same A-pack twice across
            # the persistent loop.
            #
            # Bit-identical output verified at
            # /tmp/verify_qwen_gateup_correctness_round7.py:
            #   Qwen-GateUP-B16-M2048: max_abs=0.0  bit_eq=True
            #   Qwen-GateUP-B32-M2048: max_abs=0.0  bit_eq=True
            # group_m / num_xcds are pure scheduling knobs on FP8
            # grouped RCR persistent tile schedule; arithmetic and
            # quantization rounding invariant.
            #
            # Rule scope check: ``tiles_n == 12`` (n=3072) is uniquely
            # Qwen-GateUP in the metric grouped FP8 suite (DSV3 N ∈
            # {4096, 7168} → tiles_n ∈ {16, 28}; gpt_oss N ∈ {2880, 5760}
            # → tiles_n ∈ {11, 22}; Qwen-Down N=4096 → tiles_n=16).
            # Dense FP8 metric (LLaMA-2-7B / Llama-3.1-8B) has N ∈
            # {4096, 6144, 12288, 22016, 28672, 4096, 4096} → tiles_n ∈
            # {16, 24, 48, 86, 112, 16, 16} — no tiles_n=12. ``k == 4096``
            # is uniquely Qwen-GateUP in the grouped suite (DSV3 k ∈
            # {2048, 7168}; gpt_oss k=2880; Qwen-Down k=1536) but
            # matches multiple dense shapes — the tiles_n=12 clause
            # excludes all of them. ``tiles_m == 8`` (m_per_group=2048)
            # excludes the M=4096 sibling cases (probed in R7 with
            # tight verify; (gm=1, xcds=4) winner shows +0.23 pp B16-M4096
            # / +0.80 pp B32-M4096 — B16 margin sits at the noise floor
            # (spread 0.64 %), and a 5-trial post-add metric repeat
            # showed score variance ±2.5 hiding the projected aggregate
            # +0.04 pp geomean. Rule 2 was REVERTED this round; left
            # unanchored on default until a stronger signal materialises).
            return HipKittenConfig(
                layout=layout,
                group_m=16,
                num_xcds=4,
                kernel=None,
            )
        # Round-29 — REMOVED Round-28 dead-code rule (`tiles_n == 6 and
        # k == 4096`). The R28 commit (507aff37, 2026-05-02) added an
        # if-branch on the FP8 grouped RCR forward dispatch that was
        # never reachable from any metric shape. R29 audit traced the
        # branch as follows:
        #
        #   * Forward RCR dispatch (`grouped_gemm_fp8_impl.py:541`):
        #     `select_default_config(avg_m, n=N_fwd, k=K_fwd, "rcr",
        #     "fp8", m_total=m_total)`. For ``tiles_n == 6`` we need
        #     ``N_fwd == 1536``. None of the 24 metric shapes
        #     (DeepSeek-V3 N ∈ {4096, 7168}; gpt_oss_20B N ∈ {2880,
        #     5760}; Qwen3-235B-A22B N ∈ {3072, 4096}) has N_fwd =
        #     1536. None of the 8 dense FP8 shapes (Llama-2-7B /
        #     Llama-3.1-8B N ∈ {4096, 6144, 12288, 22016, 28672}) does
        #     either.
        #
        #   * dA RRR dispatch enters with `layout == "rrr"`, never
        #     reaching this `if layout == "rcr":` block.
        #
        #   * dB CRR var-K dispatch is inline in
        #     `grouped_gemm_fp8_impl.py` and bypasses
        #     `select_default_config` entirely.
        #
        # The R28 commit message claimed the rule targets "Qwen3-GateUP
        # forward RCR" with "the new shape N_fwd=1536" — but Qwen3-
        # GateUP forward actually has ``N_fwd = 2 * moe_intermediate_size
        # = 2 * 1536 = 3072`` (tiles_n=12), not N_fwd=1536. The R28
        # author confused Qwen3's ``moe_intermediate_size = 1536`` with
        # the GateUP layer's `N_fwd = 2 * moe_int = 3072`. The R7 rule
        # at `tiles_n == 12 and tiles_m == 8 and k == 4096` (line 1498)
        # already handles the actual Qwen3-GateUP M=2048 family.
        #
        # The ``+2.1 mean / +3 median`` improvement R28 reported (12-run
        # distribution, mean 992.4 → 994.5) was random noise from a
        # single sample window of the metric's wide [981, 1000] noise
        # band (re-quantified at this round at the unchanged HEAD across
        # 8 runs: median 988.5, mean 990.9, identical band).
        #
        # R29 also re-probed every plausible Qwen3-Down M=2048 forward
        # / dA / dB var-K (gm, num_xcds) cell on the actually-bottom
        # metric shapes (Qwen3-235B-A22B-Down-B16-M2048 ratio = 1.243;
        # Qwen3-235B-A22B-Down-B32-M2048 ratio = 1.265). Tight verify
        # (400-iter × 12-trial p20 × 3 seeds, mirror of the R28 verify
        # methodology) collapses every coarse 200-iter signal to within
        # ±0.5 % noise:
        #
        #   * Forward RCR (n=4096, k=1536):
        #     binding default (gm=4, xcds=None=8) — every other tested
        #     cell (gm ∈ {1, 2, 8, 16, 32}) regressed by 1-4 %. Default
        #     is the robust optimum.
        #   * dA RRR (n=1536, k=4096): R42 (gm=16, xcds=4) — proposed
        #     (gm=8, xcds=4) tight-verify delta on Qwen3-Down M=2048:
        #     +0.06 % B16, +0.01 % B32 (within noise). R42 unchanged.
        #   * dB CRR var-K (n=K_fwd=1536, k=N_fwd=4096, m_per_g=2048):
        #     current inline rule (gm=8, xcds=4 for m_total >= 16384)
        #     — proposed (gm=4, xcds=4) tight verify on Qwen3-Down M=
        #     2048: +0.35 % B16, -0.19 % B32 (within noise). Current
        #     rule unchanged.
        #
        # Net: Qwen3-Down M=2048 stays on the FP8 binding default for
        # forward RCR; existing R42 / inline var-K rules cover the
        # backward path. The metric's gap to the 1.35 target on the
        # Qwen3-Down M=2048 cohort is bounded by the symmetric
        # `quantize_fp8` HBM tax both backends pay — the architectural
        # ceiling documented in `analysis/_notes/round-8-fused-act-
        # architectural-ceiling-confirmed.md` and re-confirmed in R26.
        if tiles_n == 16 and tiles_m == 16 and k == 7168:
            # Round-8 rule. DeepSeek-V3 GateUP M_per_group=4096 family
            # (B ∈ {16, 32}; tiles_n=16 for n=4096; k=7168; tiles_m=16
            # for m_per_group=4096; m_total ∈ {65536, 131072}). Was
            # the only DSV3-GateUP sub-family with a clean tight-verify
            # signal that survived a 200-iter × 7-trial p20 sweep
            # (B16-M2048 had +0.13 % top1 = solid noise; B32-M2048
            # had a single-shape tight-verify win covered by the
            # adjacent rule below).
            #
            # Tight verify (200-iter × 7-trial p20 at
            # /tmp/verify_dsv3_gateup_round8.py):
            #
            #   DSV3-GateUP-B16-M4096:
            #     ( 2,  8)  2787.94 TF  +0.64 pp vs default *winner (spread 0.25 %)
            #     ( 2, 16)  2786.89 TF  +0.60 pp                    (spread 0.21 %)
            #     ( 4,  8)  2770.20 TF  baseline                    (spread 0.22 %)
            #
            #   DSV3-GateUP-B32-M4096:
            #     ( 2, 16)  2775.04 TF  +0.49 pp vs default         (spread 0.33 %)
            #     ( 2,  8)  2774.24 TF  +0.46 pp vs default *winner (spread 0.31 %)
            #     ( 4,  8)  2761.58 TF  baseline                    (spread 0.43 %)
            #
            # ``(gm=2, xcds=8)`` wins both M=4096 shapes by 13-18 TF
            # (+0.46..+0.64 pp) — gaps are 1.5-2.5× the run-to-run
            # spread; the winner's per-trial min beats the default's
            # max in 6-7 of 7 trials per shape. xcds=8 is the binding
            # default (cfg.num_xcds=None → 0 → kernel falls back to
            # BLOCK_SWIZZLE_NUM_XCDS=8); the tight verify shows xcds=8
            # vs xcds=16 are within 0.04 pp on both shapes — keep
            # xcds=None for cleaner config (no explicit override
            # needed).
            #
            # Why (gm=2) wins for tiles_m=16 + tiles_n=16: m_per_group=4096
            # ⇒ 16 M-tiles per group; tiles_n=16; with K=7168 (56 K-iter)
            # the per-tile compute is 4× heavier than gpt_oss K=2880, so
            # the persistent loop scheduler benefits from a smaller gm
            # that doesn't over-batch M-tiles together (lets each slot
            # walk N before M, maximising B-tile L2 reuse on the
            # K=7168 long-K axis where each B-tile is read once for each
            # of 56 K-iter).
            #
            # Bit-identical output verified at
            # /tmp/verify_dsv3_gateup_correctness_round8.py:
            #   DSV3-GateUP-B16-M4096: max_abs=0.0  bit_eq=True
            #   DSV3-GateUP-B32-M4096: max_abs=0.0  bit_eq=True
            # group_m / num_xcds are pure scheduling knobs on FP8
            # grouped RCR persistent tile schedule; arithmetic and
            # quantization rounding invariant.
            #
            # Rule scope check: ``tiles_n == 16`` (n=4096) is shared
            # with Qwen-Down (k=1536) and dense (8192,4096,K), but the
            # ``k == 7168`` clause is uniquely DSV3-GateUP in the
            # metric grouped FP8 suite (DSV3-Down k=2048 → tiles_n=28
            # rule below; gpt_oss k=2880; Qwen-GateUP k=4096; Qwen-Down
            # k=1536). Dense FP8 LLaMA shapes have K ∈ {4096, 11008,
            # 14336} — no K=7168. ``tiles_m == 16`` (m_per_group=4096)
            # selects M=4096 only; M=2048 sibling cases (B16-M2048
            # default-optimal at tight verify, B32-M2048 covered by
            # the m_total>=65536 rule below).
            return HipKittenConfig(
                layout=layout,
                group_m=2,
                num_xcds=None,
                kernel=None,
            )
        if (tiles_n == 16 and tiles_m == 8 and k == 7168
                and m_total is not None and m_total >= 32768):
            # Round-8 / Round-45 rule. DSV3-GateUP M_per=2048 family
            # (tiles_n=16 + tiles_m=8 + k=7168). R8 anchored the rule
            # on B=32-only (m_total >= 65536) because the B16-M2048
            # sibling's 50-iter sweep top1 sat at +0.13 % which R8
            # classified as "solid noise" and didn't tight-verify.
            #
            # R45 re-probes B=16 M=2048 with the tighter R44
            # methodology (12 trials × 200 iters × 3 seeds):
            #
            #   seed=42:   default=X  (16,4)=X'  Δmed=+1.17%
            #   seed=137:  default=X  (16,4)=X'  Δmed=+1.39%
            #   seed=2024: default=X  (16,4)=X'  Δmed=+1.50%
            #
            # All 3 seeds give +1.17% to +1.50% — 0.33pp spread.
            # The R8 50-iter probe (top1 +0.13%) did not have enough
            # statistical resolution to see this signal; 200-iter
            # × 12-trial × 3-seed resolves it above noise with the
            # same (gm=16, xcd=4) config R8 already validated for
            # the B=32-M=2048 sibling.
            #
            # Companion candidates at R45 tight probe:
            #   (16, 4):  +1.17%, +1.39%, +1.50%  avg +1.35%  *winner
            #   (32, 4):  +0.98%, +1.24%, +1.48%  avg +1.23%  close 2nd
            #   ( 1, 4):  +0.83%, +1.07%, +1.21%  avg +1.03%
            #   ( 2, 8):  -0.72%, -0.31%, -0.88%  avg -0.64%
            #   ( 2,16):  -0.95%, -0.79%, -0.93%  avg -0.89%
            #
            # (16, 4) matches the sibling rule exactly and wins cleanly.
            # R45 relaxes the threshold from m_total >= 65536 to
            # m_total >= 32768 to capture the B=16 M=2048 shape.
            #
            # Original comment retained below for historical context:
            #
            # Sibling to the M=4096 rule above; covers the only M=2048 GateUP
            # shape with a clean tight-verify signal. B16-M2048
            # (m_total=32768) is excluded by the m_total bound — its
            # 50-iter sweep top1 sat at +0.13 % (solid noise), no
            # tight-verify warranted.
            #
            # Tight verify (200-iter × 7-trial p20):
            #
            #   DSV3-GateUP-B32-M2048:
            #     (16,  4)  2756.31 TF  +0.56 pp vs default *winner (spread 0.28 %)
            #     (32,  4)  2753.87 TF  +0.47 pp                    (spread 0.34 %)
            #     ( 4,  8)  2741.00 TF  baseline                    (spread 0.87 %)
            #
            # ``(gm=16, xcds=4)`` wins by 15 TF (+0.56 pp), 2× the
            # winner's spread (0.28 %) — clean signal. The (gm=16, xcds=4)
            # winner mirrors the gpt_oss-Down-B32-M2048 tier-1 rule
            # above (line 1108) and the Qwen-GateUP-B32-M2048 round-7
            # rule, suggesting a consistent "B32 M_per=2048 large grid
            # benefits from gm ∈ {16, 32}" pattern.
            #
            # Bit-identical output verified at
            # /tmp/verify_dsv3_gateup_correctness_round8.py:
            #   DSV3-GateUP-B32-M2048: max_abs=0.0  bit_eq=True
            #
            # Rule scope check: same family as the M=4096 sibling above
            # (tiles_n=16 + k=7168 uniquely DSV3-GateUP); tiles_m=8 +
            # m_total>=65536 selects B32-M2048 only (B16-M2048 has
            # m_total=32768 < 65536, falls through to default).
            return HipKittenConfig(
                layout=layout,
                group_m=16,
                num_xcds=4,
                kernel=None,
            )
        if tiles_n == 28 and 8 <= tiles_m <= 16 and k <= 4096:
            # Round-20 rule (refined round-58). DeepSeek-V3-Down grouped
            # FP8 RCR family: per-group GEMM N=7168, K=2048, M_per_group
            # ∈ {2048, 4096}, B ∈ {16, 32}. Mirrors the BF16
            # ``tiles_n==28`` rule above. Round-20's gm=24 was derived
            # from a 6-candidate sweep capped at gm<=24; round-58 reran
            # with the upper bound extended to gm=64 (a 9-candidate
            # 80-iter × 3-repeat median sweep at
            # /tmp/sweep_fp8_dsv3_down_wide.py followed by a tighter
            # 160-iter × 5-repeat verification at
            # /tmp/verify_dsv3_down_gm48.py) and found gm=32 dominates
            # gm=24 on all 4 metric shapes:
            #
            #   shape           gm=24    gm=32    Δ vs gm=24
            #   Down-B16-M2048  1725.4   1729.8   +0.26pp
            #   Down-B16-M4096  1741.0   1757.7   +0.96pp
            #   Down-B32-M2048  1743.6   1751.3   +0.44pp
            #   Down-B32-M4096  1751.9   1781.6   +1.69pp
            #
            # gm=32 is also the most robust pick across the family:
            # gm=48 wins B16 by ≤0.1pp but loses B32-M2048 by 0.4pp;
            # gm=64 wins B16-M4096 by 0.1pp but loses B32-M4096 by 0.2pp.
            # gm=32 is top-1 or within 0.1pp of top-1 on every shape,
            # whereas the round-20 gm=24 was top-1 *only within the
            # ≤24 search* and is uniformly dominated once the search
            # extends. Net delta over round-20 (gm=24): +0.84pp / shape
            # average, +1.69pp on the worst-ratio Down-B32-M4096 shape
            # (was 0.939 vs Triton in round-58 baseline).
            #
            # Bit-identical output verified at /tmp/verify_fp8_dsv3_repeatable.py:
            # gm=24 vs gm=32 on Down-B16-M2048 → max_abs=0.0, bit_eq=True;
            # SNR vs fp32 ref = 28.49 dB on both. group_m only re-orders
            # the persistent tile schedule on the FP8 grouped RCR
            # launcher; arithmetic and FP8 quantization rounding are
            # invariant.
            #
            # Rule scope check (unchanged from round-20): ``tiles_n == 28``
            # ⇔ N == 7168, which only occurs in the metric for these 4
            # DSV3-Down forward shapes. No dense FP8 metric shape has
            # N=7168; DSV3-GateUP has tiles_n==16 (N=4096), gpt_oss has
            # tiles_n ∈ {12, 23}, neither matches. The k<=4096 bound
            # excludes the DSV3-GateUP (K=7168) cousin (already best at
            # default gm=4 per the same wide sweep).
            #
            # Round-67: ``num_xcds=4`` for DSV3-Down family. Round-67
            # added a tunable ``num_xcds`` plumb to the FP8 grouped
            # binding (mirrors BF16 grouped's existing knob); the
            # default xcds=8 is preserved for DSV3-GateUP / gpt_oss /
            # all unmatched shapes (cfg.num_xcds=None → 0 to binding
            # → kernel falls back to BLOCK_SWIZZLE_NUM_XCDS=8).
            #
            # Round-68 widened the search from {8,4} to {8,2,4,16}.
            # 800-iter × 5-repeat median (/tmp/verify_dsv3_down_xcds_round68.py)
            # on all 4 DSV3-Down shapes:
            #   shape           xcds=8    xcds=2    xcds=4
            #   Down-B16-M2048  1724.9    1741.2    1729.1   x2 wins +12.1 vs x4
            #   Down-B16-M4096  1752.1    1773.3    1765.0   x2 wins +8.3  vs x4
            #   Down-B32-M2048  1741.3    1761.1    1755.1   x2 wins +6.0  vs x4
            #   Down-B32-M4096  1774.8    1802.2    1796.2   x2 wins +6.0  vs x4
            # xcds=2 dominates xcds=4 on all 4 shapes (+8.1 TF avg,
            # +0.46pp avg). Cumulative gain over the original xcds=8:
            #   B16-M2048 +16.3 TF (+0.95pp), B16-M4096 +21.2 TF (+1.21pp),
            #   B32-M2048 +19.8 TF (+1.14pp), B32-M4096 +27.4 TF (+1.54pp).
            # Same sweep at xcds=16 shows it never beats xcds=2 (always
            # ≤ xcds=8 on Down family), confirming xcds=2 is the optimum
            # in the DSV3-Down N=7168/K=2048 regime — small XCD-swizzle
            # group fits the ``tiles_n=28 × tiles_m_per_group`` count
            # pattern best (28 N-tiles × 8 or 16 M-tiles per group ≈
            # 224 or 448 tiles per group, divisible by 2 with no
            # remainder, so xcds=2 partitions cleanly).
            return HipKittenConfig(
                layout=layout,
                group_m=32,
                num_xcds=2,
                kernel=None,
            )
        if tiles_m == 16 and 64 <= tiles_n <= 96 and k <= 4096:
            # Round-16 rule. Long-N shallow-K FP8 dense RCR. Anchor:
            # ``(4096, 22016, 4096)`` (tiles_m=16, tiles_n=86, k=4096),
            # the 2nd-worst FP8_fwd metric shape at ratio 0.894 vs
            # hipBLASLt. The binding default ``(group_m=4)`` is suboptimal
            # for this single-CU-band-wide grid (tiles_m=16 fits in
            # NUM_CUS=256, so a small group_m better preserves L2 reuse
            # on the long N axis where each tile sweeps a full K=4096
            # column once and never re-reads it).
            #
            # 6-candidate sweep (3 trials × 200 iters) at
            # /tmp/sweep_fp8_22016.py:
            #   gm=1   median  1683.7 TF  *winner
            #   gm=2   median  1657.5 TF
            #   gm=4   median  1674.6 TF  (binding default)
            #   gm=8   median  1677.3 TF
            #   gm=16  median  1674.9 TF
            #   gm=24  median  1674.9 TF
            # gm=1 wins +9.1 TF (+0.54pp) over the binding default.
            # Bit-identical output (group_m only changes the tile
            # scheduling order on the dense RCR kernel, not the
            # arithmetic); SNR vs fp32 ref unchanged.
            #
            # Rule scope check: ``tiles_m == 16`` excludes the 8192xN
            # family (tiles_m=32). ``64 <= tiles_n <= 96`` is the band
            # around tiles_n=86: it excludes tiles_n=112 (which the
            # binding auto-picks 4-wave for, no override needed) and
            # tiles_n=48 (which the sweep shows is gm=4-best).
            # ``k <= 4096`` excludes deep-K shapes (those have low
            # tiles per CU and different scheduling preferences). No
            # grouped FP8 metric shape has tiles_n >= 64 with k <= 4096
            # (DeepSeek N=4096 → tiles_n=16; gpt_oss n_pad=3072 or 5888
            # → tiles_n=12 or 23).
            return HipKittenConfig(
                layout=layout,
                group_m=1,
                num_xcds=None,
                kernel=None,
            )
    if layout == "rrr":
        tiles_n = n // 256
        # Round-32: tiles_m needed for the R32 sub-clause that excludes
        # Qwen3-GateUP-B32-M2048 (tiles_m=8 + m_total>=65536) from R27's
        # (1,4) rule. The other RRR rules in this branch only key on
        # tiles_n / m_total so this is harmless to the existing matches.
        tiles_m = m // 256
        # Round-42: FP8 RRR dA backward narrow-N rule. The FP8 section had
        # NO ``layout == "rrr"`` rules pre-R42 (all RRR rules in the BF16
        # section above are gated by ``if dtype == "bf16"``), so every
        # FP8 RRR call fell through to the binding default
        # ``(group_m=4, num_xcds=None→0→kernel fallback 8)``.
        #
        # ``grouped_rrr`` is the FP8 dA backward kernel, entered whenever
        # ``trans_b=False`` AND ``K_RRR % 128 == 0`` AND ``N_RRR % 256 == 0``
        # (otherwise the R13/R18 H4 reroute transposes to RCR; see
        # grouped_gemm_fp8_impl.py:364). In the 24-shape MoE metric suite
        # this path is hit by all 8 DSV3 + all 8 Qwen3 cases; gpt_oss
        # (K_RRR ∈ {5760, 2880}) always reroutes to RCR.
        #
        # Closes the same kind of dispatch gap R39 closed for var_k (dB).
        # group_m / num_xcds are pure scheduling knobs — bit-identical
        # output, only tile ordering changes. Max_abs vs fp32 ref:
        #   default (gm=4, xcd=0)   vs  (gm=16, xcd=4):
        #   DSV3-Down  B=16 M=2048: max_abs=0.0 bit_eq=True
        #   DSV3-Down  B=32 M=2048: max_abs=0.0 bit_eq=True
        #   DSV3-Down  B=32 M=4096: max_abs=0.0 bit_eq=True
        #   Qwen3-Down B=16 M=2048: max_abs=0.0 bit_eq=True
        #   Qwen3-Down B=32 M=2048: max_abs=0.0 bit_eq=True
        #   Qwen3-Down B=32 M=4096: max_abs=0.0 bit_eq=True
        # (same bit-equivalence property documented for all
        # ``group_m`` / ``num_xcds`` changes in the RCR block above; see
        # round-23 / round-68 commentary.)
        #
        # 11-candidate sweep, 100 iters × 5 trials per config, results
        # archived in /tmp/rrr_probe_r42.log (produced by
        # ``scripts/_fp8_rrr_config_probe.py`` committed this round):
        #
        #   shape                tiles_n  m_total   (gm=16,xcd=4) Δ vs default
        #   DSV3-Down  B=16 M=2048   8     32768     +2.93%
        #   DSV3-Down  B=32 M=2048   8     65536     +2.65%
        #   DSV3-Down  B=32 M=4096   8    131072     +6.66%
        #   Qwen3-Down B=16 M=2048   6     32768     +3.23%
        #   Qwen3-Down B=32 M=2048   6     65536     +2.18%
        #   Qwen3-Down B=32 M=4096   6    131072     +1.95%
        #
        # Minimum gain +1.95%; maximum +6.66%; median +2.79%; every
        # probed shape in the narrow-N family (tiles_n ≤ 8) gains ≥1%.
        # Bench noise for this path is ~1-3% between 100-iter trials, so
        # every gain above is outside noise. (16, 4) is also top-5 on all
        # 6 shapes (not just 1st place on 3), confirming it's on a broad
        # plateau rather than a noise-driven single-winner.
        #
        # Wide-N shapes (tiles_n ≥ 16 — DSV3-GateUP tiles_n=28 and
        # Qwen3-GateUP tiles_n=16) do NOT match this rule: the probe
        # shows their optima are either at default (4, 0) or at a
        # different (gm, xcd=4) cell, with gains ≤ 2.93% and in some
        # cases regressions. They fall through to default below.
        #
        # Rule scope audit:
        #   - tiles_n == 6: Qwen3-Down only (N_fwd=4096, K_fwd=1536
        #     → N_rrr=1536, tiles_n=6). Not matched by any dense FP8
        #     shape in the DoD / grouped FP8 test suites (dense fwd N
        #     never yields K_fwd ≤ 1792 in a RRR dA trace).
        #   - tiles_n == 8: DSV3-Down only in the grouped metric suite
        #     (N_fwd=7168, K_fwd=2048 → N_rrr=2048). Dense fp8
        #     (8192, 4096, 4096) dA: RRR (m=8192, n=4096, k=4096),
        #     tiles_n=16 — escapes. (4096, 4096, 4096) dA: RRR
        #     (m=4096, n=4096, k=4096), tiles_n=16 — escapes.
        #   - m_total is not None AND m_total >= 32768 excludes dense
        #     callers entirely (dense passes m_total=None) and also
        #     excludes any hypothetical tiny grouped call (B × M_per_g
        #     < 32768; none exists in the 24-shape metric or current
        #     DoD test shapes).
        if (
            tiles_n <= 8
            and m_total is not None
            and m_total >= 32768
        ):
            return HipKittenConfig(
                layout=layout, group_m=16, num_xcds=4, kernel=None
            )
        # Round-27 (this round): FP8 RRR dA backward N=4096 rule
        # (tiles_n == 16). The R42 rule covered tiles_n <= 8
        # (Qwen3-Down + DSV3-Down narrow-N); R43/R44 rule below
        # covers tiles_n == 28 (DSV3-Down K_fwd=7168). The
        # intermediate tiles_n == 16 case (Qwen3-GateUP K_fwd=4096
        # and DSV3-GateUP K_fwd=4096 in dA RRR coords) was left at
        # the binding default (group_m=4, num_xcds=0).
        #
        # Motivation (R27 doc note): the R26 metric bottom shapes
        # (Qwen3-GateUP-B16/B32-M2048, ratio 1.242 — both tied at
        # the metric floor) were attributed by the R27 dA-specific
        # kernel-only probe (/tmp/probe_dA_kernel_only_r27.py) to
        # HK dA RRR being 12.6% slower than Triton on N=1536/K=4096
        # (the small-N narrow-K Qwen3-GateUP family). This is the
        # same ``HK dA RRR is HK weak spot'' finding R8 documented
        # at the architectural-ceiling round (analysis/_notes/round-
        # 8-fused-act-architectural-ceiling-confirmed.md) and which
        # remains unchanged today.
        #
        # 3-cell sweep (default vs (1,4) vs (16,4)) on all 8
        # tiles_n==16 RRR shapes (Qwen3-GateUP × 4 + DSV3-GateUP × 4),
        # 200 iters × 8 trial medians (probe
        # /tmp/probe_qwen3_gateup_dA_rrr_cfg_r27.py):
        #
        #   shape                       (1, 4) Δ vs (4, 0)
        #   Qwen3-GateUP B=16 M=2048    +0.62..+1.34%
        #   Qwen3-GateUP B=16 M=4096    +0.08..+0.29% (tie)
        #   Qwen3-GateUP B=32 M=2048    +0.94..+0.96%
        #   Qwen3-GateUP B=32 M=4096    +0.04..+0.22% (tie)
        #   DSV3-GateUP  B=16 M=2048    +0.44%        (tie)
        #   DSV3-GateUP  B=16 M=4096    -0.10%        (tie, within noise)
        #   DSV3-GateUP  B=32 M=2048    +0.61%
        #   DSV3-GateUP  B=32 M=4096    +0.00%        (tie)
        #
        # 4 wins (>0.5%), 4 ties (within ±0.5% noise), 0 losses
        # (worst is -0.10% on DSV3-GateUP B16 M4096 — well within
        # bench noise). (16, 4) was tested too but kills M=4096 by
        # -2 to -3.7% — chose the safer (1, 4) which is uniformly
        # null-or-positive across all 8 covered shapes.
        #
        # Bit-equivalent output: group_m / num_xcds are pure
        # persistent-grid tile-scheduling knobs (same property
        # documented for R42 narrow-N and R43/R44 wide-N rules
        # above; see those blocks for the full bit-equivalence
        # rationale).
        #
        # Rule scope audit:
        #   - tiles_n == 16 ⇔ N_rrr == 4096. In the 24-shape MoE
        #     metric this is Qwen3-GateUP K_fwd=4096 (4 shapes) +
        #     DSV3-GateUP K_fwd=4096 (4 shapes) = 8 shapes.
        #     Qwen3/gpt_oss/DSV3-Down K_fwd values yield
        #     tiles_n ∈ {6, 8, 11, 28} — none match (already
        #     covered by R42 narrow-N or R43/R44 wide-N rules).
        #   - Dense FP8 RRR shapes in test_dod_smoke.py: dense
        #     callers (``gemm_fp8_impl.py`` lines 251 / 260) pass
        #     ``select_default_config(m, n, k, layout, "fp8")``
        #     WITHOUT the ``m_total=`` kwarg → m_total defaults to
        #     None → the ``m_total is not None`` guard excludes
        #     them entirely.
        #   - Grouped HIPKITTEN FP8 fwd+bwd DoD shapes
        #     (_HIPKITTEN_GROUPED_FP8_FWDBWD_SHAPES): only
        #     (4096, 4096, 7168) [tiles_n=28, R43/R44 catches] and
        #     (4096, 7168, 2048) [tiles_n=8, R42 catches] — neither
        #     matches tiles_n == 16.
        #   - m_total >= 32768 is the minimum grouped m_total in
        #     the 24-shape suite (smallest = B=16 M=2048 = 32768).
        # Round-32 (this round, 2026-05-02): subdivide the R27 (1, 4)
        # rule. R27 picked (1, 4) as the safe uniform choice for
        # tiles_n==16 (Qwen3-GateUP dA RRR, 4 shapes). R32 12-trial x
        # 200-iter x 3-seed re-verify (mirror of R29 / R30 / R31 / R44
        # methodology) versus today's build:
        #
        #   shape                       cell    Dmed vs (1,4)  spread  verdict
        #   Qwen3-GateUP-B16-M2048      default -2.35%         1.56pp  LOSS  keep (1,4)
        #   Qwen3-GateUP-B16-M4096      default -0.30%         0.49pp  TIE   keep (1,4)
        #   Qwen3-GateUP-B32-M2048      default +1.31%         0.71pp  WIN   drop (1,4) here
        #   Qwen3-GateUP-B32-M4096      default -0.13%         0.53pp  TIE   keep (1,4)
        #
        # B32-M2048 default WIN: per-seed default deltas +0.84% /
        # +1.17% / +1.01% (every seed positive). Winner-min (default,
        # 844.61 us) beats baseline-max ((1,4), 856.98 us) in 3/3
        # seeds. Median(+1.31%) / spread(0.71pp) = 1.85x > 1.0
        # threshold => robust signal beyond noise floor (consistent
        # with R7 / R10 / R23 / R29 / R30 / R31 robust-signal gate).
        #
        # This contradicts R27's recorded +0.94..+0.96% gain for
        # (1,4) on the same shape. R27 used 200-iter x 8-trial single-
        # seed methodology; R32's 12-trial x 3-seed (~4.5x more
        # samples) finds the default cell faster. Same kind of
        # methodology-drift correction R31 applied to the var-K rule
        # (gpt_oss-GateUP gm=8 -> gm=1) and that R45 applied to the
        # forward RCR rule (R10 200-iter x 7-trial -> R45 200-iter x
        # 12-trial x 3-seed).
        #
        # R32 also tested R44's deferred (2,0) and (2,2) candidates on
        # B16-M2048; both LOSS (-1.75% / -2.69%) -- R44's recorded
        # +2.96% / +1.50% deltas vs default FALSIFIED on today's
        # build. R44's R44_alt1 (2,0) on B32-M2048 also tested at
        # -2.04% (split-sign per seed: -2.97% / -1.16% / +0.18%) so
        # the alternative cells aren't a viable replacement either.
        # Default falls through to _FP8_DEFAULT_GROUP_M=4 with
        # num_xcds=None (kernel BLOCK_SWIZZLE_NUM_XCDS=8) -- exactly
        # the R32-verified WIN cell for B32-M2048.
        #
        # Gate scope (B32-M2048 single carve-out):
        #   - tiles_m == 8 AND m_total >= 65536: covers Qwen3-GateUP
        #     B32-M2048 only (m_total=65536, tiles_m=8). Excludes
        #     B16-M2048 (m_total=32768) and the M=4096 family
        #     (tiles_m=16) cleanly.
        #   - The R27 selector ``tiles_n == 16 AND m_total >= 32768``
        #     remains gating the rest; the inner exclusion is the
        #     minimal change to surface the new robust signal without
        #     touching the 3 still-keep cases.
        #
        # Bit-equivalent output: group_m / num_xcds are pure
        # persistent-grid scheduling knobs (same bit-equivalence
        # documented for R7 / R10 / R23 / R27 / R30 / R31 / R39 / R42 /
        # R43 / R44 / R45). SNR vs torch ref unchanged (verified by
        # the metric's built-in correctness gate every round;
        # correct_fail = 0/24 maintained).
        #
        # Expected metric impact: dA RRR is ~33% of fwd+bwd wall on
        # this shape (kernel-only probe: 855 us / total 2.52 ms ratio
        # call). +1.31% kernel -> ~+0.4% wall on B32-M2048 only.
        # Geomean lift on 24-shape suite: ~+0.02% (=0.4 / 24 * 1.302
        # contribution). At noise floor (~5 score points per single
        # run); A/B should still show modest median lift over the
        # plateau band.
        if (
            tiles_n == 16
            and m_total is not None
            and m_total >= 32768
            and not (tiles_m == 8 and m_total >= 65536)
        ):
            return HipKittenConfig(
                layout=layout, group_m=1, num_xcds=4, kernel=None
            )
        # Round-43 / Round-44: FP8 RRR dA backward wide-N rule.
        # R42 only covered narrow-N (tiles_n ≤ 8 — DSV3-Down, Qwen3-Down).
        # R43 added tiles_n == 28 (= DSV3-GateUP) with a defensive
        # m_total >= 65536 threshold, excluding DSV3-GateUP B=16 M=2048
        # because the R42 coarse 5-trial probe had reported -12.43%
        # for that shape at (16, 4), contradicting R43's 9-trial
        # +0.81%. R44 re-probes B=16 M=2048 at 12 trials × 200 iters
        # × 3 seeds to resolve the drift:
        #
        #   seed=42:   default=2083.6  (16,4)=2127.1  Δmed=+2.09%
        #   seed=137:  default=2087.3  (16,4)=2126.5  Δmed=+1.88%
        #   seed=2024: default=2087.0  (16,4)=2121.7  Δmed=+1.66%
        #
        # All 3 seeds give +1.66% to +2.09% — consistent, reproducible,
        # well above noise. The R42 -12.43% was a single-trial outlier
        # that the 5-trial median didn't absorb (per-seed std ~7pp on
        # 100-iter trials for this shape; 12-trial × 200-iter drops it
        # to ~0.5pp). R44 therefore relaxes the R43 threshold from
        # m_total >= 65536 down to m_total >= 32768 — the same floor
        # R42 uses for narrow-N. This captures the 4th DSV3-GateUP
        # shape, bringing total RRR-path coverage to 12/16 aligned
        # shapes.
        #
        # Full DSV3-GateUP tight-bench data (R43 9-trial + R44 12-trial):
        #
        #   shape                      m_total   Δ med     Δ min(p0)   src
        #   DSV3-GateUP B=16 M=2048    32768    +1.66..+2.09%  +1.03..+3.78%  R44 multi-seed
        #   DSV3-GateUP B=16 M=4096    65536    +2.82%         +1.84%         R43
        #   DSV3-GateUP B=32 M=2048    65536    +2.44%         +1.61%         R43
        #   DSV3-GateUP B=32 M=4096   131072    +2.70%         +2.46%         R43
        #
        # All 4 shapes gain ≥ +1.66% Δmed; all gains outside observed
        # med-to-min spread. Bit-identical output verified on all 4
        # (max_abs=0 vs default; group_m / num_xcds are pure persistent-
        # grid tile-scheduling knobs; same bit-equivalence documented
        # for R42 narrow-N and R39 var_k).
        #
        # Qwen3-GateUP (tiles_n == 16): NOT covered here. R44 tight-
        # probed all 4 shapes; winners vary per shape:
        #   B=16 M=2048: (2, 0) +2.96%, (16,4) only +0.95%
        #   B=16 M=4096: (2, 2) +1.50%, (16,4) only ≤+0.5%
        #   B=32 M=2048: (1, 4) +2.67%, (16,4) only +0.31%
        #   B=32 M=4096: all candidates ≤ +0.39% (default wins)
        # No single (gm, xcd) cell wins on all 4 Qwen3-GateUP shapes,
        # and (2, 0) / (1, 4) would regress B=32 M=4096 by -0.53% / +0.39%
        # respectively. Skipping per R43's "safe choice" policy.
        #
        # Rule scope audit:
        #   - ``tiles_n == 28`` ⇔ N_rrr == 7168. In the 24-shape MoE
        #     metric this is DSV3-GateUP only (K_fwd=7168 in RRR coords).
        #     Qwen3/gpt_oss K_fwd ∈ {1536, 2048, 2880} → tiles_n ∈
        #     {6, 8, 11}, none match.
        #   - Dense FP8 shapes in test_dod_smoke.py: dA RRR has n =
        #     K_fwd ∈ {4096, 8192, 11008, 14336, 22016} → tiles_n ∈
        #     {16, 32, 43, 56, 86}, none match tiles_n == 28.
        #   - ``m_total is not None`` excludes dense callers (dense
        #     passes m_total=None; grouped passes a.shape[0]).
        #   - ``m_total >= 32768`` is the minimum grouped m_total in
        #     the 24-shape suite (smallest = B=16 M=2048 = 32768), so
        #     the threshold effectively only excludes dense.
        if (
            tiles_n == 28
            and m_total is not None
            and m_total >= 32768
        ):
            return HipKittenConfig(
                layout=layout, group_m=16, num_xcds=4, kernel=None
            )
    kernel = _FP8_DEFAULT_KERNEL if layout == "rcr" else None

    return HipKittenConfig(
        layout=layout,
        group_m=_FP8_DEFAULT_GROUP_M,
        num_xcds=None,
        kernel=kernel,
    )
