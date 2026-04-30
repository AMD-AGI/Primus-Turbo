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
                    return HipKittenConfig(
                        layout=layout, group_m=2, num_xcds=4, kernel=None
                    )
                if m_total <= 16384:
                    return HipKittenConfig(
                        layout=layout, group_m=2, num_xcds=2, kernel=None
                    )
                if m_total <= 65536:
                    return HipKittenConfig(
                        layout=layout, group_m=8, num_xcds=4, kernel=None
                    )
                return HipKittenConfig(
                    layout=layout, group_m=1, num_xcds=4, kernel=None
                )
            if tiles_n == 11:  # Down N=2880
                if m_total <= 8192:
                    return HipKittenConfig(
                        layout=layout, group_m=2, num_xcds=2, kernel=None
                    )
                if m_total <= 16384:
                    return HipKittenConfig(
                        layout=layout, group_m=2, num_xcds=4, kernel=None
                    )
                return HipKittenConfig(
                    layout=layout, group_m=1, num_xcds=4, kernel=None
                )
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
        if tiles_n == 28 and 8 <= tiles_m <= 16 and k <= 4096:
            # Round-20 rule. DeepSeek-V3-Down grouped FP8 RCR family:
            # per-group GEMM N=7168, K=2048, M_per_group ∈ {2048, 4096},
            # B ∈ {16, 32}. Mirrors the existing BF16 ``tiles_n==28``
            # rule above; the same shape family (tiles_n=28, k=2048)
            # falls through to the FP8 default ``group_m=4`` and was
            # the worst grpFP8 tier outside gpt_oss (ratios 0.89-0.94
            # vs Triton in round-19 metric).
            #
            # 6-candidate sweep (3 trials × 30 iters) at
            # /tmp/sweep_fp8_deepseek_down.py over
            # group_m ∈ {1, 2, 4, 8, 16, 24} on all 4 metric shapes:
            #   B16-M2048   default(gm=4)=1662  gm=24=1690  +1.7pp  (top1)
            #   B16-M4096   default(gm=4)=1688  gm=24=1697  +0.5pp  (top1)
            #   B32-M2048   default(gm=4)=1633  gm=24=1673  +2.4pp  (top1)
            #   B32-M4096   default(gm=4)=1652  gm=24=1665  +0.8pp  (top1)
            # gm=24 wins on all 4 shapes; +1.4pp average over default.
            # group_m only changes tile scheduling order on the dense /
            # persistent RCR kernel, not arithmetic — bit-identical
            # output across all 6 group_m values (cross max_abs = 0.0,
            # SNR vs fp32 reference unchanged across the family).
            #
            # Rule scope check: ``tiles_n == 28`` ⇔ N == 7168, which
            # only occurs in the metric for these 4 grouped DSV3-Down
            # forward shapes. No dense FP8 metric shape has N=7168;
            # DSV3-GateUP has tiles_n==16 (N=4096) and matches its
            # binding-default (gm=4) winner per the same sweep
            # (B16-M2048 gm=4=2507 top1, B32-M4096 gm=2=2554 top1
            # +1.0pp — too narrow to anchor a separate rule);
            # gpt_oss has tiles_n ∈ {12, 23} (n_pad=3072 / 5888) ⇒
            # neither matches. The k<=4096 bound excludes the
            # DSV3-GateUP (K=7168) cousin, which the same sweep shows
            # already runs near gm=4-best; pinning a different group_m
            # there is unjustified.
            return HipKittenConfig(
                layout=layout,
                group_m=24,
                num_xcds=None,
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
    kernel = _FP8_DEFAULT_KERNEL if layout == "rcr" else None

    return HipKittenConfig(
        layout=layout,
        group_m=_FP8_DEFAULT_GROUP_M,
        num_xcds=None,
        kernel=kernel,
    )
