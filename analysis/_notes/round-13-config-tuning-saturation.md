# Round 13 — gpt_oss FP8 Config-Tuning Saturation, B=4 Metric/Verify Divergence

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**HEAD before**: `af93b78c`
**HEAD after**: this commit (functional kernel/config bytes unchanged)
**Metric**: 793 mean → 793 mean (no functional change; documentation-only).

## Why this note

Patience counter at 8/30 with no metric improvement; 8 rounds of
config-tuning-roulette have given diminishing returns since round 12 and
this round delivered **zero** even after a fresh wide sweep + tight
verify pipeline that previously caught real wins (round-12 caught a
+1.19pp on Down-B4-M4096 by re-sweeping a "narrow" rule). This note
crystallises **why** further `(group_m, num_xcds)` tuning on the
remaining gpt_oss FP8 shapes is dead, and what the next high-leverage
direction is.

## The B=4 metric / tight-verify divergence pattern

Same structural pattern observed in round-10 (BF16 Down-B4-M2048) and
round-13 (FP8 Down-B4-M2048): the kernel-level tight verify says
``(gm=A, xcd=B)`` wins, but the full-op metric says it loses by 3-5
score points. Concrete data from this round on
``grpFP8-Down-B4-M2048`` (B=4, M=2048, N=2880, K=2880, tiles_n=11,
tiles_m=8, m_total=8192, current rule: round-7 ``(gm=2, xcd=2)``):

### Kernel-only tight verify (1500-iter × 7-repeat p20)
``/tmp/verify_fp8_down_b4_m2048_round13.py`` measures the bare HK
``grouped_dscale_fn`` directly (no quantize, no Python wrapper):

| cfg       | median p20 (TF) | spread (TF, 7 trials) | Δ vs (2, 2) |
|-----------|-----------------|----------------------|--------------|
| (2, 2)    | 1145.43         | 1143.9..1146.6 (2.7)  | baseline    |
| **(1, 1)**| **1159.11**     | **1158.7..1159.5 (0.8)** | **+1.19pp**  |
| (1, 32)   | 1158.71         | 1157.1..1159.5 (2.4)  | +1.16pp     |
| (32, 32)  | 1157.92         | 1157.5..1157.9 (0.4)  | +1.09pp     |
| (1, 8)    | 1155.55         | 1154.8..1156.7 (1.9)  | +0.88pp     |
| (16, 8)   | 1155.17         | 1155.2..1155.6 (0.4)  | +0.85pp     |
| (1, 2)    | 1142.74         | 1141.6..1144.3 (2.7)  | -0.24pp     |
| (1, 4)    | 1109.52         | 1109.2..1110.6 (1.4)  | -3.14pp     |

A clean `gm=1` plateau at xcd ∈ {1, 8, 32} — 0.8-2.4 TF spread is well
below the 12-15 TF Δ vs baseline, statistically dead-obvious win.

### Metric-aligned probe (full op, p20-of-50, isolated 1-shape)
``/tmp/metric_aligned_verify_round13.py`` calls
``turbo.ops.grouped_gemm_fp8`` with `WARMUP=10, ITERS=50` (mirrors
``scripts/_metric_hk_ratio.py::_time_op``), 5 trials. Same shape:

| cfg       | median (TF) | min (TF) | Δ median pp | Δ min pp |
|-----------|-------------|----------|-------------|----------|
| (2, 2)    | 706.31      | 702.81   | baseline    | baseline |
| (1, 1)    | 708.22      | 706.16   | +0.27pp     | +0.48pp  |
| (1, 8)    | 710.60      | 710.30   | +0.61pp     | +1.07pp  |
| (16, 8)   | 710.90      | 708.82   | +0.65pp     | +0.85pp  |

Probe predicts **all** alt cfgs improve at metric level, with `(1, 8)`
and `(16, 8)` strongest at +0.6-1.0pp. **The probe is wrong.**

### Actual full-metric runs (3 runs each)
Running `python3 scripts/_metric_grouped_only.py` with the rule edited
in-tree:

| cfg            | hk_TF (3 runs)    | ratio (3 runs)            | score (3 runs) |
|----------------|-------------------|---------------------------|----------------|
| (2, 2) baseline| 730.3 731.9 734.6 | 0.902 0.899 0.901 (~0.901) | 796 793 794 (mean 794.3) |
| (1, 1) attempt | 696.9 698.2 692.5 | 0.866 0.858 0.856 (~0.860) | 792 790 791 (mean 791.0) |
| (1, 8) attempt | 698.6 694.0 690.1 | 0.842 0.858 0.858 (~0.853) | 791 790 793 (mean 791.3) |

Both kernel-level winners regress -3 to -3.3 score in the **actual**
metric. The metric-aligned isolated-probe over-predicts by ~3-4pp on
B=4 shapes.

### Hypothesis for the divergence

The metric runs all 32 shapes back-to-back (16 BF16 then 16 FP8), each
with HK followed immediately by Triton. The Triton call following the
HK call sees an allocator/cache state that depends on what HK just did.
When HK's tile schedule changes (different gm/xcd shifts the access
pattern through HBM and the inter-call cache footprint), the *Triton*
ratio number on the same shape can move by ±5pp purely from
state-carry-over — not because Triton itself sped up. The 1-shape
isolated probe doesn't reproduce this because there's no Triton call
afterwards consuming the carry-over state.

Concrete evidence in this round's runs: `trt_TF` for Down-B4-M2048 is
- 779.0 in the round-13 baseline (1st run after rebuild)
- 814.5 / 815.3 / 813.7 in 3 reruns at `(gm=2, xcd=2)`
- 829.4 / 808.9 / 804.5 with `(gm=1, xcd=8)`

`trt_TF` swings ±25 TF (~3pp) between back-to-back metric invocations
of the **same** rule. The HK rule change correlates with `trt_TF`
shifts of the same magnitude — a strong sign the variance is in the
HK→Triton transition, not in HK's own kernel.

## Implications

### Config tuning is now truly saturated for gpt_oss

After rounds 7-12 each tightened a specific rule with measurable +0.3
to +1.7 pp wins per shape, the remaining FP8 gpt_oss ratios sit in
**ratio**:

| shape (FP8 gpt_oss only)  | last refined | rule              | ratio (R13 baseline) |
|---------------------------|--------------|-------------------|----------------------|
| GateUP-B4-M2048           | R68          | (gm=2, xcd=4)     | 0.835-0.842          |
| GateUP-B4-M4096           | R7           | (gm=14, xcd=4)    | 0.846-0.851          |
| GateUP-B32-M2048          | R69-R70      | (gm=8, xcd=4)     | 0.886-0.890          |
| GateUP-B32-M4096          | R69-R70      | (gm=8, xcd=4)     | 0.863-0.865          |
| Down-B4-M2048             | R7           | (gm=2, xcd=2)     | 0.899-0.917          |
| Down-B4-M4096             | R12          | (gm=32, xcd=4)    | 0.832-0.833          |
| Down-B32-M2048            | R8           | (gm=16, xcd=4)    | 0.896-0.899          |
| Down-B32-M4096            | (default)    | (gm=4, xcd=8)     | 0.881-0.883          |

Round 13 verified two of these were already at-or-near a kernel-level
plateau and that any nominally-positive cfg change at kernel level is
ate by the metric noise floor. In particular:
- ``GateUP-B4-M2048``: round-13 wide 9×6 sweep top1 was (gm=16, xcd=4)
  +0.17pp on 500-iter, but tight verify reversed it to -0.46pp. Current
  (gm=2, xcd=4) is at top of plateau on tight verify.
- ``Down-B32-M4096``: round-13 wide 9×6 sweep top1 was (gm=4, xcd=32)
  +0.01pp over default; default is essentially top-2.
- ``Down-B4-M2048``: tight verify says (gm=1, xcd=1) +1.19pp, metric
  says -3 score. Documented above.

### Where the remaining 0.85-0.90× gap actually lives

Per the task body's architectural analysis (gpt_oss K=2880, N=2880/5760,
B=4 small-batch ⇒ K-tail + N-tile waste + launch-bound), the gap is in
the **kernel** main loop and epilog, not in the schedule:

1. **K-tail epilog** (each gpt_oss FP8 shape pays an extra ~64-col K
   contraction in epilog because K=2880 ≡ 64 mod 128). The current
   path B fuses K-tail in epilog but still has 1 extra LDS round-trip
   per output tile — **2-4% kernel time per tile**.

2. **N-tile waste at N=2880** (Down family): with BN=256, the last
   N-tile column has only 64 of 256 columns active (75% waste). With
   BN=128 (which HK can be configured to), 22 instead of 11 N-tiles,
   and the last column is 64/128 = 50% used (a 2x improvement on the
   tail tile). Net effect on Down family: **estimated +5-8pp** if the
   register pressure / occupancy works out.

3. **B=4 launch-bound regime** (M_total ∈ {8K, 16K} → 1-2 waves of
   compute — total compute ~70-140 µs vs ~10-20 µs launch overhead).
   No amount of `gm/xcd` tuning fixes this; need either kernel-side
   amortization (M-dim multi-tile in K-tail epilog) or a different
   launch strategy (still single-launch, but reduce per-tile setup).

### The next round's lever is kernel-level, not config

Recommendation for round 14+:
- **Stop the 9×6 sweep + tight verify pipeline** for FP8 gpt_oss
  rules. The remaining wins per attempt are < kernel-level wins
  available, and 50% of attempts lose to metric noise (round-13
  pattern). Each "tuning" round has zero expected score delta now.
- **Pivot to one of**:
  - (A) **BN=128 N-tile** for Down family (N=2880). Verify whether HK
    config / kernel template supports BN=128 cleanly. If yes,
    minimal-edit experiment: hard-code BN=128 for `tiles_n_at_BN256
    == 11`, measure all 4 Down shapes. If no (would require kernel
    rewrite), defer.
  - (B) **K-tail epilog one-pass merge**: HK FP8 K-tail epilog
    currently has separate accumulator+store; merge so the K-tail
    contraction directly accumulates into the post-scale C buffer
    (saves 1 LDS round-trip per tile, ~2pp). Requires kernel edit.
  - (C) **Skip A-tile LDS staging** (mentioned in round-11 task body):
    direct HBM→register for A in the BN=256 path. Higher risk but
    higher ceiling on B=4 launch-bound shapes.
- **Build a metric-aligned timing methodology before any further
  config attempts**. The current options (kernel-only tight verify,
  isolated-shape metric-aligned probe) both fail on B=4 small-batch
  shapes by ±3-5pp. The only reliable signal is running the full
  ``scripts/_metric_grouped_only.py`` 3+ times per candidate; a
  9×6 sweep then needs 162 metric runs (~22 minutes per config-tuning
  round) — not worth the +0..+1 expected score delta vs kernel work.

## Methodology lesson (for future rounds)

For **B=4 small-batch shapes** (M_total ∈ {8192, 16384}, ≤2 waves of
work), `(group_m, num_xcds)` config decisions **must** be validated
via ≥3 full ``scripts/_metric_grouped_only.py`` runs at each candidate
versus baseline. Tight verify and metric-aligned isolated probes have
**both** failed to predict the score impact within ±3pp, and the score
is the only ground truth. For B≥16 large-batch shapes the isolated
probe is reliable because the cross-shape state-carry-over gets
washed out by the longer-running kernel.

## Bit-equality verification

No functional kernel/config bytes changed in this round (the round-13
attempts on (gm=1, xcd=1) and (gm=1, xcd=8) were both reverted in the
same session). The only change in this commit is the round-13 inline
note in ``primus_turbo/pytorch/kernels/hipkitten/config.py`` (after the
existing round-7 rule comment) plus this notes file. Metric still
reads ~793 mean.

## Why this note is committed (not just /tmp)

The 793 plateau has held for 8 rounds. Without an explicit "stop config
tuning, pivot to kernel work" artifact in the tree, the next round's
agent (who only has the chat history of the most recent few rounds)
will repeat the same wide-sweep → tight-verify → revert cycle,
burning more rounds on the same dead-end. This note's purpose is to
tell that future agent: **the lever is kernel work; the config knobs
are saturated**.
