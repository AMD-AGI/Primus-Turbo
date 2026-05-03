# Round 29 — BF16 grouped-GEMM 5-lever aggregate FALSIFIED (paired 5-run mean: -0.2, net noise-bound)

## Goal

R28 found a clean `bf16_transpose_3d` block-shape lever
`(BK, BN)=(128, 128)` for K==N with uniform-positive kernel-level
gain (+8.04 % B=4 Down, +1.08 % B=32 Down) and a paired 5-run mean
of **+3.4** — sub-+5 alone. R28 left three explicit landing options:

  * (A) Land R28 transpose as a "kernel-correctness improvement"
    commit (bit-equal, no-regression) accepting sub-+5 metric move.
  * (B) Bundle R28 transpose with the R26 3-rule dispatch aggregate
    (+0.6 measured) and R27's DSV3-GateUP dB var-K rule (+0.06
    expected): estimated +4.1 measured, still sub-+5 per R27 note.
  * (C) Find one more uniform-positive structural lever and add to
    the bundle to cross +5.

R29 executes **Option B** — a 5-lever aggregate intended to test
whether stacking every remaining allclose-safe positive lever
measured in R24-R28 crosses the +5 threshold in a single paired
5-run.

## The 5 levers bundled

| # | lever | scope | change | measured Δ (per-lever, single-probe) |
|---|---|---|---|---|
| 1 | R28 `bf16_transpose_3d` | K == N (BF16) | (BK, BN) (256, 128) → (128, 128) | +8.04 % B=4 / +1.08 % B=32 on Down transpose µs |
| 2 | R26 DSV3-Down fwd RCR | `tiles_n==28 ∧ 8≤tiles_m≤16 ∧ k≤4096` | `(gm=16, xcds=2)` → `(gm=16, xcds=4)` | avg +0.49 % uniform-positive |
| 3 | R26 gpt_oss-GateUP dB var-K | `tiles_n==11 ∧ 8≤tiles_m≤24 ∧ k≤4096` (post R24 split) | `(gm=4, xcds=4)` → `(gm=1, xcds=4)` | avg +0.98 % uniform-positive |
| 4 | R26 Qwen3-GateUP fwd RCR (NEW) | `tiles_n==12 ∧ k==4096 ∧ m_total!=None` | default `(gm=4, xcds=8)` → `(gm=1, xcds=4)` | avg +1.71 % uniform-positive |
| 5 | R27 DSV3-GateUP dB var-K (NEW) | `tiles_m==16 ∧ tiles_n==28 ∧ k≤4096 ∧ m_total!=None` | default `(gm=4, xcds=8)` → `(gm=2, xcds=8)` | avg +0.25 % uniform-positive, allclose-safe (max_abs 1.79-2.00 bf16) |

All 5 levers were independently bit-equivalent or within bf16 allclose
tolerance. Per-lever probe data comes from R28 note, R26 note (3-rule
aggregate falsification), and R27 note (extended var-K probe).

## Correctness verification

Downsized allclose check on 9 canonical shapes (one per family-config
combo, covering all 5 levers' scope):

```
PASS  DSV3-GateUP-B16-M2048              (R27 rule 5 activates)
PASS  DSV3-Down-B16-M2048                (R26 rule 2)
PASS  gpt_oss-GateUP-B4-M2048            (R26 rule 3 + R28 rule 1)
PASS  gpt_oss-GateUP-B32-M2048           (R26 rule 3 + R28 rule 1)
PASS  gpt_oss-Down-B4-M2048              (R28 rule 1)
PASS  gpt_oss-Down-B32-M2048             (R28 rule 1)
PASS  Qwen3-GateUP-B16-M2048             (R26 rule 4)
PASS  Qwen3-GateUP-B32-M4096             (R26 rule 4)
PASS  Qwen3-Down-B16-M2048               (no-op, control)
```

Full 24-shape suite: correct_fail=0/24 on both applied and baseline
runs of the paired 5-run.

## Metric verification — paired 5-run mean

Procedure: git-stash the aggregate, run 5 baseline, `git stash pop`,
run 5 applied. Metric = `scripts/_metric_grouped_bf16_weighted_wall.py`.

```
baseline:  885, 878, 888, 884, 883    →  mean 883.6  (range 878-888, σ ~3.6)
applied:   880, 885, 885, 886, 881    →  mean 883.4  (range 880-886, σ ~2.7)
```

**Δ = -0.2** (applied - baseline). Well within the 5-run noise σ ~3,
and of the **wrong sign**. Every first-run single-shot gave
misleading numbers: baseline-run-1 = 885, applied-run-1 = 880
(-5), but by run 3 they both settled around 884-886.

## Why the 5-lever aggregate failed

R27 note predicted the 4-rule dispatch aggregate (R26 + R27) at
+1.4 measured at best. R28 transpose alone measured +3.4. The paper
prediction was +3.4 + +1.4 = **+4.8 estimate**, already borderline.

The paired 5-run measured -0.2 — a ~5.0 shortfall vs the paper
estimate. Likely causes (can't disambiguate without per-lever
isolation which this round has no budget for):

  1. **R28 transpose gain is seasonal.** R28 used a different GPU
     session / warm-up order than R29. Cross-session noise on the
     `bf16_transpose_3d` kernel µs measurements can be ±1-2 µs; on
     gpt_oss-Down B=4 (where the transpose is 26 µs total wall),
     that's ±5-8 % per-run variability. The +3.4 R28 observed was
     real-at-the-time but not robust to session changes.
  2. **Dispatch rule composition is non-linear.** R26 3-rule
     aggregate measured +0.6 total (vs +1.1 sum of averages). R27
     predicted 4-rule at +1.4. Stacking 5 rules may actually net
     negative if any pair cross-interferes — e.g. the R26 Qwen3-GateUP
     fwd RCR rule might change B-tile load timing in a way that
     partly negates the transpose kernel's gain on gpt_oss-Down B=4.
  3. **Ceiling effect near 885.** The metric has been hovering at
     880-891 for 22 rounds (R23-R29). It's plausible we're at the
     local maximum for the current (dispatch rules × kernel code)
     configuration and any future dispatch-only tweaks will net to
     zero against per-run noise.

## Decision

**Revert. Documentation-only commit.** The 5-lever aggregate is
FALSIFIED — combined paired 5-run mean is -0.2, well below the +5
land threshold and of the wrong sign. Archive the bundle recipe
for future use.

This is now the **third consecutive round** attempting
dispatch-rule aggregation (R24 LANDED +5.4 4-rule; R26 FALSIFIED
3-rule +0.6; R27 closed allclose gap on 5th rule; R29 FALSIFIED
5-rule -0.2). The pattern is clear: **dispatch-rule surface is
exhausted**. Any further score movement on the BF16 24-shape
wall must come from kernel code (HipKittens .cpp) or from an
entirely new algorithmic path (e.g. H4 transpose fusion into the
kernel B-load, mentioned in R27 worst-case fallback).

## Files

* `analysis/_notes/round-29-bf16-grouped-5-lever-aggregate-FALSIFIED-net-noise-bound.md` — this note.
* `/tmp/r29_paired_5run.sh`, `/tmp/probe_round29_correctness.py`
  — probe artefacts (not committed).
* No production change (config.py and fp8_transpose.py reverted
  to HEAD 436bdc3).

## Suggested R30 next step

* **Stop attempting dispatch aggregates.** 5 consecutive rounds
  (R24 LANDED, R25-R27 FALSIFIED, R29 FALSIFIED) have now
  characterized this surface. The remaining dispatch-only
  headroom is in the noise.
* **Main line**: resume the R28 alternate — attempt to fuse the
  H4 `bf16_transpose_3d` kernel into the HK kernel's B-load
  path. Spec: modify
  `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  to accept an optional on-the-fly transpose during the first LDS
  stage of B-tile loading when the caller passes `trans_b=False`
  and the dimensions are misaligned (the case the Primus-Turbo
  H4 guard handles today by materializing a `[B, N, K]` tensor).
  On gpt_oss-Down B=4 (~5 % of wall × 3 weight = +0.6 score),
  gpt_oss-GateUP B=4 (~10 % × 3 = +1.5), and gpt_oss-Down B=32
  (~7 % × 3 = +1.5), the headroom is ~+3.5-4.0 score if fusion
  can eliminate the entire transpose pass. Not enough alone to
  cross +5 but opens a new surface.
* **Alt main line**: K-tail kernel structural probe (R9-R16's
  territory) — specifically the `KI=0 + unroll-2` inner loop
  at lines 845-1060 of `kernel_bf16_dynamic.cpp`. R16 landed
  at the local optimum here; R30 would need to explore a
  different inner-K scheduling paradigm (e.g. async copy-based
  prefetch with WMB primitives, vs the current LDS double-
  buffer path). High risk, high variance, high potential gain.
* **Last resort**: accept score plateau ~885 and pivot to DoD
  regression reduction — the DoD smoke has failed=608 last run,
  which is orthogonal to the wall metric but affects cross-
  backend confidence.
