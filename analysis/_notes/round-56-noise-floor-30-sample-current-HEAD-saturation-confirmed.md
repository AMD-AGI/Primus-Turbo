round-56-noise-floor-30-sample-current-HEAD-saturation-confirmed.md
=============================================================================

Round: 56 / 100
Date: 2026-05-10
Pre-SHA: aa587ddc (R55 docs — Direction G A-PRIORI FALSIFIED, R56 forward-pointer = noise re-characterization)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R55 forward-pointed to R56 = "re-characterize the noise floor with 30
samples on the current HEAD on GPU 3 (the daemon's pin) to set an
honest wind-down threshold". Executed: 30 consecutive
`_metric_gpt_oss_fp8_kernel.py` samples on remote MI355X, GPU 3,
HEAD = `aa587ddc` (R55, no code change since `743599f` R43 perf ship).

```
samples (chronological): 696 693 696 692 694 691 695 697 692 695
                         691 695 690 696 697 690 696 696 698 696
                         693 694 694 696 695 694 692 697 691 697

sorted ascending:        690 690 691 691 691 692 692 692 693 693
                         694 694 694 694 695 695 695 695 696 696
                         696 696 696 696 696 697 697 697 697 698

n=30  min=690  max=698  range=8
median=695   mean=694.3   σ≈2.27
no tail-mode samples (R29 had 2/23 = 9% in tail)
```

**All 30 samples landed in a tight 9-point cluster (690–698)** with no
tail-mode contamination. This is the cleanest noise distribution we have
on this metric, on this hardware, on this GPU. The current SHA's true
score is **695 ± 2.3** — there is no hidden +5 or +10 lift waiting to be
unlocked by a luckier sample.

## How this compares to the R29 baseline

R29 characterized noise on the round-28 HEAD (SHA `80cf0b69`) on the
same GPU 3 with 23 samples:

| Metric           | R29 (80cf0b69)        | R56 (aa587ddc)        | Δ                  |
|------------------|-----------------------|-----------------------|--------------------|
| Samples          | 23                    | 30                    | +7                 |
| Cluster range    | [690, 705]            | [690, 698]            | upper -7           |
| Cluster median   | 698-700               | 695                   | **-3 to -5**       |
| σ (cluster)      | ~3-4                  | 2.27                  | tighter            |
| Tail-mode rate   | 2/23 = 9%             | 0/30 = 0%             | cleaner GPU state  |

The current HEAD measures **3-5 score lower** than R29's baseline (with
no intervening kernel/dispatcher/macro change between the two SHAs that
should affect score: rounds 29-55 are entirely docs/NEUTRAL except R43
which shipped `GateUP-B4-M2048 dgrad-via-H4 num_xcds None→4`, a +1-2 %
local win on a single section's H4 path). The 3-5 score downshift is
plausibly attributable to:

1. **R29's upper tail (703, 705) is real but rare.** R29 had n=23 with
   max=705; R56 has n=30 with max=698. Resampling the same code on
   different days changes the upper tail by ~5 score (consistent with
   day-to-day GPU thermal / clock-slip variation across the docker
   container's host).
2. **Daemon's measured scores 691-696 over R51-R55 fit R56's
   distribution exactly** (cluster 690-698, median 695). The "best so
   far = 696" daemon record is the **single-sample 60th-percentile** of
   the R56 cluster — i.e., ordinary, not a real optimum.
3. **R29-vintage high samples (703, 705)** would today read as
   tail-upward outliers ~3.5σ above R56 median (695 + 3.5×2.27 ≈ 703).
   Those samples reflect the GPU being in a slightly cooler / less-
   contended state at R29 measurement time, NOT a genuine code lift
   between the two SHAs.

**Conclusion**: there is no hidden upper-cluster regime above 700 that
the daemon could re-discover by waiting for a "lucky" sample. The
distribution is what it is.

## Updated wind-down thresholds

Using R56's 30-sample distribution (median 695, σ 2.27):

| Decision boundary               | Old (R29-based)             | New (R56-based)              |
|---------------------------------|-----------------------------|------------------------------|
| Single daemon sample = "improved" if score ≥ X | 715 (R29 median + 3σ)        | **702** (R56 median + 3σ)    |
| Cluster median (5-sample probe) ≥ Y for ship   | 710 (R29 + 2σ)               | **700** (R56 median + 2.3σ)  |
| FALSIFIED if 5-sample median ≤ Z              | 685                          | **688** (R56 median - 3σ)    |
| Detectable single-sample lift (3σ above median) | +12-15                      | **+7**                       |

The detectability floor has tightened (σ shrank from ~3.5 to 2.3) but
the cluster-median ceiling is **lower** than we previously believed
(695 vs 698-700). This is consistent: a tighter, lower distribution
means past "best" samples were single-shot upper-tail draws from a
distribution centered around today's daemon median.

## What this means for the daemon's stop policy

The daemon is at:
- Streak = 14 rounds without `improved=True` (best 696 since R0 baseline 692)
- Patience = 40 rounds (so 26 rounds remaining before automatic stop)
- All 7 task-md NEW DIRECTIONS exhausted at preflight or implementation
  (A1 Stream-K — wave-pack PMC closes 1/8 cells; A3 decoupled-warps —
   prior PC paper data shows -17 to -44 % loss across 6 sizes; B —
   stream parallelism doesn't help kernel-only metric; C — KREM=64 already
   shipped on rcr / no K-tail block on var-K; D SALU coord-decode —
   would require var-K kernel rewrite, multi-round, R21 etiology open;
   E barrier-scheme — R26-R28 falsified single-barrier-drops, deeper
   restructure is multi-round; F — N/A; G cross-shape co-opt —
   predicates already at single-shape granularity)
- Macro-level levers exhausted (R22-R28 falsified VARK macros;
  R31b RCR_KTAIL_VMCNT default=8 = sweet spot; R4 vmcnt=16 metric-
  tied)
- Dispatcher per-shape predicates audited at single-shape granularity
  (R23, R7, R3, R8, R12, R50, R70, R23, R44, R45, R47 — all 8 metric
  shapes have explicit per-shape rules; per-shape sweeps exhausted)

A 26-round budget remaining at this saturated state would burn
~26×3 minutes = ~80 minutes of remote GPU time generating noise data.
**Recommended: trip an early wind-down**, either by:

1. Submitting a `kill_round = 56` or equivalent to the auto_optimize
   daemon (if such a control exists), OR
2. Letting the streak grow to patience=40 naturally (≈26 more docs
   commits), OR
3. Pivoting the daemon's task to a different metric (e.g.
   `_task_fp8_fused_act.md` Path-A track which has dedicated kernel
   work still open per the task list).

## What was changed this round

* No code changes shipped (HK working tree unchanged, PT working tree
  unchanged, no probe scripts added).
* This docs note added: empirical 30-sample noise floor characterization
  + updated wind-down thresholds.

## Files touched

* `analysis/_notes/round-56-noise-floor-30-sample-current-HEAD-saturation-confirmed.md`

## R57+ forward-pointer

Two options if the daemon does not stop:

1. **Multi-task daemon pivot** (highest EV): switch the task pointer to
   `_task_fp8_fused_act.md` (Path-A fused activation). That track has
   open kernel-template work and a different score landscape, so it
   resets the saturation streak with real headroom. Requires daemon
   config change (out of in-round scope) but is the obvious move when
   one task saturates.
2. **Permanent FALSIFIED-only docs streak** (lowest EV): keep emitting
   neutral docs commits with the saturation citation until streak hits
   patience=40. Lowest-information path. The score will continue to
   read in the 690-698 noise band, never crossing the new +7 single-
   sample detectability threshold (which would require a real +10
   median lift, and no candidate lever exists per R55).

R57 default (if daemon continues): docs-only, citing this note as
saturation evidence, no probe.

## Decision: NEUTRAL

R56 ships a docs note only. Daemon metric on the next sample expected
to land in [690, 698] per the distribution above, with median 695.
