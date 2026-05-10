# Round 63 — FP8 gpt_oss saturation reaffirmed (693 at lower band edge)

**Date**: 2026-05-10
**Run**: gpt_oss_fp8_local_20260509_143917, round 63 / 100
**HEAD (Primus-Turbo)**: f5e67680 (R62, doc-only since R55 functional baseline)
**HEAD (HipKittens)**: 49ffb984 (R17 infra — caller-alloc workspace, NEUTRAL on this metric)
**Verdict**: NEUTRAL — no in-scope lever, no functional change

## Recap of the saturation regime

| Round | Metric | Δ vs R56 noise median (695) | Within σ=2.27? |
|---|---|---|---|
| R57 | 697 | +2 | yes |
| R58 | 696 | +1 | yes |
| R59 | 696 | +1 | yes |
| R60 | 696 | +1 | yes |
| R61 | 695 |  0 | yes |
| R62 | 693 | -2 | yes (edge of -σ ≈ 692.7) |

Six consecutive rounds inside the R56-characterized 30-sample noise band
(median 695, σ 2.27, no tail). Mean of R57-R62 = 695.5, exactly the R56
median. Spread 4 = 1.76σ — entirely consistent with i.i.d. sampling of
the noise distribution on a kernel surface that has not changed since
R55.

The R62 print of 693 is the lower edge of the band (median - σ).
Two interpretations, both compatible with continued saturation:

  1. **Pure noise**: 693 is at the 16th percentile of N(695, 2.27²).
     Hitting that 1× in 6 rounds is expected (binomial p=0.16 → expect
     ~1 in 6).

  2. **Slow GPU-temperature drift**: GPU 3 may have drifted slightly
     warmer over the 6-round wallclock (~30 min). Past sessions have
     shown ~3-5 score sensitivity to GPU thermal state. This is below
     the ship threshold and orthogonal to any in-repo lever.

Either interpretation supports NEUTRAL.

## Why no in-scope action

R60-R61 closed the SKILL.md NEW DIRECTIONS A-G inventory:

| Direction | Closure | Round |
|---|---|---|
| A1 Stream-K | preflight FALSIFIED, then full impl shipped NEUTRAL | R52, HK R13a-R17 |
| A2 SplitK var-K wgrad | falsified | R33 |
| A3 decoupled-warps PC | preflight FALSIFIED + R60 audit (no PC prototype < 128x256) | R54, R60 |
| B cross-stream | blocked by metric semantics (metric serializes sections) | n/a |
| C act cache | blocked by metric semantics (metric pre-quantizes inputs) | n/a |
| D1 SALU coord-decode | SHIPPED neutral (b3a5c8db, HK side) | pre-R55 |
| D2 magic-number divide | sub-noise budget A-PRIORI FALSIFIED | R59 |
| E barrier scheme | falsified (drop-2/drop-4 catastrophic, drop-redundant 0 lift) | R26-R28 |
| F larger tiles | falsified (256x384 / 512x256 register-budget infeasible) | R32 |
| G cross-shape co-opt | falsified (dispatcher already per-shape predicate granularity) | R55 |

FORBIDDEN PATHS in `_task_gpt_oss_fp8_kernel.md` covers the macro-flag,
4-wave port, 256x128 asymmetric, small-tile-4w, dispatcher (gm/xcds/
slots/cs) on Down-B4, and quant-cache axes with multi-sample evidence.

There is **no untried in-scope lever** as of R63. The 204-score gap to
TARGET=900 is structural — out-of-scope of this daemon's round budget.

## Operator recommendation (unchanged from R62)

Terminate the daemon at R63. 37 more rounds at the current per-round
cost will produce 37 more in-band noise prints. Closing the gap to
TARGET=900 requires expanding daemon scope:

  - re-scope the metric (e.g. allow new kernel templates measured in a
    different harness),
  - hardware/firmware lever access (MI355X firmware tuning, XCD
    interconnect topology, etc.),
  - or a multi-week kernel-template research project (PC variant at
    256x256+, Stream-K with different work-stealing strategy, custom
    barrier hierarchy).

None of these can be reached inside the per-round budget of this
daemon configuration.

## Self-check vs prior NEUTRAL rounds

This is the 6th consecutive NEUTRAL round (R57 through R63). To avoid
becoming a self-perpetuating ritual: the falsification gates I checked
this round before deciding NEUTRAL:

  1. **Did any new lever appear in SKILL.md or task md since R62?** No
     (verified by reading both files this round).
  2. **Did the 30-sample noise model from R56 still fit R57-R62?** Yes
     (mean 695.5 ≈ median 695, max-deviation 1.76σ, no outliers).
  3. **Did HK HEAD change since R55 functional baseline?** Yes — R13a
     through R17 added Stream-K infra (pybind kwargs, host-side
     allocator). All shipped NEUTRAL on this metric (metric does not
     exercise Stream-K path; dispatcher rule does not select it).
  4. **Did any of the 8 shapes regress correctness?** Cannot directly
     verify without running metric (which the daemon does post-round);
     R57-R62 prints were all in valid score range (not 0), so SNR>25dB
     gate passed for all 8 shapes across all rounds. No regression
     signal.

All four gates pass. NEUTRAL is the honest verdict.

## What would change the verdict (forward pointer)

A non-NEUTRAL round becomes possible only if one of the following
out-of-scope events occurs:

  - SKILL.md or task md is updated with a new lever class,
  - a new HK kernel template lands on the prototype shelf with
    bit-equivalence on gpt_oss_20B's 8 shapes,
  - the metric script is re-scoped (FROZEN list change),
  - GPU pool changes (e.g. MI355X firmware update affecting MFMA
    throughput).

Until then: NEUTRAL doc commits per round, with the operator-facing
recommendation to terminate.
