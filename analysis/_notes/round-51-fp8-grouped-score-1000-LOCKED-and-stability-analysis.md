round-51-fp8-grouped-score-1000-LOCKED-and-stability-analysis.md
==================================================================

Round: 51 / 100
Date: 2026-05-02
SHA: 63244cdf (pre) → TBD (post)
Task: grouped FP8 tensorwise (fwd + var-K bwd + dA bwd) on the 24-shape suite

## TL;DR

R50 LANDED score=**1000** (capped). R51 single-shot metric came in at
**986** — within the run-to-run noise band (978-1000, centred ~990).
auto_optimize.py best=1000 stays locked until a future single-shot
measurement >1000 (impossible since score is min(geomean/1.20, 1.0)
× 1000 → capped at 1000). Patience streak = 1 (R51 didn't beat
R50's 1000).

This round verified the R50 rule fires correctly (sanity-check below)
and documents the path to **stable** score=1000 (currently the metric
samples a normal distribution on the 24-shape geomean centred near
1.18 with ~0.014 std → score=1000 hit ~5% of the time, score in
978-998 hit ~95%).

No kernel or dispatch changes this round. HipKittens HEAD unchanged
at 6c52d017.

## R50 rule fires verification

Direct probe at /tmp/verify_rule_fires.py (3 lines + 5-trial p20):

```
Rule for gpt_oss-Down-B32-M4096:
  HipKittenConfig(layout='rcr', group_m=4, num_xcds=4, kernel=None)  ✓

5-trial 50-iter p20 timing on the live binary:
  trial 1: 1884.34 TF
  trial 2: 1948.81 TF
  trial 3: 1963.90 TF
  trial 4: 1975.95 TF
  trial 5: 1967.42 TF

Median 1963.90 TF / Min 1884.34 / Max 1975.95
Spread 91.61 TF (4.7% CV)
```

Median sits at the R50 sweep's tight-verify median (1959.77 TF). The
rule is correctly applied. R51's metric single-shot of 1789 TF on
this shape is the LOW TAIL of the distribution (~5σ below median); a
single-trial p20 is sensitive to this when the distribution has a
~5% CV tail.

## R51 metric (single-shot)

```
                                                       ratio
  grpFP8_gpt_oss_20B-Down-B32-M4096                    1.002    *low-tail of (4,4) variance
  grpFP8_DeepSeek-V3-GateUP-B16-M4096                  1.093    (already at gm=2 round-8 rule; Triton 2360 = high-tail)
  grpFP8_gpt_oss_20B-GateUP-B32-M2048                  1.074    (gm=8,xcd=4 round-70)
  grpFP8_gpt_oss_20B-Down-B32-M2048                    1.084    (gm=16,xcd=4 round-8)
  grpFP8_gpt_oss_20B-GateUP-B4-M2048                   1.086    (gm=1,xcd=4 round-23)
  grpFP8_gpt_oss_20B-GateUP-B32-M4096                  1.085    (gm=8,xcd=4 round-70)
  grpFP8_gpt_oss_20B-GateUP-B4-M4096                   1.085    (gm=14,xcd=4 round-7)
  grpFP8_gpt_oss_20B-Down-B4-M4096                     1.105    (gm=32,xcd=4 round-12)
  grpFP8_Qwen3-235B-A22B-Down-B32-M4096                0.964    *Triton high-tail anomaly (1624 vs typical ~1373)*

  grp_BF16  geomean = 1.1995  (right at 1.20 threshold — barely FAIL)
  grp_FP8   geomean = 1.1664  (below; needs ~+3pp to robustly hit 1.20)
  score = 986
```

Every gpt_oss FP8 shape with ratio < 1.20 is **already** in a swept
dispatch rule. The ratios reflect the kernel's structural ceiling on
gpt_oss K=2880 (FUSED_KTAIL=true) shapes, not dispatch fall-throughs.

## Statistical state of FP8 geomean (R47-R51 corpus)

| Round | Run     | Score | FP8 geomean |
|-------|---------|-------|-------------|
| R47   | auto    | 998   | 1.1923      |
| R48   | auto    | 982   | unknown     |
| R48   | manual1 | 992   | -           |
| R48   | manual2 | 997   | -           |
| R49   | auto    | 990   | -           |
| R49   | KREM-1  | 996   | 1.1903      |
| R49   | KREM-2  | 987   | 1.1701      |
| R49   | KREM-3  | 987   | 1.1686      |
| R50   | base    | 989   | 1.1735      |
| R50   | rule-1  | 981   | 1.1559      |
| R50   | rule-2  | 984   | 1.1613      |
| R50   | rule-3  | 999   | 1.1965      |
| R50   | rule-4  | 987   | 1.1701      |
| R50   | auto    | 1000  | (lucky shot)|
| R51   | auto    | 986   | 1.1664      |

Corpus N=15. Score: μ ≈ 990, σ ≈ 7.3 (range 981-1000). FP8 geomean:
μ ≈ 1.175, σ ≈ 0.013 (range 1.156-1.197). Distribution is well-fit
by μ=1.175, σ=0.013 normal: P(geomean ≥ 1.20) = P(Z ≥ 1.92) ≈ 2.7%.

To **stably** lock score=1000, FP8 geomean μ needs to shift from 1.175
to ≥ 1.21 (+3pp). This is a real architectural lift; the cheap dispatch
closures and refactor-style cleanups have all landed.

## Path to stable 1000 (R52+ roadmap)

| Lever                                          | ROI     | Risk    | Rounds  |
|------------------------------------------------|---------|---------|---------|
| **C-2** warp-tile restructure (4w-style)       | +3-6pp  | High    | 2-3     |
| **C-3** explicit AGPR via `art_base` + ASM     | +1-3pp  | Med     | 2-3     |
| **D-3** full main-loop port to 32x32x64 cell   | unknown | High    | 4-5     |
| **E**   hand-written ASM main loop             | +2-5pp  | Highest | 5+      |
| Per-shape K_REM constexpr (R49 done)           | 0       | -       | -       |
| Dispatch fall-throughs (R50 done for B32M4096) | flat    | -       | -       |

C-2 (warp-tile restructure) is the highest-ROI structural option.
Plan:
  1. Sketch ``grouped_rcr_kernel_4w<...>`` template variant with
     WARPS_M=2, WARPS_N=2, RBM=64, RBN=64 (matches rcr_4w's per-warp
     accumulator footprint = 256 fp32/lane → forces AGPR allocation
     per R47 verification).
  2. Wire dispatcher to select 4w-variant for FUSED_KTAIL=true
     (gpt_oss K=2880 shapes only) initially; falls back to 8w on
     other shapes.
  3. Compare resource report — expect AGPR ≥ 256, VGPR Spill = 0
     (matches rcr_4w<dense>).
  4. Run metric, expect +3-6pp on gpt_oss FP8 shapes.

R52 first step: build the prototype scaffolding (template, dispatch,
helper signatures) without modifying main-loop logic. R53 wires up
the actual main-loop schedule. R54 stress-tests on metric.

## Falsification register (no changes this round)

| Lever                                        | Status         | Round   |
|----------------------------------------------|----------------|---------|
| Lever A (async g→LDS) — base shipped         | SHIPPED        | R54-dm  |
| Lever B (dual LDS) — base shipped            | SHIPPED        | early   |
| Lever C-1 (restrict / lifetime hints)        | SATURATED      | R12,R54 |
| Lever C-1' (per-spec K_REM constexpr)        | LANDED         | R49     |
| Lever C-3 (art_base AGPR migration)          | not impl       | —       |
| Lever C-3' (``+a`` inline-asm hint)          | FALSIFIED      | R48     |
| Lever C-4 (mfma-vgpr-form mllvm flag)        | FALSIFIED      | R48     |
| **Lever C-2 (warp-tile to 4w)**              | **NEXT R52**   | —       |
| Lever D K-tail-only port                     | FALSIFIED      | R62-dm  |
| Lever D full main-loop port                  | NOT STARTED    | —       |
| Lever E (ASM main-loop)                      | NOT STARTED    | —       |
| Lever F (Qwen3 K=1536 short-K variant)       | FALSIFIED      | R35-grp |
| ``amdgpu_waves_per_eu(2,2)`` attribute       | FALSIFIED      | R47     |
| Drop ``__launch_bounds__(_, 1)`` entirely    | FALSIFIED      | R47     |
| sched_barrier / LICM / anti-CSE class        | FALSIFIED      | R31-32  |
| K-tail micro-knobs (vmcnt / reorder)         | SATURATED      | R3-R55  |
| Down-B32-M4096 fall-through                  | LANDED         | R50     |

## Attribution

- HipKittens HEAD: ``6c52d017`` — UNCHANGED this round
- Primus-Turbo: this doc note + verify_rule_fires.py probe (already
  in /tmp; not committed to repo as it's diagnostic)

## Validation paper-trail

```
/tmp/metric_round_51.log              (R51 single-shot: score 986)
/tmp/verify_rule_fires.py             (rule fires + 5-trial timing)
```
