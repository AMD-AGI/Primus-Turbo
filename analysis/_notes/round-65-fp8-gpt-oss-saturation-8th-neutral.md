# Round 65 — FP8 gpt_oss saturation, 8th consecutive NEUTRAL print

**Verdict**: NEUTRAL. No code change shipped. 8 consecutive in-band noise
prints since R57. Recommend operator terminate the run.

## R57–R64 print sequence (8 samples, identical functional code on both repos)

```
round   metric   z vs R56 noise (median=695, σ=2.27)
R57     697      +0.88
R58     696      +0.44
R59     696      +0.44
R60     696      +0.44
R61     695       0.00
R62     693      -0.88
R63     694      -0.44
R64     695       0.00
                    n=8  mean=695.13  stdev=1.25  range=[693,697]
```

Mean 695.13 ≡ R56 median 695 (Δ=0.13, well below 0.1σ). Spread tightening
as expected for repeated draws from a stationary distribution. Max |z| =
0.88. R65 will be one more independent sample from the same distribution.

## Falsification gates (re-checked vs R64 note)

1. **Primus-Turbo HEAD**: zero functional commits since R55; only doc
   notes R55-R64. dispatcher rules in `select_default_config` unchanged.
2. **HipKittens HEAD** (`49ffb984` — same as R63/R64): Stream-K infra
   commits R12-R17 (sk_split_n / sk_partial_buf / sk_workspace_ptr ABI)
   are all explicitly tagged "production NEUTRAL" in their commit messages
   — they expose pybind kwargs for future Stream-K dispatch but the
   default code path through `grouped_rcr_kernel` is unchanged. None of
   the 8 in-scope cells route through Stream-K in `config.py`.
3. **No new lever** in the SKILL.md NEW DIRECTIONS A-G untried list.
   All seven directions are closed:
   - A1 Stream-K (HK infra shipped, dispatcher gain NEUTRAL at R63)
   - A2 SplitK var-K (R33 falsified)
   - A3 decoupled-warps (R26-R28 audited; PMC predicts negative EV)
   - B cross-stream (metric serial — out of scope)
   - C activation-cache (metric pre-quantizes — out of scope)
   - D SALU coord-decode (R54 falsified, R59 step-2 A-PRIORI falsified)
   - E barrier scheme (R26-R28 audited)
   - F larger tiles (R32 falsified)
   - G cross-shape (R55 falsified)
4. **No correctness regression**: every print in [693, 697], inside the
   R56-characterized ±2σ band; SNR>25 dB gate passed every sample.

## Why no probe ran this round

Inventing a new probe outside FORBIDDEN PATHS that has not been tried
requires a fresh PMC signal or a fresh kernel template — neither has
appeared in 8 rounds. R21 etiology (4× CTA-barrier-per-iter schedule pin
causing 60-70% MFMA pipe idle) is the standing root cause; attacking it
requires the multi-round Stream-K dispatcher routing or decoupled-warps
restructure called out in SKILL.md NEW DIRECTIONS A. SKILL.md itself
budgets these at 4-6 rounds each, exceeding single-round per-commit budget.

## Operator recommendation (5th repeat)

Terminate at R65. 35 more rounds will produce 35 more in-band noise
prints centered on 695. Closing the gap to TARGET=900 requires expanding
daemon scope — see R64 note for the three options (re-scope metric,
hardware/firmware lever, or multi-week kernel-template research).

## Forward pointer (if run continues anyway)

R65 metric forecast: 695 ± 2 score (band [690, 700]).
Any print outside [690, 700] is more likely measurement-environment
drift (GPU pin, thermal, idle-pool change) than a real signal — verify
with a 9-sample re-run on the same GPU before treating as a regression
or improvement.

The first lever that could actually move the median is wiring HK's
already-shipped Stream-K ABI (sk_split_n / sk_workspace_ptr) into the
gpt_oss dispatcher. That requires:
  1. HK side: a Stream-K kernel variant that beats persistent on at
     least one in-scope shape (currently the scaffolding is gated and
     the dispatch path is NEUTRAL).
  2. PT side: a one-line dispatcher rule selecting it for that shape.

Step 1 is missing. Until HK ships a Stream-K variant that wins at the
microbench level on one of the 8 cells, the metric cannot move from the
Primus-Turbo side alone.
