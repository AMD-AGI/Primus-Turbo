# Round 64 — FP8 gpt_oss saturation, 7th consecutive NEUTRAL print

**Verdict**: NEUTRAL. No code change shipped. 7 consecutive in-band noise
prints since R57; R63 metric=694 fits R56-characterized band (median 695,
σ 2.27, n=30) at z=-0.44. Recommend operator terminate the run.

## R57–R63 print sequence (7 samples, identical functional code)

```
round   metric   z vs R56 noise (median=695, σ=2.27)
R57     697      +0.88
R58     696      +0.44
R59     696      +0.44
R60     696      +0.44
R61     695       0.00
R62     693      -0.88
R63     694      -0.44
                    n=7  mean=695.29  stdev=1.38  range=[693,697]
```

* Sample mean 695.29 ≡ R56 median 695 (Δ=0.29, well below 1σ).
* Sample stdev 1.38 < R56 σ 2.27 (smaller spread expected at n=7).
* Max |z| = 0.88. No outlier. Distribution consistent with i.i.d. draws
  from R56 noise model + ±3 score GPU/thermal drift over 7-round wallclock.

## Falsification gates (re-checked vs R63 note)

1. **No new lever** in `/shared-data/apps/kyle/local_skill/...SKILL.md`
   (file not present on disk, last guidance loaded into prior rounds is
   still current) or `scripts/_task_gpt_oss_fp8_kernel.md` since R62.
2. **Noise model still fits**: adding R63=694 to the R57-R62 sequence
   keeps mean (695.29) within 0.13σ of R56 median; no regime change.
3. **HK HEAD**: no new commits in the gpt_oss FP8 path since R63 docs
   commit. Stream-K infra (R13a-R17 as of R63) still not dispatcher-routed
   for any of the 8 in-scope cells; per FORBIDDEN PATHS small-tile and
   4-wave variants remain falsified.
4. **No correctness regression**: all 7 prints in valid score range,
   SNR>25 dB gate passed for all 8 shapes every round.
5. **NEW DIRECTIONS A-G** (SKILL.md untried list): all closed —
   - A1 Stream-K (R13a-R17 infra shipped, dispatcher gain NEUTRAL)
   - A2 SplitK var-K (R33 falsified)
   - A3 decoupled-warps (pre-R55 PMC predicts negative EV, R26-R28 audited)
   - B cross-stream (metric serial — out of scope)
   - C activation-cache (metric pre-quantizes — out of scope)
   - D SALU coord-decode (R54 falsified, R59 step-2 magic-divide A-PRIORI falsified)
   - E barrier scheme (R26-R28 audited single-barrier-drop, R55 follow-up)
   - F larger tiles (R32 falsified)
   - G cross-shape (R55 attempted, no net win)

## Why I did not test anything new

Every in-scope lever has been:
- (a) explicitly tested and falsified across R1-R55 (FORBIDDEN PATHS
  table covers macros, 4-wave, 256x128, small-tile, Down-B4 dispatcher,
  quant-cache, ktail, vmcnt, gm/xcds joint, slots/cs joint), OR
- (b) probed-and-NEUTRAL across R56-R63 (noise band).

Inventing a new probe outside FORBIDDEN PATHS that has not been tried
requires a fresh PMC signal or a fresh kernel template — neither has
appeared in the last 7 rounds. The R21 etiology (4× CTA-barrier-per-iter
schedule pin causing 60-70% MFMA pipe idle) is the standing root cause;
attacking it requires the multi-round Stream-K / decoupled-warps /
producer-consumer restructure called out in SKILL.md NEW DIRECTIONS A —
which is OUT of single-round budget by SKILL.md's own estimate (4-6
rounds per direction).

## Operator recommendation (4th repeat)

Terminate at R64. 36 more rounds will produce 36 more in-band noise
prints centered on 695. Closing the gap to TARGET=900 requires expanding
daemon scope:
  * Re-scope metric (e.g. drop pre-quantize-outside-timer rule and let
    fused activation count), OR
  * Hardware/firmware lever (rocm release, MI355X firmware), OR
  * Multi-week kernel-template research (Stream-K full port,
    decoupled-warps producer-consumer scheme).

None of these reach inside the single-round per-commit budget the
auto_optimize loop is designed for.

## Forward pointer (if run continues anyway)

If the operator chooses to keep running:
- R65-R72 will saturate at 695±2 (predicted band [690, 700]).
- Any print outside [690, 700] is more likely measurement-environment
  drift (GPU pin, thermal, idle-pool change) than a real signal — verify
  with a 9-sample re-run on the same GPU before treating as a regression
  or improvement.
- The first lever that could actually move the median is a
  HipKittens-side Stream-K dispatch path for the 8 in-scope cells,
  conditional on the var-K wgrad path getting its own SplitK or
  decoupled-warps variant. That work needs to land in HK first, then a
  single-round dispatcher rule in `config.py` can route the cells.
  Until HK ships such a variant, this metric cannot move from the
  Primus-Turbo side alone.
