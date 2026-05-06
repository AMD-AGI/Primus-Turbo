# Round 21 — gpt_oss-Down-B32-M4096 fwd RCR (next-after-R20 lowest, 1.318): R50 narrow-sweep widening FALSIFIED

## TL;DR

R21 metric data: gpt_oss-Down-B32-M2048 (R20 falsified) remains lowest
at 1.267, gpt_oss-Down-B32-M4096 (next, ratio 1.318) was the round
target. R50's deposit (commit history) only swept gm∈{2,3,4,5,6,8} ×
xcds∈{1,2,4,8} = 24 cells — gm > 8 and xcds = 16 columns were never
tested. R10/R11 found var-K dB wins precisely on these untested-column
patterns, so a wider sweep on the FORWARD path of B32-M4096 was a
plausible lever.

R21 probe (`/tmp/probe_round_21_gpt_oss_dn_b32m4096_fwd_widesweep.py`,
55-cell × 3-trial × 300-iter median): **R50's `(gm=4, xcds=4)` is the
unique tight optimum** at 1111.03 µs with **0.35 µs spread (0.03 %
CV)** — the tightest plateau seen on this metric's gpt_oss family. No
candidate beats production by the 0.5 % noise floor.

This is **zero-source-change** — falsification + R50 wide-sweep
confirmation note only. R50 rule unchanged. Patience now 19/30.

## Selected target (per R21 metric data, ascending)

```
fusedFP8_gpt_oss_20B-Down-B32-M2048    1774.3  1400.8  1.267  <135%  (R20 FALSIFIED)
fusedFP8_gpt_oss_20B-Down-B32-M4096    1966.5  1492.0  1.318  <135%  (R21 target)
fusedFP8_Qwen3-Down-B16-M2048          1981.3  1489.4  1.330  <135%  (R29 wide-sweep verified)
fusedFP8_gpt_oss_20B-Down-B4-M2048     1342.8   999.8  1.343  <135%  (R7 rule)
fusedFP8_gpt_oss_20B-GateUP-B4-M2048   1722.1  1280.2  1.345  <135%  (R23 rule)
fusedFP8_Qwen3-GateUP-B16-M2048        2210.2  1643.7  1.345  <135%  (R7 + R15 verified)
[geomean=1.3881, below_target=6/24, score=1000]
```

R21 picks `gpt_oss-Down-B32-M4096` per the "skip just-falsified shape,
descend to next-lowest UNTRIED" interpretation of the prompt. (R20 just
falsified the cell-split lever for B32-M2048; per `Score flat or down →
revert + write falsification note (don't waste cycles iterating on the
same lever)`, the same shape's next-untested lever is also off the
table for this round.)

## Hypothesis

R50 deposit narrowly swept gm∈{2,3,4,5,6,8} × xcds∈{1,2,4,8} = 24
cells:

```
cfg            median        spread   Δ vs default
(4,  4)        1959.77 TF    26.25    +15.96 (+0.82pp)  *winner
(4,  1)        1945.76 TF    12.75    +1.95
default(4,8)   1943.81 TF    87.97    baseline
(8,  4)        1939.01 TF    11.36    -4.80
…
```

The pattern that R10/R11/R14/R15 found wins on was widening the
candidate set to xcds = {1, 2} OR gm > 8 — precisely the columns
R50 never sampled at this shape (gm=12, gm=16, gm=24, gm=32, xcds=16).
So the wide sweep was warranted.

## Probe (`/tmp/probe_round_21_gpt_oss_dn_b32m4096_fwd_widesweep.py`)

55-cell × 3-trial × 300-iter median, single-GPU pinned (HK card 5).
Top-18 by median:

```
       cfg  median_us     min_us     spread     Δ vs prod
( 4, 4)    1111.03    1110.77       0.35     +0.00us +0.00 %  *winner *prod
( 8, 4)    1118.42    1118.01       0.92     -7.39us -0.67 %
( 4, 1)    1120.55    1120.09       4.40     -9.53us -0.86 %
( 1, 4)    1122.25    1121.51       0.87    -11.23us -1.01 %
( 4,16)    1122.48    1121.81       2.59    -11.45us -1.03 %
( 4, 8)    1122.54    1121.72       4.56    -11.51us -1.04 %
( 2, 2)    1132.39    1130.55       5.13    -21.37us -1.92 %
( 8, 2)    1136.55    1135.65       3.69    -25.53us -2.30 %
( 1, 2)    1136.67    1136.26       1.03    -25.64us -2.31 %
( 2, 1)    1139.19    1138.20       3.13    -28.16us -2.53 %
( 6, 4)    1140.18    1139.80       2.22    -29.15us -2.62 %
( 2,16)    1140.50    1139.87       3.62    -29.47us -2.65 %
( 2, 8)    1140.61    1140.32       1.00    -29.58us -2.66 %
( 3, 4)    1147.26    1146.84       0.70    -36.23us -3.26 %
( 4, 2)    1149.15    1149.15       0.18    -38.13us -3.43 %
( 5, 4)    1149.39    1148.85       1.20    -38.36us -3.45 %
( 8, 1)    1152.99    1152.32       3.15    -41.96us -3.78 %
( 2, 4)    1154.47    1152.91       1.68    -43.44us -3.91 %
```

Three structural observations:

1. **`(gm=4, xcds=4)` is THE unique tight optimum.** Wins by 7.39 µs
   over the closest neighbour `(gm=8, xcds=4)` — that's 7× the
   single-trial spread of either cell, so the win is statistically
   real. Best result for `(gm=4, xcds=4)`: spread 0.35 µs (0.03 % CV)
   — **the tightest plateau seen on gpt_oss family in this metric's
   sweep history**.

2. **gm > 8 is uniformly WORSE.** Every gm∈{12,16,24,32} candidate
   regressed by 1-4 % vs `(gm=4, xcds=4)` — they're not in the top-18
   so they're below 1154 µs (-3.9 % from prod). The "narrow sweep
   missed gm > 8" hypothesis is FALSE for this shape.

3. **xcds = 16 is also WORSE.** `(gm=4, xcds=16)` at -1.03 % is the
   best of the xcds=16 column, still under prod. Higher xcds dilutes
   the persistent-grid scheduler's per-XCD work, hurting throughput
   on this large-grid shape.

The R50 reasoning analytically holds:
> B=32 M=4096 has m_total=131072 rows = 8x more tiles than B=4 M=4096.
> The B=4 sibling needed gm=32 to extract L2 reuse from a tiny grid;
> B=32 grid already saturates the GPU at any gm, so SMALL gm=4 lets
> the schedule walk the N axis (11 N-tiles < 16 M-tiles) before
> iterating M, maximising L2 hit rate on the wider B-side.

The R21 wide sweep adds the **xcd dimension**: `(gm=4, xcds=4)` wins
both narrow (R50) AND wide (R21). The 4-XCD partition is empirically
the unique sweet spot for this shape — splitting a 5632-tile schedule
across 4 of 8 XCDs (1408 per XCD vs 5632/8=704 per XCD) avoids the
per-XCD launch / drain overhead that xcds=8 over-distributes (and
xcds=16 over-distributes even more).

## Cross-shape reconciliation — the gpt_oss-Down-B32 family is now fully exhausted

| Shape                     | tiles_m | m_total | Rule         | Sweep round | Wide-sweep verified? |
|---------------------------|--------:|--------:|--------------|-------------|----------------------|
| gpt_oss-Down-B4-M2048     | 8       |    8192 | (gm=2, xcds=2)  | R7 + R13 wide-sweep | ✓ |
| gpt_oss-Down-B4-M4096     | 16      |   16384 | (gm=32, xcds=4) | R12 wide-sweep      | ✓ |
| gpt_oss-Down-B32-M2048    | 8       |   65536 | (gm=16, xcds=4) | R8 wide-sweep + R20 dA-vs-fwd cross-verify | ✓ |
| gpt_oss-Down-B32-M4096    | 16      |  131072 | (gm=4, xcds=4)  | R50 narrow + **R21 wide-sweep**            | ✓ |

All 4 cells of the gpt_oss-Down family have wide-sweep verified rules
covering at least gm∈{1..32} × xcds∈{1..16} candidates. **The
forward dispatcher for gpt_oss-Down on this kernel is fully exhausted.**

## What this means for the FROZEN list

* `gpt_oss-Down-B32-M4096 fwd RCR (gm=4, xcds=4)` rule [R50] →
  **WIDE-SWEEP CONFIRMED**, R21. Plateau is uniquely tight; no
  alternative exists at any (gm, xcds) cell within ±0.5 % noise.
* `gpt_oss-Down-B32-M4096 dA RCR-via-T (gm=4, xcds=4)` rule [shared
  with fwd via R50] → **inherited from R20's discovery** that fwd
  and dA-via-T track each other on the gpt_oss-Down family.

## Why no commit this round

* Probe was `/tmp/`-only; no source modified.
* R50 rule unchanged (the wide sweep CONFIRMS the existing rule).
* No new dispatcher cell to add; no new shape carve-out warranted.
* Note this matches the R12-R15 / R20 pattern: wide-sweep falsification
  rounds add inventory without code churn.

## Patience accounting

| Counter                              | Value         |
|--------------------------------------|---------------|
| Score this round                     | 1000          |
| Best of run                          | 1000          |
| Improved this round?                 | No            |
| Consecutive unimproved rounds        | 19/30         |
| Rounds remaining before EARLY-STOP   | 11            |
| Rounds at cap since R3               | 19            |

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-21-fused-act-gpt-oss-down-B32-M4096-fwd-rcr-widesweep-FALSIFIED.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
R21 metric (HEAD d227c2e7, pre-probe):
  geomean=1.3881  score=1000  below_target=6/24  correct_fail=0/24

R21 metric (HEAD d227c2e7, post-probe verify):
  geomean=1.3859  score=1000  below_target=6/24  correct_fail=0/24
```

## Suggested next round (R22)

R22 should NOT re-attempt:
* gpt_oss-Down-B32-M4096 (gm, xcds) widening (R21, this round) — exhausted.
* gpt_oss-Down-B32-M2048 (gm, xcds) cell-split (R20) — exhausted.
* gpt_oss-Down-B32 family wide sweep — entire family is now wide-sweep
  verified (R7, R8, R12, R20, R21).

R22 candidates worth probing — by ascending ratio with already-verified
rules excluded:
* `gpt_oss-Down-B4-M2048` (1.343, R7 rule). R7 already wide-swept this
  but with 7-trial × 500-iter (less intensive than R20/R21). A R20-style
  fwd-vs-dA cross check might reveal something — but per R20 finding,
  fwd and dA-via-T track each other on the gpt_oss family. Likely
  another quick falsification round.
* `gpt_oss-GateUP-B4-M2048` (1.345, R23 rule). I haven't traced R23's
  sweep depth yet; worth checking the rule's history first.
* **Continue maintenance hold** (R17/R18/R19 pattern). Patience 19/30
  with 11 rounds buffer remaining. This is the recommended path; the
  Primus-side dispatcher has been exhausted across **20 rounds of this
  task** and adding more falsification notes has marginal value.

The R20 + R21 sequence has now wide-sweep-verified BOTH B32-M2048 and
B32-M4096 of the gpt_oss-Down family (the two worst-ratio shapes in the
metric). The remaining 4 below-target shapes are all on rules that were
either wide-sweep verified earlier (R7, R23, R29 verified Qwen3-Down-M2048
across all 3 paths) or sit on the architectural-ceiling band per R5/R8.
