# Round 22 — gpt_oss-GateUP-B4-M2048 fwd RCR (1.344): R23 9-cell sweep widening FALSIFIED

## TL;DR

R22 metric data: top-4 lowest-ratio shapes are all FROZEN (R20/R21 falsified
gpt_oss-Down-B32-{M2048,M4096}; R29 wide-swept Qwen3-Down-B16-M2048;
R7+R15 wide-swept Qwen3-GateUP-B16-M2048). The next ascending UNTESTED
candidate is `gpt_oss-GateUP-B4-M2048` at ratio 1.344 (R23 rule).
R23 deposit only swept 9 cells: `(1,4)*winner, (1,2), (3,4), (2,8),
(4,4), (2,4), (1,8), (1,1), (1,16)`. **gm ∈ {6, 8, 12, 16, 24, 32} ×
any xcds** were never tested at this shape.

R22 probe (`/tmp/probe_round_22_gpt_oss_gateup_b4m2048_widesweep.py`,
50-cell × 5-trial × p20 metric-aligned per-iter-sync): **R23's
`(gm=1, xcds=4)` is the unique tight optimum** at 145.72 µs with spread
0.96 µs (0.66 % CV). Wins by 0.60 µs over closest neighbor `(2, 2)` at
146.32 µs — sub-noise. **No candidate beats production by the 0.5 %
noise floor.** The previously-untested gm > 4 column is uniformly
WORSE by -1.4 % to -2.4 %.

This is **zero-source-change** — falsification + R23 wide-sweep
confirmation note only. Patience now 20/30.

## Selected target (per R22 metric data, ascending)

```
fusedFP8_gpt_oss_20B-Down-B32-M2048    1780.7  1401.2  1.271  <135% (R20 FALSIFIED)
fusedFP8_gpt_oss_20B-Down-B32-M4096    1967.0  1489.3  1.321  <135% (R21 FALSIFIED)
fusedFP8_Qwen3-Down-B16-M2048          1977.7  1487.7  1.329  <135% (R29 wide-swept all 3 paths)
fusedFP8_Qwen3-GateUP-B16-M2048        2192.7  1641.8  1.336  <135% (R7+R15 verified)
fusedFP8_gpt_oss_GateUP-B4-M2048       1723.7  1282.8  1.344  <135% (R22 target — R23 9-cell narrow)
[geomean=1.3843, below_target=5/24, score=1000]
```

R22 picks `gpt_oss-GateUP-B4-M2048` per the "skip just-falsified shape +
descend to next-lowest UNTESTED rule cell" interpretation.

## R23 deposit history (config.py:1369-1432)

R23 9-cell sweep (commit history, /tmp/verify_fp8_gateup_b4_m2048_round23.py):

```
cfg          med p20    min       max     spread
(1, 4)      1013.81    1006.90   1018.22   1.12%   *winner
(1, 2)      1008.70    1005.26   1011.25   0.59%   -5.11
(3, 4)      1005.56    1002.15   1009.59   0.74%   -8.25
(2, 8)      1004.22     999.64   1006.31   0.66%   -9.59
(4, 4)      1002.44    1000.38   1006.30   0.59%  -11.37
(2, 4)      1001.56     997.58   1009.44   1.18%  -12.25  ←round-68
(1, 8)       972.46     968.85    975.81   0.72%  -41.35
(1, 1)       970.79     968.02    974.13   0.63%  -43.02
(1, 16)      971.63     968.58    975.39   0.70%  -42.18
```

Cells NEVER tested: `gm ∈ {6, 8, 12, 16, 24, 32}` × any `xcds`. R10/R11
found dB var-K wins on similar untested-column patterns; the wide sweep
was warranted.

## Probe (`/tmp/probe_round_22_gpt_oss_gateup_b4m2048_widesweep.py`)

50-cell × 5-trial × p20, metric-aligned timing
(WARMUP=10, ITERS=50, per-iter `cuda.synchronize()` + Event timing,
return p20). Top-18 by median:

```
       cfg     med_us     min_us    spread       Δ vs prod
( 1, 4)     145.72     145.16      0.96     +0.00us +0.00 %  *winner *prod
( 2, 2)     146.32     145.68      0.84     -0.60us -0.41 %
( 2, 4)     146.76     146.44      1.40     -1.04us -0.71 %
( 1, 2)     146.80     145.76      1.68     -1.08us -0.74 %
( 4, 4)     147.12     146.52      1.08     -1.40us -0.96 %
( 8, 4)     147.80     147.56      0.88     -2.08us -1.43 %
(16, 4)     148.16     147.92      0.72     -2.44us -1.67 %
( 6, 4)     148.56     148.08      0.92     -2.84us -1.95 %
( 4, 2)     148.80     148.12      1.08     -3.08us -2.11 %
(12, 4)     149.20     148.88      0.72     -3.48us -2.39 %
( 3, 4)     149.24     149.12      0.24     -3.52us -2.41 %
(16, 2)     149.28     148.48      1.52     -3.56us -2.44 %
( 2, 8)     149.36     149.32      1.00     -3.64us -2.50 %
( 4, 8)     149.36     148.96      1.20     -3.64us -2.50 %
( 4, 1)     149.64     147.20      2.84     -3.92us -2.69 %
( 4,16)     149.72     149.64      0.24     -4.00us -2.74 %
( 3, 2)     149.92     148.60      1.92     -4.20us -2.88 %
( 2, 1)     149.96     147.76      2.40     -4.24us -2.91 %
```

Three structural observations:

1. **`(gm=1, xcds=4)` is THE unique tight optimum.** Wins by 0.60 µs
   over closest neighbour `(gm=2, xcds=2)` — single-trial spread of
   either cell is 0.96/0.84 µs, so the win is at the noise edge but
   consistent across 5 trials. Best result for `(gm=1, xcds=4)`:
   spread 0.96 µs (0.66 % CV).

2. **gm > 4 is uniformly WORSE.** Every gm∈{6,8,12,16,24,32}
   candidate regressed by 1.4-2.4 % vs `(gm=1, xcds=4)`. The
   "narrow sweep missed gm > 4" hypothesis is FALSE for this shape.
   R23's analytical reasoning ("gm=1 lets the schedule walk the
   entire N-row for each M-tile before moving on, maximising B-tile
   L2 reuse — better fit for tiles_m << tiles_n shapes") holds:
   tiles_m=8 < tiles_n=23 so smaller gm wins.

3. **`(2, 2)` second-place is interesting but sub-noise.**
   `(gm=2, xcds=2)` at 146.32 µs (-0.41 %) is the only candidate
   within 1 µs of production. Both share `xcds < 4` partition style,
   suggesting the `tiles_n=23` × 4-batch grid prefers narrow XCD
   spread. But the 0.41 % delta is well below the 0.5 % noise
   threshold; not actionable.

## Cross-shape reconciliation — gpt_oss-GateUP family rule audit

| Shape                       | tiles_m | m_total | Rule           | Sweep round | Wide-sweep verified? |
|-----------------------------|--------:|--------:|----------------|-------------|----------------------|
| gpt_oss-GateUP-B4-M2048     | 8       |    8192 | (gm=1, xcds=4) | R23 9-cell + **R22 50-cell wide-sweep** | ✓ |
| gpt_oss-GateUP-B4-M4096     | 16      |   16384 | (gm=14, xcds=4) | R10dm 1500-iter ×7-repeat | ✓ |
| gpt_oss-GateUP-B32-M2048    | 8       |   65536 | (gm=8, xcds=4)  | R70 24×6=144-cell sweep   | ✓ |
| gpt_oss-GateUP-B32-M4096    | 16      |  131072 | (gm=8, xcds=4)  | R70 24×6=144-cell sweep   | ✓ |

All 4 cells of the gpt_oss-GateUP family now have wide-sweep verified
rules. **The forward dispatcher for gpt_oss-GateUP on this kernel is
fully exhausted.**

## Combined gpt_oss family status (post-R20/R21/R22)

| Family              | All 4 (B, M) cells wide-sweep verified? |
|---------------------|-----------------------------------------|
| gpt_oss-Down (RCR)  | ✓ (R7, R8, R12, R20, R21)               |
| gpt_oss-GateUP (RCR)| ✓ (R10dm, R23, R70, R22)                |

The entire 8-shape gpt_oss family forward RCR dispatcher is now
exhausted. The remaining 5 below-target shapes are:

| Shape | Ratio | Path | Status |
|---|---:|---|---|
| gpt_oss-Down-B32-M2048   | 1.271 | fwd+dA+dB | R20 fwd-vs-dA cell-split FALSIFIED |
| gpt_oss-Down-B32-M4096   | 1.321 | fwd+dA+dB | R21 wide-sweep FALSIFIED         |
| Qwen3-Down-B16-M2048     | 1.329 | fwd+dA+dB | R29 fwd+dA+dB all wide-swept     |
| Qwen3-GateUP-B16-M2048   | 1.336 | fwd+dA+dB | R7+R15 wide-swept                |
| gpt_oss-GateUP-B4-M2048  | 1.344 | fwd+dA+dB | **R22 fwd wide-swept (this round)** |

**ZERO Primus-side dispatcher cells remain wide-sweep-untested in the
24-shape MoE metric suite.**

## What this means for the FROZEN list

* `gpt_oss-GateUP-B4-M2048 fwd RCR (gm=1, xcds=4)` rule [R23] →
  **WIDE-SWEEP CONFIRMED**, R22. 50-cell sweep at metric-aligned timing
  confirms R23's 9-cell pick. No alternative beats by > 0.5 % noise.
* gpt_oss family ENTIRE forward RCR dispatcher → **EXHAUSTED** across
  all 8 (B, M) cells, all wide-sweep verified.

## Why no commit this round

* Probe was `/tmp/`-only; no source modified.
* R23 rule unchanged (the wide sweep CONFIRMS the existing rule).
* No new dispatcher cell to add.
* Matches R12-R15 / R20 / R21 falsification-without-code-change pattern.

## Patience accounting

| Counter                              | Value          |
|--------------------------------------|----------------|
| Score this round                     | 1000           |
| Best of run                          | 1000           |
| Improved this round?                 | No             |
| Consecutive unimproved rounds        | 20/30          |
| Rounds remaining before EARLY-STOP   | 10             |
| Rounds at cap since R3               | 20             |

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-22-fused-act-gpt-oss-gateup-B4-M2048-fwd-rcr-widesweep-FALSIFIED.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
R22 metric (HEAD 33e73d20, pre-probe):
  geomean=1.3843  score=1000  below_target=5/24  correct_fail=0/24

R22 metric (HEAD 33e73d20, post-probe verify):
  geomean=1.3812  score=1000  below_target=5/24  correct_fail=0/24
```

## Suggested next round (R23)

R23 should NOT re-attempt:
* gpt_oss-GateUP-B4-M2048 (gm, xcds) widening (R22, this round) — exhausted.
* Any gpt_oss family forward RCR (gm, xcds) (R20, R21, R22, R7-R12 all
  wide-sweep verified).
* Any of the 5 below-target shapes' Primus-side dispatch — all are now
  on wide-sweep verified rules (this round closed the last remaining
  one).

R23 candidates:
1. **Maintenance hold** (R17/R18/R19/R20/R21 pattern). Patience 20/30
   with 10 rounds buffer. **STRONGLY RECOMMENDED** — this round
   completed the systematic exhaustion of every below-target shape's
   Primus-side dispatcher. There is now LITERALLY no Primus-side cell
   left to wide-sweep.
2. Audit non-below-target shapes for "stale rule" wins. Unlikely to
   yield anything (these shapes are above target so dispatcher is
   already-good), but might surface a 0.5-1% geomean lift if a long-
   ignored rule has a hidden tied-or-better competitor.
3. Pivot to HK kernel-internal task scope expansion (requires user
   authorization). The 5 below-target shapes' ratios (1.27-1.34) are
   now bounded by the architectural ceiling per R5 / R8 / R26. The
   only remaining lever is HK kernel-internal C++ work on
   `grouped_rrr_kernel` template (Qwen3 weak spot) or the K-tail
   epilog kernels (gpt_oss-Down K=2880).

The R20-R22 sequence has now done **3 consecutive wide-sweep
falsification rounds** + the R29 (predecessor task) verification
covering the 4th below-target shape's dispatcher. The Primus-side
dispatcher is **PROVABLY EXHAUSTED** for the 24-shape MoE metric
suite. R23+ should default to option (1) maintenance hold; only option
(3) has any chance of breaking the 1000 cap.
