# Round 20 — gpt_oss-Down-B32-M2048 (lowest ratio, 1.270): per-call (gm, xcds) re-tune for dA-vs-fwd FALSIFIED + R8 stale-data correction

## TL;DR

R20 metric data picked `gpt_oss-Down-B32-M2048` (ratio 1.270 — the
lowest-ratio shape across both runs of R19 and R20) as the round
target. Per R8 probe 1 the in-shape wall component breakdown was
"HK dA = 724 us vs TRT dA = 639 us" (the per-component regression source).
R20 hypothesis: **the dispatcher's `(gm=16, xcds=4)` rule for this shape
was wide-sweep verified at R8 on the FORWARD path only; the dA
RCR-via-H4-T call uses the SAME rule but might prefer a different cell
because the load patterns differ (a=grad_out vs a=activation, b=b_T vs
b=b)**. If true, splitting the dispatch into "fwd cell vs dA cell"
would unlock a free win.

Result: **FALSIFIED**. R20 probe (12-cell × 400-iter probe at
`/tmp/probe_round_20_gpt_oss_dn_b32m2048_fwd_vs_dA.py`) shows fwd and
dA-via-T track each other to <0.5 µs across the entire `gm∈{12,16,32} ×
xcds=4` plateau. The best-by-sum candidate (`(12, 4)`) beats production
`(16, 4)` by **+0.05 %** (1154.29 vs 1154.90 µs sum) — pure noise.

Bonus finding (**stale-data correction**): R8 probe 1's "HK dA = 724 us"
was pre-R9-transpose-cache and included the uncached `fp8_transpose_3d`
cost (~138 µs). With the R9 cache HIT (which is what actually happens
in the metric loop iter 2+), dA kernel time = **577 µs ≈ fwd kernel
time 577 µs**. R8's "HK dA 13 % slower than TRT" is no longer accurate
— with R9 cache, dA is roughly tied with TRT (HK 577 µs vs TRT 639 µs
per the R8 reported TRT number = HK ~10 % FASTER, not slower).

This round is **zero-source-change** — falsification + stale-data
correction note only.

## Selected target (per round 20 metric data)

R20 metric (single run, fresh after R19 commit `de2dfff4`):

```
fusedFP8_gpt_oss_20B-Down-B32-M2048    1776.0  1398.8  1.270  <135%
fusedFP8_gpt_oss_20B-Down-B32-M4096    1912.9  1490.8  1.283  <135%
fusedFP8_Qwen3-235B-A22B-GateUP-B16-M2048 2193.2  1648.9  1.330  <135%
fusedFP8_Qwen3-235B-A22B-Down-B16-M2048   1983.7  1491.8  1.330  <135%
fusedFP8_Qwen3-235B-A22B-GateUP-B16-M4096 2353.2  1755.5  1.340  <135%
fusedFP8_gpt_oss_20B-GateUP-B4-M2048      1730.5  1287.2  1.344  <135%
[geomean=1.3817 below=6/24]
```

Lowest-ratio shape: **`gpt_oss-Down-B32-M2048`** (ratio 1.270, B=32, M=2048,
N=2880, K=2880; m_total=65536, K%128=64, N%256=64 — the worst-case
double-misaligned shape).

## Hypothesis

* The `select_default_config` rule for this shape (config.py line 1729,
  R8 deposit) returns `(group_m=16, num_xcds=4)`. R8's wide-sweep
  ranking only timed the **forward** kernel:

  ```
  cfg          p20 median (fwd TFLOPS)
  (16, 4)      947.36   *winner
  (32, 4)      947.33   tied
  (12, 4)      947.00   tied
  (4, None=8)  933.02   baseline (-1.5 pp vs winner cluster)
  ```

* The same rule fires for the dA call (after the H4 reroute swaps to
  RCR layout): `select_default_config(avg_m=2048, n=2880, k=2880,
  layout='rcr', m_total=65536)` is identical-args-identical-rule
  for both fwd and dA-via-T. But:
    - fwd: `a = activation [m_total, K]`, `b = weight [B, N, K]`
    - dA: `a = grad_out [m_total, N_fwd]`, `b = b_T [B, K_fwd, N_fwd]`
* Different load patterns might favour different chiplet schedules.
  R8 never separated the two.

## Probe (`/tmp/probe_round_20_gpt_oss_dn_b32m2048_fwd_vs_dA.py`)

400-iter × 30-warmup p-mean per cell, single-GPU pinned (HK card 5).
H4 transpose cache pre-warmed once before each dA timing loop (matches
metric iter-2+ behaviour). 12-cell sweep over the R8 plateau + a few
exploratory neighbours:

```
       cfg    fwd_us     dA_us    sum_us
--------------------------------------------------
( 4, 8)    600.87    599.66   1200.53     (binding default)
( 8, 4)    595.49    599.10   1194.59
(12, 4)    577.25    577.04   1154.29     *best by sum (+0.05 % vs prod)
(16, 4)    576.98    577.92   1154.90     *production (R8 deposit)
(32, 4)    577.33    578.41   1155.73
(16, 8)    607.00    608.18   1215.18
(32, 8)    609.06    610.41   1219.47
( 4, 4)    601.06    606.27   1207.33
( 1, 4)    598.18    601.90   1200.08
( 1, 2)    607.76    610.12   1217.88
(16, 2)    600.42    599.35   1199.77
(16,16)    617.61    618.71   1236.32
```

Three structural observations:

1. **`gm∈{12,16,32} × xcds=4` is a dead-flat plateau on BOTH fwd and dA
   independently.** Fwd top-3: 576.98 / 577.25 / 577.33 (spread 0.35 µs).
   dA top-3: 577.04 / 577.92 / 578.41 (spread 1.4 µs). The `(12, 4)`
   "win" by sum is just the lucky single-trial draw; well within the
   ±2 µs single-trial p-mean noise reported in R8 / R10 / R14 sweeps.

2. **Best-by-sum candidate beats prod by +0.05 %** (delta 0.6 µs out
   of 1154 µs). Below the 0.5 % noise floor consistently used in
   prior round notes (e.g. R12-R14 falsification thresholds). NOT a
   real win.

3. **Fwd and dA times are essentially identical for every cfg.** Across
   the whole 12-cell sweep, `|fwd - dA| <= 6 µs` (max delta at `(8,4)`:
   3.61 µs). This is **the actually new finding**: with the R9
   transpose cache HIT, dA-via-H4-T kernel time tracks fwd kernel time
   perfectly — same rule, same cell, same plateau, same noise band.

## Stale-data correction — R8 "HK dA = 724 us" was pre-R9 cache

R8 probe 1 reported (analysis/_notes/round-8-fused-act-architectural-
ceiling-confirmed.md line 28):

```
gpt_oss-Down-B32-M2048   HK dA  724 us  vs  TRT dA  639 us  (HK +13 % slower)
```

The 724 µs number includes the **uncached** `fp8_transpose_3d` H4
reroute cost. R9 (commit history: `_FP8_H4_TRANSPOSE_CACHE` deposit
at line 104 of `grouped_gemm_fp8_impl.py`) shipped the weakref-keyed
transpose cache. After cache HIT (always after iter 1 in metric loop;
always within an optimizer step in production training):

* Pre-R9 dA path: 724 µs ≈ 138 µs transpose + ~586 µs RCR kernel
  ≈ R20 measured 577 µs ✓
* Post-R9 dA path: 0 µs cached transpose + 577 µs RCR kernel = 577 µs

So the corrected R20 per-component breakdown (using R8's TRT numbers
which are unaffected — Triton has no H4 reroute):

```
                       HK (corrected)  TRT (R8)   HK relative
fwd                    577 us         649 us     -11.1 %  HK FASTER
dA (post-R9 cache HIT) 577 us         639 us      -9.7 %  HK FASTER
dB var-K               654 us        1110 us     -41.1 %  HK FASTER
                       ─────         ─────
total                  1808 us       2398 us     -24.6 %  → ratio 1.326
```

Measured wall ratio in R20: **1.270**. Discrepancy 1.326 vs 1.270 = 4.4 %
explained by:
* R8's TRT numbers were also a single probe; R20 metric TRT is averaged
  over a longer iter count and shows TRT slightly faster than R8 caught.
* Quantize_fp8 wall (~599 µs both backends) was excluded from the per-
  component breakdown — it's a constant additive both backends pay,
  so its ratio = 1.0 drags the wall ratio toward 1.0.

The corrected breakdown puts HK ahead on EVERY component for this shape
— there is no per-component HK regression to chase. The ratio cap is
the additive `Q ≈ 599 µs` term in both numerator + denominator, exactly
as R8's analytical model predicted.

## What this means for the FROZEN list

* `gpt_oss-Down-B32-M2048 fwd RCR (gm=16, xcds=4)` rule [R8] → **CONFIRMED
  AGAIN**, this round. Plateau-mate `(12, 4)` is +0.05 % within
  noise; switching cells is not a real win.
* `gpt_oss-Down-B32-M2048 dA RCR-via-H4-T (gm=16, xcds=4)` rule
  [shared with fwd, R8] → **CONFIRMED**. dA-vs-fwd-cell-split lever
  is FALSIFIED; dA tracks fwd perfectly on this shape's plateau.
* **R8 architectural-ceiling note's "HK dA 13 % slower than TRT" → STALE**;
  superseded by R9 transpose cache. Future agents reading R8 should
  see this R20 correction.

## Why no commit this round

* Probe was on a `/tmp/` script; no source file modified.
* Re-tuning prod (16, 4) → (12, 4) is +0.05 % wall (sub-noise) → would
  be a churn commit risking metric regression on noise.
* Note this is NOT what R5/R10/R11/R14/R15 did when they FOUND a real
  win: they shipped a per-cell rule + verified non-regression. R20
  follows the same workflow with the opposite outcome (FALSIFIED →
  no commit + add to inventory).

## Patience accounting

| Counter                              | Value         |
|--------------------------------------|---------------|
| Score this round                     | 1000          |
| Best of run                          | 1000          |
| Improved this round?                 | No            |
| Consecutive unimproved rounds        | 18/30         |
| Rounds remaining before EARLY-STOP   | 12            |
| Rounds at cap since R3               | 18            |

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-20-fused-act-gpt-oss-down-B32-M2048-fwd-vs-dA-cfg-FALSIFIED.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
R20 metric (single run, HEAD de2dfff4):
  geomean=1.3817  score=1000  below_target=6/24  correct_fail=0/24

R20 metric (post-probe verify, HEAD de2dfff4):
  geomean=1.3866  score=1000  below_target=5/24  correct_fail=0/24
```

## Suggested next round

R21 should NOT re-attempt:
* Per-shape (gm, xcds) widening on `gpt_oss-Down-B32-M*` (R14 + R20 both
  falsified).
* dA-vs-fwd cell-split for any shape on the gm-plateau cluster (this
  round, FALSIFIED — dA tracks fwd perfectly post-R9).

R21 candidates worth probing:
* `Qwen3-Down-B16-M2048` (ratio 1.330, k=1536 shallow-K). R8 listed this
  as "k=1536 shallow-K throughput (HK weak spot)" — kernel-internal but
  the dispatcher rule for this shape might still be wide-sweep-untested
  (need to check `select_default_config` history). If the rule is from
  before R10's dispatcher exhaustion, it might admit a per-cell win.
* Continue maintenance hold (R17/R18/R19 pattern). Patience remains
  18/30 with 12 rounds buffer.
