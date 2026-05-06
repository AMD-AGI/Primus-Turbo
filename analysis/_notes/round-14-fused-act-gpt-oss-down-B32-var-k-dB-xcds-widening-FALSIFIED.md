# Round 14 — gpt_oss-Down-B32 var-K dB xcds-column widening: **FALSIFIED**

## Context (entering this round)

- Wall metric `_metric_grouped_fused_wall.py` score: **1000** (capped),
  geomean 1.3865.
- Lowest 4 ratios in the wall metric (all 4 below the 1.35 target):

  ```
  fusedFP8_gpt_oss_20B-Down-B32-M2048      1.265   *worst, target 1.35
  fusedFP8_gpt_oss_20B-Down-B32-M4096      1.283
  fusedFP8_Qwen3-235B-A22B-Down-B16-M2048  1.331
  fusedFP8_Qwen3-235B-A22B-GateUP-B16-M2048 1.333
  ```

- gpt_oss-Down-B32 forward (R8 / R50 rules) and dA RCR-via-H4 reroute (R8
  shared rule) were tight-verified in earlier rounds. The remaining wall gap
  on this family is in the **dB var-K** path.
- R10 (gpt_oss-Down-B4-M4096) + R11 (gpt_oss-Down-B4-M2048) wide-swept the
  xcds candidate set on the **B=4 small grid** (m_total=8192/16384, ~2
  wave-steps) and found `(gm=1, xcds=2)` wins both shapes (+1.24% / +0.65%
  kernel-only).
- R30 set the rule `(gm=4, xcds=4) for m_total >= 65536 + k=2880 + n=2880`
  via a 12-trial × 400-iter × 3-seed verify that ONLY compared `(gm=4,xcds=4)
  vs (gm=8,xcds=4) R39` — i.e., 2-cell sweep on the gm column at xcds=4. The
  xcds={1, 2, 8, 16} columns were never swept on this rule.

## Hypothesis tested this round

The R10/R11 (gm=1, xcds=2) win pattern on gpt_oss-Down-B4 var-K dB might OR
might not transfer to gpt_oss-Down-B32 var-K dB. Same n=2880 / k=2880
geometry (11×11 per-group output tiles) but the persistent grid is 8× larger:

| Shape (B=value, M_per=value) | per-group tiles × B | wave-steps |
|---|---|---|
| gpt_oss-Down-B4-M4096 (R10) | 121 × 4 = 484 | 2 |
| gpt_oss-Down-B4-M2048 (R11) | 121 × 4 = 484 | 2 |
| gpt_oss-Down-B32-M2048 (this) | 121 × 32 = 3872 | 15.1 |
| gpt_oss-Down-B32-M4096 (this) | 121 × 32 = 3872 | 15.1 |

R12 (Qwen3-Down-B16/B32 var-K dB) FALSIFIED xcds widening on a similar
"larger grid" regime (Qwen3-Down B32: 12 wave-steps) — R39 (gm=8, xcds=4)
was confirmed optimal there. But Qwen3-Down has tile-aligned geometry
(tiles_n=16) vs gpt_oss's tile-misaligned (tiles_n=11) cross-group stall
geometry, so the falsification might not generalize.

Hypothesis: gpt_oss-Down-B32 var-K dB might still benefit from
(gm=1, xcds=2) chiplet locality despite the 15-wave-step grid, because the
11×11 cross-group stall pattern is unique to gpt_oss n=k=2880.

## Probe methodology

`/tmp/probe_round_14_gpt_oss_down_b32_var_k.py` — direct call to
`hipkitten.load_fp8().grouped_variable_k_crr_dscale` (production var-K dB
kernel) with:

- 2 shapes × 20 candidate cells covering xcds ∈ {1, 2, 4, 8, 16} × gm
  representative set ({1, 2, 4, 8, 16, 32}).
- Per cell: 3 seeds × 7 trials × 200-iter p20.
- Reference baseline: R30 rule `(gm=4, xcds=4)`.
- Mirror R10 / R11 / R12 reporting format.

## Results

### gpt_oss-Down-B32-M2048 dB (m_total=65536; 15.8 wave-steps)

```
cell                       mean      Δ vs cur   spread pp
(8, 4) R39 prev           645.81 µs  +0.25%     2.19 pp   noise (Δ < spread)
(4, 4) R30 RULE           647.45 µs   0.00%     2.20 pp   baseline
(2, 4)                    648.46 µs  -0.16%     1.43 pp   noise
(16, 4)                   651.10 µs  -0.56%     2.61 pp   noise
(1, 4)                    651.19 µs  -0.58%     1.59 pp   noise
(32, 4)                   651.81 µs  -0.67%     2.42 pp   noise
(1, 2)                    655.26 µs  -1.21%     2.27 pp   loss (Δ ≈ spread)
(2, 1) ... (8, 8) etc.    656.59-670.78 µs                clear loss (-1.57% to -3.60%)
```

This shape sits at the **measurement noise floor** for the xcds=4 column:
within-seed 7-trial spread is 1.4-2.6 pp, which exceeds the absolute Δ
between (4,4) R30 / (8,4) R39 / (2,4) / (1,4). No xcds=2 candidate beats
the xcds=4 column; (1,2) is -1.21% (~ at spread), all other xcds={1,2,8,16}
losses are clear (-1.57% to -3.60%, well above spread).

### gpt_oss-Down-B32-M4096 dB (m_total=131072; 31.6 wave-steps)

```
cell                       mean       Δ vs cur   spread pp
(4, 4) R30 RULE           1088.80 µs   0.00%     0.17 pp    *unique top
(8, 4) R39 prev           1091.22 µs  -0.22%     0.27 pp
(1, 4)                    1092.89 µs  -0.38%     0.21 pp
(2, 4)                    1095.18 µs  -0.59%     0.20 pp
(32, 4)                   1096.49 µs  -0.71%     0.27 pp
(16, 4)                   1097.32 µs  -0.78%     0.17 pp
(1, 2)                    1109.06 µs  -1.86%     0.22 pp    clear loss
(16, 2)                   1109.53 µs  -1.90%     0.19 pp
(8, 2)                    1116.53 µs  -2.55%     0.24 pp
(2, 1) ... (8, 8) etc.    1119.90-1143.60 µs               clear loss (-2.86% to -5.03%)
```

This shape has **tight spread** (0.17-0.27 pp). R30 `(gm=4, xcds=4)` is the
unique top with -0.22% to -0.78% margin over the entire xcds=4 column and
-1.86% to -5.03% margin over xcds≠4 columns. **R30 confirmed.**

## Verdict — FALSIFIED (with shape-asymmetric noise)

### Combined finding across both shapes:

- **B32-M4096**: R30 `(gm=4, xcds=4)` is empirically the unique top across
  the 20-cell wide sweep. xcds widening **does not transfer** from B=4 to
  B=32 on this shape.
- **B32-M2048**: R30 sits within noise of (8,4) R39 / (2,4); none of the
  xcds={1, 2, 8, 16} candidates clearly win. Shape is on the
  measurement noise floor — high run-to-run variance (2 pp spread) makes
  any sub-1% delta non-actionable.

**No code changes** in this round. R30 rule is preserved.

### Why xcds widening does not transfer from B=4 to B=32

| Factor                     | B=4 (R10/R11 win)                     | B=32 (R14 falsified)                  |
|---|---|---|
| Wave-steps                 | ~2                                    | ~15-32                                |
| Persistent grid            | sparse (484 tile-steps / 256 CUs)     | saturated (3872 tile-steps / 256 CUs) |
| xcds=2 effect              | keeps schedule in single chiplet pair | over-restricts parallelism            |
| L2 reuse pattern dominator | per-K B-pack (single slab)            | cross-group stall + tile interleaving |
| Optimal cell               | (gm=1, xcds=2) gm=1 walks N-axis      | (gm=4, xcds=4) gm=4 fits cross-group  |

The R10/R11 (gm=1, xcds=2) win required **both** conditions:
(a) Tiny persistent grid (≤ 2 wave-steps so chiplet-locality matters more
    than parallelism), AND
(b) Cross-group stall geometry (11×11 tiles, mismatched gm/tiles_n).

B=32 has condition (b) but not (a) — the saturated grid means xcds=2
reduces the achievable parallelism without commensurate L2 benefit, so
xcds=4 (R30 / R39 family default) remains optimal.

## Inventory update

After R14 the following var-K dB cells are formally wide-sweep verified:

| Family                       | Rule              | Probe                         |
|---|---|---|
| gpt_oss-Down-B4-M2048   (R11) | (gm=1, xcds=2)    | 17-cell × 3-seed × 7-trial    |
| gpt_oss-Down-B4-M4096   (R10) | (gm=1, xcds=2)    | 17-cell × 3-seed × 7-trial    |
| gpt_oss-Down-B32 (both) (R14) | (gm=4, xcds=4)    | 20-cell × 3-seed × 7-trial    |
| gpt_oss-GateUP-B4-M4096 (R31) | (gm=1, xcds=4)    | 12-trial × 400-iter × 3-seed  |
| gpt_oss-GateUP-B32      (R31) | (gm=1, xcds=4)    | 12-trial × 400-iter × 3-seed  |
| Qwen3-Down-B16/B32 (R12 R39)  | (gm=8, xcds=4)    | 13-cell × 3-seed × 7-trial    |
| Qwen3-GateUP m_total>=16384   | various R39       | covered by R39 sweep          |
| DSV3 family                   | various R39       | covered by R39 sweep          |
| All m_total<16384 (B=4 fallthrough) | (gm=4, xcds=0=8) | binding default, R39 sweep |

**All gpt_oss + Qwen3 var-K dB cells are now formally wide-sweep verified
across xcds={1, 2, 4, 8, 16}**. The Python-side var-K dB dispatcher is
**fully tight-verified** for the 24-shape MoE suite.

Combined with R13's DSV3-GateUP forward falsification, R12's Qwen3-Down
var-K dB falsification, and R10/R11's gpt_oss-Down-B4 var-K dB wins, the
**Primus-side dispatcher** (forward + var-K dB cells) is now considered
**exhausted as a metric-affecting lever** for this 24-shape suite.

## Suggested next round

The score has been at 1000 cap for 12 consecutive rounds (R3-R14). With
R14 closing the last un-widened var-K dB cell, two reasonable directions:

1. **Qwen3 forward wide sweep** (8 shapes) — last un-probed forward family.
   Expected outcome (per R12 + R13 + R14 pattern): all current rules
   confirmed; document and close. Same falsification / negative-result
   round as R12 / R13 / R14.
2. **Pivot to maintenance hold** — patience 12/30 still has 18 rounds of
   buffer. Confirm dispatcher exhaustion is final via a meta-summary doc
   or hold without commit. Skip patience-trigger rounds while preserving
   all the wins from R10/R11 still shipped.

Either way, no metric-score-affecting change is expected without
expanding scope (HK kernel-internal templates / new HK kernel variants /
fused activation re-attack with different load primitive).
