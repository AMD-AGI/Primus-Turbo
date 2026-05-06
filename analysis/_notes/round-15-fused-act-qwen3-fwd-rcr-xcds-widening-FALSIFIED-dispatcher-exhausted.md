# Round 15 — Qwen3 forward FP8 RCR xcds-column widening: **FALSIFIED** + Primus-side dispatcher formally exhausted

## Context (entering this round)

- Wall metric `_metric_grouped_fused_wall.py` score: **1000** (capped, 12
  consecutive rounds), geomean 1.3819.
- Below-target shapes (ratio < 1.35):
  ```
  fusedFP8_gpt_oss_20B-Down-B32-M2048      1.269 (R14 falsified, R30 confirmed)
  fusedFP8_gpt_oss_20B-Down-B32-M4096      1.315 (R14 falsified, R30 confirmed)
  fusedFP8_Qwen3-235B-A22B-Down-B16-M2048  1.332 (R29 narrow probe, this round target)
  fusedFP8_Qwen3-235B-A22B-GateUP-B16-M2048 1.342 (R7 narrow probe, this round target)
  fusedFP8_Qwen3-235B-A22B-GateUP-B32-M2048 1.348 (R7 narrow probe, this round target)
  fusedFP8_gpt_oss_20B-GateUP-B4-M2048      1.344 (R23 already wide-verified)
  ```
- After R10-R14 closures, the only remaining un-wide-swept forward FP8 RCR
  cells were the 6 Qwen3 shapes:

  | Family | Rule | Probe history |
  |---|---|---|
  | Qwen3-GateUP M=4096 (B16/B32) | (gm=1, xcds=4) R10/R45 | R10: 3 cells {(1,4), (4,4), (4,8)}; R45 verified (1,4) vs (4,8) on B16 (2 cells) |
  | Qwen3-GateUP M=2048 (B16/B32) | (gm=16, xcds=4) R7    | R7: 5 cells {(16,4), (32,4), (1,4), (4,4), (4,8)} |
  | Qwen3-Down M=2048 (B16/B32)   | default (gm=4, xcds=8) | R29: only proposed (gm=2, xcds=8); within noise → kept on default |

  Qwen3-Down M=4096 was already wide-sweep verified in R6 (28-cell sweep
  → (gm=2, xcds=8) shipped); not re-probed here.

## Hypothesis tested this round

The 6 Qwen3 forward rules above were authored before the R10/R11/R14 wide
xcds candidate-set methodology became standard; xcds={1, 2, 16} columns
were never swept on any of these rules. R10/R11 found wins on similar
"gm column with narrow xcds candidate set" gaps. Apply the same R10/R11
pattern to the 6 Qwen3 forward shapes.

Wave-step grid for the 6 shapes:

| Shape | tiles_m × tiles_n × B | Wave-steps |
|---|---|---|
| Qwen3-GateUP-B16-M2048 | 8 × 12 × 16  | 6.0  |
| Qwen3-GateUP-B32-M2048 | 8 × 12 × 32  | 12.0 |
| Qwen3-GateUP-B16-M4096 | 16 × 12 × 16 | 12.0 |
| Qwen3-GateUP-B32-M4096 | 16 × 12 × 32 | 24.0 |
| Qwen3-Down-B16-M2048   | 8 × 16 × 16  | 8.0  |
| Qwen3-Down-B32-M2048   | 8 × 16 × 32  | 16.0 |

Per the R10/R11/R14 pattern: small grids (≤2 ws) win xcds=2; large grids
(≥10 ws) tend to confirm xcds=4-8. The 6 Qwen3 shapes span 6-24 ws —
GateUP-B16-M2048 (6 ws) is the most likely candidate for a new win.

## Probe methodology

Two probes — same R13/R14 format:

- `/tmp/probe_round_15_qwen3_gateup_fwd.py` — 4 shapes × 22 cells × 3
  seeds × 7 trials × 200-iter p20 (~5 min/shape).
- `/tmp/probe_round_15_qwen3_down_m2048_fwd.py` — 2 shapes × 21 cells × 3
  seeds × 7 trials × 200-iter p20 (~1.5 min/shape).

Both call `hipkitten.load_fp8().grouped_rcr_dscale` directly (production
forward path), with cells covering xcds ∈ {0=binding-default, 1, 2, 4, 8,
16} × gm ∈ {1, 2, 4, 8, 16, 32} as relevant.

## Results — all 6 shapes confirm current rule

### Qwen3-GateUP family (4 shapes)

```
B16-M2048 (rule (16,4) R7;  6 ws):
  (16, 4)*RULE         325.44 µs   0.00%  spread 0.11pp   *unique top
  (32, 4)              325.82 µs  -0.11%  spread 0.63pp   tied within noise
  (1, 4)*M4096-rule    326.87 µs  -0.44%  spread 0.77pp
  ... 18 more cells, all -0.64% to -6.86%

B32-M2048 (rule (16,4) R7; 12 ws):
  (16, 4)*RULE         658.19 µs   0.00%  spread 0.84pp   *winner (high spread)
  (32, 4)              659.34 µs  -0.17%  spread 0.20pp
  (1, 4)*M4096-rule    661.29 µs  -0.47%
  (16, 2)              662.75 µs  -0.69%
  ... 17 more cells, all -1.00% to -5.78%

B16-M4096 (rule (1,4) R10/R45; 12 ws):
  (1, 4)*RULE          651.01 µs   0.00%  spread 0.28pp   *unique top
  (4, 1)               652.73 µs  -0.26%  spread 0.59pp   tied within noise
  (4, 8)               653.42 µs  -0.37%
  (4, 16)              653.93 µs  -0.45%
  ... 17 more cells, all -0.50% to -3.57%

B32-M4096 (rule (1,4) R10/R45; 24 ws):
  (1, 4)*RULE         1289.35 µs   0.00%  spread 0.10pp   *unique top, very tight
  (4, 8)              1296.07 µs  -0.52%  spread 0.27pp   clear loss
  (4, 16)             1296.18 µs  -0.53%
  ... 18 more cells, all -0.57% to -4.21%
```

### Qwen3-Down M=2048 family (2 shapes)

```
B16-M2048 (rule default (4, 8);  8 ws):
  (4, 1)               221.54 µs  +0.08%  spread 0.36pp   tied within noise
  (4, 0)*DEFAULT       221.71 µs   0.00%  spread 0.36pp
  (4, 16)              221.86 µs  -0.07%  spread 0.87pp   tied
  (16, 4)              227.15 µs  -2.45%  clear loss
  ... 17 more cells, all -2.58% to -7.06%
  ⇒ R29 default confirmed; xcds={1, 16} cells all within ±0.1% noise.

B32-M2048 (rule default (4, 8); 16 ws):
  (4, 0)*DEFAULT       435.86 µs   0.00%  spread 0.39pp   *unique top
  (4, 1)               436.35 µs  -0.11%  spread 0.53pp
  (4, 16)              437.77 µs  -0.44%
  (16, 4)              444.30 µs  -1.94%  clear loss
  ... 17 more cells, all -2.25% to -7.56%
  ⇒ R29 default confirmed; xcds≠8 columns all clear loss.
```

### Verdict for each shape

| Shape                  | Rule (post-R15) | Verdict |
|---|---|---|
| Qwen3-GateUP-B16-M2048 | (gm=16, xcds=4) R7   | confirmed (unique top, 0.11pp spread) |
| Qwen3-GateUP-B32-M2048 | (gm=16, xcds=4) R7   | confirmed (top, (32,4)/(1,4) within noise) |
| Qwen3-GateUP-B16-M4096 | (gm=1, xcds=4) R10/R45 | confirmed (top, (4,1)/(4,8) within noise) |
| Qwen3-GateUP-B32-M4096 | (gm=1, xcds=4) R10/R45 | confirmed (unique top, 0.10pp spread) |
| Qwen3-Down-B16-M2048   | default (gm=4, xcds=8) | confirmed (top, (4,1)/(4,16) within noise) |
| Qwen3-Down-B32-M2048   | default (gm=4, xcds=8) | confirmed (unique top, 0.39pp spread) |

**No code changes** in this round.

## Why R10/R11 win pattern does NOT transfer to Qwen3 forward

| Pattern              | Where it wins                          | Where it falsifies                      |
|---|---|---|
| (gm=1, xcds=2) | gpt_oss-Down-B4 var-K dB (2 ws) R10/R11 | All medium-large grids (R12 / R14 / R15) |
| Narrow xcds candidate set | Small grids + cross-group stall    | Already-wide-swept grids                |

Qwen3-GateUP-B16-M2048 has 6 wave-steps — closest to the small-grid regime
where R10/R11 wins, but still 3× larger than the B=4 (2 ws) regime. The
(gm=16, xcds=4) R7 rule already picks the L2-reuse-maximising config for
the medium-grid persistent loop; xcds=2 over-restricts and the (gm=1)
rule that wins for B=4 small grids loses for the medium-grid 8-cycle
N-axis traversal.

The R7 / R10 / R45 narrow probes happened to bracket the optima — same
falsification class as R13's DSV3-GateUP confirmation.

## Inventory update — Primus-side forward dispatcher fully exhausted

**After R15, every forward FP8 RCR cell in the 24-shape suite is formally
wide-sweep verified across xcds ∈ {0/1/2/4/8/16}**:

| Family | Rule (group_m, num_xcds) | Probe |
|---|---|---|
| gpt_oss-GateUP-B4-M2048   | (1, 4)  R23     | 9-cell × 7-trial verify       |
| gpt_oss-GateUP-B4-M4096   | (14, 4) R10dm   | 1500-iter × 7-repeat re-sweep  |
| gpt_oss-GateUP-B32 (both) | (8, 4)  R70     | 24×6=144-cell sweep           |
| gpt_oss-Down-B4-M2048     | (2, 2)  R7      | 40-cell × 7-trial             |
| gpt_oss-Down-B4-M4096     | (32, 4) R12     | 54-cell × 7-trial             |
| gpt_oss-Down-B32-M2048    | (16, 4) R8      | 54-cell × 7-trial             |
| gpt_oss-Down-B32-M4096    | (4, 4)  R50     | 11-cell × 7-trial             |
| DSV3-GateUP-M4096 (both)  | (2, None) R8    | **22-cell R13** ✓             |
| DSV3-GateUP-M2048 (both)  | (16, 4) R8/R45  | **22-cell R13** ✓             |
| DSV3-Down (4)             | (32, 4) R20/R58 | 9-cell × 5-repeat × 12-trial   |
| Qwen3-GateUP-M2048 (both) | (16, 4) R7      | **22-cell R15** ✓             |
| Qwen3-GateUP-M4096 (both) | (1, 4)  R10/R45 | **22-cell R15** ✓             |
| Qwen3-Down M=2048 (both)  | default (4, 8)  | **21-cell R15** ✓             |
| Qwen3-Down M=4096 (both)  | (2, 8)  R6      | 28-cell sweep                 |

**Combined forward + var-K dB inventory** (R15 closes the last gap):

| Path        | Status                   | Closing rounds |
|---|---|---|
| Forward FP8 RCR (24 shapes)   | All wide-sweep verified   | R6/R7/R8/R10/R12/R13/R15 etc. |
| dA backward FP8 (24 shapes)   | All on R8/H4 reroute paths | R3/R6/R7/R8/R34 |
| dB var-K FP8 (24 shapes)      | All wide-sweep verified   | R10/R11/R12/R14 |

**The Primus-side dispatcher (forward + dA + dB var-K) is now formally
fully exhausted as a metric-affecting lever for the 24-shape MoE suite.**

## Suggested next round

Score has been at 1000 cap for 13 consecutive rounds (R3-R15). Patience
13/30 with 17 rounds remaining buffer. With dispatcher fully exhausted,
the only remaining levers that could affect the score are:

1. **HK kernel-internal templates** — out-of-scope per "Forward only"
   task body; would require new HK kernel variants (e.g., deeper unroll
   for K=7168, K-tail epilog for K=2880 misalignment, RRR throughput
   uplift). Multi-round commit, requires HipKittens repo work.
2. **Re-attack fused activation with different load primitive** — R7
   originally falsified DTR-based fused-fwd. Some new HipKittens kernel
   variant might have changed the primitive cost balance. Multi-round.
3. **Maintenance hold** — let patience tick to 30 / 30 with doc-only
   rounds documenting the exhaustion verdict more thoroughly.

Recommended immediate action: **maintenance hold** for the next 5-10
rounds. The dispatcher exhaustion verdict is now well-documented across
R12/R13/R14/R15 falsification notes. Further dispatcher-tuning rounds
will continue to produce negative results without score change. If the
user wants to re-engage, options 1 and 2 require explicit task-scope
expansion.
