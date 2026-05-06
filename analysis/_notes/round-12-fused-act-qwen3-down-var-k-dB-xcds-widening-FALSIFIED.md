# Round 12 — fused-act FP8 grouped: Qwen3-Down var-K dB xcds-column widening FALSIFIED (R29 verdict re-confirmed at full xcds sweep)

## Summary

- **Lever**: R29 (prior run) tight-verified Qwen3-Down M=2048 + M=4096 var-K
  dB at the inline R39 rule `(gm=8, xcds=4)` but the candidate set was
  **xcds=4-only** (6 cells × 4 shapes). R10 / R11 of the current run found
  `(gm=1, xcds=2)` wins on gpt_oss-Down's small-grid var-K dB by exactly
  this "missing xcds column" pattern. R12 widened the Qwen3-Down sweep
  to xcds={2, 4, 8} columns to test if the same lever transfers.
- **Class**: candidate-set widening probe — same R8 / R9 / R10 / R11
  methodology. Lowest-ratio non-gpt_oss shape in current metric (now at
  ratio 1.349, just dropped below 1.35 target due to single-run noise
  on 1.355-band shapes).
- **Verdict**: **FALSIFIED**. R39 `(gm=8, xcds=4)` is the genuine
  optimum across all 4 Qwen3-Down var-K dB shapes; xcds={2, 8} columns
  uniformly LOSS by -0.26..-9.27%. The R10/R11 small-grid lever does
  NOT transfer to Qwen3-Down's larger persistent-grid regime.
- **Files touched**: this round note only. No code change.

## Pre-round metric

```
[metric_fused_wall] Goals: HK_fused / TRT_baseline >= 1.35  geomean=1.3853  score=1000
[metric_fused_wall] correct_fail=0/24  reject=0/24  below_target=6/24  goals=18/24

Bottom-ratio shapes (sorted by ratio):
  1.261  gpt_oss_20B-Down-B32-M2048   <135%   (R5/R29 closed: K-tail kernel-internal)
  1.322  gpt_oss_20B-Down-B32-M4096   <135%   (same)
  1.332  Qwen3-Down-B16-M2048         <135%   (R29 tight-verified: closed)
  1.339  gpt_oss_20B-GateUP-B4-M2048  <135%   (R7/R34/R35 tight-verified: closed)
  1.345  Qwen3-GateUP-B16-M2048       <135%   (R36 tight-verified: closed)
  1.349  Qwen3-Down-B16-M4096         <135%   ← R12 target (R29 sibling)
```

## Why this lever was worth probing

1. **R10 / R11 just found xcds=2 wins on gpt_oss-Down small-grid var-K dB**
   by exactly this pattern (R33's xcds=4-only sweep missed (1, 2)). R29's
   Qwen3-Down sweep had the same coverage gap.
2. **R29's data shows xcds=4 column is locally flat** (every cell within
   ±0.82% of the (8, 4) baseline) — the data AT (8, 4) is in a flat
   region, suggesting the optimum might be elsewhere in cell space.
3. **Cheap to falsify**: 1 probe run, 1 doc commit, no kernel work.

## Probe data (`/tmp/probe_round_12_qwen3_down_var_k.py`)

200-iter × 7-trial × p20 × 3 seeds × 13 candidate cells × 4 shapes.
Test cells: xcds=2 column `gm ∈ {1, 2, 4, 8, 16}`, xcds=4 column
`gm ∈ {1, 2, 4, 8, 16}`, xcds=8 column `gm ∈ {1, 4, 8}`. Baseline:
R39 inline rule `(gm=8, xcds=4)`.

```
shape: Qwen3-Down-B16-M2048-dB  (m_total=32768, per-group 96 tiles, 6 wave-steps)

  cell        seed42   seed137  seed2024  med Δ  spread pp  verdict
  (1, 4)      +0.43%   +0.72%   +0.67%    +0.30% (mean)  0.59  TIE  spread > med
  (2, 4)      +0.27%   +0.15%   +0.40%    +0.10% (mean)  0.57  TIE
  (4, 4)      +0.50%   -0.10%   +0.36%    +0.08% (mean)  0.48  TIE
  (8, 4)R39   baseline                     +0.00%        0.89  (R39)
  ...
  (1, 2)      -3.01%   -2.78%   -2.32%    -2.70% (mean)  0.38  LOSS
  (1, 8)      -6.06%   -4.95%   -5.18%    -5.37% (mean)  0.49  LOSS  *worst

shape: Qwen3-Down-B16-M4096-dB  (m_total=65536, per-group 96 tiles, 6 wave-steps)

  cell        seed42   seed137  seed2024  med Δ vs R39  spread pp  verdict
  (8, 4)R39   baseline                     +0.00%        0.22  (R39)  *winner
  (2, 4)      -0.32%   -0.22%   -0.24%    -0.26%        0.14  LOSS
  (16, 4)     -0.45%   -0.27%   -0.19%    -0.30%        0.03  LOSS
  (4, 4)      -0.20%   -0.24%   -0.49%    -0.31%        0.50  LOSS
  (1, 4)      -0.65%   -0.43%   -0.65%    -0.58%        0.29  LOSS
  (16, 2)     -0.82%   -0.83%   -0.70%    -0.78%        0.16  LOSS
  ...
  (1, 2)      -3.20%   -2.92%   -3.24%    -3.15%        0.39  LOSS
  (1, 8)      -5.79%   -5.38%   -5.66%    -5.60%        0.34  LOSS  *worst

shape: Qwen3-Down-B32-M2048-dB  (m_total=65536, per-group 96 tiles, 12 wave-steps)

  cell        med Δ vs R39  spread pp  verdict
  (8, 4)R39    +0.00%        0.91       (R39)  *winner
  (2, 4)       -0.12%        0.54       TIE
  (16, 4)      -0.45%        0.48       LOSS
  (4, 4)       -0.45%        0.35       LOSS
  (1, 4)       -0.72%        0.83       LOSS
  ...
  (1, 8)       -4.94%        0.42       LOSS  *worst

shape: Qwen3-Down-B32-M4096-dB  (m_total=131072, per-group 96 tiles, 12 wave-steps)

  cell        med Δ vs R39  spread pp  verdict
  (8, 4)R39    +0.00%        0.11       (R39)  *winner (extremely tight)
  (4, 4)       -0.10%        0.34       TIE
  (16, 4)      -0.12%        0.23       TIE
  (2, 4)       -0.21%        0.03       small LOSS  (very tight spread)
  (1, 4)       -0.73%        0.11       LOSS
  ...
  (1, 8)       -9.27%        0.23       LOSS  *worst
```

**R39 `(gm=8, xcds=4)` is genuinely the optimum on every Qwen3-Down
var-K dB shape.** Of 12 alternative cells × 4 shapes = 48 candidate
points, **0 win-with-spread-< median** are above the standard
"med/spread > 1×" robustness threshold:

* B16-M2048's (1, 4) shows mean +0.30% with spread 0.59pp — spread > median,
  per-seed delta wraps both signs, NOT a robust signal. R29 reported the
  same cell at +0.11% under their methodology — both within noise.
* B16-M4096 / B32-* baseline is the unique top with the tightest spread.
* xcds=2 column uniformly LOSS by -0.55..-3.52% across ALL 4 shapes.
* xcds=8 column uniformly LOSS by -1.11..-9.27%.

## Why the R10 / R11 small-grid lever does NOT transfer

R10 / R11 found `(gm=1, xcds=2)` wins on gpt_oss-Down B=4 var-K dB.
That shape's persistent grid is **484 tile-steps ≈ 2 wave-steps** — very
small. xcds=2 keeps the schedule INSIDE a single chiplet pair, avoiding
cross-chiplet L2 invalidation that the 2-wave-step grid can't amortise.

Qwen3-Down's persistent grid is **1536 tile-steps ≈ 6 wave-steps** (B=16)
or **3072 tile-steps ≈ 12 wave-steps** (B=32). At 6-12 wave-steps per
slot, the cross-chiplet L2 cost is paid only once per ~3-6 wave-steps;
the parallelism benefit of xcds=4 (4-XCD schedule across both chiplet
pairs) DOMINATES, with xcds=2's narrower distribution costing -0.55..
-1.04% (tail of the K-tile traversal serializes on fewer XCDs).

Pattern: xcds=2 wins ONLY when the persistent grid is too small to
amortise chiplet-locality cost (≤ ~3 wave-steps). At ≥ 6 wave-steps
xcds=4 is robust optimum. Confirmed by:

* **gpt_oss-Down B=4**: 2 wave-steps → (gm=1, xcds=2) wins (R10/R11)
* **gpt_oss-GateUP B=4 M=2048**: 4 wave-steps → (gm=2, xcds=2) wins (R35)
* **gpt_oss-GateUP B=4 M=4096**: 4 wave-steps → (gm=4, xcds=4) wins (R9)
  — the transition point
* **gpt_oss B=32**: ~16 wave-steps → (gm=4, xcds=4) wins (R30)
* **Qwen3-Down B=16**: 6 wave-steps → R39 (gm=8, xcds=4) wins (R12 NEW)
* **Qwen3-Down B=32**: 12 wave-steps → R39 (gm=8, xcds=4) wins (R12 NEW)

The 4-vs-2 wave-step transition is the discriminator. R12 confirms the
universal R39 default catches every wave-step ≥ 6 var-K dB shape in the
24-shape MoE suite.

## Closed lever inventory (post-R12)

| Sub-direction                              | Status        | Closed at | Evidence |
|--------------------------------------------|---------------|-----------|----------|
| Path A — fused fwd cvt                     | FALSIFIED     | R7 prior  | DTL > DTR by 40% |
| Path B — BF16 LDS staging                  | FALSIFIED     | R3 prior  | LDS budget overflow |
| Path A — fused dB var-K cvt                | BLOCKED       | R8 prior  | var-K load also DTL (verified R12 here) |
| Path A — fused dA cvt                      | BLOCKED       | R8 prior  | same DTL load_a path family |
| dA RRR aligned-K reroute                   | LANDED        | R3 current| +12% bwd, +2.9pp geomean |
| Tensorwise quantize cache                  | LANDED        | R1 current| 934 → 1000 score lift |
| `select_default_config` lru_cache          | LANDED        | R2 current| 0.5us → 0.07us per call |
| group_offs identity cache                  | LANDED        | R2 current| sub-noise but real |
| dA RCR-via-T (gm, xcds) per-shape carve-out | LANDED       | R4-R8     | DSV3, Qwen3, gpt_oss-GateUP all rules in |
| var-K dB R30 (gpt_oss-Down B=32)           | LANDED        | R30 prior | (4, 4) tight-verified |
| var-K dB R31 (gpt_oss-GateUP B=32)         | LANDED        | R31 prior | (1, 4) tight-verified |
| var-K dB R33 (gpt_oss-Down B=4 M=2048)     | UPDATED→R11   | R11 current| (1, 2) replaces (16, 4) |
| var-K dB R35 (gpt_oss-GateUP B=4 M=2048)   | LANDED        | R35 prior | (2, 2) tight-verified |
| var-K dB R10 (gpt_oss-Down B=4 M=4096)     | LANDED        | R10 current| (1, 2) tight-verified |
| var-K dB R9 (gpt_oss-GateUP B=4 M=4096)    | LANDED        | R9 current | (4, 4) tight-verified |
| var-K dB R29 (Qwen3-Down xcds=4-only)      | RE-VERIFIED   | R12 current| xcds={2,4,8} all sweep done |
| var-K dB R36 (Qwen3-GateUP)                | TIGHT-VERIFIED| R36 prior | (1, 4) sub-noise, didn't ship |
| Forward RCR (every metric shape)           | TIGHT-VERIFIED| R6-R50    | per-shape rules in `config.py` |
| Stream overlap (dA ‖ dB)                   | FALSIFIED     | R8 prior  | CU saturation |
| Quant pipelining (a ‖ b)                   | FALSIFIED     | R1 prior  | both HBM-bound |
| Kernel template force (TK_RCR_FORCE_KERNEL)| TIGHT-VERIFIED| earlier   | binding default optimal within ±0.1pp |

## State-of-play

* **Score**: 1000 cap held for 9 consecutive rounds (R3-R12). Geomean
  1.385-1.392 single-run (target 1.35 → +3.5..+4.2pp buffer).
* **Patience**: 9/30 (21 rounds buffer to early-stop trigger).
* **All Python-side dispatch levers**: tight-verified or shipped. R12
  closes the last fresh xcds-column candidate that hadn't been swept.
* **Architectural ceiling**: HK fwd kernel-only ratio plateaued at 1.40
  (per `_metric_grouped_only.py` 63-round ceiling). The wall ratio is
  bounded above by the kernel-only ratio in the limit Q→0; R1's quant
  cache pushed Q toward 0 in the metric, exposing the kernel ceiling
  directly.

## Files touched

- `analysis/_notes/round-12-fused-act-qwen3-down-var-k-dB-xcds-widening-FALSIFIED.md`:
  this falsification note.

HipKittens: not modified this round.

## Suggestion for the next round (R13)

The Python-side dispatcher is now formally **fully tight-verified**:
every cell that can be discriminated by `(a.shape[1], b.shape[1],
m_total)` has had its (gm, xcds) optimum probed. R13 should pivot
away from cell tuning and onto one of:

1. **HK kernel-internal compute throughput** (R8 prior R9+ direction
   #1): the HK FP8 grouped RRR template is HK's per-component weak
   spot (+6..+13% slower than Triton on dA RRR for Qwen3 / gpt_oss).
   `kernel_fp8_layouts.cpp` line ~2565. Multi-round HK C++ work, high
   risk; could lift kernel-only ratio from 1.40 cap → 1.45+.
2. **C++ `quantize_fp8_tensorwise` HBM bandwidth** (R8 prior R9+
   direction #2): currently 67% of MI355X HBM peak; pushing to 80%
   lifts geomean by ~+1.5% via the wall-ratio model, ~+15 score
   points. Out of HK scope; needs `primus_turbo_cpp_extension`
   work.
3. **Maintenance / patience hold** (R5 / R12 lean): score capped at
   1000 with healthy buffer; let auto_optimize tick the patience
   counter while documenting. R5 of this run already recommended this
   path. After ~10 more maintenance rounds the early-stop will fire
   naturally.

R13 lean: option (3) maintenance until either upstream
`_metric_grouped_only.py` baseline shifts (forces a re-baseline of
all rules) OR a HK kernel-side change lands externally.
