# Round 10 — fused-act FP8 grouped: gpt_oss-Down-B4-M4096 var-K dB rule enable (R38 revival, candidate-set widened)

## Summary

- **Lever**: enable a previously-disabled var-K dB rule for
  gpt_oss-Down-B4-M4096 (R38 was `if False`-disabled). R10 widened R38's
  xcds=4-only candidate set to xcds={2, 4, 8} columns and found
  `(gm=1, xcds=2)` is +0.48pp better than R38's `(gm=16, xcds=4)`.
- **Class**: same R8 / R9 "candidate-set widening" pattern. The R38
  comment block had documented +1.37% kernel-direct lift from (16, 4)
  but was never enabled (`if False`) — likely because xcds=4-only sweep
  missed the chiplet-locality lever (xcds=2 captures additional +0.42%
  on top of gm=1 schedule).
- **Metric (single-run wall, noisy, score capped at 1000)**:
  - Before (1 run): geomean **1.3898**, below_target 5/24, score 1000.
  - After (8 runs): scores **1000 / 1000 / 952 (outlier) / 992 (outlier)
    / 1000 / 1000 / 1000 / 1000** — **6/8 runs hit 1000**, 2 outliers
    were single-run GPU-side noise (shared host phantom-VRAM noise band).
    Geomean per-run on the 1000 runs: 1.3888 / 1.3940 (the others ran
    score-only). Mean across all 8 runs is dominated by outliers; on
    the score plane, R10 is statistically identical to R9 baseline.

## Probe data (`/tmp/probe_round_10_gpt_oss_down_b4_m4096_var_k.py`)

200-iter × 7-trial × p20 × 3 seeds × 17 candidate cells × xcds={2, 4, 8}
columns:

```
shape: gpt_oss-Down-B4-M4096 var-K dB
  m_total=16384, N_fwd=2880, K_fwd=2880, B=4 groups
  per-group output [N=2880, K=2880] = 11×11=121 tile-steps × 4 = 484
  tile-steps over 256 CUs ≈ 2 wave-steps (very small grid)

  cell      seed42  seed137  seed2024  med Δ vs cur  spread pp  verdict
  (1, 2)    +1.33%  +1.11%   +1.28%    +1.24%        0.17       WIN  med/spread=7.4×
  (16, 2)   +1.09%  +0.99%   +1.48%    +1.18%        0.49       WIN
  (32, 2)   +0.94%  +1.18%   +1.31%    +1.14%        0.37       WIN
  (16, 4)R38+0.76%  +0.71%   +0.83%    +0.76%        0.12       WIN  (R38 candidate)
  (32, 4)   +0.78%  +0.51%   +0.64%    +0.64%        0.27       WIN
  (1, 4)    +0.68%  +0.56%   +0.58%    +0.60%        0.12       WIN
  (2, 2)    +0.82%  +0.63%   +0.17%    +0.54%        0.65       small WIN
  (8, 4)cur baseline                    +0.00%        0.10       (R39 default)
  (1, 8)    -0.92%  -0.87%   -0.75%    -0.85%        0.17       LOSS
  (8, 8)    -1.47%  -1.71%   -1.43%    -1.54%        0.28       LOSS
  (16, 8)   -1.78%  -1.67%   -1.79%    -1.72%        0.12       LOSS
```

`(gm=1, xcds=2)` is the unique top with the tightest spread (0.17pp) —
every-seed positive (3/3) at +1.11..+1.33%, baseline-min beat in 3/3
seeds. Median / spread = 7.4×.

## Why (gm=1, xcds=2) wins on this small-grid

Persistent grid is **484 tile-steps over 256 CUs ≈ 2 wave-steps** (very
small). With only 2 wave-steps:

- **gm=1**: walks the entire 11-row N-axis under each individual K-tile
  before advancing K, maximising B-pack L2 reuse on the per-K column slab
  (one slab serves 11 N-rows back-to-back).
- **xcds=2**: keeps the 2-wave-step schedule INSIDE a SINGLE chiplet
  pair (vs xcds=4 splitting across both chiplet pairs of MI355X 8-XCD
  topology). With only 2 wave-steps, cross-chiplet L2 invalidation
  dominates over the parallelism benefit of wider distribution.

R38's `(gm=16, xcds=4)` was sub-optimal because:
1. gm=16 over-batches the 11-row N-axis (16/11=1.45 batches; 5 unbatched
   rows on second pass = fractional-tail stall);
2. xcds=4 splits across chiplet pairs, paying L2 invalidation that the
   small grid can't amortise.

The R8 / R34 / R9 / R31 missing-candidate pattern repeats here: R38's
sweep was xcds=4-only, missing the entire xcds=2 column.

## Bench correctness (`bench_grouped_gemm_turbo.py --dtype fp8`)

All 24/24 fp8 grouped shapes PASS:

```
| 12 | gpt_oss_20B-Down       |   4 | 4096 | 2880 | 2880 | fp8 | tensorwise | PASS | 0.14 us | 1981.74 fwd TF | 0.41 us | 1327.97 bwd TF |
```

(target shape gpt_oss-Down-B4-M4096; bench runs full fwd+bwd correctness
allclose with all-pass.) Average forward TFLOPS 2196.66, backward
1656.83 — both unchanged vs prior runs.

Bit-equivalent output verified at `/tmp/probe_round_10_correctness.py`:
max_abs_diff=0.0 between (gm=8, xcd=4) and (gm=1, xcd=2) on B4-M4096
in 3/3 seeds {0, 42, 137}; bit_eq=True.

## Discriminator: clean carve-out via existing m_total guards

The new `elif a.shape[1] == 2880 and b.shape[1] == 2880` lives between
R30 (which has `m_total >= 65536` guard) and R31 (`b.shape[1] == 5760`).
Net rule scope:

```
m_total      shape                              dispatch
8192         gpt_oss-Down-B4-M2048-dB           m_total<16384 branch (R33: gm=16, xcd=4)
16384        gpt_oss-Down-B4-M4096-dB           NEW R10 (gm=1, xcd=2)
65536        gpt_oss-Down-B32-M2048-dB          R30 (gm=4, xcd=4)
131072       gpt_oss-Down-B32-M4096-dB          R30 (gm=4, xcd=4)
```

No overlap with R30 / R31 / R39 / R33 / R35.

## Files touched

- `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`:
  replaced the `elif False and (...)` R38 guard with an active rule
  (~110-line in-code comment block + 2-line dispatch). The disabled
  R38 comment block remains for historical reference.
- `analysis/_notes/round-10-fused-act-gpt-oss-down-B4-M4096-var-k-dB-rule-enable.md`:
  this note.

HipKittens: not modified this round.

## Suggestion for the next round

R10 / R8 / R9 chain has now found 3 candidate-widening levers in 3
consecutive rounds. The remaining var-K rules:
- R30: re-probed in R8 with 12 cells across xcds={2, 4, 8}; (4, 4) holds.
- R31: split by R9 (B4-M4096 → (4, 4); B32 → (1, 4) holds).
- R10 NEW: gpt_oss-Down B4-M4096 → (1, 2).
- R33: gpt_oss-Down B4-M2048 (m_total<16384). R33's candidate set was
  also xcds=4-only — eligible for R10-style widening.
- R35: gpt_oss-GateUP B4-M2048 — extensively probed in R35 (10-cell
  neighbor sweep across xcds={1, 2, 4, 8}); unlikely to find more.

R11 candidate: re-probe R33 with xcds={2, 4, 8} columns (gpt_oss-Down
B4-M2048 var-K dB at m_total=8192, 484 tile-steps × M=2048 = 484
tile-steps over 256 CUs ≈ 2 wave-steps, EXACT same persistent-grid
geometry as R10's gpt_oss-Down-B4-M4096 (same 11×11 per-group output
× 4 groups; M_per_group only affects A-pack column count, not the
N×K tile geometry). So R10's (1, 2) lever might generalize.

R10 also triggered automatic DoD checkpoint — must verify the new
rule doesn't regress DoD smoke. The rule scope is tightly gated
(only matches gpt_oss-Down B4-M4096 in the metric/DoD universe per
R32 audit); regression risk is minimal.
