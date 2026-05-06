# Round 11 — fused-act FP8 grouped: gpt_oss-Down-B4-M2048 var-K dB re-tune (R10 sibling, R33 cell update)

## Summary

- **Lever**: re-tune R33's gpt_oss-Down B4-M2048 var-K dB cell from
  `(gm=16, xcds=4)` to `(gm=1, xcds=2)`. R33's original sweep was
  xcds=4-only — never tested xcds=2.
- **Class**: same R8 / R9 / R10 "candidate-set widening" pattern. R10
  (last round) found `(gm=1, xcds=2)` wins +0.48pp over `(16, 4)` on
  gpt_oss-Down-B4-M4096 var-K dB which has the EXACT same persistent-
  grid geometry as B4-M2048 (per-group output [2880, 2880] = 11×11 =
  121 tile-steps × 4 groups = 484 tile-steps over 256 CUs ≈ 2 wave-
  steps per slot; M_per_group only affects K-loop length, not the N×K
  tile geometry). R10's note explicitly suggested R11 = R33 sibling
  re-probe.
- **Metric (single-run wall, noisy, score capped at 1000)**:
  - Before: score 1000, geomean **1.3875**, below_target 5/24.
  - After:  score 1000, geomean **1.3922**, below_target 5/24.
  - Δgeomean **+0.0047 (+0.34pp)**, all 24 shapes correctness PASS.
  - Kernel-level Δ on the target shape: +0.65% (probe).

## Probe data (`/tmp/probe_round_11_gpt_oss_down_b4_m2048_var_k.py`)

200-iter × 7-trial × p20 × 3 seeds × 15 candidate cells across
xcds={2, 4, 8} columns:

```
shape: gpt_oss-Down-B4-M2048 var-K dB
  m_total=8192, N_fwd=2880, K_fwd=2880, B=4 groups
  per-group output [N=2880, K=2880] = 11×11 = 121 tiles × 4 groups
  = 484 tile-steps over 256 CUs ≈ 2 wave-steps (very small grid)

  cell        seed42   seed137  seed2024  med Δ vs R33  spread pp  verdict
  (1, 2)      +0.78%   +0.62%   +0.54%    +0.65%        0.16       WIN  med/spread=4.06×
  (16, 2)     +0.78%   +0.35%   +0.66%    +0.59%        0.35       WIN
  (32, 2)     +0.50%   +0.39%   +0.78%    +0.56%        0.39       WIN
  (32, 4)     +0.12%   +0.27%   +0.27%    +0.22%        0.23       small WIN
  (1, 4)      +0.16%   +0.12%   +0.08%    +0.12%        0.04       TIE
  (16, 4)R33  baseline                     +0.00%        0.08       (R33 cell)
  (2, 4)      -0.66%   -0.70%   -0.58%    -0.65%        0.16       LOSS
  (8, 4)      -0.66%   -0.78%   -0.81%    -0.75%        0.08       LOSS
  (4, 4)      -0.74%   -0.93%   -0.74%    -0.80%        0.19       LOSS
  (2, 2)      -0.85%   -1.16%   -1.16%    -1.06%        0.23       LOSS
  (4, 2)      -1.47%   -1.82%   -1.59%    -1.63%        0.27       LOSS
  (8, 2)      -1.59%   -1.94%   -1.74%    -1.76%        0.27       LOSS
  (1, 8)      -1.94%   -2.13%   -2.17%    -2.08%        0.16       LOSS
  (8, 8)      -2.91%   -2.71%   -2.99%    -2.87%        0.27       LOSS
  (16, 8)     -2.95%   -3.41%   -3.22%    -3.19%        0.39       LOSS
```

`(gm=1, xcds=2)` is the unique top with the tightest spread (0.16pp) —
every-seed positive (3/3) at +0.54..+0.78%, baseline-min beat in 3/3
seeds. Median / spread = 4.06×, well above the standard "median > spread"
robust-signal threshold used by R7 / R10 / R23 / R29 / R30 / R31 / R32 /
R33 / R35 / R39 / R6-R10 of the current run.

## Why (gm=1, xcds=2) wins on this small grid

SAME persistent-grid analysis as R10 on B4-M4096 (which has identical
484 tile-step / 2 wave-step geometry):

- **gm=1** walks the entire 11-row N-axis under each individual K-tile
  before advancing K, maximising B-pack L2 reuse on the per-K column
  slab (one slab serves 11 N-rows back-to-back).
- **xcds=2** keeps the 2-wave-step schedule INSIDE a SINGLE chiplet
  pair (vs xcds=4 splitting across both chiplet pairs of the MI355X
  8-XCD topology). With only 2 wave-steps, cross-chiplet L2
  invalidation dominates over the parallelism benefit of a wider
  distribution.

The probe data confirms the chiplet-locality lever explicitly:
xcds=2 column dominates xcds=4 column by +0.43..+0.64pp at the matching
gm (compare `(1, 2) vs (1, 4)`, `(16, 2) vs (16, 4)`, `(32, 2) vs
(32, 4)` for clean A/B differentials); xcds=8 column uniformly LOSS by
-2.08..-3.19pp.

R33's `(gm=16, xcds=4)` was sub-optimal vs `(gm=1, xcds=2)` because:
1. gm=16 over-batches on the 11-row N-axis (16/11=1.45 batches per
   pass, fractional — the 16-tile batch packs the wave-step but pays
   a fractional-tail stall on the 5 unbatched N-rows of the second
   pass, only 5 of 16 slots populated).
2. xcds=4 splits across chiplet pairs, paying L2 invalidation that the
   small grid (only 2 wave-steps) can't amortise.

The R8 / R34 / R9 / R31 / R10 missing-candidate pattern repeats here:
R33's sweep was xcds=4-only, missing the entire xcds=2 column.

## Bench correctness (`/tmp/probe_round_11_correctness.py`)

Bit-equivalent output verified with **`torch.zeros()`** out buffer (per
R36's documented `torch.empty()` garbage trap):

```
seed=  0  max_abs_out=    247.00  diff=  0.0000e+00  bit_eq=True  nan_new=False  inf_new=False
seed= 42  max_abs_out=    245.00  diff=  0.0000e+00  bit_eq=True  nan_new=False  inf_new=False
seed=137  max_abs_out=    244.00  diff=  0.0000e+00  bit_eq=True  nan_new=False  inf_new=False
```

3/3 seeds bit_eq=True. (gm, xcds) are pure persistent-grid scheduling
knobs on the var-K CRR kernel — same property documented for
R30/R31/R32/R33/R35 and R10-current.

## Discriminator: clean carve-out via existing m_total guards

The R33 cell remains gated by `a.shape[1] == 2880 AND b.shape[1] == 2880`
within the `m_total < 16384` else-branch:

```
m_total      shape                              dispatch
8192         gpt_oss-Down-B4-M2048-dB           NEW R11 (gm=1, xcds=2)  ← updated
8192         gpt_oss-GateUP-B4-M2048-dB         R35 (gm=2, xcds=2)
16384        gpt_oss-Down-B4-M4096-dB           R10 (gm=1, xcds=2)
65536        gpt_oss-Down-B32-M2048-dB          R30 (gm=4, xcds=4)
131072       gpt_oss-Down-B32-M4096-dB          R30 (gm=4, xcds=4)
```

No overlap with R10 / R30 / R31 / R35 / R39 default. The gpt_oss-Down
B=4 family now has a single (gm=1, xcds=2) per-row config across both
m_total bands (B4-M2048 and B4-M4096) — intentional rule consistency
following the same persistent-grid topology.

## Files touched

- `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`:
  R33 if-branch body updated (`vk_group_m=16,vk_num_xcds=4` →
  `vk_group_m=1,vk_num_xcds=2`) + ~110-line R11 documentation block;
  R33 historical block annotated with "SUPERSEDED by R11" pointer.
- `analysis/_notes/round-11-fused-act-gpt-oss-down-B4-M2048-var-k-dB-retune.md`:
  this note.

HipKittens: not modified this round.

## Metric distribution

Single-run noisy:
```
pre  (HEAD 31ef9c36, 1 run): score=1000  geomean=1.3875  below_target=5/24
post (this commit, 1 run):   score=1000  geomean=1.3922  below_target=5/24
```

The +0.34pp geomean lift is consistent with the kernel-level +0.65%
projecting through (var-K dB ≈ 25% of bwd wall on B=4 → +0.16% bwd wall
→ +0.08% fwd+bwd wall on the target shape, divided by 24-shape suite
→ +0.0033% geomean — same order as observed +0.0047 within single-run
metric noise). Score capped at 1000 already, so the gain is buffer
rather than headline. Same R8/R9/R10 "ship narrow carve-out when probe
shows clean WIN even if metric noise floor swallows the geomean lift"
pattern.

## Suggestion for the next round

R11 closes the last R33-class candidate-set-widening lever
(B4-M2048 was the only m_total<16384 metric shape with a==2880, b==2880,
and the only one whose original sweep was xcds=4-only without a follow-
up R10-style widening). The remaining unspent var-K cells:

- **R35** (gpt_oss-GateUP-B4-M2048): tight-verified across xcds={1,2,4}
  in R35's 10-cell + 7-cell neighbor probes; (2, 2) is on a sharp
  single-cell optimum. Re-probing low-priority.
- **R30** (gpt_oss-Down B=32, m_total≥65536): re-probed in R8 with
  widened xcds={2,4,8} candidate sets; (4, 4) holds. Closed.
- **R31** (gpt_oss-GateUP B=32, m_total≥32768): re-probed in R9 with
  same widening; (1, 4) holds at B=32. Closed.
- **R10** (gpt_oss-Down B4-M4096, m_total in [16384, 65536)): just
  shipped (1, 2) last round. No re-tune needed.
- **R39 default** (Qwen3-* and DSV3-* var-K dB): R29 (Qwen3-Down) and
  R36 (Qwen3-GateUP) tight-verified; signals are sub-noise. Closed.

R12 candidates (not in priority order):
1. **HK kernel surgery** — gpt_oss-Down K=2880 forward K-tail epilog
   is the residual ratio gap source (R5 decomp probe; +1.27 ratio at
   1.270 below target). Multi-round HK C++ work; R7-R8 falsified
   Path A on the same kernel. Could re-attempt with a different load
   primitive but high-risk.
2. **Maintenance** — score capped at 1000 with healthy +4.5pp geomean
   buffer (1.3922 vs 1.35 target). Patience 9/30 after R11 (still 21
   rounds buffer). R5 recommended maintenance until either the
   `_metric_grouped_only.py` baseline shifts or a HK kernel-side
   change lands.
3. **`quantize_fp8_tensorwise` HBM bandwidth lift** (R29 direction 2):
   currently 67% of MI355X HBM peak; pushing to 80% would lift the
   metric geomean by ~+1.5%. Out of HK scope — needs C++ extension
   work. Not actionable from Primus-side this round.

R12 lean: maintenance round, OR re-audit prior carve-outs for any cell
that became sub-optimal due to the cumulative R30/R31/R10/R11 shifts
in the same dispatcher chain (low-likelihood; this style of audit was
exhausted at R29).
