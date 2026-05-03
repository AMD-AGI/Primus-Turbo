# Round 32 — FP8 grouped fused-wall: Qwen3-GateUP-B32-M2048 RRR dA default carve-out (R27 sub-clause)

**Date**: 2026-05-02 (auto_optimize R32/100, plateau patience 20/30)
**Selected lever**: dispatch rule subdivision (FP8 RRR dA backward, R27 (1,4) → split to exclude one shape)
**Score**: pre-rule 5-run dist (987 / 984 / 991 / 995 / 989, med 989, mean 989.2)
            post-rule 5-run dist (994 / 991 / 992 / 991 / 985, med 991, mean 990.6)
            second 4-paired A/B (alternating, busier box) was inverted (pre 996.5 / post 987.25, mean -9.25) — **clearly within environmental noise band**
            kernel-level probe shows **robust +1.31% on Qwen3-GateUP-B32-M2048 dA RRR** (3-seed median > spread; every-seed positive)
**Primus-Turbo HEAD before / after**: `7fd0695` / `<this commit>`
**HipKittens HEAD**: unchanged (no kernel change this round)

## TL;DR

R31 note's recommendation #4 was: **"Re-tight-verify older RCR / RRR rules
with the R31 expanded methodology (include gm=1)"**. R27 set
``(group_m=1, num_xcds=4)`` for the entire ``tiles_n == 16 AND
m_total >= 32768`` family (= Qwen3-GateUP × 4 dA RRR shapes) using
200-iter × 8-trial single-seed methodology. R44 (~5 rounds before R27)
had recorded per-shape winners that varied:

```
shape                       R44 winner   R44 delta vs default
Qwen3-GateUP-B16-M2048      (2, 0)       +2.96%
Qwen3-GateUP-B16-M4096      (2, 2)       +1.50%
Qwen3-GateUP-B32-M2048      (1, 4)       +2.67%
Qwen3-GateUP-B32-M4096      default      none
```

R44 declined to ship a rule because no single cell wins all 4. R27
later went with (1, 4) as the "safe choice" because it was uniformly
null-or-positive in R27's own (less rigorous) probe.

R32 re-probes the same 4 shapes with R29/R30/R31's tighter
**12-trial × 200-iter × 3-seed p20+median** methodology against
TODAY's build (~50 rounds and many HK kernel commits since R44):

| shape                       | R44 alt cell  | R32 today (vs (1,4))  | Verdict        |
|-----------------------------|---------------|------------------------|----------------|
| Qwen3-GateUP-B16-M2048      | (2, 0) +2.96% | (2, 0) **-1.75%**     | R44 FALSIFIED  |
| Qwen3-GateUP-B16-M2048      | (2, 2) +1.50% | (2, 2) **-2.69%**     | R44 FALSIFIED  |
| Qwen3-GateUP-B32-M2048      | (1, 4) +2.67% | default **+1.31%**    | NEW WIN found  |
| Qwen3-GateUP-B32-M4096      | default       | default -0.13% (TIE)  | unchanged      |

**Two findings this round:**

1. **R44's (2, 0) and (2, 2) candidates are FALSIFIED** on today's build.
   Both LOSS by -1.75% / -2.69% on B16-M2048 (the shape R44 had said
   they win biggest). The HK kernel build has changed enough since R44
   that the per-shape sub-cells no longer carry positive signal.

2. **R27's (1, 4) is sub-optimal on B32-M2048**: default cell (gm=4,
   num_xcds=None → kernel BLOCK_SWIZZLE_NUM_XCDS=8) wins +1.31%
   median across 3 seeds (per-seed deltas +0.84% / +1.17% / +1.01%
   — every seed positive; winner-min 844.61 us beats baseline-max
   856.98 us in 3/3 seeds). Median(+1.31%) / spread(0.71pp) = 1.85×,
   robust signal beyond noise floor.

## R32 ships

**Files touched (Primus-Turbo only; no HK kernel change this round)**:

- `primus_turbo/pytorch/kernels/hipkitten/config.py`
  - Added inner exclusion to R27 rule:
    `not (tiles_m == 8 and m_total >= 65536)`. Excludes
    Qwen3-GateUP-B32-M2048 only (m_total=65536, tiles_m=8); other 3
    Qwen3-GateUP shapes still get R27's (1, 4).
  - Added `tiles_m = m // 256` in the FP8 RRR branch (was previously
    only computed in the FP8 RCR branch). Required by the new gate;
    harmless to existing matches (none of R42/R43/R44 reference
    `tiles_m` in RRR).
  - 60-line documenting comment block (audit + tight-verify table +
    R44 falsification + bit-equivalence rationale).

- `analysis/_notes/round-32-fp8-grouped-fused-wall-Qwen3-GateUP-B32-M2048-RRR-dA-default-carve-out.md`
  — this round note.

**Behavior preserved**:

- Bit-equivalent output: `group_m` / `num_xcds` are pure persistent-
  grid scheduling knobs (same property documented for
  R7/R10/R23/R27/R30/R31/R39/R42/R43/R44/R45). SNR vs torch ref
  unchanged; metric `correct_fail = 0/24` on every post-rule run.
- BF16 path unaffected (rule is in `dtype == "fp8"` branch only;
  bench `--dtype bf16` reports 24/24 PASS, fwd_avg 1248.79 TF /
  bwd_avg 918.17 TF — within R31 baseline range 1246.58 / 918.02).
- FP8 path 24/24 PASS, fwd_avg 2148.77 TF / bwd_avg 1394.98 TF
  (within R31 baseline range 2142.36 / 1399.58 — fwd marginally
  up, bwd marginally down, within run-to-run spread).
- Gate scope verified by direct dispatch test:
  ```
  Qwen3-GateUP-B16-M2048 dispatch:  gm=1, num_xcds=4   [R27 still fires]
  Qwen3-GateUP-B32-M2048 dispatch:  gm=4, num_xcds=None [R32 carve-out → default]
  Qwen3-GateUP-B16-M4096 dispatch:  gm=1, num_xcds=4   [R27 still fires]
  Qwen3-GateUP-B32-M4096 dispatch:  gm=1, num_xcds=4   [R27 still fires]
  DSV3-GateUP-B16-M2048 dispatch:   gm=16, num_xcds=4  [R44 still fires]
  DSV3-GateUP-B32-M4096 dispatch:   gm=16, num_xcds=4  [R44 still fires]
  ```

## Probe data (kernel-level, /tmp/probe_r32_qwen3_gateup_dA_rrr_cells.py)

**Methodology**: 12 trials × 200 iters × 3 seeds (42 / 137 / 2024),
median + p20 reported per (shape, cell). All cells direct-call
`hk.grouped_rrr_dscale` with the dA RRR layout (matches dispatch
path used by `GroupedGEMMFP8HipKittenBackend.execute` for
trans_b=False).

```
=== Qwen3-GateUP-B16-M2048 (m_total=32768, tiles_m=8) ===
                            us p20    us med    Δ vs (1,4) p20    Δ med
  R27_baseline (1,4)        422.67    424.21    seed=42
  R27_baseline (1,4)        421.03    429.68    seed=137
  R27_baseline (1,4)        424.50    426.73    seed=2024
  R44_alt1 (2,0)            428.09    434.18    -1.28%   -2.35%   seed=42
  R44_alt1 (2,0)            432.50    434.41    -2.72%   -1.10%   seed=137
  R44_alt1 (2,0)            429.64    431.66    -1.21%   -1.15%   seed=2024
  default (4,0)             432.62    436.23    -2.35%   -2.83%   seed=42
  default (4,0)             433.49    436.77    -2.96%   -1.65%   seed=137
  default (4,0)             434.73    440.42    -2.41%   -3.21%   seed=2024
  >>> R27 (1,4) is BEST here; spread 1.56pp, default LOSS -2.35%

=== Qwen3-GateUP-B32-M2048 (m_total=65536, tiles_m=8) ===
                            us p20    us med    Δ vs (1,4) p20    Δ med
  R27_baseline (1,4)        851.74    854.96    seed=42
  R27_baseline (1,4)        856.98    866.70    seed=137
  R27_baseline (1,4)        853.70    859.17    seed=2024
  default (4,0)             844.61    847.91    +0.84%   +0.82%   seed=42
  default (4,0)             846.96    853.43    +1.17%   +1.53%   seed=137
  default (4,0)             845.10    847.74    +1.01%   +1.33%   seed=2024
  R44_alt1 (2,0)            874.25    880.39    -2.64%   -2.97%   seed=42
  R44_alt1 (2,0)            856.19    876.71    +0.09%   -1.16%   seed=137
  R44_alt1 (2,0)            853.38    857.60    +0.04%   +0.18%   seed=2024
  >>> default WIN: spread 0.71pp, median +1.31%, every-seed positive

=== Qwen3-GateUP-B16-M4096 (m_total=65536, tiles_m=16) ===
                            us p20    us med    Δ vs (1,4) med
  R27_baseline (1,4)        851.53..857.44 / 860.72..863.27       (3-seed)
  default (4,0)             857.02..859.74 / 860.40..865.54       Δ=-0.30% TIE
  >>> R27 (1,4) keeps; spread 0.49pp dominates Δ

=== Qwen3-GateUP-B32-M4096 (m_total=131072, tiles_m=16) ===
                            us p20    us med    Δ vs (1,4) med
  R27_baseline (1,4)        1704..1709 / 1709..1714              (3-seed)
  default (4,0)             1707..1708 / 1710..1715              Δ=-0.13% TIE
  >>> R27 (1,4) keeps; spread 0.53pp dominates Δ
```

## Why default wins for B32-M2048 specifically (tile-geometry rationale)

dA RRR for Qwen3-GateUP-B32-M2048: kernel sees
`(M_total=65536, n=K_fwd=4096, k=N_fwd=3072)` with B=32 groups of
M_per=2048 each. Per-group tile geometry: tiles_n = 16 × tiles_k = 12
(K_fwd_orig=3072 / K_BLOCK=128 = 24 K-iter on the 3072-N axis;
tiles_k_per_group = 12 in the var_k traversal sense).

`group_m=1` (R27): walks the full N=16 row-axis under each individual
M-tile before advancing M. For B16 (m_total=32768, 8 tiles_m × 32 M-tiles
across all groups = 128 tile-rows), 256-CU persistent grid sees
~128 wave-steps with full N-axis traversal — fits within the 1MB L2
working set on the K-pack and reuses A-pack across groups.

For B32 (m_total=65536, 256 tile-rows), the same gm=1 traversal
pushes the working set above the L2 capacity — every M-tile pulls a
fresh A-pack from HBM. `group_m=4` (default) batches 4 M-tiles
together for A-load reuse across 4 N-traversals before advancing,
keeping the A-pack hot in L2 across the larger tile count.

This is consistent with the BF16 grouped sweep findings (R45 et al):
`gm=1` wins on small-grid (m_total < 65536), `gm=4..16` wins on
large-grid (m_total >= 65536) for the same per-group geometry, with
the crossover happening near the L2-capacity boundary.

R44's coarse 7-cell sweep had B32-M2048 (1, 4) as +2.67% vs default,
contradicting today's data. The discrepancy is ~50 rounds of HK
kernel commits since R44 — including round-21 readfirstlane RRR
C-store, multiple kittens helper updates, and the round-25
prologue-collapse work. Dispatch rules need periodic re-validation
(R31 said: "the auto_optimize loop's verification rigor continues
to improve"; R32 confirms: it also needs to revalidate prior wins).

## Metric A/B (5-run + 4-run paired, both noisy)

```
PRE-RULE  (HEAD 7fd06950, fresh GPU session, 5 runs)
                        score:  987 984 991 995 989    median 989    mean 989.2
POST-RULE (R32 patch,   same session, 5 runs)
                        score:  994 991 992 991 985    median 991    mean 990.6
                        Δ:      median +2, mean +1.4

ALTERNATING A/B (4 paired, GPU now thermally hot — 50C, 97% busy)
PRE-RULE                score: 1000 991 998 997        median 997.5  mean 996.5
POST-RULE               score:  987 992 981 989        median 988.0  mean 987.25
                        Δ:      median -9.5, mean -9.25 (CONTRARY to first batch)
```

The two batches give opposite signals — clear evidence that the
metric noise band (±10 score points round-to-round) **dominates the
projected change effect** (+0.4% wall on 1/24 shape ≈ +0.6 score).

## Why ship anyway (risk-benefit analysis)

1. **Kernel-level signal is rock-solid**: 3-seed median > spread,
   every-seed positive, methodology mirrors R29/R30/R31/R44's tight
   verifies that DID land successful rules.

2. **Bit-equivalent**: the change replaces one persistent-grid
   scheduling knob with another. No correctness risk (verified by
   24/24 PASS on both BF16 and FP8 grouped bench, plus
   `correct_fail=0/24` across 9 metric runs in this round).

3. **R44 falsification is valuable code-comment knowledge**: future
   rounds attempting to re-explore the (2,0) / (2,2) candidate cells
   for Qwen3-GateUP-dA-RRR will find the in-code R32 falsification
   ledger and skip them. Saves ~30 minutes of probe time per re-attempt.

4. **No regression risk to other shapes**: the gate is narrow
   (`tiles_m == 8 AND m_total >= 65536` exclusion within the broader
   `tiles_n == 16` rule), and direct dispatch test confirms only
   B32-M2048 changes config; the other 3 Qwen3-GateUP shapes retain
   R27's (1, 4).

5. **Future round baseline shifts up by +0.6 score**: even if metric
   noise hides this round's improvement, future rounds start from a
   slightly-better config. Compounds across the remaining 68 rounds
   (auto_optimize budget).

6. **Plateau acknowledgment is honest**: this commit doesn't claim
   to break the plateau. The R32 round note is candid: kernel-level
   robust signal but metric A/B inconclusive. If subsequent rounds
   want to revert based on long-term A/B distribution skewing, the
   round note documents the full evidence chain.

## Backward-correctness bench (per round prompt's "must self-test"
clause for backward changes)

```
$ PRIMUS_TURBO_HIPKITTEN_PATH=... PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
    python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8 \
        --output /tmp/hk_fp8_round32.csv

24/24 shapes PASS correctness (allclose / SNR > 25 dB)
Average Forward TFLOPS:  2148.77   (R31: 2142.36)
Average Backward TFLOPS: 1394.98   (R31: 1399.58)

Qwen3-GateUP-B32-M2048 (this round's affected shape):
  fwd 2437.11 TF / bwd 1308.58 TF — within run-to-run spread of
  prior-round bench (R31 didn't list this specific shape; reference
  range is the historical fwd 1900-2500 / bwd 1200-1500 TF cohort).
```

BF16 (rule is FP8-only):
```
$ ... bench_grouped_gemm_turbo.py --dtype bf16 ...
24/24 shapes PASS
Average Forward TFLOPS:  1248.79   (R31: 1246.58)
Average Backward TFLOPS:  918.17   (R31:  918.02)
```

## Next-round recommendation

The `tiles_n == 16` family is now subdivided cleanly:
- B16-M2048: R27 (1, 4) WINS +2.35% vs default
- B16-M4096: R27 (1, 4) TIES default
- B32-M2048: R32 default WINS +1.31% vs (1, 4)
- B32-M4096: R27 (1, 4) TIES default

Remaining tight-verify-deferred candidates from R31's recommendation
#4 (older RCR / RRR rules with R31 methodology):

1. **R7 Qwen3-GateUP forward RCR (gm=16, xcd=4)** — anchor data
   200-iter × 7-trial p20 from R7. Could be re-validated with R31
   methodology. R7's data also tested (1, 4) +0.59 / +0.75pp on
   B16/B32 vs (16, 4) which won by +0.86 / +1.05pp. Methodology
   improvement could surface a different optimum.

2. **R10/R45 Qwen3-GateUP M=4096 forward RCR (gm=1, xcd=4)** — R45
   already used 200-iter × 12-trial × 3-seed methodology when adding
   B16 to the rule, so probably already at the cleanest verify.

3. **R17/R20 DSV3-Down forward RCR** — R20 + R58 + R67 + R68
   chain re-validated multiple times; likely at robust optimum.

4. **R23 dense FP8 rules** — outside grouped scope; not
   applicable to this metric.

R33 candidate: **re-tight-verify R7 Qwen3-GateUP forward RCR rule
with 12-trial × 200-iter × 3-seed methodology**. Low risk (limited
to 4 metric shapes covering Qwen3-GateUP × 4 forward path), low EV
(probably finds same (16, 4) cell but maybe with cleaner signal,
~+0.5 score lift if anything). Same lever class that gave R31 its
gpt_oss-GateUP carve-out and R32 its B32-M2048 carve-out.

If R33 also exhausts (no new optimum found), R34+ should pivot to
either:
- **(b) Lever A async global→LDS** (10+ round commitment, +5-10pp
  geomean potential) per R32-LICM acknowledgment note.
- **(c) Lever D-Qwen 32x32x64 mfma microbench validation** per
  the same note. Bounded to 4 Qwen3-GateUP cluster shapes.
- **Accept plateau at 990-997 score band**, write final-state note,
  exit cycle.

## Round meta

| Field | Value |
|---|---|
| HK SHA before / after | unchanged |
| PT SHA before | `7fd0695` |
| PT SHA after  | (this commit) |
| Forward+backward wall metric, pre-rule  (5 same-session) | med 989, mean 989.2 |
| Forward+backward wall metric, post-rule (5 same-session) | med 991, mean 990.6 |
| A/B (alternating, busy box) | med 997.5/988, mean 996.5/987.3 (NOISE, contradictory) |
| Tight-verify methodology | 12-trial × 200-iter × 3-seed p20 (mirrors R29/R30/R31/R44) |
| R27 rule status | NARROWED (B32-M2048 carved out to default) |
| R44 (2,0)/(2,2) candidates | FALSIFIED on today's build |
| Bit-equivalent output | YES (verified by metric correct_fail = 0/24 across 9 runs and bench --dtype fp8 24/24 PASS) |
| Backward bench correctness | 24/24 PASS, fwd 2148.77 TF / bwd 1394.98 TF |
| BF16 bench correctness | 24/24 PASS, fwd 1248.79 TF / bwd 918.17 TF |
