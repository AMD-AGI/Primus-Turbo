# Round-15 — gpt_oss FP8 kernel-only ceiling: var-K wgrad fine-slots refinement + (gm, xcds) re-tune at slots=192 grid FALSIFIED

**Date**: 2026-05-08 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD `e4974d28` → R15)
**Scope**: gpt_oss FP8 kernel-only suite, wgrad var-K (CRR / `grouped_var_k_kernel_fp8`)
post the R3 `slots=192` lever shipped in commit `7060820`.
**Goal**: Find a metric-moving lever for the worst gpt_oss wgrad cells
(Down-B4-M2048: 1279 T, Down-B4-M4096: 1676 T) by re-probing along two
axes the R2 / R10 / R11 sweeps did not jointly explore: (a) finer slot
counts than R2's coarse {32, 64, ..., 192, 256} grid, and (b) the
(gm, xcds) cell when slots is pinned to the R3 lever value of 192.

## Bottom line

Both directions **falsified**. The R3 `slots=192` rule is on a sharp
local optimum (cliff between slots=184 and slots=192, smooth fall-off
slots=200..256), and at slots=192 the R10/R11 cell `(gm=1, xcds=2)`
remains the robust top for both Down-B4 wgrad shapes under 1500-iter ×
7-trial × 3-seed tight verify. The borderline R4-noted `(gm=16, xcds=4)`
candidate for Down-B32-M2048 wgrad also re-falsifies under the same
methodology — `(gm=8, xcds=4)` (R1-current) wins by -0.77 % on (16, 4)
and -1.37 % on (4, 4). **No metric-moving rule landed in R15**;
deliverable is the falsification trace + the demonstration that the
slots-axis lever is on a chiplet-aligned cliff (`slots ∈ {64×N}` are
qualitatively different from non-multiples-of-64), which retargets
R16+ at the kernel-side surgery candidates documented in R14 and
R4-prior-task.

| Direction | Anchor shape(s) | Best non-current cell | Δ vs current | Verdict |
|---|---|---|---|---|
| Fine-grained slots sweep | Down-B4-M2048 wgrad (R3 anchor)<br>Down-B4-M4096 wgrad (sibling) | slots=200<br>slots=200 | **-0.13 %** (within ±0.05 pp p20 spread)<br>**-0.54 %** (smooth degradation) | FALSIFIED |
| (gm, xcds) at slots=192 | Down-B4-M2048 wgrad<br>Down-B4-M4096 wgrad | (1, 2) WIN<br>(1, 4) coarse +0.85 % → tight -0.03 % | (R11 at ceiling)<br>**TIE within 0.05 pp** | FALSIFIED |
| Down-B32-M2048 wgrad (gm, xcds) | Down-B32-M2048 wgrad (R1-current cell) | (8, 4) WIN<br>(16, 4) -0.77 %<br>(4, 4) -1.37 % | (R1-current at ceiling) | FALSIFIED |

## Probe protocol

All probes use **kernel-only direct call** to
`hk.grouped_variable_k_crr_dscale` (the same binding entry the var-K
metric path takes via `grouped_gemm_fp8_variable_k_impl`). Three
methodologies layered:

1. **Coarse**: 200-iter × 5-trial p20 × 3-seed (median across seeds).
   Used to scan the slots ∈ {160..256} and (gm, xcds) ∈ ~11 cells.
2. **Tight**: 1500-iter × 7-trial p20 × 3-seed (median across seeds).
   Used to confirm or falsify the top-2 candidates from coarse.
3. **Bit-equivalence**: same documented property as R10/R11/R30/R31
   etc. (`group_m`, `num_xcds`, `num_slots` are all pure persistent-grid
   scheduling knobs on the var-K CRR kernel — see R3 / R30 / R31
   commentary in `grouped_gemm_fp8_impl.py:828-2002`). Numerical
   verification on this round is implicit: every cell in the sweep
   completed without NaN, and the metric's built-in correctness gate
   (SNR > 25 dB on out / dA / dB across all 8 gpt_oss shapes) is
   `0/8 FAIL` on the post-R15 working tree.

Probe scripts:

- `/tmp/_probe_round_15_vark_slots_fine.py` — fine-grained slots sweep
- `/tmp/_probe_round_15_gm_xcds_at_slots_192.py` — (gm, xcds) sweep at slots=192
- `/tmp/_probe_round_15_tight_verify.py` — 1500-iter × 7-trial × 3-seed tight A/B
- `/tmp/_probe_round_15_b32_m2048_wgrad.py` — Down-B32-M2048 wgrad re-verify

## A. Fine-grained slots sweep (R2 follow-up)

R2 swept slots ∈ {32, 64, 96, 128, 160, 192, 256, 384} on Down-B4-M2048
wgrad and identified slots=192 as the +6.14 % winner. The coarse grid
left a 64-slot gap between 192 and 256 unsampled. R15 fills it.

Anchor: Down-B4-M2048 wgrad (m_total=8192, R11 cell `(gm=1, xcds=2)`):

```
slots   TFLOPS   Δ vs 192
  160    1176    -23.42 %    (cliff: kernel under-saturates at < 6 chiplet pairs)
  176    1178    -23.25 %    (cliff)
  184    1179    -23.08 %    (cliff)
  192    1452     +0.00      *baseline (R3 rule, 6 of 8 chiplets)
  200    1450     -0.13 %    (within p20 spread of 0.05 pp)
  208    1438     -0.94 %
  216    1430     -1.54 %
  224    1417     -2.44 %
  240    1389     -4.53 %
  256    1376     -5.51 %    (R3 baseline before lever; R2 reported here)
```

Sibling: Down-B4-M4096 wgrad (m_total=16384, R10 cell `(gm=1, xcds=2)`):

```
slots   TFLOPS   Δ vs 192
  160    1481    -24.97 %    (cliff)
  176    1487    -24.43 %    (cliff)
  184    1487    -24.48 %    (cliff)
  192    1850     +0.00      *baseline (R3 rule)
  200    1840     -0.54 %
  208    1821    -1.63 %
  216    1810    -2.26 %
  224    1792    -3.24 %
  240    1753    -5.55 %
  256    1725    -7.24 %
```

### Reading

1. **Sharp cliff at slots=192**. Both shapes show 23-25 % degradation
   for slots ∈ {160, 176, 184} and a ~+18 % jump at slots=192. The
   cliff aligns with the kernel's `chiplet_transform_chunked(blockIdx.x,
   slots_eff, xcds_eff, 64)` swizzle (line 7793-7794 in
   `kernel_fp8_layouts.cpp`): chunk_size=64 means slots=192 = 3×64
   gives clean 3-chiplet-pair alignment. slots=160 = 2.5×64 leaves a
   fractional chunk that the swizzle treats as a half-populated 4th
   chunk, which serializes against the 3 full chunks. slots ∈
   {200, 208, 216, 224} are also non-multiples of 64 but degrade
   smoothly because the smooth-region cost dominates the chunk-tail
   cost when slots > 192.
2. **slots=192 is the unique sharp peak**. No fine-grained slots value
   beats it within the metric's noise floor (single-trial p20 spread
   ~0.05 pp at 1500 iters). The `slots ∈ {64, 128, 192, 256}`
   chiplet-aligned set has slots=192 as the unique top for Down-B4
   wgrad (slots ∈ {64, 128} are far below — 484 / 64 ≈ 7.5
   wave-steps/slot but the chiplet pair has too many sequential tiles
   per slot, losing parallelism).
3. **The R3 rule cannot be refined along the slots axis**. Any future
   slots tuning would require either (a) a kernel change to the
   `chiplet_transform_chunked` chunk_size (currently hard-coded
   64; line 7793-7794) or (b) a different chunking scheme that
   amortises the per-chunk cost differently for short grids.

## B. (gm, xcds) re-tune at slots=192 (R10 / R11 follow-up)

R10 / R11 swept `(gm, xcds) × slots=256` and picked `(gm=1, xcds=2)` as
the joint best for Down-B4 wgrad. With the R3 lever now active
(slots=192), the persistent grid topology has changed (484 / 192 ≈
2.52 wave-steps/slot vs 484 / 256 ≈ 1.89). R15 re-sweeps the cell
on the actual production grid.

### Coarse 200-iter × 5-trial × 3-seed sweep at slots=192

Down-B4-M2048 wgrad (the `(1, 2)` cell wins):

```
cfg          med ms     TFLOPS    Δ vs (1, 2)
( 1,  2)     0.0913     1488.8    +0.00 %    *cur (R11)
( 2,  2)     0.0914     1487.5    -0.09 %
(16,  2)     0.0914     1486.2    -0.18 %
(32,  2)     0.0915     1485.5    -0.22 %
( 8,  2)     0.0916     1483.6    -0.35 %
( 4,  2)     0.0917     1482.3    -0.44 %
( 1,  4)     0.0917     1481.6    -0.48 %
( 1,  8)     0.0922     1473.9    -1.01 %
( 1, 16)     0.0924     1470.1    -1.27 %
(16,  4)     0.0926     1467.5    -1.45 %
(32,  4)     0.0926     1466.9    -1.49 %
```

Down-B4-M4096 wgrad (coarse `(1, 4)` lead → tight TIE):

```
cfg          med ms     TFLOPS    Δ vs (1, 2)
( 1,  4)     0.1441     1885.8    +0.85 %    coarse TOP (falsified at tight)
( 1, 16)     0.1442     1884.8    +0.80 %
( 1,  8)     0.1442     1884.3    +0.77 %
( 8,  2)     0.1451     1872.9    +0.17 %
(32,  2)     0.1452     1872.3    +0.14 %
(32,  4)     0.1452     1872.3    +0.14 %
( 4,  2)     0.1452     1871.3    +0.08 %
( 1,  2)     0.1454     1869.8    +0.00 %    *cur (R10)
(16,  4)     0.1454     1869.3    -0.03 %
(16,  2)     0.1458     1864.6    -0.27 %
( 2,  2)     0.1459     1862.6    -0.38 %
```

### Tight 1500-iter × 7-trial × 3-seed verify

The Down-B4-M4096 coarse hint of `(1, 4)` +0.85 % was the only
candidate that crossed the 0.5 pp signal threshold. Tight verify
falsifies:

```
shape: Down-B4-M4096 wgrad var-K @ slots=192
  cfg          seed42   seed137   seed2024   med ms    spread pp   TF
  ( 1,  2)     0.1466   0.1472    0.1466     0.1466    0.410       1854.4   *cur
  ( 1,  4)     0.1466   0.1466    0.1466     0.1466    0.055       1853.9    -0.03 %  (TIE)
  ( 1,  8)     0.1466   0.1466    0.1467     0.1466    0.108       1853.4    -0.05 %  (TIE)

shape: Down-B4-M2048 wgrad var-K @ slots=192
  cfg          med ms    spread pp   TF        Δ vs (1, 2)
  ( 1,  2)     0.0934    0.044       1455.6    +0.00 %    *cur (R11)
  ( 1,  4)     0.0952    0.547       1428.1    -1.93 %    (clear LOSS)
```

### Reading

1. **R10/R11 `(gm=1, xcds=2)` confirmed at ceiling at slots=192**. The
   coarse `(1, 4)` lead on M=4096 (+0.85 %) was within run-to-run
   noise of the 200-iter measurement; tight 1500-iter × 7-trial settles
   it at -0.03 % TIE. Same shape's M=2048 sibling has `(1, 4)` clearly
   losing -1.93 %, so even if the M=4096 cell were a robust win the
   rule split would have to gate on `m_total == 16384`, making the
   per-rule lift smaller than the metric's 1-2 pt noise floor.
2. **The R10 / R11 picked cells generalize across the slots-256 →
   slots-192 grid switch**. This is the same property documented for
   R30 / R31 ("`group_m / num_xcds` are pure persistent-grid scheduling
   knobs"); the relative ranking among (gm, xcds) cells is preserved
   when `slots_eff` changes. The +0.85 % coarse blip on M=4096 was
   noise, not a real grid-induced re-optimum.

## C. Down-B32-M2048 wgrad (R1-current re-verify)

R4 audit flagged `(gm=16, xcds=4)` as a +0.27 % NOT-ROBUST candidate
for Down-B32-M2048 wgrad (50T spread). R1-current chose `(gm=8, xcds=4)`
post the R3-fused-act commit `ceb7e93` rebuild. R15 re-tests at
1500-iter × 7-trial × 3-seed:

```
shape: Down-B32-M2048 wgrad @ slots=0 (default 256), m_total=65536
  cfg          med ms    spread pp   TF        Δ vs (8, 4)
  ( 8,  4)     0.6495    0.721       1673.8    +0.00 %    *R1-current
  (16,  4)     0.6545    0.501       1661.0    -0.77 %
  ( 4,  4)     0.6584    0.298       1651.1    -1.37 %    (R30 original)
  (12,  4)     0.6557    0.634       1658.1    -0.95 %
  ( 8,  8)     0.6690    1.363       1625.0    -3.00 %    (default xcds)
  ( 8,  2)     0.6584    1.057       1651.1    -1.37 %
```

### Reading

1. **`(gm=8, xcds=4)` is the unique top** for Down-B32-M2048 wgrad on
   today's binding. R4 audit's `(16, 4)` +0.27 % coarse hint is
   falsified at tight verify (-0.77 % vs current). All probed
   alternatives in the (gm=4..16) × (xcds=2..8) cell lose by -0.77 %
   to -3.00 %.
2. **R1-current rule split (`gm=8` for `m_total==65536` vs `gm=4`
   for `m_total==131072`) confirmed**. The R30 universal `(gm=4,
   xcds=4)` rule it replaced now loses -1.37 % on M=2048; both shapes
   get the right cell under the R1-current m_total tiering.

## What this rules out for the wgrad section

- **Slots-axis refinement**: The R3 `slots=192` lever sits on a
  chiplet-aligned cliff (slots = 64 × 3); no in-range slots value
  beats it. Future slots tuning requires a kernel change to the
  `chiplet_transform_chunked` chunk_size or a different chunking
  scheme. The slots ∈ {32 × 5 = 160, 32 × 7 = 224} candidates were
  also tested coarse and rejected.
- **(gm, xcds) re-tune at the R3 grid**: R10 / R11 cells transfer
  cleanly from slots=256 to slots=192. The `(gm=1, xcds=2)` choice
  is robust across both grids on both Down-B4 wgrad shapes.
- **Down-B32 wgrad family re-tune**: R1-current is at ceiling. The
  R4-noted alternatives all regress by -0.77 % to -3.00 % at tight
  verify.

## What's left (R16+ candidates)

The dispatcher levers (group_m, num_xcds, num_slots) for the var-K
wgrad path are now jointly exhausted across all 8 gpt_oss wgrad
shapes (R3 / R10 / R11 / R30 / R31 / R1-current / R15). Per
R4-prior-task and R14 PMC analysis, the next score lift requires:

1. **Per-tile prologue trim in `grouped_var_k_kernel_fp8`** —
   `kernel_fp8_layouts.cpp:7716-7783` (LDS group-metadata init,
   scale fetch, swizzle offset prefill). For Down-B4-M2048 wgrad
   (484 tile-steps / 192 slots ≈ 2.5 wave-steps/slot), the per-tile
   prologue is amortised over only ~16 K-blocks of MFMA work. Trim
   estimate: -10..-30 cycles per tile prologue, ~+1..+3 % on under-
   amortised wgrad shapes. Cost: 1-2 rounds of HK kernel surgery +
   bit-equivalence probe.
2. **`KI_HINT=22` specialization for `grouped_var_k_kernel_fp8`** —
   currently a single `<KI_HINT=0>` instantiation
   (`kernel_fp8_layouts.cpp:8100`). Adding `<KI_HINT=22>` (=
   K_fwd=2880 / 128 = 22.5 K-blocks for gpt_oss) lets the compiler
   unroll the K main-loop and reduce control-flow overhead. Same
   pattern as the dense `gemm_kernel<L,KI_HINT>` (R55-era).
3. **K-tail fuse runtime gate** (R14 candidate) — the
   `FUSED_KTAIL=true` template adds register pressure that doesn't
   pay off on under-saturated grids. A runtime gate keyed off
   `wave_steps_per_cu < 3` (computable host-side) could ship the
   unfused path on Down-B4 family without regressing the saturated
   GateUP-B32 / Down-B32 family.

All three are kernel-source changes in `HipKittens/analysis/fp8_gemm/
mi350x/kernel_fp8_layouts.cpp`, not Primus dispatcher changes.

## R15 deliverables

### Primus-Turbo

- `/tmp/_probe_round_15_vark_slots_fine.py` — fine-grained slots sweep driver
- `/tmp/_probe_round_15_gm_xcds_at_slots_192.py` — (gm, xcds) sweep at slots=192 driver
- `/tmp/_probe_round_15_tight_verify.py` — 1500-iter × 7-trial tight A/B verify
- `/tmp/_probe_round_15_b32_m2048_wgrad.py` — Down-B32-M2048 wgrad re-verify
- `analysis/_notes/round-15-fp8-vark-wgrad-fine-slots-and-gm-xcds-at-r3-grid-falsified.md` (this file)
- **No `select_default_config` change.** **No `grouped_gemm_fp8_impl.py` change.**
  Metric unchanged at 680 (3-run median).

### HipKittens

- No change. (R3 already shipped the var-K kernel side `num_slots`
  plumbing; R15 just confirms the slots-axis is exhausted at the
  current `chunk_size=64` chiplet swizzle.)

## R16 plan

Pivot to **`KI_HINT=22` specialization for `grouped_var_k_kernel_fp8`**
(R16+ candidate #2 above). Concrete steps:

1. Locate the `grouped_var_k_kernel_fp8<KI_HINT>` template definition
   (`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:7764`)
   and the single `<0>` instantiation at line 8100.
2. Add a `<22>` instantiation (K_fwd / 128 = 22 main-loop iters for
   gpt_oss-Down K=2880 var-K wgrad; the `K_fwd=2880` axis is the var-K
   kernel's K-axis after CRR transpose).
3. Update `grouped_var_k_layout_globals_fp8` dispatch to pick the
   specialised template when `g.k == 22 * K_BLOCK == 2816` AND the
   K-tail fits the unrolled body. K=2880 has K_REM=64 which still
   needs the masked tail handler — verify the unrolled body
   composes correctly with the tail.
4. Bit-equivalence probe (verify max_abs_diff=0 between `<0>` and
   `<22>` on Down-B4-M2048 wgrad).
5. Tight 1500-iter × 7-trial × 3-seed verify on the 4 gpt_oss-Down
   wgrad shapes (B=4 + B=32, M=2048 + M=4096). Estimated lift +1-3 %
   per shape on the under-saturated subset.

If `<22>` does not produce a measurable kernel-only TFLOPS lift,
falsify and pivot to candidate #1 (per-tile prologue trim).
