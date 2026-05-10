round-54-A3-decoupled-warps-PRE-IMPLEMENTATION-FALSIFIED-existing-CDNA4-PC-paper-data-shows-17-44pct-loss-AND-no-256x256-prototype.md
=============================================================================

Round: 54 / 100
Date: 2026-05-10
Pre-SHA: e65c5f34 (R53 docs — A3 preflight greenlight)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R53 greenlit a 4-6 round implementation arc to port the BF16
producer-consumer (PC) prototype to FP8 var-K wgrad, projecting +22 to
+44 score. R54 reviewed two pieces of **already-existing CDNA4 evidence
that R53 did not consult** and finds the EV projection is **NEV-negative
to NEV-zero**:

  1. **Existing CDNA4 BF16 PC paper data shows PC LOSING** to the
     100%-consumer (= regular GEMM) baseline by **17 % to 44 %** across
     all measured tile sizes (3072³, 7680², 9216²) and both 8c4p and
     12c4p wave splits.
  2. **No 256×256-tile PC prototype exists** in the HK tree. All PC
     variants are 128×256 or 192×256. Production FP8 var-K is 256×256.
     Porting PC at the existing prototype tile sizes is independently
     blocked by FORBIDDEN PATHS (256×128 asymmetric tile = 7-23 % slower
     than production per task md line 132). Building a new 256×256 PC
     variant ≠ "no greenfield kernel-architecture work required" (R53
     inventory item #1 was structural-existence only, not perf-validity).

A3 (decoupled-warps) is therefore **PRE-IMPLEMENTATION FALSIFIED** at
the EV-bound level. R55 forward-pointer = Direction G (cross-shape
co-optimization, the last untried task-md direction) OR a simpler
issue-rate attack that does not restructure the wave layout.

## Evidence #1 — Existing CDNA4 PC paper benchmarks (pc_micro.png)

Source: `HipKittens/analysis/paper_experiments/producer_consumer_micro/plot.py`

This is the canonical paper-experiment data for the PC prototype family
that R53 cited. All measurements are CDNA4-targeted (verified via the
prototype Makefile at line 27-28: `--offload-arch=gfx950
-DKITTENS_CDNA4`), so they are directly relevant to the MI355X target.

The plot compares 67 % / 75 % / 100 % "consumer worker" fractions:
  * 67 % = 8c4p (4 prod + 8 cons = 12 waves)  ← R53's target prototype
  * 75 % = 12c4p (4 prod + 12 cons = 16 waves)  ← above SW occupancy limit anyway
  * 100 % = no producers (regular 8-wave ping-pong GEMM)

| Problem (M×N×K)  | 67 % (8c4p) | 75 % (12c4p) | 100 % (regular) | Δ vs 100 % @ 67 % | Δ vs 100 % @ 75 % |
|------------------|-------------|--------------|-----------------|-------------------|-------------------|
| 3072³            | 453.65      | 683.61       | 808.16          | **−44 %**         | **−15 %**         |
| 3072×4096×4096   | 708.06      | 1006.45      | 1102.71         | **−36 %**         | **−9 %**          |
| 7680³            | 872.48      | 1128.48      | 1443.73         | **−40 %**         | **−22 %**         |
| 7680×8192×8192   | 964.00      | 1251.18      | 1521.94         | **−37 %**         | **−18 %**         |
| 9216×8192×8192   | 1010.59     | 1229.37      | 1490.14         | **−32 %**         | **−18 %**         |
| 9216³            | 900.17      | 1129.15      | 1418.23         | **−37 %**         | **−20 %**         |

Mean across 6 problems:  **−38 % @ 67 %**, **−17 % @ 75 %**.

R53's projected lift from PC port = **+5 to +10 % section-mean wgrad
TFLOPS**. The actual measured BF16 PC lift on CDNA4 across 6 sizes =
**−17 % to −44 %**. For FP8 var-K to invert this, FP8 must be
structurally different in a way that overcomes a **22-49 percentage-point
swing** beyond the BF16 baseline.

R53's argument for FP8 favorability ("8× more K reduction per
tile-load") is also numerically wrong: BF16 BK=32 vs FP8 BK=128 → **4×
more K-reduction per LDS-tile-load**, not 8×. Even with this factored
in, half-cancelling the BF16 deficit puts expected FP8 lift in the
**−10 % to +0 %** range — well below R53's +5 to +10 %.

## Evidence #2 — No 256×256 PC prototype exists

Source: `HipKittens/kernels/gemm/bf16fp32/micros/producer_consumer/32x16/README.md`

| Micro | Output Tile | Wave Split | Status |
|-------|-------------|------------|--------|
| micro_02_2stage_8c4p   | **128×256** | 8c4p   | shippable |
| micro_03_3stage_8c4p   | **128×256** | 8c4p   | shippable |
| micro_04_2stage_12c4p  | **192×256** | 12c4p  | "above SW limit" per pc_micro plot (= 16 waves > occupancy budget for production grids) |
| micro_05_2stage_16c2p  | 128×256     | 16c2p  | "Above SW limit" (in archive) |
| micro_06_2stage_8c4p_64x96  | 128×256 | 8c4p | scratch/spills (in archive) |
| micro_06_2stage_8c4p_96x64  | 128×256 | 8c4p | scratch/spills (in archive) |
| micro_07_2stage_8c4p_nblock8 | 128×512 | 8c4p | scratch/spills (in archive) |

**Production FP8 var-K uses 256×256** (`ST_v2a` / `ST_v2` rows×cols =
256×128 each, 2 sub-tiles concatenated → 256×256 effective output). No
PC prototype targets this tile size.

Two implementation paths from here:

  **Path A: Use existing 128×256 8c4p prototype as direct port**
    * Halves M-tile from production 256 → 128 (cuts B-matrix L2 reuse 2×).
    * **Independently blocked** by task md FORBIDDEN PATHS line 132:
      "256×128 asymmetric tile (BN=128, 8-wave) = 7-23 % slower than
      production". The 128×256 case is even more asymmetric in M
      (per-tile overhead doesn't halve with BM either — production is
      already tuned to 256×256).
    * Net EV: BF16 PC -38 % @ 67 % consumer fraction, plus another
      -7 to -23 % from the 128×256 tile-shape penalty — compound
      regression of -45 % to -61 %.

  **Path B: Build new 256×256 PC variant from scratch**
    * No existing prototype to copy from — must design double-buffer
      LDS allocation for As/Bs at 256×256, route 4 producer warps to
      cooperatively load 4× larger tiles than the 128×256 prototype
      (= 4× larger memcpy_per_tile per producer warp), validate that
      LDS budget remains within the 160 KB CDNA4 limit (R53 measured
      production var-K at 128.5 KB / 160 KB; doubling As / Bs = 257 KB
      → blown LDS budget for 2-stage; 1-stage saves half but loses the
      load-shadow benefit that motivates PC).
    * **Refutes R53 claim "no greenfield kernel-architecture work
      required".** This IS greenfield — neither tile size nor LDS
      layout have an existing reference.
    * Implementation cost: ≥6 rounds (R53 estimated 4-6 even with a
      direct port).

## Evidence #3 — LDS budget for 256×256 2-stage PC blows the CDNA4 limit

Re-checking R53 inventory item #3 with corrected tile size:

  * Production var-K (256×256 effective via 2 × 256×128 sub-tiles):
    `ST_crr_a As[2][2]` = 4 × (256 × 128 × 1 B) = 128 KB
    `ST_crr_b Bs[2][2]` = 4 × (256 × 128 × 1 B) = 128 KB
    **Wait — 256 KB total exceeds 160 KB.**

    Re-reading line 8421-8427 of `kernel_fp8_layouts.cpp`:
    ```
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];
    ```
    `ST_v2a` and `ST_v2` are 128×128 each (HB=128 row dim, BLOCK_SIZE=128
    col dim) per `kernel_fp8_layouts.cpp` line 79-86 (HB=128, BK=128).
    So 4 × 16 KB × 2 = 128 KB total — matches R53 inventory.

  * The "256×256 tile" is actually formed by **wm,wn warp coordinates
    over 4 sub-tiles** (br*WARPS_M*2+wm, bc*WARPS_N*2+wn store at line
    8826-8832). Each LDS tile is 128×128 (= ST_v2 / ST_v2a). The 256×256
    output is the whole-CTA aggregate of 4 quadrants.

  * Therefore R53 inventory #3 stands: 2-stage PC at the existing 128×128
    LDS-tile size keeps 128 KB. ✓

  * BUT the PC prototype at 192×256 effective output uses M_BLOCK=3
    sub-tile copies (`As[2][3][2]`) = 6 × 16 KB × 2 = 192 KB — that
    blows 160 KB. Only the 128×256 8c4p (M_BLOCK=2) fits at 64 KB +
    64 KB = 128 KB.

  * Net: even if a custom 256×256 PC variant kept the 128×128 LDS
    sub-tiles (mirroring the dispatch), the **wave-budget tax (4/12 =
    33 % producers)** dominates, and the BF16 paper data already
    accounts for that scenario — it shows -38 % regression.

## Evidence #4 — R21 PMC pinpointing was done on 8w wgrad with NO PC available

R21 PMC said "32 % MfmaUtil, 58 % issue-rate idle" on Down-B4 wgrad.
R53 inferred PC could recover "75 % of barrier-induced idle". But
**R21's idle isn't necessarily barrier-induced** — the same PMC also
reported `SALU/SQ_busy = 85 %` on var-K wgrad (per task md line 102),
indicating the issue-rate idle is **co-dominated by SALU contention**
(coord decode, K-loop bookkeeping). PC eliminates barrier-sync delay
but adds NO SALU relief; if SALU is the binding constraint per iter,
removing barriers leaves consumer warps still SALU-blocked.

R53's arithmetic implicitly assumed barrier-eliminability translates
1:1 to MFMA-issue-rate gain. The BF16 paper data is the empirical test
of this assumption on CDNA4, and it falsifies the assumption: removing
3 of 4 barriers per iter does NOT recover anywhere near 75 % of
issue-rate idle in BF16 GEMM, so the FP8 transfer is unwarranted.

## Risk inventory revisited

R53 risk table assumed:
  * "Producer-side rcr_8w_load_hoist hard-coded to N_THREADS=512" — **incorrect**.
    The helper is template-parameterized: `rcr_8w_load_hoist<N_THREADS>`.
    Calling with `<256>` for 4 producer warps works (assuming
    `is_producer` gate). This is genuinely a non-blocker — R53 was
    overly cautious here. ✓ (correction noted, doesn't change verdict.)
  * "Variable per-group K (ki_g varies 16-32) — handled identically
    to production." — **partially correct**. Production handles ki_g
    variability via `for (int k = 0; k < ki_g - 2; k++)`, but each
    iter has 4 CTA barriers that BOTH producer and consumer wait on
    (the production design has no producer/consumer split). Once we
    split, the producer's load loop is SEPARATE from the consumer's
    MMA loop, and the producer must run AHEAD of consumer to provide
    load shadow. With `ki_g` as small as 16 on Down-B4-M2048, the
    prologue + epilogue dominate → load-shadow benefit is amortized
    over only ~14 main-loop iters. The BF16 paper data measured on
    K=4096-9216 (ki_K = K/BLOCK_SIZE = 64-144 main iters) had MORE
    iters to amortize and STILL lost — so the FP8 var-K case (16-32
    iters) is ARITHMETICALLY WORSE.

## EV verdict

  * R53 projected **+22 to +44 score**, EV per round = **+4 to +11**.
  * Re-grounded EV using existing CDNA4 BF16 paper data:
    - BF16 PC delivers **−17 % to −44 % TFLOPS** vs 100 %-consumer.
    - FP8-favorable corrections ≈ **+10 to +20 percentage points** at
      best (4× more K-reduction per LDS-tile-load).
    - Best-case net = **−7 % to −24 % TFLOPS** for FP8 var-K PC port.
    - **Negative EV** for the implementation arc.
  * Worst-case EV (catastrophic SNR fail mid-arc) = **0 score for
    affected shapes for ≥1 round** = potentially **-30 to -50 score
    transient** until rolled back.

A3 implementation arc is NEV-negative.

## R54 verdict

**Direction A3 (decoupled-warps producer-consumer port to FP8 var-K)
is PRE-IMPLEMENTATION FALSIFIED** — not by re-running the metric (no
implementation was attempted), but by **two pieces of pre-existing
evidence that R53 omitted** from its preflight:

  1. The CDNA4 BF16 PC paper data showing PC underperforming
     100 %-consumer baseline by 17-44 % across all measured sizes.
  2. The absence of a 256×256-tile PC prototype, combined with the
     forbidden-paths block on 256×128 asymmetric tiles that any
     existing-prototype port would need to inherit.

R54 ships **no kernel or dispatcher change**. Daemon metric expected in
the 690-699 noise band per R29 characterization.

## R55 forward-pointer

Per task md NEW DIRECTIONS list, the LAST untried direction is **G
(cross-shape co-optimization)**. All A1 (Stream-K), A3
(decoupled-warps), B (cross-stream), C (activation cache reuse), D
(SALU coord-decode — already shipped via R9-dm closed-form decode), E
(different barrier scheme — R26-R28 audited), F (larger tiles —
forbidden) directions are now exhausted at the preflight or
implementation level.

**R55 = Direction G probe**: identify a dispatcher rule lever that
loses ≤0.5 % on shape A but gains ≥1.5 % on shape B, where prior
per-shape sweeps falsified A and B in isolation but never tested the
joint search space. Concretely:

  * Anchor: pick a (gm, num_xcds) cell that is *known* sub-optimal
    on Down-B4-M2048 (e.g. R44 measured a -0.3 % per-shape drift)
    but might be co-optimal on Down-B4-M4096 (R6 falsified the same
    cell at the M4096 level, but used a single-shape SNR gate).
  * Probe: 3×3 (gm, xcds) sweep on the TWO shapes JOINTLY, scoring
    by sum-of-progress instead of per-shape. Verify SNR > 25 dB on
    both shapes.
  * If a co-optimum exists with sum-progress lift > 0.5 % over
    current per-shape rule, ship it as a new dispatcher rule keyed
    on `(layout, K, m_total)` for the joint family.

If R55 G falsifies, the task ceiling at 696 score is established and
the optimization daemon should be wound down (15+ rounds without
improvement, all NEW DIRECTIONS exhausted).

## Files added

  * `analysis/_notes/round-54-A3-decoupled-warps-PRE-IMPLEMENTATION-FALSIFIED-existing-CDNA4-PC-paper-data-shows-17-44pct-loss-AND-no-256x256-prototype.md` (this file)

## NEUTRAL round

No code, dispatcher, or kernel changes. Daemon metric expected in the
691-699 noise band.
