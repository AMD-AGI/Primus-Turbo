round-61-saturation-reaffirmed-A-G-closure-final
=============================================================================

Round: 61 / 100
Date: 2026-05-10
Pre-SHA: ef5d103d (R60 — R59 RBM=64 RBN=32 PC residual A-PRIORI FALSIFIED)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R60 closed the SKILL.md NEW DIRECTIONS A-G inventory on every leaf. R61
has no novel lever to probe. NEUTRAL round; HEAD unchanged from R60 on
both repos.

## Why nothing to ship

Per the closure table from R59 + R60:

  A1 Stream-K           FALSIFIED R52  (1/8 cells survive PMC reality check)
  A2 SplitK var-K       FALSIFIED R33
  A3 Decoupled-warps    FALSIFIED R54 (256x256 EV-bound) + R60 (RBM=64 RBN=32 residual)
  B  Cross-stream       blocked by metric serial-timing semantics
  C  Activation cache   blocked by metric pre-quantize semantics
  D1 Closed-form decode SHIPPED neutral b3a5c8db
  D2 Magic divide       FALSIFIED R59 (sub-noise budget <=4 vs sigma 2.27)
  E  Different barrier  FALSIFIED R26-R28
  F  Larger tiles       FALSIFIED R32
  G  Cross-shape co-opt FALSIFIED R55

Plus the FORBIDDEN PATHS list in `scripts/_task_gpt_oss_fp8_kernel.md`
(VARK_MAIN_UNROLL=2/4, VARK_HOIST_PREFETCH_INTO_HALF1, VARK_DROP_BARRIER_*,
VARK_SW_PIPE_HOIST_AHEAD, 4-wave production port, small-tile 4-wave,
256x128 asymmetric, quant-cache, Down-B4 dispatcher), every macro/dispatcher
axis is closed with multi-sample evidence.

## Noise model (still current)

R56 30-sample on GPU 3 (current daemon-pin): median 695, sigma 2.27, no
upper tail. R57-R60 metric prints (697, 696, 696, 696) sit cleanly inside
[median - 1sigma, median + 1sigma] = [693, 697]. The noise model has not
drifted across 4 rounds; no re-characterization needed this round.

## Why D2-style sub-noise stacking won't help

R59 closed D2 magic divide with the projection "<=4 score units, below
sigma=2.27". A natural next thought is "stack 3-4 sub-noise wins to clear
sigma". This fails on two grounds:

  1. Per the R59 audit, the D2 candidate was the LAST sub-noise lever in
     the per-K-iter SALU bucket. The other audited SALU sites are either
     already shipped (D1 closed-form decode) or a-priori falsified
     (R34/R36/R37 launch-bounds / amdgpu-num-vgpr / forward-pointer
     variants). There is no second sub-noise candidate to stack on D2.

  2. Every sub-noise lever ships a SNR risk (template/ordering changes),
     a compile-time risk (build-flag macros adding scaffolding), and a
     cross-shape regression risk. Stacking 3-4 of these without an
     individual >=1-sigma gain to anchor the search is the precise path
     R22-R28 walked when the chained falsifications produced -3 to -20
     score on cells that looked locally neutral.

## Forward pointer (rounds 62-100)

The remaining daemon budget will hold the 695 +/- 2.27-sigma noise band.
Closing the 204-score gap to user TARGET=900 requires a lever class
outside the audited A-G + macro + dispatcher space. Two candidates that
are NOT in scope of this daemon:

  * CDNA4-equivalent of Hopper warp-specialisation TMA primitives:
    hardware-unsupported on MI355X (no async-bulk-tensor copy, no
    warp-specialised barrier instructions). Would require new hardware
    or compute-firmware support.

  * Re-scoping the metric (e.g., section-weighted, or include rowwise
    quantization granularity): metric script is FROZEN per task md;
    daemon cannot modify scoring.

R61 ship verdict: docs-only NEUTRAL. Same outcome will hold for R62-R100
unless an out-of-scope lever opens up.

## Files touched this round

* `Primus-Turbo/analysis/_notes/round-61-saturation-reaffirmed-A-G-closure-final.md`
  (this doc)

No code changes. HEAD unchanged on both repos.
