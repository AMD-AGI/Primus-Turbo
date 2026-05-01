# Round 3-dm — FP8 grouped main-loop `#pragma unroll` saturated

**Date**: 2026-05-01 (Round 3 of 100)
**Primus-Turbo HEAD (entry)**: `b732ade` (round-2-dm ST_v3 docs)
**HK HEAD (entry)**: `f9f2b545` (round-2-dm ST_v3 falsified)

## Round-3 data

- Round-3 baseline metric: **815** (geomean 0.9779), same lowest-ratio
  shape `grpFP8_gpt_oss_20B-GateUP-B4-M2048 @ 0.930` as rounds 1 & 2.
- `RCR_MAIN_UNROLL` was the last untested main-loop hint not in the
  round-25 saturated-knobs inventory. Round-26 docs claimed "tested"
  without numerical evidence; this round verified with a clean 3×5-run
  sweep.

## Sweep results

Applied `TK_PRAGMA_UNROLL(N)` as a grouped-local override at the
`grouped_rcr_kernel` main-loop site (line 2161 in
`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`), leaving
dense RCR's hint unchanged at macro `RCR_MAIN_UNROLL = 2`.

| N (hint) | n | mean | median | range      |
|---|---|---|---|---|
| 1 | 5 | **816.2** | 818 | [810, 822] |
| 2 | 5 | **816.2** | 816 | [812, 819] |
| 4 | 5 | **816.4** | 818 | [812, 820] |

**Means within 0.2 score** — flat across the entire probe range.
VGPR spill counts identical across all three cells (67 / 76 / 45 / 54
for the four `<FUSED_KTAIL, N_MASKED>` template specializations),
confirming LLVM's unroll heuristic picks its own factor regardless
of the `#pragma unroll N` hint at this body size.

## Conclusion

`RCR_MAIN_UNROLL` is formally **saturated / compiler-driven**. Added
to the cumulative saturated-knobs inventory (which now covers every
exposed macro / pragma / config knob in the FP8 grouped RCR main
path). No kernel semantic change this round; reverted override to
shared `RCR_MAIN_UNROLL` macro so future dense-side tunes propagate
transparently.

## What's left (unchanged from round-2-dm closing)

Only multi-round structural projects remain:

1. **FP8 K-tail amortize across M-slab** (round-8 §1 design, 2-3
   rounds). **Recommended round-4-dm target.**
2. **FP8 direct HBM → register main loop** (task-body lever E, 4-8
   rounds).
3. **FP8 MFMA cell-shape 16x16x128 → 32x32x64** (2-3 rounds, round-12
   partially falsified at MFMA-count level).

## Round-4 suggestion

Start project (1): read-only audit of `grouped_rcr_kernel`'s
FUSED_KTAIL epilog (lines 2045-2331) to identify the M-slab boundary
and shared-accumulator insertion point. No kernel edit in round-4;
design doc only. First actual edit in round-5.

## Files touched

- HipKittens:
  `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:2155-2171`
  (comment-only; no semantic change).
  `analysis/_notes/round-3-dm-fp8-grouped-main-loop-unroll-saturated.md`
  (new).
- Primus-Turbo: `analysis/_notes/round-3-dm-fp8-unroll-sweep-saturated.md`
  (this file).

## Commits

- HipKittens: `docs(round-3-dm): FP8 grouped RCR main-loop #pragma unroll sweep saturated ({1,2,4} means all 816.2-816.4 over 5 runs)`
- Primus-Turbo: `docs(round-3-dm): FP8 grouped #pragma unroll sweep saturated`
