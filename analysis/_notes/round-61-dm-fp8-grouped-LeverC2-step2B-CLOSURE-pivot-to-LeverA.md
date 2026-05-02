# Round 61 — FP8 grouped: Lever C-2 step-2B CLOSURE, pivot to Lever A

**Date**: 2026-05-02 (R61 of 100)
**HEAD before**: 29553fb14690c890493433173d4c983a7b03edba
**Score**: baseline 977 → final 983 (within ±5 noise band, **NO regression**)
**Goal**: continue R60 debug — try temp-copy probe (R60 plan step 3); if fails, formally close C-2 path and announce pivot.

## TL;DR

Tried the R60 plan step-3 temp-copy workaround: copy `cAB[0][0]` to a
fresh temp via `copy(c00_tmp, cAB[0][0])` and store from the temp.

**Codegen radically changed but correctness pattern UNCHANGED**:

| Metric | R59 baseline | R60 (no-mul) | R61 (temp-copy) |
|---|---|---|---|
| VGPR Spill | 80 | 80 | **12** (87.5% drop) |
| ScratchSize [B/lane] | 324 | 324 | **12** |
| AGPR | 256 | 256 | 256 |
| cAB[0][0] broken base tiles | tiles[{0,1}][1] | tiles[{0,1}][1] | tiles[{0,1}][1] |
| Correctness | FAIL | FAIL | FAIL |

The temp-copy dramatically improved compiler register allocation but did
NOT fix the broken base tiles. This **proves the bug is NOT at store
time** — store-from-temp gives the same wrong output as direct AGPR
store. The bug is in the `mma_ABt(cAB[0][0], a_reg[0], b_reg[0],
cAB[0][0])` computation itself.

## Failure mode summary across all R60-R61 mitigations

| Mitigation | What it changed | Effect on cAB[0][0] |
|---|---|---|
| Sacrificial dummy (R60) | declares C_acc_4w before cAB | NONE (DCE'd) |
| Explicit per-base-tile unroll (R60) | 16-iter unroll instead of mma_ABt | tiles[{0,1}][1] partially fixed (294→86) but cAB[1][0]/[1][1] regressed (~0.3) |
| Skip mul() epilog (R60) | removes scale multiply | first 4 rows of cAB[0][0] correct (~0.001 diff), tiles[{0,1}][1] still wrong |
| Temp-copy + store-from-temp (R61) | copy(temp, cAB[0][0]) before store | spill 80→12, tiles[{0,1}][1] still wrong |

**Key insight**: All 4 mitigations produced WILDLY DIFFERENT codegen
(verified by spill count, scratch size, ISA changes), yet the broken
output region is **bit-identical** across all of them: rows 0-31, cols
16-31 of cAB[0][0] (= base tiles `[n=0..1][m=1]`).

This is characteristic of a **deterministic LLVM AGPR allocation defect
specific to this kernel shape**, not a runtime race condition.

## Root cause hypothesis (unverified)

The cAB[0][0] cell holds 16 base tiles in a 4×4 grid. LLVM allocates
these to AGPR slots. For our specific kernel:
- per-warp acc footprint = 4 cells × 64×64 fp32 = 1024 fp32/warp = 16 fp32/lane × 64 lanes = exactly fits AGPR=256
- mma_ABt produces cAB[0][0].tiles[n][m] for n,m ∈ [0,4) in some traversal order
- something in the AGPR slot assignment for `cAB[0][0]` (the FIRST acc cell to be computed) places `tiles[{0,1}][1]` in slots that get clobbered between mma write and the next mma_ABt(cAB[0][1], ...) call

Cross-cell verification (logical impossibility otherwise):
- cAB[0][1].tiles[0][1] computed correctly using same a_reg base tile
- cAB[1][0].tiles[0][1] computed correctly using same b_reg base tile
- Yet cAB[0][0].tiles[0][1] = same a × same b → BROKEN

## C-2 path closure (formal)

After **7 rounds of effort** (R54-R60):
- R54: scaffold landed (4w types)
- R57: AGPR allocation hypothesis CONFIRMED (256 AGPR / 0 spill in stub kernel)
- R58: real-load helpers preserve AGPR (256 AGPR / 8 spill)
- R59: real coords kernel — AGPR retained but cAB[0][0] correctness FAILS
- R60-R61: 4 mitigation attempts — all preserve the bug

**Conclusion**: the 4w-style 4-cell × 64×64 accumulator approach is
BLOCKED by what appears to be a deterministic LLVM AGPR allocation
defect specific to this kernel shape. Continuing to debug = high risk
of more wasted rounds with no guarantee of success.

The test kernel + complete debug docstring stays in tree as reference
material for a possible future C-2 retry (e.g. after LLVM upgrade,
different cell shape that doesn't trigger the AGPR bug, or pure-VGPR
schedule that avoids AGPR entirely).

## Pivot to Lever A (async global→LDS + MFMA pipelining)

**R62+ plan**: explore Lever A. Why this is the right next move:
- doesn't require AGPR (works with normal VGPR allocation, no LLVM bug
  in the way)
- reduces register pressure by eliminating VGPR staging of A_row_reg /
  B_row_reg between buffer_load and ds_read
- uses gfx950 `global_load_lds_dwordx4` ASM intrinsic directly (HK
  helper `rcr_8w_load_hoist` already implements this for the 8w grouped
  path; we already verified `global_load_lds` instructions are emitted
  in libtk_fp8_layouts.so)
- the 67-spill bottleneck in the production grouped kernel is in part
  due to A_row_reg / B_row_reg VGPR staging — Lever A removes this
  source of spill by going DIRECTLY to LDS

**Lever A concrete approach** (R62 starting point):
1. Identify current sync vs async transitions in `grouped_rcr_kernel`'s
   K-loop (production fp8 grouped, ~line 2800).
2. Replace synchronous `g2s_pass<i>` (which stages global → register →
   ds_write_b128 → register → mma) with a single async
   `global_load_lds_dwordx4` (global → LDS direct, MFMA reads LDS).
3. Verify spill count drops via `-Rpass-analysis=kernel-resource-usage`.
4. Run probe (BF16 grouped first as smoke, then FP8 grouped).
5. Run metric.

Risk: production grouped kernel ALREADY uses `rcr_8w_load_hoist` which
ALREADY uses `buffer_load_dwordx4 ... offen lds` (= async direct-to-LDS).
So Lever A may have already been partially applied. Need to verify and
identify what's still synchronous.

## Files touched (R61)

- `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - kept the test kernel + R60 changes
  - added temp-copy (`copy(c00_tmp, cAB[0][0])`) before store of cAB[0][0]
  - extended docstring with R61 findings + C-2 closure announcement
- `Primus-Turbo/analysis/_notes/round-61-dm-...`: this note

## Production kernel impact

NONE. Test kernel is reachable only via the dedicated
`tk_fp8_layouts.test_4w_real_coords` pybind binding. All grouped FP8
production paths byte-identical to R55.

Metric: 977 → 983 (Δ = +6, within noise band).

## Roadmap (revised, R62-R65 outlook)

- **R62**: Lever A audit + first pipelining attempt
- **R63**: Lever A — full grouped K-loop pipelining
- **R64**: Verify Lever A on Qwen3 GateUP (large K=4096, async benefits most)
- **R65**: Metric validation + commit OR falsify (if no measurable improvement)
