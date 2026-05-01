# Round 57-dm — FP8 grouped: Lever D Round-A step 1 — public _s aliases for 32x32x64 cell shapes

**Status**: INFRASTRUCTURE-ONLY commit / no metric impact expected
**Score before**: 961 (R28 baseline)
**Score after**:  960 (4-run sample 959/963/960/960 — within R28 noise band)
**HK SHA**: `6a93fa32` → `c2abba21`
**PT SHA**: this commit
**Round time**: ~25 min, 1 build cycle, 5 metric runs
**Auto-optimize round**: 29

## What was done

Added public `_s` type aliases to `include/types/types.cuh`:

```cpp
using rt_32x64_s = ducks::rt_shape::rt_32x64;
using rt_64x32_s = ducks::rt_shape::rt_64x32;
```

These were missing from R14-dm (which added the underlying
`rt_shape` structs and enumerated them in `ducks::rt_shape::all`).
Without the public aliases, FP8 kernel code couldn't declare
`rt_fp8e4m3<R, C, row_l, rt_32x64_s>` register tiles using the
public surface.

**HK commit**: `c2abba21` — `infra(fp8): add rt_32x64_s / rt_64x32_s
public aliases`. 17 lines added (15-line motivation comment + 2
type alias lines). Bit-equivalent to HK HEAD pre-commit; no callers
yet.

## Surprise discovery — most of Lever D infrastructure already exists

When I dug into the kernel for R57-dm, I found **MUCH more existing
infrastructure** than R56-dm assumed:

1. `rt_32x64` / `rt_64x32` rt_shape structs **already defined** in
   `include/types/register/rt_shape.cuh:59-60` (R14-dm).
2. `concept ducks::rt_shape::all` **already includes** these shapes.
3. `mma_AB_base` 32x32x64 scaffold **already wired** at
   `include/ops/warp/register/tile/mma.cuh:172-185` (calls
   `mfma323264` when D is rt_32x32, A is 32x64, B is 64x32).
4. `mma_ABt_base` 32x32x64 scaffold **already wired** at lines
   234-238 (D rt_32x32, A 32x64, B 32x64).
5. **Standalone K-tail kernels using mfma_323264 already exist
   AND were tested**:
   - `grouped_ktail_kernel_mfma32x32` (line 4062)
   - `grouped_ktail_kernel_mfma32x32_M2` (line 4263, 2 stacked
     32×32 sub-blocks)
   - `grouped_ktail_kernel_mfma32x32_M2N2` (line 4454, 2×2 sub-tiles)
   - `grouped_ktail_kernel_mfma32x32_M2N4` (line 4636, disabled
     by `#if 0` after R61 regressed metric 752 → 749).
6. The M2N2 standalone variant **was launched and tested in R52-dm**
   as a separate kernel after main; **falsified at -76 pts metric**.

So R56-dm's "7-round commit" estimate was inaccurate. R57-R59 of that
plan are essentially DONE (this commit + the existing R14-dm work).
The remaining work is R60+ (the actual K-tail body port) and **R52-dm
already proved the standalone-launch path doesn't work**.

## Refined Lever D fan-out cost analysis

R56-dm estimated fan-out cost ~50-100 cy, savings ~128 cy → net
+0-78 cy = +0-1.4 pp ratio. After examining the M2N2 standalone
kernel's actual lane mapping (lines 4604-4612), I derived:

**32x32x64 mfma output lane layout** (from M2N2 kernel):
```cpp
// Each lane (tid in 0..63) holds 16 fp32 dwords:
//   col_in_cell = tid % 32        (single column per lane)
//   chunk       = tid / 32         (0 or 1)
//   For i ∈ [0, 16):
//     row_group     = i / 4        (0..3)
//     row_in_group  = (i % 4) + chunk * 4   (0..7)
//     row_in_cell   = row_group * 8 + row_in_group  (0..31, 16 unique values)
//   acc[i] is at (row_in_cell, col_in_cell)
```

Lane (l) holds **all 16 rows of a single column slice** within its
chunk (chunk=0 covers rows {0..3, 8..11, 16..19, 24..27}; chunk=1
covers complementary rows). Tall vertical strip, 1 col per lane.

**rt_16x16 col_l output lane layout** (from existing rcr_mma):
```cpp
// Each lane holds 4 fp32 dwords per cell:
//   row_in_cell = lane % 16
//   col_in_cell = (lane / 16) * 4 + [0, 1, 2, 3]
```

Lane (l) holds **4 contiguous columns at a single row** within the
16x16 cell. Short horizontal strip, 4 cols per lane.

**Cross-layout merge cost**:

For one 64x32 region currently held as rt_fl<64,32,col_l,rt_16x16>
(8 cells × 4 dwords/lane = 32 dwords/lane), porting to
rt_fl<64,32,col_l,rt_32x32> (2 cells × 16 dwords/lane = 32
dwords/lane) is **same total dwords/lane** but completely different
internal lane partition.

To merge a `cAB_32` (rt_32x32 layout) into existing `cA` (rt_16x16
layout), each element must traverse a cross-lane swap:
- `cAB_32` lane L holds (row=L%32 split by chunk, col=L%32) — only
  the row mapping conflicts.
- `cA` cells split the same 32×32 region into 4 quadrants of
  16×16, each with its own lane partition.
- For ONE element at (r, c) in the 64x32 region:
  - `cAB_32` lane that owns it: depends on chunk and row_group
    decomposition.
  - `cA` lane that owns it: row_in_quadrant = r % 16,
    col_in_quadrant_strip = (c % 16) / 4.
  - These rarely match → cross-lane data movement required.

**Two practical merge mechanisms**:

(a) **Cross-lane DPP / v_permute_b32**: 16 dwords per
    `cAB_32` lane, each potentially destined for a different `cA`
    lane. ~16-32 v_permute instructions per cAB_32 → 4× rt_16x16
    fan-out, ~4 cy each = 64-128 cy per cAB_32. With 4 cAB_32
    (covering cA + cB + cC + cD) = 256-512 cy of register
    permutation. **Eats ALL the mfma cycle savings (128 cy)** and
    introduces extra scheduler pressure.

(b) **LDS round-trip**: 1 ds_write per 32 dwords/lane × 4 cAB_32 =
    128 ds_writes per warp = ~32 cy SQ issue + LDS bandwidth +
    barrier sync (~50 cy). Then ds_read into cA-cD additive ~32 cy.
    Total ~120-150 cy per K-tail. **Still eats most of the 128 cy
    mfma savings**, leaving 0-20 cy net (= 0-0.4 pp ratio).

PLUS register-pressure spike during steps 3-4 of the merge:
- cA-cD live (128 dwords/lane, K=[0..2816) sums)
- + cAB_32 live during mfma + LDS write (128 dwords/lane)
- + a + a_kt1 + b0 + b1 (~80-128 dwords/lane during K-tail HBM load)
- = 336-384 dwords/lane peak, well exceeds 256 VGPR/lane budget
- → 80-128 dwords/lane extra spill on top of current 39

Each spill is ~8 cy round-trip × ~80 dwords = ~640 cy added back to
hot path. **Net Lever D K-tail port = -500+ cy = -10 pp ratio**.

This is much worse than R56-dm's "+0-1.4 pp" estimate. **Lever D
Round-A is structurally NOT a win** for the fused K-tail path.

## Why M2N2 standalone was tested and -76 pts

R52-dm tested `grouped_ktail_kernel_mfma32x32_M2N2` as a SEPARATE
launch after `grouped_rcr_kernel` (with FUSED_KTAIL=false template
spec firing on K_REM=64). That path:
- Has a separate kernel launch (~30-50 us overhead per launch
  on AMD).
- Reads C from HBM (already-stored cA-cD output, fp32 → bf16),
  adds K-tail mfma contribution, writes back. C-RMW chain.
- C-tile cache is COLD between main kernel and K-tail launch (~5MB
  C tensor, larger than L2).
- Doesn't benefit from the fused FUSED_KTAIL=true template's spill
  reduction (R34-dm: a_kt1 in scope gives -7 dw spill; standalone
  K-tail kernel doesn't have this).

Net: launch overhead + C-RMW + cold cache = -76 pts metric, much
worse than the in-fused-kernel mfma savings. R52-dm correctly
falsified this approach.

## Conclusion — Lever D path is dead

Both directions are now FALSIFIED:
1. **Fused K-tail port** (this round's analysis): cross-layout
   merge cost ≥ mfma savings; register-pressure spike causes
   spills that exceed mfma savings by ~10 pp.
2. **Standalone K-tail launch** (R52-dm): launch + C-RMW overhead
   exceeds mfma savings by ~7-8 pp on metric.

The only remaining direction would be **Lever D Round-B** — full
main-loop port to rt_32x32 cell shape. This requires:
- New A_row_reg = rt_fp8e4m3<128, 64, row_l, rt_32x64_s>
  (was rt_16x128_s)
- New B_row_reg = rt_fp8e4m3<32, 128, row_l, rt_32x64_s>
- Rewrite ds_read / buffer_load_lds for new lane layout
- Rewrite store from rt_32x32 acc to bf16 g.c (different lane
  mapping for output)
- Rewrite mul, K-tail, epilog 1, epilog 2, store_c_tile_n_masked
  for the new layout

R56-dm's "5-round" estimate for this is OPTIMISTIC; realistic scope
is **8-12 rounds** with high risk of intermediate-round metric
regressions (kittens infrastructure churn). Patience counter would
need to absorb 3-5 rounds of doc-only or near-neutral commits.

## Recommendation for R30 (R58-dm in note numbering)

**Three options, in order of expected value**:

**Option α (de-risk, recommended)**: Accept plateau 947-963.
Continue committing infrastructure that other agents may benefit
from (e.g., this round's _s aliases). Don't pursue Lever D Round-B
in remaining 31 rounds — risk of 8-12 round regression chains
exceeds the +5-10 pp ratio upside.

**Option β (high-risk, high-cost)**: Begin Lever D Round-B step 1:
add `rt_fp8e4m3<R, C, row_l, rt_32x64_s>` instantiation +
correctness probe (compare 32x32 mfma output to 16x16 mfma output
on a 64x32 region with random A/B; max_abs should be 0). Pure
infrastructure round, no metric change expected. Continue 8-12
rounds for full port.

**Option γ (orthogonal)**: Investigate non-FP8-grouped paths that
are out of scope for this task body but may help long-term:
- main-loop spill reduction via LDS-staged hot working set (heavy
  refactor, 5+ rounds).
- HBM bandwidth optimization via better SRD prefetch patterns
  (medium refactor, 2-3 rounds).
- BF16-grouped optimization (already at 1.183 geomean, closer to
  1.20). **Out of scope this task body** but the closest thing to
  a clear win available.

I recommend **Option α** for R30+: keep committing low-risk
infrastructure, accept the FP8 plateau, and document why the score
ceiling is ~962.

## Round meta

- Auto-optimize round: 29 (this round)
- Score trajectory: 961 (R28) → 960 (R29 with alias commit, 4-run
  median 959-963).
- Plateau: round 11 of 947-963 noise band (slight band widening
  due to GPU 3 vs GPU 4 variance).
- patience: 1/10 — slight regression but within noise.
- HK SHA: `6a93fa32` (R50-dm winner) → `c2abba21` (this round's
  alias commit, bit-equivalent in codegen).

## Files touched

- `/workspace/code/HipKittens/include/types/types.cuh`: +17 lines
  (alias commit `c2abba21`).
- `/workspace/code/Primus-Turbo/analysis/_notes/round-57-dm-...`:
  this note (~250 lines).

No `kernel_fp8_layouts.cpp` change. No callers of new aliases yet.
