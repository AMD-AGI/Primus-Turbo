# Round 74 — BF16 grouped bwd var-K LDS padded swap FALSIFIED (VGPR spill 66 + 24/24 dB-allclose FAIL)

**Status:** FALSIFIED — single-line shape swap insufficient; P1 requires kernel-
structural redesign, not a shape alias change.
**Score:** baseline 875 → with-patch 0 (24/24 dB-allclose FAIL). Clean revert → 875.

## R74 Experiment (R72 P1 lever — var-K CRR LDS swizzle)

Goal: eliminate R68 PMC diagnostic's 217M LDS bank conflicts in
`grouped_var_k_kernel<0>` (CRR dB, 34% of bwd wall × 24 shapes).

Change: single-line swap of ST shape at `kernel_bf16_dynamic.cpp:4723-4724`:

```cpp
// Was:
using ST_A = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;            // 64x128, sub-tile 32x16 (16 sub-tiles, unpadded)
using ST_B = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
// Tried:
using ST_A = st_bf<K_STEP, HALF_BLOCK_SIZE, st_64x32_padded_b128_s>; // 64x128, sub-tile 64x32 (4 sub-tiles, padded)
using ST_B = st_bf<K_STEP, HALF_BLOCK_SIZE, st_64x32_padded_b128_s>;
```

## Result

### Compile (unexpectedly passes type-check)

```
kernel_bf16_dynamic.cpp:4717:1: grouped_var_k_kernel<0>:
    TotalSGPRs: 103       (was 95,  +8)
    VGPRs:      256       (unchanged, at per-wave cap)
    ScratchSize [bytes/lane]: 268 (was 0)
    Occupancy [waves/SIMD]: 2 (unchanged)
    SGPRs Spill: 0
    VGPRs Spill: 66       (was 0)   <-- severe spill
    LDS Size [bytes/block]: 272 (unchanged)
```

Builds without `static_assert` firing, but the compiler now spills 66 VGPRs
to scratch (268 bytes/lane = 268*256 threads = ~69 KB/block scratch traffic
per K-iteration). This alone would kill perf even if correctness worked.

### Metric (catastrophic correctness failure)

```
correct_fail = 24/24   (all shapes dB-allclose FAIL)
score = 0/1000
```

All 24 shapes fail the dB-allclose check. Confirms R69 note's warning:
Path B for `st_64x32_padded_b128_s` → `rt_32x16_s col_l` is listed as
"compiles only, not semantically verified". The CRR A/B layout expected
by var-K (K=fast, M/N=slow, col_l registers) does NOT round-trip through
`st_64x32_padded_b128`'s sub-tile padding scheme.

### Root cause sketch

The padded shape is designed for RCR: `st_bf<HALF_BLOCK_SIZE=128, K_STEP=64,
st_64x32_padded_b128_s>` with outer (128, 64), sub-tile (64, 32), composed
2×2. The kittens `G::load` / `G::prefill_swizzled_offsets` / `load(reg, st)`
helpers implement the K-along-rows convention for this shape.

Var-K CRR swaps the axes: `st_bf<K_STEP=64, HALF_BLOCK_SIZE=128, ...>`
with outer (64, 128). For the unpadded `st_32x16_s` (sub-tile 32x16), the
composition is 2×8 sub-tiles and the helpers implement the K-along-cols
convention correctly. When we force the padded shape onto the (64, 128)
outer dims, the composition becomes 1×4 sub-tiles — but the helpers still
iterate assuming K-along-rows, and the padded swizzle's `(row*cols + col)`
offset formula (identity within sub-tile, padding BETWEEN sub-tiles along
the row axis) lands writes/reads on the wrong LDS bytes for the 2nd-3rd-4th
column sub-tiles.

### Why VGPR spills

Each memcpy per thread was bytes_per_thread=16 and memcpy_per_tile
changed from `BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy` with
unpadded strides to a padded-stride version that the compiler is unable
to hoist out of the inner loop efficiently. The per-tile offset array
`swizzled_offsets_A/B[memcpy_per_tile/2]` also grows because the padded
subtile stride is 4128 (= 4096 + 32) vs the unpadded 4096. The compiler
cannot fold the 32B padding term into a uniform offset, so a per-lane
VGPR is allocated for each sub-tile offset — 66 VGPRs worth.

## Reverted to baseline

- HK: `git checkout -- kernel_bf16_dynamic.cpp`, rebuild confirmed
  VGPR Spill: 0, baseline 875 re-verified (24/24 PASS).
- No code committed this round.

## R75 direction (next agent cold-start)

**P1 is still the highest-priority lever** (R68 identified 217M LDS
bank conflicts in var-K; +10-15 score upside). But it requires:

### R75 prerequisite: write a standalone LDS-swizzle probe

File `/tmp/r75_varK_lds_probe.cu`:
- Allocate a 64x128 bf16 tile in LDS with `st_bf<64, 128, st_XXX>`
  for XXX ∈ {`st_32x16_s`, `st_64x32_padded_b128_s`, and a new custom
  `st_32x16_padded_b64` to be defined}.
- Fill with known pattern `src[r, c] = r * 1000 + c`.
- Issue `load(rt_reg, st_tile)` with `rt_bf<K_STEP, HALF_REG_BLOCK_M,
  col_l, rt_32x16_s>` (what var-K uses).
- Dump `rt_reg` back to HBM via shuffle/store pattern.
- Compare to fp32 reference (expect bit-exact for bf16 round-trip).

If `st_64x32_padded_b128_s` FAILs (expected from R74 result), the probe's
per-cell diff pattern tells us which sub-tile indices are wrong — that
pinpoints whether the bug is in:
- (a) `G::prefill_swizzled_offsets` HBM-voffset reconstruction (the
  `r*cols + c` formula inside st_64x32_padded_b128::swizzle — R57+ note
  in st_shape.cuh:346-352 says this only works for the RCR (128, 64)
  outer layout because of how padding composes with `subtile_padding=32`).
- (b) `load(reg, st_subtile)` lane→byte addressing in
  `shared_to_register.cuh` for the col_l rt_32x16_s variant.

### R75 fallback: define a new padded shape for CRR

Create `st_32x16_padded_b64` (sub-tile 32x16, padding 32 between
sub-tiles along the row dim to break the 128B bank alignment). Should
preserve the existing 2×8 sub-tile composition used by var-K's helpers,
only adding inter-sub-tile padding. Lower risk than switching the whole
shape structure.

Expected outcome: 217M → ~50M LDS bank conflicts (mirror RCR's 0 conflicts),
dB var-K wall 5-10% faster, score +8-12.

## No code change this round

HK + Primus working trees clean. Baseline 875 verified 24/24 PASS.

## Chat-window note

Current chat is at 90/90 min — R75 cold-start. Next agent must re-derive
context from R72-R74 notes. This R74 note is the concrete P1 implementation
report + exact failure mode + fallback direction.
