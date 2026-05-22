# Architectural Rewrite Design — Closing dsv3-up Gap to <5%

## Why source tweaks are exhausted (13 rounds done)

R1-R13 ISA+PMC-driven attempts. Final state:
- dsv3-up worst: 12.85% gap (down from 17.30%, -4.5pp via R3+R4)
- Stall analysis: SQ_WAIT_INST_ANY/SQ_BUSY = 4.8 (CUs wait 4.8x more than busy)
- SQ_WAIT_INST_LDS only 11% of waits — **LDS is NOT primary bottleneck**
- 89% of waits are mfma issue stalls + vmem prefetch waits
- Spill VGPRs = 37 (structural: 4 acc + persistent state > register file)

ISA verified: `ds_read_b128_tr_b8` does NOT exist on gfx950 (llvm-mc confirmed).
Only ds_read_b64_tr_b8 for fp8 transposed load. So bandwidth-doubling via wider
transpose-read is impossible at instruction level.

## Required architectural rewrites (4 candidates)

### Option A: B LDS pre-transpose + mma_ABt (~250-400 LOC)
- Change RRR's B from `mma_AB(c, A, B_col)` to `mma_ABt(c, A, B_row_transposed)`
- Requires new `prefill_swizzled_offsets_transposed` that computes per-lane
  V offsets to write B[k][n] → LDS[n][k] (transpose at LDS write time)
- Switch RRR's Bs from ST_v2 to ST_row layout
- Replace `load_col_from_st` with kittens' standard `load` (ds_read_b128)
- mma_ABt(C, A, B_row) computes A * (B^T)^T = A * B (math preserved)
- **Estimated gain**: ~5-7pp (halves B LDS instructions: 40 → 20 per main+epi)
- **Risk**: lane-mapping correctness — needs careful verification of mfma operand
  layout vs ds_read_b128 byte ordering. Likely 1-2 iterations of debugging.

### Option B: Per-tile state SMEM hoist (~200 LOC)
- Move `a_gl_g`, `c_gl_g` view copies and `soA`/`soB` swizzle arrays to SMEM
- Per-tile in main loop: read from SMEM into SGPR (one-time hoist)
- Saves ~16-20 VGPRs → spill 37 might drop to 0
- **Estimated gain**: ~2-4pp (eliminate scratch ops in init + post-mma)
- **Risk**: SMEM bandwidth/latency for soA/soB reads in main loop may negate gain

### Option C: Split-K with grid reduction (~400 LOC)
- For high-K shapes (dsv3-up K=7168), split K into 2 or 4 sub-Ks
- Each sub-K kernel computes partial cA
- Final reduction kernel sums partials
- **Estimated gain**: ~8-12pp on dsv3-up (better cache + dispatch)
- **Risk**: requires new kernel pipeline, reduction overhead, atomic conflicts

### Option D: 32x32 mfma + tile restructure (~500 LOC)
- Use `v_mfma_f32_32x32x64_f8f6f4` (1 instruction = 4x throughput of 16x16x128)
- Per-warp tile 32x32 instead of 16x16, reduces per-warp acc tile count
- **Estimated gain**: ~5-8pp (lower instruction issue overhead)
- **Risk**: per memory `[fp8 RRR 32×32 flawed premise]` 32x32 alone doesn't help
  for fixed BLK_M×BLK_N. Need coordinated tile reshape.

## Recommended path (priority order based on cost/benefit)

1. **Option A (B LDS pre-transpose)** — file comment explicitly marks as
   actionable lever; biggest predicted gain; well-defined scope.
2. Option B can follow as complementary (different bottleneck).
3. Options C and D are higher-risk/higher-effort alternatives.

## What's needed to execute Option A

- **AMD MFMA operand layout docs** for `v_mfma_f32_16x16x128_f8f6f4` B operand
  lane-to-element mapping. Without this, ds_read_b128 output to register may
  not match mfma operand layout, causing wrong results.
- Alternative: empirical verification via small test kernel that does
  ds_read_b128 + mma_ABt for both layouts (B/B^T), comparing against reference.
- 1-2 sessions of focused work for implementation + correctness verification.

## Current state (committed)
- HK 1b7646a6 (R3 b0+b1 split)
- HK 9c73e332 (R4 epilog b0 prefetch)
- HK 6c7327bc (RCR bn128 race fix from prior campaign)
- HK 05768db7, bcf2b13a (RRR bn128 race fix)
- HK a8702e68 (mma.cuh helper)
- PT 3rdparty mirrors for all
