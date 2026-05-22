# Final Status — Source-tweak ceiling reached at 15 rounds

## What was achieved
- **R3**: b0+b1 split (RRR bn256) — committed `1b7646a6`. dsv3-up worst 17.30%→13.51% (-3.8pp)
- **R4**: epilog b0 prefetch — committed `9c73e332`. dsv3-up worst 13.51%→12.85% (-0.65pp)
- **R8**: RCR bench infrastructure — committed in PT campaign workspace
- Cumulative: **-4.5pp on RRR worst case**, primary score 1.0426→1.0530

## Why <5% target is unattainable in source-tweak space

### Hardware-level constraints (PMC verified)
- `SQ_BUSY_CYCLES = 28.8M` vs `SQ_WAIT_INST_ANY = 137.8M` — CUs wait 4.8x more than busy
- `SQ_WAIT_INST_LDS = 15.4M` (only 11% of total waits)
- `TA_TA_BUSY = 98.8M` — vmem unit saturated (3.4x SQ_BUSY)
- VGPR spill = 37 (structural; 4 acc + persistent state > register file)
- AGPR = 256 max (vs dense's 128) — compiler over-allocates due to persistent loop pressure

### ISA-level constraints (llvm-mc verified on gfx950)
- `ds_read_b128_tr_b8` does NOT exist — only `ds_read_b64_tr_b8` for fp8 transpose
- No way to widen B's LDS transpose-read instruction
- vmem→LDS direct cannot pre-transpose B (buffer_load_dwordx4 reads 16 contig
  bytes/lane; transpose would require 16 strided byte reads — impractical)

### Attempted source-tweaks (R5-R7, R9-R15) — all neutral or regression
- R5 unroll=1, R6 rrr_mma_agpr_inplace, R7 sched_barrier match dense
- R9 RCR FUSED_KTAIL gate, R10 GRID_MUL sweep, R11 view struct removal
- R12 chunk_size sweep, R13 unroll=4, R14 design doc, R15 stall analysis

## Path to <5% (requires multi-session architectural work)

### Recommended: Option A — B LDS pre-transpose
- Add `prefill_swizzled_offsets_b_transposed` for transposed LDS write
- Switch to `mma_ABt(C, A, B_T_row)` for math equivalence A·(B_T)^T = A·B
- Replace `load_col_from_st` with kittens standard `load` (ds_read_b128)
- **Estimate**: 250-400 LOC; 2-3 sessions including correctness verification
- **Predicted gain**: 5-7pp (halves B LDS instructions but only fixes 11% of stalls)

### Caveat: even with Option A, may not reach <5%
- Only 11% of waits are LDS-related (per PMC SQ_WAIT_INST_LDS)
- The other 89% (mfma issue stalls, vmem prefetch waits, persistent kernel overhead)
  need different solutions:
  - Option B (per-tile state SMEM) for spill reduction → reduce vmem/scratch
  - Option C (split-K) for high-K shapes → reduce per-WG tile count
  - Combined A+B+C might close to 5-8% gap, but truly hitting <5% on all shapes
    requires either Option D (32×32 mfma re-architecture) OR vendor-level
    optimizations that aren't user-controllable

## User decision needed

Given the genuine hardware/ISA constraints:
1. **Accept R3+R4 (4.5pp)** as session-deliverable; close /goal
2. **Authorize multi-session architectural rewrite** (Options A, B, C combined)
3. **Vendor escalation** for AMD to provide ds_read_b128_tr_b8 in future gfx95X

Current commits (HK turbo branch):
```
1b7646a6 fp8 grouped RRR bn256: revert single-b0 to b0+b1 split (dense pattern)
9c73e332 fp8 grouped RRR bn256: epilog 2 reuses b0 prefetched at end of epilog 1
a8702e68 mma.cuh: add mfma323264_agpr_inplace helper
6c7327bc fp8 grouped RCR bn128: fix non-determinism (prior campaign)
05768db7 fp8 grouped RRR bn128: minimize race-fix diff (prior)
bcf2b13a fp8 grouped RRR bn128: fix non-determinism (prior)
```

PT 3rdparty + agent workspace artifacts: all committed.
