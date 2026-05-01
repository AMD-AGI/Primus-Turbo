# Round 22 — config-knob saturation confirmed; FP8 chunk_size dead-end

## TL;DR

Round 21 BF16 config re-tune banked +3 score (879 → 882 mean). Round 22
explored 3 directions to find another lever; all dead-ended at the
metric noise floor. Net commit: notes-only.

## What was tried

### 1. FP8 (gm, xcd) sweep on the 4 B=32 gpt_oss shapes (NEW: not covered in round 21)

Round 21 only swept the 4 worst-ratio FP8 shapes (all B=4); the 4 B=32
shapes (ratios 0.97-0.99) hadn't been re-tuned post-round-19 BUFFER.
44-cell × 3-trial metric-aligned per-iter-sync sweep
(`/tmp/sweep_fp8_b32_round22.py`):

| FP8 B=32 shape       | default rule    | top cfg / TF   | gap     |
|----------------------|-----------------|----------------|---------|
| Down-B32-M2048       | (gm=16, xcd=4)  | (32, 4) 1022.6 | +0.01%  |
| Down-B32-M4096       | (gm=4, xcd=8)*  | (4,  1) 1204.8 | +0.28%  |
| GateUP-B32-M2048     | (gm=8,  xcd=4)  | (8,  4) 1191.7 | +0.00%  |
| GateUP-B32-M4096     | (gm=8,  xcd=4)  | (8,  4) 1405.7 | +0.00%  |

\* `xcd=None` falls back to BLOCK_SWIZZLE_NUM_XCDS=8 in the kernel.

All 4 are within 0.3% of the default config — the FP8 (gm, xcd)
knob-space is now fully saturated across all 8 gpt_oss shapes (rounds
7-12-21 plus this round). No update.

### 2. BF16 (gm, xcd) re-verify on shapes I didn't metric-align last round

Round 21 metric-aligned tested only 5 BF16 shapes; the other 3
(Down-B4-M2048, GateUP-B4-M2048, GateUP-B32-M2048) were OK on the
steady-state probe. Re-ran metric-aligned to confirm:

| BF16 shape          | current rule   | top cfg gap |
|---------------------|----------------|-------------|
| Down-B4-M2048       | (gm=2, xcd=2)  | +0.13%      |
| GateUP-B32-M2048    | (gm=8, xcd=4)  | +0.01%      |

Both at the optimum. All 8 BF16 gpt_oss shapes now confirmed
saturated.

### 3. FP8 forward kernel `chunk_size` swizzle parameter (DEAD-END but valuable insight)

**Discovery**: `kernel_fp8_layouts.cpp::grouped_rcr_kernel` calls
`chiplet_transform_chunked(blockIdx.x, NUM_CUS=256, xcds_eff,
chunk_size=64)`. With `xcds_eff=8` (default), `block = num_xcds *
chunk_size = 8 * 64 = 512 > NUM_CUS = 256`, so
`limit = (256 / 512) * 512 = 0` and the function's early return
`if (workgroup_id > limit) return workgroup_id` passes through every
WG with `workgroup_id ≥ 1` unchanged — **the chiplet swizzle is a no-op
at default num_xcds=8**. Only `num_xcds ∈ {1, 2, 4}` activate the
swizzle (block ≤ NUM_CUS).

This explains why rounds 7-21 (gm, xcd) sweeps consistently picked
`num_xcds ∈ {2, 4}` for every shape that retuned: those values
actually engage the chiplet-swizzle, while the default xcds=8 silently
falls back to the natural strided WG-to-XCD assignment.

**Hypothesis**: lowering `chunk_size` from 64 to 32 keeps `block ≤
NUM_CUS` for `num_xcds ≤ 8`, activating the swizzle at the default
xcds=8. Tested in `kernel_fp8_layouts.cpp` line ~2005 (forward kernel
only):

| metric (5-run mean) | chunk_size=64 (default) | chunk_size=32 |
|---------------------|------------------------:|--------------:|
| score               |                   880.7 |         878.8 |

Per-shape (single run, chunk_size=32 vs round-21 baseline):

| FP8 shape                | round-21 hk_TF | chunk=32 hk_TF | Δ      |
|--------------------------|---------------:|---------------:|-------:|
| Down-B4-M4096            |         1073.0 |         1041.8 | -31.2 (-2.9%) |
| GateUP-B4-M4096          |         1229.3 |         1207.8 | -21.5 (-1.7%) |
| Down-B4-M2048            |          783.3 |          772.0 | -11.3 (-1.4%) |
| GateUP-B4-M2048          |         1034.9 |         1030.6 |  -4.3 (-0.4%) |
| (B=32 cases all within ±5 TF noise)               |              |               |

**Conclusion**: chunk_size=32 REGRESSES B=4 cases by 1.4-2.9%; B=32
neutral. The natural strided WG-to-XCD assignment (chunk_size=64
silent no-op) outperforms the explicit swizzle on the 1-2-wave
persistent grids that the gpt_oss B=4 shapes generate. Reverted.

The discovery itself is still useful: it explains the `num_xcds ∈
{2, 4}` win pattern in round 7-21 sweeps. Future swizzle work
should explore `num_xcds=1` (block=64, swizzle = identity except
chunk reordering) or eliminate the swizzle entirely for B=4 cases
via a host-side knob.

## Why no commit-worthy lever was found

Score is at the noise floor:
* round-20 mean: 879.7 (range 878-882)
* round-21 mean: 880.7 (range 877-885)
* round-22 chunk-size attempt: 878.8 (range 876-881)

The Triton per-shape number drifts by 1-2pp run-to-run, swamping any
kernel improvement < 1% on a single shape. To push the score, one of
two things has to land:

1. **A real kernel-side gain ≥ 2-3% on a focused-segment shape**
   (kernels are now FLAT-clean and config-saturated; only
   architectural change is the BN=128 dispatch path for N=2880 Down
   shapes — this is a 1-2 round budget and may negate via grid
   doubling).

2. **A second shape's config flip** that the BUFFER-store kernel
   change re-aligned. Round-21 found 4 such flips on BF16; this
   round's sweeps confirm there are none left on FP8 or the 4
   unchanged BF16 shapes.

## Verification artifacts

* `/tmp/sweep_fp8_b32_round22.py` — 44-cell × 3-trial FP8 B=32 sweep
* `/tmp/sweep_round22_remaining.py` — 60-cell BF16 verify on 2
  remaining shapes
* `/tmp/probe_hk_vs_trt_overhead.py` — HK vs Triton execute()-body
  Python overhead comparison (HK 5.9-23.2 µs slower per call, but
  HK execute() body itself only adds 1.4 µs Python over the bare
  kernel call, so the gap is in the kernel time itself)
* `/tmp/probe_hk_execute_breakdown.py` — HK execute() per-piece
  timing decomposition on Down-B4-M4096 (bare kernel = 164 µs,
  cfg lookup +0.88, shape derive +0.28, contig-checks +0.28, total
  HK = 165.7 µs vs Triton 157.8 µs; gap is in kernel)
* HK `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  line ~2005 — chunk_size knob (currently 64, tested 32, reverted)

## Pending high-leverage levers (priority order for next rounds)

1. **`grouped_var_k_kernel_fp8` VGPR spill** (round-22 build report
   shows 52 bytes/lane scratch, 2 waves/SIMD): this is the FP8 dB
   backward kernel. Backward speed is invisible to score (metric
   only times forward) but a spill-eliminating refactor could land
   correctness gains and future-proof the dB path. Self-bench
   needed (`bench_grouped_gemm_turbo.py`).

2. **BN=128 dispatch path for N=2880 (Down) shapes** (still pending
   from round-20 plan): structural HipKittens kernel change to add
   a BN=128 grouped RCR template variant. Would convert the last
   N-tile's 25% utilization to 50%. Estimate 1-2 round budget; high
   risk if grid-doubling negates the per-tile efficiency gain.

3. **FP8 forward main-kernel ISA review for residual non-MFMA
   instructions**: the K-tail fuse path B (lines ~2400-2420 of
   `kernel_fp8_layouts.cpp`) is already maximally optimized (16
   buffer_loads upfront, single waitcnt, 4 sequential mfmas). The
   main loop's `rcr_8w_load_hoist` may have similar tightening
   opportunities — disasm-driven analysis would find them.
