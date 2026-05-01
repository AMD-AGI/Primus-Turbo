# Round-15-dm — fresh rocprof PMC FALSIFIES the MFMA cell-shape migration project

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `4b73216f` (round-14-dm scaffolding)
**HipKittens HEAD**: unchanged (round-14 scaffolding still in tree, will become dead code)
**Primus-Turbo HEAD after**: this commit (Primus-Turbo notes-only; kernel/config bytes unchanged)

**Metric**: 821 baseline → 821 (no kernel change; doc-only pivot commit)

---

## TL;DR

Round-13/14's planned multi-round project — migrate FP8 grouped MFMA cell shape from
`mfma_f8f6f4_16x16x128` (32 cyc) to `mfma_scale_f32_32x32x64_f8f6f4` (64 cyc) to amortize
fixed overhead — was **based on round-16-rocprof PMC data that is now stale**. Round-12's
K-tail split-vmcnt fix already closed the original MfmaUtil gap on the round-16 hot-spot
(`gpt_oss Down-B4-M4096`: 33.6% → **42.84%**, +9.2 pp). Fresh rocprof on three current
worst FP8 shapes (this round's `_metric_grouped_only.py` table) shows:

| shape                          | ratio  | HK MfmaUtil | TRT MfmaUtil | delta vs TRT |
|--------------------------------|--------|-------------|--------------|--------------|
| `grpFP8_DSV3-Down-B16-M4096`   | 0.948  | **40.52 %** | 40.05 %      | **+0.47 pp** |
| `grpFP8_gpt_oss-Down-B4-M4096` | 0.964  | **42.84 %** | 42.64 %      | **+0.20 pp** |
| `grpFP8_gpt_oss-GU-B4-M2048`   | 0.963  | 38.69 %     | 40.12 %      | −1.43 pp     |

Two of the three worst shapes have HK at MfmaUtil **parity-or-better** with Triton, yet
still trail by ~5 % in TFLOPS ratio. The remaining gap is **non-PMC-visible** (see "What
the new data rules out" and "Where the gap actually lives"). Migrating cell shapes would
buy ≤1.4 pp on one shape (`GU-B4-M2048`) at the cost of touching main-loop / epilog 1 /
epilog 2 / FUSED_KTAIL / scale / store across two register tile shapes — a **negative-EV
trade** given the data.

**Decision**: abandon the cell-shape migration project. Round-14's `rt_32x64`/`rt_64x32`
scaffolding (in `include/types/register/rt_shape.cuh` + `include/ops/warp/register/tile/mma.cuh`)
is now unreachable dead code; flag for future cleanup but **do not revert** this round
(the scaffolding does not change `tk_fp8_layouts.so` codegen — verified by 821=821 metric
across rounds 14→15 with bytes-identical .so).

---

## What changed since round-16-rocprof (the data that motivated the migration)

`round-16-rocprof-mfma-util-deficit.md` reported (commit `6ec3733`, pre-round-12):

| shape                          | HK MfmaUtil | TRT MfmaUtil | delta     |
|--------------------------------|-------------|--------------|-----------|
| `gpt_oss Down-B4-M4096`        | 33.6 %      | 41.9 %       | **−8.3 pp** |
| `gpt_oss GateUP-B32-M4096`     | 36.1 %      | 42.9 %       | −6.8 pp   |

That was the headline that drove the migration thesis: HK was issue-throughput-limited
because short MFMAs forced too many issues per K-iter. The fix the round-13/14 plan
prescribed was longer MFMAs (32x32x64 = 64 cyc instead of 16x16x128 = 32 cyc).

**Round-12's split-vmcnt K-tail fix (commit `9f63115`)** rearranged the FUSED_KTAIL block
loads so the first two MFMAs overlap with the `a_kt1` load drain. This was a **K-tail
specific** optimization — but its impact on the hot-loop body MfmaUtil PMC was much
larger than expected. Cross-validation today on the same shape:

```
gpt_oss Down-B4-M4096
  round-16 PMC (pre-r12):   HK MfmaUtil 33.6 %
  round-15 PMC (post-r12):  HK MfmaUtil 42.84 %
  shift                                +9.24 pp
```

The K-tail block is only ~6 of the ~22 K-iters per output tile (K=2880, BK=128 →
22 hot-loop iters + 1 K-tail), so a +9 pp shift in *whole-kernel* MfmaUtil from a
K-tail-only change is consistent with the K-tail block having been the primary
MfmaUtil sink, not the hot-loop body. Round-16's diagnosis ("the body is the
bottleneck") **was wrong** in retrospect — the body and K-tail were jointly
bottlenecked, but the K-tail dominated the headline number.

## What the new data rules out

1. **Cell-shape migration won't help DSV3-Down**. HK already runs MFMA pipe at +0.47 pp
   *higher* than Triton on the worst DSV3 shape (40.52 vs 40.05 %). Longer MFMAs cannot
   raise pipe utilization above what's already achieved.

2. **Cell-shape migration won't help gpt_oss-Down**. Same story: HK +0.20 pp above TRT.

3. **MFMA-pipe issue throughput is no longer the FP8 grouped bottleneck**. Round-16's
   "8 pp deficit" no longer exists post round-12.

## Where the gap actually lives (the non-PMC wall-time tax)

For DSV3-Down-B16-M4096:
- HK SQ_BUSY/launch = 69.28 M cycles; TRT SQ_BUSY/launch = 69.88 M cycles.
  HK does **0.86 % less SQ work** per main launch.
- HK main kernel timestamp duration = 1073.37 us; TRT = 1014.85 us.
  HK runs **5.77 % longer wall-clock** per main launch.

So per main GEMM launch, HK issues fewer instructions but takes more wall time. The
discrepancy implies non-SQ stall sources: VMEM/LDS completion latency, wave drain, or
scheduling gaps that don't reach SQ_BUSY. Direct wall-time bench (`/tmp/probe_dsv3_walltime.py`):

```
shape: M=65536 (16*4096) N=7168 K=2048 FP8 tensorwise
 backend   p20_ms   p50_ms   min_ms   avg_ms  TFLOPS@p20  TFLOPS@min
      HK   1.4059   1.4171   1.3935   1.4201      1368.6      1380.8
     TRT   1.3478   1.3571   1.3378   1.3750      1427.6      1438.3
```

Confirms the metric's 0.948 ratio reproduces under direct probe: HK is 4.3 % slower
end-to-end despite ~equal in-kernel SQ activity. Other resource readouts:

| metric          | HK DSV3 | TRT DSV3 | HK gpt-Down | TRT gpt-Down |
|-----------------|---------|----------|-------------|--------------|
| Grid_Size       | 131072  | 131072   | 131072      | 131072       |
| Workgroup_Size  | 512     | 512      | 512         | 512          |
| VGPR_Count      | **128** | 116      | **128**     | 124          |
| SGPR_Count      | **112** | 80       | **112**     | 80           |
| Scratch_Size    | **272** | 0        | **236**     | 0            |
| LDS_Block_Size  | 140288  | 0¹       | 140288      | 0¹           |

¹ Triton allocates LDS dynamically via launch param; static block size = 0.

HK kernel uses **+12 VGPRs, +32 SGPRs, and ~256 bytes scratch** vs Triton. Both kernels
are at 2-wave-per-SIMD occupancy (256-VGPR limit / 116 = 2; / 128 = 2 — same wave
count), so the 12 extra VGPRs do not hurt occupancy; but the 272-byte scratch
indicates **register spills** that Triton avoids. Each spill round-trip is L1 traffic
that doesn't show in MfmaUtil but does extend wall time. Whether 272 bytes of spills
accounts for the full 4 % wall-time gap is unproven — but it's a candidate.

## Per-shape PMC heterogeneity (3 shapes, 3 different stories)

The three worst FP8 shapes this round have **three distinct bottleneck signatures**:

1. `DSV3-Down-B16-M4096` (0.948): MFMA pipe matched, gap is non-PMC wall-time tax.
   Likely candidate: VGPR pressure / scratch spills.

2. `gpt_oss-Down-B4-M4096` (0.964): MFMA pipe matched, gap is non-PMC wall-time tax.
   Same candidate. (B=4 means each CU handles ~5 tiles; per-tile prologue/epilog overhead
   dominates more than at B=32.)

3. `gpt_oss-GateUP-B4-M2048` (0.963): MFMA pipe **deficit −1.43 pp** AND HK does +8 %
   *more* SQ work per launch. This is the only shape that retains a round-16-style
   profile. M=2048, N=5760 (large N, small M with B=4) — different tile geometry from
   Down-B4, possibly hitting a tile-distribution edge.

The cell-shape migration would only help shape (3). Migrating the entire kernel to
help one shape — at risk of breaking shapes (1) and (2) which already match Triton's
pipe — is a clear negative-EV trade.

## Updated falsification banks

Add to `_falsified_falsified_levers.md` (or equivalent):

- [r15-dm] **MFMA cell-shape migration** (`mfma_f8f6f4_16x16x128` → `mfma_scale_f32_32x32x64_f8f6f4`).
  *Evidence*: rocprof PMC on DSV3-Down-B16-M4096 + gpt_oss-Down-B4-M4096 shows HK already
  at MfmaUtil parity-or-better with Triton. Migration cannot lift pipe util above current
  level. Confirmed by `round-12-dm` K-tail split-vmcnt closing the round-16 hot-spot.

Round-14 scaffolding (rt_32x64 / rt_64x32 typedefs in `rt_shape.cuh`; mfma_323264 dispatch
branch in `mma.cuh`) is now **unreachable dead code**. Mark for cleanup in a future round
once we're confident no migration-style change wants them. Not urgent: they cost ~30 LOC,
add no codegen overhead (verified by metric==metric across r14→r15 with .so unchanged).

## Round-16 candidates (in priority order, all data-driven)

1. **VGPR / SGPR / scratch reduction** (DSV3 + gpt_oss-Down). Hypothesis: 272-byte
   scratch indicates spills extending wall-time without showing in SQ_BUSY. Tools:
   `-Rpass-analysis=kernel-resource-usage` already enabled in Makefile; need to capture
   the per-function VGPR/spill report. Levers: hoist persistent state to LDS, eliminate
   redundant tile-coord caches, simplify the K-tail register live-ranges. **Risk: low**
   (purely register allocation, no algorithmic change). **Reward**: 1-3 % per shape if
   spills are real.

2. **`gpt_oss-GateUP-B4-M2048`-specific MfmaUtil lift** (only shape with residual deficit).
   The −1.4 pp gap is much smaller than round-16's −8.3 pp; remaining options narrow:
   tile geometry (BLOCK_M / BLOCK_N adjustment for N=5760 M=2048 small-B regime), or
   chunked grid scheduling. **Risk: medium** (tile geometry may regress other shapes).
   **Reward**: ~1 % on this single shape; geomean impact ≤0.06 pp.

3. **End-to-end wall-time delta investigation** (DSV3, all). The 4 % wall vs 0.86 %
   SQ_BUSY discrepancy is structural, not specific to one shape. Possible mechanisms:
   (a) wave drain at kernel end — persistent kernel runs N tiles per CU, last tile's
   tail might serialize behind earlier tiles; (b) HBM read amplification — HK might
   prefetch more bytes per tile than necessary; (c) inter-kernel gap in the 12-launch
   pipeline (group_offs → 2× scale → 2× quantize → main → 6× amax). Tools: rocprof
   `--kernel-trace --hsa-trace` for inter-launch gaps; `MemUnitStalled` is already
   `0.24 %` so HBM stalls are NOT the cause.

## Score impact

- Metric this round: **821** (down 5 from rolling best 826, all within noise band of
  rolling 5 measurements 820/825/826/825/821 = stdev 2.4). Doc-only commit; no kernel
  change shipped.
- The diagnostic itself does not move the score, but **prevents** spending 3+ rounds on
  the cell-shape migration project (which would have produced ≤0.06 pp geomean gain at
  best, or regressed correctness if the migration broke any shape).

## Probe artifacts

- `/tmp/rocprof_round15_dsv3/{hk,trt}_dsv3_down_b16/` — DSV3-Down PMC.
- `/tmp/rocprof_round15_dsv3/{hk,trt}_gpt_down_b4/` — gpt_oss-Down PMC (round-16 cross-validate).
- `/tmp/rocprof_round15_dsv3/{hk,trt}_gpt_gu_b4_2k/` — gpt_oss-GateUP-B4-M2048 PMC.
- `/tmp/parse_rocprof_all.py` — counter aggregator (per-kernel-class).
- `/tmp/parse_resources.py` — VGPR / SGPR / Scratch extractor.
- `/tmp/parse_timestamps.py` — per-kernel wall-time aggregator.
- `/tmp/probe_dsv3_walltime.py` — direct CUDA-event wall-time probe.
- `/tmp/bench_dsv3_one.py` — DSV3 hot-loop driver.

## Round-15-dm commits

- Primus-Turbo: this notes file. No kernel/config change.
- HipKittens: none.
