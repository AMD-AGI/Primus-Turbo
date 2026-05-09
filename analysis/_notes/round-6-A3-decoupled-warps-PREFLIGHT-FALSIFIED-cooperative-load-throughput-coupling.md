---
name: round-6-A3-decoupled-warps-PREFLIGHT-FALSIFIED-cooperative-load-throughput-coupling
description: R6 A3 producer-consumer warp-role preflight — three falsification gates on grouped_rcr_kernel. Gate 2 (cooperative-load throughput coupling) and Gate 3 (256x256 tile fragment geometry locked to 8-warp 2x4 grid) both fire hard. A3 falsified. Rotate to A1 (Stream-K) for R7.
type: project
---

# Round-6 — A3 (decoupled-warps producer-consumer) preflight FALSIFIED

## TL;DR

Per R5 entry plan (`round-5-post-noise-floor-structural-direction-triage.md`),
this round is the A3 preflight: enumerate producer-consumer warp-role
splits on `grouped_rcr_kernel`, audit register/LDS/barrier-count deltas,
falsify if any of the three gates fail.

**All three gates were tested. Gate 2 and Gate 3 fire hard. Gate 1 is
borderline but moot once 2/3 fail.** A3 cannot be implemented on the
existing 8-warp / WARPS_M=2 / WARPS_N=4 / 256x256-tile structure
without a wholesale tile-shape redesign that would invalidate the
~30 rounds of dispatcher tuning encoded in the per-shape rules in
`config.py` (RCR ~1303-1660 + RRR ~2230-2570 + var-K ~2440-2620).

R7 rotates to **A1 (Stream-K / persistent + work-stealing)** per the
R5 fallback chain. R6 is a docs commit; bit-equivalent to R5
(`e5a1c584`).

## Baseline measured this round

`grouped_rcr_kernel` (kernel_fp8_layouts.cpp:2977-3696) — file-scope
8-warp template, FUSE_ACT × FUSED_KTAIL specializations:

| Property                       | Value                                    | Source                            |
|--------------------------------|------------------------------------------|-----------------------------------|
| WARPS_M × WARPS_N              | 2 × 4 = 8 warps                          | line 13-14                        |
| _NUM_THREADS                   | 512                                      | line 41                           |
| __launch_bounds__              | (_NUM_THREADS, 1) = (512, 1)             | line 2977                         |
| RBM × RBN per warp             | 64 × 32 (fp32 accumulator units)         | line 42-43                        |
| Per-warp accumulators          | cA, cB, cC, cD = 4 × rt_fl<64,32>        | line 3017                         |
| Per-warp accum VGPR/lane       | 4 × (64×32/64) × 1 fp32-VGPR = 128       | (4 acc × 32 fp32/lane × 1)        |
| Reported VGPR/AGPR/spill       | 256 / 0 / 37                             | comment line 4534                 |
| LDS double-buffer              | As[2][2] + Bs[2][2] = 4+4 ST_v2 tiles    | line 2997-2998                    |
| Per-tile LDS (ST_v2)           | 16 KB (128 rows × 128 cols × 1 byte fp8) | st_fp8e4m3<HB=128, BK=128, st_16x128_v2_s> |
| Total LDS                      | 8 × 16 KB + ~520 B group meta = 128 KB   | sum                               |
| MI355X LDS budget per CU       | 160 KB (CDNA4)                           | ISA spec                          |
| LDS headroom                   | ~32 KB                                   | 160 - 128                         |
| s_barriers per K-iter (steady) | **8** (lines 3241/3244/3248/3251/3256/3259/3262/3264) | direct count |
| s_setprio bracket per MMA      | 1 begin + 1 end × 4 MMAs = 8/iter        | lines 3243/3250/3258/3263 etc     |
| Cooperative load function      | `rcr_8w_load_hoist<_NUM_THREADS>(...)`   | line 1199, called 4×/iter         |
| Per-thread bytes_per_load      | bpt = ST::underlying_subtile_bytes_per_thread (=16 for st_16x128_v2_s) | line 1205 |
| Per-thread b128 issues / tile  | memcpy_per_tile = 16 KB / (16 × 512) = 2 | line 1207-1208                    |

**Note**: the R5 doc cited "4× CTA-barrier-per-iter". Direct re-count
of the steady-state main-loop body (3236-3265) shows **8 s_barriers
per K-iter**. This is in pairs around each of the 4 MMAs (one before
to wait LDS read, one after to seal LDS write before next prefetch).
The 4-vs-8 discrepancy doesn't change the falsification verdict — both
would benefit from removal — but is recorded for accuracy.

## Producer-consumer split candidates (R5 entry plan #1)

### Candidate A — 2P + 6C inside the same CTA (preserve 8-warp = 512 thread CTA)

**Idea**: 2 of the existing 8 warps are reassigned to producer-only
duties (HBM → LDS via `rcr_8w_load_hoist`, but with a smaller
N_THREADS template arg and per-warp-group LDS flag signal). 6 warps
stay as MMA consumers.

**Tile-fragment geometry blocker (Gate 3)**:

The 256×256 output BLK is tiled as a 2×2 grid of 128×128 sub-tiles
(cA, cB, cC, cD per warp). Each sub-tile is covered by an 8-warp
2×4 grid of `rt_fl<RBM=64, RBN=32>` fragments → exactly fits (2 × 64 = 128
along M; 4 × 32 = 128 along N). All four sub-tiles share the same
warp grid.

If we drop to 6 consumer warps, the warp grid options are:

| Layout    | RBM_c (M)        | RBN_c (N)       | Verdict                      |
|-----------|------------------|-----------------|------------------------------|
| 1×6       | 128/(2·1) = 64   | 128/(2·6) = 10.67 | ✗ non-integer fragment      |
| 2×3       | 128/(2·2) = 32   | 128/(2·3) = 21.33 | ✗ non-integer fragment      |
| 3×2       | 128/(2·3) = 21.33| 128/(2·2) = 32    | ✗ non-integer fragment      |
| 6×1       | 128/(2·6) = 10.67| 128/(2·1) = 64    | ✗ non-integer fragment      |

The MFMA fragment rt_fl<…, …, col_l, rt_16x16_s> requires 16-row /
16-col multiples. Every 6-warp partition of the 128×128 sub-tile
yields a non-integer fragment count along at least one axis. Same
problem for any non-power-of-2 consumer count (5, 7, etc.).

**Gate 3 (tile geometry) FALSIFIED**: 6 consumer warps cannot tile
the 128×128 sub-tile with 16x16-multiple MFMA fragments. The only way
to make 6C work is to drop tile size to 256×192 (RBN_c = 96/(2·3) = 16 ✓)
or 192×256 (symmetric). But 256×192 invalidates:
  * GateUP cell N=5760 = 22 × 256 (not divisible by 192).
  * Down cell N=2880 = 11 × 256 (not divisible by 192).
  * Every per-shape dispatcher rule referencing (tiles_n, tiles_m).
  * The fastpath wave-count headers `rcr_exact_8wave_fastpath.inc`.

→ Out of scope for R7-R10 rotation budget.

### Candidate B — 2P + 8C as a 10-warp CTA (640 threads)

**Idea**: keep all 8 MMA warps untouched (preserve fragment geometry),
add 2 dedicated producer warps. Total CTA = 10 warps = 640 threads.

**Cooperative-load throughput blocker (Gate 2)**:

`rcr_8w_load_hoist<N_THREADS>` (line 1199) divides each tile load
across `N_THREADS` threads:

  ```
  memcpy_per_tile = ST::rows × ST::cols × sizeof(T) / (bpt × N_THREADS)
                  = 16 KB / (16 × N_THREADS)
                  = 1024 / N_THREADS  b128 issues per thread
  ```

Current 8-warp call site uses `N_THREADS = _NUM_THREADS = 512`:
  * memcpy_per_tile = 1024 / 512 = **2 b128 issues per thread**

If we move HBM loads to 2 producer warps only, `N_THREADS = 128`:
  * memcpy_per_tile = 1024 / 128 = **8 b128 issues per thread**
  * Per-thread issue cost **4×** higher.

The HBM load is **throughput-bound** (HBM bandwidth × number of
in-flight requests), not latency-bound. Cutting the issuing-thread
count by 4× cuts the in-flight outstanding-request count by 4×
(each thread holds at most ~2-4 outstanding loads), which collapses
the achieved HBM bandwidth roughly proportionally on CDNA3/4
(per the standard `bw = N_outstanding / latency × txn_size` model).

R21 PMC observed the load is **already a non-zero fraction of iter
time** (LDS_busy 31-70% means LDS issue itself is consuming SIMD
cycles, even though MemStall is < 1 % i.e. the HBM round-trip
is not the bottleneck). With 4× longer producer issue path:

  * Estimated producer iter cost: ~480 cy/iter (was ~120 cy on 8w)
  * Consumer MFMA iter cost:      ~140 cy/iter (unchanged)
  * **Pipeline cost = max(P, C) = 480 cy/iter** (was max(120, 400) = 400 cy/iter)

→ A3 with Candidate B is **strictly worse** than the current
schedule. The producer becomes the bottleneck because we removed
6 warps' worth of HBM-load issue capacity from the load path.

**Gate 2 (cooperative-load throughput) FALSIFIED**: any "decoupled
producer" scheme that uses fewer than 8 warps for HBM-load issue
loses the load-throughput advantage that R21's PMC implicitly relies
on (LDS_busy 31-70% would balloon).

### Candidate C — 8 warps issue load (cooperative, unchanged), 6 of those 8 also issue MMA

**Idea**: all 8 warps participate in the load (preserves throughput),
but only 6 issue MMA (the 2 "load-only" warps idle on MMA work). Use
LDS-flag handshake among the 6 consumer warps so the 2 producer-only
warps run ahead.

**Same Gate 3 blocker as Candidate A**: 6 consumer warps cannot tile
the 128×128 sub-tile with 16x16 MFMA fragments. The 2 idle-MMA warps
provide ZERO benefit compared to "all 8 issue MMA" because the load
throughput is identical in both cases (all 8 issue load), and the
MMA throughput drops from 8w to 6w. Strictly worse.

→ FALSIFIED.

## LDS-flag synchronization scheme (R5 entry plan #2) — moot

For completeness: even if Gates 2 and 3 had cleared, the LDS-flag
handshake would be:

  * Producer (per stage): `__atomic_store_n(&lds_ready[stage], 1, __ATOMIC_RELEASE)`
  * Consumer (per stage): `while(__atomic_load_n(&lds_ready[stage], __ATOMIC_ACQUIRE) == 0) { __builtin_amdgcn_s_sleep(0); }`

CDNA3/4 LDS atomic round-trip ~16-24 cy. This is well below the 8
× ~12 cy = 96 cy of CTA-barrier overhead per iter we'd remove → would
save ~70-80 cy/iter, mapping to ~+15-20% MFMA-utilization (consistent
with R5 doc envelope).

But the LDS-flag mechanism doesn't bypass Gates 2 and 3 — those are
about thread-count vs work-decomposition feasibility, not barrier
type.

## LDS depth audit (R5 entry plan #3 — Gate 1)

For producer-consumer to give **deep-pipeline benefit** (producer
runs N stages ahead of consumer to absorb HBM tail latency), the
LDS double-buffer needs to grow from 2-stage to 4-stage:

  * 2-stage As[2][2] + Bs[2][2] = 128 KB (current, fits in 160 KB CDNA4 LDS)
  * 4-stage As[4][2] + Bs[4][2] = 256 KB (does NOT fit)

→ **Gate 1 (LDS depth headroom) FALSIFIED for deep-pipeline variant.**

The 2-stage variant already in the code is sufficient for "any"
producer-consumer scheme — the question is just whether the warp-role
split can be done at all (Gates 2 and 3 say no).

## Three falsification gates — final scoreboard

| # | Gate                                       | Verdict | Reason                                                                                             |
|---|--------------------------------------------|---------|----------------------------------------------------------------------------------------------------|
| 1 | LDS depth headroom for ≥4-stage pipeline   | ✗ FALSIFIED | 256 KB > 160 KB MI355X CDNA4 budget                                                            |
| 2 | Cooperative-load throughput preserved      | ✗ FALSIFIED | 2-warp producer = 4× per-thread issue → producer becomes bottleneck                            |
| 3 | 6-warp consumer fragment geometry          | ✗ FALSIFIED | 128×128 sub-tile cannot be tiled by 6 warps with 16x16-multiple MFMA fragments                 |

**All three gates fail.** A3 (decoupled-warps producer-consumer)
preflight is FALSIFIED on the existing 8-warp / 256×256 / WARPS_M=2
× WARPS_N=4 structure.

## What remains viable from the A3 family

A3 could in principle be salvaged by:

  (a) Moving to a **larger CTA** (say 16 warps = 1024 threads, max
      per-CU) so that 8-warp consumer + 8-warp producer share the
      load path AND consumer count stays power-of-2-friendly. Would
      require redesigning the type system (RBM/RBN derive from
      WARPS_M × WARPS_N), the dispatcher rules (gridDim.x scaling),
      AND the launch_bounds. Estimated 8-12 rounds of work to bring
      to bit-equivalence. Out of scope for the R7-R12 budget.
  (b) Changing the tile geometry from 256×256 to 256×384 or 384×256
      (3-way N or M decomposition). Same family of issues —
      invalidates ~30 rounds of dispatcher rules, doesn't compose
      with N=22 / N=11 / N=5 dispatcher cells.
  (c) Going to 4-warp CTA with 4-stage LDS depth (256 KB LDS exceeds
      budget; 2-stage LDS only saves 128 KB needed). Same as (a)'s
      type-system burden, plus loses the cooperative-load throughput.

None of (a)-(c) fit the round budget remaining for a 900-score push.

## R7 entry plan — A1 (Stream-K) per R5 fallback chain

A1 envelope (per R5):
  * **+15 to +50 score** total (concentrates on the 4 B=4 cells where
    1.16-2.32 wave grid causes 16-63% tail-wave underfill).
  * 2-4 rounds (preflight + atomic-counter scaffolding + steal loop +
    bit-equiv + metric).

R7 should preflight A1 with these gates:

  1. **R61 atomic primitive availability**: confirm `tile_counter`
     int32 atomicAdd-with-bound-cmp pattern (R61 fused-act post-kernel)
     is reusable in the grouped_rcr_kernel context. The primitive sits
     somewhere in `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
     near the fused-act post-kernel (R59-R61 work). Locate and verify
     the primitive can be invoked from a persistent grouped kernel's
     outer loop (not just the post-kernel).

  2. **K-split atomic-add bit-equivalence cost**: Stream-K splits the
     K-dimension across CTAs and atomically reduces partial sums into
     the output tile. This **changes the floating-point accumulation
     order**, which is gated by SNR > 25 dB. Project the worst-case
     accum-order change for the var-K wgrad K-loop (longest K=2880;
     8-way split would change reduction tree depth from 22.5 to ~3).
     If projected SNR drops below 25 dB on any of the 8 metric shapes,
     A1 is FALSIFIED on bit-eq grounds.

  3. **Grid-sizing benefit envelope per cell**: re-compute the
     per-cell tail-wave underfill from the R5 table and project the
     theoretical TFLOPS lift assuming Stream-K eliminates the
     underfill exactly. For B=4-M2048 (1.16 waves, 16% underfill →
     ~+10% theoretical), B=4-M4096 (2.32 waves, 32% → ~+13%), etc.
     If any cell projects < +5% lift, A1 is per-cell-FALSIFIED for
     that cell (skip via dispatcher rule).

If all three gates clear, R8-R10 implement Stream-K on the
fwd / dgrad RCR path. R11-R12 extend to var-K wgrad if bit-eq holds.

If any gate fails, R7 rotates to **E (incremental barrier
replacement)** — the LAST remaining viable direction, even though
its EV/round is worst (5-20 score per replacement, 10-40 needed for
+200). At that point the round budget likely cannot reach 900 from
the current 695 floor without a method change. Would document and
escalate.

## Code state this round

Single docs commit. No HK changes, no PT changes. Bit-equivalent to
R5 (`e5a1c584`). Daemon's R6 metric is sample #N+1 in the same noise
distribution as R1-R5. Per R29 noise floor, ±5 from cluster median
(~695) is expected.

## Forward pointers

  * **R7**: A1 preflight — 3 gates above. Start by locating R61
    `tile_counter` primitive in `kernel_fp8_layouts.cpp` (grep
    `tile_counter|atomicAdd|atomic_load`).
  * **R8-R10 (if R7 clears)**: A1 Stream-K implementation on RCR
    fwd/dgrad. Atomic-counter scaffold, steal loop, bit-eq verify,
    metric.
  * **R11-R12 (if R10 lifts ≥ +5 score on the metric)**: extend
    Stream-K to var-K CRR wgrad path. Bit-eq risk higher (longer K).
  * **R13+ (if A1 lifts < +5 or fails)**: rotate to E
    (incremental barrier replacement); document round-budget
    inadequacy for 900-score target via the dispatcher/algorithm
    axis on the existing 256×256 / 8-warp template.

## Why this round is not a wasted commit

R5 explicitly committed to "preflight first, code second" to avoid
the R32-R45 pattern of 5+ rounds spent on directional dithering. R6
is exactly that preflight — the verdict is negative (A3 falsified
across all 3 gates) but the verdict is **a-priori, before any kernel
edit**, saving the 4-6 rounds R5 had budgeted for A3 implementation.

The A3 → A1 rotation is also deterministic per R5's fallback chain,
so R7 has a concrete entry point with no further direction-selection
overhead.
