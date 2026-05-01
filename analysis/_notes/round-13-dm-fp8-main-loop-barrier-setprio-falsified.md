# Round-13-dm — FP8 grouped main-loop barrier + setprio(0) drop both FALSIFIED

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `9f631156` (round-12-dm K-tail split-vmcnt)
**HipKittens HEAD before**: `25386036` (round-12-dm K-tail split-vmcnt)
**HipKittens HEAD after**: notes-only (kernel reverted to round-12 state)
**Primus-Turbo HEAD after**: this commit (Primus-Turbo notes-only; kernel/config bytes unchanged)

**Metric**: 826 baseline mean → 826 (no functional change; both probes reverted)

---

## Round target (per the round-13 prompt rule)

Lowest FP8 ratio shape after round-12 = **`grpFP8_DeepSeek-V3-Down-B16-M2048` at
0.951** (HK 1142.4 TF / Triton 1201.2 TF). All 4 DSV3-Down shapes
(0.951–0.975, K=2048 N=7168, fully aligned, no FUSED_KTAIL, no N-tail)
now dominate the gap. Round-12's K-tail split-vmcnt only addressed the
gpt_oss FUSED_KTAIL=true variants and was correctly invariant on DSV3.

The DSV3-Down shapes are pure main-loop bound: `grouped_rcr_kernel`
main loop body runs at MFMA Util ≈ 33.6 % (Triton ≈ 41.9 %, –8.3 pp,
**round-16 PMC** in `analysis/_notes/round-16-rocprof-mfma-util-deficit.md`).
Round-16 listed two **untried** main-loop levers:

1. Drop `s_setprio(0)` post-MFMA (4 × per K-iter)
2. Drop one or more main-loop `s_barrier`s (4 post-MFMA per K-iter)

Round-25's saturated-knobs inventory does **not** list either, so both
were genuinely unprobed. This round tested both, in priority order.

## Why this looked promising (model)

Per K-iter the FP8 grouped main loop has 8 `__builtin_amdgcn_s_barrier()`
calls for 4 short MFMAs (`mfma_f8f6f4_16x16x128_f8f8`, ~32 cyc each):

```
section_A: load_b(b0) → load_a(a) → prefetch → TK_WAIT_LGKM(4) → s_barrier (1)
           → lgkmcnt(0) → setprio(1) → mfma cA → setprio(0) → s_barrier (2)
section_B: load_b(b1) → prefetch → s_barrier (3)
           → lgkmcnt(0) → setprio(1) → mfma cB → setprio(0) → s_barrier (4)
section_C: load_a(a)  → prefetch → s_barrier (5)
           → lgkmcnt(0) → setprio(1) → mfma cC → setprio(0) → s_barrier (6)
section_D: prefetch → TK_WAIT_VMCNT(8) → s_barrier (7)
           → setprio(1) → mfma cD → setprio(0) → s_barrier (8)
```

Modeling per K-tile cost: 4 × 32 cyc MFMA + 8 × 30 cyc s_barrier ≈
128 + 240 = **368 cyc**, MfmaUtil = 128/368 = **34.8 %**, matching the
observed 33.6 %. The same model on BF16 (`mfma_*_bf16` ~64 cyc, same
8 barriers) yields 256/(256+240) = 51.6 %, matching observed 50.8 %.
**The model implies that halving FP8's barriers would nearly double
MfmaUtil and close the entire 0.951 → 1.0 ratio gap on DSV3-Down.**

Round-13-dm tested this hypothesis directly.

## Probe 1 — Remove the 4 post-MFMA s_barriers (lines 2182, 2189, 2196, 2201)

### Setup

Kept all 4 *pre*-MFMA barriers (lines 2179, 2186, 2193, 2199, after each
section's load + prefetch + lgkm/vmcnt drain) — those are obviously
load-bearing because the next section's collective `rcr_8w_load_hoist`
writes to a shared LDS slot that this section's `load_b`/`load_a` is
reading. Removed only the 4 *post*-MFMA barriers (between the MFMA's
`setprio(0)` and the next section's first instruction).

Hypothesis (now falsified): each MFMA writes only this wave's VGPR
accum (cA/cB/cC/cD), and the next section's loads read VGPR-disjoint
slots (`b1`, `As[tic][1]`, `b_tile(tic, 0)` re-prefetched 2 iters ago)
already drained by the prior iter's `RCR_STEADY_VMCNT=8` cap. Removing
the post-MFMA barrier should let the compiler hoist the next section's
loads/prefetch *across* the MFMA, hiding ~30 cyc × 4 = ~120 cyc of
barrier latency per K-iter (≈3 % of K-tile wall on DSV3-Down ki_dyn=16).

### Result

```
[metric_grouped_only] grp_FP8 vs triton geomean=0.0133 (n=16)
correctness FAIL shapes: 15/16
```

**15/16 FP8 shapes failed forward-SNR** (SNR 14.3 dB – 23.7 dB, threshold
25 dB). Score **826 → 11**.

### Why the hypothesis was wrong (mechanism)

The breakage is on the **collective** nature of `rcr_8w_load_hoist`. Each
wave issues its own `buffer_load_b128` ops to fill its own portion of the
shared LDS slot. The LDS layout for `b_tile(tic, 0)` (and similar) is
**cross-wave-interleaved**: wave A writes lanes 0/2/4/…, wave B writes
lanes 1/3/5/…  When the next iter's `load_b(b0, b_tile(tic, 0), wn)`
fires per-wave, each wave's `ds_read_b128` pulls 128 bits per lane —
which means each wave's read **straddles addresses written by the
*other* wave**. For the read to be correct, *all* waves' prefetch DMA
must have drained.

Without the post-MFMA `s_barrier`, wave A can race ahead into the next
section's prefetch issue while wave B is still finishing its MFMA. The
LDS slot ends up partially populated when wave A's later iter reads it,
producing the observed 14–24 dB SNR floor (consistent with mild lane-
interleaved corruption, not total garbage). Pre-MFMA barriers do not
fix this: they sync waves at the lgkmcnt drain *before* the MFMA, but
the LDS slot reuse race is *after* the MFMA, when the next section's
prefetch issues a write into the slot wave A's prior `load_b(b0)` was
reading from.

**Verdict**: post-MFMA `s_barrier`s are load-bearing for cross-wave
LDS-slot-reuse coherence. Cannot be removed in isolation. Reverted.

## Probe 2 — Drop the 4 post-MFMA s_setprio(0) calls

### Setup

Kept all `s_setprio(1)` before each MFMA. Removed `s_setprio(0)` after
mfma cA / cB / cC (kept the very last one after cD, mostly for symmetry
with the `s_setprio(0)` at the end of every other epilog). The wave
keeps priority 1 across the next section's load + prefetch + waitcnt +
barrier window until the next setprio(1) tags the following MFMA.

Hypothesis: raising priority during the load+prefetch+barrier window
makes *this* wave's vmem queue pull HBM data faster.

Round-2 falsified removing `setprio` *together with* `sched_barrier`
(–5.6 %); round-16 noted that setprio in isolation was never tested.
Round-13-dm tests this gap.

### Result (5-run median)

```
runs    = [818, 814, 818, 814, 816]
mean    = 816.0
median  = 816
stdev   = 2.0
range   = 814 – 818
baseline mean = 826
Δ      = -10 score (5/5 runs below baseline 822-826 band)
```

**Δ = –10 score, 5/5 runs below baseline.** Outside the noise band
(baseline 1-σ ≈ 1.3 from round-14 calibration). Falsified.

### Why the hypothesis was wrong (mechanism)

At occupancy 2 waves/SIMD, the `s_setprio(1)` is a **wave-priority
hint** to the issue arbiter — when both this wave and the co-occupant
wave are eligible, this wave wins. The post-MFMA `s_setprio(0)` is the
key handoff point: it returns the wave to default priority during the
next section's load + prefetch + barrier window, *letting the
co-occupant wave's MFMA fire while we're load-bound*. Removing
`s_setprio(0)` causes this wave to hog the issue slot through the entire
load/prefetch/wait/barrier window; the co-occupant wave's MFMA
backs up behind ~30 cyc of "we're loading, you wait" hints.

Net: this wave's loads complete *slightly* earlier, but the co-occupant
wave's MFMA throughput drops by more — the SIMD's dual-issue pipeline
loses its overlap. The score regression confirms the latter dominates.

**Verdict**: post-MFMA `s_setprio(0)` is load-bearing for 2-wave
co-occupant scheduling. Cannot be removed. Reverted.

## Updated saturated-knobs inventory (cumulative through round-13-dm)

| knob | direction tested | result | round |
|---|---|---|---|
| `RCR_INIT0_VMCNT` | both | saturated; raise races | 24 |
| `RCR_INIT1_VMCNT` | both | saturated; raise races | 24 |
| `RCR_STEADY_VMCNT` | both | saturated | 5 |
| `RCR_PREFETCH_LGKM` | both | saturated | 5 |
| `RCR_EPILOGUE_VMCNT` | both | saturated; raise races | 25 |
| `RCR_TWO_TILE_MID_VMCNT` | rounds 6, 15 | dead-end (grouped path) | 6, 15 |
| `RCR_TWO_TILE_MIN_KI` | round 15 | no-op for grouped | 15 |
| `RCR_MAIN_UNROLL` | {1, 2, 4} | identical asm | 16, 3-dm |
| main-loop `sched_barrier(0)` | round 4 | **removable +0.45 pp** | 4 (removed) |
| main-loop `s_barrier` post-MFMA | round 13-dm | **load-bearing** (-100% breaks) | **this rd** |
| main-loop `s_setprio(0)` post-MFMA | round 13-dm | **load-bearing** (-10 score) | **this rd** |
| epilog `sched_barrier(0)` | round 25 | **load-bearing** (-1 pp) | 25 |
| FUSED_KTAIL block load order + split-vmcnt | round 12-dm | **+13 score** | 12-dm (committed) |
| FUSED_KTAIL block 3-stage vmcnt split | round 13-dm | dropped, ≤4 cyc/tile saving | analysis only |
| `chunk_size` (chiplet) | round 22 | 64 baseline optimal | 22 |
| `(group_m, num_xcds)` | rounds 1-dm, 10-dm, 21, 23 | per-shape table converged | 21, 23, 10-dm |
| `BLOCK_SWIZZLE_NUM_XCDS` | rounds 12-14 | 8 = MI355X HW | 14 |
| BN=128 dispatch path | round 24 | falsified | 24 |
| FP8 dense 2-tile main loop port | round 6 | catastrophic (-144) | 6 |
| FP8 LDS `ST_v2`→`ST_v3` swap | round 2-dm | falsified (SNR fail) | 2-dm |
| FP8 WARPS_M/WARPS_N flip 2/4→4/2 | round 6-dm | catastrophic | 6-dm |
| FP8 grouped parallel LDS-init for s_offs | round 9-dm | perf no-op | 9-dm |
| FP8 grouped launch_bounds MIN=2 | round 5-dm | no-op | 5-dm |
| FP8 grouped launch_bounds MIN=4 | round 7-dm | no-op | 7-dm |
| Host overhead trim (Python) | round 22 | 1.7 µs ≈ 1 % B=4 wall | 22 |

**All single-knob 1-round levers exposed to the FP8 grouped RCR kernel
are now saturated.** No remaining 1-round main-loop knob to test.

## Why the round-16 model overestimated removability

The simple model "each barrier costs 30 cyc, MfmaUtil = mfma / (mfma +
overhead)" treats barriers as **detachable serial overhead**. The
reality on a 2-wave-occupancy SIMD with collective LDS prefetch is that
barriers are doing **two distinct jobs at once**:

1. **Compute-side overlap gating**: the MfmaUtil model. Removing this
   role *would* save the modeled cycles.
2. **Coherence-side gating**: forcing all waves to a synchronization
   point before subsequent collective LDS writes (probe 1) and forcing
   a wave-priority handoff for the co-occupant wave's MFMA throughput
   (probe 2).

The simple model captures only role (1). The 2 falsifications above
are both role-(2) effects that the round-16 PMC counters cannot
distinguish from role (1). Lesson: **MfmaUtil deficit ≠ removable
overhead** when the deficit is a coherence-gate cost, not a scheduling
slack.

## What's left for round-14+ (multi-round, structural)

All single-knob levers are now exhausted. Round-14+ must commit to one
of the **multi-round structural projects** previously listed. In
priority order:

### 1. FP8 MFMA cell-shape `16x16x128` → `32x32x64` (2-3 rounds)

HK uses `mfma_f8f6f4_16x16x128_f8f8` (8w warp tile, 8 MMA / wave, ~32
cyc/MFMA). Triton uses `mfma_scale_f32_32x32x64_f8f6f4` (4 MMA / wave,
2× the K per MFMA, ~64 cyc/MFMA). With 2× MFMA latency, **the same 8
barriers per K-tile would amortize over 2× the MFMA cycles**, raising
MfmaUtil from 34.8 % to ~52 %. This *directly attacks* the bottleneck
the round-13-dm probes failed to exploit via barrier removal. Cost:
re-derive `RBM/RBN` and lane-cell mapping; `St_subtile`'s 4-lane HW
transpose layout must be re-mapped because cell-shape change alters
which lanes own which (m, n) accum coords.

### 2. K-tail epilog amortize across multi-tile-M (2 rounds)

Persistent wg processes multiple BM-rows per outer K-iter, sharing one
K-tail epilog across them. Reduces K-tail relative cost from 2-4 % →
0.5-1 %. Helps **only** the 8 K-misaligned gpt_oss shapes (K=2880
K_REM=64). Risk: VGPR pressure (multiple BM-rows × MMA = +N VGPRs/wg).

### 3. AGPR migration (high-risk, multi-round)

Round-8-dm flagged this as a candidate. Migrate hot accum VGPRs to
AGPRs, freeing VGPRs for occupancy. Risk: register-class plumbing for
mfma operands.

### 4. K-tail single-load merge (2 rounds; gpt_oss-only)

Currently path B does an extra `rcr_8w_load_hoist` for the K-tail block
before fold-into-cA/cB/cC/cD. Folding the K-tail vmem into the
last-K-tile prefetch slot inside Epilog 2 would amortize. Path B and
Epilog 2 schedule co-design.

## Recommendation for round-14

Start **option 1 (MFMA cell-shape change)**. It's the only one with the
mechanism to address the DSV3-Down 0.951 ratio, which is the *current*
bottleneck (4/16 FP8 shapes at 0.951–0.975 vs Triton). Round-14 step 1
should be a contained scaffold:
- Locate every site in `kernel_fp8_layouts.cpp` that hard-codes
  `mfma_f8f6f4_16x16x128` or its 8-MMA-per-warp tile shape.
- List the `RBM × RBN` derivation and the `St_subtile` lane mapping.
- Plan the single-section migration (one section first, verify SNR,
  then expand).

Failure mode is graceful: if intermediate compile breaks, partial
progress is committable as a `WIP` round.

## Files / commits

* HipKittens: no commit this round (both probes reverted to round-12-dm
  baseline `25386036`; kernel byte-identical to that SHA).
* Primus-Turbo: this commit —
  `analysis/_notes/round-13-dm-fp8-main-loop-barrier-setprio-falsified.md`.

Self-bench: not required (no backward path touched; only forward kernel
probes which round-12 metric covers, both reverted post-test).
