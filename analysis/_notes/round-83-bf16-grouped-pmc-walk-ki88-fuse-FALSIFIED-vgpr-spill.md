# Round 83 — bf16 grouped GEMM weighted wall

> **Context:** auto_optimize round 6 / 100, MI355X. Continuation of the
> R82 PMC instrumentation pivot.

**Status:** PMC instrumentation **DONE** + **R83 KI=88 FUSE
specialization FALSIFIED** (9-VGPR spill cancels full-unroll gain;
3-run mean 882.3 vs baseline 882-883).

| run | weighted score | gpt_oss geomean |
|-----|---------------|-----------------|
| baseline (R82 commit ec03006) | 882-883 (±2 noise) | 1.0934 |
| R83 KI=88 FUSE attempt        | 881 / 883 / 883    | 1.0907 / 1.0940 / 1.0954 |
| post-revert (back at ec03006) | 881               | 1.0906 |

Net delta = within noise band (±2 score, ±0.005 gpt_oss geomean).
Required gain ≥ +5 → **REVERT**.

## Part A: PMC walk on gpt_oss-Down-B4-M2048 (R82's R83 plan)

Ran `rocprofv3 --pmc LDSBankConflict MfmaUtil MemUnitStalled
OccupancyPercent --kernel-include-regex grouped` on the target shape
(B=4 M=2048 N=2880 K=2880, fwd+bwd, 13 iter post-warmup).

Per-kernel mean PMC values (DSV3-GateUP-B4-M1024 warmup excluded):

| kernel                          |  n  |  dur_us | LDSBC% | MfmaU% | MemSt% |  Occ%  |
|---------------------------------|-----|---------|--------|--------|--------|--------|
| `grouped_var_k_kernel<RCR>`     | 14  |  4135   |  16.0  |  42.6  |  0.11  |  16.4  |
| `grouped_kernel<RCR,0,FUSE>`    | 26  |  3811   |   0.0  |  43.0  |  0.15  |  15.9  |
| `grouped_kernel<RRR,64>` warmup |  1  |  1744   |   9.3  |  74.0  |  0.08  |  23.2  |
| `grouped_kernel<RCR,112>` warmup|  1  |  1359   |   0.0  |  64.7  |  0.05  |  20.7  |

**Findings:**

1. **LDS bank conflict is NOT the dominant bottleneck.** The RCR FUSE
   kernel runs at LDSBC=0% — perfect LDS layout. The var-K kernel is
   at LDSBC=16% — moderate, but well under the 217M absolute count
   R68 reported (R68's claim was extrapolated from a smaller workload
   slice; the per-kernel-time fraction is the right denominator).
2. **HBM bandwidth is NOT the bottleneck.** MemUnitStalled = 0.11%
   (var-K) and 0.15% (RCR FUSE) — kernels are compute-bound, not
   memory-bound.
3. **MFMA utilization IS the bottleneck.** The two gpt_oss-Down hot
   kernels are at 43% MFMA, vs 74% for DSV3 warmup `KI=64`. The
   gpt_oss kernels are leaving ~30 percentage points of MFMA pipe
   idle — the comparable DSV3 / Qwen3 KI-specialized variants close
   that gap via compile-time-bound full-unroll.
4. **Occupancy is low (16% on gpt_oss FUSE) but matches the build
   report's 2 waves/SIMD ceiling.** All BF16 kernels in this codebase
   sit at 256 VGPRs / 2 waves/SIMD — not a per-kernel regression,
   structural ceiling. Increasing waves would require dropping to
   ≤128 VGPRs which has historically failed (R74's swap probe spilled
   66 VGPRs).

### Verdict

R68's "LDS bank conflict is the var-K bottleneck" claim is
**REFUTED** by direct PMC measurement on this round's binary. The
custom-LDS-swizzle direction R82 floated for R84+ would not have moved
the metric.

The actual bottleneck is **MFMA pipe idle cycles caused by KI=0
generic + #pragma unroll 2** on the FUSE path. KI=64-specialized
DSV3 kernels show this gap is closeable in principle: 64.7-74% MFMA
util on warmup vs 43% on gpt_oss. That gap = ~+50% throughput at
constant (memory-bound, occupancy-bound) ceiling.

## Part B: R83 lever — KI=88 specialization for the FUSE template

**Hypothesis:** add `grouped_kernel<RCR, 88, FUSED=true>` so the
gpt_oss K=2880 path takes a compile-time-bounded full-unroll over 88
main_loop_iter calls (matching the R52 KI=88 / FUSED=false spec but
with the K-tail epilog block live). KI=88 = `g.ki = 2816/32 = 88`,
which all 8 gpt_oss shapes hit (any RCR shape with K%128=64 and
K_main=2816). The forward path AND the post-R80 H4-rerouted dA path
both go through `grouped_kernel<RCR, ?, FUSE>`, so the specialization
benefits both directions.

**Build report:**

| kernel                          | VGPRs | Spill | Scratch B/lane | Occ |
|---------------------------------|-------|-------|----------------|-----|
| `grouped_kernel<RCR,0,FUSE>` (existing baseline) | 250 | 0 | 0 | 2 waves/SIMD |
| `grouped_kernel<RCR,88,FUSE>` (R83 attempt)      | 256 | **9** | **40** | 2 waves/SIMD |
| `grouped_kernel<RCR,88,non-FUSE>` (R52 reference)| 256 | 1 | 8 | 2 waves/SIMD |

The KI=88 / FUSE variant lands at the 256 VGPR ceiling AND spills 9
extra VGPRs to scratch — much worse than R52's KI=88 non-FUSE (1
VGPR spill, 8 B/lane scratch). The K-tail epilog block lives across
the 88-iter main-loop full-unroll, holding ~6-8 extra registers
through the entire kernel body. Path B's HBM-direct K-tail load
(`buffer_load_b128` + register-direct DO_MMA) needs 4 in-flight A_tile
+ 4 B_tile + zero-init shadow regs visible in epilog 2's exit state.

**Metric** (3 runs, post-build):

| run | score | gpt_oss geomean | DSV3 | Qwen3 |
|-----|-------|-----------------|------|-------|
| 1   |  881  | 1.0907          | 1.1224 | 1.1126 |
| 2   |  883  | 1.0940          | 1.1224 | 1.1111 |
| 3   |  883  | 1.0954          | 1.1222 | 1.1110 |
| **mean** | **882.3** | **1.0934** | 1.1223 | 1.1116 |

**Pre-change baseline (today, run 1):** 882 / 1.0934 / 1.1203 / 1.1136.
**Post-revert (binary back at ec03006):** 881 / 1.0906 / 1.1219 / 1.1112.

Both the change and revert sit inside the same ±2 score / ±0.005
gpt_oss geomean noise band — no signal. **The full-unroll gain is
exactly compensated by the 9 VGPR spills + 40 B/lane scratch
overhead.** Spills hit ~10-cycle latency per access; 9 spill regs
× ~3-4 hot-path uses per K-tile = ~25-35 hot-path scratch ld/st per
main loop iter, i.e. ~55-77 cycles/iter overhead vs ~150 cycles of
K-tile compute saved by full-unroll → net zero at the kernel level,
matching the metric.

### Verdict

**FALSIFIED**. KI=88 FUSE template's K-tail epilog block raises live
register count past the 256 VGPR ceiling, forcing 9-VGPR spills that
cancel the full-unroll instruction-scheduling gain. The simpler
fully-fused spec is not viable until the FUSE epilog's live state is
trimmed.

## Part C: direction for R84

PMC + R83 jointly map out the surface:

* **MFMA util is recoverable** in principle (DSV3 KI=64 hits 74%;
  gpt_oss FUSE sits at 43%; ~30pp = ~+50% ratio room).
* **Naive KI specialization fails** because the FUSE epilog block
  pushes live state past the 256 VGPR ceiling (R83).
* **Splitting the FUSE epilog into a standalone tail launch** would
  let the main kernel use KI=88 non-FUSE (R52, no spill) BUT
  reintroduces the K-tail RMW double-launch wall (= the original
  ~30-35% wall-time penalty FUSE was added to eliminate).

Two structural levers worth a look in R84:

1. **Trim the FUSE K-tail epilog's live state** so KI=88 / FUSE fits
   in 256 VGPRs without spill. The K-tail HBM-direct load pulls 4 +
   4 = 8 reg tiles (~64 VGPRs) live through epilog 2; if we can
   issue the K-tail load INSIDE epilog 2 (overlapping the 8 K-tile-
   N-2 stores with the K-tail HBM fetches) we free up the 8 reg
   tiles' live-range straddle. ~2-3 day kernel surgery, high risk
   (compiler may re-spill the moved tiles).
2. **Investigate why DSV3 KI=64 RRR hits 74% vs gpt_oss KI=0 RCR
   FUSE 43%.** Layout (RRR vs RCR) controls the MFMA register-tile
   shapes; the FUSE epilog's 8 K-tile reuse may be pessimised by
   RCR's `rt_16x32_s` B-tile vs RRR's `rt_32x16_s`. PMC vs PMC
   pair-test on a controlled shape (B=4 M=2048 N=4096 K=4096 RCR
   FUSE forced + RRR forced) would isolate the layout effect from
   the FUSE-eligible vs non-eligible effect — orthogonal probe to
   R83's KI specialization.

R84 candidate: lever (2) — narrower-scope PMC pair test (no kernel
change, ~30 min), confirms whether layout or FUSE epilog is the
binding constraint before committing to the bigger lever (1).

## Files touched

* `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  — KI=88 FUSE template instantiation + dispatcher switch case.
  **REVERTED** (file matches ec03006 baseline; `git diff` clean).
* This round note (Primus-Turbo).
* `/tmp/r6_pmc_target.py`, `/tmp/r6_analyze_pmc.py`,
  `/tmp/r6_pmc_counters.txt`, `/tmp/r6_pmc_out/pmc_1/r6_varK_counter_collection.csv`
  — PMC artifacts (offline only, not committed).

## Suggestion for R84

Run a PMC pair test forcing gpt_oss-Down-B4-M2048 through:
* RCR FUSE (current path) — `KI=0`, MfmaU=43%
* RCR non-FUSE + standalone K-tail kernel — to compare MFMA util
  in the main-loop body without the FUSE epilog's live-state weight

If non-FUSE main + K-tail standalone hits MfmaU > 50% in the main loop,
the FUSE epilog's register pressure is the binding constraint and
lever 1 (epilog live-state trim) becomes the next attack. Otherwise
the layout-level limit is binding and we should pivot to dB var-K
swizzle (R82's original R84 plan, now properly scoped to the var-K
kernel where LDSBC=16% — a smaller absolute lever but at least real).
