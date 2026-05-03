# Round 14 — BF16 grouped KI_HINT=44 + FUSED_KTAIL=true specialization (FALSIFIED, REVERTED)

**Status:** FALSIFIED — Lever A2c (full-unroll the K=2880 main loop via a
KI_HINT=44 + FUSED_KTAIL=true specialization) is correctness-clean (SNR
49.58 dB on all 6 probe shapes) but **regresses metric -30 score**
(843 vs 873 baseline) due to 28-30 VGPRs spilling to scratch. All 8
gpt_oss BF16 grouped shapes slowed; gpt_oss family geomean dropped
1.074 → 1.012.

## Observation that motivated the round

Reading `kernel_bf16_dynamic.cpp` lines 3984-4010 + 4203-4222 revealed
that K=2880 (gpt_oss BF16 grouped) routes to
`grouped_kernel<L, 0, true>` (KI_HINT=0 dynamic, FUSED_KTAIL=true) via
`launch_one_grouped_fuse<L>`. Inside `device_gemm_tile_body`
(line 689-707), the KI_HINT=0 branch uses `#pragma unroll 2` on the
main loop:

```cpp
} else {
    const int num_tiles = num_tiles_dyn;
    #pragma unroll 2
    for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
}
```

while the KI_HINT > 0 branch (used by DSV3 K=7168 → KI=112, Qwen3
K=4096 → KI=64) does **full** `#pragma unroll` for RCR/RRR:

```cpp
if constexpr (KI_HINT > 0) {
    constexpr int num_tiles = KI_HINT;
    if constexpr (L == Layout::CRR) { #pragma unroll 2 ... }
    else { #pragma unroll  // FULL UNROLL
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    }
}
```

R10 attributed 13 pp of the 17 pp gpt_oss vs DSV3 MFMA-util gap to
"short-K main-loop saturation (per-tile fixed cost amortizes worse)";
the unroll-factor split looked like a more concrete cause: K=2880
gets a 2-iter compiler scheduling window while DSV3 K=7168 gets a
55-iter window — fewer cross-iteration vmem / MFMA / LDS interleave
opportunities, more loop-branch overhead.

Adding `template grouped_kernel<RCR, 44, true>` and
`template grouped_kernel<RRR, 44, true>` (K=2880 → fast_k=2816 → ki=44)
plus a `case 44: launch_one_grouped_fuse_ki<L, 44>(g)` dispatch branch
would route gpt_oss K=2880 to a fully-unrolled main loop while keeping
the FUSED_KTAIL block intact (same `device_gemm_tile_body` template,
same K-tail epilog).

## R14 implementation

`HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

1. Added explicit instantiation block at lines ~4014-4034:
   ```cpp
   template __global__ void grouped_kernel<Layout::RCR, 44, true>(...);
   template __global__ void grouped_kernel<Layout::RRR, 44, true>(...);
   ```
2. Added `launch_one_grouped_fuse_ki<L, KI>` template helper mirroring
   `launch_one_grouped_fuse<L>` but with `KI_HINT > 0` (line ~4046).
3. Modified the `if (fuse_ktail_eligible)` branch in `dispatch_grouped`
   to switch on `g.ki`:
   ```cpp
   switch (g.ki) {
       case 44:  launch_one_grouped_fuse_ki<L, 44>(g); break;
       default:  launch_one_grouped_fuse<L>(g); break;
   }
   ```

## Correctness gate

`/tmp/probe_round14_correctness.py` — fp32 reference vs HK output,
max_abs + SNR over 6 shapes (5 fuse-eligible K=2880 + 1 no-fuse
control K=4096):

| shape | K | K%128 | path | max_abs | SNR (dB) | pass |
|---|---|---|---|---:|---:|---|
| gpt_oss-GateUP-B32-M2048 | 2880 | 64 | KI=44 fuse | 0.0156 | 49.58 | OK |
| gpt_oss-Down-B32-M2048   | 2880 | 64 | KI=44 fuse | 0.0156 | 49.58 | OK |
| gpt_oss-GateUP-B32-M4096 | 2880 | 64 | KI=44 fuse | 0.0156 | 49.58 | OK |
| gpt_oss-Down-B32-M4096   | 2880 | 64 | KI=44 fuse | 0.0156 | 49.58 | OK |
| gpt_oss-GateUP-B4-M2048  | 2880 | 64 | KI=44 fuse | 0.0156 | 49.58 | OK |
| qwen3-GateUP-B16-M4096   | 4096 |  0 | KI=64 std  | 0.0156 | 49.60 | OK |

All 6 PASS at clean BF16-MFMA accumulation SNR (≥ 49 dB). Numerical
schedule unchanged (KI_HINT only affects code-gen unroll factor, not
the MMAs / loads / epilogs). Control shape (qwen3 K=4096) confirms
the dispatch change is K-gated correctly: it lands on `KI_HINT=64`,
not the new `KI_HINT=44` path.

## Resource report (the critical signal)

`-Rpass-analysis=kernel-resource-usage` for the 4 relevant kernels
(grouped_kernel<L, KI, FUSED_KTAIL>, persistent grouped, line 3667):

|                                  | KI=0 fuse=T | KI=44 fuse=T | Δ |
|----------------------------------|-------------:|-------------:|---|
| **RCR (Layout 0)**               |              |              |   |
| VGPRs                            |     248      |     **256**  | +8 (CAP) |
| ScratchSize (bytes/lane)         |       0      |     **116**  | +116 |
| VGPRs Spill                      |       0      |      **28**  | +28 |
| Occupancy (waves/SIMD)           |       2      |       2      | 0 |
| **RRR (Layout 1)**               |              |              |   |
| VGPRs                            |     249      |     **256**  | +7 (CAP) |
| ScratchSize (bytes/lane)         |       0      |     **124**  | +124 |
| VGPRs Spill                      |       0      |      **30**  | +30 |
| Occupancy (waves/SIMD)           |       2      |       2      | 0 |

Full unroll on the 21-iter main loop (RCR / RRR) capped VGPRs at
256 and forced 28-30 register spills to scratch memory (per-lane).
The compiler couldn't reuse the `A_tile / B_tile_0 / B_tile_1`
register names across the 21 unrolled `main_loop_iter` copies — each
copy wants fresh registers for its prefetch loads to overlap with
the prior copy's MMAs, and at full unroll that working set exceeds
the VGPR budget.

Comparison points:
* KI=56 + fuse=F (existing K=3584 case if any): VGPRs=256, Spill=14,
  Scratch=60 — also spills but less.
* KI=112 + fuse=F (DSV3 K=7168): VGPRs=256, Spill=0, Scratch=0 — no
  spill because longer K means deeper main loop and the compiler
  allocates a different register tiling pattern that fits.

The KI=44 + fuse=T case has the WORST spill profile of any kernel in
the binary (more than the existing KI=832 / KI=462 cases that already
spill). Adding the FUSED_KTAIL block on top of the unrolled main
loop is what tipped the working set past the budget.

## Performance gate (where the change failed)

|                  | metric score | gpt_oss geomean | DSV3  | Qwen3 |
|------------------|-------------:|----------------:|------:|------:|
| Baseline (HEAD = 76f400b) | 873 | 1.074  | 1.119 | 1.113 |
| R14 (KI=44 fuse spec)     | **843** | **1.012** | 1.117 | 1.112 |
| Revert verify             | 885 | 1.097  | 1.124 | 1.115 |

Per-shape gpt_oss before → after with R14:

| shape | baseline | R14 | Δ pp |
|---|---:|---:|---:|
| GateUP-B4-M2048    | 1.067 | 1.020 | -4.7 |
| Down-B4-M2048      | 1.090 | 1.053 | -3.7 |
| GateUP-B4-M4096    | 1.089 | 1.052 | -3.7 |
| Down-B4-M4096      | 1.086 | 1.030 | -5.6 |
| GateUP-B32-M2048   | 1.054 | 0.984 | -7.0 |
| Down-B32-M2048     | 1.044 | 0.974 | -7.0 |
| GateUP-B32-M4096   | 1.083 | 1.009 | -7.4 |
| Down-B32-M4096     | 1.084 | 0.981 | -10.3 |

All 8 gpt_oss shapes regressed; B=32 shapes regressed more than B=4
(7-10 pp vs 4-6 pp), consistent with the spill traffic scaling with
the per-CU tile workload (B=32 has more tiles, so more cross-tile
spill traffic).

DSV3 / Qwen3 / no-fuse shapes are unaffected within noise — the
dispatch gate (`fuse_ktail_eligible && g.ki == 44`) correctly scopes
the change to gpt_oss K=2880 only.

## Mechanistic explanation

R10 was right that there's a "short-K main-loop saturation" gap, but
it's NOT cleanly fixable by full unroll because:

1. The main_loop_iter body uses ~85 lines of source covering 4 G::loads
   (A and B prefetch for tile+2), 4 load_*_subtile (LDS reads into the
   active register tile), 4 MMAs, plus barriers/waitcnts. The active
   register working set inside the body is ~50-60 VGPRs.
2. Full unroll requires the compiler to either:
   (a) Reuse the same A_tile/B_tile_*/C_accum names across iterations
       — limits ILP, defeats the unroll benefit
   (b) Allocate fresh register names per iteration — explodes the live
       range, exceeds 256 VGPR budget
3. At KI=44 the compiler chose (b) and overflowed → spill.

The KI=112 (DSV3) case **also chose (b)** but didn't spill because
the cross-iteration dependency chain is **longer** (more MMAs per
prefetch, so the compiler can stagger live ranges across more iters).
KI=44 has half the main-loop iters → twice as compressed live ranges
→ spill.

`#pragma unroll 4` would be a middle ground but isn't expressible in
the current codebase without a new template branch in
`device_gemm_tile_body`.

## Why R10's prediction was off

R10 said:
> Lever A2 (short-K main loop): Reduce per-tile prologue/epilog/tile-
> transition overhead → ~+13 % speedup ceiling, +70 score.

R10 framed A2 as a **fixed-overhead amortization** problem, but the
real cause of the gap is **compiler scheduling depth** (unroll-factor
gives more interleave opportunities). A2's ceiling of +13% assumes
fixing the gap; this round shows the obvious fix (full unroll) is
**negative-EV** because of the spill cost.

The "real" A2 lever would need to:
* Find a working-set-bounded unroll factor (4? 8?) that gives some
  scheduling depth without spilling, OR
* Restructure `main_loop_iter` to use fewer concurrent register tiles
  (e.g., interleave 2 inner iters at the source level), OR
* Manual ASM scheduling annotations in the unroll-2 path that mimic
  what full unroll would have done (very high risk).

## Lessons for R15+

1. **Don't full-unroll without checking spill first.** The
   `-Rpass-analysis=kernel-resource-usage` flag is in the Makefile
   already — R14's mistake was building, then probing correctness,
   then probing performance. Should have read the resource report
   FIRST and reverted before measuring (would have saved 1 round).

2. **R10's "main-loop saturation" gap is real but not unroll-fixable
   on this kernel structure.** Future A2-class levers must respect
   the 256 VGPR / no-spill budget. Candidate angles:
   - Per-warp tiling reduction (smaller HALF_REG_BLOCK_M / N) — would
     reduce per-warp working set but also reduce per-tile work
     (= more tiles, more fixed overhead — tension with the original
     amortization argument).
   - Manual instruction-level scheduling annotations (sched_barrier,
     setprio) inside the unroll-2 main loop body — same set of MMAs
     and loads, just different relative ordering. R11's A1 attempt
     was in this family; got -0.3 score (sub-noise). Diminishing
     returns.

3. **The fuse path's KI=0 dynamic kernel may already be near-optimal
   on this codebase's instruction-scheduling patterns.** R10 / R11 /
   R14 (this) have all attempted to find +5 score on the K=2880 path
   and falsified. Combined falsification surface size: ~3 levers.
   Strong evidence the kernel-side ceiling on gpt_oss K=2880 is
   <+5 score from current state without a different kernel
   architecture.

4. **Score plateau acceptance is overdue.** Best=891, current
   plateau=873-887 across rounds 7-14 = ±10 noise band. The
   per-round +5 commit threshold cannot reliably be cleared on the
   gpt_oss K=2880 surface alone.

## Compliance check

* No metric file edits.
* No `can_handle` tightening.
* No host syncs.
* HipKittens C++ reverted via `git checkout` after performance
  falsification; rebuild restored baseline (885 ≈ pre-R14 873-887
  noise band).
* No new dispatcher rules persisted.
* Doc-only commit in Primus-Turbo this round.
* Numerical correctness gate PASSed (49.58 dB on all 6 probes) —
  even if the change had landed, it wouldn't have been a correctness
  regression.

## What R14 DID

1. Re-ran R14 baseline metric (873; gpt_oss geomean 1.074 — within
   noise of R8-R13 plateau).
2. Read `kernel_bf16_dynamic.cpp:3984-4010, 4023-4032, 4203-4222`
   — found K=2880 routes to KI_HINT=0 + `#pragma unroll 2`.
3. Implemented `KI_HINT=44 + FUSED_KTAIL=true` specialization +
   `launch_one_grouped_fuse_ki<L, KI>` helper + `case 44:` dispatch
   branch.
4. Built (~55s wall, full build).
5. Read resource usage: VGPRs cap, +28-30 spill, +116-124 scratch
   bytes/lane on the new kernels.
6. Ran correctness probe — all 6 shapes PASS at SNR ≥ 49 dB.
7. Ran metric — score 843, all 8 gpt_oss shapes regressed -3.7 to
   -10.3 pp.
8. Reverted HK kernel via `git checkout`, rebuilt.
9. Verified revert metric = 885 (within noise of pre-R14 baseline).
10. Wrote this FALSIFIED note.

## Files touched (Primus-Turbo only — HK reverted)

* `analysis/_notes/round-14-bf16-grouped-KI44-fuse-specialize-FALSIFIED-spill.md` — this note.

No code changes shipped. HipKittens repo: `git checkout` reverted the
kernel. Probe script archived at `/tmp/probe_round14_correctness.py`.

## R15 plan (pivot)

The kernel-level surface for gpt_oss K=2880 has now seen 3 falsified
attempts (R11 A1 early-issue prefetch, R14 A2c full-unroll, plus the
implicit closure of A2a/A2b after R10's analysis). Recommended pivot
options:

1. **R15 — partial-unroll search**: Try `#pragma unroll 4` and
   `#pragma unroll 8` (would need a new template branch). Risk: same
   spill mechanism; reward: +5-15 score if it finds the sweet spot
   between scheduling depth and register pressure. Prerequisite:
   resource-report check FIRST, abort if spill > 8 VGPRs.

2. **R15 — accept plateau, write summary doc**: Document the noise
   band (±10), best=891, and the closed kernel-level surface for
   gpt_oss K=2880. Recommend the auto_optimize loop pivot to
   FP8 wall (out of this run's scope, but the user can redirect).

3. **R15 — DSV3 / Qwen3 push (Phase B per task body)**: After
   accepting gpt_oss plateau, the +14% to 1.25 on DSV3+Qwen3 would
   add ~+40 score. R8 falsified DSV3 var-K cfg, but the K%128==0
   forward path may have unexplored levers (Lever B1 — MFMA pipeline
   scheduling on K=4096/7168 main loop). Different territory than
   K-tail; might find +5-10 score there. Prerequisite:
   `rocprofv3 valuMfmaUtil` profile on DSV3-GateUP-B16-M4096 to
   confirm Lever B1 has headroom (R9 measured 79.7 % util on this
   shape — only 5pp from the 85 %+ cluster ceiling, marginal).

R15 should pick option 1 if spill-aware (low cost, real chance of
+5), or option 3 if the DSV3 profile shows >5pp util headroom.
Option 2 is the "safe accept" that records the closure but doesn't
move the score.
