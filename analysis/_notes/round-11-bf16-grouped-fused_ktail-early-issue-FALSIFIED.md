# Round 11 — BF16 grouped FUSED_KTAIL early-issue prefetch (FALSIFIED)

**Status:** FALSIFIED — Lever A1 (K-tail HBM-load early-issue) is correctness-clean
and resource-neutral, but kernel speedup is sub-noise (ΔScore < 0 vs baseline
within the ±10 noise band).

## What R10 set up

R10 isolated the K-tail cost to ~4 pp MFMA / ~3 % kernel time (much smaller than
R9's 17 pp estimate). R10 outlined Lever A1 (early-issue K-tail HBM loads with
a separate A register set) as a +7 % kernel-speedup ceiling with risk in VGPR
pressure. R11 was scoped to **implement and measure** Lever A1.

## R11 implementation

`kernel_bf16_dynamic.cpp:863-916`, RCR FUSED_KTAIL block. Change:

```diff
-auto load_a_kt = [&](int m_slab) ... {
-    *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][w].data[0]) = v;
+auto load_a_kt = [&](A_reg_t& A_dst, int m_slab) ... {
+    *reinterpret_cast<__uint128_t*>(&A_dst.tiles[h][w].data[0]) = v;
 };

-load_a_kt(0);
-load_b_kt(B_tile_0, 0);
-load_b_kt(B_tile_1, 1);
-asm("s_waitcnt vmcnt(0)");
-DO_MMA(C[0][0], A_tile, B_tile_0, ...);
-DO_MMA(C[0][1], A_tile, B_tile_1, ...);
-load_a_kt(1);
-asm("s_waitcnt vmcnt(0)");
-DO_MMA(C[1][0], A_tile, B_tile_0, ...);
-DO_MMA(C[1][1], A_tile, B_tile_1, ...);
+A_reg_t A_tile_kt_1;
+load_a_kt(A_tile,      0);
+load_b_kt(B_tile_0,    0);
+load_b_kt(B_tile_1,    1);
+load_a_kt(A_tile_kt_1, 1);   // EARLY ISSUE — overlap slab-1 A HBM with slab-0 MMAs
+asm("s_waitcnt vmcnt(0)");    // single drain
+DO_MMA(C[0][0], A_tile,      B_tile_0, ...);
+DO_MMA(C[0][1], A_tile,      B_tile_1, ...);
+DO_MMA(C[1][0], A_tile_kt_1, B_tile_0, ...);
+DO_MMA(C[1][1], A_tile_kt_1, B_tile_1, ...);
```

Goal: replace two serial `vmcnt(0)` drains with one drain after issuing all
16 K-tail vmem ops upfront. The slab-1 A HBM load now overlaps with slab-0
MMAs instead of starting after them.

## Resource analysis (compile gate)

Built with `-Rpass-analysis=kernel-resource-usage`. Compared baseline vs R11
on the production grouped kernel:

|                                  | Baseline | R11   | Δ |
|----------------------------------|---------:|------:|---|
| **`grouped_kernel<RCR, KI=0, FUSED_KTAIL=true>`** | | | |
| VGPRs                            |     248  | 248   | 0 |
| SGPRs                            |      96  | 96    | 0 |
| ScratchSize (B/lane)             |       0  | 0     | 0 |
| VGPRs Spill                      |       0  | 0     | 0 |
| LDS Size (B/block)               |     544  | 544   | 0 |
| Occupancy (waves/SIMD)           |       2  | 2     | 0 |

The compiler folded `A_tile_kt_1` into the dead VGPRs that `A_tile / B_tile_0
/ B_tile_1` would have held during epilog 2. Net delta = 0 across all
instantiated grouped + dense kernels.

(26 kernels in baseline already spill — those are unrelated `gemm_kernel<KI≥56>`
and `grouped_kernel<KI≥56, FUSED_KTAIL=false>` instantiations that pre-existed
this work; R11 leaves them unchanged.)

## Correctness gate

`/tmp/probe_round11_correctness.py` — fp32 reference vs HK output, max-abs +
SNR over 6 shapes covering both K-tail (K=2880) and no-K-tail (K=2816, 4096,
7168) paths, gpt_oss / DSV3 / Qwen3 families, B ∈ {4, 16, 32}:

| shape | max_abs | SNR (dB) | pass |
|-------|--------:|---------:|------|
| gpt_oss-GateUP-B32-M2048  (K%128=64, FUSED_KTAIL) | 0.0156 | 47.82 | OK |
| gpt_oss-Down-B32-M2048    (K%128=64, FUSED_KTAIL) | 0.0156 | 47.82 | OK |
| gpt_oss-GateUP-B4-M2048   (K%128=64, FUSED_KTAIL) | 0.0156 | 47.82 | OK |
| DSV3-GateUP-B16-M4096     (K%128=0,  no K-tail)   | 0.0312 | 47.85 | OK |
| Qwen3-GateUP-B16-M4096    (K%128=0,  no K-tail)   | 0.0156 | 47.84 | OK |
| synth K=2816              (no K-tail, A/B sanity) | 0.0156 | 47.82 | OK |

All shapes PASS at clean BF16-MFMA accumulation SNR (≥ 47 dB).

## Performance gate (where the change failed)

4-run metric samples on the same GPU (HIP_VISIBLE_DEVICES=3 pinned by
auto_optimize.py):

|             | run 1 | run 2 | run 3 | run 4 | mean | spread |
|-------------|------:|------:|------:|------:|-----:|-------:|
| **Baseline (HEAD = 9dc4dc0)** | 879 | 881 | 881 | 881 | 880.5 | 2 |
| **R11 (early-issue prefetch)** | 885 | 874 | 880 | 882 | 880.2 | 11 |

R11 mean ≈ baseline mean. Δ = -0.3 score, well within run-to-run spread.
R11 doesn't measurably move the score; the +5 commit threshold is not met.

## Mechanistic explanation

R10 measured K-tail cost = 56 µs / kernel ≈ 2.7 % of total time.

Best-case A1 saving = the part of K-tail HBM stall that previously sat
serially after slab-0 MMAs. Hard upper bound: half of K-tail time = ~28 µs ≈
1.4 % kernel speedup (since the slab-0 drain remains the same).

GCN vmcnt is a single counter across all SRDs; A and B use different SRDs
(`a_srsrc_base`, `b_srsrc_base`), so completion order is not strict in-order.
This forces the single drain to `vmcnt(0)` (wait for all 16), which is
roughly the same wall-time as the current `vmcnt(0)` after 12 ops + a second
`vmcnt(0)` after 4 ops — the limiting latency is the slowest-completing op,
not the count.

In other words: in the current GCN scheduling model the change reorders WHEN
the slab-1 A vmem op is **issued**, not when it **completes**. Since the
issue rate is bound by SRD throughput (~1 op/cycle) and the wait is bound by
HBM round-trip latency (~hundreds of cycles), early-issue saves only ~16
cycles of issue overlap × 22 K-tails per CU × 256 CUs / 256 CUs = ~352 ns
total ≈ 0.02 % kernel speedup. Below noise.

## Lessons for R12+

1. **A1 is genuinely small leverage.** R10's +7 % ceiling was an upper-bound;
   the realised saving in the GCN model is closer to 0.02-1 %. Do not pursue
   A1 further on this kernel.
2. **VGPR pressure is NOT a bottleneck for additive work in the FUSED_KTAIL
   block.** The compiler reuses dead VGPRs from `A_tile / B_tile_0 / B_tile_1`
   when the new register tile has the right liveness. Future K-tail-specific
   optimisations can add more registers without occupancy hit, IF they touch
   the K-tail path only.
3. **The same may NOT be true for changes that touch the main loop or
   epilog 2.** Adding registers there would compete with `A_tile / B_tile_0 /
   B_tile_1 / C_accum` which are all live concurrently.

## R12 plan (pivot)

Track A2 (short-K main-loop / per-tile fixed-overhead reduction) is now the
only remaining lever for gpt_oss > +5 score. Per R10 plan, R12 should:

1. **Profile per-tile timing** via `rocprofv3` markers around the persistent-
   kernel `for (int tile = ...)` loop body, separating:
   * Prologue (A/B initial K-stripe load + LDS write)
   * Main loop (K_TWO_TILE × 22 iters for K=2880)
   * Epilog 1 + epilog 2 drain
   * K-tail (already known: ~3 % of per-tile)
   * C-store
   * Tile transition (back to top of persistent loop, group decode, etc.)
2. The key question: how big is the **tile-transition + prologue** cost
   relative to main loop? If > 10 % of per-tile time, A2a (pre-issue next
   tile's prologue HBM load during current tile's C-store) is the candidate.
3. If A2 leverage is also small, the BF16 grouped suite is at its kernel-
   level performance ceiling and the only remaining levers are dispatch
   re-tunes (R7, R26, R21 already saturated) — pivot to FP8 if the FP8 wall
   needs work, or accept the current ceiling.

## Compliance check

* No metric file edits.
* No `can_handle` tightening.
* No host-syncs.
* No persistent HipKittens C++ changes (kernel was modified, tested,
  reverted via `git checkout`).
* No new dispatcher rules.
* Doc-only commit in Primus-Turbo this round.

## What R11 DID

1. Re-ran R11 baseline metric (882, sample 1; mean 880.5 across 4 runs).
2. Read the `FUSED_KTAIL` block of `kernel_bf16_dynamic.cpp:779-1117`.
3. Confirmed grouped-kernel VGPRs = 248 / 256 (`-Rpass-analysis`).
4. Implemented the early-issue prefetch with `A_tile_kt_1`.
5. Built (~50s wall, full build).
6. Confirmed VGPR / SGPR / spill / LDS all unchanged vs baseline.
7. Ran correctness probe — all 6 shapes PASS at SNR ≥ 47 dB.
8. Ran 4-run metric — R11 mean 880.2 vs baseline 880.5. Sub-noise.
9. Reverted HK kernel via `git checkout`.
10. Wrote this FALSIFIED note.

## Files touched (Primus-Turbo only)

* `analysis/_notes/round-11-bf16-grouped-fused_ktail-early-issue-FALSIFIED.md` — this note.

No code changes shipped. HipKittens repo: `git checkout` reverted the
kernel; archive of the attempted change kept at
`/tmp/kernel_round11_v1.cpp.keep` for R12+ reference if A2 needs to revisit.
