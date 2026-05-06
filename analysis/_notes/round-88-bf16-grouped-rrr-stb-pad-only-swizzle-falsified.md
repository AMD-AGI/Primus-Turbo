# Round 88 — bf16 grouped GEMM weighted wall (auto_optimize round 11/100)

> **Context:** auto_optimize round 11/100. R87 (round 10) PMC pivot
> identified Lever B5 (RRR-only `st_32x16_v2_s` ST_B, pad-only peer) as
> the lowest-risk single-round attack on the DSV3/Qwen3 native-RRR dA
> path's inferred 1 BC/Inst. Build-gate set: `grouped_kernel<RRR,0,FUSE=true>`
> ≤ 256 VGPR ∧ ≤ 4 scratch. R88 implements + paired-tests Lever B5.
> Outcome: build-gate PASSES but metric is FLAT — falsified.

**Status:** R88 = **FALSIFIED**. HK change reverted. Primus working tree
holds only this docs note (matches R87's pattern of clean-revert + plan
update). Score stays in the 883 ±2 noise band.

| run                                            | n | mean | std | gpt_oss geomean | DSV3 geomean | Qwen3 geomean |
|------------------------------------------------|--:|-----:|----:|----------------:|-------------:|--------------:|
| R87 baseline (commit 4bbc00e, post-revert)     | 5 | 883.4 | 1.0 | 1.094           | 1.122        | 1.111         |
| R88 v2-swizzle (RRR ST_B = st_32x16_v2_s)      | 5 | 882.8 | 1.0 | 1.094           | 1.120        | 1.111         |

Δ-mean = −0.6 score, well within the per-run ±1 σ noise floor. No
detectable family-level change. **Falsified at the +5-score acceptance
gate.**

## Implementation (reverted, recorded for R12+)

`include/types/shared/st_shape.cuh` — added `st_32x16_v2` peer: same
within-subtile swizzle as `st_32x16` (rows 0-15 identity, rows 16-31
XOR col bit 3); only difference is `subtile_padding = 16` (one BF16
row, 32 banks × 4 B). The padding shifts subtile-i's start bank from
bank 0 to bank `4*(i mod 8)`, breaking the 128-B LDS bank-row alias
on stride-128 ds_read patterns. Identity within-subtile swizzle
preserves `prefill_swizzled_offsets`'s HBM voffset reconstruction
(global_to_shared.cuh:149-151) — same correctness pattern as
`st_64x32_padded_b128` (st_shape.cuh:353).

`include/types/types.cuh` — added `using st_32x16_v2_s = ducks::st_shape::st_32x16_v2;` alias.

`analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp` (line 3734) —
extended `grouped_kernel<L, KI, FUSE>`'s ST_B 2-way conditional to a
3-way: `RCR → st_16x32_s | RRR → st_32x16_v2_s | CRR → st_32x16_s`.
ST_A unchanged. `gemm_kernel` unchanged. `grouped_var_k_kernel`
unchanged (line 4768-4769 keeps `st_32x16_s` for ST_A and ST_B).

## Build-gate verdict (passed, but metric still flat)

Resource report diff vs R87 baseline (`grouped_kernel<RRR, KI, FUSE>`,
all instantiations actually emitted):

| KI    | FUSE | Baseline VGPR/Scratch | R88 VGPR/Scratch | Δ Scratch | Occ |
|------:|:----:|----------------------:|-----------------:|----------:|----:|
|     0 | 0    | 246 / 0               | 250 / 0          | 0         | 2   |
|     0 | 1    | 250 / 0               | 254 / 0          | 0         | 2   |
|    56 | 0    | 256 / 56              | 256 / 72         | +16       | 2   |
|    64 | 0    | 254 / 0               | 256 / 8          | +8        | 2   |
|    88 | 0    | 254 / 0               | 256 / 8          | +8        | 2   |
|   112 | 0    | 256 / 56              | 256 / 72         | +16       | 2   |
|   128 | 0    | 256 / 56              | 256 / 72         | +16       | 2   |
|   172 | 0    | 256 / 56              | 256 / 72         | +16       | 2   |
|   224 | 0    | 256 / 20              | 256 / 16         | -4        | 2   |
|   256 | 0    | 254 / 0               | 256 / 8          | +8        | 2   |
|   296 | 0    | 256 / 56              | 256 / 72         | +16       | 2   |
|   448 | 0    | 254 / 0               | 256 / 8          | +8        | 2   |
|   462 | 0    | 256 / 56              | 256 / 72         | +16       | 2   |
|   832 | 0    | 256 / 20              | 256 / 44         | +24       | 2   |

Layout 0 (RCR) and Layout 2 (CRR) `grouped_kernel` instantiations
**byte-identical** to baseline. `grouped_var_k_kernel<0>` **byte-identical**
(only line-number shift from the `kernel_bf16_dynamic.cpp` edit). All
RRR kernels keep Occ 2 — gate strictly passes for the gated kernel
(KI=0 FUSE=true: 254 VGPR / 0 scratch / Occ 2).

But +8-+24 byte/lane scratch on the hot RRR KIs (KI=64=Qwen3 K=4096 dA;
KI=112=DSV3 K=7168 dA; KI=128=Qwen3 K=8192; KI=172/296/462) introduces
extra spill traffic. Per-lane-byte → per-WG bytes per scratch access:
8-72 B/lane × 256 lanes/WG = 2-18 KB/WG. With ~2 spill loads per main
loop K-tile and ~K/64 K-tiles per group, this puts the swizzle-induced
scratch traffic in the same ballpark as the LDS-BC time we hoped to
eliminate. Net wash.

## Why the swizzle didn't move the needle (post-mortem)

R87 inferred RRR's ST_B usage at "~1 BC/Inst" without direct PMC
measurement (only var_k's 2 BC/Inst was directly counted, with var_k
explicitly attributed to 2× st_32x16_s in CRR). That inference was
weakest at: **the within-subtile swizzle's bank-conflict pattern is
already well-handled by `st_32x16_s`'s rows-16-31 col-bit-3 XOR**;
adding pad-only inter-subtile spacing breaks no additional within-row
alias. Cross-subtile reads are mostly sequential during the main
loop's K-step iteration, where the K-tile counter increments by 1 →
subtile-i and subtile-(i+1) are read on different cycles, NOT the
same 32-lane wave issue. So the v2 padding doesn't reduce within-wave
conflicts, only the (rare) cross-subtile-overlap conflicts that
weren't a measurable bottleneck.

The +scratch on hot KIs that we expected to be VGPR-neutral materialised
because: even though the swizzle function math is bit-identical, the
3-way `std::conditional_t` introduces a distinct ST_B type per layout
→ separate code path → independent register-allocator decisions in
each instantiation → nominal +scratch as the allocator picks slightly
different live-range placements (this is the same kind of compiler
sensitivity that bit R83-R86's swizzle attacks).

## R12 plan: pivot to Lever B6 — var_k VGPR live-range trim

R87 already laid out the multi-round pivot to var_k. R88 confirms B5
is dead → R12+ executes B6. Recap (lifted from round-87 Part D):

1. **R12 (single round)**: profile var_k VGPR live-range using
   `__attribute__((noinline))` markers at suspected high-pressure
   helpers (cooperative-cumsum prologue, group bookkeeping, per-tile
   coord lambdas). Build, read per-helper VGPR usage from
   `-Rpass-analysis`. **No metric change expected this round** —
   produce a per-helper budget table.

2. **R13 (single round)**: recompute-instead-of-store the highest
   live-range per-tile group-row offset that's currently in registers
   but derivable from the iteration counter on-the-fly. Target:
   `grouped_var_k_kernel<0>` 256 VGPR → 244-248 VGPR (8-12 VGPR margin
   restored). Metric should stay flat (var_k path unchanged
   functionally; only register pressure relaxes).

3. **R14-R15 (1-2 rounds)**: with ≥ 8 VGPR headroom in var_k, retry
   the swizzle/permutation landing for var_k's CRR `(ST_A, ST_B)`
   path (the actual 50%-BC bottleneck per R87 PMC, +9-27 weighted
   score upper bound). Apply v2 swizzle to ST_B only first, then
   ST_A only, then both — gate each at +scratch-aware Occ-2 budget.

Lever A1 (forward K-tail HBM stall, RCR FUSE=true at 250/0/Occ 2)
also needs the var_k prologue trim before its dual-A prefetch +8-VGPR
ask becomes affordable. Stay on B6 → A1 sequencing.

## What does NOT work (recap of falsified streak — R83 through R88)

| Round | Lever                                               | Verdict | Cause                                                    |
|-----:|-----------------------------------------------------|---------|----------------------------------------------------------|
|   83 | RCR FUSE=true KI=88 dual-A prefetch                 | -3 score | grouped_kernel<RCR,88,1> +9 spill drops Occ 2→1          |
|   84 | RCR FUSE=true KI=44 LDS pre-shuffle                 | -7 score | grouped_kernel<RCR,44,1> +28 spill drops Occ 2→1          |
|   85 | var-K KI=32/64 inner-loop unroll                    | -4 score | grouped_var_k_kernel KI=32: +14 spill; KI=64: +18 spill   |
|   86 | st_32x16_v2 within-half swizzle (var_k + grouped CRR) | -5 score | var_k +17 spill; grouped CRR +8 spill                     |
|   87 | (diagnostic / pivot only — PMC + plan)              | -      | (no code change)                                          |
|   88 | st_32x16_v2 RRR ST_B (pad-only)                     | flat   | Build-gate passes (254/0/Occ 2) but +8-24 scratch on hot RRR KIs cancels any 1-BC/Inst gain; net wash within ±1 σ noise. |

The pattern: **EVERY single-round VGPR-adjacent change to BF16 grouped
or var_k regresses or stays flat**. The shared root cause is the var_k
kernel sitting at the 256-VGPR + Occ-2 ceiling with no slack — until
that's relaxed (Lever B6 in 1-2 rounds), no swizzle / prefetch /
permutation lever has a viable VGPR budget. R12 starts the prologue
trim explicitly.

## Files touched

* `/workspace/code/Primus-Turbo/analysis/_notes/round-88-bf16-grouped-rrr-stb-pad-only-swizzle-falsified.md`
  (this file)
* `/tmp/build_round11.log` (-Rpass-analysis with v2 swizzle, kept
  offline for R12 helper-VGPR comparison reference)

HipKittens working tree: clean (NFS lock files only).
Primus-Turbo working tree: this docs note + the un-staged R10 artefacts.

## Metric / numbers

* R87 baseline (5 runs): 883, 885, 883, 882, 884 → mean 883.4, std 1.0.
* R88 v2-swizzle (5 runs): 881, 883, 883, 884, 883 → mean 882.8, std 1.0.
* Δ-mean = −0.6 (well below per-run noise floor).
* All 24 shapes PASS correctness in both configurations.
* Per-shape correctness probe (DSV3 GateUP B=16 M=2048 N=4096 K=7168,
  Triton-vs-HK rel-err): y=0.42 %, dA=0.53 %, dB=0.73 % — all below
  the 5 % BF16 tolerance.

## Recommendation for round 12

Execute Lever B6 step 1 — `__attribute__((noinline))` profile of
`grouped_var_k_kernel`'s helpers (cooperative-cumsum prologue,
group-row offset lambdas, per-tile bookkeeping). Goal: produce a
per-helper VGPR budget table to identify the smallest-cost target
for R13's recompute-instead-of-store trim. **Expect no metric change
in R12** (this is a profile-only round). R13's actual trim should
land 8-12 VGPR margin without functional change.
