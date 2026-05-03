# Round-38 — BF16 grouped GEMM: port FP8 R37 K-tail `a_kt1 + vmcnt(8) split` to BF16 RCR fuse — **FALSIFIED** (mean delta -1.4, sub-noise)

## Status
**FALSIFIED** (paired 4-run baseline mean 883 vs 5-run R38 change mean 881.6,
delta -1.4 well within noise). HipKittens kernel reverted. Primus-Turbo
gets this docs-only round note.

## Selected target
Per R38 baseline (HEAD `9038b10`, score=879):
- Lowest-progress shapes, weight 3:
  1. **gpt_oss_20B-GateUP-B32-M2048** ratio 1.049 (4 metric calls in)
  2. gpt_oss_20B-Down-B32-M2048      ratio 1.059
  3. gpt_oss_20B-GateUP-B32-M4096    ratio 1.087
  4. gpt_oss_20B-Down-B32-M4096      ratio 1.081

R9 (rocprofv3 PMC) had already nailed these as **MFMA-Util 63.4 %** vs
the K%128==0 path's 75-80 % — the FUSED_KTAIL epilog
(kernel_bf16_dynamic.cpp:799-916, RCR branch lines 800-916) is the
single bottleneck on gpt_oss K=2880. R37-fp8
(`round-37-dm-fp8-ktail-issue-reorder-b0b1aa_kt1-wins-plus16pts-via-llvm-liveness-reshape.md`)
demonstrated +16 metric points by porting an
`a_kt1` second-register-tile + `[b0, b1, a, a_kt1]` issue order +
`vmcnt(8)` split to the FP8 grouped K-tail. **BF16 has never had
this evolution** — the BF16 K-tail block is still at the round-5 path-B
state (`8deea208 perf(grouped-bf16): fuse K-tail into RCR main kernel
epilog (path B)`) with single A_tile, single `vmcnt(0)`, and issue
order `[a, b0, b1] → mma → load_a → mma`.

R38 lever: direct port of the FP8 R3+R7+R12+R37 cumulative pattern.

## Change applied (HipKittens, reverted before commit)
`analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp` — RCR FUSED_KTAIL
block at lines 905-916:

```diff
-            auto load_a_kt = [&](int m_slab) ... {
-                ...
-                *reinterpret_cast<__uint128_t*>(
-                    &A_tile.tiles[h][w].data[0]) = v;
-            };
+            A_reg_t A_tile_1;          // R38: 2nd A register for slab 1
+            auto load_a_kt = [&](A_reg_t& A_tile_dst, int m_slab) ... {
+                ...
+                *reinterpret_cast<__uint128_t*>(
+                    &A_tile_dst.tiles[h][w].data[0]) = v;  // dst by ref
+            };
             auto load_b_kt = ...;  // unchanged
-            // M slab 0
-            load_a_kt(0);
-            load_b_kt(B_tile_0, 0);
-            load_b_kt(B_tile_1, 1);
-            asm volatile("s_waitcnt vmcnt(0)");
-            DO_MMA(C_accum[0][0], A_tile, B_tile_0, ...);
-            DO_MMA(C_accum[0][1], A_tile, B_tile_1, ...);
-            // M slab 1 (B tiles unchanged — share K-tail across slabs)
-            load_a_kt(1);
-            asm volatile("s_waitcnt vmcnt(0)");
-            DO_MMA(C_accum[1][0], A_tile, B_tile_0, ...);
-            DO_MMA(C_accum[1][1], A_tile, B_tile_1, ...);
+            // R38 issue order: smaller B first.
+            // 24 buffer_load_b128 total = 4 + 4 + 8 + 8.
+            load_b_kt(B_tile_0, 0);             // 4 b128 (FIRST)
+            load_b_kt(B_tile_1, 1);             // 4 b128
+            load_a_kt(A_tile,   0);             // 8 b128 → A (slab 0)
+            load_a_kt(A_tile_1, 1);             // 8 b128 → A_tile_1 (slab 1, LAST)
+            asm volatile("s_waitcnt vmcnt(8)");  // first 16 retired = b0+b1+a
+            DO_MMA(C_accum[0][0], A_tile,   B_tile_0, ...);
+            DO_MMA(C_accum[0][1], A_tile,   B_tile_1, ...);
+            asm volatile("s_waitcnt vmcnt(0)");  // drain a_kt1
+            DO_MMA(C_accum[1][0], A_tile_1, B_tile_0, ...);
+            DO_MMA(C_accum[1][1], A_tile_1, B_tile_1, ...);
```

24-line diff (45 inserts, 11 deletes including comment expansion).

## Resource report — A_tile_1 added at zero cost
`grouped_kernel<RCR, KI_HINT=0, FUSED_KTAIL=true>` resource usage,
`-Rpass-analysis=kernel-resource-usage`:

```
                       baseline    R38 (after)
TotalSGPRs                  96         96
VGPRs                      248        248
AGPRs                        0          0
ScratchSize [B/lane]         0          0
Occupancy [waves/SIMD]       2          2
SGPR Spill                   0          0
VGPR Spill                   0          0
LDS Size [B/block]         544        544
```

**Identical**. Adding `A_tile_1` (32 VGPR/lane in source) did NOT
inflate the compiled kernel — LLVM was already allocating registers
efficiently (pre-existing 248/256 ceiling means ≤8 VGPR of slack
that the new tile reuses from dead prologue scratch). This rules
out the VGPR-pressure-regression hypothesis; the falsification
below is purely about the K-tail wall savings being too small to
move the metric.

## Metric (paired)
Baseline (HEAD `9038b10`, kernel binary unchanged):
```
score = 879  (initial)
score = 882
score = 888
score = 883
mean  ≈ 883.0  (range 879-888, σ ≈ 3.7)
```

R38 change (kernel rebuilt with the diff above):
```
score = 878
score = 880
score = 885
score = 878
score = 887
mean  ≈ 881.6  (range 878-887, σ ≈ 4.4)
```

**Mean delta: -1.4 score** (R38 - baseline). Both ranges overlap
heavily (878-888 range covers BOTH means within σ). NOT a +5
candidate; in fact, slightly negative if anything. Per-family
geomean from one R38 run vs the corresponding baseline:

```
                       baseline (879)   R38 (878)   Δ
gpt_oss_20B   geomean   1.0850          1.0846     -0.04 pp
DeepSeek-V3   geomean   1.1222          1.1238     +0.16 pp
Qwen3-235B    geomean   1.1137          1.1118     -0.19 pp
```

All deltas <0.2 pp = pure noise.

## Why the FP8 R37 mechanism does NOT transfer to BF16

FP8 R37 attributed +16 score to two coupled effects:
1. **HBM-overlap pipelining** of the trailing 8 `a_kt1` loads with
   the leading cA/cB MFMAs via the `vmcnt(8)` split.
2. **LLVM liveness reshape** — the `A_row_reg a_kt1;` declaration
   reshuffles VGPR allocation in the main loop, improving spill
   placement on K-iter-heavy paths (FP8 R37 noted DSV3 K=7168 with
   56 K-iters amplified the mechanism most).

BF16 R38 inherited (1) at the same magnitude (~32-64 cyc/output
tile saved on the K-tail block, where K_STEP=64 means 24 b128
total = 384 B/lane), but **gets ~0 from (2)** for four reasons:

1. **`FUSED_KTAIL=true` template scope is gpt_oss-only in BF16.**
   The `if constexpr (FUSED_KTAIL)` gate (kernel_bf16_dynamic.cpp:799)
   means the new `A_reg_t A_tile_1;` declaration only emits in the
   `<RCR, 0, true>` template instance. DSV3 (K∈{2048, 7168},
   K_REM=0) and Qwen3 (K∈{1536, 3072, 4096}, K_REM=0) all route
   through `<RCR, 0, false>` — a separate compiled instance,
   COMPLETELY UNTOUCHED by R38. FP8 won big on DSV3 because FP8 R34
   had separately extended `FUSED_KTAIL=true` to K_REM=0 shapes
   (sole purpose: capture liveness reshape effects on DSV3). BF16
   has no R34 equivalent — no path for DSV3 / Qwen3 to benefit.
2. **BF16 K-tail is NOT VGPR-pressure-bound.** Resource report
   bit-identical pre/post. FP8 R37's biggest gains came from
   redistributing spill placement; BF16 has 0 spill before AND
   after. The mechanism for liveness gain has no spill state to
   improve.
3. **gpt_oss main loop is short (44 K-iters / 22 main_loop_iter
   pairs vs FP8 DSV3's 56).** Even if liveness were to help,
   amplification would be half the FP8 case.
4. **gpt_oss share of HBM-overlap savings is small.**
   GateUP-B32-M2048 has M_total/HALF_BLOCK_SIZE × ceil_div(N=5760,
   BLOCK_SIZE) = 256 × 23 ≈ 5888 output tiles distributed across
   ~304 CUs ⇒ ~20 tiles/CU. Each saves ~32-64 cyc on the K-tail
   epilog ⇒ 640-1280 cyc/CU = 0.3-0.6 µs/CU at ~2 GHz. Kernel wall
   ≈ 1-2 ms, so the saving is ~0.05-0.1 % wall — far below the
   ~0.5 % metric-noise floor.

Result: the change is correct (24/24 PASS) and clean (zero VGPR
pressure delta), but the metric impact is genuinely sub-noise on
the BF16 weighted-wall. Do **not** retry this lever standalone.

## What R39+ should attack instead

The R9 rocprof finding (gpt_oss K-tail = 63 % MFMA util) is still
the highest-leverage bottleneck on paper, but BF16-specific
structural barriers make the FP8-derived recipes ineffective.
Three remaining attack surfaces to consider:

### Option A — gpt_oss main-loop MFMA scheduling
gpt_oss K=2880 has 22 main_loop_iter calls (each = 2 K-tiles × 8
DO_MMAs = 16 MFMAs). The main loop body
(kernel_bf16_dynamic.cpp:600-686) uses `__builtin_amdgcn_s_setprio(1)`
around each DO_MMA pair — that's already in. But the K-tail block
(905-916) does NOT use `s_setprio`. Quick mirror experiment:

```cpp
// Around K-tail DO_MMAs — mirror main loop pattern
__builtin_amdgcn_s_setprio(1);
DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
__builtin_amdgcn_s_setprio(0);
```

Effort: 10 lines. Risk: very low. Expected: marginal (the K-tail
MFMA throughput is ~32-64 cyc total — priority bumps don't help
much when the unit is non-contended).

### Option B — port FP8 R34 (K_REM=0 FUSED_KTAIL=true) to BF16
Extend the `fuse_ktail_eligible` predicate
(kernel_bf16_dynamic.cpp:4194-4201) to allow K_REM=0 shapes
through the FUSED_KTAIL=true template. Currently:

```cpp
const bool fuse_ktail_eligible =
    ((L == Layout::RCR) ...) &&
    (g.bpc > 0) && (g.ki >= 2) &&
    (K_rem_for_fuse == K_STEP) && lds_k_tail_safe_for_fuse;
```

would become `K_rem_for_fuse ∈ {K_STEP, 0}`. The FUSED_KTAIL=true
body's `if (g.fast_k < g.k)` runtime gate already short-circuits
when K_REM=0 (line 800 outer gate, line 858 in path-A RRR scope).
But the OUTER `if constexpr (FUSED_KTAIL)` would now apply to DSV3
/ Qwen3 too — A_tile_1 declaration enters scope, LLVM liveness
graph reshapes for the compile-time-included branch, even though
the runtime branch never fires.

Risk: the dispatcher's grouped/non-grouped branch tree at lines
4203-4232 has many switch cases keyed on `g.ki`; each ki value
currently routes to `launch_one_grouped<L, KI>` (FUSED_KTAIL=false).
Extending fuse to K_REM=0 means a SECOND template tree for
FUSED_KTAIL=true K_REM=0 shapes — code-bloat risk. Need to verify
the build size doesn't explode and the linker doesn't reject the
new instantiations.

Estimated impact (mirror FP8 R34): +5 to +20 score from liveness
reshape on DSV3/Qwen3 main loops (those run K=7168, 4096, 3072 —
56, 32, 24 K-iters; potential per-iter spill savings amplify
through the hot path).

This is the **next-highest-priority lever** if a structural change
is acceptable.

### Option C — rocprofv3 the actual K-tail epilog wall on gpt_oss
R9 captured `MfmaUtil 63.4 %` whole-kernel — does NOT separate the
main loop from the K-tail. Need a marker-bracketed PMC capture
that times only the K-tail block to confirm what fraction of total
wall it consumes. If K-tail is e.g. 5 % of wall, even a 50 %
K-tail speedup nets only 2.5 % overall = ~+1 score (sub-noise).
If K-tail is 20 % of wall, 50 % speedup nets 10 % = ~+30 score.

Without this measurement we keep chasing micro-optimizations on
a block that may not justify the effort.

R39 should pick ONE of A / B / C, ideally C first (free
information; tells us whether to invest in B or pivot away).

## Files touched
- HipKittens: `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  (changed and reverted; no commit).
- Primus-Turbo: `analysis/_notes/round-38-bf16-grouped-fp8-r37-port-FALSIFIED-mech-non-transfer.md`
  (this file, docs-only commit).

## Commit
- Primus-Turbo: docs-only round note (this commit).
- HipKittens: unchanged (revert + rebuild verified to match the
  pre-R38 baseline binary).
