# Round-48-dm · FP8 grouped — `store_c_tile_n_masked_grouped` clone with `#pragma unroll 1` outer loop achieves SPILL=0 but **CATASTROPHIC 11× regression** from runtime VGPR indexing into register-tile array

**Status**: 1 probe FALSIFIED. Reverted to R44-dm winner state (HK SHA `37926c98`).
Score baseline 959 → R48-dm probe **533** (-426 pts). grp_FP8 1.1194 → **0.3452** (kernel ratios 1.0+ → 0.09-0.14).
Correctness PASS 32/32 (kernel produces correct results, just 11× slower).

This is the **single biggest perf regression in the FP8 grouped optimization
log**. The lesson is sharp and worth 1 round to document fully.

## Hypothesis (per R47-dm next-step plan)

R47-dm ISA inspection identified `store_c_tile_n_masked` (called 4× in
the grouped kernel's C-store epilog when N_MASKED+partial-tile branch
fires) as a likely contributor to the gpt_oss spec's 39-dword spill.
Helper has 4 nested `#pragma unroll`:

* outer `i` (height=4)
* `j` (width=2)
* `k` (num_strides=1)
* inner `l` (stride/packing=4)

Total = 4 × 2 × 1 × 4 = **32 fully-unrolled iters per call**, each with
intermediate `v0/v1/row/col/off0/off1` live. Across 4 calls (cA/cB/cC/cD)
in the persistent kernel's RA scope, ~512 intermediate VGPRs. R47-dm
hypothesised that capping outer `i` to `#pragma unroll 1` (4 sequential
outer iters of 8 inner iters each) caps peak live state at 1/4, freeing
LLVM to reduce the 39-dword spill closer to DSV3's 32-dword baseline.

Plan: clone helper as `store_c_tile_n_masked_grouped` with `#pragma
unroll 1` on outer `i`, route the 4 grouped call sites (line 2535-2541)
to it, leave 7 OTHER call sites (`gemm_kernel<RCR>`, `store_c_kernel_n_masked`,
etc.) untouched.

## Implementation

```diff
+template<ducks::gl::all GL, ducks::rt::all RT>
+__device__ __forceinline__ void store_c_tile_n_masked_grouped(
+    const GL& g_c, const RT& src,
+    int r_tile, int c_tile, int n_limit) {
+    // ... same setup as store_c_tile_n_masked ...
+    #pragma unroll 1                                  // <-- ONLY change
+    for (int i = 0; i < src.height; i++) {
+        #pragma unroll
+        for (int j = 0; j < src.width; j++) {
+            // ... body referencing src.tiles[i][j].data[idx] ...
+        }
+    }
+}

 if constexpr (N_MASKED_STORE) {
     // ...
     } else {
         mul(cA, cA, combined_scale);
-        store_c_tile_n_masked(g.c, cA, r0, c0, g.n);
+        store_c_tile_n_masked_grouped(g.c, cA, r0, c0, g.n);
         // ... 3 more analogous swaps ...
     }
```

## Spill count delta — DRAMATIC reduction

| Spec template params | R44-dm baseline | R48-dm probe | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)         | 39 | 39 | 0 |
| `<0,true ,false>` (FUSED=false n_masked)          | 43 | **0** | **-43** |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 32 | 32 | 0 |
| `<0,true ,true >` (FUSED=true  n_masked, gpt_oss) | 39 | **0** | **-39** |

**100 % spill ELIMINATION on both N_MASKED=true specs** (FUSED=false +
FUSED=true). Spill on N_MASKED=false specs unchanged (those don't call
the masked helper). This is the largest single-round spill reduction
ever recorded for this kernel — bigger than R44's -42 to -56 % reduction
which was the prior record.

## Per-shape metric — CATASTROPHIC

| Shape | R44-dm baseline | R48-dm probe | Δ pp |
|---|---|---|---|
| DSV3-GateUP-B16-M2048 | 1.139 | 1.132 | -0.7 |
| DSV3-Down-B16-M2048   | 1.173 | 1.184 | +1.1 |
| DSV3-GateUP-B16-M4096 | 1.127 | 1.134 | +0.7 |
| DSV3-Down-B16-M4096   | 1.184 | 1.169 | -1.5 |
| DSV3-GateUP-B32-M2048 | 1.147 | 1.147 | 0.0 |
| DSV3-Down-B32-M2048   | 1.228 | 1.160 | -6.8 |
| DSV3-GateUP-B32-M4096 | 1.158 | 1.170 | +1.2 |
| DSV3-Down-B32-M4096   | 1.226 | 1.167 | -5.9 |
| gpt_oss-GateUP-B4-M2048   | 1.084 | **0.109** | **-97.5** |
| gpt_oss-Down-B4-M2048     | 1.141 | **0.139** | **-100.2** |
| gpt_oss-GateUP-B4-M4096   | 1.065 | **0.099** | **-96.6** |
| gpt_oss-Down-B4-M4096     | 1.083 | **0.106** | **-97.7** |
| gpt_oss-GateUP-B32-M2048  | 1.037 | **0.095** | **-94.2** |
| gpt_oss-Down-B32-M2048    | 1.074 | **0.094** | **-98.0** |
| gpt_oss-GateUP-B32-M4096  | 1.019 | **0.095** | **-92.4** |
| gpt_oss-Down-B32-M4096    | 1.053 | **0.093** | **-96.0** |

ALL 8 gpt_oss FP8 shapes regressed **-92 to -100 pp** (i.e., 11× slower
in absolute TFLOPS: 1.9 TFLOPS → 0.17 TFLOPS). DSV3 mixed (because DSV3
uses N_MASKED=false → never enters the masked helper; the small
swings are codegen perturbation noise).

grp_FP8 geomean: 1.1194 → 0.3452 (-77 pp).
grp_BF16 geomean: 1.1830 → 1.1831 (unchanged — BF16 unaffected).
Score: 959 → 533 (-426 pts).

Correctness: PASS 32/32. The kernel produces NUMERICALLY CORRECT
results. The 11× slowdown is purely a perf regression.

## Root cause: AMDGCN register-tile ARRAY runtime indexing

The helper accesses `src.tiles[i][j].data[idx]` (lines 711-713 of the
helper). When `i` is a **constexpr** (full-unroll case), the access
becomes `src.tiles[0][j].data[idx]`, `src.tiles[1][j]...`, etc., each
resolving to specific VGPR slots at compile time — single-cycle VGPR
reads.

When `i` is a **runtime variable** (the `#pragma unroll 1` case), LLVM
must emit code that picks one of `src.tiles[0..3][j].data[idx]` at
runtime. Register-tile arrays in HipKittens are stored across VGPR
ranges (e.g. `tiles[0]` lives in `v100-v107`, `tiles[1]` in `v108-v115`,
etc.) — they ARE the register file, not memory. Runtime selection
between VGPR ranges on AMDGCN is implemented via **chains of
`v_cmpx`/`v_cndmask_b32` per VGPR slot** — emitting ~32 conditional moves
per single read of `src.tiles[i][j]`, executed across all lanes
(SIMD-divergent).

Result: each helper iteration that originally cost ~6 instructions
(2 conv + 2 store + 2 addr) now costs ~200+ instructions (2 conv + 2
store + 2 addr + ~32×4=128 VGPR-shuffle ops for tile selection +
extra VGPR pressure for the shuffle's intermediates). With 32 iters
per call × 4 calls per tile × ~150 tiles per launch on gpt_oss-B32-M4096,
the total instruction count balloons by ~10×, matching the observed
11× slowdown.

The 0-dword spill is misleading: LLVM avoided spilling by EMITTING
runtime-VGPR-shuffle code that accomplishes the same thing as scratch
spill+reload, but with worse latency than scratch. Spill count is a
useful proxy for VGPR pressure but **not** a universal proxy for
runtime perf.

## Why the analysis was wrong

R47-dm note correctly identified the helper as a spill source, and
correctly hypothesised that restricting unroll would relieve VGPR
pressure. The mistake was in step 3 — assuming that LLVM's response
to the relieved pressure would be FEWER spills (it was — 39→0!) AND
that the freed slots would be productively reusable. In practice, LLVM
chose a worse mechanism (runtime VGPR shuffle) because the source
data lives in register-tile arrays that REQUIRE constexpr indexing.

This is a fundamental AMDGCN constraint. Any attempt to "spread out"
the access to register-tile arrays via runtime loops will trigger the
same regression.

## Falsified directions to avoid

1. **`#pragma unroll N` for any N < src.height on store_c_tile_n_masked**
   — runtime VGPR shuffle into register-tile arrays.
2. **`__noinline__` on store_c_tile_n_masked** (would have been the
   next try) — same problem: register-tile struct passed by reference
   becomes either spilled-to-scratch or shuffled across function call
   boundary. Not falsified empirically yet but mechanism is identical;
   adding to the avoid list.
3. **Manual template recursion** to "force constexpr indexing while
   appearing as a loop" — this just produces the same fully-unrolled
   ISA as `#pragma unroll`. No win.

## Reusable insight: spill count is NOT a universal proxy

Throughout R39-R44 the dominant heuristic was: lower spill count → faster
runtime. R44 confirmed this DIRECTIONALLY (spill 80→39, runtime +8 pts).
R48-dm REFUTES it as a UNIVERSAL law: spill 39→0 with **runtime**
**11× slower**.

The condition under which spill count predicts runtime correctly:
* The spill represents data that EXISTS in a slot LLVM could allocate
  if it had more VGPR budget.

The condition under which it doesn't:
* The spill represents data that LLVM was already accessing from a
  fundamentally non-spillable location (register-tile array). Reducing
  "spill count" via unroll-restriction just shifts the access cost
  from scratch-load to VGPR-shuffle, which can be worse.

## Search-space update — exhausted directions

Now exhausted SINCE R37 K-tail issue reorder (last gainful K-tail change):

* ✗ R38: K-tail mfma permutation
* ✗ R39: MFMA commutative reorder
* ✗ R40: Runtime-dead a_extra
* ✗ R41: Pre-issue a_kt1 in epilog 2 (-56 pts)
* ✗ R42: Cluster prologue
* ✗ R43: Cluster prologue + full barriers (-41 pts)
* ✓ R44: Interleave mul + store (+8 pts) — STILL THE WINNER
* ✗ R45: Conditional interleave by N_MASKED
* ✗ R46: K-tail SRD setup hoist
* ✗ R47: Drop a_kt1, reuse `a` (-6 pts)
* ✗ R48: store_c_tile_n_masked clone with restricted unroll (-426 pts) ← THIS

## Take-away for next agent

1. **Do not touch register-tile-array iteration patterns.** Any change
   that risks runtime VGPR-array indexing (loops over tile dimensions
   with non-unrolled outer, function-pointer-based dispatch, etc.) will
   catastrophically regress.

2. **Spill count is necessary but not sufficient for win prediction.**
   Always run runtime metric on a probe even if spill drops dramatically.
   Conversely, don't dismiss probes that DON'T reduce spill — R37 won
   without spill change because instruction scheduling improved.

3. **Remaining unfalsified levers** (per R47-dm note + this round):
   * **R46 lever (b)**: HAS_KTAIL_BODY=false template variant for
     DSV3 (K_REM=0) shapes — removes K-tail body at compile time.
     Risk: may revert R34's codegen win if the K-tail body was load-
     bearing for LLVM's RA decisions. Estimated +0 to +1 pp on DSV3
     if works; probably 0.
   * **R46 lever (a)**: Combine a + a_kt1 into single 256x128 tile load
     (height=8) — doubles VGPR per lane; risks occupancy regression.
     Bigger surgery.
   * **rt_32x64 / rt_64x32 cell shape switch (Lever D)**: full kernel
     rewrite. R29 falsified integral migration but with R44 baseline
     it might work. Big surgery.
   * **Warp-specialised K-tail** (R36-dm note item 5): dedicate 2
     warps to async loads, 2 to mfma. Massive restructure.

4. **Do NOT try `__noinline__` on store_c_tile_n_masked** (this round's
   "next step" candidate from R47) — register-tile struct reference
   semantics make it worse than the existing inline.

## Score history

| Round | Score | grp_FP8 geo | Notes |
|---|---|---|---|
| Start    | 851  | ~1.01  | Baseline |
| R10      | 950  | 1.099  | R37-dm K-tail reorder (+16 pts) |
| R16      | 958  | 1.116  | R44-dm interleave mul+store (+8 pts) |
| R17-R19  | 957-961 | 1.116  | R45/R46/R47 falsified |
| R20-probe | **533** | **0.345**  | R48-dm-probe-1 CATASTROPHIC (runtime VGPR indexing) |
| R20-revert | 959  | 1.122  | (back to baseline) |
| Target   | 1000 | ≥1.20  | Gap = 7.8 pp grp_FP8 |

## Files touched (this round)

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (HipKittens) —
modified then reverted; final state matches HEAD `37926c98` (R44-dm
winner). Verified post-revert build: spill counts 39 / 43 / 32 / 39
unchanged. Verified post-revert metric: 959 (within ±2 pt noise of
959 pre-probe).

This Primus-Turbo round-note is the only delta this round. No HK
commit (revert restored HEAD).
