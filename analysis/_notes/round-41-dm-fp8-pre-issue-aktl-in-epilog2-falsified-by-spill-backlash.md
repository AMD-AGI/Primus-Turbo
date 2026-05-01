# Round-41-dm · FP8 grouped — pre-issue `a_kt1` in epilog 2 FALSIFIED by R33-pattern spill backlash

**Status**: 1 probe FALSIFIED. Reverted to R37 winner state.
Score 950 → 894 (probe) → 949 (revert, within noise of 950).

## Hypothesis

The K-tail for gpt_oss (K_REM=64) issues 24 buffer_load_b128 (8 a_kt1
+ 4 b0 + 4 b1 + 8 a) and waits with split-vmcnt: `vmcnt(8)` for first
16 retired (b0+b1+a under R37 issue order), fire `cA, cB`, then
`vmcnt(0)` for remaining 8 (a_kt1), fire `cC, cD`.

R36-dm analysis estimated K-tail ~648 cy total, dominated by HBM
latency. If we **pre-issue the 8 `a_kt1` buffer_loads inside epilog 2's
cA→cB transition**, those loads get ~150 cy head-start (overlap with
epilog 2's cB+cC+cD mfmas + barriers), and the K-tail block only needs
to wait for `b0+b1+a` (16 loads, ~520 cy max). Combined sequence:

* Epilog 2 cA→cB: issue 8 a_kt1 loads (out-of-band, no vmcnt change)
* Epilog 2 cB, cC, cD mfmas + barriers (~150 cy hide)
* K-tail: issue b0+b1+a (16 loads); `vmcnt(8)` waits 16 retired (FIFO
  = a_kt1 + b0 + b1 retired); fire `cC, cD` (use a_kt1 + b0/b1);
  `vmcnt(0)` waits remaining 8 (a); fire `cA, cB`.

Reorder mfma to `[cC, cD, cA, cB]` preserves R12-dm split-vmcnt
advantage (cC/cD fire while `a` still in flight). Expected savings:
~5 cy direct K-tail + LLVM register-allocation reshape similar to
R34-dm's `a_kt1` declaration trick (+17 pts on DSV3 via codegen).

## Implementation

```diff
 // function-scope declarations
 A_row_reg a; B_row_reg b0, b1;
+A_row_reg a_kt1;          // hoisted from inside if constexpr (FUSED_KTAIL)
 rt_fl<...> cA, cB, cC, cD;
```

```diff
 // Epilog 2: cA mfma block ...
 __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
 __builtin_amdgcn_s_barrier();

+if constexpr (FUSED_KTAIL) {
+    if (g.fast_k < g.k) {
+        // Inline 8 raw_buffer_load_b128 → a_kt1.tiles[0..3][0]
+        // (lane-cell mapping identical to R3 path B)
+        ...
+    }
+}

 load_b(b1, b_tile(tic, 1), wn);
 ...  // rest of epilog 2 unchanged
```

```diff
 // K-tail block:
-A_row_reg a_kt1;             // (deleted; now at function scope)
-load_a_kt(a_kt1, 1);         // (deleted; pre-issued in epilog 2)
 load_b_kt(b0, 0); load_b_kt(b1, 1); load_a_kt(a, 0);
 asm volatile("s_waitcnt vmcnt(8)");
-rcr_mma(cA, a, b0); rcr_mma(cB, a, b1);   // (was: cA/cB first)
+rcr_mma(cC, a_kt1, b0); rcr_mma(cD, a_kt1, b1);  // cC/cD first
 asm volatile("s_waitcnt vmcnt(0)");
-rcr_mma(cC, a_kt1, b0); rcr_mma(cD, a_kt1, b1);
+rcr_mma(cA, a, b0); rcr_mma(cB, a, b1);
```

## Spill count delta (`-Rpass-analysis=kernel-resource-usage`)

| Spec template params | Pre-R41-dm | Post-R41-dm | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)  | 67 | 67  | 0    |
| `<0,true ,false>` (FUSED=false n_masked)   | 76 | 76  | 0    |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 72 | **100** | **+28** |
| `<0,true ,true >` (FUSED=true  n_masked,  gpt_oss)| 82 | **107** | **+25** |

The 16-VGPR `a_kt1` register lives across epilog 2's ~150 cy of mfma +
barrier sequence. LLVM is forced to spill ~25-28 additional dwords of
hot state (cA/cB/cC/cD partials + b0/b1 + sgpr scratch) to scratch
buffer to make room for a_kt1's 16 VGPRs.

This is the EXACT R33 pattern: trying to extend a register's live
range to enable scheduling-level optimization triggers an LLVM spill
backlash that costs more than any HBM latency hidden.

## Per-shape regression (R37 baseline=950 → R41-dm probe=894)

| Shape | Pre-R41 | Post-R41 | Δ pp |
|---|---|---|---|
| DSV3-GateUP-B16-M2048 | 1.116 | 1.076 | -4.0 |
| DSV3-Down-B16-M2048   | 1.154 | 1.064 | -9.0 |
| DSV3-GateUP-B16-M4096 | 1.130 | 1.095 | -3.5 |
| DSV3-Down-B16-M4096   | 1.184 | 1.011 | **-17.3** |
| DSV3-GateUP-B32-M2048 | 1.152 | 1.112 | -4.0 |
| DSV3-Down-B32-M2048   | 1.206 | 1.045 | **-16.1** |
| DSV3-GateUP-B32-M4096 | 1.158 | 1.125 | -3.3 |
| DSV3-Down-B32-M4096   | 1.208 | 1.042 | **-16.6** |
| gpt_oss-GateUP-B4-M2048   | 1.058 | 0.881 | **-17.7** |
| gpt_oss-Down-B4-M2048     | 1.119 | 0.937 | **-18.2** |
| gpt_oss-GateUP-B4-M4096   | 1.015 | 0.854 | **-16.1** |
| gpt_oss-Down-B4-M4096     | 1.052 | 0.875 | **-17.7** |
| gpt_oss-GateUP-B32-M2048  | 1.025 | 0.884 | **-14.1** |
| gpt_oss-Down-B32-M2048    | 1.030 | 0.897 | **-13.3** |
| gpt_oss-GateUP-B32-M4096  | 0.983 | 0.859 | **-12.4** |
| gpt_oss-Down-B32-M4096    | 1.014 | 0.881 | **-13.3** |

grp_FP8 geomean: 1.0979 → 0.9723 (-0.126).
ALL 16 FP8 shapes regressed. Worst hits: DSV3-Down family and
gpt_oss-GateUP family (-13 to -18 pp each).

Correctness: PASS (correct_fail=0/32, reject=0/32). Numerics OK.

## Why the analysis was wrong

R36-dm analysis underestimated the spill cost. The reasoning was
"a_kt1 = 16 VGPRs added; epilog 2 has 256 VGPR budget; should fit".
But LLVM's register pressure is not budget-limited — it's
scheduling-limited. With 67-82 dwords already spilled at the R37
baseline (signaling LLVM is at the spill knee), adding 16 more live
VGPRs across a tight schedule pushes it over the cliff. The 25-28 new
scratch_load/store pairs in the main loop (NOT the K-tail) cost more
cycles than any K-tail HBM hide.

This is ALSO the same lesson as R30 (sched_group_barrier saved spill
on paper but regressed runtime), R33 (epilog 1 mfma merge), and R39
(MFMA commutative reorder was neutral — LLVM optimum register
allocation isn't reachable via simple permutation).

## Search-space implications

After R37 (K-tail issue order win) + R38 (permutation class
saturated) + R39 (mfma reorder neutral) + R40 (runtime-dead a_extra
regressed) + R41 (pre-issue regressed):

The R37 K-tail block is at a **local LLVM register-allocation
optimum** that ANY perturbation pushes off. The block is small enough
that the LLVM scheduler has fully explored it; we're up against a
discrete codegen barrier, not a continuous knob.

**Real bottleneck**: gpt_oss's main-loop overhead per K-iter is large
relative to its short K_iter=22. Per-tile fixed overhead (binary
search ~70 cy + prologue ~800 cy + epilog 1+2 ~300 cy + K-tail ~650
cy + store ~100 cy ≈ 1920 cy) is amortized over only 22 K-iters
(~14400 cy main loop). Fixed-overhead share = 1920/16320 ≈ 12%, vs
~6% for DSV3 (K_iter=56). This 6 pp differential matches the
empirical gap.

## Remaining unexplored levers (for R12+)

Already-falsified:
* ✗ K-tail permutation (R38)
* ✗ K-tail mfma reorder (R39, R41)
* ✗ Runtime-dead a_extra (R40)
* ✗ Pre-issue a_kt1 in epilog 2 (R41)
* ✗ MFMA cell shape migration (R15, R29)
* ✗ N_MASKED spec collapse (R35)
* ✗ Sched_group_barrier (R30)
* ✗ Compiler hints (R28)
* ✗ Readfirstlane (R29)
* ✗ Prologue collapse (R31, R32)

Still unexplored:
1. **Port dense's 2-tile main loop to grouped** (RCR_TWO_TILE_MIN_KI
   = 28 in dense; gpt_oss ki_dyn=22 < 28 wouldn't qualify even with
   port). Dense's threshold suggests this WON'T help low-K shapes.
   Skip.
2. **Pre-issue `a` (M-slab 0) instead of `a_kt1`** — the bottleneck
   load by FIFO order. Would need a SECOND function-scope A_row_reg
   to avoid main-loop `a` clobber. Likely same spill backlash.
3. **Reduce per-tile fixed overhead** (binary search → 4 sequential
   cmp instead of 6, prologue collapse, epilog 1+2 merge). Each was
   tried and failed (R31, R32, R33).
4. **Attack via BF16 layer**: BF16 grouped is at 1.18, gpt_oss
   shapes have similar K=2880 issue. If a shared BF16 fix transfers,
   it may attack FP8 too. But task body says BF16 is frozen for this
   round.
5. **Warp-specialized K-tail**: dedicate 2 warps to async loads, 2
   to mfma. Massive restructure. Defer.

## Score history

| Round | Best | grp_FP8 geo | Notes |
|---|---|---|---|
| Start | 851 | ~1.01 | Baseline |
| R7 | 932 | 1.0247 | R34-dm FUSED_KTAIL extension (+17 pts) |
| R10 | 950 | 1.0987 | R37-dm K-tail reorder (+16 pts) |
| R11 | 950 | 1.0987 | (re-confirm) |
| R12 | 946 | (noise) | R40-dm runtime-dead falsified |
| R13 | 950 | 1.0944 | R41-dm pre-issue falsified, reverted |
| Target | 1000 | ≥1.20 | Gap ≈ 10 pp grp_FP8 |

## Files touched (this round)

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (HipKittens) —
modified then reverted; final state matches HEAD `04f82d49`.

No Primus-Turbo source changes. This note is the only Primus-Turbo
delta.

## Take-away for the next agent

* Stop pre-issuing K-tail loads into epilog 2. Spill backlash is too
  strong (R41 = -56 pts).
* Stop reordering K-tail mfmas. R39 + R41 confirm both are NEUTRAL or
  worse.
* The R37 K-tail block's local optimum is discrete; further wins must
  come from OUTSIDE the K-tail block.
* The gpt_oss bottleneck is fundamental main-loop fixed-overhead, not
  K-tail. Lever choice for next round should target either (a)
  prologue cost reduction (very high risk per R31/R32), (b)
  restructuring the persistent loop to amortize per-tile setup
  across multiple tiles, or (c) accept that the geomean ceiling is
  ~1.10 with current kernel architecture and look at warp
  specialization.
