# Round-42-dm · FP8 grouped — prologue cluster-reorder HALVES LLVM spill but breaks wm==1 half-barrier sync

**Status**: 2 prologue-reorder variants tested. Both FALSIFIED via fwd-NaN
on DSV3-GateUP-B16-M2048. Reverted.
Score 949 → 816 (B-first cluster) → 819 (B-first stage-0-only) →
820 (A-first cluster) → 947 (revert).

## Hypothesis

R37-dm won +16 pts by reordering K-tail loads from interleaved to
clustered B-first `[B0, B1, A, A_kt1]`. The current FP8 grouped
**prologue** uses strict interleaving `[B0, A0, B1, A1]` for the 4
stage-0 loads. Hypothesis: applying the same R37 cluster-first trick
to the prologue should yield similar LLVM register-allocation
reshape benefits.

## Variants tested

### Variant 1: B-first cluster, both stages (full reorder)

```cpp
// Stage 0:
rcr_8w_load_hoist(b_tile(tic, 0), ...);  // B
rcr_8w_load_hoist(b_tile(tic, 1), ...);  // B  (was A0)
rcr_8w_load_hoist(As[tic][0], ...);      // A  (was B1)
rcr_8w_load_hoist(As[tic][1], ...);      // A
if (wm == 1) __builtin_amdgcn_s_barrier();
TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
__builtin_amdgcn_s_barrier();
// Stage 1: similar B-cluster reorder
```

### Variant 2: B-first cluster, stage-0 ONLY (stage-1 untouched)

Same as variant 1 but stage-1 keeps original `[B0, A0, B1]` order.

### Variant 3: A-first cluster, stage-0 ONLY

```cpp
rcr_8w_load_hoist(As[tic][0], ...);      // A  (was B0)
rcr_8w_load_hoist(As[tic][1], ...);      // A  (was A0)
rcr_8w_load_hoist(b_tile(tic, 0), ...);  // B  (was B1)
rcr_8w_load_hoist(b_tile(tic, 1), ...);  // B  (was A1)
```

## Static codegen impact (`-Rpass-analysis=kernel-resource-usage`)

All 3 variants produce IDENTICAL spill numbers — clustering itself
(not which side is first) is the codegen lever:

| Spec template params | Baseline | Clustered (any) | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)  | 67 | 67  | 0    |
| `<0,true ,false>` (FUSED=false n_masked)   | 76 | 76  | 0    |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 72 | **36** | **-36 (-50%)** |
| `<0,true ,true >` (FUSED=true  n_masked,  gpt_oss)| 82 | **56** | **-26 (-32%)** |

**This is the largest static spill reduction observed since R3-dm.**
The clustered prologue gives LLVM significantly more freedom to share
SGPRs (A and B base pointers folded across consecutive loads) and
re-use VGPR slots (consecutive same-side loads have identical lane
patterns).

## Runtime: ALL variants FAIL with fwd-NaN on the same shape

| Variant | grp_FP8 geo | score | correctness |
|---|---|---|---|
| Baseline (interleaved [B0,A0,B1,A1]) | 1.0962 | 949 | PASS 32/32 |
| B-first both stages   | 0.8126 | 816 | **FAIL: DSV3-GateUP-B16-M2048 fwd-NaN** |
| B-first stage-0 only  | 0.8172 | 819 | **FAIL: DSV3-GateUP-B16-M2048 fwd-NaN** |
| A-first stage-0 only  | 0.8195 | 820 | **FAIL: DSV3-GateUP-B16-M2048 fwd-NaN** |

The NaN is **deterministic, same shape every run, same FAIL mode**.

## Root cause: wm==1 half-barrier sync is order-dependent

The prologue uses an asymmetric half-barrier pattern:

```cpp
// Stage 0 loads (4)
if (wm == 1) __builtin_amdgcn_s_barrier();  // ONLY wm==1 issues
TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
__builtin_amdgcn_s_barrier();  // full barrier (all warps)
```

Paired with C-store epilog:

```cpp
if (wm == 0) __builtin_amdgcn_s_barrier();  // ONLY wm==0 issues
// ... store(g.c, ...) ...
asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
__builtin_amdgcn_s_barrier();  // full barrier
```

Across persistent-loop iterations, wm==1's "extra barrier" at line
2147 pairs with wm==0's "extra barrier" at line 2479 (from previous
iteration's C-store). They form a balanced workgroup-wide sync.

The exact barrier counter increment depends on the ORDER and TIMING
of preceding LDS writes. `rcr_8w_load_hoist` internally does
`s_mov m0, <lane offset> + buffer_load_dwordx4 ... lds`. When 4
consecutive calls all target B-tiles (clustered), the m0 broadcast
pattern is different from interleaved B-A-B-A: different SGPR
fold/spill behaviour upstream of the barrier intrinsic. This shifts
when each lane's ds_write actually hits the LDS bank, and the
half-barrier sync's "wave 0/wave 1 arrival timing" assumptions
break.

**Empirically falsified**: ANY within-stage cluster-reorder breaks the
sync, regardless of which side (A or B) is clustered first or whether
stage-1 is also clustered.

## Comparison to R31-dm (prologue collapse)

R31-dm collapsed the 2 INIT stages into 1 and BROKE wm==1 half-
barrier sync (5/16 DSV3 fwd-NaN). R32-dm confirmed inter-barrier
vmcnt(4) is load-bearing.

R42-dm extends this lesson: even **preserving** the 2-stage structure
+ vmcnt scheme, just **reordering loads within stage 0**, breaks the
same wm==1 sync. The half-barrier pattern is sensitive to load order,
not just stage count.

The prologue's interleaved `[B0, A0, B1, A1]` order is a load-bearing
correctness invariant.

## Per-shape regression (B-first cluster, both stages)

Largest hits (vs baseline 949):
- DSV3-GateUP-B16-M2048: 1.124 → **0.000 (NaN)**
- DSV3-Down-B16-M4096: 1.184 → 1.113 (-7 pp)
- DSV3-Down-B32-M4096: 1.207 → 1.128 (-8 pp)
- gpt_oss-Down-B4-M2048: 1.115 → 1.108 (-0.7 pp)
- All other FP8 shapes: -0 to -2 pp

So even WITHOUT the NaN, the perf would have regressed slightly on
2 DSV3-Down shapes (-7 to -8 pp). The "spill reduction" did NOT
translate to runtime gain — likely because the new schedule has
worse instruction-level parallelism in the main loop (LLVM moved
spills around but didn't actually reduce critical-path stalls).

This further reinforces R30/R33/R39/R41's lesson: **static spill
count is a misleading proxy for runtime perf in this kernel**.
LLVM's "spill" counter measures one cost dimension; the actual
runtime is dominated by HBM/MFMA pipeline stalls that re-allocation
shifts but doesn't eliminate.

## Take-away for next agent

1. **Do NOT** try any prologue load reorder. The wm==1 half-barrier
   sync requires `[B0, A0, B1, A1]` interleaving (or trivially
   different orderings within the same interleave class — untested,
   but B-A-A-B / A-B-B-A / etc. likely also fail).

2. **Promising lever** for a future round (NOT this round, too risky):
   replace the wm==1 / wm==0 half-barrier pair with a different sync
   mechanism (e.g., wave-uniform SGPR signal + full s_barrier, or
   barrier-counter-based `s_barrier_signal/wait` pair). If the new
   sync mechanism is order-independent, then clustered prologue
   becomes legal → potentially -26 to -36 dwords of spill on FUSED=
   true specs → could be the biggest single codegen win in the
   trail. But it's a significant refactor (touches both prologue
   sync AND C-store sync) and needs careful numerical validation
   across all 32 shapes + DoD smoke. Estimated 2-3 rounds of work.

3. The R37 K-tail block + R34 FUSED-extension have already harvested
   the easy LLVM liveness wins. **All future codegen-only levers
   look exhausted**.

4. The realistic remaining headroom is in **reducing the per-tile
   fixed overhead** (binary search, prologue, epilog 1+2 sync). All
   three are documented as fragile. A coordinated overhaul of the
   sync pattern (point #2) is the only viable path.

## Score history

| Round | Best | grp_FP8 geo | Notes |
|---|---|---|---|
| Start | 851 | ~1.01 | Baseline |
| R7 (cron) | 932 | 1.025 | R34-dm FUSED extension (+17 pts) |
| R10 (cron) | 950 | 1.099 | R37-dm K-tail reorder (+16 pts) |
| R11-R13 | 950 | 1.094 | R38/R39/R40/R41 falsified |
| R14 | 949 | 1.090 | R42-dm prologue cluster falsified (this) |
| Target | 1000 | ≥1.20 | Gap = 11 pp grp_FP8 |

## Files touched (this round)

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (HipKittens) —
modified 3 times then reverted; final state matches HEAD `04f82d49`.

No Primus-Turbo source changes. This note is the only Primus-Turbo
delta.
