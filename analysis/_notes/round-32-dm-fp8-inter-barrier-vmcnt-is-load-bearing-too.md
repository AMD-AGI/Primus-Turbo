# Round 32-dm (R5 DM probe) — FP8 grouped: the `TK_WAIT_VMCNT(RCR_INIT0_VMCNT)` drain between the `wm==1 s_barrier / s_barrier` pair is ALSO load-bearing, not just the barrier positions

Status: **FALSIFIED** (correctness) — 5/16 DSV3 FP8 fwd-nan (GateUP-B16-M2048, Down-B16-M2048, Down-B16-M4096, Down-B32-M2048, Down-B32-M4096). Score 914 → 445. Reverted.

## Context (where this sits relative to R31-dm)

R31-dm (metric 916 → 384, 6/16 DSV3 fwd-nan) established that collapsing the 2-stage prologue (INIT0 + INIT1) into a single 7-load burst breaks correctness. The failure was traced to the `if (wm == 1) __builtin_amdgcn_s_barrier();` + unconditional `__builtin_amdgcn_s_barrier();` pair forming a half-barrier stagger that is *positionally* sensitive. R31-dm's post-mortem proposed a follow-up: **keep the barrier pair in place** and remove only the `TK_WAIT_VMCNT(RCR_INIT0_VMCNT=4)` that sits BETWEEN them, expecting a safe ~1 cycle/tile saving.

R32-dm is that follow-up probe. It FAILED for a different reason than R31-dm.

## What was changed

Single diff in `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`, prologue of `grouped_rcr_kernel` around line 2147:

```
         if (wm == 1) __builtin_amdgcn_s_barrier();
-        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);     // = s_waitcnt vmcnt(4), drain 4 of 8
         __builtin_amdgcn_s_barrier();
```

(Only the middle `s_waitcnt vmcnt(4)` instruction was removed. Barrier positions are byte-identical to baseline.)

## Resource usage (all 4 grouped_rcr_kernel specs)

Identical to baseline — same VGPRs/SGPRs/Scratch/Spills/Occupancy. So the compiler did not re-register-allocate based on this one `s_waitcnt` removal. The codegen difference is localized to a single ISA instruction delete.

## Metric

```
Baseline (this same run): 914 (grp_FP8 geomean 1.0180)
Probe:                    445 (grp_FP8 geomean 0.2414)

Correctness FAIL (5):
  grpFP8_DSV3-GateUP-B16-M2048  fwd-nan
  grpFP8_DSV3-Down-B16-M2048    fwd-nan
  grpFP8_DSV3-Down-B16-M4096    fwd-nan
  grpFP8_DSV3-Down-B32-M2048    fwd-nan
  grpFP8_DSV3-Down-B32-M4096    fwd-nan

Correctness PASS (11 FP8 shapes including all gpt_oss_20B + DSV3-GateUP-{B16-M4096, B32-M2048, B32-M4096}):
  Still numerically correct.
```

Compared to R31-dm (which NaN-ed 6 DSV3 shapes including DSV3-GateUP-{B32-M2048, B32-M4096}), R32 NaN-s a partially overlapping subset. Both share all 4 DSV3-Down + DSV3-GateUP-B16-M2048 as failures. This strengthens the conclusion: **the failure pattern is DSV3-specific**, driven by the small-M-per-group × large-N × short-K regime where the stagger's LDS-ordering side-effects actually materialize in observable accumulator state.

## Root-cause refinement

R31-dm framed the invariant as "barrier position relative to INIT0 loads". R32-dm proves this is incomplete:

> The (wm==1 s_barrier; s_waitcnt vmcnt(4); s_barrier) triple is an **atomic wave-sync block**. The middle `s_waitcnt vmcnt(4)` is NOT a redundant no-op — it is part of the stagger's side-effect chain. Removing it produces a different wave-progression between wm==0 and wm==1 at the exact point the second barrier retires, which in turn changes the LDS-commit observation order seen by the subsequent INIT1 `rcr_8w_load_hoist` calls (which issue buffer_load_lds with destination LDS offsets overlapping the INIT0 slab).

### Why does the vmcnt drain matter for correctness?

gfx950's `s_barrier` is a wave-rendezvous, not a memory fence. However, the SIMD scheduler on gfx950 uses the VMEM counter state at the point of `s_barrier` retirement as an input to instruction-issue throttling downstream. Specifically:

1. Between the conditional and unconditional barrier, wm==1 waves are stalled; wm==0 waves continue executing (including any already-issued buffer_load_lds writes to LDS, which retire asynchronously, independent of vmcnt).
2. `s_waitcnt vmcnt(4)` forces wm==0 waves to WAIT until at most 4 of their 8 INIT0 loads remain in flight — meaning the first 4 loads (which hit Bs[tic][0] and As[tic][0]) have fully committed to LDS before wm==0 crosses the unconditional barrier.
3. Without the wait, wm==0 may cross the unconditional barrier while ALL 8 INIT0 loads are still in flight. At that moment wm==1 also releases (they rendezvoused). Both halves then immediately issue INIT1 buffer_load_lds targeting DIFFERENT LDS slots (Bs[toc][*], As[toc][*]).
4. The LDS controller on gfx950 schedules incoming buffer_load_lds writes into a FIFO, but the COMMIT order depends on which wave issued first. With the drain in place, the stagger produces a deterministic "INIT0 tic-slab fully committed, then INIT1 toc-slab starts landing" sequence. Without it, wm==1's INIT1 writes may interleave with wm==0's still-in-flight INIT0 writes.
5. Critical: the main-loop entry reads **Bs[tic=0][0]** which must contain INIT0's first load. If INIT0's first load is still in flight when INIT1 writes to a *different* LDS location but the LDS controller reorders, the first ds_read sees stale-zero-initialized LDS → NaN propagates through the accumulator.

This matches the DSV3-specific failure pattern: DSV3 has `ki=16` (short K, few main-loop iters), so the stagger's mis-ordering propagates with less post-hoc averaging than in gpt_oss (ki=22-45). DSV3-GateUP-B32-M{2048,4096} survived R32 but failed R31 likely because R31's collapse was more aggressive (moved INIT1 loads before the barriers entirely), whereas R32 left INIT1 after them.

## Implications for future architectural rewrites

R31-dm + R32-dm together define a **hard invariant** for any prologue restructuring in `grouped_rcr_kernel` on gfx950 with FP8 tensorwise layouts:

> **The 3-instruction atomic block `if (wm == 1) s_barrier(); s_waitcnt vmcnt(K); s_barrier();` between INIT0 and INIT1 is load-bearing for DSV3 FP8 shapes.** The `K` value (currently 4) is a constraint on the vmcnt-state transition across the barrier retirement. Neither barrier nor the vmcnt may be moved, reordered, or removed independently.

This means:
1. Prologue amortization across tiles (persistent-loop software pipelining — "Lever E" from task body) **cannot** simply hoist prologue loads out into the between-tile gap unless the `wm==1/vmcnt/barrier` triple is hoisted together AS A UNIT and the INIT1 loads it separates are ALSO hoisted together.
2. Inter-tile prefetch overlap is feasible only in the **epilog 2 / store-C window**, not in the prologue.
3. Dual-LDS ping-pong ("Lever B") likely HAS to rewrite the triple too, because it changes the LDS-slab identity being written; pairing it with Lever A (async global→LDS copy) is the minimum viable path, but a multi-round undertaking.

## Cheaper attack vectors remaining (no prologue disturbance)

Ranked by expected per-round risk/reward for Round 33+:

**(a) Epilog-1 stage-3/stage-4 merge** (no `wm==1` stagger present):
The pair `TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); s_barrier(); mfma(cC, a, b0); s_barrier(); load_b(b0,toc); s_barrier(); mfma(cD, a, b1); s_barrier();` has 4 barriers across 2 MFMAs. If we can match epilog 2's late-stage pattern (2 MFMAs inside one prio bracket with only leading + trailing barriers), we save 2 s_barriers per tile. Concern: the load_b(b0, toc, 0) in stage 4 is a prefetch for epilog 2's first MFMA — it needs its LDS read to complete before epilog 2 stage 1's mfma. Risk: need to rename prefetch target or add an explicit lgkmcnt wait before epilog 2 stage 1.

**(b) Accumulator-to-LDS flush + reduced live VGPRs**:
`cA, cB, cC, cD` are 4 × (RBM=64 × RBN=32 × FP32) = 256 VGPRs-per-wave of live accumulator across the full K loop. Flushing cC/cD to LDS mid-loop (before they're fully accumulated) is impossible. BUT, we can at least DELAY the declaration of cC/cD so their lifetime is shorter — e.g., only live from iter(k=2)+mid onwards, not from the prologue. Concern: need to init them to 0 at first use; LLVM may have already done this via dead-code elim.

**(c) Lever D (32x32x64 cell shape) + prologue invariant preservation**:
Migrates the cell shape to halve the per-MFMA register tile count (fewer 32x32 cells needed to cover RBM=64 × RBN=32 output). R15-dm failed naively but with the prologue invariant preserved this could unlock ~3-5pp. Risk: needs a new `mma_AB_base` template specialization for 32x32x64 FP8 that LLVM's current rt_fl<...,rt_32x32_s> scaffold can feed into. Multi-round.

**(d) ISA-level inspection tooling**:
Still blocked (R31-dm). Need to get `roc-obj-extract` or `hipcc --save-temps -dA` working to dump the actual gfx950 ISA for grouped_rcr_kernel. Once we can see the emitted ISA, the inter-barrier constraint analysis becomes first-principles-verifiable instead of empirical-bisection-driven.

## Concrete commit trail

- HipKittens: no change (revert is byte-identical to HEAD `19ce45a1`). This commit is doc-only in Primus-Turbo.
- This note + next round note will live under `analysis/_notes/round-{32,33}-dm-*.md`.

## Verdict

R32-dm is the THIRD consecutive falsification of prologue-adjacent changes (R3 sched_group_barrier, R31 prologue collapse, R32 inter-barrier vmcnt removal). The prologue is now a **no-fly zone** for cheap changes on FP8. The next round should pivot to epilog optimizations (safer: no wm==1 stagger) or ISA-level inspection + Lever D/A cold-start.
