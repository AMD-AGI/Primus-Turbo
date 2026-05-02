# Round 32 — FP8 grouped: artificial loop-iter dependency to defeat IR-level LICM hoist (FALSIFIED)

## Summary

R32 attempted IR-level LICM defeat for the 30-dw prologue spill in `<0,T,T>`
(gpt_oss FUSED_KTAIL+N_MASKED template). The mechanism was an
`asm volatile("" : "+v"(laneid_loop_dep))` clobber on a value derived as
`laneid + ((gt - blockIdx.x) & 0)` (provably 0 but opaque to LLVM IR
analysis), threaded through `store_c_tile_n_masked` via a new
`laneid_in` parameter.

**Result: FALSIFIED at metric and at the structural level.**

| Variant | scratch_store (TT) | scratch_load (TT) | spill (TT) | Metric (3-trial median) |
|---|---|---|---|---|
| R30 baseline (no probe) | 60 | 285 | 37 dw | 979 |
| R30 anti-CSE asm-volatile on offsets | 60 | 165 (-42%) | 37 dw | 977 (-2 pts, FALSIFIED) |
| R31 sched_barrier(0) | 60 (no asm change) | 285 | 37 dw | 977 (no codegen change) |
| **R32 IR-level LICM defeat** | **60** | **166 (-42%)** | **38 dw (+1)** | **977 (-2 pts, FALSIFIED)** |

R32 is functionally identical to R30 at the asm level (same -42% reload
reduction via CSE-break across the 4 masked-path call sites), with the
extra cost of +1 VGPR live (laneid_loop_dep itself).

## What R32 actually proved

The **prologue spill_store count is unchanged** because the spilled
offsets belong to the **fast path** (kittens shared `store(g.c, ...)` helper),
not the masked path. The fast-path 4 stores still CSE-hoist their 30
unique per-lane buffer-store offsets to function entry, where they spill
across the K-iter (256 VGPR cap).

R32's loop-dep mechanism only affects the masked-path 4 stores (because
that's where the new helper signature was applied). Net effect on the
masked path: offsets are no longer CSE'd → computed locally per call →
no spill_store contribution. Net effect on the fast path: zero (kittens
shared helper unchanged).

**Disasm proof** (objdump on `_Z18grouped_rcr_kernelILi0ELb1ELb1EEv...`):

```text
R30 baseline:           60 scratch_store,  285 scratch_load   (CSE shared)
R32 loop-dep masked:    60 scratch_store,  166 scratch_load   (masked-path CSE broken)
                                          ^^^ -42% same as R30
```

`scratch_store` line numbers in disasm: 198-330 (function PROLOGUE),
identical to baseline. The spill is structurally bound to the fast-path
hoist that we did not touch.

## Why we stopped here (didn't extend to fast path)

To eliminate the prologue spill entirely, both fast and masked paths
would need to defeat LICM. That requires either:

1. **Modifying kittens shared `store(...)` helper in
   `include/ops/warp/memory/tile/global_to_register.cuh`** to take an
   optional `laneid_in` parameter. Affects ALL kernels (dense + grouped,
   BF16 + FP8) → high blast radius for a defeated lever class. Risk of
   regressing dense FP8 / BF16 grouped that are already close to target.

2. **Copying the kittens `store(...)` body locally into
   `kernel_fp8_layouts.cpp`** (~70 lines for col_layout RT) and using
   it for the fast-path branches. Scoped, but introduces a maintenance
   fork of the kittens helper — every change to the upstream `store(...)`
   would need to be replicated.

R30 + R31 + R32 already established three independent FALSIFICATION
points for the spill-reduction lever class:

* **R30**: anti-CSE asm-volatile DIRECTLY on offset compute → -42% reload,
  -2 pts metric.
* **R31**: sched_barrier(0) at K-iter boundary → no asm change at all
  (wrong tool layer).
* **R32**: IR-level loop-iter dependency → -42% reload (same as R30),
  -2 pts metric, +1 dw spill.

All three confirm the same conclusion: **reload count reduction does NOT
translate to metric improvement** because

(a) spill RELOADS are L1-cached after the first pass through the loop
    (~21 cy effective, not 80 cy VMEM cost), and the gpt_oss kernel
    loops over enough tiles to amortize the cold-cache cost; and
(b) the prologue spill_store cost (30 × 80 cy at function entry,
    once per work-group) is **untouched** by any of these levers
    because they all operate on the loop-body offset uses, not the
    function-entry CSE'd hoist itself.

To break (b) we would need to either prevent the hoist (requires
modifying upstream kittens or duplicating ~70 lines per template) OR
reduce the live-VGPR pressure that forces the spill in the first place
(requires the architectural-level Lever A/B from the original task brief
— async global→LDS or dual-LDS ping-pong, both deferred as
high-effort high-risk rewrites).

## Decision

**Acknowledge structural ceiling at 977-981 for `<0,T,T>` template.**

The R30 → R31 → R32 sequence has saturated the "anti-CSE / anti-hoist /
loop-body offset shuffling" lever class. Three independent attacks
yielded zero metric improvement. Continuing with variants (e.g.,
extending R32 to fast path, or copying kittens `store` locally) is
predicted by the disasm to give at most **~30 fewer prologue
scratch_store + reload save** — but these are amortized across the per-CU
tile count, and three rounds of evidence say the metric impact will be
within ±2 pts noise band.

The remaining uncovered levers from the original task brief are:

* **Lever A (async global→LDS, +5-10pp)**: requires `global_load_lds_*`
  intrinsic adoption + restructuring the K-iter pipeline. Major rewrite.
* **Lever B (dual/triple LDS ping-pong, +3-6pp)**: LDS budget
  constrained (139796 bytes used vs 64KB cap on gfx950 — actually
  already over LDS budget per-CB, double-buffer impossible without
  reducing tile size).
* **Lever D / 32x32x64 mfma**: previously falsified on original
  16-shape suite; could re-probe on Qwen GateUP only since K=4096 is
  different from the falsification load. Requires R38+ pre-validation
  gate (single-warp mfma_323264 vs mfma_1616128 microbench, ≥3pp
  required to continue).

## Files touched (then reverted)

* `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  — Added `laneid_in` parameter to `store_c_tile_n_masked` (default -1
  preserves dense-kernel callers' semantics), threaded
  `laneid_loop_dep` through 4 grouped-kernel masked-path call sites
  via `asm volatile("" : "+v"(...))` clobber. **Reverted** post-metric.

## Metric trace

Baseline (R31 sanity, single trial): 978
R32 trial 1: 977
R32 trial 2: 979
R32 trial 3: 977
R32 median: 977 (-1 vs baseline single-trial, -2 vs R30 3-trial median 979)

Spill counts (kernel resource remarks):

| Template | R30 baseline | R32 (scoped to N_MASKED branch) |
|---|---|---|
| `<0,F,F>` (dead) | 54 | 54 |
| `<0,T,F>` (dead) | 38 | 39 (+1) |
| `<0,F,T>` (DSV3+Qwen) | 34 | 34 |
| `<0,T,T>` (gpt_oss) | 37 | 38 (+1) |

DSV3+Qwen template unchanged (no metric risk). gpt_oss template +1 dw
spill (consistent with the -1 pt median).

## Next-round recommendation

R33 should **NOT** attempt:

* Extending R32 to fast path (predicted +0 to -2 pts based on R30/R32
  precedent, since the prologue spill cost dominates and reloads are
  L1-amortized).
* Any further "shuffle the offset compute" probe.
* Any sched_barrier mask sweep (R31 falsified the entire intrinsic
  for IR-level hoisting concerns).

R33 should EITHER:

* **(a) Accept plateau at 977-981 and stop** — patience=4/30, but the
  asm-level evidence is conclusive. Document final note and exit-cycle.
* **(b) Pivot to Lever A (async global→LDS, gfx950 `global_load_lds_*`
  intrinsic)** — major rewrite, multi-round commitment. Predicted
  +5-10pp on geomean if it lands, with significant risk of correctness
  regressions during the rewrite. Only viable with 10+ rounds of budget.
* **(c) Pivot to Lever D-Qwen (32x32x64 mfma on Qwen GateUP only)** —
  requires R33 single-warp microbench validation gate before any
  kernel-level change. Bounded scope (4/24 cases), predicted upper
  bound +3-5pp on Qwen GateUP cluster only ≈ +0.5-1pp on geomean.

Recommendation: **(c)** is the only positive-expected-value lever
remaining within a 5-round budget. **(b)** is high-EV but high-cost.
**(a)** is correct if the user wants to lock the current best (981) and
stop spending budget on FP8 grouped.
