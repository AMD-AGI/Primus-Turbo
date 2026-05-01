# Round-40-dm · FP8 grouped — MFMA commutative reorder + runtime-dead `a_extra` both FALSIFIED

**Status**: 2 probes tested. Reverted to R37 winner state.
Score unchanged: 950 baseline = 950 after revert.

## R39-dm: MFMA commutative reorder — NEUTRAL

Swap cA/cB and cC/cD mfma order inside the K-tail block (accumulators
are independent → commutative):

```cpp
// Before (R37 state):
rcr_mma(cA, a,     b0);  // A = ... + a·b0
rcr_mma(cB, a,     b1);
asm vmcnt(0);
rcr_mma(cC, a_kt1, b0);
rcr_mma(cD, a_kt1, b1);

// R39 probe:
rcr_mma(cB, a,     b1);
rcr_mma(cA, a,     b0);
asm vmcnt(0);
rcr_mma(cD, a_kt1, b1);
rcr_mma(cC, a_kt1, b0);
```

**Result**: 950 → 950 (exact tie, measurement noise level). No codegen
impact. LLVM treats commutative mfmas identically for scheduling
regardless of source-code order.

**Conclusion**: MFMA commutative reorder does NOT reshape LLVM's
liveness graph. The cA/cB/cC/cD register allocation is determined by
their USAGES (in `mul()` + `store()` later), not by mfma issue order.

## R40-dm: runtime-dead `a_extra` with forced load+mfma — REGRESSION (−35 pts)

Hypothesis: add a volatile-guarded runtime-dead branch that writes
`a_extra` and uses it in a dead mfma. LLVM can't DCE volatile-guarded
code, so `a_extra`'s liveness extends through the dead block,
reshaping register allocator's decisions for the live code.

```cpp
A_row_reg a_extra;
volatile uint32_t never_fire = 0;
if (never_fire) {
    load_a_kt(a_extra, 0);   // 8 buffer_loads to a_extra
    rcr_mma(cA, a_extra, b0); // dead mfma using a_extra
}
// real K-tail mfma sequence below
```

**Result**: 950 → **915 (−35 pts)**. grp_FP8 geomean drops 1.0989 →
1.0174. Correctness PASS but performance catastrophic.

**Post-mortem**:
- VGPR spill count unchanged at 52 (resource usage remark).
- LLVM allocated registers for the dead mfma's input/output tiles
  (a_extra expands to several VGPRs, cA already live).
- The `volatile` load of `never_fire` forces an actual HBM/LDS access
  per tile per warp — extra bandwidth cost.
- The dead branch occupies ICache lines, pushing out hot-path code.
- Net: register pressure + branch cost + icache pollution >> any
  theoretical liveness-reshape benefit.

**Lesson**: runtime-dead-but-syntactically-live code costs MORE than
it helps. R34's `a_kt1` succeeded because it was RUNTIME-LIVE for
gpt_oss (actually executed) and merely LIVENESS-LIVE for DSV3 (in
scope but dead). The costs were:
- DSV3: zero runtime cost, liveness-reshape benefit.
- gpt_oss: actual useful work + liveness-reshape benefit.

In contrast, R40 introduces code that is **never useful at runtime**
for ANY spec, and imposes an unconditional volatile memory access
per-tile. Fundamentally asymmetric cost-benefit vs R34.

## Permutation class exhaustion (R38-dm)

Combined with R38-dm: the K-tail permutation-in-4-loads class is
exhausted. Only R37's `[b0, b1, a, a_kt1]` is empirically beneficial.
Other angles (mfma reorder, dummy decls, runtime-dead branches) all
fail.

## Remaining search space for R13+

Ideas not yet tested:

1. **Split load_a_kt lambda into fewer-load units**: instead of 8
   b128 loads per call, 4-load or 2-load units. Gives LLVM explicit
   scheduling points. Example:
   ```cpp
   load_a_kt_half<0>(a_kt1, 1);  // loads h=0,1 (4 b128)
   load_a_kt_half<1>(a_kt1, 1);  // loads h=2,3 (4 b128)
   ```

2. **Main-loop RCR_STEADY_VMCNT tuning** — task body says vmcnt is
   frozen but specifically "K-tail vmcnt". Main-loop vmcnt might be
   untouched (unclear from frozen list — verify before trying).

3. **Prologue optimization for short-K** — gpt_oss has 22 K-iters vs
   DSV3 56. Prologue (2 INIT stages) is ~9% overhead for gpt_oss vs
   3.6% for DSV3. But R31/R32 showed prologue is extremely fragile
   (NaN on ANY change). Any prologue probe needs heavy numerical
   validation.

4. **Attack gpt_oss through BF16 layer** — BF16 gpt_oss has similar
   issues (spec <0,true,true> path), with K=2880 too. BF16 already
   at 1.18x. If there's a shared optimization, it might work for
   both.

5. **Warp-specialised K-tail** — gpt_oss K-tail uses the same 4 warps
   as main loop. If we could dedicate 2 warps to K-tail async loads
   and 2 to mfma, we'd hide HBM latency. Massive restructuring.

Next round should pick option 1 (split lambda) as lowest-risk probe.

## Score history

| Round | Best score | grp_FP8 | Best FP8 shape |
|---|---|---|---|
| Start | 851 | 1.010 | — |
| R7 | 932 | 1.025 | all 8 DSV3 +3-14 pp |
| R10 | 950 | 1.099 | DSV3-Down-B32-M4096 = 1.203 ≥ target |
| R11 | 948 | 1.094 | (noise) |
| R12 | 950 | 1.099 | unchanged |
| Target | 1000 | ≥1.200 | all 16 ≥ 1.20 |

Gap: grp_FP8 geomean needs +10 pp. Currently DSV3 at 1.13-1.20 (close);
gpt_oss at 0.98-1.11 (long pole). Closing the gpt_oss gap by ~5-7 pp
per shape → grp_FP8 ≈ 1.15 → score ~975.
