# Round 58 — BF16 grouped, fwd K_STEP-stale-case discovery + KI=24/32/44-fuse SPILL FALSIFIED

## Goal coming in

R57 falsified the FUSED_KTAIL overlap hypothesis and recommended R58 =
work-stealing persistent kernel for wave imbalance. R58 starting metric
(GPU 3, single sample): score=907, worst shape =
`gpt_oss-GateUP-B32-M2048` ratio=1.034 (HK 1056.3 / TRT 1021.2,
weight 3) — TILES = 5888, iters/CU = 23 (perfectly balanced, NO wave
imbalance).

Today's worst shape is **NOT** wave-imbalanced; work stealing won't
help. Pivot to the next R57-doc lever: **KI specialization audit**.

## Discovery — STALE-CASE in dispatcher KI specs

Reading the dispatcher (kernel_bf16_dynamic.cpp line 4239+):

```cpp
switch (g.ki) {
    case 48:  ... R53 spec for "Qwen3-Down K=1536"
    case 56:  ...
    case 64:  ... covers Qwen3-GateUP K=4096
    case 88:  ... R52 spec for "gpt_oss K=2880"
    case 112: ... covers DSV3-GateUP K=7168
    case 128: case 172: case 224: case 256: case 296: case 448:
    case 462: case 832:
    default:  KI_HINT=0 dynamic
}
```

R52's comment on `case 88`: `// covers gpt_oss K=2880 (g.ki = 2816/32 = 88)`.
R53's comment on `case 48`: `// covers Qwen3-Down K=1536 (g.ki = 1536/32 = 48)`.

**Both are stale.** `K_STEP = 64` today (line 5: `constexpr int K_STEP = 64`).
`g.ki = fast_k / K_STEP` (line 4140). Re-deriving for current metric K
values:

| Family | K | fast_k | g.ki today | Dispatcher route today |
|--------|---|--------|------------|------------------------|
| DSV3-GateUP | 7168 | 7168 | **112** | `case 112` ✓ |
| DSV3-Down | 2048 | 2048 | **32** | NO CASE → KI=0 dynamic |
| Qwen3-GateUP | 4096 | 4096 | **64** | `case 64` ✓ |
| Qwen3-Down | 1536 | 1536 | **24** | NO CASE → KI=0 dynamic |
| gpt_oss-* | 2880 | 2816 | **44** | fuse_eligible → KI=0 fuse |

**Only 8/24 metric shapes** (DSV3-GateUP and Qwen3-GateUP families)
hit a compile-time KI spec. The other **16/24** (DSV3-Down,
Qwen3-Down, all gpt_oss) fall through to `KI_HINT=0 dynamic` (or its
fuse variant for gpt_oss). R52's `case 88` and R53's `case 48` are
dead code for metric — they would require K=88×64=5632 or
K=48×64=3072 which no metric shape uses.

This OUGHT to be a +20-30 score lift opportunity (if the missing
KI=24/32/44 specs work as cleanly as KI=64/112).

## Three KI spec attempts (all FALSIFIED)

### Attempt 1 — `INSTANTIATE_K_GRP(24)` + `INSTANTIATE_K_GRP(32)` (non-fuse)

Resource report (post-build):

```
                    SGPR  VGPR  VGPRspill  occ
L0_KI24_F0 (RCR):    83   256      20      2  ← SPILL
L0_KI32_F0 (RCR):    83   256      20      2  ← SPILL
L0_KI48_F0 (RCR):    86   256       0      2
L1_KI24_F0 (RRR):    83   256      20      2  ← SPILL
L1_KI32_F0 (RRR):    83   256      20      2  ← SPILL
L1_KI64_F0 (RRR):    88   256       0      2
L2_KI24_F0 (CRR):    84   256      16      2  ← SPILL
L2_KI32_F0 (CRR):    84   256      16      2  ← SPILL
L2_KI64_F0 (CRR):    82   256      16      2  ← SPILL
```

KI=24 / 32 spill 16-20 VGPR on **all 3 layouts** including RCR. R53
saw KI=48 spill 16-20 only on RRR/CRR (RCR-only spec was the workaround);
KI=24/32 RCR also spills, so the RCR-only workaround does NOT apply.

Counter-intuitive: smaller compile-time KI = MORE spill. KI=48
(23-iter unroll) is clean while KI=24 (11-iter unroll) spills 20.
Hypothesis: LLVM's GCN scheduler uses an aggressive software
pipelining heuristic for short fully-unrolled loops, doubling A_tile /
B_tile register lifetimes; longer loops (KI=48+) trip a different
heuristic that pipelines less aggressively.

Spill cost ≈ 20 dwords × ~50-cycle VMEM scratch latency × N reuses ≈
1000+ cycles per kernel. Compare to dynamic-K loop overhead
(~5 cycles/iter × 12 iters = 60 cycles). The compile-time spec is
**~17× WORSE** than the dynamic K it would replace.

### Attempt 2 — `<RCR, 44, true>` fuse spec (gpt_oss K=2880 RCR fwd)

Refactored `launch_one_grouped_fuse<L>` → `launch_one_grouped_fuse<L, KI = 0>`,
added dispatcher `if constexpr (L == RCR) { switch (g.ki) { case 44:
launch_one_grouped_fuse<L, 44>(g); break; default: launch_one_grouped_fuse<L, 0>(g); } }`.

Resource sweep across `#pragma unroll {full, 4, 1}`:

```
                    SGPR  VGPR  VGPRspill  ScratchSize  occ
L0_KI0_F1   (KI=0): 96    248       0          0       2  ← BASELINE
L0_KI44_F1  full:    87   256      28          ~120    2  ← SPILL
L0_KI44_F1  unroll4: 87   256      28          ~120    2  ← SPILL (same)
L0_KI44_F1  unroll1: 93   256       9           40     2  ← SPILL
```

`#pragma unroll 1` (NO unrolling at all) STILL spills 9 VGPR vs
KI=0's 0 spill at 248 VGPR. The compile-time KI=44 ALONE costs
17 VGPR vs dynamic (8 absolute increase to 256 VGPR + 9 spill).

Where? KI_HINT propagates to:
* Line 698-711: main loop bound + unroll directive (gated above).
* Line 720: `tile = (KI_HINT > 0) ? KI_HINT - 2 : num_tiles_dyn - 2`
  (epilog 1/2 starting tile — constexpr arithmetic).
* Line 861, 1000: K-tail block's `k_tail_tile = KI_HINT > 0 ? KI_HINT
  : num_tiles_dyn` (used in HBM offset compute for direct-HBM loads).

None of these should INCREASE pressure — constants typically reduce
allocation cost. The 17-VGPR delta is likely from LLVM's GCN scheduler
treating compile-time-bound K-tile counts as a software-pipelining
opportunity even at `#pragma unroll 1`, since the function-body
graph still exposes the constant.

Disassembly diff KI=0 vs KI=44 fuse @ unroll 1 would identify the
spill site precisely. Deferred to R59 due to time budget.

### Attempt 3 — Per-layout RCR-only fallback for KI=24/32

Tried gating `INSTANTIATE_K_GRP(24)` to RCR-only (mirroring R53's KI=48
workaround). RCR variant ALSO spilled 20 VGPR (build report above) —
RCR-only fallback DOES NOT salvage KI=24/32. Distinct from R53's KI=48
where RCR was clean. Spill mechanism for KI<48 is layout-independent.

## Correctness

Not run — all attempts reverted before validation. Build resource report
alone disqualified each spec on perf grounds (spill > expected dynamic-K
loop overhead).

## Outcome

* **No kernel commit.** WT reverted to R55 baseline (HK SHA
  237ca6b1bdd7e432e3d0ad97bf2082f3cb150e62) — verified clean via 5×
  paired metric runs (median 917, mean 914.8 — within R55-baseline
  noise envelope from R57).
* This doc + STALE-CASE discovery + 3-attempt failure analysis is the
  R58 deliverable.

## R59 next-action surface

Updated priorities given R58 findings:

1. **DISASSEMBLY DIAGNOSTIC** (pre-requisite for fixing KI<48 spill):
   Build KI=44 fuse @ unroll 1 with `-save-temps`, diff `.s` output
   vs KI=0 dynamic same template. Identify exact instruction(s)
   responsible for the 17-VGPR delta. Without root cause, further
   KI-spec attempts are blind. **Recommended R59.**

2. **Work-stealing persistent kernel** (R57's recommended R58, not
   relevant for today's worst shape but still a structural lever for
   the noisy-day-worst B=4 family). Atomic global tile counter,
   eliminates wave imbalance on tiles=736 (B=4 M=2048). Estimated
   2-5% lift on those 4 shapes (weight 3) under contention-low
   conditions.

3. **Per-block fixed-overhead reduction**:
   * (a) Hoist cumsum scan out of kernel via Primus-Turbo passing a
     pre-computed cum-tiles array (infrastructure work; clean).
   * (b) Reduce per-iter SRD setup / chiplet swizzle. Already tight.

4. **Prologue HBM load overlap with init phase**: kernel's outer
   scope does cumsum scan + LDS init via 1 thread (lines 3726-3742),
   then `__syncthreads()`, then per-iter SRD setup. Could the init
   thread also issue the first iter's prologue HBM `buffer_load_lds`
   ops in parallel? Saves ~50 cycles per kernel call. Marginal but
   clean to implement.

5. **DEPRIORITIZED**: KI=24/32/44 spec attempts (all spill, root
   cause unclear). Re-attempt only after disassembly diagnostic (R59
   item 1) reveals a clean fix.

## Side correction

Update R52/R53 doc commit messages to flag the K_STEP=64 stale-case:
the `case 48` and `case 88` switch arms are dead code for the current
metric. Either add deprecation comments OR remove them in a future
cleanup commit (defer; non-urgent — stale cases just get skipped at
runtime, no perf cost).
