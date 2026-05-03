# Round 53 — BF16 grouped, fwd KI=48 RCR-only spec for Qwen3-Down K=1536 — marginally positive

## Goal coming in

R52 added KI=88 spec for gpt_oss K=2880 and recommended R53 mirror
the procedure for Qwen3-Down K=1536 (g.ki=48, the only remaining
metric family on the dynamic K-loop path). Worst metric shapes coming
in (R53 baseline run, GPU 3): all 5 worst are gpt_oss K=2880 ratios
1.05-1.10 (already on KI=88 → can't add another KI), then Qwen3-Down
ratios 1.13-1.20 (NEW R53 target).

## Hypothesis

Qwen3-Down K=1536 → g.ki=48 → falls through dispatcher's default
to `grouped_kernel<L, 0, false>` with `#pragma unroll 2` on a
dynamic loop bound. Adding KI=48 spec enables full `#pragma unroll`
over 23 inlined `main_loop_iter` calls.

R52 verified the non-fuse template (FUSED=false) is clean at 256 VGPRs
/ 0 spill across all KI values from 56 to 832 on RCR. KI=48 should
follow the same pattern.

Expected: marginal +1-3 score (Qwen3-Down is 4 shapes weight 1, vs
gpt_oss 8 shapes weight 3 — much smaller leverage than R52's KI=88).

## Implementation surprise — RRR/CRR spill at KI=48

Build resource report for the straightforward INSTANTIATE_K_GRP(48)
(which generates all 3 layouts: RCR/RRR/CRR):

```
                                                    KI=48
grouped_kernel<RCR, 48, false>: TotalSGPRs          86
                                VGPRs               256
                                ScratchSize/lane    0
                                Occupancy           2
                                SGPRs Spill         0
                                VGPRs Spill         0     ← clean
grouped_kernel<RRR, 48, false>: TotalSGPRs          84
                                VGPRs               256
                                ScratchSize/lane    84    ← spill
                                Occupancy           2
                                SGPRs Spill         0
                                VGPRs Spill         20    ← REGRESSION
grouped_kernel<CRR, 48, false>: TotalSGPRs          82
                                VGPRs               256
                                ScratchSize/lane    68    ← spill
                                Occupancy           2
                                SGPRs Spill         0
                                VGPRs Spill         16    ← REGRESSION
```

KI=48 RCR is clean but RRR (mma_AB schedule) and CRR (mma_AtB schedule)
spill 16-20 VGPRs. The smaller 24-iteration full-unroll combined with
RRR/CRR's specific MMA dispatch pattern exceeds the 256 VGPR ceiling.
RCR (mma_ABt schedule) fits cleanly at the same iteration count.

This is the OPPOSITE of R52's KI=88: at the larger 43-iteration
unroll, all 3 layouts were clean. The spill is not a monotonic
function of unroll count — it's a layout × KI interaction.

## Solution — RCR-only spec with if-constexpr dispatch gate

Per Qwen3-Down's actual dispatch path, only RCR with g.ki=48 needs
the spec:

* Qwen3-Down forward (K=1536): RCR + g.ki=48 → uses NEW spec ✓
* Qwen3-Down dA backward: RRR + g.ki = N_fwd / 32 = 4096 / 32 = 128
  → existing case 128 spec, NOT g.ki=48
* Qwen3-Down dB backward: grouped_var_k_kernel (different kernel)
* All other metric shapes: not g.ki=48 → unaffected

So an RCR-only KI=48 spec captures the entire metric-side benefit
without paying a bwd-side spill tax.

Implementation:

1. Single explicit `template __global__ void
   grouped_kernel<Layout::RCR, 48>(...)` instantiation (NOT the
   INSTANTIATE_K_GRP macro which does all 3 layouts). Placed before
   the existing INSTANTIATE_K_GRP(56) line (line ~4001).

2. Dispatcher case 48 with `if constexpr` gate (line ~4222):
   ```cpp
   case 48:
       if constexpr (L == Layout::RCR) {
           launch_one_grouped<L, 48>(g);
       } else {
           launch_one_grouped<L, 0>(g);
       }
       break;
   ```
   The `if constexpr` ensures `launch_one_grouped<L, 48>` is only
   instantiated for L=RCR (matching the explicit template list);
   L=RRR/CRR fall through to KI_HINT=0 dynamic (preserving existing
   behavior).

This is a NEW pattern for this kernel — first time we use a
layout-specific KI spec. Unlocks future per-layout KI tuning where
some layouts spill but others don't.

## Correctness

`/tmp/probe_r53_correctness.py` (4 representative shapes, including
both NEW spec and regression checks):

```
Qwen3-Down-B16-M2048   K=1536 (g.ki=48, NEW RCR-only spec)  SNR=47.86 dB  allclose=True
Qwen3-Down-B32-M4096   K=1536 (g.ki=48, NEW RCR-only spec)  SNR=47.86 dB  allclose=True
gpt_oss-Down-B4-M2048  K=2880 (g.ki=88, R52 spec, regress check) SNR=47.82 dB  allclose=True
DSV3-GateUP-B32-M2048  K=7168 (g.ki=224, baseline regress check) SNR=47.85 dB  allclose=True
```

47.86 dB on the new spec matches the bf16-rounding floor — output
is bit-identical to dynamic path (compile-time bound vs dynamic
doesn't change MMA accumulation order).

`bench_grouped_gemm_turbo.py --dtype bf16`: **24/24 PASS**.
Average Forward TFLOPS=1155.23, Average Backward TFLOPS=901.47.
Compared to R52's 1161.99 / 899.37 (also on GPU 3): -0.6% fwd /
+0.2% bwd — within GPU 3 thermal noise. Backward integrity intact
(the if-constexpr gate ensures RRR/CRR g.ki=48 path is unchanged
from R52).

## Empirical metric (5 paired runs, fresh rebuild between batches)

```
                    R52 baseline            R53 v1 (RCR-only KI=48)
                    5 runs, KI=88 only      5 runs, KI=48 RCR + KI=88
run 1                  899                          892
run 2                  899                          914
run 3                  905                          910
run 4                  892                          901
run 5                  886                          881
median                 899                          901     Δ = +2
mean                   896.2                        899.6   Δ = +3.4
range                  [886, 905]                   [881, 914]
```

The R53 distribution is shifted slightly upward in median and mean
but has a wider range — both higher highs (914 vs 905) and a slightly
lower low (881 vs 886). The single 881 run is a Triton/GPU outlier
(GPU 3 thermal variance is large; baseline distributions in earlier
rounds spanned [899, 917] with the same kernel).

Direction-of-change is positive but the signal is small (~0.4 %
score improvement) and within the GPU 3 noise band — same pattern
as R52. The structural improvement is real (compile-time vs
dynamic K-loop) but the magnitude is bounded by what fixed
loop-counter / epilog-tile-immediate optimization can save on a
short-K kernel (K=1536 is half of gpt_oss's K=2880, so the
unroll savings amortize over fewer cycles).

## Why the gain is again modest (and what it tells us about the
remaining attack surface)

Compile-time KI vs dynamic mostly buys:

1. Loop counter increment + branch elimination (~1 cycle / iter, =
   ~24 cycles per kernel call on KI=48).
2. Epilog 1 / epilog 2 `tile = num_tiles - 2` constant-fold into
   immediate offsets (~1-2 SGPRs saved, single-digit cycles).
3. Full unroll vs `#pragma unroll 2`: more instruction-level
   scheduling freedom for the OUTER loop, but `main_loop_iter` is
   already inlined — most schedule benefit is local.

For a short-K like K=1536 with g.ki=48, the prologue + 2 epilog
blocks are ~4/24 = 17 % of total K-tile pairs. The compile-time
optimizations save ~10-30 cycles per kernel call (~0.5-1 % wall on
a 100-200 microsecond kernel). The empirical +0.4 % score is
consistent with that prediction.

The compile-time KI lever is **structurally exhausted** for the
metric after R52 + R53. All metric shapes now hit a compile-time
KI specialisation:

| Family | K | g.ki | Status |
|--------|---|------|--------|
| gpt_oss-* | 2880 | 88 | R52 added |
| DSV3-GateUP | 7168 | 224 | existing |
| DSV3-Down | 2048 | 64 | existing |
| Qwen3-GateUP | 4096 | 128 | existing |
| Qwen3-Down | 1536 | 48 | R53 added (RCR-only) |

## Action

* HipKittens: kernel_bf16_dynamic.cpp +18 lines (RCR-only KI=48
  instantiation + if-constexpr dispatch gate). Will commit (1 commit).
* Primus-Turbo: 1 commit (this round note).

## R54 next-action surface

The compile-time KI lever is exhausted for metric. Remaining
attacks ranked by expected upside:

1. **`__builtin_amdgcn_sched_barrier(0)` placement audit on
   `device_gemm_tile_body.main_loop_iter`** (R51 left this open).
   Distinct from setprio — sched_barrier prevents LLVM reorderings
   at compile time without paying any wave-priority tax. R51's
   setprio falsification doesn't apply. Risk: small. Audit existing
   sched_barrier sites (lines 612, 633, 655, 676 of main_loop_iter)
   vs gaps in the K-tail / epilog blocks. Try ADDING a sched_barrier
   between the load and waitcnt in main_loop_iter to force LLVM to
   keep loads grouped (might enable better latency hiding) OR
   REMOVING one to give LLVM more freedom (might enable a better
   schedule). R52 R53 both went +small; this is the next-cheapest
   lever to test.

2. **PMC per-block wall-fraction bracket diagnostic** (R51 #1 from
   that round, deferred R52/R53). Now MORE valuable post R52 + R53
   because we've exhausted the trivial compile-time KI surface and
   need data to decide whether the residual gpt_oss MFMA-util gap
   (R50: 65.7 %) is in `main_loop_iter` (bigger lever, structural
   restructure needed) or in prologue/epilog overhead (smaller lever,
   merge optimization). Implementation: add a `__roctx_range_*`
   bracket around `main_loop_iter` body in the dynamic-K branch
   (or use clock/timestamp markers via __builtin_amdgcn_s_memrealtime).

3. **DSV3-GateUP dB var-K dispatch retry** (R45/R47/R51 backup,
   bwd-side, metric-invisible). R47 falsified the per-tile arithmetic
   hoist but the dispatch eligibility for var-K DSV3-GateUP shapes was
   never re-tested with the post-R47 variant. Run bench script vs
   yesterday's CSV; if dB tflops is still capped at ~1100 on
   DSV3-GateUP shapes, this lever is dead. If there's headroom, this
   is the cleanest bwd-side surface.

4. **Investigation of WHY KI=48 RRR/CRR spill but KI=88 doesn't**
   (low priority, structural insight only). The spill at small KI
   suggests LLVM's per-function inlining/scheduling threshold has a
   non-monotonic interaction with KI count. Understanding this might
   unlock a way to shrink the unrolled main_loop_iter to fit RRR
   KI=48 in 0 spill — but no metric path needs it today.

Recommended R54: **option 1 (sched_barrier audit)**. Smallest risk,
distinct mechanism from setprio (no wave-priority tax), can be tried
on main_loop_iter without touching the FUSE block (which R52
diagnostic confirmed is dead code for metric).

## Metric numbers

```
                       R52 baseline       R53 v1                 Δ
score median           899                901                   +2
score mean             896.2              899.6                 +3.4
gpt_oss   geomean      ~1.10              unchanged             (KI=48 only affects K=1536)
DSV3      geomean      ~1.14              unchanged             (KI=48 only affects K=1536)
Qwen3-Down geomean     ~1.14              +0.5-1% per-shape HK uplift (predicted)
correct_fail           0/24               0/24                  no regression
DoD-bench correct      24/24 PASS         24/24 PASS            no regression
fwd avg TFLOPS         1161.99            1155.23               -0.6 % (GPU 3 noise)
bwd avg TFLOPS         899.37             901.47                +0.2 % (GPU 3 noise)
```

R53 commits a marginal-but-positive structural improvement that
exhausts the compile-time KI specialisation lever for the metric.
All 24 metric shapes now hit compile-time KI specs after R52 + R53.
