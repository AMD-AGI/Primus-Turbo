# Round 59 — BF16 grouped, fwd KI=44 fuse spec @ #pragma unroll 1 — perf NEUTRAL

## Goal coming in

R58 attempted KI=44 fuse spec (gpt_oss K=2880, RCR fwd) and found
build-time spill (28 VGPR @ full unroll, 9 VGPR @ #pragma unroll 1)
but **never actually ran perf measurements** — disqualified each
spec on static spill cost alone. R59 plan: actually measure metric
wall to determine whether the 9-VGPR spill at unroll 1 sits in a
cold path (epilog / K-tail) and is therefore wall-invisible.

R59 starting metric (GPU 3, 1 sample): score=921, worst shape =
`gpt_oss-GateUP-B32-M2048` ratio=1.046 (HK 1087.4 / TRT 1039.0,
weight 3) — same family as R58's worst (1.034 → 1.046, contention
drift). gpt_oss family routes through fuse path → KI=44 spec
candidate is the ONE intervention point for 8 weight-3 shapes.

## Implementation (NOT committed — perf NEUTRAL)

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

1. Refactor `launch_one_grouped_fuse<L>` → `launch_one_grouped_fuse<L, KI = 0>`
   (1 line; KI defaulted so RRR fuse stays at KI=0).

2. Dispatcher fuse-eligible branch: split RCR/RRR cases. RCR switches
   on `g.ki`: `case 44 → launch_one_grouped_fuse<L, 44>(g)`; default
   stays KI=0. RRR unconditionally KI=0.

3. Add `template __global__ void grouped_kernel<Layout::RCR, 44, true>(...)`
   instantiation.

4. Gate main-loop unroll directive on FUSED+KI>0:
   `else if constexpr (FUSED_KTAIL) { #pragma unroll 1 ... }` (between
   the existing CRR `unroll 2` and RCR/RRR `unroll` branches).

## Resource report

```
                    SGPR  VGPR  VGPRspill  ScratchSize  occ
L0_KI0_F1   (KI=0): 96    248       0          0       2
L0_KI44_F1  unroll1: 93   256       9          40       2  ← R59 candidate
```

8 VGPR allocated + 9 spill = 17 VGPR delta vs KI=0 dynamic. R58 doc
suspects spill source is LLVM's GCN scheduler treating compile-time
KI as a software-pipelining opportunity even at unroll 1, since the
function-body graph still exposes the constant.

## Correctness

`/tmp/probe_r55_correctness.py` — 5 representative shapes (incl.
fuse-path `gpt_oss-Down-B4-M2048`): 47.83-47.86 dB SNR, all
allclose=True (bf16-rounding floor).

## Metric (5×R59 paired with 5×baseline, GPU 3, single shell session)

| batch        | runs                              | median | mean   |
|--------------|-----------------------------------|-------:|-------:|
| R59          | 912, 901, 896, 898, 882           |  898   | 897.8  |
| Baseline     | 917, 904, 884, 889, 895           |  895   | 897.8  |
| **Δ**        |                                   | **+3** | **0**  |

Pair-by-pair (interleaved):
−5, −3, +12, +9, −13 → 2 positive / 3 negative; mean diff = 0,
median diff = −1 (sign of pair median).

**NEUTRAL.** The compile-time KI=44 specialization gain (constexpr
loop bound + epilog tile / K-tail HBM offset constant propagation)
exactly offsets the 9-VGPR spill cost. Net wall impact = 0.

## Why neutral (not negative as R58 suspected)

The 9-VGPR spill DOES sit partly in a cold path. Likely sites:

* K-tail block (lines 856-926) — runs once per persistent iter.
  ~3 iters/block on B=32 M=2048 = 3 invocations per CU. Spill/reload
  amortized across 3 iters of MMA work (~3000 cycles each) =
  negligible.
* Epilog 1/2 — runs once per persistent iter. Same amortization.
* Main loop body — would be hot (44 K-tile passes per iter), but
  unroll 1 keeps the unrolled body small enough that the spill is
  apparently OUTSIDE the inner loop.

The compile-time KI gain offsets exactly enough to net out:
constexpr `KI - 2 = 42` for epilog tile arithmetic + constexpr
`k_tail_tile = 44` saves ~5 cycles per K-tail invocation, mirrored
for ~3 invocations/CU = ~15 cycles saved per CU per kernel call,
matching the spill cost amortized.

## Decision: REVERT (no commit)

Net wall = 0 → not worth the maintenance cost of a custom KI spec
that doesn't generalize (KI=44 is gpt_oss-specific; future K values
would need separate analysis). Drop the 4-file change set and stay
on the simpler dynamic-K fuse template.

## R60 next-action surface (recap + new)

1. **DISASSEMBLY DIAGNOSTIC** (R58/R59 deferred): build with
   `-save-temps` and diff `.s` for KI=44 fuse @ unroll 1 vs KI=0
   fuse same template. Identify spill site precisely. If spill is
   eliminable (e.g., via tighter register hint or restructuring of
   k_tail_tile compute), KI=44 fuse spec becomes a clean +0.5-3.8 %
   per-shape lift on gpt_oss family (8 shapes, weight 3).

2. **Work-stealing persistent kernel** — R57 recommended; addresses
   B=4 wave imbalance (2.88 iters/CU, MFMA 50%). Atomic global tile
   counter, eliminates 32-CU 2-iter-vs-3-iter split. Estimated 2-5%
   per-shape lift on B=4 family (4 shapes, weight 3) under
   contention-low conditions. Highest-confidence structural lever.

3. **Per-block fixed-overhead reduction**:
   * Hoist cumsum scan via Primus-Turbo passing a pre-computed
     cum-tiles array (infrastructure; clean but cross-repo).
   * Inline asm-tighten chiplet swizzle / SRD setup (already tight).

4. **Prologue HBM-load overlap with init phase** — issue first iter's
   `buffer_load_lds` ops in parallel with cumsum scan. ~50 cycles
   saved per kernel call. Marginal but local.

5. **DEPRIORITIZED**: KI=24/32 non-fuse spec retry (R58 falsified;
   layout-independent spill of 16-20 VGPR per RCR/RRR/CRR).

## Outcome

* No kernel commit. WT reverted to R55 baseline (HK SHA
  237ca6b1bdd7e432e3d0ad97bf2082f3cb150e62).
* This doc + R58 KI=44 fuse spill perf measurement (NEUTRAL not
  NEGATIVE) is the R59 deliverable.
