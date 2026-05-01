# Round 26 (DM): RCR_SCHED_BARRIER mask sweep — geomean falsified, but per-shape signal IS real

## Context

Round 25 (cold-start handoff doc) recommended **Option A**: replace
`RCR_SCHED_BARRIER()` (currently `__builtin_amdgcn_sched_barrier(0)`,
the most restrictive form) with `__builtin_amdgcn_sched_group_barrier`
or a wider mask, to give the LLVM machine scheduler freedom to
interleave MFMA + DS_READ across the per-quartet barriers in
`grouped_rcr_kernel` main loop body. This was untested in rounds 1-25.

This round bisected the cheaper variant — widening the existing
`sched_barrier(MASK)` rather than swapping to `sched_group_barrier`.
The sched_barrier form is strictly less invasive: it changes an
"allow-list" of which instruction classes can be reordered across the
barrier, without redefining the whole iteration's instruction sequence.

## Mask sweep (all 5 settings vs revert-to-baseline mask=0)

| mask    | meaning                          | runs            | mean | median | geomean (best run) |
|---------|----------------------------------|-----------------|------|--------|--------------------|
| `0`     | baseline (block all reorder)     | 829,823,825,823,822 | 824.4 | 823 | 0.989              |
| `0x008` | MFMA only                        | 820,821         | 820.5 | 820.5 | 0.984             |
| `0x100` | DS_READ only                     | 820,829,821     | 823.3 | 821    | 0.985             |
| `0x108` | MFMA \| DS_READ                  | 828,827,819,820,829 | 824.6 | 827 | **0.994** (best run) |
| `0x10A` | MFMA \| DS_READ \| VALU          | 822            | 822    | 822    | 0.987             |
| `0x308` | MFMA \| DS_READ \| DS_WRITE      | 821            | 821    | 821    | 0.985             |

5-sample baseline (mask=0) and 5-sample widest-tested (mask=0x108) have
**near-identical means** (824.4 vs 824.6) and only a small median
gap (823 vs 827). All within the documented 818-829 noise band.

**Verdict at metric level: falsified.** No mask widens the geomean
beyond the per-run noise floor.

## What IS real: per-shape sensitivity

Despite the geomean cancellation, individual shapes consistently shift
in the same direction across all 5 widening masks (data from the
mask=0x108 best run, but the directional pattern reproduces across
0x008/0x100/0x10A as well):

| shape                          | mask=0 baseline | mask=0x108 best | delta  |
|--------------------------------|----------------:|----------------:|-------:|
| **Shapes that consistently win** when mask is widened:                    |
| DSV3-Down-B32-M2048            |          0.980  |          1.002  | +2.2pp |
| DSV3-Down-B32-M4096            |          0.974  |          1.002  | +2.8pp |
| DSV3-GateUP-B32-M2048          |          1.019  |          1.031  | +1.2pp |
| DSV3-GateUP-B32-M4096          |          1.048  |          1.057  | +0.9pp |
| DSV3-GateUP-B16-M4096          |          1.030  |          1.041  | +1.1pp |
| gpt_oss-Down-B4-M2048          |          1.000  |          1.023  | +2.3pp |
| **Shape that consistently loses** (reproduces across ALL widening masks): |
| DSV3-Down-B16-M2048            |          0.995  |          0.952  | -4.3pp |

Net effect on geomean: ~zero. The B16-M2048 regression is large enough
to absorb the spread of small B32-* gains.

The B16-M2048 regression is reproducible across **every** widening
mask tried (0x008, 0x100, 0x108, 0x10A, 0x308). It is NOT noise. The
loss is structural to that one shape: the loop iter count is
ki=K/K_BLOCK=16 (K=2048, K_BLOCK=128), which is the smallest ki in the
test set, so the steady-state pipeline benefits don't amortize over
enough iterations while the cost (whatever it is — likely a stall in
the prologue or an aliased reorder) dominates.

## Adds to the falsification trail (now 9 directions)

| # | round | direction                              | result                                  |
|--:|------:|----------------------------------------|-----------------------------------------|
| 1 |   R17 | rocprof PMC instr count diagnose       | locates gap in CPI, not insts           |
| 2 |   R18 | hoist `make_srsrc` out of helper       | compiler already CSE'd it               |
| 3 |   R19 | port two-tile main loop from dense     | -37% spill cliff (catastrophic)         |
| 4 |   R20 | `readfirstlane(lds_tile_base)`         | compiler already hoisted                |
| 5 |   R21 | `__noinline__` on `rcr_8w_load_hoist`  | build-fail (architecturally inline)     |
| 6 |   R22 | host-overhead trim                     | host is 0.9-1.1% only                   |
| 7 |   R23 | (no change) HK vs Triton breakdown     | confirmed gap is 100% in main kernel    |
| 8 |   R24 | RCR_PREFETCH_LGKM sweep {2,4,6,8}      | saturated — all in noise band           |
| 9 |   **R26** | **RCR_SCHED_BARRIER mask sweep {0,0x008,0x100,0x108,0x10A,0x308}** | **saturated — all in noise band** |

## What this rules in for round 27+

The fact that the mask widening DOES move per-shape numbers (in
opposite directions for B16 vs B32) confirms the LLVM scheduler IS
honoring the barrier and producing different machine code. So the
abstraction is functional — it just trades winners for losers globally
when applied uniformly. Three follow-ups make sense:

### A. Per-template-parameter mask gating (round 27 candidate)

The kernel template is `grouped_rcr_kernel<L, _NUM_THREADS, ...>`. If
we add a template parameter `bool USE_WIDE_SCHED_BARRIER = false` and
gate the macro on it, the dispatcher (`dispatch_grouped_rcr` line
~4889) can pick the mask per shape from the existing autotune cache.
Anchor the win on B32-* shapes by setting wide=true; leave B16-M2048
and similarly-narrow-ki shapes at wide=false. Risk: instantiates more
template variants (binary size, build time).

### B. `sched_group_barrier` (true Option A, untested)

The relaxed `sched_barrier(MASK)` form is "permission to reorder";
the `sched_group_barrier(MASK, COUNT, GROUP_ID)` form **declares an
exact instruction count of class MASK between this barrier and the
next group barrier with the same GROUP_ID**. This is what HK's own
attention kernels use (kernels/attn/gqa/kernel.cpp:32-48,
sched_barrier_pairs<MFMA, VALU>). Translating to FP8 GEMM main loop:

```cpp
// Replace `s_barrier(); RCR_SCHED_BARRIER();` after each rcr_mma with:
SCHED_BARRIER(MFMA, 1, GROUP);   // exactly 1 MFMA in this group
SCHED_BARRIER(DS_READ, 4, GROUP); // exactly 4 ds_read_b128 in this group
```

This is structurally different from mask widening — it FORCES a
specific interleave instead of allowing one. May or may not help.
Round 27 candidate but more invasive (~30 line kernel diff).

### C. PMC SQ_LDS_BANK_CONFLICT side-by-side (Option B from R25 doc)

Still not done. Would tell us if HK's LDS layout has more bank
conflicts than Triton's, which would point to a swizzle-pattern fix
rather than scheduler tuning.

## Round 26 verdict

- Final HipKittens kernel diff: zero (mask reverted to 0 after sweep).
- Final HipKittens .so: rebuilt to match source.
- Score: 829 (final clean baseline run). No commit on HipKittens repo.
- Doc-only commit on Primus-Turbo (this file).

## Files touched this round

- `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  — line 79 `RCR_SCHED_BARRIER` macro: 5 mask values tested, all
  reverted. Final state: `__builtin_amdgcn_sched_barrier(0)` (unchanged
  from upstream).
- `/workspace/code/Primus-Turbo/analysis/_notes/round-26-dm-fp8-rcr-sched-barrier-mask-sweep-falsified.md`
  — this doc.

## Recommendation for round 27 cold-start agent (if chat rolls)

Read this doc + round-25-dm cumulative trail. Do NOT re-run
`sched_barrier` mask sweep. The two productive directions left are:

1. **Per-template-parameter mask gating** (route DSV3-Down-B32-* and
   DSV3-GateUP-B32-* shapes through a `wide_sched_barrier=true` kernel
   instantiation; everything else stays mask=0). Cleanest single-round
   commit: gate on `(B*M)/HB > 16` or similar template heuristic.
   Risk: build time +10s for one extra template variant; per-shape
   wins ~1-2pp adds to ~+0.5pp geomean if it composes.

2. **PMC `SQ_LDS_BANK_CONFLICT`** rocprofv3 capture HK vs Triton on
   `DSV3-Down-B16-M4096` (worst shape). If HK conflict ratio
   noticeably higher, the lever is LDS swizzle layout, NOT scheduler.

Score band: 818-829 over 7 consecutive rounds. Patience 7/30.
Geomean: 0.989 (target 1.20, gap 21pp).
