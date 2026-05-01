# Round 20 (death-march) — FP8-vs-BF16 grouped spill divergence is the load-bearing finding; readfirstlane on lds_tile_base is a NOOP

## Round 19 plan revisited

Round-19 doc promoted "template-specialize two-tile" to #1 priority and
"audit kernel code-size to reduce 67-VGPR-spill baseline" to #2. This
round explored a focused #2 lever: add `__builtin_amdgcn_readfirstlane`
to `lds_tile_base` inside `rcr_8w_load_hoist` to force SGPR-resident
LDS slot base (mirror BF16 `grouped_kernel:3763` pattern).

Before committing to either #1 or #2, this round also collected the
build resource summary across **all** kernel families (FP8 dense vs FP8
grouped vs BF16 grouped) — and that diagnostic is what should drive
round 21+.

## Key diagnostic — spill divergence by kernel family

Single `make -j` build, all template specs:

| Kernel | Spec | VGPR Spill | Scratch bytes/lane |
|---|---|---|---|
| FP8 `gemm_kernel<RCR>` (dense, has BOTH single + two-tile) | KI=0 | **12** | **48** |
| FP8 `gemm_kernel<RRR>` | KI=0 | 8 | 36 |
| FP8 `gemm_kernel<CRR>` | KI=0 | 24 | 100 |
| **FP8 `grouped_rcr_kernel`** | <0,0,0> (vanilla) | **67** | **272** |
| FP8 `grouped_rcr_kernel` | <0,1,0> (N_MASKED) | 76 | 308 |
| FP8 `grouped_rcr_kernel` | <0,0,1> (FUSED_KTAIL) | **48** | **196** |
| FP8 `grouped_rcr_kernel` | <0,1,1> (both) | 58 | 236 |
| **BF16 `grouped_kernel`** (RCR, KI=0, FUSED=false) | — | **0** | **0** |
| BF16 `grouped_kernel` (RCR, KI=0, FUSED=true) | — | 0 | 0 |
| BF16 `grouped_kernel` (RCR, KI=56) | — | 14 | 60 |

**Three observations of high signal:**

1. **FP8 dense kernel spills 12 VGPR** despite containing BOTH single-tile
   AND two-tile main-loop bodies. So the two-tile body itself is NOT
   inherently spill-causing. (Reaffirms round-19's spill-cliff finding:
   adding the body to grouped via runtime branch was -37%, but dense
   has both bodies and only 12 spill.)

2. **BF16 grouped spills 0 VGPR** with the same persistent-loop
   bookkeeping (LDS group metadata cache, branch-free 6-step binary
   search, group-by-M/N swizzle, per-tile accumulator zero, per-tile
   epilog). So the persistent-loop pattern is NOT inherently
   spill-causing either.

3. **FP8 grouped <0,0,0> spills 67 VGPR**. This is the spec used by all
   8 DSV3 metric shapes (K=2048 / N=4096-7168 fully aligned). The 67
   spill / 272 scratch is the rocprof-localized FLAT bottleneck from
   round-17.

The conclusion is that the 67-spill is **specifically caused by FP8
grouped's combination** of:
* persistent-loop bookkeeping (which BF16 grouped also has, with 0
  spill)
* FP8-specific load helper `rcr_8w_load_hoist` (12 calls per K-iter, vs
  BF16's `device_gemm_tile_body` shared lambda calling `G::load`)
* FP8 register tile sizes (`rt_fp8e4m3<RBM=64, BK=128, row_l, rt_16x128_s>`
  for A_row_reg = 32 VGPR/thread, vs BF16's `rt_bf<HALF_REG_BLOCK_M=64,
  K_STEP=64, row_l, rt_16x32_s>` = ~16 VGPR/thread)

Counter-intuitive corollary: **FUSED_KTAIL=true reduces grouped spill
from 67 → 48** (DSV3 spec → gpt_oss spec). Adding the `raw_buffer_load_b128`
direct-to-VGPR K-tail block changes the compiler's register allocation
heuristic in a way that frees registers in the main loop. This contradicts
round-19's two-tile finding (where adding code increased spill by 14)
and suggests **the spill-cliff is highly sensitive to WHAT code is
added**, not just code volume.

## This round's intervention — `readfirstlane` on lds_tile_base

`rcr_8w_load_hoist` already calls `__builtin_amdgcn_readfirstlane` on
the per-pass `off32` (line 522). But `lds_tile_base = reinterpret_cast<
uintptr_t>(&dst.data[0])` (line 507-508) does NOT — it relies on the
compiler to recognize the wave-uniform property and propagate SGPR
allocation through the `lds_tile_base + warp_linear_offset +
lds_subtile_id * subtile_padding` chain.

BF16 `grouped_kernel` (kernel_bf16_dynamic.cpp:3763) explicitly forces
SGPR via:

```c++
uint32_t a_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
    reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
```

Hypothesis: forcing `lds_tile_base` to SGPR via the same readfirstlane
pattern in the FP8 helper may help the compiler keep the per-call lds
addresses fully SGPR-resident, freeing VGPRs for other uses.

Implementation (1 line change in `rcr_8w_load_hoist`):

```c++
const uintptr_t lds_tile_base = static_cast<uintptr_t>(
    __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&dst.data[0]))));
```

## Result — falsified, fully NOOP

Build resource:

| Spec | Before | After |
|---|---|---|
| grouped_rcr_kernel<0,0,0> spill | 67 | **67** |
| grouped_rcr_kernel<0,1,0> spill | 76 | **76** |
| grouped_rcr_kernel<0,0,1> spill | 48 | **48** |
| grouped_rcr_kernel<0,1,1> spill | 58 | **58** |
| gemm_kernel<RCR> spill (dense) | 12 | **12** |
| All ScratchSize values | unchanged | unchanged |

Wall-time after revert: 819 / 830 / 818 over 3 runs (mean 822 ± 5),
indistinguishable from baseline 824.

The compiler was ALREADY hoisting the LDS base into SGPR, exactly as
the round-18 SRD hoist falsification predicted: when a wave-uniform
value is computed inside a `__forceinline` helper and used immediately
in inline-asm `s` constraints (which the per-pass `off32` is), the
compiler recognizes the dependency chain and propagates SGPR class.
Explicit `readfirstlane` is a no-op in this case.

## Implication for round 21+

The 67-spill is REAL (rocprof'd as 2.99M FLAT instructions, 2.9× Triton)
but is NOT addressable via:
1. ✗ Hoisting SRD construction (round 18)
2. ✗ Hoisting LDS slot base (this round)
3. ✗ Adding two-tile body via runtime branch (round 19)

What HAS NOT been tested yet:

**(A) Port BF16's `device_gemm_tile_body` pattern to FP8 grouped.**
BF16 grouped uses a shared lambda/template helper that takes
precomputed SGPR-resident LDS offsets (`a_lds_00..11`, `b_lds_00..11`)
as parameters. The helper is `__forceinline` but its parameter contract
gives the compiler stronger register-class hints than FP8's per-call
`reinterpret_cast<uintptr_t>(&dst.data[0])` pattern. Multi-round port,
high risk (mirrors round-19 code-size cliff if not done as REPLACE).

**(B) Convert FP8 grouped main-loop loads from LDS-staged to direct
buffer_load_b128 → VGPR (FUSED_KTAIL pattern).** The FUSED_KTAIL=true
spec has 19-VGPR-LESS spill than =false (48 vs 67). Replicating its
direct-HBM-to-VGPR load pattern in the main loop may transfer that
spill reduction to all specs. Risk: gives up LDS prefetch pipelining,
might lose more wall-time than it saves in spill.

**(C) Reduce FP8 register tile widths.** `A_row_reg =
rt_fp8e4m3<RBM=64, BK=128, row_l, rt_16x128_s>` = 32 VGPR/thread.
Halving BK to 64 (matching BF16's K_STEP=64) would halve A_row_reg's
per-thread VGPR count to 16, freeing 16 VGPRs and shrinking the per-
K-iter footprint. But this requires changing the entire main-loop
schedule (each MFMA covers half the K depth → 2x more iterations) and
the LDS layout. Multi-round refactor, high risk.

**(D) Move FUSED_KTAIL block into a `__device__ __noinline__` helper.**
Currently it's inlined into grouped_rcr_kernel. Splitting into a
non-inline call would isolate its register pressure from the main
loop. But would add function-call overhead per K-tail iteration. Worth
a single-round probe.

## Falsified levers (cumulative, for next round's prompt)

1. BF16 K_STEP=64→32 port (round 26) — out of scope.
2. FP8 chunk_size sweep (round 22) — dead-end.
3. FP8 MFMA cell-shape migration (rounds 14-15) — falsified by rocprof.
4. FP8 main-loop barrier/setprio/vmcnt micro-knobs (rounds 13-16) —
   saturated.
5. FP8 grouped SRD hoist for SALU reduction (round 18) — compiler CSEs.
6. FP8 grouped two-tile main-loop port via runtime branch (round 19) —
   spill cliff -37%.
7. **FP8 grouped `readfirstlane` on `lds_tile_base` (THIS ROUND, round
   20)** — compiler already hoists. Confirms round-18 lesson:
   wave-uniform values inside `__forceinline` helpers used immediately
   in inline-asm `s` constraints are auto-promoted to SGPR; explicit
   readfirstlane is a no-op for this pattern.

## Status

**Reverted**. Doc-only commit in Primus-Turbo. No HipKittens commit
this round. Score after revert: 822 ± 5 (within baseline noise).

Round-20 metric SHA: this commit (Primus-Turbo).
HK SHA: unchanged.

## Round 21 starting move

Recommend **option D** (FUSED_KTAIL `__noinline__` probe) as a
short-circuit experiment FIRST: it tests whether the FUSED_KTAIL
block's register-allocation interaction with the main loop is what
caused the surprising spill REDUCTION (67 → 48) when FUSED_KTAIL=true.
If `__noinline__` makes the FUSED_KTAIL=true spec spill MORE (toward
67), it confirms the inline interaction is responsible — and the
inverse (force FUSED_KTAIL block style into the main loop) is the
direction. If `__noinline__` doesn't change anything, the spec
divergence is from different LLVM heuristic states (compile-time
template parameter affects optimization flags), and a different lever
is needed (try option B or C).
