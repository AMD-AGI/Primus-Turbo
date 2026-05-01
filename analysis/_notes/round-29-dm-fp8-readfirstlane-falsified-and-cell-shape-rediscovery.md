# Round-29-dm — FP8 grouped: `readfirstlane` on binary-search LDS reads REGRESSED, plus rediscovery of cell-shape falsification

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `8df19707` (round-28-dm doc-only commit)
**HipKittens HEAD**: unchanged (probe edits made + reverted)
**Primus-Turbo HEAD after**: this commit (Primus-Turbo notes-only)

**Metric**: 858 → 854 (within ±5 noise band; identical kernel bytes after revert)

---

## TL;DR

Two findings this round:

1. **NEW falsification**: explicit `__builtin_amdgcn_readfirstlane` wrapping
   the wave-uniform LDS reads in `grouped_rcr_kernel`'s persistent-loop
   binary search (lines 2076, 2080, 2082, 2083) **REGRESSED** the
   `<0,F,T>` and `<0,T,T>` template specs by **+17 to +20 spill slots
   each** (DSV3 K-misaligned: 48→68 spills, gpt_oss N+K-misaligned:
   58→75 spills). The `<0,F,F>` and `<0,T,F>` specs were unaffected
   (+1 SGPR, identical spill count). Reverted.

2. **Critical rediscovery**: round-1 (= round-28-dm) of this 60-round
   chat planned to scaffold the MFMA cell-shape migration
   (`16x16x128 → 32x32x64`) as the next high-leverage lever. **This
   project was already FALSIFIED in round-15-dm** (commit `7b08432a`,
   `analysis/_notes/round-15-dm-fp8-rocprof-falsifies-mfma-cell-shape-migration.md`).
   Round-15-dm rocprof PMC on the worst FP8 shapes shows HK at MfmaUtil
   parity-or-better with Triton (DSV3-Down +0.47 pp, gpt_oss-Down
   +0.20 pp); migrating cell shapes cannot lift pipe utilization above
   what's already achieved. Round-28-dm's plan was based on stale
   round-13/14 thinking; this round redirects the plan.

---

## Round 29 hypothesis (refuted)

The 67 VGPR spills in `grouped_rcr_kernel<0,F,F>` come from somewhere —
round-28-dm clarified they are NOT MFMA accumulators (those are in
AGPRs `a0..a255` per ISA inspection) but rather "persistent non-MFMA
state" carried across the main-loop iterations.

The persistent-loop prologue computes wave-uniform tile-coord state
from LDS reads (`s_cum_tiles[mid]`, `s_offs[group_idx]`) that are
identical for every lane in a wave (same address, broadcast read).
Hypothesis: LLVM may pessimistically place these LDS-load results in
VGPRs (since `ds_read_b32` writes to a VGPR by default), inflating
the live VGPR set carried into the main loop body. Forcing SGPR
allocation via explicit `__builtin_amdgcn_readfirstlane(...)` should
free 7–8 VGPR slots — exactly the margin needed to drop the 67-slot
spill cliff.

## Experiment — patch applied

In `kernel_fp8_layouts.cpp` (~lines 2003-2084 of the `grouped_rcr_kernel`,
not the RRR or var-K twins):

```cpp
// pid (line 2003) — wave-uniform after chiplet_transform_chunked
int pid = __builtin_amdgcn_readfirstlane(chiplet_transform_chunked(
    blockIdx.x, NUM_CUS, xcds_eff, 64));

// Inside binary search (line 2076) — wrap each LDS read result
const int sval = __builtin_amdgcn_readfirstlane(s_cum_tiles[mid]);
if (gt >= sval) lo = mid;

// Post-search prologue state (lines 2080-2083)
const int tile_start = __builtin_amdgcn_readfirstlane(s_cum_tiles[lo]);
const int m_start_g  = __builtin_amdgcn_readfirstlane(s_offs[group_idx]);
const int m_end_g    = __builtin_amdgcn_readfirstlane(s_offs[group_idx + 1]);
```

## Resource-usage delta (`-Rpass-analysis=kernel-resource-usage`)

| spec | TotalSGPRs | Spill | ScratchSize | ΔSGPR | ΔSpill | verdict |
|------|-----------:|------:|------------:|------:|-------:|---------|
| baseline `<0,F,F>` | 65 | **67** | 272 | – | – | – |
| r29-dm  `<0,F,F>` | 66 | 67     | 272 | +1 | 0   | **no-op** |
| baseline `<0,T,F>` | 71 | **76** | 308 | – | – | – |
| r29-dm  `<0,T,F>` | 71 | 76     | 308 | 0  | 0   | **no-op** |
| baseline `<0,F,T>` | 79 | **48** | 196 | – | – | – |
| r29-dm  `<0,F,T>` | 84 | 68     | 276 | +5 | **+20** | **REGRESSION** |
| baseline `<0,T,T>` | 83 | **58** | 236 | – | – | – |
| r29-dm  `<0,T,T>` | 87 | 75     | 304 | +4 | **+17** | **REGRESSION** |

For the K-aligned specs (`<0,F,F>`, `<0,T,F>`), the explicit readfirstlane
was either silently dropped by the compiler (`<0,T,F>`: 0,0 delta) or
materialized one extra SGPR with no VGPR savings (`<0,F,F>`: +1 SGPR,
0 spill change). For the K-misaligned specs (`<0,F,T>`, `<0,T,T>` —
which carry the FUSED_KTAIL block's `a_kt1` register tile and tighter
register pressure already), the explicit readfirstlane evidently
forced an artificial VGPR materialization for the readfirstlane
operand that cascaded into +17 to +20 NEW spill slots, despite
adding 4-5 SGPRs.

## Why the hypothesis was wrong

LLVM's uniform-value analysis on AMDGCN already tracks divergent vs
uniform values. For LDS reads with a uniform address, the analysis
recognizes the result is uniform across active lanes and SGPR-allocates
the consumer ops (the binary-search compare `gt >= s_cum_tiles[mid]`,
the assignment `lo = mid`, etc.). The compiler-generated code path was
already optimal at the binary-search granularity.

By wrapping the LDS read with **explicit** readfirstlane, we forced
the LDS load result to first land in a VGPR (the readfirstlane builtin
has VGPR-source semantics), then issued a `v_readfirstlane_b32`
instruction to extract lane-0 into an SGPR. This added a VGPR
materialization step that wasn't there in the baseline. For specs
already at register-pressure ceiling (FUSED_KTAIL=true with `a_kt1`),
this nudge was enough to push 17-20 additional values into scratch.

This precisely mirrors round-20's earlier finding (`round-20-dm-fp8-readfirstlane-on-lds-tile-base-was-already-hoisted.md`):
"compiler was already doing this" for `lds_tile_base` in
`rcr_8w_load_hoist`. The same conclusion now extends to the
binary-search LDS reads in the persistent-loop prologue:

> Explicit `__builtin_amdgcn_readfirstlane` on values the compiler
> can already prove uniform is at best a no-op; on tightly-packed
> kernels it can be **harmful** by forcing an artificial VGPR-
> materialization round-trip.

## Round-15-dm rediscovery (correction to round-28-dm plan)

Round-28-dm's "next-round plan" recommended starting the cell-shape
migration as a template-specialized `bool USE_32x32_CELL = false` flag
on `grouped_rcr_kernel`. **This direction was already exhaustively
falsified** in round-15-dm (commit `7b08432a`):

| shape (round-15-dm rocprof) | ratio | HK MfmaUtil | TRT MfmaUtil | delta |
|-----------------------------|------:|------------:|-------------:|------:|
| DSV3-Down-B16-M4096         | 0.948 | **40.52 %** | 40.05 %      | **+0.47 pp** |
| gpt_oss-Down-B4-M4096       | 0.964 | **42.84 %** | 42.64 %      | **+0.20 pp** |
| gpt_oss-GateUP-B4-M2048     | 0.963 | 38.69 %     | 40.12 %      | −1.43 pp     |

Two of three worst FP8 shapes have HK at MfmaUtil parity-or-better with
Triton; longer MFMAs cannot raise pipe utilization above what's
already achieved. Migrating buys ≤ 1.4 pp on a single shape at the
cost of touching main-loop / epilog 1 / epilog 2 / FUSED_KTAIL /
scale / store across two register-tile shapes — a clear negative-EV
trade. The `rt_32x64`/`rt_64x32` scaffolding from round-14
(`include/types/register/rt_shape.cuh` + `mma.cuh`) remains
unreachable dead code; flagged for cleanup, not blocking.

## Where the gap actually is (round-23-dm verdict, still standing)

Round-23-dm's wall-time-vs-PMC head-to-head probe (HK 1.07× cycles
for 1.01× instructions) localized the residual ~6-8 % FP8 grouped
gap to **pipeline efficiency / cycles-per-instruction**, NOT
instruction count, NOT register pressure (occupancy-stalls), NOT
host overhead. R26-dm and R27-dm then exhausted single-knob
pipeline-scheduling levers (sched_barrier MASK ∈ {0,
0x008, 0x100, 0x108, 0x10A, 0x308}; sched_barrier added 1x/2x
post-MMA-A,C; `s_setprio(1)→s_setprio(3)` 4 sites — all in noise band).

Round-28-dm then ruled out compiler-level register-allocation hints
(`amdgpu_waves_per_eu(2,4)`, `amdgpu_waves_per_eu(3)`,
`amdgpu_num_vgpr(192)`) — silently ignored by LLVM, no codegen change.

Round-29-dm (this round) rules out manual readfirstlane on the
binary-search LDS reads.

## Updated falsification table (now 16 directions across r17-r29-dm)

| # | round | direction                                | result |
|--:|------:|------------------------------------------|--------|
| 1 |   R17 | rocprof PMC instr count                  | gap is in CPI, not insts |
| 2 |   R18 | hoist `make_srsrc` out of helper         | compiler already CSE'd   |
| 3 |   R19 | port two-tile main loop from dense       | -37% spill cliff         |
| 4 |   R20 | `readfirstlane(lds_tile_base)`           | compiler already hoisted |
| 5 |   R21 | `__noinline__` on `rcr_8w_load_hoist`    | architecturally inline   |
| 6 |   R22 | host-overhead trim                       | host = 0.9-1.1% only     |
| 7 |   R23 | (no change) HK vs Triton breakdown       | gap = main-kernel CPI    |
| 8 |   R24 | `RCR_PREFETCH_LGKM` sweep {2,4,6,8}      | saturated                |
| 9 |   R26 | `RCR_SCHED_BARRIER` mask sweep (macro)   | macro doesn't reach grouped main loop |
|10 |   R27 | `+sched_barrier(0x108)` 1x/2x grouped main loop | mean = baseline   |
|11 |   R27 | `+sched_barrier(0x100)` 1x post-MMA-A    | mean = baseline          |
|12 |   R27 | `s_setprio(1) → s_setprio(3)` 4 sites    | mean = baseline          |
|13 |   R28-dm | `__attribute__((amdgpu_waves_per_eu(2,4)))` | silently ignored      |
|14 |   R28-dm | `__attribute__((amdgpu_waves_per_eu(3)))`  | silently ignored       |
|15 |   R28-dm | `__attribute__((amdgpu_num_vgpr(192)))`    | silently ignored       |
|16 | **R29-dm** | **explicit readfirstlane on binary-search LDS reads** | **+17 to +20 spills on `<0,F,T>` / `<0,T,T>`; reverted** |
| – |   R15-dm | MFMA cell-shape migration (rocprof PMC)    | parity-or-better; ≤ 0.06 pp geomean |

## What's left (rounds 30+)

After r15-r29 cumulative falsification, the remaining productive
levers are all **architecturally invasive**:

1. **`grouped_var_k_kernel_fp8` register reduction (untargeted)** —
   the dB backward kernel reports SGPR=79, VGPRs=256, **52 spills**,
   ScratchSize=212. Smaller absolute spill count than the forward
   `<0,F,F>` (67), but still 52 active spill slots. Backward is
   metric-invisible, so any change must be hand-benched
   (`benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8`) and
   the fwd+bwd TFLOPS + correctness pasted into the commit message
   per the round-2 instruction. Risk: similar to forward — anything
   that seems to lighten pressure may regress in the FUSED_KTAIL-
   equivalent variants.

2. **`a_kt1` register-tile elimination (FUSED_KTAIL=true specs)** —
   the scoped `A_row_reg a_kt1` declaration inside the K-tail block
   (line 2289) carries 32 VGPRs of state for the round-12-dm
   2-stage `vmcnt(8)/vmcnt(0)` overlap optimization. Eliminating
   it (revert to single-tile sequential `a` overwrite) saves 32 VGPRs
   at the cost of ~50-100 cyc per output tile (lost overlap). Net:
   may flip the spill cliff for `<0,F,T>`+`<0,T,T>` specs (which
   are at 48/58 spills today). Risk: high — directly opposes the
   round-12-dm shipped optimization, and round-12-dm's K-tail PMC
   shift was the specific change that closed the round-16 hot-spot
   from 33.6 % → 42.84 % MfmaUtil. Likely net-negative.

3. **N=2880 partial-tile-aware dispatch (gpt_oss only)** — bpc=12 with
   the last col-tile only 64-wide (25 % utilization) wastes 8.3 %
   of the per-group MFMA work on `gpt_oss-*` shapes. A separate
   N-tail correction kernel (analogous to existing
   `grouped_ntail_kernel_lds_rrr` for BF16) would let the main
   kernel run with bpc=11 and N=2816 (256-aligned, no `N_MASKED_STORE`),
   eliminating both the wasted partial-tile MFMAs and the
   `<0,T,*>` template specs entirely. Risk: medium — major
   architectural change; need to verify can_handle and dispatch
   logic preserves BF16 path. Reward: hits gpt_oss specifically
   (4 of 16 shapes × ~1.0pp expected each) so ~+0.25 pp geomean.

4. **`__builtin_amdgcn_sched_group_barrier` (NOT the same as
   `sched_barrier` mask)** — `sched_group_barrier(MASK, COUNT,
   GROUP_ID)` declares an exact instruction count of class MASK
   between this barrier and the next group barrier with the same
   GROUP_ID, FORCING a specific interleave instead of allowing one.
   HK's own attention kernels use this (`kernels/attn/gqa/kernel.cpp:32-48`).
   Round-26-dm's mask sweep was a STRICTLY DIFFERENT lever (just
   widening the existing `sched_barrier(MASK)` allow-list).
   Round-25 documented this as "Option A, untested" and round-26
   chose the cheaper variant; round-27 then exhausted the
   alternative `+sched_barrier(MASK)` direction. The
   `sched_group_barrier` direction remains the single highest-EV
   pipeline-scheduling lever NOT yet tested. Risk: medium (~30 line
   kernel diff, not template-gated). May or may not move CPI.

## Round 29-dm verdict

- HipKittens kernel diff: zero (probe applied + reverted; `.so` rebuilt
  to match source — bytes-identical).
- HipKittens HEAD: unchanged (no commit).
- Primus-Turbo: this notes file only.
- Score band still 818-829 over r17-r28; round-29-dm metric reading
  854 (down 4 from R28-dm 858, all within run-to-run noise; same
  16/16 below 1.20 target).

## For round 30+ cold-start

Read this doc + round-15-dm + round-23-dm + round-26/27-dm. **Do
NOT** retry: cell-shape migration (R15-dm), `readfirstlane` on any
uniform value (R20+R29-dm), `RCR_SCHED_BARRIER` mask sweep (R26/R27),
single-knob `sched_barrier` adds (R27), `s_setprio` direction sweep
(R27), compiler hints on register count (R28-dm). The cleanest
single-round commit still pending is `sched_group_barrier` (lever
#4 above) — same 30-line diff scope as the R27 single-knob attempts
but a structurally different scheduling primitive that R26 explicitly
flagged as "untested, may or may not help." If that fails, the
remaining levers are all architecturally invasive (#1-#3) and need
multi-round porting plans.
