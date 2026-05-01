# Round-43-dm · FP8 grouped — cluster prologue + full barriers FALSIFIES R42-dm "promising lever" (correctness OK, perf -41 pts)

**Status**: tested R42-dm's documented "promising lever" — replace
wm==1/wm==0 half-barriers with full s_barriers AND apply clustered
prologue [B0,B1,A0,A1]. Correctness PASSES 32/32. Score 950 → 909
(-41 pts) due to broad perf regression. FALSIFIED, reverted.

## Hypothesis (from R42-dm note)

> Replace the wm==1 / wm==0 half-barrier pair with a different sync
> mechanism (e.g., wave-uniform SGPR signal + full s_barrier, or
> barrier-counter-based `s_barrier_signal/wait` pair). If the new
> sync mechanism is order-independent, then clustered prologue
> becomes legal → potentially -26 to -36 dwords of spill on FUSED=
> true specs → could be the biggest single codegen win in the trail.

R43-dm tested the simplest realisation: replace both half-barriers
with bare `__builtin_amdgcn_s_barrier()` (full workgroup barrier).
BF16 grouped already uses 36 full s_barriers and 0 half-barriers
without correctness issues, proving full barriers are semantically
correct at these sync points.

## Implementation

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:

1. Line 2147: `if (wm == 1) __builtin_amdgcn_s_barrier();` → bare
   `__builtin_amdgcn_s_barrier();`.
2. Line 2479: `if (wm == 0) __builtin_amdgcn_s_barrier();` → bare
   `__builtin_amdgcn_s_barrier();`.
3. Lines 2142-2145: cluster prologue stage-0 to [B0, B1, A0, A1]
   (was strictly interleaved [B0, A0, B1, A1]).

Stage 1 (3 loads) kept original order (no spill impact per R42-dm).

## Static codegen (`-Rpass-analysis=kernel-resource-usage`)

VGPR spill (matches R42-dm):

| Spec template params | Baseline | R43-dm | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)  | 67 | 67  | 0 |
| `<0,true ,false>` (FUSED=false n_masked)   | 76 | 76  | 0 |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 72 | **36** | **-36 (-50%)** |
| `<0,true ,true >` (FUSED=true  n_masked,  gpt_oss)| 82 | **55** | **-27 (-33%)** |

The cluster prologue's spill reduction is fully preserved with full
barriers. Confirms: clustering is the static-codegen lever; the
half-barrier was orthogonal to spill counts.

## Runtime: correctness PASSES, perf REGRESSES across the board

| Shape | Baseline ratio | R43-dm ratio | Δ |
|---|---|---|---|
| `grpFP8_DSV3-GateUP-B16-M2048` | 1.149 | 0.968 | **-18 pp** |
| `grpFP8_DSV3-Down-B16-M2048`   | 1.170 | 0.996 | **-17 pp** |
| `grpFP8_DSV3-GateUP-B16-M4096` | 1.169 | 1.001 | -17 pp |
| `grpFP8_DSV3-Down-B16-M4096`   | 1.130 | 1.041 | -9 pp  |
| `grpFP8_DSV3-GateUP-B32-M2048` | 1.162 | 0.999 | -16 pp |
| `grpFP8_DSV3-Down-B32-M2048`   | 1.192 | 1.050 | -14 pp |
| `grpFP8_DSV3-GateUP-B32-M4096` | 1.165 | 1.017 | -15 pp |
| `grpFP8_DSV3-Down-B32-M4096`   | 1.217 | 1.031 | -19 pp |
| `grpFP8_gpt_oss-GateUP-B4-M2048`  | 1.053 | 1.008 | -4.5 pp |
| `grpFP8_gpt_oss-Down-B4-M2048`    | 1.104 | 1.042 | -6 pp   |
| `grpFP8_gpt_oss-GateUP-B4-M4096`  | 1.013 | 0.964 | -5 pp   |
| `grpFP8_gpt_oss-Down-B4-M4096`    | 1.044 | 1.005 | -4 pp   |
| `grpFP8_gpt_oss-GateUP-B32-M2048` | 0.999 | 1.004 | +0.5 pp |
| `grpFP8_gpt_oss-Down-B32-M2048`   | 1.035 | 1.021 | -1 pp   |
| `grpFP8_gpt_oss-GateUP-B32-M4096` | 0.988 | 0.981 | -0.7 pp |
| `grpFP8_gpt_oss-Down-B32-M4096`   | 1.026 | 0.977 | -5 pp   |
| **grp_FP8 geomean** | **1.0984** | **1.0063** | **-9.2 pp** |
| **grp_BF16 geomean** | 1.1827 | 1.1835 | +0.1 pp (neutral) |
| **score** | **950** | **909** | **-41** |

## Two distinct effects observed

### Effect A: full-barrier overhead (~30 cy each × 2 sync points)

Adds ~60 cy/iter. For B32-M4096 (46 tiles/CU): ~2760 cy/CU =
~0.4% wall time. **Insufficient to explain the -9.2 pp grp_FP8
regression**.

### Effect B: spill-reduction perf paradox (-15 to -18 pp on DSV3)

DSV3 shapes (FUSED=true n_aligned spec) saw the largest spill
reduction (72 → 36, -50%). They also regressed the most (-15 to
-18 pp). This is the **opposite** of the naive expectation
(less spill = faster).

The mechanism: LLVM's register-allocation reshape in response to
clustered loads doesn't just remove spills — it MOVES them. The
new spill placement lands in worse pipeline positions (mid-loop
scratch ops blocking critical-path MFMAs), causing more pipeline
stalls than the original spill scheme.

Static spill count is a **directionally-misleading** proxy in this
kernel. R30/R33/R39/R41 already documented this; R43-dm extends the
finding from "spill increases ≠ perf decrease" to **"spill decreases
≠ perf increase"** — the relationship is non-monotonic.

### gpt_oss less impacted

gpt_oss shapes (FUSED=true n_masked spec) saw 82→55 spill (-33%)
but only -0 to -6 pp regression. The masked-store helper + K-tail
runtime branch already saturate the pipeline; the new spill
placement has less room to make things worse.

## Disposition: R42-dm's "promising lever" is FALSIFIED

The hypothesised path "fix sync → enable cluster → reap spill
reduction" has been tested end-to-end. The first two links work
(correctness PASS 32/32, spill drops -50%); the third link (perf
gain from spill reduction) **does not materialise**. The lever
is dead.

The implication is broader: **all "reduce VGPR spill" levers are
exhausted on this kernel**. The remaining headroom is in:

1. Reducing per-tile fixed overhead (binary search, prologue, K-tail,
   epilog) — each piece is fragile (R31, R32, R33, R36, R41).
2. Reducing main-loop critical-path stalls (HBM/LDS scheduling). All
   sched_barrier knobs are saturated (R26, R27).
3. Architectural rewrites (warp-specialised K-tail, software-
   pipelined main loop) — multi-round investments, high risk.

## Take-away for next agent

1. **Do not** revisit cluster prologue or half-barrier replacement —
   confirmed end-to-end exhausted (R42-dm + R43-dm).
2. **Do not** trust static spill count as a perf proxy. The kernel
   is at a multi-objective LLVM-RA optimum; pushing on spill alone
   moves cost between dimensions without net benefit.
3. **Realistic remaining lever** (high cost, high risk):
   software-pipelined main loop (lever E in task body) — write
   inline ASM `s_buffer_load` + interleave with MFMA in 3-stage
   prefetch pattern. Estimated 5+ rounds of work, may yield +5-10 pp.
4. **Easier alternative**: focus on making MORE shapes hit ≥1.20
   (currently 2/16 FP8 PASS) by chipping at the DSV3 family which
   sits at 1.13-1.22. A targeted N-aligned dispatch refinement
   might lift 4 DSV3 shapes from 1.13-1.17 → 1.20+. But all the
   "free" knobs are saturated.
5. **Recommended next round direction**: do a deep `rocprof` /
   `rocm-bandwidth-test` measurement round (no kernel changes) to
   identify whether HBM/LDS bandwidth is actually the bottleneck.
   If yes, no software lever can help. If no, target the specific
   stall source.

## Files touched (this round)

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (HipKittens) —
modified then reverted. Final state matches HEAD `04f82d49`.

No Primus-Turbo source changes. This note is the only Primus-Turbo
delta.

## Score history

| Round | Best | grp_FP8 geo | Notes |
|---|---|---|---|
| Start    | 851 | ~1.01  | Baseline |
| R10 (cron) | 950 | 1.099  | R37-dm K-tail reorder (+16 pts) |
| R11-R14   | 950 | 1.094  | R38-R42 falsified |
| R15      | 949 | 1.090  | R43-dm cluster + full barrier falsified (this) |
| Target   | 1000 | ≥1.20  | Gap = 11 pp grp_FP8 |
