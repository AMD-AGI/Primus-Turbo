# Round 63 — FP8 grouped: Lever F (KI_HINT short-K specialization) FALSIFIED

**Date**: 2026-05-02 (R63 of 100)
**HEAD before**: 3d3a21aedbf8c1431d92db514e0e695da6b7ea0c
**Score**: baseline 978 → KI_HINT applied 906 (-72!) → reverted 978 (no regression)
**Goal**: per R62 plan, prototype Lever F — `KI_HINT={12,32}` template
specializations for Qwen Down (K=1536, ki=12) and Qwen GateUP (K=4096, ki=32),
expecting +2-3pp via compile-time loop unroll.

## TL;DR

**FALSIFIED IMMEDIATELY ON FIRST METRIC RUN.** Compile-time KI_HINT
specialization caused a **catastrophic regression** on Qwen FP8 cases:

| Spec | Spill | Scratch [B/lane] |
|---|---|---|
| `<0, false, true>` (current Qwen path) | **34** | 140 |
| `<12, false, true>` (KI_HINT=12 — Qwen Down) | **49** | 200 |
| `<32, false, true>` (KI_HINT=32 — Qwen GateUP) | **49** | 200 |

Compile-time loop unroll exposed too many parallel live ranges that
LLVM couldn't fold. Per-shape Qwen FP8 ratios:

| Shape | Before (KI_HINT=0) | With KI_HINT | Δ |
|---|---|---|---|
| Qwen3 GateUP B16 M2048 | 1.182 | **0.676** | -0.506 |
| Qwen3 Down   B16 M2048 | 1.171 | **0.758** | -0.413 |
| Qwen3 GateUP B16 M4096 | 1.168 | **0.721** | -0.447 |
| Qwen3 Down   B16 M4096 | 1.136 | **0.736** | -0.400 |
| Qwen3 GateUP B32 M2048 | 1.167 | **0.693** | -0.474 |
| Qwen3 Down   B32 M2048 | 1.188 | **0.771** | -0.417 |
| Qwen3 GateUP B32 M4096 | 1.186 | **0.750** | -0.436 |
| Qwen3 Down   B32 M4096 | 1.174 | **0.760** | -0.414 |

Score: 978 → **906 (-72 points)** → reverted → 978.

## Why it failed

The KI_HINT mechanism uses `ki_dyn = (KI_HINT > 0) ? KI_HINT : g.ki`
(line 2355). With `KI_HINT > 0`, `ki_dyn` is a compile-time constant,
so the K-loop `for (int k = 0; k < ki_dyn - 2; k++)` becomes
fully unrollable.

The current K-loop body has 4 `mma_ABt` calls + 4 `rcr_8w_load_hoist`
calls per iter. With 12 (or 32) iters fully unrolled, that's 48 (or 128)
mma_ABt calls + 48 (or 128) load calls inlined — each carrying its own
A/B register live range. LLVM tried to keep these all live in parallel
(for instruction scheduling) which exceeded the VGPR budget by ~15
slots, forcing spill 34 → 49.

The runtime loop (KI_HINT=0) lets LLVM REUSE the same A/B register
slots across iterations because it doesn't see them all at once.

## Lessons

1. **Compile-time loop unroll is NOT free for register-pressured kernels.**
   The "tighter liveness analysis" benefit only manifests when the loop
   body is small enough to fit comfortably. For our 4-mma-cell K-loop,
   each iter is too register-heavy to unroll without spilling.

2. **Spill report is a real signal.** Spill 34 → 49 (+44%) translates
   directly to +44% scratch traffic + scratch_load latency in the K-loop.
   For Qwen K=12 (1536), that's ~12 scratch-load stalls per K-iter ×
   12 iters = ~144 wasted cycles per tile = ~30% of tile time. Matches
   the observed 30-50% perf regression.

3. **Skill: when in doubt, check spill report BEFORE running metric.**
   Could have caught this earlier; the spill 34 → 49 was a clear red
   flag before metric confirmed the regression.

## Lever scoreboard update

| Lever | Status | Comment |
|---|---|---|
| A. Async load | **APPLIED** | rcr_8w_load_hoist (R62 audit) |
| B. LDS ping-pong | **APPLIED** | As[2][2], Bs[2][2] |
| C-1. Reduce VGPR | plateau | spill 34-37, no easy win |
| C-2. Force AGPR | **CLOSED R61** | LLVM bug |
| D. 32x32x64 mfma | **FALSIFIED** R37+ | |
| E. Manual ASM main loop | NOT TRIED | high risk, last resort |
| **F. Qwen-Down K=1536** | **FALSIFIED R63** | KI_HINT regressed -72 |

**5 of 6 standard levers exhausted**. Only Lever E (manual ASM) and
exotic ideas remain.

## Files touched (R63)

- `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - **REVERTED** all R63 changes after FALSIFY
  - left a 8-line FALSIFY comment in the dispatcher to prevent re-trying
- `Primus-Turbo/analysis/_notes/round-63-dm-...`: this note

## Production kernel impact

NONE (changes reverted). Metric: 978 → 978 (no net change).

## Roadmap (R64+)

The Lever F falsification leaves us with no clear path to score >985.
Options for R64+:

1. **Try Lever E (manual ASM main loop)**: high effort, multi-round.
   Probably needs 3-5 rounds of careful ISA editing. High risk of
   correctness regression.

2. **Try a "dummy declaration" trick** (R34-style): add an unused
   register tile declaration that perturbs LLVM's liveness graph
   in a beneficial way. Cheap to try (1 round).

3. **Accept plateau at 977-983**. With 30-patience budget, we have
   ~18 rounds before patience runs out. Documenting the closure
   decisions for posterity is also valuable.

4. **Explore exotic Lever G — single-warp CRR / cluster-cross load**:
   experimental, no precedent. Very risky.

R64 will try option 2 (dummy declaration trick) as the cheapest
remaining experiment. If falsified, R65 commits to Lever E or
plateau acceptance.

## Worst 5 FP8 cases (R63 baseline, unchanged from R62)

```
1. gpt_oss GateUP B32 M4096    1.077 → 1.080 (mixed noise)
2. gpt_oss Down   B32 M2048    1.104
3. gpt_oss GateUP B32 M2048    1.107
4. gpt_oss GateUP B4  M4096    1.108
5. gpt_oss Down   B32 M4096    1.111
```

Note: the worst-5 are ALL gpt_oss, not Qwen. R62-planned focus on
Qwen was already a marginal-impact play (4 cases × +2-3pp = +0.5pp
geomean). gpt_oss is the real bottleneck.
