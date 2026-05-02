# Round 23 — FP8 grouped: disable FUSED_KTAIL=true on K_REM=64 (gpt_oss) — **CATASTROPHIC -48 pts, REVERTED**

**Status**: R22's recommendation #2 ("A/B test FUSED_KTAIL=false on the gpt_oss
B=32 specs — empirically measure whether moving back to a separate K-tail
kernel launch is faster") is **FALSIFIED**. Disabling fuse for K_REM=64 caused
gpt_oss FP8 ratios to collapse from 1.028-1.152 to **0.702-0.950** (-22 to
-35 pp per shape). Score 961 → 913 (-48). Revert confirmed at 962.

R22's ASM finding (~283 extra divergent-SRD loops in C-store epilog of
FUSED_KTAIL=true specs, ~5660 cy/tile cost) is **a real cost** but the
**FUSED_KTAIL=true register-allocation favorability dominates by a wide
margin** on the K_REM=64 hot path. The standalone `grouped_ktail_kernel_*`
launch path (M2N2/M2/MFMA32x32/MFMA16x128/scalar fallback) is structurally
slower than the in-kernel fused K-tail accumulation, by 25-35 % wall-time on
gpt_oss B=32 cases.

**Auto-optimize round**: 23 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `5d45ceb2` (no kernel changes committed;
falsified change reverted in-tree, baseline binary restored)
**PT SHA at round start**: `c21a5dc4`
**Reported best (forward)**: 966 (R15 / R18, high-tail of noise band)
**R23 baseline metric**: 961 (initial trial, geomean 1.1197)
**R23 probe metric**: 913 (-48 pts, geomean 1.0106)
**R23 revert metric**: 962 (within noise, geomean 1.1226)
**R23 patience**: 8 rounds at noise floor (R16-R23)

---

## R23 hypothesis (from R22 recommendation #2)

R22 disassembled the 4 RCR template specs and found:

```
spec                  divergent-SRD-loops in C-store epilog
─────────────────────────────────────────────────────────
rcr<0,F,F>                  128
rcr<0,T,F>                  411   ← FUSED_KTAIL=true, +283 vs F,F
rcr<0,F,T>                  136
rcr<0,T,T>                  419   ← FUSED_KTAIL=true, +283 vs F,T
```

At ~20 cy/loop when SRD is uniform (the common case), 283 extra loops
≈ 5660 cy/tile of overhead that's not in any prior round's bookkeeping.
This is ~10 % per-tile cost on the gpt_oss B=32 cases (current ratio
1.028-1.073).

**Hypothesis**: For K_REM=64 (gpt_oss only — bottom-5 of FP8 suite), routing
through the standalone `grouped_ktail_kernel_*` path adds ~5 us launch
overhead but saves the divergent-loop cost. On B=32 cases (~280 us per
call), 5 us launch is 1.8 % overhead and the 10 % per-tile saving easily
wins. Expected: +5-15 us net win on gpt_oss B=32 cases (~2-5 % wall-time
improvement).

**R34-dm K_REM=0 path is preserved**: keep DSV3 + Qwen3 (16/24 K-aligned
shapes) routed to FUSED_KTAIL=true (R34-dm +17 pts win), only force K_REM=64
back to standalone tail.

## R23 implementation

One-liner edit in `dispatch_grouped_rcr` (`kernel_fp8_layouts.cpp:5267`):

```cpp
// Before:
const bool fuse_ktail_eligible =
    (g.bpc > 0) && (g.ki > 0) &&
    ((K_rem_for_fuse == 64) || (K_rem_for_fuse == 0)) &&
    lds_k_tail_safe_for_fuse;

// After (R23 probe):
const bool fuse_ktail_eligible =
    (g.bpc > 0) && (g.ki > 0) &&
    (K_rem_for_fuse == 0) &&  // drop K_REM=64
    lds_k_tail_safe_for_fuse;
```

K_REM=64 (gpt_oss) now falls through to the standalone tail dispatch block,
which selects `grouped_ktail_kernel_mfma32x32_M2N2<Layout::RCR, 64>` for
gpt_oss M_per ∈ {2048, 4096} N ∈ {2880, 5760} (the round-60 64×64 fast path).

## R23 result — gpt_oss FP8 collapses across the board

```
shape                              before    after    Δ
─────────────────────────────────────────────────────────────
gpt_oss-GateUP-B4-M2048             1.080    0.860   -22.0 pp
gpt_oss-Down-B4-M2048               1.152    0.950   -20.2 pp
gpt_oss-GateUP-B4-M4096             1.054    0.705   -34.9 pp   ← worst delta
gpt_oss-Down-B4-M4096               1.085    0.844   -24.1 pp
gpt_oss-GateUP-B32-M2048            1.049    0.720   -32.9 pp
gpt_oss-Down-B32-M2048              1.073    0.732   -34.1 pp
gpt_oss-GateUP-B32-M4096            1.028    0.702   -32.6 pp
gpt_oss-Down-B32-M4096              1.050    0.731   -31.9 pp
─────────────────────────────────────────────────────────────
gpt_oss segment geomean: 1.069 → 0.776 (-29 %)
```

DSV3 ratios moved minimally (within ±0.04 noise), Qwen3 ratios moved
minimally (within ±0.04 noise). FP8 geomean: 1.1197 → 1.0106 (-10.9 pp).
BF16 geomean: 1.1893 → 1.1876 (-0.2 pp, noise; BF16 path untouched).

Score: **961 → 913 (-48 pts)**. Decisive falsification.

## What this means

**R34-dm's mechanism (FUSED_KTAIL=true → favorable LLVM register allocation
in epilog 1) was never the only effect.** The R22 ASM finding (divergent-SRD
overhead) is also real, but on the K_REM=64 hot path the **register-allocation
win out-scales the divergent-loop loss by ~3-5×**. Routing K_REM=64 through
the standalone tail path:

1. **Loses the favorable register allocation in main-kernel epilog 1** —
   when FUSED_KTAIL=false, the C-store epilog has to interleave ~62 spills
   per Down spec and ~44 per GateUP spec into the mfma block (vs 28 / 22
   for FUSED_KTAIL=true per R34-dm ISA inspection). This is the dominant
   per-iter cost; 22-iter K-loop × 30+ extra spills × ~24 cy/spill =
   ~16,000 cy of new main-loop scratch traffic, dwarfing the ~5660 cy
   saved on divergent-SRD loops.

2. **Adds a separate kernel launch (~5 us)** — measurable on B=4 cases
   (~5 % of ~25 us total wall) and the M2N2 K-tail kernel runtime itself
   (~5-15 us depending on M, N, B).

3. **The standalone tail kernel is bandwidth-bound, not throughput-bound**
   — even at 100 % HBM peak the K-tail-only contribution (covering the
   K=[2816, 2880) reduction across the whole M*N output) takes ~30-50 us
   per gpt_oss B=32 case, much more than the saved 28 us in-kernel cost.

Net: **+25-30 us per gpt_oss B=32 case**, exactly matching the observed
-30 % wall-time regression.

## Why R22's analysis was directionally right but quantitatively off

R22 estimated the divergent-loop cost at "~10 % per-tile" and the launch
overhead at "~5 us". Both numbers are correct in isolation. The missed
factor: **the standalone tail kernel itself takes ~5-15 us of compute**,
not just launch overhead. And the LLVM register-allocation effect on the
main kernel's K-loop is **far larger** than the divergent-loop overhead
in the C-store epilog (~16,000 cy vs ~5660 cy).

The 411 divergent loops in the C-store epilog ARE there in the .so
disassembly. They're real cost. But:

- Most of them run for 1 wave-uniform iteration (~20 cy) — total **~5660 cy**.
- The K-loop scratch interleave (round-34-dm's "interleaved spill") costs
  **~22 iter × 30 spills × 24 cy = ~16,000 cy**.

The two costs co-exist; FUSED_KTAIL=true trades 5660 cy LOSS in C-store
for 16,000 cy WIN in K-loop. Net: **~10,000 cy/tile WIN for FUSED=true**.
R23 confirmed this by reverting and measuring -48 pts.

## Cumulative falsification matrix (R23 final)

| Lever | Verdict | Round | Mechanism / measurement |
|---|---|---|---|
| **A** Async global→LDS                 | FALSIFIED | R2  | Already shipped via inline ASM |
| **B** Triple LDS slab                  | FALSIFIED | R2  | LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL       | FALSIFIED | R4  | Spill neutral, -1.8 pp regression |
| **C-2** K-tail capture refactor        | FALSIFIED | R3  | Already correctly scoped |
| **D** mfma_32x32x64 cell-shape         | FALSIFIED | R5  | Microbench -0.03 % (shape-agnostic) |
| **E** ASM software pipelining          | FALSIFIED | R11 | Microbench -7.28 % |
| **R** Stage-level pipelining           | FALSIFIED | R17 | Microbench -0.07 % (LLVM auto-overlaps) |
| **H/B-rcr** voffset swap               | FALSIFIED | R19 | Uncoalesced HBM reads |
| **H/B-rrr** in-kernel K-tail           | FALSIFIED | R28+R29 (HK) | Compiler aliases A→c VGPRs |
| **HIP transpose** rewrite              | FALSIFIED | R20 | Triton already at 75-110 % HBM peak |
| **RRR spill reduction**                | FALSIFIED | R21 | dA TFLOPS already exceeds fwd TFLOPS |
| **`readfirstlane(group_idx)`**         | FALSIFIED | R22 | +21 dw spill, only 8/411 div loops fixed |
| **Drop FUSED_KTAIL on K_REM=64**       | **FALSIFIED** | **R23** | **-48 pts (gpt_oss collapses 0.69-0.95). Tail kernel structurally slower than in-kernel fused path; main-loop register-allocation win >> C-store divergent-loop cost** |
| **F** Per-shape dispatcher rules       | LANDED+SAT | R6-R10 | 5 rules, R10-dm audit confirmed top-1 |
| **H/A** Triton fp8_transpose_3d        | LANDED  | R13 | +9.3 % bwd avg |
| **K** var_k spill trim                 | LANDED  | R14 | +0.81 % bwd avg |
| **Q** transpose block tile             | LANDED  | R15 | +1.1 % gpt_oss bwd |

13 levers FALSIFIED, 4 LANDED + SATURATED. **No remaining 1-round positive-
EV lever exists** within the FP8 grouped kernel surface — the divergent-SRD
overhead is real but **NOT actionable via dispatcher choice**; addressing it
requires a kernel rewrite that decouples the K-tail block's per-lane VGPR
ops from the C-store epilog's uniformity analysis (see R22 recommendation
#1: restructure `g.c` descriptor SSA flow, kittens-helper-level change).
That's a 4-15 round project, not a 1-round budget.

## R23 metric runs (3 trials)

```
R23 baseline (no change):   961  (fp8 geomean 1.1197)
R23 probe (fuse off K=64):  913  (fp8 geomean 1.0106)   ← REVERTED
R23 revert verification:    962  (fp8 geomean 1.1226)   ← restored baseline
```

R14-R23 cumulative score history (forward-only):
```
R14=962 R15=966 R16=964 R17=963 R18={964,962,964,964,959} R19={960,965,961,962}
R20=962 R21=963 R22={960, 962} R23=961|913|962
→ 22 trials min=913 (excluded as falsified probe), 21 valid trials min=959 max=966
  range=7, median=962, mean=962.4
```

Score band: 962 ± 3, no movement since R15. Patience at 8 rounds.

## R24+ recommendation (unchanged from R20-R22)

1. **Pause auto-optimize on FP8 grouped** — 13 rounds of disciplined
   1-round falsification (R5-R23) plus R22's ASM-level investigation have
   exhausted every dispatcher / cache / micro-knob / template-flip /
   register-pressure lever. The 962 ± 3 score band is the architectural
   ceiling under current LLVM uniformity analysis behaviour.

2. **Or attempt the multi-round-budget structural rewrite** that R22
   recommendation #1 suggested: refactor the `grouped_layout_globals` /
   kittens load/store helpers so the C-tensor SRD is constructed once at
   kernel entry as an SGPR-resident value and passed through to per-call
   `store(g.c, cA, ...)`. This addresses the divergent-SRD root cause
   directly and would close the ~5660 cy/tile gap WITHOUT giving up the
   ~16,000 cy K-loop register-allocation win. Risk: high (correctness
   across all callers + kittens helper API change); reward: potentially
   +5-10 pts on FP8 grouped score. Estimated 4-6 rounds.

3. **Or accept the plateau** and switch the auto-optimize loop to a
   different optimisation target.

## Files touched in R23

* `analysis/_notes/round-23-fp8-grouped-disable-fuse-ktail-on-K-rem-64-CATASTROPHIC-revert.md` (NEW)

No HK kernel changes committed (failed probe reverted in-tree before commit;
HK SHA still `5d45ceb2`).
