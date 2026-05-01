# Round 24 (DM): RCR_PREFETCH_LGKM sweep saturated — knob is dead

## Status
- HEAD start: `06037054`
- Best: 829 (R18)
- This round baseline (LGKM=4): 824
- 5 consecutive rounds without improvement (818-829 noise band, mean 822)

## Hypothesis (from R23 doc)
R23 confirmed the gap is pipeline stalls (8.4× CPI vs instruction ratio).
The recommended next move was MFMA scheduling adjustments. Before doing
heavyweight changes (`s_setprio(3)`, `sched_group_barrier`), test whether
the existing `RCR_PREFETCH_LGKM` knob has any signal. If the knob is
dead, then schedule slack isn't the issue and we need a different
attack.

The current value is `RCR_PREFETCH_LGKM=4` (line 55 of
`kernel_fp8_layouts.cpp`). RRR uses 8. Higher value = more in-flight
LDS reads allowed before MFMA fires. Lower = force MFMA to wait for
LDS to drain more aggressively.

## Sweep result

| LGKM | metric score | delta vs LGKM=4 |
|-----:|-------------:|----------------:|
|    2 |          820 |             -4  |
|    4 (baseline) |  824 |       0  |
|    4 (re-baseline at end) | 819 |  -5  |
|    6 |          824 |              0  |
|    8 |          820 |             -4  |

All 5 measurements span 819-824 — exactly the **818-829 noise band**
characterized in R22-23. The knob has no signal — every value sits in
the noise.

## Falsification

`RCR_PREFETCH_LGKM` is **saturated at 4**. Tuning it ±2/±4 does not
move the metric beyond noise. This rules out "main-loop LDS-vs-MFMA
overlap is wrong by a factor of 2" as the bottleneck.

By the same token, this means the round-23 hypothesis "pipeline stalls
are tunable via wait-counter slack" is also weakened. If LGKM slack
doesn't help, then VMCNT slack (already in falsified list, R14/16/22/24/25)
also won't help. This narrows the remaining attack surface significantly.

## What this means for rounds 25+

The CYCLE-PER-INSTRUCTION gap from round 23 (HK 1.07× Triton) is NOT
attributable to:
  - Instruction count (R17 PMC: 1.01× — basically tied)
  - Host overhead (R22 probe: 0.9-1.1%)
  - LDS-vs-MFMA wait slack (R24 sweep: saturated)
  - Inline footprint (R18, R20, R21 falsified)
  - Code volume (R19 spill cliff)
  - MFMA cell shape (R14, R15 falsified)
  - Dispatch chunking (R16 falsified)
  - Per-shape config (R7-69 saturated)

What's left in the toolbox:
  - **A. `__builtin_amdgcn_sched_group_barrier`** to constrain the
    LLVM machine scheduler. The kernel uses `sched_barrier(0)` (block
    all reorder) but never `sched_group_barrier(MASK, COUNT, SYNC_ID)`
    (constrain instruction-class ordering). Triton's MLIR backend uses
    these; HK's hand-written kernel does not.
  - **B. LDS bank conflict audit on `ds_read_b128` for FP8 K_BLOCK=128**.
    Round-23 wall-time gap might be from LDS read serialization, not
    from MFMA pipelining. Need to PMC `SQ_LDS_BANK_CONFLICT` HK vs
    Triton on the same shape.
  - **C. Accept ~825 as ceiling** — the score has been within 818-829
    for 5 rounds, drifting in the noise band. The metric goal of 1.20×
    requires ~22% improvement; we're at parity (1.00×). Triton is
    well-optimized; HK is well-optimized; the architectural gap is
    real and small.

## Round 24 verdict

Tested LGKM 2/4/6/8 for the RCR steady-state main loop. All within
the 818-829 noise band. Reverted to LGKM=4. Knob is saturated.

**Metric**: 819 (final after revert; same as round 21).

## Recommended round 25+ starting move

Chat is rolling over (82+/90 min). Next agent will cold-start.

1. **Read `analysis/_notes/round-{17..24}-dm-*.md`** (8 falsifications,
   each documents a specific dead-end direction).
2. **Do NOT retry**: SRD hoist (R18), two-tile port (R19), readfirstlane
   (R20), `__noinline__` (R21), host trim (R22), MFMA cell migration
   (R14-15), chunk_size sweep (R16), per-shape config (R7-69), VMCNT
   sweep (R14/16/22/24/25), **LGKM sweep (R24, this doc)**.
3. **Do try**:
   - `__builtin_amdgcn_sched_group_barrier(MFMA, 4, 0)` insertion
     between MFMA blocks (LLVM ≥18 supported)
   - rocprof PMC on `SQ_LDS_BANK_CONFLICT` HK vs Triton same shape
4. **Honest signal test**: any commit that doesn't beat 829 by ≥10 is
   noise. Don't chase noise. After round 30, if score is still
   <840, escalate to user that the FP8 grouped main kernel may
   have hit its architectural ceiling vs Triton.
