# Round 17 — FP8 grouped main loop micro-knobs FALSIFIED (3 attempts)

## TL;DR

Round 16's rocprof breakdown identified an 8pp MfmaUtil deficit between HK
FP8 grouped main loop (33.6%) and Triton (41.9%) on `Down-B4-M4096`, and
listed 4 candidate micro-levers ordered by leverage. Round 17 falsified the
two ostensibly safest ones (positions 2 and 1 of that list) and one
adjacent knob:

| # | Lever | Variant tried | Result |
|---|---|---|---|
| 1 | Drop one of 4 post-MFMA `s_barrier` | Remove last (after MFMA(cD)) | **REGRESS** — MfmaUtil 33.6 → 31.8 (−1.8pp), SQ_BUSY +6% |
| 2 | `RCR_PREFETCH_LGKM` lower-bound test | 4 → 1 | **REGRESS** — VGPR spills jump 67 → 90/76/72 across kernels (compiler can't keep ops live with shorter LGKM window) |
| 3 | Drop `s_setprio(0)` only (keep `(1)`) | Remove all 4 setprio(0) per K-iter | **REGRESS** — MfmaUtil 33.6 → 30.7 (−2.9pp), SQ_BUSY +8%, VGPR spills 99/90/67 |

All three reverted in the same session. **Score this round: 794** (unchanged
from round-16 best of 795 within the established ±3pp noise band).

## Why each one regressed (and what that tells us)

### (1) Removing post-MFMA `s_barrier` regresses MFMA throughput

Counter-intuitive. The hypothesis was that the post-MFMA cross-warp barrier
is redundant because MFMA writes only to registers (no LDS write/race). But
removing it dropped MfmaUtil by −1.8pp.

**Mechanism**: The barrier acts as a *convergence point* that lets the WG
issue MFMAs in synchronised bursts. Without it, faster waves race ahead and
stall at the next iter's `s_waitcnt lgkmcnt(0)` while slower waves are
still in MFMA. Net: instruction issue is staggered across waves, MFMA pipe
sees more bubbles, util drops.

**Conclusion**: All 4 post-MFMA `s_barrier`s in the FP8 RCR grouped main
loop are load-bearing — they're not just LDS-correctness fences but
WG-wide MFMA throughput stabilisers. **Don't try removing other s_barriers
in this loop.**

### (2) `RCR_PREFETCH_LGKM=1` triggers register spills

Round-5 swept `{2,4,8}` and called the knob saturated. Round-17 tested the
lower bound {1}: the compiler emits VGPR spills (76-90 lanes per kernel
across all three FP8 RCR variants — dense, grouped, var-K) because a
shallower LGKM window forces it to keep more live values across LDS waits.
Spill traffic on a register-bound kernel is catastrophic.

**Conclusion**: `RCR_PREFETCH_LGKM` is bounded below by 2 (round-5) and
above by 8 (round-5). Inside that range, MfmaUtil is flat. Knob is dead.

### (3) Removing `s_setprio(0)` regresses MfmaUtil and triggers spills

Round-2 falsified bundled removal of `s_setprio(1/0)` + `sched_barrier`
together (−5.6%). Round-4 isolated the `sched_barrier` part as the
*removable* one. Round-17 tested the third combination: keep `setprio(1)`
(round-4 confirmed load-bearing for MFMA priority bias), drop `setprio(0)`
only.

Result: MfmaUtil 33.6 → 30.7 (−2.9pp), SQ_BUSY +8%, **and** VGPR spills
appear (99/90 lanes in dense and grouped 2-tile variants).

**Mechanism**: `s_setprio(0)` is *both* a runtime priority restoration
*and* a compile-time scheduling barrier. The clang HIP backend treats it
as a fence in the instruction scheduler — without it, MFMAs and loads get
reordered in ways that lengthen register lifetimes and exceed the VGPR
budget.

**Conclusion**: `s_setprio(0)` is load-bearing. The pair `(1)/(0)` must
stay together. **Setprio is fully nailed down — don't revisit.**

## Updated leverage list (round 16 → round 17)

| # | Lever | Status | Risk if revisited |
|---|---|---|---|
| 1 | Drop redundant `s_setprio` toggles | **DEAD** (round-2 + round-4 + round-17 all converge) | High |
| 2 | Collapse paired `s_barrier+s_waitcnt` | **DEAD** (round-17 single-barrier removal regressed −1.8pp) | High |
| 3 | Hoist k+2 prefetch addresses out of K-iter body | UNTESTED | Medium-high (refactor) |
| 4 | Direct HBM→register path | DEAD-on-prior-attempt (round 5/6) | High structural |
| 5 | (NEW) Reduce 8 waves/WG to 4 waves/WG | UNTESTED | Very high (re-derive MFMA fragment layout) |
| 6 | (NEW) Wider MFMA op (16x16x32 → 16x16x64) | UNTESTED | Hardware-dependent (CDNA4 ISA support?) |

After this round, **every "safe" single-knob change has been falsified**.
The remaining options are all multi-round structural rewrites with
significant rollback risk.

## Mechanism evidence summary

The FP8 grouped RCR main loop's instruction schedule is at a *local
optimum* under the constraint of:
- 8 waves/WG (= 1 WG/CU because LDS=140 KB)
- 2-stage LDS double-buffer for A and B
- Cross-warp barrier between MFMA phases
- `s_setprio(1/0)` priority bias around each MFMA

To escape this local optimum, at least *one* of those four constraints
must be relaxed structurally. None of them can be tweaked with a single
`#define` or a `if constexpr` gate — they all require reshaping the kernel
body.

## What the rocprof telemetry rules in / rules out

✓ Ruled in (still candidate paths):
  1. Reduce static instruction count between MFMAs (smaller body).
  2. Shorten cross-warp synchronisation distance (fewer waves/WG).

✗ Ruled out by round 17:
  - Removing individual `s_barrier`/`s_setprio`/lowering LGKM window. None
    of these reach the rocprof MfmaUtil target without breaking other
    constraints.

## Probe artefacts (round 17 specific)

- `/tmp/rocprof_round16/hk_no_last_barrier_counter_collection.csv`
- `/tmp/rocprof_round16/hk_no_setprio0_counter_collection.csv`

## Round-17 commits

- Primus-Turbo: this notes file. No kernel change.
- HipKittens: none (all 3 micro-knob experiments reverted; falsified at the
  rocprof level, no commit warranted).

## Suggested next-round directions (all multi-round)

1. **Probe Triton's actual ISA**: dump the SQ instruction stream of
   `_grouped_fp8_persistent_gemm_kernel` with `att-perfcounters` and
   compare the MFMA-to-non-MFMA instruction ratio against HK's. This may
   reveal whether Triton's higher MfmaUtil comes from (a) fewer non-MFMA
   instructions, (b) a different MFMA op type, or (c) a different
   warp-level layout.
2. **8-wave → 4-wave WG re-layout**: drop the `8w` cooperative load to a
   `4w` variant; halve LDS staging slabs; recompute MFMA fragment
   distribution. Estimated 3-5 rounds. High risk of perf regression on
   DSV3 [watch] shapes.
3. **Wider MFMA op**: probe CDNA4 ISA for `mfma_16x16x64` or
   `mfma_32x32x32` variants that issue more FLOPs per cycle. If
   available, halves the # of MFMAs per K-iter.
