# Round 14 — gpt_oss focus: 3 HipKittens micro-knob experiments FALSIFIED + metric ceiling calibration

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**HEAD before**: `fd9580fd`
**HEAD after**: this commit (Primus-Turbo notes-only; kernel/config bytes unchanged)
**HipKittens commit**: `d778c3d0` (notes-only — see HK
`analysis/_notes/round-8-grouped-sched-barrier-removal-falsified.md`)
**Metric**: 794 (round-13 best) → 794 (no functional change).

## Why this note

Patience counter at 9/30 with no metric improvement. Round 13 documented
config-tuning saturation; this round investigated **HipKittens kernel
micro-knobs** (sched_barrier compile-time hints + `s_waitcnt vmcnt`
threshold) as the next-most-likely source of free score points. All
three experiments came back neutral or slightly negative. This note
captures the **calibration finding** that's more important than any
single experimental result.

## TL;DR — three HK kernel experiments, all FALSIFIED

| # | experiment                               | n_runs | mean Δ vs base | min Δ | verdict |
|---|------------------------------------------|--------|----------------|-------|---------|
| 1 | BF16 grouped main_loop_iter `sched_barrier(0)` removal (gated `!FUSED_KTAIL`) | 8 | -1.0 | -3 | revert  |
| 2 | FP8 grouped Epilog 1+2 `RCR_SCHED_BARRIER()` removal (3 hints/tile)           | 8 | -0.1 |  -1 | revert  |
| 3 | FP8 `RCR_STEADY_VMCNT` knob: 8 → 4 (tighter)                                  | 10 | -0.3 |  -1 | revert  |
|   | FP8 `RCR_STEADY_VMCNT` knob: 8 → 12 (looser)                                  | 5  | -0.8 |  -1 | revert  |

Mechanism analysis is in HipKittens `analysis/_notes/round-8-grouped-
sched-barrier-removal-falsified.md`; this note focuses on the
**Primus-Turbo-side metric calibration** that came out of those
experiments.

## CALIBRATION: `auto_optimize.py` "best=794" under-counts true ceiling

The `auto_optimize.py` round-13 final report logs `best=794` from a
single-run-per-round metric history. This round measured the
**baseline distribution** with 10 metric runs on the unmodified HK
binaries:

```
RCR_STEADY_VMCNT=8 (BASELINE):
  scores = [792, 797, 794, 794, 795, 793, 794, 794, 795, 794]
  mean   = 794.2
  median = 794
  stdev  = 1.32
  min    = 792
  max    = 797
```

So:

* **The true round-13 metric mean is 794.2**, not 794.
* **The true ceiling is 797**, +3 points above what `auto_optimize` logs.
* **The 1-sigma noise band is ~±1.3 score points**.
* **Discriminating power for a single round-vs-round comparison: ±2 points
  (95% CI on the mean of 1 sample is ±2.6, but median compares fine
  inside ±1.3).**

This means:

* Single-round metric deltas of `+1` (the kind round-9..13 shipped) are
  inside the noise band for an N=1 baseline. The wins were real *only*
  because each came with mechanism evidence (gm/xcd tight-verify TF
  delta, host-overhead microsecond breakdown) that survived a 5-run
  re-test.
* Future micro-knob experiments need **≥10 runs each side** and
  **≥1.5 pp median shift** to claim a real win, OR mechanism evidence
  with matching kernel-level instrumentation.
* The plateau at "score ~794" since round 11 does NOT mean the work is
  stuck — it means we're **measurement-limited at single-run scale**.

## Why the "FP8 round-3/4 mechanism → BF16" extrapolation keeps failing

This round was the second independent failure of "port FP8 micro-win
to BF16" attempts (round-7 was the first, round-8 / this round is the
second):

* **Round-7 (BF16 K-tail single-wait port)**: -11 score, +5 SGPR spill
  + 1 VGPR spill cascade. Mechanism: BF16 grouped is operating at a
  different point on the register-pressure × scheduler-freedom surface
  than FP8 grouped.
* **Round-8 / this round (BF16 main-loop sched_barrier removal)**:
  -1 mean, -3 min. Mechanism: BF16's `main_loop_iter` already has
  tighter operand dependencies (4 `s_waitcnt lgkmcnt(0)` + 1
  `TK_WAIT_LGKM` per 2-K-iter call) than FP8's 1-2-per-K-iter pattern,
  leaving the back-end machine scheduler less reorder freedom even
  with the compile-time hint removed.

**Pattern**: FP8 micro-wins exploit FP8's looser MMA-around-wait
structure; BF16's tighter structure already constrains the scheduler
to a near-optimal order. The two kernels have different limiting
resources (BF16 ≈ register / scheduler, FP8 ≈ memory / scheduler).
Mechanical port from FP8 → BF16 is **no longer a viable strategy**
for the remaining ~30 pp gap to target=1.20.

## What's NOT yet falsified for round 15+ (priority order)

Strict ranking by hypothesis weight × cost (best ROI first):

### 1. K-tail amortize (M-dim multi-tile per persistent wg) — high ROI, high cost

Round-30 start guidance ④. A persistent wg processes multiple BM-rows
of one K-tile, sharing one K-tail epilog across them. Reduces K-tail
relative cost from 2-4 % → 0.5-1 %. Requires kernel structural change
(shared accumulator per-row, single end-of-tile K-tail call). Risk:
VGPR pressure (multiple BM-rows × MMA = +N VGPRs/wg). Estimated
2-3 rounds of work; estimated +1-3 pp on K-misaligned shapes (8/16
gpt_oss shapes have K=2880).

### 2. N=2880 / N=5760 BN=128 sweep — medium ROI, medium cost

Round-30 start guidance ②. Default BN=256 leaves the last column of
N-tiles at 25 % utilization (64-col-active × 256-col-tile). BN=128
doubles that to 50 % but doubles grid size. **Round-30 start said
Triton uses BN=256 too, but that data point is from before the round
11/12/13 host-overhead trim that bumped overall ratio +1-2 pp on every
shape**. Worth a fresh measurement. Estimated 1-2 rounds of HK kernel
work; estimated +0.5-2 pp on N=2880 shapes (4/16 gpt_oss).

### 3. Per-shape XCD-pinning sweep on B=32 unbound shapes — small ROI, low cost

Typically +0.1-0.3 pp per shape. Only worth doing in batch (sweep all
B=32 shapes once, write rules into `select_default_config` once).

### 4. FP8 Down-layer rocprof breakdown — calibration

The FP8 grouped focus shapes show absolute TFLOPS far below Triton
(e.g., FP8-Down-B4-M4096 985 TF vs Triton 1183 TF, 16.7 % gap). Worth
running rocprof on this case to identify whether the gap is HBM
bandwidth, MFMA issue rate, or LDS bank conflicts before committing
to a structural change.

## Files / commits

* HipKittens: `d778c3d0` —
  `analysis/_notes/round-8-grouped-sched-barrier-removal-falsified.md`
  (3 micro-knob falsifications + ceiling calibration).
* Primus-Turbo: this commit —
  `analysis/_notes/round-14-micro-knob-sweep-falsified.md` (this
  file).

Self-bench: not required (no backward path touched, no kernel
edited, only documentation).
