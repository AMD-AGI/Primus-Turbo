# Round-4 — `RCR_KTAIL_VMCNT=16` FALSIFIED at metric (tied median, 10×10 samples)

## TL;DR

The task md (`scripts/_task_gpt_oss_fp8_kernel.md`, "ONLY MARGINAL WIN STILL
AVAILABLE FROM MACROS" section) recommended shipping `RCR_KTAIL_VMCNT=16`
based on a prior 9-sample observation of "median 697 vs baseline 696 = +1
sub-threshold (95% CI ±1.45) … zero-risk to ship". A 10-sample-each
re-test on this run's binding (HK working tree at default `vmcnt=8` vs
flipped to `16`, `dbg_remote.sh` rebuild on remote MI355X
`HIP_VISIBLE_DEVICES=3`) reproduces the **tied** median outcome but at
zero net delta — the "+1" claim does not hold up beyond noise.

```
vmcnt=8  (baseline, 10 samples, sorted): 693 693 693 694 695 696 696 697 697 698
vmcnt=16 (candidate, 10 samples, sorted): 692 693 694 695 695 696 697 697 698 699

  median:  695.5  vs  695.5     (Δ = 0.0)
  mean:    695.2  vs  695.6     (Δ = +0.4, within 95% CI ±1.45)
  range:   693-698 vs 692-699   (essentially identical envelopes)
```

Per the multi-sample falsification protocol (≥9 samples per cell), the
two distributions are statistically indistinguishable: the +0.4 mean
shift is ~22% of the noise CI; the median ties exactly; the cumulative
distribution functions interleave (5 of 10 vmcnt=16 samples ≤ vmcnt=8
median, 5 ≥). **The "ship vmcnt=16 zero-risk" recommendation in the
task md does not move the metric.**

## Why this matters relative to R31b

R31b's per-cell evidence (`round-31b-rcr-ktail-vmcnt-falsified.md`)
sweep showed:

```
Cell                         vmcnt=4    vmcnt=8 (default)   vmcnt=12
Down_B4_M2048_fwd            1485.5 T   1543.5 T            1485.5 T   (-3.8%)
Down_B4_M4096_fwd            1972.9 T   2013.2 T            1970.1 T
GateUP_B4_M2048_fwd          1887.4 T   1931.4 T            1884.8 T
```

R31b's mechanistic theory: **LLVM inserts an implicit additional
`s_waitcnt` before the mfma whenever the user-hint `vmcnt(N)` is too
loose to satisfy the data dep on `a`'s VGPRs**. cA reads `a`'s 8
elements; `a` is the LAST 8 of 24 in-flight loads → vmcnt(8) is the
minimum that drains all of `a`. Any value > 8 (= 12, 16) would expose
the read-before-write hazard, forcing LLVM's schedule fallback. The
prediction was that vmcnt=16 would regress identically to vmcnt=12
(i.e., -3.8 % on Down_B4_M2048 fwd).

This round's metric-level result is consistent with R31b at the
section-mean level: **the per-cell -3.8 % regression on Down_B4 fwd
(if it occurs at vmcnt=16) is masked by tied or marginal gains
elsewhere when averaged across 8 shapes × 3 sections** — net 0
median, +0.4 mean.

Either:
1. LLVM's fallback at vmcnt=16 happens to land on a different
   schedule than at vmcnt=12 (e.g., the looser hint triggers a
   no-op/identity transform rather than the conservative serialised
   wait), so Down_B4 fwd is NOT regressed by -3.8 % at vmcnt=16, OR
2. Down_B4 fwd IS regressed but the section-mean dilution + other
   shapes' offsetting movement keeps the metric flat.

Case (1) would require a per-cell A/B at vmcnt=16 (mirror of R31b's
methodology but with the missing 4th column) to confirm. Not pursued
this round — the metric-level decision is already final: NEUTRAL,
nothing to ship.

## What was changed this round

* No code changes shipped. HK working tree restored to
  `RCR_KTAIL_VMCNT 8` (the R31b-default value).
* No probe scripts added (the `dbg_remote.sh` for-loop calling the
  canonical `_metric_gpt_oss_fp8_kernel.py` was the entire experiment;
  results captured in this note).

## Files touched

* `analysis/_notes/round-4-rcr-ktail-vmcnt-16-FALSIFIED-tied-median-at-10-samples.md`
  (this doc).
* No HK changes (vmcnt flipped to 16, measured, flipped back to 8 —
  net diff = 0).

## Forward pointer for round-5

The task md's "ONLY MARGINAL WIN STILL AVAILABLE FROM MACROS" section
is now closed: vmcnt=16 falsified at metric level (this round),
vmcnt=4/12 falsified at per-cell level (R31b). The `RCR_KTAIL_VMCNT`
macro is permanently retained at default=8 as zero-cost scaffolding
but no value moves the metric.

The remaining un-attempted directions per the task md (NEW DIRECTIONS
A-G, with D explicitly closed in R48 already):

| Direction | Status              | Effort |
|---|---|---|
| A1 Stream-K (RCR fwd/dgrad) | un-attempted | 4-6 rounds |
| A2 SplitK (var-K wgrad)     | un-attempted | 2-3 rounds |
| A3 Decoupled-warps          | un-attempted | 4-6 rounds |
| B Cross-stream (PT-side)    | needs metric-script audit first | 1 round preflight |
| C Activation cache reuse    | metric pre-quantizes → 0 EV on score | closed by inspection |
| D SALU coord-decode (var-K) | CLOSED in R48 (EV ≤+1) | — |
| E Per-warp-group barriers   | un-attempted | 3-5 rounds |
| F Larger tiles (256x384 etc.) | un-attempted; 256x128 was -7..23% (FORBIDDEN) | 2-3 rounds |
| G Cross-shape co-optimization | un-attempted (per-shape rules tuned to local optima only) | 1-2 rounds |

Round-5 recommended priority:

1. **B preflight (1 round)** — read `_metric_gpt_oss_fp8_kernel.py`
   timing structure to determine whether dgrad and wgrad are timed on
   separate streams or with explicit syncs. If serial, the autograd
   `Function.backward` already returns a single (grad_a, grad_b) tuple
   — overlapping the two `grouped_gemm_fp8_*_impl` calls inside the
   backward via `torch.cuda.Stream` could give near-2× wall on bwd
   sections IF the metric measures wall-clock between the autograd
   call and the next sync. **Cheap to determine; high upside if
   metric supports it.** Should be done before committing to A1
   (multi-round Stream-K port).
2. **G cross-shape (1-2 rounds)** — sweep one cell's dispatcher
   neighbour values that have been falsified per-cell, and measure
   the 8-shape metric impact. The per-cell falsification protocol
   uses the cell's own SNR/TFLOPS as the gate; it does not detect
   "this cell loses 0.5 % but its sibling gains 1.5 %" patterns.
   Pick a candidate from R44/R45-falsified (gm-xcd drift) — those
   were closed at single-cell granularity, never at 8-shape.
3. **A1 Stream-K (4-6 rounds)** — only if 1 & 2 fail to deliver.

Avoid re-testing the macro-flag axis (now triple-closed: R8/R9,
R13, R31b, R4-this-round) and the dispatcher axis on Down_B4 cells
(exhausted across R1-R4, R10-R13, R34-R45, R52). All fresh budget
should go to architectural work.
