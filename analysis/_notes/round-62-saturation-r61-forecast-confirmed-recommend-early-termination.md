round-62-saturation-r61-forecast-confirmed-recommend-early-termination
=============================================================================

Round: 62 / 100
Date: 2026-05-10
Pre-SHA: 9ada4eed (R61 — saturation reaffirmed, A-G closure final)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R61 forecast: *"Same outcome will hold for R62-R100 unless an
out-of-scope lever opens up."* R62 confirms: no novel lever. NEUTRAL,
HEAD unchanged on both repos.

**Recommendation to operator**: terminate this run. Continuing to R100
will produce 38 more identical NEUTRAL rounds at fixed cost-per-round.
Score is pinned at noise band 695 +/- 2.27 with no in-scope lever
remaining; the daemon will not break that ceiling.

## Why this is round-62 not a duplicate of round-61

R57-R61 all docs-only NEUTRAL, all on same HEAD (b3a5c8db ship of D1
closed-form decode is the last functional change). R62 simply observes
that R61's forward-pointer played out as predicted: no audit gap opened,
no out-of-scope lever materialised between rounds. This note is
intentionally short to avoid recapitulating the R59/R60/R61 closure
tables verbatim --- those three docs already cover the exhaustion
inventory in detail.

## What an operator should do

1. Stop the daemon (kill the process pointed at by `auto_optimize_logs/
   gpt_oss_fp8_local_20260509_143917/run.pid` if still running).
2. To make further progress on the 695->900 score gap, expand scope
   beyond the daemon's frozen surface. Three concrete classes the
   daemon cannot reach:
   * Re-scope the metric (e.g., section-weighted, drop the saturated
     wgrad cells, or add rowwise/blockwise granularity coverage). Today
     the metric is FROZEN per `scripts/_task_gpt_oss_fp8_kernel.md`.
   * Hardware/firmware-class lever: a CDNA4 equivalent of warp-spec TMA
     would unblock A3 decoupled-warps, but is not a software change.
   * Build a new kernel template not yet in `kernel_fp8_layouts.cpp`
     (e.g., a producer-consumer scheme with persistent grid + per-warp
     async load + barrier-less inner loop). This was sketched in R52-R54
     work but blocked by the AGPR allocator alias bug for 4-wave at
     RBN=64; resolving it requires either an LLVM patch or rewriting
     the accumulator layout to 64 fp32/lane (open research direction,
     multi-week, not within a daemon round budget).

## Files touched this round

* `Primus-Turbo/analysis/_notes/round-62-saturation-r61-forecast-confirmed-recommend-early-termination.md`
  (this doc)

No code changes. HEAD unchanged on both repos.
