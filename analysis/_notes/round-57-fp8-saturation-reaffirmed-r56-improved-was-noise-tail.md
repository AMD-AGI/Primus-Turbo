# Round 57 — FP8 gpt_oss saturation reaffirmed; R56 `improved=True` was a noise-tail sample, not a code lift

## TL;DR

- R56 daemon metric returned **697** (label `improved=True`, new "best"
  697 vs prior 696) but R56 shipped **zero functional code change**:
  `git diff aa587ddc..36c14183 -- . ':!analysis/_notes/'` is empty.
  The +1 score is therefore by definition noise — specifically a
  +1σ upper-tail draw from the R56 noise model (30-sample median 695,
  σ 2.27 on this exact GPU pin / SHA).
- Working tree at HEAD = **bit-equivalent** to R55 (aa587ddc) for
  every code path the metric exercises. The "improvement" recorded
  by the daemon for R56 is a measurement artifact, not progress.
- All seven task-md NEW DIRECTIONS (A1, A3, B, C, D, E, F, G) are
  preflight- or post-implementation-FALSIFIED through R55. Macro
  flag space exhausted (R22-R28, R31b). Per-shape dispatcher
  predicates audited at single-shape granularity for all 8 metric
  cells (R23, R7, R3, R8, R12, R44, R45, R47, R50, R70). Stream-K
  scaffolding shipped HK-side at R12-R17 but production-NEUTRAL —
  the wave-pack PMC model in R52 closes 1/8 cells above the +25%
  ship gate.
- No new lever surfaced this round. R57 ships docs only. Daemon's
  next sample expected to land in [690, 698] per the R56 distribution.

## Why this is not a "wait one more round to see if it sticks" situation

R56's "best=697" was a single sample. The R56 30-sample empirical
distribution on this exact (GPU 3, SHA aa587ddc, code-bit-equivalent
to current HEAD) characterised:

```
range  [690, 698]
median 695
mean   694.3
σ      2.27
P(score ≥ 697 | no lift)  ≈ 18%   (~1σ tail)
P(score ≥ 700 | no lift)  ≈ 1.5%
P(score ≥ 702 | no lift)  ≈ 0.13% (3σ — single-sample detect threshold)
```

A single sample at 697 from this distribution is fully consistent with
the null hypothesis "no code change → no lift". The daemon's
`improved=True` flag fires on *strict-greater-than-prior-best*, which
turns the upper-tail of the noise band into spurious "best so far"
ratchets every ~5-6 rounds at this σ. This is mechanical, not
informative.

**Implication for the streak counter**: the daemon resets
`patience_streak` to 0 on R56 because of the noise tail, even though
no underlying code state changed. The patience-40 stop heuristic is
therefore not protecting against runaway saturation rounds — it gets
extended every time the noise band reaches into the upper tail.

## What the daemon should do (out-of-round-scope, recorded for the operator)

Per R56 §"R57+ forward-pointer", the two options remain:

1. **Multi-task pivot** — point the daemon at `_task_fp8_fused_act.md`
   (Path-A fused activation) which has open kernel-template work and
   a different score landscape. Highest EV; resets the saturation
   streak with real headroom.
2. **Wait out patience** — keep emitting docs commits in the noise
   band until the streak converges to 40. Lowest information; ~1.5
   GPU-hours of metric noise per remaining 26 rounds, and nothing
   ships.

Either is the operator's call; an in-round Claude cannot change the
daemon config (FROZEN list per SKILL.md).

## Why no probe this round

A probe requires a candidate hypothesis. There is none on the
remaining shortlist:

| Direction | Status @ R56 |
|---|---|
| A1 Stream-K work-stealing | Wave-pack PMC closes 1/8 cells (R52); HK scaffold shipped R12-R17 NEUTRAL |
| A2 SplitK (var-K wgrad)   | Subsumed by D-class blocker (var-K closed-form coord decode shipped R9 NEUTRAL) |
| A3 Decoupled-warps        | Pre-implementation FALSIFIED via existing CDNA4 PC paper data (-17 to -44% loss across 6 sizes) (R54) |
| B Cross-stream parallelism| Doesn't help kernel-only metric (verified R56 forward-pointer) |
| C KREM=64                 | Already shipped on rcr; var-K has no K-tail block (R50 a-priori falsified) |
| D SALU coord-decode       | BF16 var-K closed-form port shipped (HK b3a5c8db / round-9), production NEUTRAL on metric |
| E Barrier-scheme          | R26-R28 single-barrier-drops falsified; deeper restructure is multi-round, no candidate |
| F Larger tiles (256x384/512x256) | Register/LDS budget blocker per task-md §F; no headroom on production at 256 VGPR / 37 spill |
| G Cross-shape co-opt      | A-priori FALSIFIED — predicates already at single-shape granularity (R55) |

There is no eighth direction in the task-md or recent-rounds notes
that has not been touched. R57 honest verdict: nothing to probe.

## Files touched this round

- `analysis/_notes/round-57-fp8-saturation-reaffirmed-r56-improved-was-noise-tail.md` (this file)

No HipKittens change. No Primus-Turbo functional change.

## Forward pointer (R58)

If the daemon continues at `_task_gpt_oss_fp8_kernel.md`:

R58 default = identical to R57 = docs-only, citing R56 noise
characterization + this saturation re-affirmation. Repeat until
either (a) operator pivots the daemon task, or (b) patience-40
fires.

If the daemon is pivoted to `_task_fp8_fused_act.md` between R57
and R58, the entry point is the most recent dm-track note (e.g.
`round-56-fp8-grouped-C2-step2-still-deferred-fresh-chat.md` for
the parallel `_metric_grouped_only` track, which still has C-2
step-2 mechanical recipe open per R55-R56-dm forward pointers).

## Decision: NEUTRAL (docs-only, saturation re-affirmation)
