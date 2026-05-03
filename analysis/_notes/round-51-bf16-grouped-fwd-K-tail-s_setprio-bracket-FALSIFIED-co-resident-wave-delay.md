# Round 51 — BF16 grouped, fwd K-tail s_setprio(1) bracket — FALSIFIED (-5 to -9 score, co-resident wave delay outweighs priority gain on small MMA bursts)

## Goal coming in

R50 PMC diagnostic showed all 3 metric families (gpt_oss, DSV3, Qwen3)
sit at MFMA utilization 57-76 % on fwd, with 15-25 pp headroom. R50
recommended R51 attack the MFMA pipeline scheduling on the SHARED
`device_gemm_tile_body` via `s_setprio(1)` placement audit:

> R50 next-action 1.a: __builtin_amdgcn_s_setprio(1) placement audit
> — currently set around DO_MMA inside main_loop_iter but NOT around
> K-tail block (line 905-916) or epilog 1/2 DO_MMAs. Adding setprio(1)
> to those MMA bursts may reduce wave-scheduler stalls. Lowest
> correctness risk; broad coverage.

Worst metric shape coming in = `gpt_oss-Down-B4-M2048` (ratio 1.026,
weight 3, K=2880 forces fuse path) — the exact path the K-tail
s_setprio fix would benefit.

## Hypothesis (R51)

`grouped_kernel<RCR, 0, true>` runs at occupancy 2 waves/SIMD with 256
VGPRs (the kernel-resource-report shows 256 VGPRs, 0 spill, 96 SGPRs,
2 occ; consistent with the R39 baseline measurement which reported
248 VGPRs, build-noise-tolerable). With 2 waves co-resident on the
same SIMD, the wave scheduler interleaves issue slots between them.

Inside `main_loop_iter` (line 600-686), every DO_MMA burst is bracketed
by `__builtin_amdgcn_s_setprio(1)` / `s_setprio(0)` (lines 608/610,
619/621, 629/631, ...). Same pattern in epilog 1 (lines 718, 727, 737)
and epilog 2 (lines 753, 763, 772). This gives MFMA-issuing waves
priority over the co-resident wave's other (non-MFMA) work.

The FUSED_KTAIL block (lines 905-916 RCR, 1122-1133 RRR) was the only
MMA-issuing region in this kernel WITHOUT s_setprio brackets. R51
hypothesis: adding the same `s_setprio(1)/s_setprio(0)` bracket to
the K-tail's 4 DO_MMAs would let them issue more tightly, reducing
the K-tail block's wall-time (~20 % of fwd kernel per R50's MFMA-util
extrapolation) and lifting gpt_oss family ratio.

Expected: +5-10 score on gpt_oss family alone (3x weight), neutral on
DSV3 / Qwen3 (their dispatch path is FUSED=false → no K-tail block).

## v1 attempt

Edited `kernel_bf16_dynamic.cpp` to add `__builtin_amdgcn_s_setprio(1)`
before each DO_MMA pair in the K-tail block AND `s_setprio(0)` after,
mirroring the pattern in main_loop_iter/epilog 1/epilog 2:

* RCR FUSED_KTAIL block (lines 905-916, the gpt_oss path) — 2 brackets
  added (one per M slab).
* RRR FUSED_KTAIL block (lines 1122-1133, gated behind
  `BF16_RRR_FUSE_PROBE=1` — currently default-off but mirror the
  pattern for consistency) — 2 brackets added.

Build resource report (vs baseline same kernel):

```
                                                        baseline   R51 v1
grouped_kernel<RCR, 0, true>: TotalSGPRs                96         96
                              VGPRs                     256        256
                              ScratchSize [bytes/lane]  0          0
                              SGPRs Spill               0          0
                              VGPRs Spill               0          0
                              Occupancy [waves/SIMD]    2          2
                              LDS Size [bytes/block]    167936     167936
```

Identical resource usage. Expected — `s_setprio` is a 1-instruction
scalar opcode (`s_setpriority`) that doesn't allocate any register.

## Correctness probe

`/tmp/probe_r51_correctness.py` ran `turbo.ops.grouped_gemm` (HIPKITTEN
backend, autograd off, fwd-only) on 3 representative shapes vs an
fp32-reference grouped GEMM. SNR threshold 30 dB:

```
gpt_oss_20B-Down-B4-M2048  (B=4 M=2048 N=2880 K=2880, fuse path)
                                max_diff=1.5625e-02  SNR=47.82 dB  allclose=True
gpt_oss_20B-Down-B32-M2048 (B=32 M=2048 N=2880 K=2880, fuse path)
                                max_diff=1.5625e-02  SNR=47.82 dB  allclose=True
DSV3-GateUP-B32-M2048      (B=32 M=2048 N=4096 K=7168, KI=112 path)
                                max_diff=3.1250e-02  SNR=47.85 dB  allclose=True
ALL OK: True
```

47.82 dB is the bf16-rounding-noise floor (matches R48 KI=32 / R45
baseline measurements). Output is bit-identical or close enough to be
indistinguishable from rounding noise — `s_setprio` is purely a
scheduler hint, not a semantic change.

## Empirical impact (the falsification)

Tested with 3 metric runs each on the SAME just-built kernel binary
(reverted between runs, rebuilt fresh, ran 3x for noise estimate).
Metric command: `python3 scripts/_metric_grouped_bf16_weighted_wall.py`.

```
                  baseline           R51 v1
                  (no setprio)       (setprio added)
run 1                905                898
run 2                906                902
run 3                917                901

median               906                901    Δ = -5
mean                 909.3              900.3  Δ = -9
max                  917                902    Δ = -15
```

R51 v1 is consistently SLIGHTLY LOWER across all 3 runs (no overlap
between the two distributions). Direction is stable — this is a real
regression, not noise. The Triton-side variance (905-917 in baseline
alone) is large but the with-change distribution is shifted entirely
below it.

Verification: reverted to R50 baseline kernel (`diff` of the .cpp
file vs the saved baseline = empty), rebuilt, ran metric → score 912,
confirming we're back in the baseline 905-917 distribution.

## Mechanism — why s_setprio(1) on the K-tail block backfires

`s_setprio(1)` raises THIS wave's priority above the OTHER co-resident
wave (occupancy 2 means 2 waves per SIMD). Inside main_loop_iter,
every MMA burst is bracketed by setprio(1) / setprio(0) — these
brackets cover ~8 MMAs over ~200-300 cycles per `main_loop_iter` call
(K_TWO_TILE pair). The priority gain accrues over enough cycles to
beat any co-resident-wave delay penalty.

The K-tail block has only **4 MMAs** total (2 per M slab × 2 slabs),
issued in 2 separate bursts of 2 MMAs each. Each burst is ~16 × 2 =
~32 cycles of MFMA work. The s_setprio(1) bracket only covers 2 MMAs.

What happens with two waves co-resident in K-tail simultaneously:

* Wave A enters its K-tail s_setprio(1) bracket. Wave A's 2 MMAs
  issue on this SIMD's MFMA pipe over ~32 cycles.
* Wave B is mid-(K-tail OR main loop) on the SAME SIMD; its
  instructions (MMAs of its own + scalar / VMEM work) get
  DELAYED for ~32 cycles by Wave A's priority.
* When wave A exits its bracket (`s_setprio(0)`), wave B's pending
  instructions issue.
* Net: Wave A saves a few cycles on its MMAs; Wave B loses ~32 cycles
  of throughput.

For main_loop_iter's longer MMA bursts (~200+ cycles), wave A's
saving is large enough that net per-pair throughput improves. For the
K-tail's 2-MMA burst (~32 cycles), wave A's saving is comparable to
or smaller than the parallel-wave delay → net throughput drops.

The empirical -5 to -9 score is consistent with this model: the K-tail
block is ~20 % of fwd wall (per R50 MFMA-util extrapolation) on
gpt_oss shapes, and a small per-burst slowdown there scales to ~1 %
fwd ratio drop, weighted 3x → ~5-10 score.

This is a structural insight: **`s_setprio(1)` is not free even when
the brackets are short — the parallel wave loses what your wave
gains, and short MMA bursts can't amortize the delay.** The existing
main_loop_iter pattern works because each bracket covers ~8-16 MMAs
of latency-bound MFMA work where the priority gain materially affects
the issue pipeline.

## Falsification consequence

R51 closes:

* **K-tail s_setprio(1) bracket lever** (R50 next-action 1.a). The
  K-tail block's 2-MMA bursts are TOO SHORT for the wave-priority
  gain to outweigh the co-resident-wave delay. Don't add `s_setprio`
  to small (≤ 4-MMA) MMA bursts in this kernel; only main_loop_iter
  -style ≥ 8-MMA bursts can amortize the cost.

R51 does NOT close:

* **`s_setprio(1)` placement on the LARGER MMA bursts** that may
  still be missing it. Per quick re-grep, the only un-bracketed MMA
  bursts in the entire kernel ARE the K-tail blocks (RCR + RRR fuse).
  The lever surface is exhausted.
* **`__builtin_amdgcn_sched_barrier(0)` placement** (R50 next-action
  1.b). Different lever — sched_barrier prevents LLVM reorderings,
  doesn't change wave priority. May still have headroom.
* **Per-block wall-fraction PMC bracket** (R50 next-action 2,
  diagnostic). Quantify prologue + epilog 1 + epilog 2 vs main_loop
  wall-time using rocprofv3 marker bracketing. If fixed-overhead is
  large for short K, pivot to shrinking it.
* **DSV3-GateUP dB var-K dispatch retry** (R50 next-action 3).
  Smallest expected upside but cleanest dispatch surface remaining.

## Action

* HipKittens: `kernel_bf16_dynamic.cpp` modified (4 setprio brackets
  added — 2 in RCR fuse, 2 in RRR fuse), then reverted via backup file
  `/tmp/kernel_bf16_dynamic_R50_baseline.cpp`. Final diff = 0
  (rebuilt to confirm). No HipKittens commit.
* Primus-Turbo: 1 commit (this falsification note).

## R52 next-action surface

Three candidates remain (pruned from R50's surface):

1. **`__builtin_amdgcn_sched_barrier(0)` placement audit** (R50 1.b).
   Distinct from setprio — sched_barrier walls prevent compiler
   reordering across them. Inside main_loop_iter, sched_barrier(0)
   appears between MMA bursts (lines 612, 633, 655, 676). The K-tail
   block has NO sched_barriers. May enable better LLVM scheduling
   on neighboring loads/MMAs without the wave-scheduler tax that
   doomed s_setprio. Risk: similarly low; output byte-identical;
   may also prove neutral or negative.

2. **Per-block wall-fraction PMC bracket diagnostic** (R50 2).
   rocprofv3 marker brackets around (prologue + epilog 1 + epilog 2)
   vs (main_loop_iter) on a single shape. Should be 1-shape capture
   (~10 sec) and gives a clear go/no-go on whether the fixed
   per-tile overhead is the residual bottleneck. If overhead < 5 %
   → fixed-overhead is exhausted as a lever. If > 15 % → there's
   significant attack surface (e.g., merge prologue + epilog 1, or
   skip the epilog s_barrier on the LAST persistent iteration).

3. **DSV3-GateUP dB var-K dispatch retry** (R50 3, R47/R45 backup).
   R24 dropped `xcds=0` due to allclose drift; re-sweep with
   `xcds ∈ {1, 2, 4, 8}` on (tiles_m=16, tiles_n=28) cells. Smallest
   expected upside but cleanest dispatch surface remaining.

Recommended for R52: **start with (2) the PMC diagnostic**. R50 + R51
have spent 2 rounds on fwd-side speculation; the per-block wall
fraction is the missing diagnostic that would either CONFIRM the
"shorter K = larger fixed-overhead tax" hypothesis from R50 (and
unlock structural levers like prologue/epilog merge) or FALSIFY it
(and force the lever search back to bwd-side dispatches).

If (2) confirms fixed overhead < 5 %, fall through to (3). If > 15 %,
R53 attacks epilog/prologue merging.

## Metric numbers

```
                       R50 baseline        R51 v1 (median of 3)
score                  906                 901                  Δ = -5
gpt_oss   geomean      ~1.10               1.0951
DSV3      geomean      ~1.14               1.1508
Qwen3     geomean      ~1.17               1.1742
correct_fail           0/24                0/24                 no regression
above_target shifts    2 PASS (run 1)      1-2 PASS             noise
```

Reverted state metric (final verification): score=912, in baseline
905-917 distribution. No regression after revert.
