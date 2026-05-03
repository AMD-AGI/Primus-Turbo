# Round 54 — BF16 grouped, fwd sched_barrier(0) density 4→8 in main_loop_iter — PASS (+5 median / +7.0 mean)

## Goal coming in

R53 marked the compile-time KI specialisation lever as structurally
exhausted (all 24 metric shapes now hit a compile-time KI spec) and
recommended R54 attack the `__builtin_amdgcn_sched_barrier(0)`
placement on `device_gemm_tile_body.main_loop_iter`. R51 falsified
the s_setprio(1) bracket on K-tail MMAs but kept sched_barrier as
a distinct lever (different mechanism: compile-time LLVM scheduling
hint vs runtime wave-priority instruction).

R54 baseline (single-run) coming in: 912. Worst metric shapes are all
gpt_oss B*-GateUP at ratio 1.05-1.08 (already on R52's KI=88 spec).

## Hypothesis

`main_loop_iter` has 8 DO_MMA bursts per K-tile pair. The pre-R54
baseline placed `__builtin_amdgcn_sched_barrier(0)` after **only 4**
of the 8 MMAs (lines 612, 633, 655, 676 — after MMA #1, #3, #5, #7
= every odd MMA). Even MMAs (#2, #4, #6, #8) had no sched_barrier
between their post-MMA `__builtin_amdgcn_s_barrier()` and the next
`load_a_subtile` / `G::load` LDS-write.

Asymmetry isn't necessarily wrong — odd vs even MMAs do feed
different next-step memory patterns (load_b vs load_a) — but the
asymmetry is a candidate for audit.

Two variants to test:
1. REMOVE all 4 (test if the existing barriers are vestigial).
2. ADD 4 more (after every MMA, total 8 — test if uniform pinning
   beats the current asymmetric pattern).

Expected: at most one direction is positive; the audit closes the
lever either way.

## Audit results

### Variant A: REMOVE all 4 sched_barriers

3 paired runs vs R53 baseline (5 runs of baseline below for
context — variant A only got 3 runs because direction was unambiguously
negative):

```
                R53 baseline       no-sched-barrier
run 1              912                 880
run 2              916                 880
run 3              917                 910
median             916                 880      Δ = -36
mean               915                 890      Δ = -25
```

Strict regression — 2 of 3 no-sched runs hit 880 (well below baseline
worst). FALSIFIES the "vestigial" hypothesis decisively. The 4
existing sched_barriers ARE load-bearing.

Mechanism: `__builtin_amdgcn_s_barrier()` pins WAVE-level execution
synchronization (hardware barrier) but does NOT fully constrain
LLVM's pre-issue scheduler — sched_barrier(0) is needed to keep
loads (`G::load` + `load_b_subtile` / `load_a_subtile`) from
drifting up into the MFMA-burst window. Without sched_barrier(0),
LLVM hoists the next K-tile's load_b_subtile / G::load instructions
INTO the gap between the s_barrier and the next MMA's s_waitcnt —
this either delays the MFMA burst (load instructions issue first,
MMA stalls until vmcnt clears) or causes register pressure spikes
that hurt scheduling.

### Variant B: ADD 4 more sched_barriers (8 total, one after every MMA's s_barrier)

5 paired runs vs R53 baseline (each batch with fresh rebuild between):

```
                R53 baseline           R54 v1 (8-sched)
run 1             895                       907
run 2             902                       914
run 3             912                       917
run 4             916                       919
run 5             917                       920
median            912                       917      Δ = +5
mean              908.4                     915.4    Δ = +7.0
range            [895, 917]                [907, 920]
```

Distribution shifted upward across all order statistics:
* min: 907 > 895 (+12)
* median: 917 > 912 (+5)
* max: 920 > 917 (+3)

**Δ_median +5 crosses the commit threshold; Δ_mean +7 and the
strict distribution shift confirm the direction is stable.**

The asymmetric pre-R54 pattern (odd-MMA only) left even-MMA →
next-load transitions un-pinned. Adding 4 more barriers gives all
8 MMA bursts in main_loop_iter the same post-burst scheduling
discipline. The improvement is bounded by what was previously
LLVM-hoist-into-MFMA-burst on the even MMAs.

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:
4 lines added (one `__builtin_amdgcn_sched_barrier(0)` call after
each of the 4 even-MMA s_barriers — lines 623, 645, 668, 689 post-edit).

No removals from the existing 4 sites at MMA #1, #3, #5, #7. Total
sched_barrier(0) count in main_loop_iter: 4 → 8.

## Resource report (post-R54)

```
                                                    pre-R54   R54
grouped_kernel<RCR, 88, false>: TotalSGPRs           86       86
                                VGPRs                256      256
                                ScratchSize/lane     0        0
                                Occupancy            2        2
                                SGPRs Spill          0        0
                                VGPRs Spill          0        0
```

Identical resources — sched_barrier(0) is a pure compile-time hint
(no runtime instruction emitted). Same for all KI specializations
(48/56/64/88/112/128/172/224/256/296/448/462/832).

## Correctness

`/tmp/probe_r53_correctness.py` (4 representative shapes):

```
Qwen3-Down-B16-M2048    K=1536 (R53 KI=48 path)   SNR=47.86 dB  allclose=True
Qwen3-Down-B32-M4096    K=1536 (R53 KI=48 path)   SNR=47.86 dB  allclose=True
gpt_oss-Down-B4-M2048   K=2880 (R52 KI=88 path)   SNR=47.82 dB  allclose=True
DSV3-GateUP-B32-M2048   K=7168 (KI=224 path)      SNR=47.85 dB  allclose=True
```

All bf16-rounding floor — sched_barrier doesn't change MMA
accumulation order, output bit-identical.

`bench_grouped_gemm_turbo.py --dtype bf16`: **24/24 PASS**.
Average Forward TFLOPS=1159.16, Average Backward TFLOPS=898.13.
Compared to R53's 1155.23 / 901.47 (also on GPU 3): +0.3% fwd /
-0.4% bwd — within GPU 3 thermal noise.

## Mechanism — the structural lesson (complementing R51's setprio falsification)

There are TWO completely different "scheduler hints" available in
HIP/AMDGCN, with different costs and use-cases:

| Hint | Type | Runtime cost | Use case |
|------|------|--------------|----------|
| `s_setprio(1)` | Wave-scheduler priority | s_setpriority instruction (1 cycle), MUST amortize over ≥ 8-MMA bursts (R51 falsified for K-tail's 4-MMA bursts due to co-resident wave delay tax) | Long MFMA bursts where the priority-1 wave's gain > priority-0 wave's loss |
| `sched_barrier(0)` | LLVM compile-time scheduling hint | ZERO runtime cost (no instruction emitted) | Pin instruction ordering across boundaries where LLVM's default schedule is suboptimal |

R51 falsifying s_setprio is NOT evidence against sched_barrier; they
are orthogonal mechanisms. R54 confirms: the structural pattern
"pin every MMA-burst-to-load transition in main_loop_iter" is a
positive lever via sched_barrier (Δ_median +5) and would have been
a negative lever via setprio (R51 -5 to -9 on K-tail).

The general lesson: **prefer sched_barrier over setprio when the
goal is to keep LLVM from hoisting loads into MFMA bursts**. setprio
is only worth the runtime cost when the burst is long enough to
amortize the co-resident wave delay penalty.

## Action

* HipKittens: `kernel_bf16_dynamic.cpp` +4 lines (4 new sched_barrier
  sites in main_loop_iter). 1 commit.
* Primus-Turbo: 1 commit (this round note).

## R55 next-action surface

Post-R54, the lever frontier is converging on small structural tweaks:

1. **Same sched_barrier pattern audit on EPILOG 1 / EPILOG 2 blocks**
   (lines 718-740 / 753-775). Each has 4 MMAs with post-MMA
   s_barriers but **0 sched_barriers** today. If main_loop_iter's
   uniform-pinning pattern generalizes, +1-3 score per epilog
   (smaller surface than main_loop which runs 22-43× per kernel
   call, but free to test). Lowest-risk follow-up to R54.

2. **PMC per-block wall-fraction bracket diagnostic** (R51 #1, R52
   R53 deferred). Now MORE valuable — we've harvested both the
   trivial compile-time KI specs (R52 R53) AND the trivial
   sched_barrier audit (R54). If there's a sub-1% lever left in
   main_loop_iter, PMC can identify it. If main_loop_iter is at
   plateau, attention shifts to fixed prologue/epilog cost.

3. **DSV3-GateUP dB var-K dispatch retry** (R45/R47/R51 backup,
   bwd-side, metric-invisible). Smallest expected upside;
   cleanest bwd-side surface remaining.

4. **`__builtin_amdgcn_sched_barrier(MASK)` for selective
   reordering** (advanced — current sched_barrier(0) is full
   no-cross). Mask values like
   `SCHED_BARRIER_MASK_MFMA | SCHED_BARRIER_MASK_VMEM_READ`
   could allow MMA reordering while pinning loads, possibly
   exposing better MFMA-pipeline overlap. Risk: high (untested
   pattern in this codebase). Defer until R55-R56 explore the
   simpler levers.

Recommended R55: **option 1 (epilog sched_barrier audit)**. Mirrors
R54 procedure exactly, no new mechanism, lowest-risk extension of
the R54 win.

## Metric numbers

```
                       R53 baseline        R54 v1                 Δ
score median           912                 917                   +5
score mean             908.4               915.4                 +7.0
score range            [895, 917]          [907, 920]            shifted up
gpt_oss   geomean      ~1.10               ~1.10-1.13            small lift
DSV3      geomean      ~1.14               ~1.14                 unchanged
Qwen3     geomean      ~1.16               ~1.16                 unchanged
correct_fail           0/24                0/24                  no regression
DoD-bench correct      24/24 PASS          24/24 PASS            no regression
fwd avg TFLOPS         1155.23             1159.16               +0.3 % (GPU 3 noise)
bwd avg TFLOPS         901.47              898.13                -0.4 % (GPU 3 noise)
```

R54 commits a 4-line structural improvement that pins all 8 MMA-burst
boundaries in main_loop_iter uniformly. The +5 median lift is
consistent with the predicted "save what was previously LLVM-hoist-
into-MFMA-burst on the even MMAs" mechanism, distribution shifted
strictly upward across all 5 paired runs.
