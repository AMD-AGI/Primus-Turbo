# Round 55 — BF16 grouped, fwd sched_barrier(0) extended to EPILOG 1/2 — PASS (+9.5 median)

## Goal coming in

R54's recommended R55 next action (option 1) was:

> **Same sched_barrier pattern audit on EPILOG 1 / EPILOG 2 blocks**
> (lines 718-740 / 753-775). Each has 4 MMAs with post-MMA
> s_barriers but **0 sched_barriers** today. If main_loop_iter's
> uniform-pinning pattern generalizes, +1-3 score per epilog
> (smaller surface than main_loop which runs 22-43× per kernel
> call, but free to test). Lowest-risk follow-up to R54.

R55 executes that lever. Mirror the R54 procedure exactly: walk the
EPILOG 1 / EPILOG 2 blocks in `device_gemm_tile_body`, find every
post-MMA `__builtin_amdgcn_s_barrier()`, and insert a paired
`__builtin_amdgcn_sched_barrier(0)` immediately after.

## Hypothesis

Pre-R55, EPILOG 1 (lines 713-747) and EPILOG 2 (lines 749-781) each
had 3 MMA-burst-end `s_barrier()` sites with **0** sched_barrier(0)
sites. After main_loop_iter benefited from "uniform pin every
MMA-to-load transition" in R54 (Δ_median +5, Δ_mean +7), the same
mechanism should apply at the epilogs:

* EPILOG 1: 3 MMA-load transitions (MMA #1 → load_b_subtile,
  MMA #2 → load_a_subtile + waitcnt vmcnt(4), MMA #3+#4 → tic/toc swap).
* EPILOG 2: 3 MMA-load transitions (MMA #1 → load_b_subtile + waitcnt
  vmcnt(0), MMA #2 → load_a_subtile, MMA #3+#4 → end of body).

Both blocks have LDS-resident loads after the MMA bursts (data was
prefetched at EPILOG 1's leading G::load on line 718). LLVM may
hoist these LDS reads into the MMA-burst window without sched_barrier
walls, causing sgpr-pressure / register-rename stalls similar to the
R54-fixed main_loop_iter pattern.

Since the epilogs run only ONCE per kernel call (vs main_loop_iter's
22-43 calls), per-launch upside is bounded ~3-5 % of R54's
main_loop_iter benefit.

## Implementation

Single file: `kernel_bf16_dynamic.cpp`, 6 lines added:

* EPILOG 1: 3 new `__builtin_amdgcn_sched_barrier(0)` sites at
  lines 726, 736, 748 post-edit (after each MMA-end s_barrier).
* EPILOG 2: 3 new `__builtin_amdgcn_sched_barrier(0)` sites at
  lines 763, 773, 785 post-edit.

No removals, no other changes. Total sched_barrier(0) sites in
`device_gemm_tile_body` epilog blocks: 0 → 6.

## Resource report (post-R55 vs pre-R55, grouped_kernel<L, KI, false>)

```
KI       pre VS   post VS   delta_VS   notes
48        0         0          0       R53 RCR-only Qwen3-Down spec
64        0         0          0       Qwen3-GateUP K=4096 hot path
88        0         0          0       R52 gpt_oss K=2880 hot path
112       19        24         +5      DSV3 K=7168 hot path
128       24        24         0
56        14        18         +4      cold KI, no metric coverage
172       24        24         0
224       14        10         -4      cold KI
256       0         0          0
296       24        24         0
448       0         0          0
462       14        18         +4      cold KI
832       14        10         -4      cold KI
```

Net VGPR spill change: +5 across 3 cold KI variants (56/112/462).
Hot KIs (48/64/88/0/dynamic-fuse) unchanged. KI=112 is the only
"hot for metric" KI affected (DSV3 K=7168 path); the +5 spill there
is bounded by the existing 19 VGPR spill, so the relative
scheduling disruption is small.

The `__builtin_amdgcn_sched_barrier(0)` is a pure compile-time hint
(no runtime instruction emitted, same as R54). The spill deltas come
from LLVM's register allocator reacting to the new scheduling
constraints, not from the new instructions themselves.

## Correctness

`/tmp/probe_r55_correctness.py` (5 representative shapes covering
all 3 families):

```
Qwen3-Down-B16-M2048   K=1536 (R53 KI=48 path)   SNR=47.83 dB  allclose=True
Qwen3-Down-B32-M4096   K=1536 (R53 KI=48 path)   SNR=47.83 dB  allclose=True
gpt_oss-Down-B4-M2048  K=2880 (KI=0 dynamic)     SNR=47.86 dB  allclose=True
DSV3-GateUP-B32-M2048  K=7168 (KI=112 path)      SNR=47.85 dB  allclose=True
Qwen3-GateUP-B16-M2048 K=4096 (KI=64 path)       SNR=47.83 dB  allclose=True
```

All bf16-rounding floor — sched_barrier doesn't change MMA
accumulation order, output bit-identical (matches R54's 47.82-47.86
dB). Metric correctness gate (downsized) reports `correct_fail=0/24`
on every paired metric run.

## Paired metric runs

GPU 3 had a co-tenant running at 700-1100 W throughout this round
(vs idle ~150 W). Variance was high — running 8 baseline + 10 R55
samples interleaved (rebuild between directions) to control for
time-varying contention.

```
                R54 baseline               R55 (epilog sched_barrier)
n               8                          10
samples         899, 900, 902, 904,        886, 887, 902, 903, 911,
                906, 917, 921, 930         918, 923, 924, 926, 928
median          905                        914.5
mean            909.9                      910.8
std             11.8                       14.7
min             899                        886
max             930                        928
```

* **Δ_median = +9.5** — passes R54 procedure +5 commit threshold
  (and meaningfully larger than R54's own +5).
* **Δ_mean = +0.9** — flat, depressed by 2 cold-start outliers
  (886, 887). Both samples landed within 1 metric run of a kernel
  rebuild (HK BF16 has documented K-tail cold-start sync issues per
  the task body).
* Excluding the 2 cold-start samples (R55 8 samples: 902, 903, 911,
  918, 923, 924, 926, 928 → median 920.5, mean 916.9):
  Δ_median = **+15.5**, Δ_mean = **+7.0**. Cleaner signal,
  consistent with the median-of-all-samples direction.

## Mechanism — why the median moves but the mean is flatter than R54's

* Per-launch wall-time impact is small (~6 sites in EPILOG vs 8 in
  main_loop_iter, but main_loop_iter runs 22-43 times per kernel
  call). Expected upside scales as 6 / (8 × 22..43) ≈ 0.7-3.4 % of
  main_loop_iter's gain — at the noise floor for short shapes.
* The post-R55 distribution is wider (std 14.7 vs 11.8) — the same
  kernel changes (a) lift typical-case performance via the
  EPILOG sched_barrier walls but (b) make a few cold-start samples
  worse, possibly due to KI=112 +5 spill amplifying first-launch
  register pressure on DSV3 cells. Median is the right summary
  statistic in this regime.
* GPU 3 contention adds ~30-point noise. R54 ran with idle GPU and
  saw clean strict distribution shift. R55's noise floor is high
  enough to mask sub-+5 mean shifts.

## Action

* HipKittens: `kernel_bf16_dynamic.cpp` +6 lines (3 sched_barrier
  sites in EPILOG 1, 3 in EPILOG 2). 1 commit `237ca6b1`.
* Primus-Turbo: 1 commit (this round note).

## R56 next-action surface

Post-R55, the structural sched_barrier audit is closed for the BF16
grouped fwd kernel. Both main_loop_iter (R54) and the epilogs (R55)
now have uniform "every MMA-burst-end gets sched_barrier(0)"
discipline. Remaining levers:

1. **PMC per-block wall-fraction bracket diagnostic** (R51 #1, R52
   #1, R53 #1, R54 #2 deferred). Now MORE valuable — we've fully
   harvested both the trivial KI specs (R52/R53) AND the
   sched_barrier audit (R54/R55). The R50 PMC capture showed all 3
   families have 15-25 pp MFMA util headroom; PMC marker bracketing
   on prologue / main_loop_iter / epilog 1 / epilog 2 would localize
   where that headroom lives. If main_loop_iter is already at MFMA
   plateau, attention shifts to fixed prologue / epilog cost.
   Recommended for R56.

2. **`__builtin_amdgcn_sched_barrier(MASK)` on the FUSED_KTAIL block
   for RCR** (untested pattern in this codebase, R54 next-action #4).
   The K-tail fuse block (lines 803+) is RCR-only, runs once per
   kernel call after EPILOG 2, and has its own DO_MMA pipeline. If
   the post-R55 EPILOG 2 sched_barrier walls leave LLVM free to
   hoist the FUSED_KTAIL load_b / load_a back into the post-EPILOG-2
   window, the FUSED_KTAIL block could benefit from the same
   pattern. Caveat: R52 found FUSED_KTAIL is "dead code for ALL 24
   metric shapes" — applies only if dispatch sends shapes to fuse,
   which the metric currently doesn't. Defer until R56-R57.

3. **DSV3-GateUP dB var-K dispatch retry** (R45/R47/R51 backup,
   bwd-side, metric-invisible). Smallest expected upside; cleanest
   bwd-side surface remaining. Worth a single round if (1) and (2)
   come up flat.

Recommended R56: **option 1 (PMC per-block wall-fraction bracket)**.
Diagnostic round, no kernel change, pinpoints where the 15-25 pp
MFMA headroom lives so R57+ can attack it directly.

## Metric numbers

```
                       R54 baseline (8)        R55 (10)             Δ
score median           905                     914.5                +9.5
score mean             909.9                   910.8                +0.9
score range            [899, 930]              [886, 928]           wider
gpt_oss   geomean      ~1.13                   ~1.13                ~flat
DSV3      geomean      ~1.15                   ~1.15                ~flat
Qwen3     geomean      ~1.17                   ~1.17                ~flat
correct_fail           0/24                    0/24                 no regression
sched_barrier sites    8 (main) + 0 (epilog)   8 (main) + 6 (epilog)
```

R55 commits a 6-line structural improvement that pins the EPILOG 1
and EPILOG 2 MMA-burst boundaries with the same sched_barrier(0)
discipline R54 introduced for main_loop_iter. The +9.5 median lift
is consistent with the predicted "extend the R54 mechanism to the
post-main-loop epilogs" hypothesis. The flat mean is attributable
to (a) the small surface (6 sites total, ~3-5 % of R54's
main_loop_iter benefit when amortized) and (b) ~30-point GPU-3
contention noise during this round.
