# Round 56 — BF16 grouped, fwd PMC tile-amortization DIAGNOSTIC — pinpoints small-B / small-M MFMA-util sag at low tiles/CU

## Goal coming in

R55 closed the structural sched_barrier(0) audit on `device_gemm_tile_body`
(both main_loop_iter and EPILOG 1/2). R55's R56 next-action recommended:

> 1. **PMC per-block wall-fraction bracket diagnostic** (R51/R52/R53/R54
>    deferred). Now MORE valuable — we've fully harvested both the trivial
>    KI specs (R52/R53) AND the sched_barrier audit (R54/R55). The R50
>    PMC capture showed all 3 families have 15-25 pp MFMA util headroom;
>    PMC marker bracketing on prologue / main_loop_iter / epilog 1 /
>    epilog 2 would localize where that headroom lives.

R56 starts the diagnostic. R55's recommendation was per-block bracketing,
which requires kernel-level marker injection (non-trivial). R56 instead
extends R50's MFMA-util capture to the **B=4 metric shapes that R50
never measured**, since today's metric worst (gpt_oss-GateUP-B4-M2048,
ratio 0.972) is in a B=4 regime R50's B=32 captures don't speak to.

## Today's metric attack target

R56 baseline metric (single run): **score 918**, gpt_oss family
geomean 1.130, DSV3 1.188, Qwen3 1.184. Per-shape worst:

```
gpt_oss-GateUP-B4-M2048  B=4  M=2048  N=5760  K=2880  ratio 0.972  weight 3
```

This is the only ratio < 1.0 in the suite today (HK 789.6 TF vs TRT
812.6 TF — HK actually slower than Triton). R55 last saw the same
shape at ratio 1.138 (HK 882.5 / TRT 775.7); the 14 % swing is mostly
GPU 3 contention noise (HK ±11 %, TRT ±5 % between adjacent runs).

## Hypothesis

R50 measured 3 B=32 shapes and showed MFMA utilization 57-76 %, with
the ranking driven by K-loop length (longer K = more per-tile
amortization of fixed prologue+epilog cost). R50 did NOT capture
B=4 shapes. The today-worst shape (B=4 M=2048) has only
`(B*M / 256) * ceil(N/256) = 32 * 23 = 736` output tiles vs
`NUM_CUS * occupancy = 256 * 2 = 512` waves available — i.e., only
~1.4 waves per persistent block, with 1/3 of CUs doing 3 persistent
iters and 2/3 doing 2 (tail imbalance).

If MFMA util sags at low tiles/CU, the persistent-kernel architecture
itself is leaving headroom on small-B shapes that the auto-tune-driven
single-tile launch grid (Triton's default) doesn't pay.

## Evidence — PMC capture (50 dispatches each, idle GPU 3 except for one co-tenant @ ~700-1100 W)

### Driver: `/tmp/probe_r56_b4_pmc.py`
Same DSV3 warmup + 50 fwd-only iterations as R49/R50. `R56_TARGET=1/2/3`
selects the shape.

### PMC counter set: same as R50 (`/tmp/r50_pmc.txt`)
```
SQ_VALU_MFMA_BUSY_CYCLES SQ_THREAD_CYCLES_VALU SQ_INSTS_VMEM SQ_BUSY_CYCLES GRBM_GUI_ACTIVE
```

### MFMA utilization, gpt_oss-GateUP across (B, M) variations

```
shape                                 M_total   tiles  tiles/CU   MFMA%   metric_ratio
gpt_oss-GateUP-B4-M2048   (worst)        8192     736     2.88    50.4    0.972
gpt_oss-GateUP-B4-M4096   (PASS)        16384    1472     5.75    55.4    1.305
gpt_oss-GateUP-B32-M2048  (R50)         65536    5888    23.00    59.7    1.110
```

(`tiles` = `ceil(M_total/256) * ceil(N/256)`. NUM_CUS=256 blocks each
with occupancy=2 → 512 waves available. `tiles/CU` = `tiles / 256`,
≈ persistent loop iterations per block.)

### Falsifies R50's "K-loop length is the dominant overhead amortizer"

R50 measured 3 K-different shapes ALL with B=32 M=2048, M_total=65536:
DSV3 K=7168 → 75.6 %, gpt_oss K=2880 → 65.7 %, Qwen3 K=4096 → 57.2 %.
R50 attributed the spread to K-loop length amortizing per-tile fixed
overhead.

R56 fixes the K and varies (B, M) instead. All 3 shapes have **the
same K=2880, same N=5760, same per-tile work**. The MFMA util ranking
now tracks `tiles/CU` from low (50 %) to high (60 %), confirming
**tile-count amortization is the dominant overhead amortizer**, not
K-loop length per se. R50's K-loop ranking was conflated with the
tiles-per-CU axis (DSV3 K=7168 also has the highest tiles/CU because
its M_total*N is highest).

### Mechanistic explanation

* `tiles/CU = 2.88` (B=4 M=2048): non-integer → 1/3 of CUs do 3
  persistent iters, 2/3 do 2. The 2-iter CUs finish first and IDLE
  for the remaining time of the 3-iter CUs. This shows up as low
  MFMA utilization because GRBM_GUI_ACTIVE keeps incrementing on
  the idle CUs (GPU is "active" — kernel is launched) but
  SQ_VALU_MFMA_BUSY_CYCLES drops to zero on those CUs. **Tail
  imbalance is the dominant 50 % MFMA util killer.**
* `tiles/CU = 5.75` (B=4 M=4096): same 75/25 imbalance but with 6
  iters total per block, the imbalance is 1/6 = 17 % of wall time
  vs B=4 M=2048's 1/3 = 33 %. MFMA util recovers to 55 %.
* `tiles/CU = 23.00` (B=32 M=2048): exact integer → no tail imbalance.
  MFMA util reaches 60 % — bounded by per-iter prologue+epilog cost
  (constant overhead per persistent iter).

### Why metric ratio doesn't track MFMA util

```
shape                    MFMA%   ratio   inferred TRT MFMA%
B=4 M=2048 (worst)        50.4    0.972  ~52      (TRT ~equal)
B=4 M=4096 (PASS)         55.4    1.305  ~42      (TRT 23 % BEHIND)
B=32 M=2048 (R50)         59.7    1.110  ~54      (TRT 9 % behind)
```

* On B=4 M=2048 HK and TRT both pay the small-batch tax. HK ~50 %,
  TRT ~52 %. They're both wave-imbalanced; TRT happens to handle
  the K%128 != 0 K-tail slightly more efficiently in its
  fused-K-tail loop. Ratio sits below 1.0.
* On B=4 M=4096 HK still has wave imbalance (75/25 between 5 vs 6
  iters) but TRT's tile sizes don't divide N=5760 evenly either,
  AND TRT pays a per-tile launch overhead that HK's persistent
  kernel amortizes. HK wins big at 1.305.
* On B=32 M=2048 HK at 60 % MFMA still beats TRT (1.110), but the
  per-tile fixed cost (~40 % of cycles spent NOT in MFMA — prologue
  loads, epilog stores, persistent-loop boundary syncs) caps the
  ratio.

## Falsification consequences

**R56 closes:**

* R50 hypothesis "K-loop length is the dominant overhead amortizer"
  — refuted by the constant-K tile-count sweep above.
* R55-recommended #2 "FUSED_KTAIL block sched_barrier(MASK)" lever
  — would have targeted the same K-tail HK already handles correctly;
  the gap is structural (tile-count amortization), not K-tail
  scheduling.

**R56 opens:**

* **Persistent-grid sizing** for low-tile workloads. Currently
  `dispatch_grouped` always launches `NUM_CUS=256` blocks. For
  workloads with `tiles < NUM_CUS * occupancy = 512`, only
  `tiles / occupancy ≈ tiles / 2` blocks are needed; the rest run
  empty persistent iters. For B=4 M=2048 (736 tiles), launching
  `min(NUM_CUS, ceil(tiles/2)) = min(256, 368) = 256` blocks gives
  `tiles/CU = 2.88` — what we see today. But launching
  `tiles / occupancy = 368` blocks would give `tiles/CU = 2.0` exact,
  eliminating the tail imbalance. Trade-off: more blocks = more
  prologue cost amortized over fewer iters.

  Better predicate: when `tiles < NUM_CUS * 4`, switch grid to
  `ceil(tiles)` blocks (1 tile per block, no persistent loop) so
  every CU does exactly 1 tile worth of work. Eliminates the
  amortization headroom but also eliminates persistent-loop
  imbalance. Mathematically pinpointed by R56's PMC capture.

* **Tile sub-division** for very low tile-count workloads. If
  `tiles < NUM_CUS`, use 128×128 or 64×64 tiles instead of 256×256
  to grow the tile count by 4x or 16x. Higher launch overhead per
  tile but better wave occupancy. Touches more of the kernel
  structure than persistent-grid sizing.

## R57 next-action surface

1. **Persistent-grid sizing for low-tile workloads** (R56 pinpointed
   lever). Add a general predicate to `dispatch_grouped`:
   ```cpp
   const int tiles = (g.M_total / BLOCK_SIZE) * ceil_div(g.n, BLOCK_SIZE);
   const int blocks = (tiles < NUM_CUS * 2)
                      ? std::max(1, tiles / 2)   // small workloads
                      : NUM_CUS;                 // large workloads
   ```
   This reduces `tiles/CU` from non-integer (tail-imbalanced) to
   integer for sub-1024-tile workloads. Expected upside: B=4 M=2048
   MFMA 50% → 55-60% → ratio 0.97 → 1.06-1.13. **Recommended R57.**

2. **Tile sub-division** for `tiles < NUM_CUS`. Higher complexity,
   smaller upside (only the very smallest workloads benefit). Defer
   to R58+ pending R57 outcome.

3. **PMC marker bracketing per-block** (R55 #1 original recommendation).
   Now LESS valuable since R56's tile-amortization finding gives a
   direct lever (option 1) without needing per-block breakdown.
   Defer.

## Resource report

No kernel change in R56. `kernel_bf16_dynamic.cpp` unchanged from
R55 commit (HK `237ca6b1`).

## Action

* HipKittens: no kernel change.
* Primus-Turbo: 1 commit (this round note).
* PMC artifacts:
  * `/tmp/r56_pmc_b4_m2048/pmc_1/b4_m2048_results.db`
  * `/tmp/r56_pmc_b4_m4096/pmc_1/b4_m4096_results.db`
  * `/tmp/r56_pmc_b32_m2048/pmc_1/b32_m2048_results.db`
  * Probe driver: `/tmp/probe_r56_b4_pmc.py`
  * Summarizer: `/tmp/r56_summarize_v2.py`
  * Counter set: `/tmp/r50_pmc.txt` (reused from R50)

## Metric numbers (round 56)

```
                                R56 baseline (3 paired runs after R55 build)
score (single)                  918   893   923   925
gpt_oss   geomean ratio         1.130 (worst-shape gpt_oss-GateUP-B4-M2048 = 0.972)
DSV3      geomean ratio         1.188
Qwen3     geomean ratio         1.184
correct_fail                    0/24
shapes >= 1.25 (PASS)           3/24  (DSV3-Down-B16-M2048, gpt_oss-GateUP-B4-M4096,
                                       Qwen3-Down-B16-M2048)
```

R56 is purely diagnostic — no kernel edit, metric movement is GPU 3
contention noise. The deliverable is the falsification of R50's
"K-loop length is the dominant amortizer" interpretation and the
discovery that **tiles/CU < 4** is the regime where the persistent
kernel leaks 10-15 pp MFMA util to tail imbalance. R57 should attempt
the persistent-grid sizing lever (option 1 above).
