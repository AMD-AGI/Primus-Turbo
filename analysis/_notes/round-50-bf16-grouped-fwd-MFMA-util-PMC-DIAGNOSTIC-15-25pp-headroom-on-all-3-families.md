# Round 50 — BF16 grouped, fwd MFMA-utilization PMC diagnostic — 15-25pp HEADROOM ON ALL 3 FAMILIES

## Goal coming in

R49 closed the LDS bank-conflict lever (A3) with PMC-confirmed
`SQ_LDS_BANK_CONFLICT = 0` on both gpt_oss K%128 != 0 fuse path AND
DSV3 K%128==0 baseline. R49 next-action surface ranked:

1. **PMC-driven fwd MFMA utilization on gpt_oss vs DSV3** (R49 backup
   #3, R48-listed un-tried). Free PMC datapoint, single-shape capture,
   gives clear signal whether fwd-side optimization has headroom.
2. DSV3-GateUP dB var-K dispatch retry (R47 backup, weight 1x).
3. Store batching 4 → 1 in `grouped_var_k_kernel` epilog (high
   correctness validation cost).

R50 executes step 1: capture MFMA utilization for the worst gpt_oss
shape AND triangulate against KI=64 (Qwen3) and KI=112 (DSV3) baselines
for the K%128==0 fast path.

## Hypothesis (R50)

Per R49's recommended go/no-go criterion: if MFMA utilization on the
gpt_oss fuse path (KI=0 dynamic, FUSED_KTAIL=true) is < 90 % then there
is real fwd-side scheduling headroom; if ≥ 90 % the fuse path is
saturated and we should pivot to bwd levers (R47 backup #2 / #3).

Per R49's mechanical model: gpt_oss has K=2880 (K%128=64) which forces
both (a) the FUSED_KTAIL block (K-tail HBM-to-register reload + 4
extra DO_MMAs) AND (b) the dynamic-bound `#pragma unroll 2` schedule
on the main loop (KI=44 fuse spec was R39-falsified at 28 VGPR spill).
Either or both could leak 5-15 % from peak MFMA issue rate.

## Evidence

### PMC capture setup

* `/tmp/probe_r49_lds_workload.py` (R49 driver, reused) — 50 fwd
  iterations + DSV3 cold-start warmup, single-shape grouped GEMM via
  `turbo.ops.grouped_gemm` HIPKITTEN backend, autograd off.
* `/tmp/probe_r50_qwen3.py` — Qwen3 variant of the same driver.
* `/tmp/r50_pmc.txt` — PMC counter set:
  `SQ_VALU_MFMA_BUSY_CYCLES SQ_THREAD_CYCLES_VALU SQ_INSTS_VMEM
   SQ_BUSY_CYCLES GRBM_GUI_ACTIVE`.
* `rocprofv3 --kernel-include-regex "grouped_kernel"` to filter to the
  active fwd grouped instantiation only (skip K-tail correction kernel,
  scalar tail kernel).
* MfmaUtil derived from `rocprofv3 -L`:
  `MfmaUtil = reduce(SQ_VALU_MFMA_BUSY_CYCLES, sum) /
              (reduce(GRBM_GUI_ACTIVE, max) * SIMD_NUM) * 100`,
  `SIMD_NUM = 1024` (256 CU × 4 SIMD/CU on MI355X).
* PMC artifacts: `/tmp/r50_pmc_gpt_oss/`, `/tmp/r50_pmc_dsv3/`,
  `/tmp/r50_pmc_qwen/`.

### MFMA utilization comparison (3 instantiations, fwd RCR)

```
                                         dispatch     MFMA
shape                              KI    dur (us)    util (%)   ratio @ R50
gpt_oss-GateUP-B32-M2048           0     1833        65.7       1.056
   (K=2880, K%128=64, FUSED=true)
DSV3-GateUP-B32-M2048             112    2934        75.6       1.132
   (K=7168, K%128=0,  FUSED=false)
Qwen3-GateUP-B32-M2048             64    1376        57.2       1.109
   (K=4096, K%128=0,  FUSED=false)
```

### Falsifies R49 hypothesis "fuse path is the unique bottleneck"

R49's pre-hypothesis: gpt_oss fuse path is materially lower MFMA util
than the K%128==0 paths because of (a) the FUSED_KTAIL block and (b)
the dynamic-K loop. The data only partially supports this:

* gpt_oss vs DSV3: gpt_oss is **9.9 pp LOWER** (65.7 % vs 75.6 %).
  Direction matches the hypothesis but magnitude is HALF what R49
  estimated (model said ~16 pp from MFMA / GRBM ratio extrapolation).
* gpt_oss vs Qwen3: gpt_oss is **8.5 pp HIGHER** (65.7 % vs 57.2 %).
  Qwen3 — the SHORTER K%128==0 path — has the LOWEST MFMA util of
  the three. KI=64 compile-time unroll (Qwen3) is NOT mfma-saturating.

The MFMA util ranking by K (longer K = better):

```
DSV3   K=7168  KI=112   75.6 %
gpt_oss K=2880 KI=0     65.7 %    (fuse path, includes K-tail block)
Qwen3  K=4096  KI=64    57.2 %
```

This suggests the MFMA util gap is driven primarily by **K-loop length
amortizing per-tile / per-launch overhead**, NOT by the fuse path's
K-tail block per se. Smaller K → shorter main loop → larger relative
weight of:
* Per-persistent-iteration zero+store epilog (~constant)
* Per-tile dispatch overhead (chiplet swizzle, cumsum scan, B SRD
  rebuild on group transitions)
* Prologue / epilog 1 / epilog 2 (3 K_TWO_TILE pairs of work, fixed)

For Qwen3 K=4096 = ki_main=64 → 32 K_TWO_TILE pairs total. Subtract 3
(prologue + 2 epilogs) → only 29 K_TWO_TILE pairs in `main_loop_iter`.
For gpt_oss K=2880 → ki_main=44 → 22 K_TWO_TILE pairs total → 19 in
main_loop_iter. For DSV3 K=7168 → ki_main=112 → 56 K_TWO_TILE pairs →
53 in main_loop_iter.

The fixed prologue+epilog block is ~6 % of DSV3's main loop, ~14 % of
gpt_oss's, ~10 % of Qwen3's. This is consistent with the MFMA util
ranking but doesn't fully explain the Qwen3 < gpt_oss inversion.

### Per-call HBM bandwidth context (orthogonal datapoint)

```
shape              SQ_INSTS_VMEM/disp   dur(us)    VMEM/us
gpt_oss            2.36e7               1833       12,860
DSV3               3.42e7               2934       11,640
Qwen3              1.57e7               1376       11,430
```

VMEM rate is ~similar across shapes (~11.5-13 K instructions / us).
HBM bandwidth is NOT the limiter — all three shapes stream at similar
rates. The kernel is MFMA-throughput-limited, just at a level below
peak.

## Falsification consequence

R50 closes:

* **R49 hypothesis "gpt_oss fuse path is uniquely bottlenecked"**.
  All three KI/fuse combinations show MFMA utilization in the
  57-76 % range — none are MFMA-saturated. The gpt_oss fuse path
  is NOT a clear outlier; it sits IN BETWEEN Qwen3 (lower) and DSV3
  (higher). The R49-recommended pivot ("if MFMA util ≥ 90 % → fuse
  path is saturated → fall through to (1) DSV3-GateUP dB var-K
  dispatch retry") triggers a NULL: neither branch holds — fuse
  path has headroom but it's not uniquely large.

* **R49 hypothesis "K-tail block is the dominant fuse-path overhead"**
  is also falsified by the Qwen3 datapoint. Qwen3 has NO K-tail block
  (KI=64 compile-time, FUSED=false) and yet has LOWER MFMA util than
  the gpt_oss fuse path. The K-tail block's overhead is NOT what's
  capping the fuse path at 65 %.

R50 opens:

* **All 3 families have 15-25 pp MFMA util headroom.** Lifting MFMA
  util uniformly by ~10 pp would:
  - gpt_oss ratio: 1.056 × (75.6/65.7) ≈ 1.215 (+15 %)
  - DSV3 ratio:    1.132 × (90/75.6) ≈ 1.348 (+19 %, OVER target)
  - Qwen3 ratio:   1.109 × (75.6/57.2) ≈ 1.466 (+32 %, OVER target)

  The largest score upside lives on the K%128==0 fast path
  (DSV3 + Qwen3, 16 weight-1 shapes), NOT on the gpt_oss K-tail.
  This is a significant pivot from the prior 7-round attack on the
  gpt_oss fuse path (R39, R41-R49 all targeted gpt_oss-specific
  bottlenecks).

* **Gap structure: prologue/epilog overhead amortization**. The
  fixed prologue + epilog 1 + epilog 2 block (3 K_TWO_TILE pairs of
  work) is roughly the same wall-time across all three K values, so
  shorter-K kernels pay a larger relative tax. This is structural —
  shrinking the prologue/epilog requires kernel-body restructuring
  (probably not 1-round work). But measuring per-block wall fraction
  with rocprofv3 marker bracketing on the prologue / main-loop /
  epilog blocks would FALSIFY OR CONFIRM this hypothesis cheaply.

## R51 next-action surface

Three candidates, ranked by upside:

1. **MFMA pipeline scheduling audit on the fast-path
   `device_gemm_tile_body` main_loop_iter** (R50 lever B1, expanded).
   Per round-task-body lever B1: "If MFMA util < 90 %, the kernel is
   not MFMA-saturated → there's room for schedule tweaks." All three
   shapes are at 57-76 % MFMA — pipeline schedule almost certainly
   has room. Two attack vectors:
   * `__builtin_amdgcn_s_setprio(1)` placement audit — currently set
     around DO_MMA inside main_loop_iter but NOT around K-tail block
     or epilog 1/2 DO_MMAs. Adding setprio(1) to those MMA bursts may
     reduce wave-scheduler stalls.
   * `__builtin_amdgcn_sched_barrier(0)` placement audit — the
     reorderings LLVM does between `s_waitcnt lgkmcnt(0)` and DO_MMA
     determine how tightly MFMAs pack. Tighter sched_barrier walls
     around MMA bursts may improve issue rate.

2. **Per-block wall-fraction PMC bracket** (diagnostic, like R50).
   Quantify prologue + epilog 1 + epilog 2 wall-time vs main_loop_iter
   wall-time. If fixed-overhead is large (> 20 %) for short K, pivot
   to shrinking it. If small (< 5 %), eliminate it as a lever and
   pivot to MFMA pipeline.

3. **DSV3-GateUP dB var-K dispatch retry** (R49 backup #1, R47/R45).
   Smallest expected upside (DSV3 already at ratio 1.13-1.15) but
   cleanest dispatch surface remaining. R24 dropped `xcds=0` due to
   allclose drift; re-sweep with `xcds ∈ {1, 2, 4, 8}` on
   (tiles_m=16, tiles_n=28) cells.

Recommended for R51: **start with (1.a)** — `s_setprio(1)` placement
audit. Lowest correctness risk (purely scheduler hint, byte-identical
output), 1-line edit per MMA burst, broad coverage (every fwd
grouped_kernel call benefits). If +5 score → land + R52 attacks
(1.b). If neutral → pivot to (2) the per-block wall fraction
diagnostic.

## Action

* HipKittens: no change.
* Primus-Turbo: 1 commit (this diagnostic note).
* PMC capture artifacts: `/tmp/r50_pmc_gpt_oss/pmc_1/gpt_results.db`,
  `/tmp/r50_pmc_dsv3/pmc_1/dsv_results.db`,
  `/tmp/r50_pmc_qwen/pmc_1/qwen_results.db`.
  Probe drivers: `/tmp/probe_r49_lds_workload.py`,
  `/tmp/probe_r50_qwen3.py`. PMC config: `/tmp/r50_pmc.txt`.
  Summarizer: `/tmp/r50_summarize_pmc_v2.py`.

## Metric numbers

```
                       baseline (R49)   R50 (no kernel)   delta
score                  879/880          879               flat (no change)
gpt_oss_20B  geomean   1.0886           1.0869            ±0.002 (noise)
DeepSeek-V3  geomean   1.1209           1.1209            flat
Qwen3        geomean   1.1129           1.1133            flat
correct_fail           0/24             0/24              no change
```

R50 is purely diagnostic — no kernel edit, no metric movement
expected. The deliverable is the falsification of R49's "fuse path
is uniquely bottlenecked" hypothesis and the pivot toward MFMA
pipeline scheduling on the SHARED main_loop_iter (which all 3
families exercise).
