# Round 60 — BF16 grouped, fwd 1-tile-per-block low-tile dispatch — FALSIFIED

## Goal coming in

R56 PMC diagnostic identified `gpt_oss-GateUP-B4-M2048` (tiles=736,
2.88 iters/CU, MFMA util 50.4%) as the worst metric shape and proposed
two structural levers in priority order:

1. (R57) FUSED_KTAIL ↔ EPILOG 2 sched_barrier overlap — FALSIFIED in R57.
2. **(R56 → R57+ recommended) Persistent-grid sizing for low-tile workloads**
   — UNTRIED through R57/R58/R59. R60 attempts the most direct form:
   1-tile-per-block (1tpb) for shapes with `tiles ∈ (0, NUM_CUS*4) ∧
   tiles % NUM_CUS != 0`.

R60 starting metric (GPU 7, single sample): score=910, gpt_oss family
geomean 1.121, DSV3 1.152, Qwen3 1.176. Per-shape worst (matching R56
pattern):

```
gpt_oss-GateUP-B4-M2048  B=4 M=2048 N=5760 K=2880  ratio 1.023  weight 3
gpt_oss-Down-B4-M2048    B=4 M=2048 N=2880 K=2880  ratio 1.136  weight 3
gpt_oss-GateUP-B4-M4096  B=4 M=4096 N=5760 K=2880  ratio 1.257  weight 3 (PASS)
```

The two `B=4 M=2048` shapes both have wave-imbalanced tile counts:
- `tiles=736`, `736 mod 256 = 224` → 224 CUs do 3 iters, 32 do 2 iters.
- `tiles=384`, `384 mod 256 = 128` → 128 CUs do 2 iters, 128 do 1 iter.

R56 measured the M_total=2048 vs M_total=4096 vs M_total=65536 PMC
sweep at constant K=2880 and showed MFMA% tracks `tiles/CU` cleanly
(50.4% / 55.4% / 59.7% as `tiles/CU` rises 2.88 / 5.75 / 23.0). The
implied bound: erasing the imbalance lifts MFMA% from 50.4 →
~57.2%, ratio 1.023 → ~1.16.

## Hypothesis (FALSIFIED)

**1tpb mode**: when `tiles ∈ (0, NUM_CUS*4) ∧ tiles % NUM_CUS != 0`,
launch `grid = tiles` (each block does exactly one (group, tile)
pair, no persistent loop). With reported HW occupancy 2 (R59 build
report: VGPR=248, occ=2 on the `<RCR, KI=0, FUSED=true>` template
that gpt_oss routes through), the GPU dispatch engine fills 256 CUs ×
2 = 512 simultaneous waves, naturally absorbing 2 dispatch rounds for
any tile count up to 1024 without static-partition tail.

Predicted wall (per R60 model):
* 1tpb: `2 × (prologue + iter)` ≈ `2 × (50 + 150) µs` = 400 µs
* persistent: `prologue + 3 × iter` ≈ `50 + 450` = 500 µs (slowest CU)
* expected: 1tpb +25% wall reduction → ratio 1.023 → ~1.28.

## Implementation (R60 candidate, ALL REVERTED)

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

1. **Kernel-side grid-stride parameterization** (lines 3719, 3823):
   ```diff
   -    int pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, 64);
   +    int pid = chiplet_transform_chunked(blockIdx.x, gridDim.x, g.num_xcds, 64);
   ...
   -    for (int gt = pid; gt < total_tiles; gt += NUM_CUS) {
   +    for (int gt = pid; gt < total_tiles; gt += gridDim.x) {
   ```
   `chiplet_transform_chunked(_, 256, 8, 64)` is identity (limit = 0,
   early-out fires for all `blockIdx.x > 0`), so the persistent-grid
   path is bit-identical to baseline. For `gridDim.x ∈ {384, 736}`
   (1tpb), `chiplet_transform_chunked(_, 736, 8, 64)` permutes within
   the 0..511 range and is identity for 512..735 — every block still
   gets a unique pid in [0, gridDim.x).

2. **Host-side adaptive grid** (lines 4042-4062):
   ```cpp
   static inline int compute_grouped_grid(int M_total, int bpc) {
       if (bpc <= 0 || M_total <= 0) return NUM_CUS;
       const int tiles = (M_total / BLOCK_SIZE) * bpc;
       if (tiles > 0 && tiles < NUM_CUS * 4 && (tiles % NUM_CUS) != 0) {
           return tiles;
       }
       return NUM_CUS;
   }
   // launch_one_grouped / launch_one_grouped_fuse:
   //   const int grid = compute_grouped_grid(g.M_total, g.bpc);
   //   grouped_kernel<...><<<dim3(grid), ...>>>(g);
   ```
   General predicate (no per-(M,N,K) hardcode). Fires only on shapes
   with `tiles ∈ (0, 1024) ∧ tiles % 256 != 0`. For the 24-shape MoE
   metric this catches exactly `tiles ∈ {384, 736}` (the 2 worst
   imbalanced gpt_oss B=4 shapes); the 22 integer-iters/CU shapes
   (`tiles % 256 == 0`) keep the conventional NUM_CUS persistent grid.

## Resource report (R60 candidate, GPU 7, build clean)

```
                                SGPR  VGPR  VGPRspill  occ
RCR KI=0   non-fuse:             90   244       0       2
RCR KI=48  (Qwen3-Down):         87   256       0       2
RCR KI=64  (Qwen3-GateUP):       87   256       0       2
RCR KI=88  (gpt_oss non-fuse):   87   256       0       2
RCR KI=112 (DSV3-GateUP K=7168): 87   256      24       2
RCR KI=0,FUSED=true (gpt_oss):   97   248       0       2  ← R55 baseline 96/248/0/2
RRR KI=0,FUSED=true:             99   249       0       2
CRR KI=0:                        94   244       0       2
```

`+1 SGPR` on the FUSED=true template tracks the `gridDim.x` runtime
read replacing the `NUM_CUS` constexpr. All other KI specs unchanged
from R55 baseline.

## Correctness

`/tmp/probe_r60_correctness.py` via Primus-Turbo `turbo.ops.grouped_gemm`
(BackendType.HIPKITTEN), 5 representative shapes:

```
                                                           mode        SNR     allclose
gpt_oss-GateUP-B4-M2048   tiles=736                       1tpb       47.85 dB  True
gpt_oss-Down-B4-M2048     tiles=384                       1tpb       47.85 dB  True
gpt_oss-GateUP-B4-M4096   tiles=1472                      persistent 47.85 dB  True
DSV3-GateUP-B16-M2048     tiles=2048                      persistent 47.85 dB  True
Qwen3-Down-B16-M2048      tiles=2048 (RCR KI=48)          persistent 47.83 dB  True
```

All 5/5 PASS at bf16 floor (47.83-47.85 dB). The 1tpb path is
functionally equivalent to persistent — no MMA / store / reduction
bug introduced.

## Metric (single-sample, GPU 7)

```
config                                          score   gpt_oss  DSV3   Qwen3   below_tgt
baseline (HK 237ca6b1, before any R60 work)     910     1.121   1.152   1.176   22/24
R60 1tpb (kernel + adaptive grid)               882     1.045   1.151   1.234   19/24
R60 kernel-only (gridDim.x stride, grid=256)    901     (kernel-only neutral, control)
baseline post-revert (3 paired samples)         920/896/890 (median 896)
```

**1tpb mode regressed −28 score** (882 vs 910 baseline single-sample;
−5 vs the 920/896/890 baseline median 896). Per-shape: the two
1tpb-targeted shapes BOTH lost ratio:

```
gpt_oss-GateUP-B4-M2048  baseline ratio 1.023 → 1tpb ratio 0.880   (−14 %)  HK 789.6 → 703.4 TFLOPS
gpt_oss-Down-B4-M2048    baseline ratio 1.136 → 1tpb ratio 0.858   (−25 %)  HK 689.3 → 536.2 TFLOPS
```

`HK TFLOPS` itself dropped by 11-22 % on the 1tpb shapes. This is the
HK kernel slowing down, not Triton speeding up — the hypothesis is
falsified at the kernel-wall level, not just at the ratio level.

The kernel-only mod sample (901) sits squarely in the post-revert
baseline noise envelope (890-920 across 3 samples). The
`NUM_CUS → gridDim.x` substitution is itself **neutral** (within noise);
the regression is from the host-side grid-shrinking, not the kernel mod.

## Why 1tpb falsified (root cause)

The R60 model assumed prologue ≈ 50 µs and iter ≈ 150 µs, giving 1tpb
2 × 200 = 400 µs vs persistent 50 + 450 = 500 µs. **The HK TFLOPS
drop reveals the prologue (or some other fixed per-block cost) is
much larger than 50 µs**. Working backward from the observed HK
TFLOPS:

```
gpt_oss-GateUP-B4-M2048 (tiles=736):
  baseline persistent: HK 789.6 TFLOPS = 6 × 8192 × 5760 × 2880 / wall
                                       = 8.156e11 / wall
                       wall = 8.156e11 / 789.6e9 = 1033 µs
  1tpb:                 HK 703.4 TFLOPS = 1159 µs  (+126 µs over baseline)
```

If 1tpb has wall = 2 × (prologue + iter) and persistent has wall =
prologue + 3 × iter, then:

```
1033 = P + 3I       ...(persistent)
1159 = 2P + 2I      ...(1tpb)
```

Solving: P = 1033 - 3I, substitute: 1159 = 2(1033 - 3I) + 2I = 2066 - 4I,
so 4I = 907 → I = 227 µs, P = 1033 - 681 = 352 µs.

**The prologue cost is ~352 µs, not 50 µs** — about ~34 % of total
persistent wall on this shape. This includes:
- O(G) cumsum scan + LDS init (single-thread, ~50 cycles/group × G=4 = ~200 cy).
- SRD setup (~50 cy).
- chiplet swizzle (~50 cy → identity here, cheap).
- LDS swizzled-offset prefill `prefill_swizzled_offsets` (cooperative, ~few hundred cy).
- HBM B SRD initial bounds compute (`make_srsrc`, ~50 cy).
- First-iter prologue (B SRD rebuild for group_idx=0, A/B initial HBM
  loads via `buffer_load_lds`, LDS-write + `s_waitcnt` drain, first
  MMA pair issuance).

The dominant cost is likely the FIRST iter's prologue (HBM fetch of
the first A/B subtiles + LDS write + `s_waitcnt` drain) — this is
amortized 1× over 3 iters in persistent (33% per-tile prologue cost)
but pays in full per tile in 1tpb (100% per-tile prologue cost).

Net: 1tpb's 1.5× extra dispatch round (`2 vs 1`) is overcompensated by
the 3× extra prologue cost.

## Falsification consequences

**R60 closes**:

* R56 R57-recommended #1 lever ("persistent-grid sizing for low-tile
  workloads" — switch to 1tpb when `tiles < NUM_CUS * 4`) — falsified
  empirically. The R56 model under-estimated the prologue cost by ~7×.
* R56 R57-recommended #2 lever (smaller persistent grid, e.g.
  `grid = tiles / desired_iters_per_block = 184` blocks for tiles=736)
  — implicitly falsified by the same prologue argument. With 184
  blocks each doing 4 iters, wall = `P + 4I` = 352 + 908 = 1260 µs vs
  baseline 1033 µs. Same prologue, more iters per block → worse.

**R60 opens**:

* **Work-stealing via global atomic counter** (R57's R58 recommended
  option 2). This was the OTHER lever R56 left on the table. Unlike
  static partition (R60 1tpb) or smaller grid, work-stealing keeps
  256 blocks (1 prologue per CU) AND eliminates the wave imbalance
  by dynamically claiming tiles. Predicted wall (with prologue=352 µs,
  iter=227 µs):
  ```
  static persistent: wall = P + 3I = 1033 µs (slowest CU does 3 iters)
  work-stealing:     wall = P + ceil(736/256) × I_eff = 352 + 3 × 227 × (736/(3×256)) ≈ P + 2.875 × I = 1005 µs
  ```
  ~2.7 % wall reduction, ratio 1.023 → ~1.05. Smaller than R60's
  predicted 25 % but real (no prologue inflation). Plumbing cost: 1
  device int counter, 1 atomicAdd per persistent iter (~50 cy
  uncontended), 1 zero-init per kernel launch.

* **Reduce prologue cost on B=4 shapes** — 352 µs is 34 % of total
  wall. The `prefill_swizzled_offsets` cooperative LDS write may be
  reducible if some of the offsets are constant across iters.
  `make_srsrc` cost is fixed.

* **Combined work-steal + reduced prologue** — possibly the only
  path to lift gpt_oss B=4 ratios above 1.05 without rewriting the
  kernel architecture.

## Outcome

* **No HipKittens commit** (working tree reverted to R55 baseline,
  HK SHA `237ca6b1bdd7e432e3d0ad97bf2082f3cb150e62`). Build verified
  clean, post-revert metric in noise envelope (890-920 across 3
  samples, median 896 vs starting 910).
* **Primus-Turbo: 1 commit** (this round note + R60 1tpb falsification
  data + prologue-cost bound).
* This doc + the prologue-cost solve from observed HK TFLOPS is the
  R60 deliverable.

## R61 next-action surface

Updated priorities given R60 prologue-cost discovery (P ≈ 352 µs,
~34% of total wall on `gpt_oss-GateUP-B4-M2048`):

1. **Work-stealing via global atomic counter** (now highest priority,
   was R56→R57+ recommended #2). Plumbing cost: add `int* tile_counter`
   field to `grouped_layout_globals`, allocate small device buffer on
   `grouped_layout_globals` construction in Primus-Turbo, zero-init
   pre-launch in `dispatch_grouped`. Replace static `for (gt = pid; gt
   < total_tiles; gt += NUM_CUS)` with dynamic atomicAdd-claim. Predicted
   lift: ratio 1.023 → ~1.05 on the worst shape, +5-8 score net.
   Falsifiable in 1 round.

2. **Prologue cost diagnostic** (PMC bracketing). R60's prologue-cost
   bound (352 µs) was inferred analytically from the 2-equation
   walltime system; a direct PMC capture between kernel entry and
   first-iter MMA would localize the dominant prologue cost
   (cumsum scan vs swizzle prefill vs first-iter HBM load). Without
   this diagnostic, prologue-reduction levers are blind.

3. **Per-XCD chiplet swizzle for the actual NUM_CUS=256 grid** (lower
   priority; cache-locality lever). R60's investigation revealed
   `chiplet_transform_chunked(_, NUM_CUS=256, 8, 64)` is identity
   for the entire grid (limit = 0). The intended XCD-coordinated
   B-tile reuse never happens. A correct chiplet swizzle for the
   256-block grid would re-order pid such that XCD-local CUs sweep
   adjacent N-cols. Only worthwhile if the prologue lever (item 1)
   is exhausted.

4. **DEPRIORITIZED**: KI specialization retries (R58/R59 falsified;
   compile-time KI consistently spills 9-20 VGPR via LLVM scheduling).

## Action

* HipKittens: no kernel change (working tree clean post-revert).
* Primus-Turbo: 1 commit (this round note).
* Probe artifact: `/tmp/probe_r60_correctness.py` (5 cases, all PASS
  on the R60 candidate before revert; can be reused as the work-
  stealing R61 correctness probe).
* Metric logs:
  * `/tmp/metric_round_60.log` — R60 starting baseline (910).
  * `/tmp/metric_round_60_after.log` — R60 1tpb candidate (882, regression).
  * `/tmp/metric_round_60_kernel_only.log` — kernel-only mod (901, neutral).
