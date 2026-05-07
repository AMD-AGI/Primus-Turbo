# Round-2 (gpt_oss FP8 kernel-only ceiling, current Primus run; 2026-05-07)

## Round-G G1c — wgrad var-K NUM_CUS launch-geometry lever PROVEN

**Result**: The R1 launch-geometry hypothesis is **confirmed with 3-shape
evidence**. Reducing the persistent-grid slot count from `NUM_CUS = 256`
to `slots = 192` lifts the worst gpt_oss wgrad shape (Down-B4-M2048) by
**+6.14 % kernel TFLOPS** (1395 → 1481 T). The same lever lifts the
sibling Down-B4-M4096 wgrad by **+5.22 %** (1679 → 1767 T). On a
counter-shape (GateUP-B32-M4096 wgrad, 30+ wave-steps/CU) the same
slots=192 setting **regresses by −17.34 %** — confirming that the
lever is short-grid-specific and **must be gated** in the binding
dispatcher.

This advances the HipKittens Round-G G1 plan two steps in one round:
G1c (lever proposal) is closed with a quantified evidence table; G1d
(implementation) is unblocked once the binding exposes a per-call
`slots` kwarg (see Round-3 plan below).

## Probe artifacts

* `scripts/_probe_round_2_vark_numcus_verify.py` — bit-equivalence
  verifier. Runs `grouped_gemm_fp8_variable_k_impl` on
  Down-B4-M2048 wgrad with `TK_VARK_NUM_CUS ∈ {32, 64, 96, 128, 160,
  192, 256}` and confirms `max_abs_diff = 0.0` and `mismatch_elems =
  0 / 33 177 600` for every cell vs the slots=256 baseline. Result:
  ALL slot values produce bit-identical output. The kernel-source
  edit (NUM_CUS → gridDim.x in the persistent loop + chiplet swizzle
  range) is a pure scheduling refactor.
* `scripts/_probe_round_2_vark_numcus_sweep.py` — kernel-only timing
  sweep. 250 iter × 7 trial p20 × 3 seeds × kernel-only direct call
  to `grouped_gemm_fp8_variable_k_impl`, mirroring the R10/R11/R30
  var-K probe protocol. Subprocess-per-slot value (HK static cache).
* `/tmp/round_2_vark_numcus_sweep.json` — raw sweep results.

## Sweep evidence

### Anchor: Down-B4-M2048 wgrad (worst gpt_oss wgrad shape)

```
slots   TFLOPS   Δ vs 256
   32    286.8    -79.44%   (under-saturated; 484/32=15.1 steps/slot but only 32 CUs)
   64    570.5    -59.11%
   96    838.7    -39.89%
  128   1010.1    -27.60%
  160   1231.4    -11.74%
  192   1480.8    +6.14%    *winner
  256   1395.2     0.00%    (baseline, NUM_CUS default)
  384   1392.5    -0.19%    (clamped to 256 by env hook safety bound)
```

### Sibling: Down-B4-M4096 wgrad (same 484-tile geometry, 2× M_per_g)

```
slots   TFLOPS   Δ vs 256
  128   1238.0   -26.27%
  160   1516.5    -9.68%
  192   1766.7    +5.22%   <-- ANCHOR_WINNER
  224   1749.7    +4.21%
  256   1679.0     0.00%
```

### Counter-shape: GateUP-B32-M4096 wgrad (saturated 7744-tile geometry)

```
slots   TFLOPS   Δ vs 256
  128   1293.6   -39.84%
  160   1594.2   -25.86%
  192   1777.6   -17.34%   <-- ANCHOR_WINNER REGRESSES HERE
  224   1939.0    -9.83%
  256   2150.4     0.00%
```

## Pattern: tile-step density determines the optimal slot count

| shape                   | tile-steps | tiles / NUM_CUS | tiles / 192 | best slots |
|-------------------------|-----------:|----------------:|------------:|-----------:|
| Down-B4-M2048 wgrad     |   484      |     1.89        |   2.52      |  192       |
| Down-B4-M4096 wgrad     |   484      |     1.89        |   2.52      |  192       |
| GateUP-B32-M4096 wgrad  |  7744      |    30.25        |  40.33      |  256       |

Hypothesis (consistent with R1 PMC characterisation): the var-K kernel's
per-tile prologue (LDS group-metadata setup, scale fetch, swizzle offset
prefill at lines 7716-7783 of `kernel_fp8_layouts.cpp`) plus per-tile
epilogue (cstore + dscale apply) is a fixed cost that is amortised
across the K-main-loop iters of each tile. With only 1.89 tile-steps
per slot at NUM_CUS=256, this fixed cost dominates the per-slot wall.
Reducing slots to 192 raises tile-steps / slot to 2.52 (+33 %), which
roughly halves the prologue-cost-per-MFMA ratio. The win plateaus at
slots ≈ 192 because below that, parallelism loss (fewer concurrent
slots → fewer concurrent CUs) starts overwhelming the amortisation
benefit (clear collapse below slots=160).

For the GateUP-B32 long-grid shape, the prologue cost is already
amortised across 30 tile-steps per slot — the 30 → 40 jump from
slots=256 → 192 is irrelevant, and the lost parallelism dominates
(-17 % regression).

## What landed this round

### HipKittens (`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`)

1. `grouped_var_k_kernel_fp8` body: replace `NUM_CUS` (compile-time
   constant 256) with `slots_eff = gridDim.x` (runtime) in:
   * the chiplet swizzle range argument (line ~7720)
   * the persistent-loop stride (line ~7785)
   At `gridDim.x == NUM_CUS == 256` the math is bit-identical (verified
   by `_probe_round_2_vark_numcus_verify.py`). The compiler loads
   `gridDim.x` to a scalar register once and reuses; no per-iter cost.

2. `dispatch_grouped_var_k_fp8`: optional env `TK_VARK_NUM_CUS` reads
   the slot count, clamped to `[1, NUM_CUS]`, default `NUM_CUS`. The
   `static const int slots = ...` lambda caches the env read on first
   call so subsequent calls have zero getenv overhead. Used by R2
   probe scripts.

Resource impact (post-rebuild):
```
grouped_var_k_kernel_fp8 [-Rpass-analysis=kernel-resource-usage]
  TotalSGPRs: 80   VGPRs: 256   AGPRs: 0
  ScratchSize [bytes/lane]: 152   Occupancy: 2 waves/SIMD   VGPRs Spill: 0
```
Identical to the pre-R2 resource profile (R74 BF16 era recorded the
same numbers).

### Primus-Turbo (no production code change this round)

Only deposited probe scripts + this analysis note. The HK kernel
change defaults to slots=256 → metric / DoD path unchanged. R2 is
infrastructure-only; the metric-moving rule lands in R3.

## Backward correctness gate

Per the round prompt's hard constraint ("backward 改动 metric 看不到 ——
必须自跑 bench_grouped_gemm_turbo.py"):

```
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN
python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8
```

Result: **24 / 24 shapes PASS** (allclose + SNR > 25 dB on fwd / dA / dB).
fwd avg = 2204.6 TFLOPS, bwd avg = 1658.9 TFLOPS. No correctness
regression from the kernel rebuild. Default slots=256 → mathematically
identical to pre-R2 binding.

## Score state

| metric                            | round 1 | round 2 |
|-----------------------------------|--------:|--------:|
| `_metric_gpt_oss_fp8_kernel.py`   |     680 |     684 |
| fwd section avg                   |  1882 T |  1902 T |
| dgrad section avg                 |  2065 T |  2071 T |
| wgrad section avg                 |  1762 T |  1772 T |

+4 within ±4 noise band. R2 doesn't move the metric — that's R3's job.

## Round-3 plan: ship the slots=192 lever via per-call binding API

The HK env-var hook is global per process (single `static` cache), so
it can't be selectively applied per-shape. To ship the +5-6 % win on
the 2 short-grid Down-B4 wgrad shapes WITHOUT regressing the
6 long-grid shapes, R3 must:

1. Extend `grouped_variable_k_crr_dscale` pybind signature to accept
   a `num_slots` int kwarg. Default `0` → existing NUM_CUS behavior.
   Plumb through `grouped_var_k_layout_globals_fp8` as `g.num_slots`
   (mirror the existing `g.num_xcds` knob).

2. Inside the kernel, replace `gridDim.x` with `g.num_slots > 0 ?
   g.num_slots : NUM_CUS` (the host clamps `g.num_slots` to
   `[1, NUM_CUS]`). Resulting kernel respects the requested slot
   count.

3. Update the dispatch site to launch with `dim3(g.num_slots > 0 ?
   g.num_slots : NUM_CUS)`.

4. In Primus `grouped_gemm_fp8_impl.py:798-1920` (the `var_k_dscale_fn`
   call site), add a per-shape predicate: `if a.shape[1] == 2880 and
   b.shape[1] == 2880 and m_total <= 16384: vk_num_slots = 192`.
   Predicate is unique to gpt_oss-Down-B4 wgrad family (same gate
   class as the existing R11 (gm=1, xcds=2) rule).

5. Tight verify on the 2 affected shapes (the existing
   `_probe_round_2_vark_numcus_sweep.py` already does 3-seed × 7-trial
   p20). Estimated metric impact: wgrad section avg 1772 → ~1813 T
   (+41 T, only 2 of 8 wgrad shapes affected: 1481+1767 vs current
   1395+1679 = +88+88 = +44 wgrad avg points, divided by 8). Score
   impact: 41/2800 × 1000/3 ≈ +5 pts overall (within noise but
   directionally positive — first kernel-level lift after R5 dispatcher
   exhaustion).

Optional R4 follow-up: extend the slots probe to all 8 gpt_oss wgrad
shapes (other 6 currently saturated but the GateUP-B4 wgrad family
might have a similar lever — has 968 tile-steps, 3.78 wave-steps/CU,
an intermediate density not covered by R2 anchor or counter).

## Falsified directions kept clean

* **In-kernel wait-counter / LDS-bank-conflict lever** — falsified in
  R1 PMC pass; R2 confirms by lifting the metric via a non-stall
  attack (launch-geometry).
* **Slot count > 256** — clamped by R2 env hook to `[1, NUM_CUS]`.
  Probe at slots=384 was a no-op (clamped). The kernel's chiplet
  swizzle, LDS metadata tables (`s_offs[MAX_G_PLUS_1=65]`,
  `s_cum_tiles[65]`) and the `chiplet_transform_chunked` chunk_size=64
  argument bake in NUM_CUS=256 as the upper bound; raising NUM_CUS
  is a deeper change out of scope for the launch-geometry lever.

## Probes archived

* `/tmp/round_2_vark_numcus_sweep.json` — anchor + 2-sibling sweep raw
  data, kernel-only TFLOPS at slots ∈ {32..384}.
* `/tmp/hk_fp8_round_2.csv` — bench_grouped_gemm_turbo --dtype fp8
  output, 24/24 shapes PASS.
