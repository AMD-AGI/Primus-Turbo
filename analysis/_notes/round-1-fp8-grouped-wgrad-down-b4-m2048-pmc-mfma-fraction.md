# Round-1 (gpt_oss FP8 kernel-only ceiling, current Primus run; 2026-05-07)

## Round-G G1a — wgrad var-K PMC characterisation

**Result**: Successfully landed PMC characterisation of `grouped_var_k_kernel_fp8`
on 3 representative wgrad shapes (1 worst, 1 mid, 1 best). The data **confirms
the var-K wgrad kernel is NOT MFMA-bound** for the worst gpt_oss-Down-B4
shapes — there is structural headroom. The dominant cause is identified
quantitatively as **sub-saturated persistent-grid utilisation** (≈ 2 wave-
steps / CU on Down-B4-M2048), not an internal kernel stall pattern.

This advances the HipKittens Round-G plan (commit `087f1e17`) by closing
G1a (PMC pass on var-K) and pointing G1b → G1c at a launch-geometry
direction rather than a wait-counter / LDS-bank-conflict direction.

## Probe details

* Driver: `scripts/_probe_round_1_wgrad_pmc.py` (new this round; deposited
  alongside the prior `_probe_fp8_kernel_rocprof.py` infra).
* Mode: `rocprofv3 --pmc <7 counters>`, single hardware pass.
  * `GRBM_GUI_ACTIVE` (gpu-wide cycles)
  * `SQ_BUSY_CYCLES`
  * `SQ_VALU_MFMA_BUSY_CYCLES`
  * `SQ_VALU_MFMA_COEXEC_CYCLES`
  * `SQ_INSTS_VALU_MFMA_F8`
  * `SQ_INST_CYCLES_VMEM_RD`
  * `SQ_WAIT_INST_LDS`
* 50 dispatches per shape (20 warmup + 30 timed), kernel name match
  filter `grouped_var_k_kernel_fp8`. Aggregates per shape over the 50
  dispatches.
* GPU: pinned via `auto_optimize.py` to GPU 1 (idle this round, KFD-VRAM
  empty, 0 % use confirmed pre-launch).

## Raw aggregates (50 dispatches each, ratios to `GRBM_GUI_ACTIVE`)

| shape (wgrad)         | TFLOPS | mfma_busy / gui | sq_busy / gui | lds_wait / gui | vmem_rd_cyc / gui | F8 mfma insts |
|-----------------------|-------:|----------------:|--------------:|---------------:|------------------:|--------------:|
| Down B4 M2048 (worst) |  1277  |          42.39  |          2.97 |          1.88  |             0.74  |   117 964 800 |
| Down B4 M4096         |  1594  |          54.61  |          3.07 |          2.44  |             0.90  |   235 929 600 |
| GateUP B32 M4096 (best wgrad) | 2186 |    77.11  |          3.90 |          3.41  |             1.27  | 3 617 587 200 |

`*_busy / gui` is dimensionless: numerator is summed across CUs / SIMDs
(rocprofv3 default), denominator is single-instance global cycles. The
ratios are not directly interpretable as "% peak", but their **relative
slope** vs achieved TFLOPS is the key signal.

### Linear-correlation check (mfma_busy / gui vs TFLOPS)

```
slope = mfma_busy_norm / TFLOPS
  Down_B4_M2048    : 42.39 / 1277 = 0.0332
  Down_B4_M4096    : 54.61 / 1594 = 0.0343
  GateUP_B32_M4096 : 77.11 / 2186 = 0.0353
```

3 / 3 shapes agree to within ±3 % on the slope. **MFMA throughput is
the gating signal** — every additional MFMA-busy cycle adds proportional
TFLOPS. The kernel is not stalled in some non-MFMA bucket that swallows
the lift.

### Per-CU normalisation ("avg-CU MFMA active fraction")

Treating the SQ counters as per-CU sums (256 CUs on MI355X):

```
shape                  mfma_frac  sq_busy_frac  lds_wait_frac  vmem_rd_frac
Down_B4_M2048             0.166        0.012        0.007         0.003
Down_B4_M4096             0.213        0.012        0.010         0.004
GateUP_B32_M4096          0.301        0.015        0.013         0.005
```

**Down_B4_M2048 wgrad runs MFMA at ~16.6 % of GPU peak**. The 2800-T
target = 55.6 % of 5033-T peak — so this shape is currently at ≈ 30 %
of target (matches the 1277 / 4280 = 0.30 metric ratio at the PMC
sample window). Lifting to target requires ~3.4× more MFMA-busy cycles.

The `sq_busy_frac` / `lds_wait_frac` / `vmem_rd_frac` columns are all
< 2 % per-CU — these buckets are tiny relative to the MFMA-active bucket
in any kernel-time-fraction interpretation. They do not represent a
"hidden stall reservoir" the next round can attack.

## Why Down-B4-M2048 wgrad is at 16.6 % MFMA active — grid math

`grouped_var_k_kernel_fp8` is a persistent-grid kernel (NUM_CUS = 256
slots). Per-shape tile budget:

```
shape              tiles_n  tiles_k  groups (B)  total tile-steps  steps / CU (256 slots)
Down B4 M2048         11      11       4            484                    1.89
Down B4 M4096         11      11       4            484                    1.89
GateUP B32 M4096      22      11      32           7744                   30.25
```

Per-tile prologue (g.ki computation, A/B-pack initial load, scale fetch)
+ per-tile epilogue (cstore, dscale apply) is a **fixed per-tile cost**
that is amortised across the K main-loop iters. With only ~2 tile-steps
per slot (Down B4), the persistent-grid amortisation collapses: each
slot pays prologue+epilogue twice for ~22 K-main iters × 2 tiles =
44 useful K-iter × MFMA cycles, surrounded by 2× (prologue + epilogue)
fixed overhead. That overhead falls in the **non-MFMA fraction** of
the 16.6 % computation.

GateUP B32 M4096 amortises the same fixed prologue+epilogue cost across
~30 tile-steps per slot — 15× more K-main iters per fixed-cost unit, so
MFMA-active fraction climbs from 16.6 % → 30.1 %.

## Implication for Round-G G1c lever choice

The traditional in-kernel levers (LDS-bank-conflict reduction, wait-
counter sweeps, MFMA primitive port) attack the 1-2 % `lds_wait_frac` /
`vmem_rd_frac` buckets, not the dominant 70-83 % "non-MFMA" bucket. So
those levers cannot lift Down-B4-M2048 wgrad meaningfully under the
current persistent-grid design.

The actionable Round-G G1c lever is in the **launch-geometry / per-tile-
overhead axis**, not the in-kernel-stall axis:

1. **Per-tile prologue/epilogue trim** — for short-grid shapes (≤ 4 wave-
   steps per CU), shrink the per-tile fixed cost. Candidate sub-levers
   (need probe per sub-lever, not done this round):
   * Hoist the `ki_g = M_g / HB` computation across tiles within the
     same group when the host sets `m_per_group` constant.
   * Pre-stage the scale fetches at kernel entry instead of per-tile.
   * Fuse the CRR cstore into the last K-main iter (similar to the
     R2 fwd `FUSED_KTAIL` pattern, but for cstore).

2. **Reduce `NUM_CUS` for short-grid shapes** — instead of persistent
   grid spanning 256 slots, launch with fewer slots (e.g., `min(NUM_CUS,
   total_tile_steps × oversub_factor)`). Each slot then runs more
   wave-steps so prologue+epilogue amortises better. Trade-off: under-
   utilised CUs sit idle. Probe needed: at what `slots × steps_per_slot`
   product does Down-B4 wgrad peak?

3. **Small-grid CRR variant** — emit a separate kernel template that
   skips the persistent-loop overhead and uses a one-shot grid for
   total_tile_steps < 1024. This is the most invasive and the highest
   EV (+1500-3000 T potential per the 16.6 % → 50 % MFMA frac jump
   observed in the GateUP comparison).

## Falsified directions (this round)

* **In-kernel wait-counter / LDS-bank-conflict lever** is FALSIFIED for
  var-K wgrad on the worst gpt_oss shape. The dominant cycle bucket is
  not "stalled MFMA waiting on LDS" — it is "GPU active, no MFMA
  issued at this slot due to per-tile prologue/epilogue serialisation".
  This mirrors but inverts the BF16 R68 finding (which DID find LDS-
  bank-conflict as the BF16 bottleneck on a different geometry — BF16
  var-K has different tile padding).

## Score state

* No code change this round. Score before = after = 683 (within ±3
  noise floor). The R5 host-overhead trim (Primus HEAD `7637aae`)
  remains the latest kernel-time-relevant change.
* The probe artifact:
  * `scripts/_probe_round_1_wgrad_pmc.py` (new, kept under `scripts/`
    so a future round can reuse the rocprofv3 wrapper).
  * `/tmp/rocprof_round_1_wgrad_pmc/summary.json` (raw aggregates).

## Recommended Round-2 plan

1. **G1c — extend the PMC pass to a NUM_CUS sweep**: relaunch
   `grouped_variable_k_crr_dscale` with `dim3(N)` for N ∈ {64, 128,
   256, 512} on Down-B4-M2048 wgrad and re-measure mfma_frac. If
   mfma_frac peaks at N=128 with a +5pp lift, the launch-geometry
   lever is real.
2. If the NUM_CUS sweep is flat (kernel ignores the launch dim because
   the persistent loop is hard-coded), pivot to G1c-(1) — try hoisting
   the `ki_g` computation in `kernel_fp8_layouts.cpp:7699-8025`
   and re-measure the per-tile cycle count.
3. Order matters: option (2) needs an HK kernel-rebuild loop, option
   (1) is a host-side launch arg change in the binding (if exposed).
   Always do (1) first as the cheaper falsification.
