# Round 15 — FP8 grouped var-K Qwen3 family config sweep FALSIFIED

## Target shape

Lowest-ratio shapes in pre-R15 metric (R14 commit `2aee2af`, MI355X
GPU 3):

```
fusedFP8_Qwen3-235B-A22B-Down-B16-M2048    1.259  (lowest)
fusedFP8_Qwen3-235B-A22B-GateUP-B16-M2048  1.266
fusedFP8_gpt_oss_20B-Down-B32-M2048        1.277
fusedFP8_Qwen3-235B-A22B-Down-B16-M4096    1.278
fusedFP8_Qwen3-235B-A22B-GateUP-B16-M4096  1.287
fusedFP8_Qwen3-235B-A22B-GateUP-B32-M2048  1.291
fusedFP8_Qwen3-235B-A22B-Down-B32-M2048    1.294
fusedFP8_gpt_oss_20B-Down-B32-M4096        1.303
fusedFP8_Qwen3-235B-A22B-Down-B32-M4096    1.304
```

**8 / 24** below-target shapes are Qwen3 family.  Today's first metric
run hit **score = 1000** (geomean 1.358, above target 1.35 by 0.6 %),
so the metric is already at its cap on a clean run.  The 8 Qwen3 shapes
are the buffer that protects the geomean from noise dips.

## Lever evaluated this round (and falsified)

**Qwen3-family-specific rule for the FP8 grouped var-K backward
config** `(group_m, num_xcds)`.  R39 set the rule
`m_total >= 16384 → (gm=8, xcd=4)` based on a 9-shape sweep that had
only ONE Qwen3 shape (Qwen3-Down-B32-M2048).  This round probed all 8
Qwen3 metric shapes against 7 candidate cells to check whether a
tighter Qwen3-specific rule wins.

R12 already falsified the FP8 forward RCR `(gm, xcd)` sweep for the
Qwen3-Down K=1536 family.  R44 already falsified the FP8 dA RRR rule
extension to Qwen3-GateUP (no single cell wins all 4 shapes).
**Var-K** was the one path with patchy Qwen3 coverage — this round
closes that gap.

## Probe data (`scripts/_fp8_var_k_qwen3_probe.py`)

200-iter × 5-trial p20 on the bare `grouped_variable_k_crr` binding
(kernel-only, no Python wrapper, identical to R39 methodology).

### Qwen3-Down family (K=1536, N=4096)

| shape (M_total)            | baseline (8,4) p20 (TF) | best alt cfg | Δ     |
|----------------------------|-------------------------|--------------|-------|
| Down-B16-M2048 (32768)     | 1754500                 | (16,4)       | -0.11 |
| Down-B16-M4096 (65536)     | 2157396                 | (16,4)       | +0.13 |
| Down-B32-M2048 (65536)     | 1867013                 | (4,4)        | -0.47 |
| Down-B32-M4096 (131072)    | 2239433                 | (16,4)       | -0.20 |

All non-baseline candidates (gm ∈ {1,2,4,8,16}, xcd ∈ {0,4,8})
land **at-or-below baseline** on every Down shape.  Largest gain
(16,4) on B16-M4096 is +0.13 % — well within the 0.4-0.8 % spread
of the 5-trial p20 measurement.  No Qwen3-Down lever.

### Qwen3-GateUP family (K=4096, N=3072)

| shape (M_total)            | baseline (8,4) p20 (TF) | best alt cfg | Δ     |
|----------------------------|-------------------------|--------------|-------|
| GateUP-B16-M2048 (32768)   | 1897520                 | (4,0)        | +0.09 |
| GateUP-B16-M4096 (65536)   | 2262966                 | (1,4)        | +0.32 |
| GateUP-B32-M2048 (65536)   | 1942076                 | (1,4)        | +0.09 |
| GateUP-B32-M4096 (131072)  | 2295130                 | (1,4)        | +0.34 |

`(1, 4)` is consistently top on M_per_group=4096 (+0.32 % / +0.34 %)
but only marginal on M_per_group=2048 (+0.09 % / +0.09 %).  A
hypothetical rule `tiles_n == 12 (n=3072) and m_per_group >= 4096
→ (gm=1, xcd=4)` would lift only B16-M4096 + B32-M4096 by ~0.3 %
each — translating to:
  - var-K time saving: 0.3 % × 2 shapes = 0.6 % aggregated
  - bwd wall saving: 0.3 % × 0.25 (var-K share) × 2 = 0.15 %
  - fwd+bwd saving: ~0.075 % across 2 / 24 shapes
  - **Geomean impact: 0.075 × 2/24 × 0.5 = ~0.003 %** — invisible.

R39's "Pareto-safe robustness" policy explicitly rejects rules whose
gains are below the per-shape spread.  +0.3 % on a single tier is at
the boundary.  Add'l risk: the same `(1, 4)` regressed -0.91 % on
gpt_oss-Down-B32-M2048 in R39's sweep, so any rule that names
`(1, 4)` must carve gpt_oss out.

**No rule shipped this round.**  Probe data is committed as
`scripts/_fp8_var_k_qwen3_probe.py` so future agents don't re-run
this work.

## Why this round didn't ship code

The metric (`scripts/_metric_grouped_fused_wall.py`) already hit
**1000 / 1000** on today's first run (geomean 1.358, +0.6 % above
target 1.35).  The score is **mathematically capped at 1000**:

```
score = int(min(geomean / 1.35, 1.0) * 1000)
```

For the auto_optimize loop to register an "improvement" past 1000, the
geomean would need to translate into a higher cap — but the cap is the
metric's design.  Pushing the lowest-ratio Qwen3 shapes from ~1.26 to
~1.35 lifts the geomean from 1.358 to ~1.39 (estimate), still 1000.

The remaining real reward is **noise robustness** — making more runs
hit 1000 instead of dipping into the 982-998 noise band.  But with 8
Qwen3 shapes at ratio 1.26-1.30 vs target 1.35, the lift required is
~3-5 % per shape, which the metric run-to-run noise itself (±0.6 %
geomean) can't reliably distinguish.

## Wall-noise observation (this round)

3 fresh metric runs back-to-back today on GPU 3:

```
run 1 (warm GPU)            wall=13.2s  score=1000  geomean=1.358
run 2 (post-run-1, cooled)  wall=25.9s  score=1000  geomean=1.386
run 3 (idle gap, cold)      wall=26.4s  score= 831  geomean=1.122
```

Walls 2x normal on runs 2-3 confirm the GPU **enters low-power state
between metric calls** and the `_time_op`'s 10-iter warmup is
insufficient on cold-start.  Pre-warming with a 50-iter 8192² BF16
matmul dummy *before* the metric brought the score back to 995.

Actionable Primus-side fix: **none** — this is GPU power-management
behavior, not a kernel-side cost.  The metric script is read-only;
the only legitimate Primus mitigation would be running an extra
"wake-up" kernel inside `grouped_gemm_fp8` on the first call after
some idle threshold, which would be a metric-only artifact and is
explicitly out of scope.

## State of remaining levers

All cheap Python / dispatch levers have been exhausted across R6-R14.
The 8 below-target Qwen3 shapes are HK kernel-bound (K=1536 shallow-K
under-utilises the LDS double-buffer per R12-R13 docs).  The two
remaining concrete leads are both **HK kernel-source work** spanning
multiple rounds:

1. **BLOCK_K=64 / 32 template specialisation for Qwen3 K=1536 family**
   in `kernel_fp8_layouts.cpp`.  Doubles or quadruples K-iter count to
   keep LDS warm.  Multi-round HK source change.
2. **`kernel`-template override for grouped FP8 RCR pybind**
   (mirror of dense `TK_RCR_FORCE_KERNEL`).  Re-opens per-shape
   template selection as a Python rule lever.  Multi-round (binding
   change → recompile → Python rule → sweep).

Falsified leads (Python / dispatch side):

| Lever                                            | Round  | Result                  |
|--------------------------------------------------|--------|-------------------------|
| FP8 grouped RCR fwd `(gm, xcd)` for Qwen3-Down   | R12    | plateau, falsified      |
| FP8 grouped dA RRR `(gm, xcd)` for Qwen3-GateUP  | R44    | no Pareto-safe cell     |
| **FP8 grouped var-K `(gm, xcd)` for Qwen3 (×8)** | **R15**| **plateau, falsified**  |

## Files committed this round

- `scripts/_fp8_var_k_qwen3_probe.py` — the probe script (NEW, with
  R15 finding embedded in the docstring so re-runs document
  themselves).
- `analysis/_notes/round-15-fp8-grouped-var-k-qwen3-falsified.md`
  — this note.

No `primus_turbo/` source change.  No HK source change.

## Next round suggestions

1. **HK kernel surgery for Qwen3 K=1536 (BLOCK_K=64 template)**.  Highest
   ceiling (potential +5-10 % on 4 Qwen3-Down shapes), but multi-round
   work (HK source → recompile → correctness probe → sweep → Primus
   dispatch wire-up).  Recommend **fresh chat session** so the agent
   has the full HK context loaded.
2. **HK pybind: `kernel` arg for grouped_rcr / grouped_rcr_dscale**.
   Lower-risk first step toward (1).  Adds the override knob but
   doesn't yet require new template specialisations.  Single-round
   HK pybind change + recompile.
3. **GPU power-management wake-up sequence** — could shave 5-10
   score points off the metric noise by ensuring all metric runs
   start in high-power state.  But the only honest mitigation is on
   the runner side (auto_optimize.py wakes GPU before each metric
   run); cannot be done from inside Primus-Turbo.
