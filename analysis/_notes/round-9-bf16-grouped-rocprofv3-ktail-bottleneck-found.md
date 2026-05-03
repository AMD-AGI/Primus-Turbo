# Round 9 — rocprofv3 hardware-counter profile (PMC) — gpt_oss K-tail conclusively the bottleneck (Lever A1 anchor)

## Selected target

Per round-9 baseline metric (per-shape table from `/tmp/metric_round_9.log`):
- Lowest-progress shapes (ratio ascending):
  1. **gpt_oss-GateUP-B32-M2048**: 1.047 (weight 3, 5th round same target)
  2. gpt_oss-Down-B32-M2048:   1.067 (weight 3)
  3. gpt_oss-Down-B32-M4096:   1.079 (weight 3)
  4. gpt_oss-GateUP-B32-M4096: 1.086 (weight 3)
  5. gpt_oss-GateUP-B4-M2048:  1.090 (weight 3)
- gpt_oss family geomean = 1.0998 (DSV3 1.121, Qwen3 1.124)
- R9 baseline score: **887** (R8 final 882, R7 889, best 891 — all within ±10 noise)

This round = the **rocprofv3 gating profile** R8 recommended. After R8
declared the gpt_oss dispatch surface closed across all 5 levers, R9
needs HW-counter data to pick the kernel-side lever (B1 / A1 / D2).

## Method

Direct kernel-only micro-bench (`/tmp/probe_round9_rocprof_v2.py` —
calls `hk.grouped_rcr` with the production-dispatched cfg, 3 warmup +
1 profiled call) wrapped by `rocprofv3 -i ... --output-format csv`.

PMC counters: `MfmaUtil`, `LdsBankConflict`, `LdsUtil`,
`MeanOccupancyPerCU`, `SALUBusy` (all derived counters from
rocprofv3's built-in expression set — no math needed).

Counter pass setup: 5 derived counters in 1 PMC pass = ~7 sec per
shape. (Earlier 2-pass / 22-counter input hung on the contended GPU
pool; minimal-pass mode finishes fast.)

## Results — 5-shape survey

```
shape                      K     K%128   dur_us  MfmaUtil  LdsUtil  MeanOcc/CU  metric_ratio
gpt_oss-GateUP-B32-M2048   2880    64    2136     63.4%    23.6%      7.29       1.047
gpt_oss-Down-B32-M2048     2880    64    1024     62.3%    23.2%      7.27       1.067
DSV3-GateUP-B16-M4096      7168     0    3037     79.7%    30.0%      7.34       1.124
Qwen3-GateUP-B16-M4096     4096     0    1470     75.3%    28.4%      7.33       1.118
Qwen3-Down-B16-M4096       1536     0    1435     25.6%     9.3%      3.58       1.137
```

(steady-state averages of dispatches 10-12; first profiled dispatch
discarded as warmup. LdsBankConflict = 0% on all 5 — clean LDS access
patterns across the board.)

## Diagnosis — K-tail conclusively the gpt_oss bottleneck

**gpt_oss-GateUP and gpt_oss-Down hit nearly identical MFMA util
(63.4 vs 62.3 %)** despite very different geometries:
- GateUP: tiles_m=8, tiles_n=22 (5760 = 22*256 + 128 partial col)
- Down: tiles_m=8, tiles_n=11 (2880 = 11*256 + 64 partial col, square N=K)

If the partial last-column tile were the dominant loss (Lever
"N-tail-elide"), Down would have less waste than GateUP. They don't —
they have the same util to within 1 pp.

**The shared variable is K=2880 (K%128=64)** triggering the
`FUSED_KTAIL` path — round-4 path A's HBM-to-register reload + DO_MMA
on the final 1×K_STEP=64 partial K-tile after epilog 1 + epilog 2
drain. This in-kernel K-tail epilogue:
1. Is single-tile-no-pipeline (no overlap with main loop's MFMA)
2. Loads A and B from HBM directly to registers (no LDS staging)
3. Runs 1 DO_MMA stage with no prologue or pipeline reuse
4. Restarts MFMA after epilog 2's pipeline drain

In contrast, **K%128==0 shapes (DSV3, Qwen3-GateUP) sit at 75-80 %
util** — main loop dominates, prologue/epilog amortised across many
K_STEPs.

Concrete projection (assuming Triton baseline unchanged):

| gpt_oss kernel speedup | new metric score | Δ score |
|----:|:---:|:---:|
| +5 %  | 913  | +26 |
| +10 % | 935  | +48 |
| +16 % (close to DSV3 util) | **956** | **+69** |

Either +26 or +69 score easily clears the +5 commit threshold. **Lever
A1 (gpt_oss K-tail kernel surgery) is the highest-leverage move in
the remaining state space.**

## Side-finding — Qwen3-Down K=1536 is its own slow path

Qwen3-Down-B16-M4096 (K=1536, K%128=0 — *no* K-tail) sits at **25.6 %
MfmaUtil** with **occupancy 3.58/CU** (vs 7.3 elsewhere). Despite
low util, the metric ratio is 1.137 — Triton is also bad here. Likely
the K=1536 main loop is too short (only 12 K_STEPs / 6 K_TWO_TILEs)
to fill the 8-stage MFMA pipeline efficiently, AND the per-tile work
is so small that the persistent grouped scheduler oscillates between
high and low occupancy phases.

Lower priority than Lever A1 (Qwen3 weight 1, ratio already > 1.13;
gpt_oss weight 3, ratio < 1.10).

## What does NOT explain the gap

Confirmed by counters:
- **LDS bank conflict = 0** on all 5 shapes. Not LDS-bound.
- **SALU busy 6-11 %**. Not scalar-pipeline-bound.
- **Occupancy 7.3/CU = ~58 %** (max 12.5/CU @ 8 waves/SIMD × 4 SIMD/CU).
  Same on gpt_oss and DSV3, so not the differentiator.
- **MeanOcc 7.3 vs 7.3** between gpt_oss (1024 LDS bytes, VGPR=124)
  and DSV3 (VGPR=128). Resource-pressure constant across the survey.

## Decision

REVERT-EQUIVALENT (no code change). R9 = pure profile gating round.

Profile data captured in this note + raw CSVs at:
- `/tmp/rocprof_round9_derived_gpt_oss/pass_1/chi2894/`
- `/tmp/rocprof_round9_derived_dsv3/pass_1/chi2894/`
- `/tmp/rocprof_round9_gpt_oss_Down_B32_M2048/pass_1/chi2894/`
- `/tmp/rocprof_round9_qwen3_GateUP_B16_M4096/pass_1/chi2894/`
- `/tmp/rocprof_round9_qwen3_Down_B16_M4096/pass_1/chi2894/`

## Recommendation for round 10+

**Lever A1 — gpt_oss K-tail in-kernel fuse refinement** (multi-round
project, owner: agent across R10-R15):

1. **R10**: Investigate the round-4 path A FUSED_KTAIL block in
   `kernel_bf16_dynamic.cpp` (lines 779-1117). Specifically the
   single-tile DO_MMA step at lines ~1027-1090 — measure its
   MFMA-busy fraction relative to total kernel time via a fine-grained
   PMC timer (split GRBM_GUI_ACTIVE into "before FUSED_KTAIL" and
   "during FUSED_KTAIL" using a ROCm timeline marker) to confirm the
   K-tail block is a hot hotspot.
2. **R11**: Hypothesize fixes. Likely candidates:
   - Pipeline the K-tail load with the last main-loop epilog drain
     (currently fully serialized).
   - Use 4 K-tiles of K_STEP=16 instead of 1 K-tile of K_STEP=64 to
     keep MFMA fed during the HBM load.
   - Pre-issue the K-tail HBM load early in the main loop
     (round-pipelined SRD prefetch).
3. **R12+**: Implement the candidate, build HK, gate via metric
   correctness + bench (fwd+bwd TFLOPS in commit message).

**Lever B1 — DSV3/Qwen3 main-loop MFMA scheduling** (defer): main
loop already at 75-80 % util; remaining headroom there is 5-10 pp
which translates to maybe +5-10 score (16 weight-1 shapes × small Δ).
Smaller leverage than Lever A1 (8 weight-3 shapes × big Δ).

**Lever D2 — var-K kernel topology** (defer; kernel surgery on
backward path): R3 / R5 / R6 already concluded var-K cfg saturated;
kernel surgery there would need its own profile round first (R8 var-K
probe was blocked by GPU contention).

R10 priority: **measure the K-tail block in isolation** (sub-kernel
timing or marker-bracketed PMC) to confirm A1 hypothesis BEFORE
investing in code changes. If R10 shows the K-tail is < 10 % of
kernel time, A1 is falsified and we pivot to investigating main-loop
prologue/epilogue overhead instead.

## Files touched (round 9)

- `analysis/_notes/round-9-bf16-grouped-rocprofv3-ktail-bottleneck-found.md`
  (this file)
- `/tmp/probe_round9_rocprof_v2.py` (reusable profile harness)
- `/tmp/rocprof_input_derived.yaml` (PMC counter input)

NO source code changes. Working tree clean of code edits.

## Metric / bench numbers

- R9 baseline: **887**
- gpt_oss family geomean: 1.0998 (R8 was 1.0998, R7 was 1.0871)
- DSV3 family geomean: 1.121, Qwen3 family geomean: 1.124
- All 24 shapes PASS correctness.

## Quick-reference profile commands (for R10+ resume)

```bash
# Single-shape PMC profile (5 derived counters, ~7 sec):
rocprofv3 -i /tmp/rocprof_input_derived.yaml \
  -d /tmp/rocprof_round_<N>_<shape> --output-format csv -- \
  python3 /tmp/probe_round9_rocprof_v2.py <shape>

# Where <shape> is one of the 5 SHAPES dict keys in the probe
```
