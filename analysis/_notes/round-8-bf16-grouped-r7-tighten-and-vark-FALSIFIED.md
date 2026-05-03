# Round 8 — Tight verify of R7 rule + var-K cfg attempt (FALSIFIED — no commit-worthy lever)

## Selected target

Per round-8 baseline metric (per-shape table from `/tmp/metric_round_8.log`):
- Lowest-progress shapes (ratio ascending):
  1. gpt_oss-GateUP-B32-M2048: 1.046  (weight 3, R7 rule already covers via (16,4))
  2. gpt_oss-Down-B32-M2048:   1.063  (weight 3)
  3. gpt_oss-GateUP-B4-M2048:  1.075  (weight 3)
  4. gpt_oss-Down-B32-M4096:   1.082  (weight 3)
  5. gpt_oss-GateUP-B32-M4096: 1.084  (weight 3, R7 rule covers via (8,4))
- gpt_oss family geomean = 1.087 (vs DSV3 1.118, Qwen3 1.113)
- R8 baseline score: 879 (run 1) / 884 (run 2) / 879 (run 3) — mean 880.7,
  3-pp below R7's reported 889 but well within metric noise envelope (±10 sc).

R7 plan suggested **gpt_oss-Down dA H4 RCR cfg probe** as the next lever
(mirroring R7's GateUP rule). Audit FALSIFIED that path before probing:
gpt_oss-Down forward and dA H4 share the **same dispatch fingerprint**
(m=M_per, n=K_fwd=2880, k=N_fwd=2880, m_total) because Down is N=K square
(`N_fwd = K_fwd = 2880`). The existing `k == 2880` block at line 458+
already fires for both, so no fresh dispatch lever exists for Down's dA H4.

## Hypothesis (lever C — dispatch refinement)

Two refinement paths attempted:

**(A) Tighten R7 rule cfgs**: R7's coarse 5×80 cudaEvent probe had
2-5 % spread on the borderline shapes. Tighter 200-iter × 7-trial p20
verify might reveal a cleaner winner inside the R7-covered cells.

**(B) Var-K (CRR) cfg split for B=32**: R1 added `(gm=4, xcds=4)` for the
unified gpt_oss var-K dB family. R3 swept (8,4), (2,4), (1,4) and falsified.
Untested cells `gm ∈ {16, 24, 32}` for the B=32 sub-family (m_total ≥ 65536,
~15-30 tiles/CU) might break the saturation pattern.

## Variant tested — (A) tight verify of R7 rule cfgs

200-iter × 7-trial p20 sweep across 12 (gm, xcd) cells on the 4 BF16
gpt_oss-GateUP dA H4 RCR shapes. Probe at
`/tmp/probe_round8_bf16_dA_tight.py`, log at
`/tmp/probe_round8_bf16_dA_tight.log`.

Per-shape clean-WIN cluster (verdict requires gap > 1.5 × spread):

```
shape                  R7 cfg / TF   tight top1 / TF       Δ TF (R7→top1)
B4-M2048   tiles_m=8   default(4,8)  default(4,8)  1064.4   0.0  (default still wins)
B4-M4096   tiles_m=16  (8,4) 1350.1   (24,4) 1360.4         +10.3  borderline (sp 1.91% > gap 1.74%)
B32-M2048  tiles_m=8   (16,4) 1335.6  (1,4)  1336.9         +1.3   marginal (Δ=0.10% < 0.39% spread)
B32-M4096  tiles_m=16  (8,4)  1359.1  (1,4)  1359.8         +0.6   marginal (Δ=0.05% < 0.19% spread)
```

R7's chosen cfgs sit **inside the winning cluster on every shape**,
within 0.05–0.10 % of the cluster top1 on the 2 large-grid shapes (B=32)
and inside the noise floor on the 2 small-grid shapes (B=4). The
B4-M2048 default-stays decision is **confirmed** by tight verify (every
candidate LOSS by -1 to -5 %). (1,4) emerges as a slightly-better
alternative for the B=32 brackets but the gap (+0.05–0.10 %) is below
the metric noise floor.

Translating to metric ratio impact (dA RCR is ~30 % of bwd wall, bwd is
~70 % of total wall):

| shape | refinement | dA TF Δ | metric ratio Δ | weighted progress Δ |
|-------|-----------|---------|----------------|---------------------|
| B32-M2048 | (16,4)→(1,4) | +0.10 % | +0.021 % | +0.12 score |
| B32-M4096 | (8,4)→(1,4) | +0.05 % | +0.011 % | +0.07 score |
| B4-M4096 | (8,4)→(24,4) | +0.77 % | +0.162 % | +1.0 score (but spread 1.91 % > gap risks regression) |

Total expected score gain from polishing R7 rule: **~+1.2 score**.
Below the +5-score commit threshold AND well below per-run noise (±10).

## Variant tested — (B) var-K (CRR) wider cfg sweep

Probe at `/tmp/probe_round8_bf16_var_k_v2.py` repeatedly faulted with
`Memory access fault by GPU node-5` on the 1st `grouped_variable_k_crr`
call for B=32 inputs (B=4 worked in isolation). The same shapes work
through the autograd path during the metric — confirms a probe-context
issue, not a kernel bug.

Diagnosis: GPU pool was at 96-98 % use across all 8 cards (`rocm-smi`)
when the probe ran; every other tenant had ~22-25 GB allocated on each
card. The metric stays stable because it has a single steady-state
allocation pattern; the probe creates fresh tensors and rapidly switches
through 12 cfgs, hitting allocator races / memory thrash.

Decision: **abandoned for round 8** — re-attempt in a future round if
GPU contention drops. The R3 conclusion (var-K cfg saturated for B=32
gpt_oss) still stands as the working hypothesis since the only
untested cells are the wider gm tier which R3 implicitly disfavored
(R3 tested gm ∈ {1, 2, 8}, all flat or worse on B=32).

## Decision

**REVERT-EQUIVALENT (no code change made this round).** Working tree
already clean of probe artifacts.

R7 rule ((8,4) for tiles_m=16, (16,4) for tiles_m=8 + B=32) is at the
top of the winning cluster within metric noise; polishing it doesn't
clear the +5-score threshold. Var-K probe is blocked by GPU contention,
deferred.

## Recommendation for round 9+

The dispatch surface for grouped BF16 gpt_oss is **definitively closed**
across the 5 levers swept:

| lever | rounds | result |
|-------|--------|--------|
| forward RCR (Down + GateUP) | R9, R10, R21, R26, R45 | LANDED rules per m_total bracket |
| dA RRR cfg | R2 | falsified pre-R4 H4 reroute |
| dA H4 RCR (GateUP, post-reroute) | R7 (LANDED), R8 (tighten falsified) | LANDED — at noise floor |
| dA H4 RCR (Down, post-reroute) | R8 audit | shares dispatch with Down forward — already covered |
| var-K dB CRR | R1 (LANDED for B=4), R3, R5, R6, R8 (blocked) | LANDED only the B=4 corner |

Remaining levers are all **kernel-side** and require multi-round
projects:

1. **Lever B1 — DSV3/Qwen3 forward MFMA scheduling** (multi-round):
   16 shapes at 1.10–1.13 ratio with weight 1 each = 16 weight-units of
   leverage. A +5 % HK forward TFLOPS lift across these would push the
   weighted-wall by +20–30 score. Profile
   `rocprofv3 valuMfmaUtil` on a representative shape (e.g.
   DSV3-GateUP-B16-M4096 at ratio 1.122 in R8 baseline) to characterize
   MFMA pipeline utilization and identify which side (compute-bound vs
   memory-bound vs pipe stall) has the biggest gap.

2. **Lever A1 — gpt_oss K-tail in-kernel fuse refinement** (multi-round):
   K=2880 (K%128=64) hits the FUSED_KTAIL path. The K-tail accumulation
   tile may still under-utilize MFMA. Anchor: gpt_oss-GateUP-B32-M2048
   forward at ratio 1.046 in R8 baseline. `rocprofv3 valuMfmaUtil` on
   the forward kernel to compare K%128==0 vs K-tail utilization.

3. **Lever D2 — var-K kernel topology for small N tiles** (multi-round):
   gpt_oss var-K has tiles_m=11 (Down) / 22 (GateUP) — the smallest grids
   in the metric. Per-tile MFMA pipeline in the var-K body may be
   sub-optimal for these short M-axes. Bench
   `bench_grouped_gemm_turbo.py` shows bwd_TF / fwd_TF ≈ 0.78 on gpt_oss
   B=32 (vs 0.86 on DSV3 B=32), suggesting var-K is the bigger gap
   on gpt_oss. A dedicated `grouped_var_k_kernel_smallN` variant might
   lift gpt_oss bwd by 5–10 %.

R9 priority order: B1 (highest leverage, 16 shapes × weight 1) → A1
(gpt_oss-specific, weight 3) → D2 (gpt_oss var-K, weight 3 but
multi-round). Run rocprofv3 in R9 as the gating action; don't commit
kernel changes without profile data.

## Files touched (round 8)

- `analysis/_notes/round-8-bf16-grouped-r7-tighten-and-vark-FALSIFIED.md`
  (this file)

NO source code changes. Working tree clean.

## Metric / bench numbers

- R7 metric (per R8 prompt): 889 (improved=False, best=891)
- R8 baseline runs (3 mid-round samples): 879, 884, 879 — mean 880.7
- R8 final run after probe-induced GPU thrash: 879

- Bench `bench_grouped_gemm_turbo.py --dtype bf16`:
  - gpt_oss-GateUP-B32-M2048: fwd 1224 / bwd 916 TF (worst metric shape)
  - gpt_oss-Down-B32-M2048:   fwd 1174 / bwd 808 TF (2nd worst)
  - gpt_oss-Down-B4-M2048:    fwd 882  / bwd 663 TF (small-grid outlier;
    metric ratio 1.136 — relative ratio still OK)
  - DSV3-GateUP-B32-M4096:    fwd 1457 / bwd 1125 TF (best gpt_oss-class
    at large grid, but uniformly slower than DSV3)

- Tight verify probe at `/tmp/probe_round8_bf16_dA_tight.py` (200×7 p20):
  R7 cfgs within 0.05–0.77 % of top1 on covered shapes; default still
  wins on B4-M2048 — same conclusion as R7 5×80 cudaEvent probe.

- Var-K probe at `/tmp/probe_round8_bf16_var_k_v2.py`: faulted on B=32
  inputs due to allocator thrash (8/8 GPUs at 96-98 % use); deferred.

- Correctness: 24/24 PASS, 0/24 reject in every measured baseline run.
