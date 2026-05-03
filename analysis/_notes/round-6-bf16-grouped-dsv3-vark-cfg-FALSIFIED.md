# Round 6 — DSV3 var-K (CRR / dB) cfg rule (FALSIFIED) — closes R5+R6 dispatch sweep

## Selected target

Per round-6 baseline metric (per-shape table from
`/tmp/metric_round_6.log`):
- Lowest-progress shapes (ratio ascending):
  1. gpt_oss-GateUP-B32-M2048: 1.045 (weight 3, 4th round same target)
  2. gpt_oss-Down-B32-M2048:   1.046 (weight 3)
  3. gpt_oss-Down-B4-M4096:    1.075 (weight 3)
  4. gpt_oss-GateUP-B32-M4096: 1.081 (weight 3)
- Round-6 starting baseline: ~881 single (mean ~886 over recent runs);
  best historical 884.
- gpt_oss family geomean = 1.0840
- DSV3 family geomean    = 1.1212
- Qwen3 family geomean   = 1.1218

## Hypothesis (lever C — dispatch, completing round 5's pivot)

Round 5 found Qwen3 var-K cfg saturated at default `(4, 8)`. The
parallel question: is DSV3 var-K cfg also saturated, or is there a
DSV3-specific win? Audit confirmed DSV3 var-K dB tile geometries
were also falling through to default:
* DSV3-GateUP: K_fwd=7168 → tiles_n=28; N_fwd=4096 → tiles_m=16
* DSV3-Down:   K_fwd=2048 → tiles_n=8;  N_fwd=7168 → tiles_m=28

No prior CRR rule matches `tiles_m=16 ∧ tiles_n=28` (R1 gpt_oss is
tiles_n=11, R5 Qwen3 is tiles_n ∈ {6, 16}, dense LLaMA is
tiles_n=16 ∧ tiles_m≥32).

Hypothesis: same `(gm=4, xcds=4)` mirror — to definitively rule out
CRR cfg as a lever before pivoting to kernel-side work.

## Variant tested

Combined rule for both DSV3 sub-families:
```python
if (
    layout == "crr"
    and (
        (tiles_n == 28 and tiles_m == 16)  # DSV3-GateUP
        or (tiles_n == 8 and tiles_m == 28)  # DSV3-Down
    )
    and k <= 4096
    and m_total is not None
    and m_total >= 32768
):
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

## Bench results (per-shape backward TFLOPS, same-window comparison)

`bench_grouped_gemm_turbo.py --dtype bf16` (100-iter, 20 warmup).
Same GPU contention level for both runs (back-to-back).

```
shape                            base bwd_TF  R6 bwd_TF   Δ TF
DSV3-GateUP-B16-M2048              1007.47       999.42    -8.05
DSV3-Down-B16-M2048                1025.14      1021.02    -4.12
DSV3-GateUP-B16-M4096              1124.49      1119.90    -4.59
DSV3-Down-B16-M4096                1166.58      1168.06    +1.48
DSV3-GateUP-B32-M2048              1008.05       999.95    -8.10
DSV3-Down-B32-M2048                1022.23      1016.11    -6.12
DSV3-GateUP-B32-M4096              1121.25      1107.81   -13.44
DSV3-Down-B32-M4096                1151.06      1145.39    -5.67

GateUP family (4 shapes): mean Δ = -8.5 TF (consistently negative)
Down family   (4 shapes): mean Δ = -3.6 TF (mostly negative)

All 8 DSV3 shapes Δ ≤ +1.5 TF (7 of 8 negative).

Average BF16 backward TFLOPS: 985.79 -> 982.40  (Δ -3.39, regression)
Forward TFLOPS:               1244.95 -> 1241.95 (~unchanged)
Correctness: 24/24 PASS (fwd, bwd_x, bwd_w all True per shape).
```

## Diagnosis — DSV3 var-K cfg space saturated (largest grids)

DSV3 var-K shapes have the LARGEST grids in the metric:
* DSV3-GateUP-B16: tiles_per_group = 28*16 = 448,
  total = 16*448 = 7168 tiles, ~28 tiles/CU at xcds=8.
* DSV3-GateUP-B32: 32*448 = 14336 tiles, ~56 tiles/CU.
* DSV3-Down-B16:   16*224 = 3584 tiles, ~14 tiles/CU.
* DSV3-Down-B32:   32*224 = 7168 tiles, ~28 tiles/CU.

At ≥14 tiles/CU, xcds=8 fully saturates (each XCD averages
~3-7 tiles/CU/XCD with ki_g=32 K-iters per tile). Reducing to
xcds=4 doubles per-XCD load AND halves the inter-XCD
parallelism — net regression of ~5-13 TF per shape.

The pattern across rounds 1, 5, 6 is now clear:

| family | tiles/CU @ xcds=8 | (4,4) result vs default |
|--------|-------------------|-------------------------|
| gpt_oss B=4    | ~4                | **+ win** (R1, +1.2 % geomean) |
| gpt_oss B=32   | ~12-30            | flat to slight + (R1, "noise") |
| Qwen3 B=16     | ~6-12             | **flat / slight regression** (R5, -4 score) |
| Qwen3 B=32     | ~12-24            | **flat / regression** (R5)    |
| DSV3 B=16      | ~14-28            | **regression** (R6, -3.4 TF avg) |
| DSV3 B=32      | ~28-56            | **regression** (R6, -8.5 TF GateUP) |

Conclusion: `(gm=4, xcds=4)` only wins on the **B=4 small-grid corner
(tiles/CU < ~6)**. All other var-K tile geometries have enough
work to saturate xcds=8; reducing parallelism hurts.

## Decision

REVERT. (Working tree clean — no config.py change kept.)
Falsification note committed (this file).

**Lever C (CRR var-K dispatch) is now definitively closed.** Round 1
captured the only positive corner (gpt_oss B=4); rounds 5 and 6
swept the remaining 4 sub-families and confirmed all are saturated.
No further CRR cfg work should be attempted without a kernel-side
change to the var-K topology.

## Recommendation for round 7+

The dispatch surface for grouped BF16 is now exhausted across both
forward (rounds 9, 21, 26, 70, 45-style sweeps) and backward (rounds
1, 5, 6 var-K sweeps + round 2 dA H4 RCR + round 3 var-K split).
The remaining attack vectors are all kernel-side:

1. **Lever D step 2 — var-K kernel topology** (multi-round, similar
   effort to round 4's H4 fast transpose): the var-K kernel uses
   `device_gemm_tile_body<CRR>` shared with forward; CRR's B-side
   load pattern may be sub-optimal for gpt_oss's small-N tiles
   (N_fwd=2880 → tiles_m=11). Investigate per-tile MFMA pipeline
   in the var-K body; a dedicated `grouped_var_k_kernel_smallN`
   variant might lift gpt_oss bwd by 5-10 %.

2. **Lever A1 — in-kernel masked K-tail loop** (multi-round): the
   round-3 path A `FUSED_KTAIL` is already enabled for gpt_oss
   forward (K=2880, K%128=64). But the K-tail accumulation tile
   may still under-utilize MFMA pipeline. Profile with
   `rocprofv3 valuMfmaUtil` on the worst gpt_oss-B32-M2048 fwd
   to characterize.

3. **Lever B1 — DSV3/Qwen3 forward MFMA scheduling** (multi-round):
   16 shapes at 1.10-1.13 ratio; +5 % HK forward TFLOPS would lift
   metric ratio +5 % across all 16 = +20-30 score ceiling. Profile
   `rocprofv3 valuMfmaUtil` on a representative shape (e.g.
   DSV3-GateUP-B16-M4096 ratio 1.122) to confirm headroom exists.

Round 7 recommendation: **rocprofv3 profile** the 3 worst metric
shapes (gpt_oss-GateUP-B32-M2048, gpt_oss-Down-B32-M2048,
DSV3-Down-B16-M4096) to identify which kernel path has the most
MFMA-utilization headroom. Then commit to one of D2 / A1 / B1
based on the profile data. Multi-round project from there.

## Files touched (round 6)

- `analysis/_notes/round-6-bf16-grouped-dsv3-vark-cfg-FALSIFIED.md`
  (this file — also closes R5 dispatch sweep)

NO source code changes (config.py reverted to round-4 state).
Working tree clean.

## Metric / bench numbers

- Round-6 starting baseline: 881 single (per-shape table in
  `/tmp/metric_round_6.log`)
- Bench DSV3 8 shapes: avg bwd TF 985.79 → 982.40 with (4,4)
  (Δ -3.39 TF, 7 of 8 negative)
- Correctness: 24/24 PASS, 0/24 reject in every measured run.
