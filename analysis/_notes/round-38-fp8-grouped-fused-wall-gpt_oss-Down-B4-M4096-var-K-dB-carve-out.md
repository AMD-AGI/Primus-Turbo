# Round 38 — fp8-grouped fused-wall: gpt_oss-Down-B4-M4096 var-K dB carve-out (gm=16, xcds=4)

## Lever / phase

Lever F (dispatch tuning of HK var-K CRR scheduling knobs). Same class as
R30 / R31 / R33 / R35 — narrow per-shape carve-out of `(group_m, num_xcds)`
on the `grouped_variable_k_crr_dscale` host wrapper, gated by a general
shape predicate that matches exactly one metric shape.

Path A fused-act fusion remains architecturally falsified (R7 / R8); this
run series is in dispatch-tuning saturation territory.

## Targeted shape (chosen from R37 metric stderr)

`gpt_oss-Down-B4-M4096` var-K dB. Last round metric showed this shape at
ratio 1.313 (`<135%`). It was sitting in R30's "else" fallback branch at
`(gm=8, xcds=4)`, never specifically tuned. R33 had probed `(gm=16)` on
this shape under earlier (less-tight) methodology and concluded TIE,
leaving the rule untouched.

## Re-probe methodology (R32-class, kernel-only)

`/tmp/probe_r38_gpt_oss_down_b4_m4096_var_k_db.py`:

* 12 trials × 200 iters × 3 seeds (42 / 137 / 2024)
* Direct call to `hipkitten.grouped_variable_k_crr_dscale` (kernel-only,
  no autograd / quant / Python orchestration overhead).
* Cells swept (vs current `(8, 4)` baseline):
  `(2,4), (2,2), (1,4), (32,4), (8,2), (8,8), (16,2), (16,4), (4,4)`.

### Result table

| cell      | seed=42 Δ% | seed=137 Δ% | seed=2024 Δ% | med Δ% | spread pp | verdict |
|-----------|------------|-------------|--------------|--------|-----------|---------|
| (16, 4)   | +1.35%     | +1.43%      | +1.24%       | +1.37% | 0.23      | **WIN  med/spread=5.96×** |
| (32, 4)   | +1.55%     | +1.41%      | +1.30%       | +1.42% | 0.12      | **WIN  med/spread=11.83×** |
| (4, 4)    | +0.24%     | +0.12%      | −0.04%       | +0.09% | 0.08      | TIE     |
| (1, 4)    | +0.86%     | +0.74%      | +0.49%       | +0.66% | 0.09      | LOSS *  |
| (2, 4)    | +0.55%     | +0.37%      | +0.25%       | +0.37% | 0.05      | TIE     |
| (8, 8)    | −1.96%     | −1.28%      | −1.34%       | −1.53% | 0.63      | LOSS    |
| (16, 2)   | TIE        | TIE         | LOSS         | tie    | wide      | TIE     |
| (8, 2)    | LOSS       | LOSS        | LOSS         | loss   | n/a       | LOSS    |
| (2, 2)    | LOSS       | LOSS        | LOSS         | loss   | n/a       | LOSS    |

(LOSS * means: 3-seed median is positive, but baseline-max beats winner-min in
≥1 of 3 seeds → not a robust every-seed WIN.)

`(16, 4)` and `(32, 4)` are the only cells with **every-seed positive**
delta clear of run-to-run spread (med/spread = 5.96× and 11.83× resp.,
both well above the standard "median > spread" robust-signal threshold
applied in R7 / R10 / R23 / R29 / R30 / R31 / R32 / R33 / R35).

### Cell choice

`(gm=16, xcds=4)` selected over `(gm=32, xcds=4)`:

* Both are within 0.05 pp of each other (med 161.92 vs 160.84 us; +1.37 vs
  +1.42 %).
* `(gm=16)` is **rule-consistent with R33's sibling carve-out** for
  `gpt_oss-Down-B=4-M=2048` (in the m_total<16384 branch), keeping the
  whole `gpt_oss-Down-B=4` family on a single per-row config.
* No measurable benefit to differentiating; minimises rule-chain
  divergence.

## Geometric reasoning

`gpt_oss-Down-B4-M4096` var-K dB:
- m_total = 16384, N_fwd = K_fwd = 2880 → tiles_n = tiles_k = 11.
- 4 groups × 121 tile-steps per group = 484 tile-steps over NUM_CUS=256
  persistent slots ≈ 2 wave-steps per slot.
- Same persistent-grid topology as R33's B4-M2048 sibling (121 × 4 = 484).
- `(gm=16)` packs 16 N-tiles per pass = 1.45× the N-axis (tiles_n=11),
  saturating L2 on the per-K B-pack and amortising the sparse persistent
  grid.
- R30's `(gm=4)` wins for B=32 (3872 tile-steps) because the much larger
  grid shifts the tile-batching trade-off — gm-4 preserves L2 on the
  cross-group stall avoidance there. Gate predicates separate the two.

## Bit-equivalence verification

`/tmp/probe_r38_correctness.py`:

```
== gpt_oss-Down-B4-M4096 var-K dB bit-equivalence ==
  cells: (8, 4) baseline vs (16, 4) candidate vs (32, 4) sibling
  seed=0    max_abs(out)=360.00  diff(8,4 vs 16,4)=0.0e+00 bit_eq=True   diff(8,4 vs 32,4)=0.0e+00 bit_eq=True
  seed=42   max_abs(out)=378.00  diff(8,4 vs 16,4)=0.0e+00 bit_eq=True   diff(8,4 vs 32,4)=0.0e+00 bit_eq=True
  seed=137  max_abs(out)=368.00  diff(8,4 vs 16,4)=0.0e+00 bit_eq=True   diff(8,4 vs 32,4)=0.0e+00 bit_eq=True
```

`group_m` and `num_xcds` are pure persistent-grid scheduling knobs on the
var-K CRR kernel — same arithmetic / FP8 quant rounding invariance
documented for every (gm, xcds) RCR / RRR / var-K rule in
`primus_turbo/pytorch/kernels/hipkitten/config.py` (R30/R31/R32/R33/R35
and the R36 methodology fix to use `torch.zeros()` for output buffers
in bit-equivalence probes — applied here).

R36 `torch.empty()` garbage trap pre-empted: probe initialises every
output buffer with `torch.zeros()` so unwritten padding never produces
a spurious diff.

## Rule scope check

Predicate: `m_total ∈ [16384, 65536)` AND `k = K_fwd = 2880` AND
`n = N_fwd = 2880`. In the 24-shape MoE metric this matches **only**
gpt_oss-Down-B=4-M=4096:

* gpt_oss-Down B4-M2048 (m_total=8192) → falls to the m_total<16384
  R33 branch ((16, 4)).
* gpt_oss-Down B32-* (m_total ≥ 65536) → R30 ((4, 4)) above.
* gpt_oss-GateUP B*: b.shape[1] = N_fwd = 5760 ≠ 2880 → excluded.
* DSV3/Qwen3 B*: a.shape[1] = K_fwd ∈ {1536, 2048, 4096, 7168} ≠ 2880
  → excluded.

No other metric / DoD / dense FP8 shape matches the
`(16384 ≤ m_total < 65536, k=2880, n=2880)` scope.

## Files touched

Primus-Turbo:

* `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py` —
  in the `m_total >= 16384` branch of the var-K dB rule, refactor the
  `(k=2880, n=2880)` clause from a flat `m_total >= 65536` predicate
  into a nested `if m_total >= 65536: (gm=4, xcds=4) else: (gm=16,
  xcds=4)`. This adds the new B=4-M=4096 carve-out without touching
  the R30 B=32 sibling or any other rule chain. Heavy comment block
  explains methodology, every-seed delta table, geometric reasoning,
  rule scope, and bit-equivalence verification.

HipKittens: no changes.

## Metric (4-run paired comparison, GPU 3, MI355X)

```
Baseline  ((gm=8, xcds=4)):   scores = [992, 998, 987, 996]   median=994   gpt_oss-Down-B4-M4096 ratio median=1.326
Carve-out ((gm=16, xcds=4)):  scores = [987, 986, 998, 995]   median=991   gpt_oss-Down-B4-M4096 ratio (1st run)=1.337   (Δ=+1.1pp)
```

* **Shape-level signal**: gpt_oss-Down-B4-M4096 ratio improved by
  ~+1.1 pp (1.326 → 1.337) — matches the +1.37 % kernel-time prediction
  within Triton-side noise.
* **Geomean / score**: indistinguishable from baseline within the
  metric's ~5-point noise floor (carve-out median 991 vs baseline
  median 994). Per the R30/R31/R32/R33/R35 pattern, this is below the
  detection threshold for a single 1pp-on-1-of-24-shape carve-out
  (~+0.05 pp geomean = ~+0.5 score points).
* **Hard correctness gate**: 0/24 correctness fail in every run (both
  baseline and carve-out).

Per the run-series convention: ship narrow, kernel-real, robust-across-
seed, bit-equivalent carve-outs even when the geomean lift is below the
metric's noise floor. They compound, and matter once any future
architectural-tier improvement amplifies var-K dB as a wall fraction.

## Backward-path validation (mandatory for dB-touching changes)

```
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
  python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8 --output /tmp/hk_fp8_r38.csv
```

All 24 cases PASS. Targeted row 12:

```
gpt_oss_20B-Down B=4 M=4096 K=2880 N=2880 fp8 tensorwise  PASS
  fwd 0.14 ms  1982.82 TFLOPS   bwd 0.44 ms  1248.70 TFLOPS
```

(Avg fwd 2201.60 TFLOPS, avg bwd 1477.15 TFLOPS, no NaN/Inf, no SNR
fail.)

## DoD impact

This change touches only the per-rule branch inside
`grouped_gemm_fp8_impl.py` for the `(k==2880, n==2880, 16384<=m<65536)`
scope. It does not touch the autograd entry, the dispatcher, the
`Float8QuantConfig`, the `quantize_fp8` C++ pipeline, the custom-op
registration, or any RCR / RRR / dense FP8 path. Per the task prompt,
"only grouped HIPKITTEN path" → quick metric is sufficient; full DoD
not required. Last DoD score = -1394 (SHA 2f9b0e5b), unchanged here.

## Lessons applied

* **R36 `torch.empty()` trap**: every probe in this round used
  `torch.zeros()` for the output buffer (var-K dB writes the entire `c`
  buffer per group, but using zeros rules out any padding-region
  garbage in the bit-eq probe).
* **R32-class methodology**: 12-trial × 200-iter × 3-seed kernel-only
  direct dispatch — fine enough resolution to detect the +1.37 %
  signal that R33's coarser methodology classified as TIE.
* **R30 / R31 / R33 / R35 nested-rule pattern**: refactor flat predicate
  into nested `if m_total >= X` only when adding a sibling carve-out —
  preserves R30 / R31 contract for the existing B=32 family.

## Round summary

* **Lever**: dispatch tuning, var-K CRR scheduling knobs.
* **Files**: Primus-Turbo
  `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`.
* **Metric**: 4-run paired noise — baseline median 994, carve-out
  median 991 (within ±5pt noise band); shape-level Δ = +1.1 pp on
  gpt_oss-Down-B4-M4096 (1.326 → 1.337).
* **Correctness**: 0/24 fail every run; bench_grouped_gemm_turbo all
  24 PASS.
* **Commit**: see Primus-Turbo HEAD post-commit.
* **HipKittens**: no changes.

## Suggestion for next round

1. The 24-shape metric stderr table is dominated by Qwen3 shapes at
   ratio ~1.27. These are the highest-impact shapes left
   (geomean weight × distance from 1.35), but R32/R36 have already
   tight-verified Qwen3 var-K dB and Qwen3-GateUP-B16-M2048 dA RRR as
   saturated at their current rules. Worth checking Qwen3 forward RCR
   (`grouped_rcr_dscale`) one more time with R32-class methodology on
   the few cells not yet exhaustively swept (e.g. `(gm=8, xcds=2)`,
   `(gm=4, xcds=4)`, `(gm=16, xcds=2)` for Qwen3-Down-B16-M4096).
2. If Qwen3 RCR is also saturated at default, the only remaining lever
   is host-side overhead reduction — e.g. cache `group_offs` /
   `group_sizes` tensors across consecutive autograd passes when the
   batch shape is identical (common in steady-state training). Probe
   first with a microbench of `dispatch_grouped_gemm_fp8_impl` to see
   how much wall comes from Python-side bookkeeping vs the kernel
   itself.
3. The metric is plateaued at ~991-998 (noise band). Consider dropping
   the patience threshold or pivoting to a different optimization
   target (e.g. weight FP8 caching, which is currently out of scope per
   the task body but would lift all 24 shapes uniformly).
