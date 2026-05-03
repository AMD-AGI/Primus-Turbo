# Round 3 — gpt_oss var-K (CRR / dB) cfg sweep (FALSIFIED)

## Selected target

Per round-3 baseline metric:
- Lowest-progress shape: **gpt_oss-GateUP-B32-M2048** (ratio=0.773,
  weight=3, progress=0.618). Same shape as rounds 1 + 2 (still worst).
- Round-3 starting baseline: **774** single run (mean ~779 over 5 runs)
- Best-historical (per harness): 817 (from round 1, lower GPU contention)
- Bench script (more stable than metric) backward TFLOPS for the
  8 gpt_oss shapes (Triton-independent absolute speed):

| shape | bench bwd TF (baseline R1 cfg) |
|-------|-------------------------------|
| gpt_oss-GateUP-B4-M2048  | 619 |
| gpt_oss-Down-B4-M2048    | **563**  ← lowest |
| gpt_oss-GateUP-B4-M4096  | 856 |
| gpt_oss-Down-B4-M4096    | 753 |
| gpt_oss-GateUP-B32-M2048 | 649 |
| gpt_oss-Down-B32-M2048   | 618 |
| gpt_oss-GateUP-B32-M4096 | 845 |
| gpt_oss-Down-B32-M4096   | 789 |

`_M2048` shapes consistently slower than `_M4096`; `Down` slower than
`GateUP` per B-tier. Reflects per-tile var-K work and tiles/CU
distribution.

## Hypothesis (lever C — dispatch)

Round-1's CRR rule applies `(gm=4, xcds=4)` uniformly to all 8 gpt_oss
var-K dB shapes. Round-1 commentary noted B=32 cases were "mostly
noise" with this single cfg. Round 3 hypothesis: split the rule by
sub-family (B and N) to use cfg better-suited to each tile-geometry
tier, especially for the slowest shape Down-B4-M2048.

## Variants tested (bench script, 100-iter `torch.utils.benchmark`)

Bench is **much more stable** than the metric (variance ~5 TF / shape
vs ±15 score on the metric weighted-wall). Used as the per-shape
cfg comparison tool this round.

### Variant A — B=32 GateUP only `(gm=8, xcds=4)`

Rule: `layout == "crr" and tiles_n == 11 and tiles_m == 22 and k <= 4096
and m_total >= 32768` → `(gm=8, xcds=4)` (otherwise round-1 rule fires
unchanged).

| shape (B=32 GateUP only changed) | base bwd TF | v(8,4) bwd TF | Δ |
|----------------------------------|-------------|---------------|---|
| gpt_oss-GateUP-B32-M2048         | 649.02      | 645.03        | −4.0 |
| gpt_oss-GateUP-B32-M4096         | 845.42      | 843.97        | −1.5 |
| gpt_oss-Down-B32-M2048 (control) | 618.04      | 605.90        | −12.1 |
| gpt_oss-Down-B32-M4096 (control) | 789.26      | 785.39        | −3.9 |

Control shapes (rule does NOT match) varied by 4–12 TF — that IS the
bench noise floor for this round. Target shapes' Δ (−4.0 / −1.5)
sit inside that noise. **(8,4) for B=32 GateUP: FLAT** (slight
regression sign, within noise).

### Variant B — B=32 GateUP only `(gm=2, xcds=4)`

Same scope as A, smaller group_m (more tile distribution).

| shape (B=32 GateUP only changed) | base bwd TF | v(2,4) bwd TF | Δ |
|----------------------------------|-------------|---------------|---|
| gpt_oss-GateUP-B32-M2048         | 649.02      | 647.34        | −1.7 |
| gpt_oss-GateUP-B32-M4096         | 845.42      | 846.21        | +0.8 |

**(2,4) for B=32 GateUP: FLAT** (within noise on both signs).

### Variant C — Down B=4 only `(gm=1, xcds=4)`

Rule: `layout == "crr" and tiles_n == 11 and tiles_m == 11 and k <= 4096
and m_total <= 16384` → `(gm=1, xcds=4)`.

| shape (Down B=4 only changed)   | base bwd TF | v(1,4) bwd TF | Δ |
|----------------------------------|-------------|---------------|---|
| gpt_oss-Down-B4-M2048            | 563.62      | 564.77        | +1.2 |
| gpt_oss-Down-B4-M4096            | 753.05      | 751.84        | −1.2 |

**(1,4) for Down B=4: FLAT** (sign flip across shapes, both within
noise).

## Diagnosis — var-K kernel cfg space is saturated

Three variants spanning the 4-way `(gm, xcds)` neighbourhood around
the round-1 `(gm=4, xcds=4)` choice all sit inside ±5 TF bench noise
on their target shapes. The var-K kernel is genuinely cfg-saturated
for gpt_oss tile geometries:

* `tiles_per_group ∈ {121 (Down), 242 (GateUP)}` × B
  `∈ {4, 32}` gives a wide spread of total tiles
  (484 / 968 / 3872 / 7744) but the persistent grid + chiplet-
  swizzle (`chiplet_transform_chunked(blockIdx.x, NUM_CUS,
  num_xcds, 64)`) achieves near-saturation at `(gm=4, xcds=4)`
  for all 4 tiers.
* Per-tile work scales with `ki_g = M_per / K_STEP ∈ {32, 64}`,
  which is high enough that XCD-split granularity (`xcds=4` vs `8`)
  is dominated by per-tile compute time, not inter-XCD scheduling.
* `group_m` (WGM in `device_gemm_tile_body`'s tile-window scheduler)
  trades B-side cache reuse against work distribution; sweep shows
  WGM=4 is the geomean optimum, with WGM ∈ {1, 2, 8, 11} all
  within 0.5pp at any single tier.

The real backward bottleneck is split between:
1. **Var-K kernel per-tile work**: low TFLOPS (~500 TF for Down-B4)
   because ki_g is small (32 K-iters per tile). Persistent kernel
   has only ~480 tiles total → ~1.9 tiles/CU, with each tile only
   doing 32 K-iter steps. CUs spend significant time idle between
   tiles.
2. **dA H4 transpose + contiguous() copy** (round-2 finding):
   ~265 µs (B=4) to ~2.1 ms (B=32) per dA call. Pure mem-bound HBM
   work, 100% serialized with the dA kernel.

Both require kernel-side work to fix:
- For (1): a different var-K kernel topology (more tiles per launch
  via finer N-axis tiling, or merged-B reduction within one tile)
  is needed. Multi-round.
- For (2): kernel-side strided-B RCR variant (skip `.contiguous()`),
  OR fix the BF16 RRR phantom-read bug to enable native RRR for
  K_RRR=2880 cases (round-3..8 archive shows this is hard). Multi-
  round.

## Decision

REVERT all 3 variants. None move bench backward TFLOPS outside the
±5 TF noise floor on their target shapes. Working tree clean.
Falsification note committed (this file).

## Recommendation for round 4

**Stop sweeping var-K cfg.** Three round-3 variants confirmed the
round-1 `(gm=4, xcds=4)` is at the local optimum for all 4 cfg-
neighbourhood directions tested. Don't retry without a new
hypothesis (e.g. WGN swap when num_pid_n > num_pid_m branch fires
— which it doesn't for gpt_oss).

Two remaining attack vectors with potentially-non-zero leverage:

1. **Lever D (kernel) — H4 transpose elimination**: write a BF16
   RCR kernel variant that accepts `b` with last-dim stride != 1
   (post-transpose strided B). Skip `.contiguous()` in
   `GroupedGEMMHipKittenBackend.execute`. Estimated wall savings
   ~265 µs (B=4) to ~2.1 ms (B=32) per gpt_oss dA call. Score
   gain: +30–50 if the kernel modification is bit-equivalent
   (probably +20–30 in practice once measurement noise is
   averaged). Multi-round (kernel + correctness verify + cfg).
2. **Lever B1 (DSV3/Qwen3) — push the 1.10–1.14 cluster across 1.25**:
   16 shapes (DSV3 + Qwen3) all sit at progress 0.85–0.90. Each
   would-be PASS adds (1.0 - 0.88) * 1 / 40 = +0.3 score. 16
   shapes at full PASS = +4.8 score per pp, +14.4 score for the
   full 1.10 → 1.25 jump. Achievable via MFMA pipeline scheduling
   (round-7..18 in HipKittens history showed similar k%128==0
   wins are real but slow; +1pp / round is the norm).

Round 4 recommendation: start **Lever D step 1** — investigate the
BF16 RCR kernel B-load path (line 583+ in kernel_bf16_dynamic.cpp)
to plan how to add a strided-B variant. The kernel uses
`G::load(Bs[...], b_gl, b_coord(...), swizzled_offsets_B,
b_srsrc_base, b_base, b_lds_...)`. The `b_coord` macro and the
`swizzled_offsets_B` precompute likely assume K-stride 1; need to
characterize what the H4 transpose's stride pattern would imply
and whether a parallel `_strided_b` kernel template could share
most of the code path.

## Files touched (round 3)

- `analysis/_notes/round-3-bf16-grouped-gpt-oss-vark-cfg-sweep-FALSIFIED.md`
  (this file)

NO source code changes (config.py reverted to round-1+round-2 state).
Working tree clean.

## Metric numbers

- Round-3 starting baseline: 774 single / 779 5-run mean
- Variants A/B/C: bench backward TFLOPS comparison only (metric not
  re-run for each variant — bench noise floor of ±5 TF is the gating
  criterion for cfg work; if bench shows flat, metric has 5x more
  noise so will also show flat).
- Round-3 final (post-revert) baseline: 778–781 (5-run mean 779.4)
- All 24 correctness PASS, 0/24 reject in every measured run.
