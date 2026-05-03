# Round 5 — Qwen3 var-K (CRR / dB) cfg rule (FALSIFIED)

## Selected target

Per round-5 baseline metric:
- Lowest-progress shape: **gpt_oss-GateUP-B32-M2048** (ratio=1.040,
  weight=3, progress=0.832). 4th round same target — but now ratio
  jumped from 0.77 to 1.04 after round 4's H4 fast transpose landed.
- Round-5 starting baseline (5-run mean): **886.2** (best historical 884
  per harness, slight noise drift).

Per-family geomean:
* gpt_oss_20B  = 1.0880  (was 0.87 pre-R4)
* DSV3         = 1.1174
* Qwen3        = 1.1264

DSV3 / Qwen3 still at 1.10–1.13 — Lever B1 territory per round-4
recommendation. 16 shapes, weight 1 each = 16 weight units; pushing
all to 1.25 = ceiling +14 score.

## Hypothesis (lever C — dispatch)

Same root-cause analysis as round 1 (gpt_oss var-K dB). Audit found
the existing CRR rules in `select_default_config` only cover:
* `tiles_n == 16 and tiles_m >= 32` (dense LLaMA mlp_gate_up dB)
* `tiles_n == 11 and 8 <= tiles_m <= 24` (round-1 gpt_oss var-K dB)

Qwen3 var-K dB tile geometries are:
* Qwen3-GateUP: K_fwd=4096 → tiles_n=16; N_fwd=3072 → tiles_m=12
* Qwen3-Down:   K_fwd=1536 → tiles_n=6;  N_fwd=4096 → tiles_m=16

Neither matches existing rules — both fell through to BF16 default
`(gm=4, xcds=8)`. Hypothesis: mirror round-1's `(gm=4, xcds=4)` choice
(reduces XCD over-split for medium-grid var-K shapes).

## Variant tested

Combined rule for both Qwen3 sub-families:
```python
if (
    layout == "crr"
    and (
        (tiles_n == 16 and tiles_m == 12)  # Qwen3-GateUP
        or (tiles_n == 6 and tiles_m == 16)  # Qwen3-Down
    )
    and k <= 4096
    and m_total is not None
    and m_total >= 32768
):
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

Catches all 8 Qwen3 var-K dB shapes (GateUP+Down × B∈{16,32} × M∈{2048,4096}).

## Bench results (per-shape backward TFLOPS, same-window comparison)

`bench_grouped_gemm_turbo.py --dtype bf16` (100-iter
`torch.utils.benchmark`, 20 warmup):

```
shape                            base bwd_TF  R5 bwd_TF   Δ TF
Qwen3-GateUP-B16-M2048               955.0        947.0       -8.0
Qwen3-Down-B16-M2048                 930.7        950.8      +20.1
Qwen3-GateUP-B16-M4096              1063.1       1057.0       -6.1
Qwen3-Down-B16-M4096                1077.9       1077.5       -0.5
Qwen3-GateUP-B32-M2048               959.0        952.3       -6.7
Qwen3-Down-B32-M2048                 947.2        959.4      +12.2
Qwen3-GateUP-B32-M4096              1068.7       1070.2       +1.4
Qwen3-Down-B32-M4096                1079.9       1071.3       -8.6

GateUP family (tiles_n=16, tiles_m=12):
  4 shapes, mean Δ = -4.9 TF (consistently slight negative)
Down family (tiles_n=6, tiles_m=16):
  4 shapes, mean Δ = +5.8 TF (mixed, noise-bound)

Average BF16 backward TFLOPS: 987.11 -> 987.19  (Δ +0.08, FLAT)
Correctness: 24/24 PASS (fwd, bwd_x, bwd_w all True per shape).
```

## Metric results

```
                         baseline (no rule)   with R5 rule
5-run mean               886.2                882.0
runs                    887/888/890/882/884   886/880/883/879/882

Δ score: -4.2 (regression, within ±5 noise but consistently negative)
```

## Diagnosis — Qwen3 var-K cfg space saturated at default

Unlike gpt_oss (round 1: gain on B=4 small grids), Qwen3's var-K
shapes are LARGER grids:
* Qwen3-GateUP-B16: tiles_per_group = 16*12 = 192,
  total = 16*192 = 3072 tiles, ~12 tiles/CU at xcds=8.
* Qwen3-GateUP-B32: 32*192 = 6144 tiles, ~24 tiles/CU.
* Qwen3-Down-B16:   16*96  = 1536 tiles, ~6 tiles/CU.
* Qwen3-Down-B32:   32*96  = 3072 tiles, ~12 tiles/CU.

At ≥6 tiles/CU, xcds=8 already covers each XCD with ~0.75-3
tiles/CU/XCD, sufficient to amortize XCD swizzle overhead.
Reducing to xcds=4 doubles per-XCD load (1.5-6 tiles/CU/XCD) but
also doubles the serialization within each XCD. The per-tile
work in var-K (ki_g=32 K-iters) is short enough that reducing
parallelism hurts slightly more than it helps.

For Qwen3-GateUP (4 shapes), `(4, 4)` consistently regressed by
6-8 TF per shape — clear sign that xcds=8 was already optimal.
Qwen3-Down (4 shapes) showed mixed +20 / -8 / +12 / -0.5 results,
classic noise-bound flat distribution.

## Decision

REVERT. (Working tree clean — no config.py change kept.)
Falsification note committed (this file).

## Recommendation for round 6

Don't extend the round-1 gpt_oss var-K rule to Qwen3 — different
grid sizes have different optimal cfg.

Two remaining attack vectors with non-zero leverage:

1. **DSV3 var-K dB cfg (NOT TESTED yet)**: DSV3-GateUP has
   tiles_n=28, tiles_m=16 (large grid: 16*448=7168 tiles
   B=16, 14336 B=32 = 28-56 tiles/CU at xcds=8). Likely
   xcds=8 default is also optimal here, but worth a quick
   bench probe to confirm. Lower priority than (2) below.

2. **Lever D step 2 — var-K kernel topology**: The var-K kernel
   uses the same `device_gemm_tile_body<CRR>` as the forward
   grouped kernel. The CRR layout has B-side load striding
   that may be sub-optimal for the gpt_oss tile geometries.
   Multi-round kernel work (similar effort to round 4's H4
   fast transpose).

3. **Lever B1 — DSV3/Qwen3 forward MFMA pipeline scheduling**:
   The K%128==0 fast path in `kernel_bf16_dynamic.cpp` 's
   forward grouped kernel runs MFMA at less than peak (likely
   <90%). Profile with rocprofv3 valuMfmaUtil to find which
   shapes can benefit. If forward TFLOPS lifts +5%, the
   metric ratio for DSV3/Qwen3 jumps proportionally.

Round 6 recommendation: **bench probe DSV3 var-K dB cfg** (similar
risk profile to this round, ~3 min experiment) — if DSV3 is also
saturated at default, definitively rule out CRR cfg as a lever
and pivot to kernel-side work (Lever D2 or A1).

## Files touched (round 5)

- `analysis/_notes/round-5-bf16-grouped-qwen3-vark-cfg-FALSIFIED.md`
  (this file)

NO source code changes (config.py reverted to round-4 state).
Working tree clean.

## Metric numbers

- Round-5 starting baseline (5-run mean): **886.2** (range 882–890)
- With R5 rule (5-run mean): **882.0** (range 879–886)
- Δ score: **-4.2** (within ±5 noise but consistently negative
  across all 5 runs; bench-side data confirms slight regression)
- Correctness: 24/24 PASS, 0/24 reject in every measured run.
