# Round 2 — gpt_oss dA H4-rerouted RCR cfg rule (FALSIFIED)

## Selected target

Per round-2 baseline metric (`scripts/_metric_grouped_bf16_weighted_wall.py`):
- Lowest-progress shape: **gpt_oss-GateUP-B32-M2048** (ratio=0.781, weight=3,
  progress=0.625). Same shape as round 1 (still the worst).
- Round-2 starting baseline (4-run mean): **819.25** (single run had reported
  817 best)
- Best-historical (per metric harness): 817

## Hypothesis (lever C — dispatch)

The dA backward path for `gpt_oss-GateUP` goes through
`GroupedGEMMHipKittenBackend.execute()`'s **H4 reroute**: the RRR call
`(a:[M_total, N_fwd=5760], b:[B, 5760, K_fwd=2880])` triggers the H4 gate
because `b.shape[-1]=2880 % BLOCK_SIZE=256 != 0`, so `b` is transposed
to `[B, 2880, 5760]` + `.contiguous()` and the kernel runs as RCR
`(M=M_per, N=2880, K=5760)`.

For this RCR call: `tiles_n = 2880 // 256 = 11` (n is the `K_fwd` post-
H4), `tiles_m ∈ {8, 16}` (M_per), `k = 5760`. The dispatch falls through
to BF16 default `(group_m=4, num_xcds=8)` — no prior rule matches:
- Gpt_oss K=2880 RCR rule above checks `k == 2880` (this is k=5760).
- Cube-small rule below checks `tiles_m == tiles_n` (8 != 11, 16 != 11).
- DSV3-GateUP-M2048 / Qwen-Down rules check `tiles_n == 16`.

**Hypothesis**: mirror the round-1 var-K (CRR) rule's `(gm=4, xcds=4)`
for the same family — Round 1 found that gpt_oss var-K (small B grid)
benefits from `xcds=4` reducing XCD over-split.

## Variants tested + 4-run measurements

GPU: `HIP_VISIBLE_DEVICES=3` (auto-pinned). GPU contention varied
intermittently across the round (early baseline 819, late baseline 778
— no change in repo state, just neighbor workload).

### Variant A — full coverage (k=5760, tiles_n=11, tiles_m∈{8,16})

```python
if (layout == "rcr" and tiles_n == 11 and tiles_m in (8, 16) and k == 5760):
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

Catches **all 4** gpt_oss-GateUP dA H4-rerouted shapes (B∈{4,32},
M_per∈{2048,4096}).

| run | baseline (819.25 mean) | variant A (4,4 all) |
|-----|------------------------|---------------------|
| 1   | 813                    | 815                 |
| 2   | 836                    | 801                 |
| 3   | 807                    | 801                 |
| 4   | 821                    | 814                 |
| **mean** | **819.25**         | **807.75**          |

**Δ = −11.5 score** (variant A regresses vs baseline).

### Variant B — B=4-only sub-rule (`m_total <= 16384`)

```python
if (layout == "rcr" and tiles_n == 11 and tiles_m in (8, 16)
    and k == 5760 and m_total is not None and m_total <= 16384):
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

Catches only the 2 B=4 GateUP dA H4-rerouted shapes (M_per∈{2048,4096});
B=32 falls back to default `(4,8)`.

| run | baseline (now 778 mean) | variant B (4,4 B=4 only) |
|-----|-------------------------|--------------------------|
| 1   | 776                     | 779                      |
| 2   | 779                     | 777                      |
| 3   | 777                     | 776                      |
| 4   | 781                     | 773                      |
| **mean** | **778.25**          | **776.25**               |

**Δ = −2.0 score** (within noise; effectively flat).

(Baseline shifted 819 → 778 between variant A and variant B runs due
to neighbor GPU workload; both variant runs were measured against the
same-window baseline so the comparison is internally consistent.)

## Diagnosis

The reason `(4,4)` works for round-1 var-K (CRR) but fails here:
- Round-1's gpt_oss CRR call is the **variable-K kernel** which has
  high per-tile cost (each output tile scans the full K_var = M_per
  reduction across all B groups). Even at B=32 (~10+ tiles/CU), the
  per-tile time is long enough that XCD-split granularity matters
  less; `xcds=4` doesn't under-utilize.
- This round's gpt_oss dA H4-rerouted RCR call is the **K-aligned
  forward kernel** with K=5760 (no K-tail; 90 K-iters). For
  B=32 (m_total ∈ {65536, 131072}), the persistent grid has
  ~11–22 tiles/CU at xcds=8 — already well-utilized; forcing
  xcds=4 doubles the per-XCD tile load and serializes work that
  would otherwise run on more XCDs in parallel. Net regression.
- For B=4 (m_total ∈ {8192, 16384}), the variant B sub-rule did not
  show detectable improvement — likely the K=5760 K-aligned RCR
  kernel's per-tile cost is high enough that even at 1.4 tiles/CU
  the default xcds=8 doesn't suffer from idle XCDs (each XCD's
  CUs alternate between ~few tiles efficiently).

The **real bottleneck** for the gpt_oss dA H4 path is the
`b.transpose(-2, -1).contiguous()` itself, not cfg:
- `gpt_oss-GateUP` b shape `[B, 5760, 2880]`:
  - B=4: ~132 MB rd+wr ≈ ~265 µs
  - B=32: ~1.06 GB rd+wr ≈ ~2.1 ms
- The BF16 RCR kernel reads `b[g]` row-major contiguous in K (last
  dim stride 1). After `transpose(-2, -1)` without `.contiguous()`,
  the per-`[g]` slice has stride `(1, N_orig)` — K is NOT contiguous,
  so the kernel mis-loads. The `.contiguous()` materializes the
  required row-major layout via a memcpy kernel.

## Why the variance dominates this round

Single-run score noise: ±15 score during this round (775–836 across
8 runs of identical baseline code). Detection floor for cfg rule
changes is ~+15 expected score. Variants A/B sit at −11.5 / −2.0 —
both inside the noise floor on the favorable end and below it on the
unfavorable end. Cannot commit either without violating the round
rule "score ≥ prior best + 5".

## Decision

REVERT (working tree already clean — no config.py change kept).
Falsification note committed (this file).

## Recommendation for round 3

The dispatch surface for the dA H4-rerouted RCR path is essentially
saturated — default `(4,8)` is near-optimal for the K-aligned
K=5760 kernel across all m_total tiers. Two alternative attacks:

1. **Lever D — H4 transpose elimination** (kernel C++ surgery).
   Either:
   - (a) Rewrite the BF16 RCR kernel to accept strided-B (last-dim
     stride != 1) for the H4 case — eliminates `.contiguous()`
     entirely, saves ~265 µs (B=4) to ~2.1 ms (B=32) per dA call.
     Estimated metric gain: +30-50 score on gpt_oss-GateUP family
     (gpt_oss-GateUP-B32-M2048 dA wall ~9 ms → ~7 ms = ratio
     0.78 → ~1.0+).
   - (b) Fix the BF16 RRR phantom-read bug (the original H4 reason)
     so `K_fwd=2880, N_fwd=2880` RRR can run native without
     transpose. Round-3..8 attempts are documented in
     `/workspace/code/HipKittens/analysis/_notes/round-{3..7}-bf16-rrr-*.md`
     (got SNR 18.68 → 25.45 dB but allclose still FAIL). Hard.

2. **Lever A2 — var-K B=32 kernel surgery**: round-1 noted B=32
   gpt_oss var-K (dB) was "mostly noise" (B=32 ratio 0.753 → 0.763
   marginal). The var-K kernel's per-group reduction may have a
   serial bottleneck across B that the persistent kernel can't
   parallelize. Profile via `rocprofv3` to characterize.

3. **Lever B1 — DSV3/Qwen3 MFMA pipeline scheduling**: DSV3 / Qwen3
   geomean is 1.13–1.17, several shapes already at 1.20+. Pushing
   them above 1.25 is +0.5–2 score per shape. Cumulative ceiling
   without gpt_oss progress: +20–40 score on the K%128==0 fast
   path. Lower per-shape leverage but more reliable measurement
   (these shapes don't have the H4 transpose noise contributor).

For round 3, recommend **Lever D(a)** — kernel-side H4 transpose
elimination. Single biggest leverage on gpt_oss family; well-
defined bit-for-bit equivalence proof (just swap stride loads in
the RCR kernel's B-load path); estimated +30-50 score. Multi-round
project (kernel rewrite + correctness verify + cfg re-tune for
strided-B path), but each round can ship a sub-step.

## Files touched (round 2)

- `analysis/_notes/round-2-bf16-grouped-gpt-oss-da-h4-rcr-cfg-FALSIFIED.md`
  (this file)

NO source code changes. Working tree clean.

## Metric numbers

- Round-2 starting baseline: 819 (single run) / 819.25 (4-run mean)
- Best-historical (per harness): 817
- Round-2 attempts:
  - Variant A (4,4 all): mean 807.75 (Δ −11.5)
  - Variant B (4,4 B=4 only): mean 776.25 (Δ −2.0 vs same-window
    baseline 778.25)
- Round-2 final score: **778** (single run; matches stable baseline
  under current GPU contention)
- All 24 correctness PASS, 0/24 reject in every measured run.
