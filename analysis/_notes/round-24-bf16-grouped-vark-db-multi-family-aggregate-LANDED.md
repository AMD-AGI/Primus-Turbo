# Round 24 — BF16 grouped GEMM: dB var-K multi-family aggregate (LANDED)

## Goal (per R23 recommendation)

R23 falsified a single-family fix for `gpt_oss-Down` dB var-K
(kernel-only +1.52 % avg, but flat on the metric). The R23 note
explicitly recommended pursuing an **R20-style multi-family
aggregate**: probe the 4 still-on-DEFAULT dB var-K families
(DSV3-GateUP, DSV3-Down, Qwen3-GateUP, Qwen3-Down — all on
`(gm=4, xcds=8)`), bundle their best cells with the previously
falsified `gpt_oss-Down` cell, and lift 16+ shapes simultaneously
so the combined wall delta crosses the metric noise floor.

R24 baseline = **884** (single run; 5-run mean **879.6**), 24/24
PASS. Bottom 4 ratios still gpt_oss B=32 (1.056-1.090, 3x weight).

## Probe — `scripts/_bf16_vark_db_multi_family_probe.py`

For each family, 12 (gm, xcds) cells × 5 trials × 120 iters per
shape, on all 4 metric shapes (B ∈ {16, 32}, M_per ∈ {2048, 4096}).
HK runtime warmed up via autograd fwd+bwd in metric order to
sidestep the K-tail cold-start sync-fault discovered in R22.

Results (kernel-only Δ vs production cfg):

| family | tiles_m × tiles_n | k_arg | prod | best uniform-positive cell | avg Δ | min Δ | max Δ |
|---|---|---|---|---|---|---|---|
| gpt_oss-Down (R23 finding) | 11 × 11 | M_per | (2, 32) cube | **(gm=1, xcds=4)** | +1.52 % | +0.72 % | +2.31 % |
| Qwen3-Down | 16 × 6  | M_per | (4, 8) DEFAULT | **(gm=1, xcds=4)** | +1.56 % | +1.15 % | +2.10 % |
| Qwen3-GateUP | 12 × 16 | M_per | (4, 8) DEFAULT | **(gm=1, xcds=4)** | +1.10 % | +0.49 % | +1.70 % |
| DSV3-Down | 28 × 8  | M_per | (4, 8) DEFAULT | **(gm=16, xcds=4)** | +0.69 % | +0.22 % | +1.24 % |
| DSV3-GateUP | 16 × 28 | M_per | (4, 8) DEFAULT | (gm=2, xcds=0) | +0.28 % | +0.15 % | +0.48 % |

Three of four newly-probed families converged on the same
`(gm=1, xcds=4)` winner — same persistent-grid scheduler state
preferred when tiles_m ∈ {12, 16} with tiles_n ∈ {6, 16}. DSV3-Down
(tiles_m=28 — widest N tile in the metric var-K suite) preferred
`(gm=16, xcds=4)`, mirroring the existing tiles_m=8 + B=32 RCR
rule's gm=16 choice (analogous large-N batching property).

## What landed

5 family-specific dispatch rules added before the cube rule
(line 691) in `primus_turbo/pytorch/kernels/hipkitten/config.py`,
each gated on `layout == "crr" and m_total is not None` so dense
LLaMA callers (always pass m_total=None) are untouched:

```
tiles_m=11 ∧ tiles_n=11 ∧ k≤4096  →  (gm=1, xcds=4)   gpt_oss-Down
tiles_m=16 ∧ tiles_n= 6 ∧ k≤4096  →  (gm=1, xcds=4)   Qwen3-Down
tiles_m=12 ∧ tiles_n=16 ∧ k≤4096  →  (gm=1, xcds=4)   Qwen3-GateUP
tiles_m=28 ∧ tiles_n= 8 ∧ k≤4096  →  (gm=16, xcds=4)  DSV3-Down
```

DSV3-GateUP rule (`tiles_m=16, tiles_n=28 → (gm=2, xcds=0)`) was
ATTEMPTED first but **dropped after metric run 1 reported 4/4
DSV3-GateUP shapes failing dB-allclose vs Triton reference**.
The kernel-only probe verified `bit_eq=True` only against (gm=4,
xcds=4) — not against (gm=2, xcds=0) itself. The xcds=0 chiplet
bypass produces a different cross-CU accumulation order that's
self-consistent within HK but drifts >2-ULP vs Triton on this
specific (deep-K=7168) geometry. DSV3-GateUP dB var-K stays on
default (gm=4, xcds=8). 4-rule aggregate landed.

## Verification

### Metric (5-run means, head-to-head)

| | run1 | run2 | run3 | run4 | run5 | mean |
|---|---|---|---|---|---|---|
| baseline (HEAD 2758c1f) | 877 | 885 | 879 | 878 | 879 | **879.6** |
| after-rule              | 887 | 884 | 883 | 885 | 886 | **885.0** |

Δ = **+5.4** (clears +5 commit threshold). New best individual run
887 ties the previous historical best 887 family. After-rule
range 883-887 is fully above the baseline mean, indicating the
delta is not a tail artifact. correct_fail=0/24 in all 5 after-runs.

Per-family dB-side improvements implicit in the metric: the 16
metric shapes covered by the 4 rules each gained roughly +0.7 to
+1.6 % at the kernel level on the dB var-K leg, which is
~25-33 % of fwd+bwd wall, so ~+0.2 to +0.5 % wall lift per shape;
4 Qwen-Down + 4 Qwen-GateUP + 4 DSV3-Down × 1× weight + 4
gpt_oss-Down × 3× weight = effective 24 weight units lifted
simultaneously, which is the "stack to overcome noise floor"
mechanism R20 used.

### bench_grouped_gemm_turbo.py --dtype bf16

24/24 PASS (`/tmp/bench_grouped_round24.log`):

* Average Forward TFLOPS: **1267.47**
* Average Backward TFLOPS: **1024.15**

Head-to-head with R20 (cffcc3d2) baseline tracked in repo:

| shape | fwd TF | bwd TF |
|---|---|---|
| DSV3-GateUP B=32 M=4096 | 1462.67 | 1140.44 |
| DSV3-Down   B=32 M=4096 | 1278.95 | 1179.11 |
| Qwen3-GateUP B=32 M=4096 | 1375.03 | 1087.61 |
| Qwen3-Down   B=32 M=4096 | 1146.60 | 1111.12 |
| gpt_oss-Down B=32 M=4096 | 1203.22 |  970.54 |

dB var-K is one of three bwd legs (also dA RRR + accumulator);
the +5.4 metric lift is the projected ratio impact summed across
the 16 affected shapes.

### Correctness

The metric's bf16 allclose check passed on 24/24 in all 5
after-rule runs. The dropped DSV3-GateUP cell was the only
allclose-positive case discovered (and dropped pre-commit).

## Why this worked when R22 + R23 didn't

Same mechanism as R20's 3-rule dA RRR aggregate:

1. R22 single-family rule (gpt_oss B=32 var-K split): kernel +0.7 %,
   covers 8 metric weight units (4 shapes × 1-3 weight)  → wall
   delta below noise floor (~3-pt metric range).
2. R23 single-family rule (gpt_oss-Down dB var-K cube fix):
   kernel +1.52 %, covers 12 weight units (4 shapes × 3 weight)
   → wall delta still below noise floor.
3. R24 four-family rule aggregate: combined kernel +0.7 to +1.6 %,
   covers 24 weight units (16 shapes, weighted) → wall delta
   crosses the +5 metric noise floor.

The principle: kernel-only wins under +2 % at the dB var-K leg
need 15+ shape-weight units lifted simultaneously to register on
the wall-time metric. Single-family rules cover at most 12.
Four-family aggregates cover 24. R20 hit the same threshold via
3 dA RRR rules (12 dA shapes × dA's larger ~50 % wall fraction).

## Files

* `primus_turbo/pytorch/kernels/hipkitten/config.py` — +124 lines,
  5 rules added (1 from R23 + 4 from R24), 1 NOTE comment about
  the dropped DSV3-GateUP cell.
* `scripts/_bf16_vark_db_multi_family_probe.py` — archived probe
  for re-use on future bf16 var-K tuning rounds.
* `analysis/_notes/round-24-bf16-grouped-vark-db-multi-family-aggregate-LANDED.md` — this note.

## Next round suggestions

* **R25 — extend probe to dA RRR families that R20 left out**:
  R20's aggregate covered DSV3-GateUP/DSV3-Down/Qwen-GateUP dA RRR
  (3 rules). Qwen-Down dA RRR + all gpt_oss dA RRR (5 families,
  20 shapes) are still on whatever they're dispatched to today.
  Mirror R24's structure: probe per-family, find uniform-positive
  cells, bundle into one commit. dA RRR has larger wall fraction
  (~50 % of bwd) than dB var-K (~25 %) per R22 arithmetic, so
  even smaller per-family kernel wins should aggregate to a
  larger metric delta.
* **R25 alternate — DSV3-GateUP dB var-K with allclose-safe cells**:
  Re-probe DSV3-GateUP with the explicit constraint that each
  candidate cell must `allclose` against Triton reference (not just
  bit-eq vs another HK cell). Likely candidate: (gm=1, xcds=4) —
  matches the other 3 family winners and would extend uniform
  coverage to 5/6 families.
* **Bottom-4 still gpt_oss B=32**: ratios 1.056-1.090. The
  fwd-side wins from R1 (gpt_oss-GateUP 4096^2 x11008-style cube)
  carry these shapes; the dB var-K win from this round shaves
  another 0.2-0.4 pp but they remain the lowest. Future rounds
  should look at whether the K-tail (K%128≠0) fwd kernel itself
  has any (gm, xcds) headroom for B=32 — that's the only leg of
  gpt_oss B=32 that hasn't been kernel-probed in any round.
