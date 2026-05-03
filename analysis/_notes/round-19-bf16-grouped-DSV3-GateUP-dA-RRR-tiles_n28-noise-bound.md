# Round 19 — BF16 grouped GEMM: DSV3-GateUP dA RRR tiles_n==28 (gm=16, xcds=4) — REAL SIGNAL, NOISE-BOUND, REVERTED

## Context (continuation of R18 momentum)

R18 landed the **first** non-falsified lever in 6 rounds: BF16 dA RRR
narrow-N rule (`tiles_n == 6`) for Qwen3-Down. R18's next-round
suggestion was to extend the same FP8→BF16 transfer pattern to the
remaining 12 uncovered BF16 dA RRR shapes:

| family       | tiles_n | shapes | FP8 reference | source   | status (pre-R19) |
|--------------|---------|--------|---------------|----------|------------------|
| Qwen3-Down   | 6       | 4      | (16,4) +1.95-3.23% | R42 | LANDED (R18)     |
| DSV3-Down    | 8       | 4      | (16,4) +2.65-6.66% | R42 | uncovered        |
| Qwen3-GateUP | 16      | 4      | (1,4) mixed   | R27/R32  | uncovered        |
| DSV3-GateUP  | 28      | 4      | (16,4) +1.66-2.82% | R43/R44 | uncovered        |

R19 attacks the strongest evidence FP8 reference (R43/R44 12-trial ×
200-iter × 3-seed methodology) — **DSV3-GateUP tiles_n==28**.

## Per-shape baseline (R19 metric, before change)

```
gpt_oss_20B   geomean = 1.111  (target 1.25, weight 3x)
DeepSeek-V3   geomean = 1.120  (target 1.25, weight 1x)
Qwen3-235B    geomean = 1.119  (target 1.25, weight 1x)
score = 892   ←  ABOVE prior R7 best (891) — but a high-variance single sample
```

DSV3-GateUP per-shape baseline (the rule's target subfamily):
* B16-M2048: 1.125
* B16-M4096: 1.123
* B32-M2048: 1.125
* B32-M4096: 1.138

## Hypothesis (Lever C-RRR-tiles_n28)

Add BF16 RRR rule mirroring the FP8 R43/R44 rule at line 2497:

```python
if (layout == "rrr"
        and tiles_n == 28
        and m_total is not None
        and m_total >= 32768):
    return HipKittenConfig(layout=layout, group_m=16, num_xcds=4, kernel=None)
```

Scope: tiles_n==28 (n=7168) is uniquely DSV3-GateUP dA in BF16 metric.
Same kernel-template-shared logic as R18: dA RRR reads only W
(single-source HBM stream), persistent-grid scheduling optimum is
dtype-stable.

## Result — per-shape DSV3-GateUP (5-run mean post-rule)

| shape                          | baseline | run1  | run2  | run3  | run4  | run5  | mean  | Δ      |
|--------------------------------|----------|-------|-------|-------|-------|-------|-------|--------|
| DSV3-GateUP B16-M2048          | 1.125    | 1.143 | 1.150 | 1.146 | 1.142 | 1.140 | 1.144 | +1.9pp |
| DSV3-GateUP B16-M4096          | 1.123    | 1.146 | 1.154 | 1.146 | 1.149 | 1.144 | 1.148 | +2.5pp |
| DSV3-GateUP B32-M2048          | 1.125    | 1.139 | 1.140 | 1.135 | 1.144 | 1.142 | 1.140 | +1.5pp |
| DSV3-GateUP B32-M4096          | 1.138    | 1.152 | 1.150 | 1.148 | 1.150 | 1.149 | 1.150 | +1.2pp |
| (avg)                          |          |       |       |       |       |       |       | **+1.78pp** |

**ALL 4 DSV3-GateUP shapes improved 1.2-2.5pp, every run.** Signal
matches FP8 R43/R44 prediction (+1.66-2.82%). Spread within shape ≤ 0.6pp
across 5 runs.

## Result — overall score 5×5 paired comparison

5 baseline samples (R18 active, no R19 rule):

| sample | source                    | score | gpt_oss | DSV3  | Qwen3 |
|--------|---------------------------|-------|---------|-------|-------|
| b1     | R18 round metric          | 880   | -       | -     | -     |
| b2     | R19 baseline (pre-R19)    | 892   | 1.111   | 1.120 | 1.119 |
| b3     | post-revert run 1         | 876   | 1.080   | 1.121 | 1.115 |
| b4     | post-revert run 2         | 875   | 1.080   | 1.118 | 1.112 |
| b5     | post-revert run 3         | 887   | 1.100   | 1.121 | 1.119 |
| mean   |                           | 882   | 1.093*  | 1.120 | 1.116 |

5 samples with R19 rule active:

| sample | score | gpt_oss | DSV3  | Qwen3 |
|--------|-------|---------|-------|-------|
| r1     | 874   | 1.072   | 1.127 | 1.115 |
| r2     | 877   | 1.079   | 1.131 | 1.113 |
| r3     | 890   | 1.107   | 1.124 | 1.114 |
| r4     | 881   | 1.088   | 1.128 | 1.116 |
| r5     | 880   | 1.088   | 1.124 | 1.113 |
| mean   | 880   | 1.087   | 1.127 | 1.114 |

Δ score = 880 - 882 = **-2 (within noise)**.

Per-family Δ:
* gpt_oss: 1.087 - 1.093 = -0.6pp (rule does NOT touch gpt_oss path → run-to-run noise)
* DSV3:    1.127 - 1.120 = +0.7pp ← rule's real effect (DSV3-GateUP +1.78pp on 4 of 8 DSV3 shapes)
* Qwen3:   1.114 - 1.116 = -0.2pp (rule does NOT touch Qwen3 path → run-to-run noise)

Score budget breakdown:
```
Score Δ = 480·Δgpt_oss + 160·Δdsv3 + 160·Δqwen3
       = 480·(-0.006) + 160·(+0.007) + 160·(-0.002)
       = -2.88 + 1.12 + (-0.32)
       = -2.08
```

## Conclusion — REAL SIGNAL, NOISE-BOUND, REVERTED

The rule has a **deterministic, reproducible +1.78pp avg gain on DSV3-GateUP**
(all 4 shapes, all 5 runs). The expected score gain from the per-family
DSV3 +0.7pp lift is +1.1 score — but it is masked by gpt_oss noise
(±0.6pp from 24 weight units → ±2.9 score) and Qwen3 noise (±0.2pp →
±0.3 score), giving a net 5-sample mean Δ of -2 score.

Per the task body's "Flat or down → revert" criterion, the score
test fails. **Revert** but document the real per-shape signal so a
future round can pick this lever back up if either:
1. the metric noise floor decreases (unlikely on 50-100 iter sampling), or
2. multiple sub-noise levers are committed simultaneously and their
   aggregate effect crosses the +5 score threshold, or
3. the auto_optimize gate switches to per-family geomean instead of
   weighted score (which would directly pick up the +0.7pp DSV3 signal).

## Key insight — per-shape signal vs metric noise asymmetry

After R18 + R19, the BF16 dA RRR transfer pattern is now characterized:

| family       | tiles_n | per-shape Δ   | per-family Δ | score Δ | metric verdict |
|--------------|---------|---------------|--------------|---------|----------------|
| Qwen3-Down   | 6       | +0.85pp avg   | +0.7pp       | +5-8    | LANDED (R18)   |
| DSV3-GateUP  | 28      | +1.78pp avg   | +0.7pp DSV3  | -2 → 0  | NOISE (R19)    |

Both have similar real per-family magnitudes but different metric verdicts
because the score test sees the gpt_oss noise floor.

## Closed surface (BF16 grouped, post-R19)

| lever                                  | round | result      |
|----------------------------------------|-------|-------------|
| FUSED_KTAIL early-issue prefetch       | R11   | falsified   |
| KI=44 + fuse=T full unroll             | R14   | falsified   |
| KI=44 + fuse=T partial unroll          | R15   | falsified   |
| KI=0 dynamic + unroll-4                | R15   | falsified   |
| KI=24 + KI=32 short-K spec             | R16   | falsified   |
| BF16 var-K Qwen-Down (gm=8,xcds=4)     | R17   | falsified   |
| BF16 dA RRR tiles_n==6 (gm=16,xcds=4)  | R18   | LANDED      |
| BF16 dA RRR tiles_n==28 (gm=16,xcds=4) | R19   | NOISE-BOUND |

## Next-round suggestion

Two paths forward:

1. **Try a different family** — DSV3-Down dA RRR (tiles_n==8, R42 FP8
   transfer). R18 broad-rule probe data showed mixed BF16 outcome on
   DSV3-Down (one regression, one tie, two wins) so a focused
   per-tiles_n=8 microbench might find a better cell than (16,4). If
   DSV3-Down tiles_n==8 has a consistent positive optimum, +0.5pp avg
   on 4 weight-1 shapes = +1 score per family — same noise problem.
2. **Pivot away from per-family levers** — at this point we've exhausted
   the easy FP8→BF16 transfer levers. Remaining levers all give ≤ +1pp
   per-family which is below the metric's noise floor. Either accept
   the plateau (~880 ± 10) or write a consolidated closure doc that
   captures the per-shape signal evidence so future agents don't redo
   the same exploration.

R20 should pick between these explicitly rather than burn another
falsified-by-noise round.
