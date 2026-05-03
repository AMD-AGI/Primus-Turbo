# Round 17 — BF16 grouped GEMM: Qwen3-Down dB var-K CRR (gm=8, xcds=4) — FALSIFIED

## Pivot rationale (from R16)

R11 / R14 / R15 / R16 closed the BF16 grouped kernel KI-specialization
surface (4 falsified levers spanning gpt_oss K=2880 fuse path + the
short-K K%128==0 paths). R16 next-step recommendation: pivot **AWAY**
from kernel KI surgery, into dispatch / cfg side.

R17 picks the **dB var-K CRR dispatch** for the lowest non-gpt_oss
shape — Qwen3-Down — which currently falls through to the binding
default (gm=4, xcds=0 → kernel BLOCK_SWIZZLE_NUM_XCDS=8).

## Per-shape baseline (R17 metric, before change)

```
gpt_oss_20B   geomean = 1.082  (target 1.25, weight 3x)   -- closed surface
DeepSeek-V3   geomean = 1.123  (target 1.25, weight 1x)
Qwen3-235B    geomean = 1.114  (target 1.25, weight 1x)
score = 877
```

Lowest non-gpt_oss-K=2880 shapes (sorted by ratio):

| shape                                | ratio |
|--------------------------------------|-------|
| Qwen3-Down-B16-M4096                 | 1.096 |
| Qwen3-Down-B32-M2048                 | 1.100 |
| DSV3-Down-B16-M4096                  | 1.106 |
| Qwen3-Down-B32-M4096                 | 1.110 |

## Hypothesis (Lever C-varK-Qwen)

`grouped_gemm_impl.py:426` calls `select_default_config(n=N_fwd=4096,
k=K_fwd=1536, m_per_group, "crr", "bf16", m_total)` for BF16 dB var-K.
Tracing the existing BF16 CRR rules in `config.py`:

| rule                                                 | matches Qwen-Down? |
|------------------------------------------------------|--------------------|
| `tiles_m >= 64 and tiles_n == 16` (LLaMA mlp_gate_up)| no — tiles_m=16    |
| `32 <= tiles_m < 64 and tiles_n == 16 and k <= 4096` | no — tiles_m=16    |
| `tiles_m <= 16 and tiles_n >= 32 and k > 4096`       | no — tiles_n=16    |
| `tiles_n == 11 and 8 <= tiles_m <= 24` (gpt_oss)     | no — tiles_n=16    |

→ falls through to default (gm=4, xcds=0→8).

The FP8 path (`grouped_gemm_fp8_impl.py:793`) has shipped an inline
rule since R39: `if m_total >= 16384: (gm=8, xcds=4)`. R39's
methodology = 11-cell × 5-trial p50 × 9-shape sweep showing +1-3%
kernel-only on m_total>=16384 family.

**Hypothesis**: the BF16 var-K kernel uses the same persistent-grid
scheduling template; the FP8-tuned (gm=8, xcds=4) should also win on
BF16. All 4 metric Qwen3-Down dB var-K calls hit m_total ∈
{32768, 65536, 131072} — qualify under R39's >=16384 threshold.

## Patch

`config.py` — insert before the BF16 CRR fallback:

```python
if layout == "crr" and tiles_n == 16 and 8 <= tiles_m <= 16 and k == 1536:
    return HipKittenConfig(layout=layout, group_m=8, num_xcds=4, kernel=None)
```

Rule scope: `tiles_n=16 (n=4096) AND tiles_m∈[8,16] (m_per_g∈[2048,4096])
AND k==1536 (uniquely Qwen-Down)` — narrowly Qwen3-Down dB var-K, no
overlap with dense LLaMA CRR (k ∈ {4096, 8192}) or gpt_oss var-K (tiles_n=11).

## Result (R17 metric, after change)

```
gpt_oss_20B   geomean = 1.091  (+0.9 pp; rule does not touch gpt_oss path → noise)
DeepSeek-V3   geomean = 1.121  (-0.2 pp; rule does not touch DSV3 path → noise)
Qwen3-235B    geomean = 1.108  (-0.6 pp; affected family — REGRESSED)
score = 880  (+3, sub-noise)
```

Per-shape Qwen3-Down (only family touched by the rule):

| shape                          | before | after | Δ      |
|--------------------------------|--------|-------|--------|
| Qwen3-Down-B16-M2048           | 1.145  | 1.126 | -1.9pp |
| Qwen3-Down-B16-M4096           | 1.096  | 1.086 | -1.0pp |
| Qwen3-Down-B32-M2048           | 1.100  | 1.086 | -1.4pp |
| Qwen3-Down-B32-M4096           | 1.110  | 1.101 | -0.9pp |

**ALL 4 Qwen3-Down shapes regressed.** Average -1.3pp on the family
the rule was designed to help.

correctness_fail = 0/24 (numerics intact; pure perf regression on the
target family).

## Conclusion — FALSIFIED

For BF16 var-K dB on Qwen3-Down (n=4096, k=1536, m_total ∈ [32k, 131k]),
**(gm=4, xcds=8) default beats (gm=8, xcds=4) by 0.9-1.9 pp**.

This contradicts the FP8 R39 result on similar tile geometry. Why?

Hypothesis: the FP8 var-K kernel reads two scale factors per group
plus the data, so the per-tile bandwidth profile differs from BF16
(which reads only the data). FP8's heavier bandwidth pressure benefits
from `gm=8` batching that amortizes the LDS load across more N-tiles
per A-pack reuse window. BF16's lighter bandwidth makes the same
batching over-bias toward LDS, leaving the `gm=4` default's tighter
N-walk schedule preferable. **FP8 cfg tunes do not transfer to BF16
on this shape geometry.**

The +3 score gain in the post-change run was driven by random noise
on shapes the rule does not touch:
* gpt_oss family +0.9 pp (run-to-run drift; rule does not match)
* DSV3 family -0.2 pp (run-to-run drift; rule does not match)
This noise band is consistent with the R7-R17 baseline range
[868, 891] (std ~7 score points across 11 rounds).

## Action

- Primus-Turbo `config.py` reverted to baseline (HEAD 84718491).
- No HK kernel touched.
- No `git push`.

## Closed surface (BF16 grouped, post-R17)

| lever                              | round | result      |
|-----------------------------------|-------|-------------|
| FUSED_KTAIL early-issue prefetch  | R11   | falsified   |
| KI=44 + fuse=T full unroll        | R14   | falsified   |
| KI=44 + fuse=T partial unroll     | R15   | falsified   |
| KI=0 dynamic + unroll-4           | R15   | falsified   |
| KI=24 + KI=32 short-K spec        | R16   | falsified   |
| BF16 var-K Qwen-Down (gm=8,xcds=4)| R17   | falsified   |

## Next-round suggestion

Two possible directions left:

1. **Targeted cfg sweep on the BF16 var-K path** — instead of borrowing
   FP8's (gm=8, xcds=4), microbench the actual BF16 var-K kernel on
   Qwen-Down + DSV3-Down geometries. R39's FP8 sweep used 11 cells; a
   focused 6-cell BF16 sweep (gm ∈ {2,4,8}, xcds ∈ {4,8}) would
   resolve the (gm, xcds) optimum directly. Best-case: +1-3 pp on
   the 8 Down dB shapes (~+5-8 score). Falsification risk: high — R29
   M=2048 FP8 sweep already found the cell within noise.

2. **Accept plateau (~877) and write consolidated phase-A/B closure doc**.
   The R7-R17 metric range [868, 891] indicates we're at a ~880 ± 10
   plateau; further +5 score commits will be increasingly hard to
   distinguish from noise. The auto_optimize loop's patience=30 will
   eventually give up; a consolidated doc captures the closed-surface
   evidence so the next phase's agent doesn't re-explore exhausted levers.
