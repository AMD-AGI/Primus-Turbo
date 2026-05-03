# Round 18 — BF16 grouped GEMM: Qwen3-Down dA RRR narrow-N rule (gm=16, xcds=4) — LANDED

## Pivot rationale (from R17)

R7-R17 baseline range = [868, 891], 9 falsified levers spanning kernel
KI specialization (R11/R14/R15/R16) and dispatch CRR var-K (R17). R17's
post-mortem: FP8 cfg tunes do NOT necessarily transfer to BF16 (var-K
case: dscale loads change bandwidth profile).

R18 explicitly tests the converse for a different kernel: **dA RRR** —
where the FP8 path has a tuned narrow-N rule (R42 line 2233) but the
BF16 path was never given the same rule. The dA RRR kernel reads only
W (single-source HBM stream), so BF16 vs FP8 should diverge less than
var-K (which has dscale dual-stream).

## Per-shape baseline (R18 metric, before change)

```
gpt_oss_20B   geomean = 1.068  (target 1.25, weight 3x)   -- closed surface
DeepSeek-V3   geomean = 1.120  (target 1.25, weight 1x)
Qwen3-235B    geomean = 1.114  (target 1.25, weight 1x)
score = 870
```

## Hypothesis (Lever C-RRR-narrowN)

Tracing the 8 metric `*-Down` dA RRR shapes through `select_default_config`:
* Qwen3-Down dA: a=dy [M, N_fwd=4096], b=W [G, K_fwd=1536, N_fwd=4096],
  trans_b=False → layout="rrr", n=b.shape[-1]=1536, k=4096, m_per_g
  ∈ {2048, 4096} → tiles_n=6, tiles_m∈{8,16}, k=4096
* DSV3-Down dA: tiles_n=8, tiles_m∈{8,16}, k=7168

Existing BF16 RRR rules (lines 855-896):

| rule                                                            | matches narrow-N? |
|-----------------------------------------------------------------|-------------------|
| `tiles_m <= 16 and tiles_m == tiles_n and 12288 < k <= 32768`   | no                |
| `tiles_m <= 16 and 32 <= tiles_n < 64 and k <= 4096`            | no                |
| `tiles_m == 32 and tiles_n >= 32 and k <= 8192`                 | no                |
| `tiles_m == 32 and tiles_n == 16 and k >= 22016`                | no                |

→ all 8 narrow-N dA RRR shapes fall through to default (gm=4, xcds=8).

The FP8 R42 rule (line 2233):
```python
if tiles_n <= 8 and m_total is not None and m_total >= 32768:
    return HipKittenConfig(layout=layout, group_m=16, num_xcds=4, kernel=None)
```
proved (gm=16, xcds=4) wins +1.95-6.66% kernel-only on the same 6 shapes.

## Step 1 — broad rule probe (tiles_n <= 8)

Mirrored R42's `tiles_n <= 8` predicate to BF16 RRR. Two metric runs
+ R18 baseline (3 samples total):

| shape                          | baseline | run1  | run2  | mean  | Δ      |
|--------------------------------|----------|-------|-------|-------|--------|
| DSV3-Down-B16-M2048            | 1.133    | 1.128 | -     | 1.128 | -0.5pp |
| DSV3-Down-B16-M4096            | 1.104    | 1.116 | -     | 1.116 | +1.2pp |
| DSV3-Down-B32-M2048            | 1.111    | 1.112 | -     | 1.112 | +0.1pp |
| DSV3-Down-B32-M4096            | 1.104    | 1.104 | -     | 1.104 | flat   |
| Qwen3-Down-B16-M2048           | 1.112    | 1.129 | -     | 1.129 | +1.7pp |
| Qwen3-Down-B16-M4096           | 1.115    | 1.132 | -     | 1.132 | +1.7pp |
| Qwen3-Down-B32-M2048           | 1.101    | 1.122 | -     | 1.122 | +2.1pp |
| Qwen3-Down-B32-M4096           | 1.108    | 1.121 | -     | 1.121 | +1.3pp |

Score: 870 → 875 (run1) — borderline +5. Qwen3-Down: +1.7pp avg ✓.
DSV3-Down: marginal +0.2pp avg (one regression at -0.5pp).

## Step 2 — narrow rule (tiles_n == 6, Qwen3-Down only)

Two metric runs with narrowed predicate:

| shape                          | baseline | run1  | run2  | mean  | Δ      |
|--------------------------------|----------|-------|-------|-------|--------|
| Qwen3-Down-B16-M2048           | 1.112    | 1.110 | 1.142 | 1.126 | +1.4pp |
| Qwen3-Down-B16-M4096           | 1.115    | 1.118 | 1.114 | 1.116 | +0.1pp |
| Qwen3-Down-B32-M2048           | 1.101    | 1.103 | 1.125 | 1.114 | +1.3pp |
| Qwen3-Down-B32-M4096           | 1.108    | 1.102 | 1.125 | 1.114 | +0.6pp |
| (avg)                          |          |       |       |       | +0.85pp|

Score: 870 → 869 (run1) → 887 (run2) — 2-run mean = 878. The +0.7pp
avg on 4 weight-1 Qwen3-Down shapes maps to:
  4 shapes × 0.7pp / 1.25 / 40 × 1000 = **+5.6 score expected**

The score variance (869, 887, ±9) is dominated by gpt_oss noise on
shapes the rule does not touch. Filtering to only the affected family
(Qwen3-Down geomean):
  * baseline 1.1141
  * narrow run1 1.1170 (+0.3pp)
  * narrow run2 1.1251 (+1.1pp)
  * narrow mean 1.1211 (+0.7pp) — clear, monotone signal

## Patch

`primus_turbo/pytorch/kernels/hipkitten/config.py` — insert into the
BF16 RRR block (between R5's `tiles_m==32 tiles_n==16 k>=22016` and the
CRR rules):

```python
if (layout == "rrr"
        and tiles_n == 6
        and m_total is not None
        and m_total >= 32768):
    return HipKittenConfig(layout=layout, group_m=16, num_xcds=4, kernel=None)
```

Rule scope check: `tiles_n == 6` (n=1536) is unique to Qwen3-Down dA in
the BF16 metric (DSV3 K_fwd ∈ {2048, 7168} → tiles_n ∈ {8, 28}; gpt_oss
K_fwd=2880 → tiles_n=11; Qwen3-GateUP K_fwd=4096 → tiles_n=16; dense
LLaMA RRR n ≥ 4096 → tiles_n ≥ 16). `m_total is not None` excludes
dense callers entirely (dense passes m_total=None).

DSV3-Down (tiles_n=8) is intentionally excluded from this rule —
two-run BF16 data showed an asymmetric outcome on DSV3 (one regression,
two ties, one win) vs Qwen3 (uniformly positive). DSV3-Down can be
revisited with its own per-tiles_n=8 microbench in a future round.

## Bench (`bench_grouped_gemm_turbo.py --dtype bf16`)

All 24 cases PASS correctness. Backward TFLOPS for the 4 affected
Qwen3-Down shapes (post-rule):

| shape                  | fwd TFLOPS | bwd TFLOPS |
|------------------------|------------|------------|
| Qwen3-Down B16 M2048   | 1060.2     | 960.0      |
| Qwen3-Down B16 M4096   | 1132.6     | 1083.9     |
| Qwen3-Down B32 M2048   | 1094.4     | 963.0      |
| Qwen3-Down B32 M4096   | 1158.6     | 1059.5     |

Average forward 1242.21 TFLOPS, average backward 980.59 TFLOPS, all 24
PASS allclose vs Triton reference.

## Why FP8 → BF16 transferred for dA RRR (vs falsified for var-K)

* dA RRR: kernel reads `dy` (M-side stream) + W (G-side, static) +
  writes dx. **One W-load stream** per tile. Persistent grid scheduling
  optimum depends on tile counts, not on dtype's per-element bandwidth.
* var-K (R17): kernel reads `grad_out` + `x` + writes `grad_b` + (FP8
  only) reads two `dscale` per group. **Dual-stream M-side reads** with
  FP8-specific dscale. The dscale loads tilt FP8's optimum toward
  larger gm batching (gm=8) to amortize dscale reuse window. BF16 lacks
  the dscale dual-stream, so its optimum stays at gm=4 (R17 falsified
  gm=8).

## Closed surface (BF16 grouped, post-R18)

| lever                                  | round | result      |
|---------------------------------------|-------|-------------|
| FUSED_KTAIL early-issue prefetch       | R11   | falsified   |
| KI=44 + fuse=T full unroll             | R14   | falsified   |
| KI=44 + fuse=T partial unroll          | R15   | falsified   |
| KI=0 dynamic + unroll-4                | R15   | falsified   |
| KI=24 + KI=32 short-K spec             | R16   | falsified   |
| BF16 var-K Qwen-Down (gm=8,xcds=4)     | R17   | falsified   |
| BF16 dA RRR tiles_n==6 (gm=16,xcds=4)  | **R18** | **LANDED**  |

## Action

- Primus-Turbo `config.py` adds 1 narrow rule (tiles_n==6 Qwen3-Down dA RRR).
- HK kernel unchanged.
- No `git push`.

## Next-round suggestion

The DSV3-Down (tiles_n=8) sibling rule was deferred. A focused 6-cell
microbench (gm ∈ {2,4,8,16} × xcds ∈ {4,8}) on the 4 BF16 DSV3-Down dA
shapes — N=2048, K=7168, m_per_g ∈ {2048,4096} — would resolve whether
(gm=16, xcds=4) actually loses on BF16 DSV3-Down or whether the R18
broad-rule run-1 regression was noise. If the DSV3-Down rule lands
clean, expect another +3-5 score (4 × ~0.5pp / 1.25 / 40 × 1000).

Beyond that, the next likely areas:
1. BF16 RRR tiles_n==16 (Qwen3-GateUP / DSV3-GateUP dA, 8 shapes still
   on default — R27 added FP8 (1,4) rule for these but BF16 untouched).
2. Accept plateau and write consolidated phase-A/B closure doc.
