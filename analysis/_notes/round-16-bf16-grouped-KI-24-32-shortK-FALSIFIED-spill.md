# Round 16 — BF16 grouped GEMM: KI_HINT=24 / KI_HINT=32 short-K specialization (FUSED_KTAIL=false) — FALSIFIED

## Pivot rationale (from R15)

R11 / R14 / R15 closed the kernel-side surgery surface for `gpt_oss_20B`
(K=2880 + FUSED_KTAIL=true) — four falsified levers (early-issue prefetch,
KI=44 full-unroll, KI=44 partial unroll, KI=0 dynamic + unroll-4) all
hit either VGPR spill (28-30) or schedule disruption.

R16 pivots to **Phase B** — short-K K%128==0 paths in DSV3 / Qwen3.

## Per-shape baseline (R16 metric, before change)

```
gpt_oss_20B   geomean = 1.093  (target 1.25, weight 3x)   -- closed surface
DeepSeek-V3   geomean = 1.119  (target 1.25, weight 1x)
Qwen3-235B    geomean = 1.117  (target 1.25, weight 1x)
score = 882
```

Lowest non-gpt_oss-K=2880 shapes:

| shape                                | ratio | path                |
|--------------------------------------|-------|---------------------|
| Qwen3-Down-B32-M4096                 | 1.104 | K=1536, ki=24, KI=0 |
| Qwen3-Down-B32-M2048                 | 1.107 | K=1536, ki=24, KI=0 |
| DSV3-Down-B32-M4096                  | 1.110 | K=2048, ki=32, KI=0 |
| DSV3-Down-B16-M2048                  | 1.116 | K=2048, ki=32, KI=0 |
| Qwen3-Down-B16-M2048                 | 1.124 | K=1536, ki=24, KI=0 |
| Qwen3-Down-B16-M4096                 | 1.142 | K=1536, ki=24, KI=0 |

All `*-Down` shapes have `K % 128 == 0`, ki not in the existing
specialization list `{56, 64, 112, 128, 172, 224, 256, 296, 448, 462, 832}`,
so they fall through to `default: launch_one_grouped<L, 0>` (KI=0 dynamic,
`#pragma unroll 2` schedule).

## Hypothesis (Lever B-spec-shortK)

Add `KI_HINT=24` and `KI_HINT=32` specializations (FUSED_KTAIL=false) so
short-K paths get compile-time loop unrolling instead of dynamic + unroll-2.

Existing data point: `KI=56 + fuse=F` is production-positive despite
spilling 12-16 VGPRs. So a moderately spilling specialization can still
beat KI=0 dynamic on short-K.

## Patch

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

```
INSTANTIATE_K_GRP(24);   // R16 — Qwen3-Down K=1536
INSTANTIATE_K_GRP(32);   // R16 — DSV3-Down K=2048
case 24:  launch_one_grouped<L, 24> (g); break;
case 32:  launch_one_grouped<L, 32> (g); break;
```

## Build resource analysis

| KI                     | VGPRs | Spill (VGPRs) | Note                |
|------------------------|-------|---------------|---------------------|
| KI=0 dynamic (fuse=F)  | 224   | 0             | unroll-2, baseline  |
| KI=24 (R16, fuse=F)    | 256   | 16-20         | full unroll, NEW    |
| KI=32 (R16, fuse=F)    | 256   | 16-20         | full unroll, NEW    |
| KI=56 (production)     | 256   | 12-16         | full unroll, BENEF. |
| KI=64 (production)     | 256   | 0             | full unroll, BENEF. |

KI=24/32 spill in same range as production KI=56 → resource gate PASSED.

## Result (R16 metric, after change)

```
gpt_oss_20B   geomean = 1.072  (-2 pp; noise — gpt_oss path unaffected)
DeepSeek-V3   geomean = 1.088  (-3 pp; affected: DSV3-Down ki=32)
Qwen3-235B    geomean = 1.071  (-5 pp; affected: Qwen3-Down ki=24)
score = 860  (-22)
```

Per-shape regression (the 8 shapes that switched from KI=0 to KI=24/32):

| shape                          | before | after | Δ      |
|--------------------------------|--------|-------|--------|
| DSV3-Down-B16-M2048            | 1.116  | 1.064 | -5.2pp |
| DSV3-Down-B16-M4096            | 1.120  | 1.052 | -6.8pp |
| DSV3-Down-B32-M2048            | 1.118  | 1.054 | -6.4pp |
| DSV3-Down-B32-M4096            | 1.110  | 1.038 | -7.2pp |
| Qwen3-Down-B16-M2048           | 1.124  | 1.024 | -10.0pp|
| Qwen3-Down-B16-M4096           | 1.142  | 1.018 | -12.4pp|
| Qwen3-Down-B32-M2048           | 1.107  | 1.017 | -9.0pp |
| Qwen3-Down-B32-M4096           | 1.104  | 1.010 | -9.4pp |

correctness_fail = 0/24 (numerics intact; pure perf regression).

## Conclusion — FALSIFIED

For short-K paths (K=1536 ki=24, K=2048 ki=32), the spill cost of
full-unroll specialization (16-20 VGPRs to scratch per iteration) >
unroll benefit. Unlike KI=56 (which has 56 K_TWO_TILE iterations to
amortize spill traffic), KI=24/32 have 24/32 iterations — too few to
hide the additional scratch traffic.

The KI=0 dynamic + unroll-2 schedule is at a **structural local optimum**
for `K % 128 == 0` && `ki < 56` shapes: the 224-VGPR / 0-spill
unroll-2 pipeline beats the 256-VGPR / 16-20-spill full-unroll pipeline.

## Closed surface

After R11 / R14 / R15 / R16, the BF16 grouped kernel KI specialization
landscape is closed:

| ki bucket          | best config                | spill | status            |
|--------------------|----------------------------|-------|-------------------|
| ki ≤ 32 (short-K)  | KI=0 dynamic + unroll-2    | 0     | OPTIMAL (R16 ✗)   |
| ki = 44 + fuse     | KI=0 dynamic + unroll-2    | 0     | OPTIMAL (R14 ✗)   |
| ki = 44 + nofuse   | (untested)                 | -     | -                 |
| ki = 56 / 64       | KI=56 / KI=64 spec         | 0-16  | OPTIMAL (prod)    |
| ki = 112+          | KI=ki spec                 | 0     | OPTIMAL (prod)    |

## Action

- HK kernel reverted to baseline (4264680-byte .so confirmed).
- Score remains at 882 baseline (no `git push`).
- Next-round suggestion: pivot AWAY from kernel KI specialization;
  explore dispatch / cfg side or accept plateau and document.
