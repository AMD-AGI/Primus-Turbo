# Fast fp8 wgrad for mega MoE backward — findings + kernel request (#407 follow-up)

## Context
mega MoE backward wgrads: `dW2[g] = dispatch_l2_grad[g]^T @ act_weighted[g]`,
`dW1[g] = grad_l1[g]^T @ pool_x[g]` — variable-K TN, contraction over the pool tokens.
The pool is BLOCK_M(256)-padded per expert (`handle[9]/[10]` are PADDED group_lens/offs;
tight per-expert counts are NOT exposed in the handle).

## What works today (post main-merge e54a7ab)
| wgrad kernel | granularity | speed vs bf16 | real-pool SNR |
|---|---|---|---|
| `grouped_gemm_variable_k_impl` (Triton bf16) | — | 1.00x | 55.6 dB (ref) |
| `grouped_gemm_fp8_variable_k_impl` MX_BLOCKWISE (Triton) | per-1x32 mxfp8 | **0.85-1.09x (no win)** | 22.5 dB |
| `..._tensorwise_flydsl_kernel` (#407 4-wave whole-loop) | per-tensor | **1.4-1.8x** | **0.00 dB** |

So there is a fast FlyDSL wgrad (#407, 1.4-1.8x) and a padding-robust one (mxfp8 Triton,
22.5 dB) — but NOT both in one kernel.

## Root cause of the tensorwise 0 dB (measured)
A granularity sweep on realistic data (experts differ up to 8x in magnitude; padding rows
present) computing wgrad over REAL rows only:

| padding | tensorwise | per-group tw | rowwise | (naive mxfp8) |
|---|---|---|---|---|
| zeroed | 28.5 | 28.5 | 28.6 | 11.6 |
| finite garbage (±20) | 28.6 | 28.5 | 28.5 | 11.9 |

=> **granularity is NOT the accuracy discriminator** — tensorwise/per-group/rowwise are all
~28 dB even with finite garbage padding and 8x cross-expert spread. The real-pool 0 dB is
therefore **pathological pool padding** (large/stale/uninit values) blowing up the single
GLOBAL amax so real values quantize to ~0. The Triton mxfp8 path survives only because its
grouped quant masks padding via group_offs AND per-1x32 localizes any damage.

## The fix / request (cheap, grounded in the #407 kernel structure)
`_compile_grouped_tn_wgrad_4wave` uses the UNSCALED `Mfma16x16x128` + a per-tensor scalar
epilogue scale (`StoreCPerTensor`). So:
- **mxfp8 (per-1x32)** = scale per K-block INTO the main loop -> must thread `MfmaScale16x16x128`
  + a scale loader through the hand-scheduled bare-asm whole-loop. Expensive, high-risk.
- **per-group tensorwise** = still an EPILOGUE scale, just `a_scales`/`b_scales` of length G
  (indexed by the tile's group) instead of 1. Cheap: a `StoreCPerTensor` tweak + relax the
  `numel()==1` gate in `GroupedGEMMFP8VariableKFlyDSLBackend.is_supported`. Per-group amax
  (over each expert's real rows) is padding-robust (28.5 dB) AND captures cross-expert spread.

**Recommended ask to kyle:** add a **per-group (length-G) tensorwise** scale mode to the
4-wave whole-loop wgrad (epilogue vector-of-scalars, no main-loop change). Host computes the
per-group amax over real rows (needs tight per-group counts — either expose them from the
prologue, or compute amax with a group_offs-masked reduction that skips padding). Expected:
keep the 1.4-1.8x AND reach ~28 dB. This is the minimal robust+fast wgrad; a full per-1x32
mxfp8 fast kernel is the heavier alternative if per-group tensorwise SNR proves insufficient
in end-to-end training loss.

## Whole-step payoff bound (so the request is scoped honestly)
dW1+dW2 ~= 0.45-1.4 ms; a 1.5x wgrad saves ~0.15-0.47 ms. The mega backward is dominated by
the comm-bound dispatch/combine (STEP1 alone ~2.3 ms), so the fast wgrad is a real but bounded
whole-step win. The already-landed win is the mxfp8 FORWARD (L1 1.4x). Backward fp8 net gain
at these DSv3 EP8 shapes is small; a compute-bound regime (large tokens/expert) would widen it.
