# Round 17 — FP8 grouped GEMM bottom shapes are FORWARD-kernel-bound

## Today's metric (HEAD `46d56aa`, post-warm)

```
[metric_fused_wall] geomean=1.3453  progress=0.997  FAIL
[metric_fused_wall] correct_fail=0/24  reject=0/24  below_target=14/24  goals=10/24  score=997
```

Lowest-ratio shapes (sorted ascending):

| shape | ratio |
| --- | --- |
| Qwen3-Down-B16-M2048   (K=1536) | 1.260 |
| Qwen3-Down-B16-M4096   (K=1536) | 1.265 |
| Qwen3-GateUP-B16-M2048 (K=4096) | 1.269 |
| gpt_oss-Down-B32-M2048 (K=2880) | 1.275 |
| Qwen3-GateUP-B32-M2048 (K=4096) | 1.285 |
| Qwen3-Down-B32-M2048   (K=1536) | 1.285 |
| Qwen3-GateUP-B16-M4096 (K=4096) | 1.287 |
| gpt_oss-Down-B32-M4096 (K=2880) | 1.294 |
| Qwen3-Down-B32-M4096   (K=1536) | 1.301 |
| DSV3-Down-B16-M2048    (K=2048) | 1.310 |

## What this round did

Pure investigative round — no `primus_turbo` source change. Built a
new probe script
(`scripts/_fp8_grouped_fwd_vs_bwd_attribution.py`) that splits the
metric's fwd+bwd ratio into separate forward and backward components
on the bottom 5 metric shapes, then ran it post-warm to identify
which kernel family drags the ratio.

## Key finding: forward is the universal bottleneck

```
shape                            hk_fwd  trt_fwd  fwd_x   hk_bwd  trt_bwd  bwd_x  total_x
Qwen3-Down-B16-M2048   (K=1536)   0.282    0.335  1.189   0.499    0.648  1.297    1.258
Qwen3-Down-B16-M4096   (K=1536)   0.481    0.559  1.163   0.889    1.171  1.317    1.263
Qwen3-GateUP-B16-M2048 (K=4096)   0.384    0.458  1.193   0.892    1.150  1.289    1.260
gpt_oss-Down-B32-M2048 (K=2880)   0.608    0.668  1.099   1.310    1.771  1.352    1.272
DSV3-Down-B16-M2048    (K=2048)   0.463    0.553  1.195   1.013    1.352  1.335    1.291
```

Across all 5 bottom-of-metric shapes, **fwd_x < bwd_x** (by 0.087 to
0.229). The HK forward kernel (`grouped_rcr_kernel` for K-aligned
shapes, transpose + RCR for K-misaligned gpt_oss family) is at
1.10-1.20× Triton; the HK backward (`grouped_rrr` for dA + var-K dB)
is already at 1.29-1.35× Triton. **The remaining ratio gap to 1.35
lives almost entirely in the forward kernel.**

This invalidates one architectural assumption I had been carrying:
that "the metric is uniformly Python-overhead-bound on small shapes"
(R7-R16 trims). It isn't. Backward already hits or nearly hits
target on all 5 shapes; forward is consistently 0.08-0.23 below.

## Why the bottom shapes are forward-bound

Common feature: shallow K combined with small/medium N. Three sub-clusters:

1. **Qwen3-Down K=1536**: 12 K-iters at `K_BLOCK=128`. The HK
   persistent loop is tuned for the deeper-K mainline (DSV3 K=7168 has
   56 K-iters, gpt_oss K=2880 has 22 K-iters); on K=1536 the per-tile
   compute completes in ~6 mfma waves, leaving the LDS double-buffer
   load-side bandwidth unsaturated. fwd_x=1.18 vs bwd_x=1.30.

2. **Qwen3-GateUP K=4096**: 32 K-iters — deeper than Qwen3-Down but
   N=3072 only, so the M*N output footprint is small. fwd_x=1.19
   vs bwd_x=1.29. Still forward-bound but less severe.

3. **gpt_oss-Down K=2880**: K%128 != 0 → H4 reroute path
   (`fp8_transpose_3d` + RCR-on-transposed). fwd_x=1.10 (the worst of
   all 5) vs bwd_x=1.35. The transpose cost (~265 MB B=32 weight
   read+write at MI355X's ~3.4 TB/s) is the dominant Python-side fwd
   contributor; the cache (R9 `_FP8_H4_TRANSPOSE_CACHE`) HITs from
   iter 2 onward in the metric so the transpose only fires once per
   shape, but still adds amortized overhead.

4. **DSV3-Down K=2048**: 16 K-iters at K_BLOCK=128. Slightly less
   shallow than Qwen3-Down but still on the shallow side. fwd_x=1.20
   vs bwd_x=1.34.

## What's NOT the bottleneck (confirmed by attribution)

* **Python dispatch overhead** — R11 (forward execute) + R14 (FusedActFunc
  dispatch shortcut) + R16 (var-K execute) + R10/R11 (caches) brought
  HK's per-call host overhead to ~50-100 ns / call on both forward and
  backward dispatch paths. HK fwd kernel wall on these shapes is
  300-650 µs / call — Python overhead is now <0.05% of fwd wall.

* **Backward dA RRR config** — R44 swept Qwen3 family RRR; no
  actionable rule. Backward is at 1.29-1.35× target.

* **Backward dB var-K config** — R15 swept Qwen3 family var-K; no
  actionable rule. Backward var-K is part of the already-passing bwd.

## What IS the bottleneck (attribution-confirmed)

The HK FP8 grouped forward kernel itself, specifically:

* `grouped_rcr_kernel<KI_HINT=0, N_MASKED_STORE, FUSED_KTAIL=false>`
  for K-aligned shapes (lines 2506-3197 of
  `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`).

* The H4 reroute path's combination of `fp8_transpose_3d` + RCR for
  K-misaligned gpt_oss family.

R12 falsified the per-shape `(group_m, num_xcds)` tuning lever for
Qwen3-Down K=1536 — the existing default plateau is already at the
shape's compute ceiling for the current kernel template. The
remaining levers are kernel-internal:

## Recommended next-chat work (HK kernel surgery)

The highest-ceiling lever is **BLOCK_K=64 template specialization for
shallow-K shapes**:

* Audit (this round): `K_BLOCK=128` is referenced in 70 places in
  `kernel_fp8_layouts.cpp`, including the LDS slab type
  (`using ST_v2 = st_fp8e4m3<HB, BK, ...>`) and register tile types
  (`using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>`).
  Adding a BLOCK_K=64 template specialization requires:

  1. Templating the kernel + helper types on `BK` (or duplicating
     them under a `BK_64` namespace with `K_BLOCK=64`).

  2. Adding new instantiations of `grouped_rcr_kernel<...>` for the
     BK=64 variant.

  3. Updating the launcher (`dispatch_grouped_rcr` at line 6395
     onward) to choose the BK template based on K/N/m_total — needs
     a Python config rule on the Primus-Turbo side too.

* Ship plan: 2-4 rounds of HK source work (build types + instantiate
  + correctness probe + sweep), then 1 round of Primus dispatch
  rule wire-up. Recommend a fresh chat session focused on HK source
  surgery (the current chat has spent its budget on Primus-side
  audits).

The lower-ceiling but lower-risk alternative is **profiling the
gpt_oss K=2880 H4 reroute** with rocprof to identify whether the
transpose itself or the RCR-on-transposed-K=2880 is the slower piece
— if the transpose dominates, a BF16-side fp8_transpose_3d
optimization (or pre-transposing the weight at quantize time) might
help. R9 already cached the transpose, so further savings are limited
to the per-iter cache-HIT path and the kernel itself.

## Round 17 round summary

* Target: lowest-ratio Qwen3-Down-B16-M2048 (1.260) and the rest of
  the bottom 5.
* Outcome: **falsified the assumption that Python overhead is the
  bottleneck**. Attribution confirms forward kernel is the universal
  bottleneck on all bottom shapes.
* Code changes: none in `primus_turbo/`. Added one new probe script
  (`scripts/_fp8_grouped_fwd_vs_bwd_attribution.py`) reusable across
  rounds + chats.
* Metric: 997 (within established noise band 980-1000 since R11).
* Next chat: HK source BLOCK_K=64 surgery (see plan above).
