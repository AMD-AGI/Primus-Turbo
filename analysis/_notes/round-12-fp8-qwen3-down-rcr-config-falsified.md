# Round 12 — Qwen3-Down FP8 RCR config search FALSIFIED + state-of-play

## Target shape

Lowest ratio in pre-R12 metric (today, fresh run on MI355X / GPU 3):

```
fusedFP8_Qwen3-235B-A22B-Down-B16-M2048   1610.7  1293.7  ratio=1.245  <135%
```

R11 closed grad_out Q-tax via per-layer delayed-scale (1-pass cvt).
The remaining gap on this shape is the **HK forward kernel itself**:
HK 1610 TF vs Triton 1294 TF, ratio 1.245 — gap to target 1.35 is
8.4 pp.  Per-call breakdown (post-R10/R11):

```
fwd kernel   ~219 µs   (HK kernel-only TFLOPS ~1890 — sweep below)
Q_a          cached    (R10 hit)
Q_b          cached    (R9  hit)
bwd dA       ~204 µs   (RRR rule already in place; R42 tight-verified)
bwd dB var-K ~?  µs   (R39 rule in place: m_total>=16384 → gm=8 xcd=4)
Q_grad_out   ~83 µs    (R11 1-pass cvt)
```

R6 explicitly skipped Qwen3-Down-M=2048 from the FP8 RCR tile-config
rule, citing a -3.95 pp regression at `(gm=2, xcd=8)` on M=2048 in
that round's sweep.  R6 didn't publish a wider-than-(gm=2) sweep on
M=2048 — re-probed this round.

## What was probed

`/tmp/sweep_qwen3_down_b16_m2048_fp8_fwd.py` — 18-cell `(group_m,
num_xcds)` grid sweep on Qwen3-Down-B16-M2048 + B32-M2048 sibling,
200 iters × p20 timing.  All non-default cells regressed; default
`(gm=4, xcds=0 → kernel BLOCK_SWIZZLE_NUM_XCDS=8)` plus its explicit
twins `(4, 8)` / `(4, 16)` cluster at the top:

```
Qwen3-Down-B16-M2048
  ( 4,  8)   1881.3   +1.20% (within noise — see tight verify)
  ( 4, 16)   1876.9   +0.97%
  ( 4,  0)   1858.9   +0.00% baseline
  (16,  4)   1823.4   -1.91%
  (32,  4)   1820.5   -2.07%
  ( 1,  4)   1815.1   -2.36%
  ( 4,  4)   1808.7   -2.70%
  ( 4,  2)   1789.9   -3.72%
  ( 8,  8)   1772.6   -4.64%
  ( 1, 16)   1764.1   -5.10%

Qwen3-Down-B32-M2048
  ( 4,  0)   1902.2   +0.00% *winner
  ( 4,  8)   1897.4   -0.25%
  ( 4, 16)   1894.1   -0.42%
  (16,  4)   1851.6   -2.66%
  ( 1,  4)   1846.3   -2.94%
  ( 2,  *)   1817..1834  -3.6% to -4.9%
```

## Tight verify on the apparent (4, 8) signal

`/tmp/verify_qwen3_down_b16_m2048_fp8_fwd.py` — 5-trial × 300-iter
× p20 median across all 4 Qwen3-Down M={2048, 4096} shapes:

```
shape                cfg     median TF   spread%   Δ vs (4, 0)
B16-M2048           (4, 0)   1889.96     0.78%     +0.00%
                    (4, 8)   1893.43     0.79%     +0.18%
                    (4, 16)  1891.69     0.13%     +0.09%
                    (4, 4)   1821.17     0.07%     -3.64%
B32-M2048           (4, 0)   1895.17     0.47%     +0.00%
                    (4, 8)   1895.00     0.46%     -0.01%
                    (4, 16)  1897.26     0.74%     +0.11%
                    (4, 4)   1845.30     0.35%     -2.63%
B16-M4096           (4, 0)   1836.09     0.33%     +0.00%
                    (4, 8)   1840.19     0.32%     +0.22%
                    (4, 16)  1839.70     0.25%     +0.20%
                    (4, 4)   1818.12     0.21%     -0.98%
B32-M4096           (4, 0)   1880.82     0.44%     +0.00%
                    (4, 8)   1881.51     0.37%     +0.04%
                    (4, 16)  1883.65     0.36%     +0.15%
                    (4, 4)   1848.69     0.46%     -1.71%
```

Δ for `(4, 8)` / `(4, 16)` vs `(4, 0)` is +0.04 % to +0.22 % — **within
the 0.13–0.79 % run-to-run spread**; the first sweep's +1.20 % on
B16-M2048 was a single-trial fluke that the 5-trial median absorbs.
Bit-equivalence verified for `(4, 0)` vs `(4, 8)` on all 4 shapes
(`max_abs_diff = 0.0`), confirming the binding's 0 → 8 fallback.

`(gm=4, xcds=4)` is a clean LOSER on every shape (-0.98 % to
-3.64 %), which matches R6's narrative: **the Qwen3-Down K=1536
family sits on a flat plateau around the binding default; gm=4 is
optimal, and any explicit xcd value other than the default-mapped 8
or its near-neighbour 16 regresses**.

No FP8 RCR tile-config rule lift available for this family.

## Other tile-config angles already covered

* dA RRR — `tiles_n <= 8 and m_total >= 32768 → (gm=16, xcds=4)`
  (R42); active for all 4 Qwen3-Down dA paths.
* dB var-K — `m_total >= 16384 → (gm=8, xcds=4)` (R39); active for
  all 4 Qwen3-Down dB paths (B≥16 gives m_total ≥ 32768).
* Q-tax — R9 (b cache) + R10 (a cache) + R11 (grad_out delayed
  scale) close the activation/weight quantize lever.

## State of play

The 24-shape geomean is **0.991 of target** (today's first run; best
recorded 999 across recent runs).  All cheap levers — caches at
3 levels, dispatch tile-config rules at the 3 layouts (RCR / RRR /
CRR), grad_out delayed scale — are now wired.  The remaining ~10–15
score points to push every shape past `>= 1.35` will require **HK
kernel-internal work**, not Python-side dispatch:

1. **Qwen3 family has K=1536 (12 K-iters per tile-step, BLOCK_K=128) —
   shallow vs DSV3 K=2048, gpt_oss K=2880, DSV3-GateUP K=7168**.  The
   HK persistent loop is tuned for the deeper-K mainline; on shallow-K
   shapes the per-tile compute completes in ~6 mfma waves and the LDS
   double-buffer is under-used.  Worth probing whether a smaller
   `BLOCK_K` template (= 64 or 32, doubling the K-iters) on the
   Qwen3-Down/Qwen3-GateUP K∈{1536, 4096} family changes the LDS
   reuse profile.  This is HK-side `tk_fp8_layouts.cpp` work, not a
   Python config rule.

2. **`grouped_dscale` binding signature has no `kernel`-template
   override for grouped FP8 RCR**.  The dense FP8 RCR path uses
   `force_rcr_kernel(env)` to pick between 4-wave / 8-wave templates;
   the grouped binding always runs template 8 internally.  If the
   grouped binding gains a `kernel` arg (mirror of the dense
   pybind11 signature), per-shape template selection becomes a
   Python rule lever again — could yield 1-3 pp on Qwen3-Down per
   the dense FP8 win pattern.

3. **Forward delayed-scale fallback for `a` MISS path (R12 idea
   considered, declined this round)**.  When R10's tensor cache
   misses (different `a` per call but same layer), today the path
   falls back to a full 2-pass quantize.  A symmetric delayed-scale
   fallback (mirror of R11) would 1-pass cvt on miss — 30 % faster.
   No metric impact (metric loop hits R10 every iter after warmup),
   so explicitly skipped this round; useful as a real-training
   completion item for the activation pipeline.

## No code change committed

This round is a docs-only round: re-probed an explicitly-skipped R6
slot, confirmed the FP8 RCR plateau, recorded the falsification for
the chain of work.  No primus_turbo code change.  No HK code change.
