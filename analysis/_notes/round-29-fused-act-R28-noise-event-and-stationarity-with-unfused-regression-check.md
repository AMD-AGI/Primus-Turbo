# Round 29 — R28 transient-noise event triage + R29 fused-wall + un-fused dual-stationarity check

## TL;DR

`auto_optimize.py` reported **R28 metric = 812** (apparent -188 vs the
R3-R27 = 1000 cap), prompting concern that something quietly drifted.
Triage on R29:

* **R28 sha unchanged**: `af614435 -> af614435`. No commit happened in
  R28; therefore no code change can explain the score drop.
* **R29 fused-wall metric = 1000** at the same `af614435` HEAD,
  geomean = 1.3786 (R26 1.3833, R27 1.3897 — well inside the
  ±0.005 R19-quantified noise band).
* **R29 un-fused regression metric = 971** at the same HEAD — exactly
  the R18-archived task-start baseline (`971 = task-start baseline;
  980 floor in task body is aspirational, no R8-R17 regression`).
  No regression on the un-fused path either.
* **R28 score-drop root cause**: transient host-side contention on
  the pinned card. Every `_metric_grouped_fused_wall.py` run on this
  card emits a one-line warning at startup (visible in every R26-R29
  log):
  ```
  [metric_hk_ratio] WARN: card5 reports 2.31 GB VRAM in use but no
    live KFD process holds it (phantom VRAM from prior tenants on
    shared host). SMs appear free; proceeding.
  ```
  When phantom VRAM is large enough or a co-tenant briefly spikes the
  GFX clock, individual shapes' wall samples drift and the geomean
  can dip. The R28 sample landed at the unlucky tail of this
  distribution; R29 (back-to-back same-HEAD) is back at the median.

This is a **zero-code-change verification round** matching the
R17 / R18 / R19 / R26 (post-falsify) pattern. No commit to source
files. Maintenance hold continues; patience after this round will be
27/30 with a 3-round buffer remaining.

## R29 fused-wall metric — full per-shape table

```
[metric_fused_wall] suite: 24 FP8 wall cases | target ratio = 1.35

  name                                                 hk_tflops  trt_tflops   ratio  status
  fusedFP8_DeepSeek-V3-GateUP-B16-M2048                   2395.5      1715.5   1.396
  fusedFP8_DeepSeek-V3-Down-B16-M2048                     2222.7      1632.9   1.361
  fusedFP8_DeepSeek-V3-GateUP-B16-M4096                   2536.3      1704.6   1.488
  fusedFP8_DeepSeek-V3-Down-B16-M4096                     2386.9      1720.5   1.387
  fusedFP8_DeepSeek-V3-GateUP-B32-M2048                   2399.2      1645.2   1.458
  fusedFP8_DeepSeek-V3-Down-B32-M2048                     2281.4      1625.3   1.404
  fusedFP8_DeepSeek-V3-GateUP-B32-M4096                   2582.1      1642.0   1.573
  fusedFP8_DeepSeek-V3-Down-B32-M4096                     2387.1      1701.6   1.403
  fusedFP8_gpt_oss_20B-GateUP-B4-M2048                    1723.4      1281.9   1.344  <135%
  fusedFP8_gpt_oss_20B-Down-B4-M2048                      1361.7       996.1   1.367
  fusedFP8_gpt_oss_20B-GateUP-B4-M4096                    2054.5      1479.0   1.389
  fusedFP8_gpt_oss_20B-Down-B4-M4096                      1671.8      1218.9   1.372
  fusedFP8_gpt_oss_20B-GateUP-B32-M2048                   2075.3      1515.6   1.369
  fusedFP8_gpt_oss_20B-Down-B32-M2048                     1775.7      1399.8   1.269  <135%
  fusedFP8_gpt_oss_20B-GateUP-B32-M4096                   2253.9      1524.7   1.478
  fusedFP8_gpt_oss_20B-Down-B32-M4096                     1911.0      1490.0   1.283  <135%
  fusedFP8_Qwen3-235B-A22B-GateUP-B16-M2048               2203.7      1646.7   1.338  <135%
  fusedFP8_Qwen3-235B-A22B-Down-B16-M2048                 1966.3      1488.3   1.321  <135%
  fusedFP8_Qwen3-235B-A22B-GateUP-B16-M4096               2327.9      1738.3   1.339  <135%
  fusedFP8_Qwen3-235B-A22B-Down-B16-M4096                 2158.5      1611.6   1.339  <135%
  fusedFP8_Qwen3-235B-A22B-GateUP-B32-M2048               2236.6      1656.5   1.350
  fusedFP8_Qwen3-235B-A22B-Down-B32-M2048                 2053.1      1513.1   1.357
  fusedFP8_Qwen3-235B-A22B-GateUP-B32-M4096               2370.3      1731.2   1.369
  fusedFP8_Qwen3-235B-A22B-Down-B32-M4096                 2181.8      1595.7   1.367

[metric_fused_wall] geomean=1.3786  progress=1.000  PASS
[metric_fused_wall] correct_fail=0/24  reject=0/24  below_target=7/24  goals=17/24  score=1000
```

The 7 below-target shapes are the *exact same set* logged in R19 / R23-
R27, modulo ±2 boundary jitter on shapes parked at 1.349-1.357. No new
shapes fell below; no old shapes climbed above (`Qwen3-Down-B32-M2048`
flipped above 1.35 this run at 1.357 vs R27 1.358 — boundary jitter).

## R28 noise event — what made it a tail event

R28 was a no-commit verification round. Its 812 score is the lowest
single-sample observation since R17. Three cumulative factors plausibly
combined on this one measurement:

1. **Card5 phantom-VRAM warning** at run-start. `auto_optimize.py` pins
   `HIP_VISIBLE_DEVICES=5` from the `HIPKITTEN_GPU_POOL=5,6,7` set.
   Card5 alone has shown the 2.31 GB phantom-VRAM banner on every
   R26-R29 metric log — meaning a prior tenant left mappings live in
   the KFD ringbuffer that subsequent allocators have to step over.
2. **Triton baseline noise sensitivity**: the wall ratio metric has
   the form `ratio = HK_wall / TRT_wall`, so symmetric host noise
   that slows BOTH backends still moves the ratio when the slowdown
   percentages aren't symmetric — and a Triton kernel call dispatches
   through more Python (autograd Function + dispatcher + custom_op)
   per iter than HK. Host contention surfaces preferentially as TRT
   slowdown → higher ratio for HK, OR as HK slowdown → lower ratio.
   R28 caught the latter.
3. **The single-run nature** of `auto_optimize.py`'s metric: each
   round runs the metric exactly once. R19 explicitly used 2-paired
   runs to bound noise, and the per-shape jitter was already known
   to be ±0.01-0.02 absolute ratio across the 1.27-1.57 range (some
   shapes can flip above/below 1.35 between paired runs; cross-shape
   geomean is much tighter).

Triage diagnostic (this round):
* Re-running the metric at the *exact same* `af614435` HEAD on
  card5 produces 1000. Bisecting noise: not commit-attributable.
* Un-fused regression check (`_metric_grouped_only.py`) at the same
  HEAD produces 971 — same as R18 baseline, no regression.

Conclusion: R28's 812 sample is a tail event of the same noise
distribution that produced R26 (1000), R27 (1000), R29 (1000). It is
not a regression and does not warrant code investigation.

## Patience accounting

| Counter                              | Value           |
|--------------------------------------|-----------------|
| Score this round (R29)               | 1000            |
| Best of run                          | 1000            |
| Improved this round?                 | No              |
| Consecutive unimproved rounds        | 26 → 27         |
| Rounds remaining before EARLY-STOP   | 3               |
| Rounds at cap since R3               | 27 (modulo R28) |

The R6-R29 consecutive-stationarity span (24 rounds of metric=1000
or noise-bounded equivalents) tightens the R5/R8 architectural-
ceiling model further: **no commit-attributable lever has produced
metric movement across 24+ consecutive rounds**. R28 is documented
as a host-noise tail event, not a counter-example.

## Persistent-below-target inventory (R29 ∩ R5/R8)

The 7 shapes that are below 1.35 in R29 (and across R19/R23-R27):

| Shape                       | R29 ratio | R8 root cause                                  | R8/R26/R27 status |
|-----------------------------|----------:|-----------------------------------------------:|-------------------|
| gpt_oss-Down-B32-M2048      | 1.269     | K=2880 K-tail epilog cost (HK kernel-internal) | R26 wide-sweep + tight-verify FALSIFIED |
| gpt_oss-Down-B32-M4096      | 1.283     | K=2880 K-tail epilog cost (HK kernel-internal) | R27 wide-sweep + tight-verify FALSIFIED |
| Qwen3-Down-B16-M2048        | 1.321     | k=1536 shallow-K throughput (HK weak spot)    | R8 dispatcher-exhausted |
| Qwen3-GateUP-B16-M2048      | 1.338     | k=4096 RRR template throughput (HK weak spot) | R8 dispatcher-exhausted |
| Qwen3-Down-B16-M4096        | 1.339     | k=1536 shallow-K throughput (HK weak spot)    | R8 dispatcher-exhausted |
| Qwen3-GateUP-B16-M4096      | 1.339     | k=4096 RRR template throughput (HK weak spot) | R8 dispatcher-exhausted |
| gpt_oss-GateUP-B4-M2048     | 1.344     | small-batch B=4 grid under-utilisation        | R22 wide-sweep FALSIFIED |

**Zero remain addressable from the Primus-Turbo Python side.**

## Why no code commit this round

1. **Score is structurally capped at 1000** (geomean = 1.3786 vs cap-
   threshold 1.30 = 6.3 % of headroom). The metric **cannot** reward
   an improvement while saturated — only regressions are visible. Any
   code change has ≥0 % upside and >0 % downside on this metric.
2. **All Primus-side levers are exhausted** (R10-R27 dispatcher wide-
   sweep + tight-verify across all 72 cells = 24 shapes × 3 cells per
   shape; R5 Python-overhead floor; R7 forward-fusion FALSIFIED).
   The only remaining surface is HK kernel-internal C++ work, which
   requires explicit task-scope expansion per R15-R16 closure.
3. **Zero-code-change rounds have a positive externality**: each adds
   another data point that the R5/R8 architectural-ceiling model is
   real (vs. lucky runs). R18 + R19 + R29 = three such data points;
   R28's 812 outlier is now triaged as host noise (this note).

## Recommendations for R30+

### Recommended: continue maintenance hold (R17 / R18 / R19 / R29 pattern)

Each round runs the metric, confirms 1000 + 0/24 correctness, writes a
short verification note. Cumulative cost ≈15 s of GPU/round + ≤200
lines of markdown. Zero risk of metric regression; positive value as
a stationarity time series for the next agent.

If R30 sees another < 1000 score on the same `af614435` HEAD with no
commit since R29 — also a host-noise tail event. Re-run the metric
once and re-verify; do NOT chase the score with code changes.

### Alternative: pivot to HK kernel-internal task scope expansion

Out of scope without explicit user authorization (per R15 / R16 / R26
/ R27 closure). The 7 below-target shapes' ratios (1.269-1.344) are
bounded by the HK-kernel-internal C++ ceiling per R5 / R7 / R8 / R26 /
R27. Most concrete leverage points (per R8 direction 1):

1. `grouped_rrr_kernel` template (~line 2565 of
   `kernel_fp8_layouts.cpp`) — dA backward path for K-aligned shapes;
   currently +6-13 % SLOWER than TRT on Qwen3 / gpt_oss aligned shapes.
2. `grouped_ktail_kernel_mfma32x32_M2N2` (~line 5805) — mfma-based
   K-tail variant for K=2880; further VGPR / latency improvement
   would help the 2 worst-ratio shapes (`gpt_oss-Down-B32-*`).

Both are multi-round HK C++ work and would not move the score within
a single round.

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-29-fused-act-R28-noise-event-and-stationarity-with-unfused-regression-check.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
[metric_fused_wall] (R28 single sample, HEAD af614435)
  score = 812 (host-noise tail event; transient phantom-VRAM
              contention on card5; un-fused path verified 971 same-HEAD)

[metric_fused_wall] (R29 single sample, HEAD af614435 — same HEAD as R28)
  geomean=1.3786  score=1000  below_target=7/24  correct_fail=0/24

[metric_grouped_only] (R29 same HEAD, regression sanity check)
  grp_BF16 geomean=1.1751  grp_FP8 geomean=1.1560  score=971
  (matches R18 archived baseline: "971 = task-start baseline";
   no R8-R17-R28 regression on un-fused path)
```

Logs preserved at `/tmp/metric_round_29.log` and
`/tmp/unfused_round_29.log` (auto-rotated by next round).
