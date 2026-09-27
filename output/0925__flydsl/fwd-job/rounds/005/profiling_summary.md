# Profiling summary — round 5, op/current (`kn_fmha_fwd_prefill_a16w16_m32x8_bshd`)

> **Benchmark caveat.** All rates below come from `benchmark.py` on an **unprofiled** run: median of 101 calls, L2 flushed before every timed call, palindromic arm order, and `beat` (aiter gfx1250 ASM forward) measured in the same run. Rates from the profiler are not used anywhere. The card is VR-throttled: the sclk DPM maximum is 1100 MHz, and prod sustains about 988–1030 MHz.

## Benchmark

| shape | (B, Sq, Skv, Hq, Hkv, D) | current TFLOP/s | beat TFLOP/s | current/beat | sclk MHz |
|---|---|---:|---:|---:|---|
| fast | (1, 1024, 1024, 8, 2, 128) | 38.27 | 69.96 | 0.547 | 1100 |
| proxy | (1, 4096, 4096, 32, 8, 128) | 703.21 | 1008.13 | 0.698 | 1027–1062 |
| prod | (4, 8192, 8192, 32, 8, 128) | 1083.33 | 1410.87 | **0.768** | 1019–1031 |

Details: [benchmark-results.md](benchmark-results.md). This kernel accounts for 94.4% of prod time and 85.6% of fast time ([kernel.yaml](kernel.yaml)).

## Verdict

- **prod: `latency`, confidence low.** The limit is on-chip execution efficiency; compute/issue cannot be excluded.
  - HBM is excluded because cycles per call stay constant when the clock moves.
  - Clock can recover at most +11%.
  - Beat does 1.30× the work per second in the same card state.
  - [6-bound-analysis](6-bound-analysis/analysis.md), [5-power-wall-analysis](5-power-wall-analysis/analysis.md)
- **fast: `latency`, confidence medium-high.**
  - The grid underfills the machine: 32 WGs on 256 CUs.
  - Per-call and cold-cache cost is high: 35.3 µs warm vs 56.2 µs flushed.
  - The kernel is not power-limited.
  - [6-bound-analysis](6-bound-analysis/analysis.md)
- `bound: latency` goes into state.yaml through `verdict.bound`, which is the prod verdict.

## Candidates (ranked; prod first per correction 4; none priced, pricing is on card)

1. **c1 (prod):** per-cycle efficiency of the KV main loop, specifically WMMA feed and softmax VALU/WMMA overlap in the schedule. About 23% of the gap to beat is not clock. See [6-bound-analysis](6-bound-analysis/analysis.md) and [benchmark-results.md](benchmark-results.md).
2. **c2 (prod):** energy per cycle.
   - With random data the clock drops to 988 MHz; with zeros it is 1053 MHz, and time falls by 7.2%.
   - Reducing switching activity pays twice, in cycles and in clock. The clock part is capped at about +11%.
   - See [5-power-wall-analysis](5-power-wall-analysis/analysis.md).
3. **c3 (fast):** grid underfill, which caps work at ≤12.5% of CUs. Get more WGs for small grids by splitting Q rows or KV. Must not regress prod. See [kernel.yaml](kernel.yaml) and [6-bound-analysis](6-bound-analysis/analysis.md).
4. **c4 (fast):** per-call / cold-cache cost (1.59× warm→flushed). Before pricing this, measure warm vs flushed for both current and beat; that is the cheapest next measurement.

Non-findings are listed in [profiling.yaml](profiling.yaml) under `non_findings`: no spill; I-cache is not a factor; HBM is not the prod bound; the prod grid fills the machine; occupancy alone does not explain the gap; power is not at the socket cap.

## No data

- **Bytes at any memory level:** gfx1250 has no byte counter, so there is no roofline, and L2/LDS bandwidth can be neither ruled in nor ruled out.
- **WMMA utilisation:** the counters read zero.
- **Stall/issue attribution:** closed, and PC sampling is forbidden.
- **Thread trace:** [4-thread-trace](4-thread-trace/provenance.yaml) was skipped because ATT sees no FlyDSL kernels.
- **Panel metrics:** [3-kernel-metrics](3-kernel-metrics/provenance.yaml) was skipped because rocprof-compute is unavailable.
- **Compute peak:** bf16 WMMA FLOP/cycle is undocumented.
- **VGPR count:** 232 read, about 464 by correction 1; not confirmed from the ISA.
- **beat warm time at fast:** not measured.

Counter evidence (SQ busy, waves, I-cache) is in [2-kernel-profiling](2-kernel-profiling/kernel-kn_fmha_fwd_prefill_a16w16_m32x8_bshd/analysis.md).
