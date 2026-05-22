# Round-2 BASELINE — after RRR+RCR race fix

## Hypothesis
No code change yet. Re-baseline after RRR race fix (HK 05768db7) + RCR race fix (HK 6c7327bc) + dispatcher gate. Test if race fix changed perf landscape.

## Single change
None. Kernel = HEAD `6c7327bc`.

## Results

primary_score (geomean ratio) = **1.1524**  | min_ratio = **0.8277**  | n_pass = **20/20** (all correctness pass — race fix verified)

### Per-shape (sorted by ratio asc — worst first)

| label                 | dense | grp  | route | ratio  | gap%  |
|-----------------------|------:|-----:|-------|-------:|------:|
| dsv3-up-B4-M4096      |  2560 | 2119 | bn256 | 0.8277 | 17.23 |
| dsv3-up-B4-M8192      |  2622 | 2188 | bn256 | 0.8343 | 16.57 |
| dsv3-up-B16-M4096     |  2560 | 2144 | bn256 | 0.8375 | 16.25 |
| dsv3-up-B16-M8192     |  2622 | 2206 | bn256 | 0.8413 | 15.87 |
| qwen-down-B16-M8192   |  1784 | 1534 | bn128 | 0.8598 | 14.02 |
| dsv3-down-B4-M8192    |  1569 | 1625 | bn128 | 1.0357 | -3.57 |
| dsv3-down-B16-M8192   |  1569 | 1664 | bn128 | 1.0603 | -6.03 |
| dsv3-down-B16-M4096   |  1532 | 1650 | bn128 | 1.0773 | -7.73 |
| qwen-down-B4-M8192    |  1784 | 1811 | bn128 | 1.0153 | -1.53 |
| dsv3-down-B4-M4096    |  1532 | 1787 | bn128 | 1.1667 | -16.67|
| qwen-up-B16-M4096     |  1660 | 1939 | bn128 | 1.1682 |  ...  |
| (other 9 shapes ratio > 1, omitted)                         |

## Attack surface analysis
- **15 cases** already pass (ratio ≥ 0.95 → gap ≤ 5%)
- **5 cases fail**: 4× dsv3-up (route=bn256) + 1× qwen-down-B16-M8192 (route=bn128)

dsv3-up shape: N=4096 K=7168 → BN=256 path; gap 15-17% consistent across B/M_g
qwen-down-B16-M8192: K=1536 large batch → bn128 path; bandwidth-bound per prior H14 analysis

Note: qwen-down-B16-M8192 was previously flagged as bandwidth-bound (physics-bound). Race fix may or may not have changed this — re-bench will tell.

## Next: Round 3 — focus dsv3-up bn256 path
Target: 4 dsv3-up shapes 17%→5%. Possible levers:
1. BN=256 RRR autotune param sweep (RRR_PREFETCH_LGKM, RRR_STEADY_VMCNT)
2. BN=256 RRR chunk_size sweep
3. BN=256 RRR main loop instruction ordering (analogous to bn128 H6+H7)
