# Round-8 — RCR baseline + analysis

## RCR baseline (post race fix only)
primary_score = 1.0620, min_ratio = 0.8345

### Per-shape RCR
| label                 | dense | grp  | route | ratio  | gap%  |
|-----------------------|------:|-----:|-------|-------:|------:|
| dsv3-down-B16-M8192   |  1968 | 1642 | bn256 | 0.8345 | 16.55 |
| qwen-down-B16-M8192   |  1812 | 1518 | bn256 | 0.8376 | 16.24 |
| dsv3-down-B4-M8192    |  1968 | 1662 | bn256 | 0.8448 | 15.52 |
| qwen-down-B16-M4096   |  1653 | 1465 | bn256 | 0.8859 | 11.41 |
| dsv3-up-B16-M8192     |  2826 | 2518 | bn256 | 0.8911 | 10.89 |
| qwen-down-B4-M8192    |  1812 | 1625 | bn128 | 0.8965 | 10.35 |
| dsv3-up-B4-M8192      |  2826 | 2539 | bn256 | 0.8983 | 10.17 |
| dsv3-down-B4-M4096    |  1918 | 1733 | bn128 | 0.9033 |  9.67 |
| (other 12 shapes pass gap <= 9% or grouped > dense)             |

7 shapes have gap > 5%. Worst: dsv3-down/qwen-down with M_g=8192 (BN=256 route).

## Architectural observation
RCR's BN=256 main loop ALREADY mirrors dense pattern (b0+b1 split, mma cA-cB-cC-cD,
epilog 1 prefetches b0 for epilog 2). So the RRR R3+R4 optimizations don't apply.

RCR-specific lever candidates:
1. AGPR pressure: RCR bn256 also uses 256 AGPR (same as RRR bn256). VGPR spill 24
   (RCR per memory). Smaller than RRR spill 37 but similar order.
2. K-loop overhead: small-K shapes (K=1536, K=2048) have higher per-tile fraction
   spent in prologue/epilog. Per-tile dispatch overhead matters more.
3. M_g=8192 stress: large per-group M means more tiles per group; tile dispatcher
   overhead accumulates.

## Test infrastructure
`quick_test_bench_rcr.py` — RCR variant of quick_test_bench.py. 20 shapes,
hk_grouped_rcr_fp8 vs hk_gemm_fp8 "rcr". Sweeps bn∈{0, 128, -128} for grouped.
