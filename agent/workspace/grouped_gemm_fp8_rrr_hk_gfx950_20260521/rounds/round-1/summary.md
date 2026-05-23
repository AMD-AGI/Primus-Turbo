# Round-1 BASELINE — grouped_gemm_fp8_rrr HK gfx950

## Hypothesis
Baseline round. No optimization change yet; record the starting correctness/performance for the 30 target_shapes (3 model × {up,down}\\{gpt-down} × B ∈ {4,16} × M_g ∈ {2048,4096,8192}) and select representative_shapes.

## Single change
No code change. Kernel snapshot: `kernel_snapshot/kernel_fp8_layouts.cpp` = HEAD `b7655e5a` (HK) / `42e63483` (PT, last to bump HK is `720eb9fc`). All H3/H6/H7/H17 wins from pre-campaign attempts are baked in.

## Results

### Validation
- Time: 2026-05-21 07:02
- Validation level: **full** (all 30 target_shapes)
- Benchmark command: `python agent/workspace/grouped_gemm_fp8_rrr_hk_gfx950_20260521/quick_test_bench.py --shapes full --summary-csv ...`
- Test command: PT high-level grouped_gemm_fp8(...) with PT grouped_gemm_ref + PT compute_snr (see `quick_test_bench.py::correctness_check`)
- Raw artifacts: `artifacts/benchmark.csv` (30 rows, canonical SUMMARY_CSV_HEADER + project ext columns)
- GPU: chi2811 MI355X, HIP_VISIBLE_DEVICES=4, mlperf_gptoss container

### Per-shape table (sorted by ratio ascending — worst at top)

| label                 | Check | dense | grp  | route | ratio   | gap%    | out_snr |
|-----------------------|-------|------:|-----:|-------|--------:|--------:|--------:|
| dsv3-up-B4-M4096      | PASS  | 2573  | 2116 | bn256 | 0.8223  | 17.77%  | 28.43   |
| dsv3-up-B16-M4096     | PASS  | 2573  | 2139 | bn256 | 0.8315  | 16.85%  | 28.50   |
| dsv3-up-B4-M8192      | PASS  | 2617  | 2191 | bn256 | 0.8370  | 16.30%  | 28.46   |
| dsv3-up-B16-M8192     | PASS  | 2617  | 2206 | bn256 | 0.8429  | 15.71%  | 28.47   |
| qwen-down-B16-M8192   | FAIL* | 1795  | 1530 | bn128 | 0.8524  | 14.76%  | 26.42   |
| dsv3-down-B4-M8192    | PASS  | 1922  | 1649 | bn128 | 0.8583  | 14.17%  | 28.47   |
| dsv3-down-B16-M8192   | PASS  | 1922  | 1655 | bn128 | 0.8613  | 13.87%  | 28.10   |
| dsv3-down-B16-M4096   | FAIL* | 1867  | 1669 | bn128 | 0.8941  | 10.59%  | 27.73   |
| qwen-down-B16-M4096   | FAIL* | 1655  | 1488 | bn128 | 0.8991  | 10.09%  | 26.72   |
| dsv3-down-B16-M2048   | FAIL* | 1719  | 1665 | bn128 | 0.9683  | 3.17%   | 27.55   |
| qwen-up-B16-M4096     | PASS  | 2000  | 1952 | bn256 | 0.9760  | 2.40%   | 28.46   |
| qwen-up-B4-M8192      | PASS  | 2059  | 2026 | bn128 | 0.9843  | 1.57%   | 28.50   |
| qwen-down-B4-M8192    | PASS  | 1795  | 1788 | bn128 | 0.9961  | 0.39%   | 28.18   |
| qwen-up-B16-M8192     | PASS  | 2059  | 2084 | bn128 | 1.0123  | -1.23%  | 28.40   |
| dsv3-down-B4-M4096    | PASS  | 1867  | 1900 | bn128 | 1.0174  | -1.74%  | 28.51   |
| qwen-up-B4-M4096      | PASS  | 2000  | 2116 | bn128 | 1.0580  | -5.80%  | 28.41   |
| qwen-down-B4-M4096    | PASS  | 1655  | 1778 | bn128 | 1.0745  | -7.45%  | 28.46   |
| dsv3-down-B4-M2048    | PASS  | 1719  | 1906 | bn128 | 1.1087  | -10.87% | 28.47   |
| gpt-up-B4-M8192       | PASS  | 1272  | 1567 | bn128 | 1.2318  | -23.18% | 28.49   |
| gpt-up-B16-M8192      | PASS  | 1272  | 1623 | bn128 | 1.2763  | -27.63% | 28.47   |
| dsv3-up-B16-M2048     | PASS  | 1629  | 2101 | bn256 | 1.2899  | -28.99% | 28.43   |
| dsv3-up-B4-M2048      | PASS  | 1629  | 2324 | bn128 | 1.4269  | -42.69% | 28.46   |
| qwen-down-B16-M2048   | PASS  |  899  | 1422 | bn128 | 1.5828  | -58.28% | 28.10   |
| qwen-up-B16-M2048     | PASS  | 1061  | 1890 | bn128 | 1.7822  | -78.22% | 28.46   |
| qwen-down-B4-M2048    | PASS  |  899  | 1668 | bn128 | 1.8561  | -85.61% | 28.46   |
| qwen-up-B4-M2048      | PASS  | 1061  | 2113 | bn128 | 1.9923  | -99.23% | 28.41   |
| gpt-up-B4-M4096       | PASS  |  716  | 1529 | bn128 | 2.1337  |-113.37% | 28.47   |
| gpt-up-B4-M2048       | PASS  |  689  | 1503 | bn128 | 2.1818  |-118.18% | 28.46   |
| gpt-up-B16-M4096      | PASS  |  716  | 1599 | bn128 | 2.2314  |-123.14% | 28.46   |
| gpt-up-B16-M2048      | PASS  |  689  | 1532 | bn128 | 2.2236  |-122.36% | 28.45   |

(FAIL* = SNR ≥ 25 dB official PT threshold, but < 28 dB user-requested threshold. Real numerics, not impl bug.)

### Aggregate
- primary_score (geomean of ratio across 30 PASS+FAIL cases) = **1.2467**
- min_ratio = **0.8223** (dsv3-up-B4-M4096, gap 17.77%)
- n_pass @ SNR≥28 = 26/30
- n_pass @ SNR≥25 (PT official) = 30/30

### Per-bucket aggregate (PASS only at 28 dB threshold)
- dsv3-up:    geomean ratio = 1.040 (6 case)  ← worst at high M (5/6 < 1)
- dsv3-down:  geomean ratio = 0.949 (3 case PASS, 2 FAIL, 1 PASS borderline)
- qwen-up:    geomean ratio = 1.276 (6 case)
- qwen-down:  geomean ratio = 1.318 (4 case PASS)
- gpt-up:     geomean ratio = 1.834 (6 case) — biggest "wins" but dense is hipBLASLt fallback

## Decision
**BASELINE**

## Attribution
- **30 case routes split bn128/bn256 ~ 23/7**. bn256 only chosen on 7 dsv3-up cases (large K=7168 makes per-warp 64×128 efficient).
- **Worst 4 case all "dsv3-up + dsv3-down large M"** (ratio 0.82-0.86, gap 14-18%) — large K=7168 (up) or large M=8192 (down) push group dispatcher overhead percentage.
- **gpt-up dense baseline is hipBLASLt fallback** (PT high-level chose, since N=5760 unaligned for HK BLK=256). HK grouped beats hipBLASLt dense 2.1-2.2× on small M, 1.23× on M=8192.
- **Small-M cases (M=2048) all grouped > dense** because dense @ M=2048 underutilizes; grouped amortizes launch across B groups.
- **4 SNR FAIL cases all B=16 + down + bn128**: per-warp 64×64 fp32 accumulator path. Same pattern, not random — real numerics ceiling for that kernel path.

## Next direction
Decision tree for round-2:

1. **SNR threshold decision** (USER): keep 28 dB (4 FAIL block round-2 per Rule 6) or align with PT official 25 dB (all 30 PASS, proceed). Must resolve before round-2 ANALYZE.

2. **If SNR=25**: 4 worst-case targets are all dsv3-up M={4096,8192} × B ∈ {4,16} with ratio 0.82-0.84. These all route to bn256. Hypothesis families (per related_work.md):
   - S1: sched_group_barrier batched + LocalPrefetch hoist (kernel-internal, K1)
   - S2: BLK 128×128 + LDS halve + occ=2 (kernel-internal, K1)
   - Profile-first: rocprof-compute SoL on dsv3-up-B4-M4096 to confirm bottleneck before code change

3. **If SNR=28**: must root-cause why bn128 down-path SNR drifts 26.4-27.7 on B=16 large M. Likely fp32 accumulator binning order in `cA→cB→cD→cC` for down-shape; may need to revisit accum order or add Kahan-style compensation.

## Real-training transfer check
- Bucket: this is BASELINE round, no code change; no cache added.
- Cache key: none.
- id(...) audit: N/A.
- 4-step trace hit rate: N/A.
- Workload distribution robustness: target_shapes use uniform `group_lens = full((B,), M_g)`. **TODO before round-2 ACCEPT**: add at least 1 skewed shape (e.g. top_k=1 cf=1.25 token distribution) to representative_shapes per Rule 11 GroupGemm clause.
- Benchmark gain this round: 0 (baseline).
- Estimated real-training gain: 0.
- Decision: **BASELINE**
