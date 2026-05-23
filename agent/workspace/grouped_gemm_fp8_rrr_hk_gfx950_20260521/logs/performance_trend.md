# Performance Trend — grouped_gemm_fp8_rrr HK on gfx950 (MI355X)

Append-only. One row per completed round. `primary_score` = geomean(grouped_TFLOPS / dense_TFLOPS) across PASS shapes; `min_ratio` = min per-shape ratio (hard gate, any -5% step triggers rollback).

| Round | Status | Description | Fwd Avg TFLOPS (grouped) | Fwd Peak TFLOPS | dense_avg TFLOPS | primary_score | min_ratio | vs Baseline | Key Finding |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BASELINE | HK b7655e5a + PT 42e63483 (H3/H6/H7/H17 baked in) | 1786 | 2324 | 1641 | 1.2467 | 0.8223 | — | 30 case; 4 SNR border (B=16+down+bn128 path); 7 worst all dsv3-up/down @ large M |
