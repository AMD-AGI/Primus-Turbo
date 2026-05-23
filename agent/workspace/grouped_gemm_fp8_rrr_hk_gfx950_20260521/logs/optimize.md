# grouped_gemm_fp8_rrr HK Optimization Log

## Basic Information
- Target operator: grouped_gemm_fp8_rrr (FP8 e4m3 tensorwise, A row-major B row-major C row-major)
- Implementation language: HIP
- Backend: HK (HipKittens `hk_grouped_rrr_fp8`)
- Target GPU: gfx950 (MI355X)
- Campaign: agent/workspace/grouped_gemm_fp8_rrr_hk_gfx950_20260521
- Start time: 2026-05-21 06:38
- Current status: PREPARE_ENVIRONMENT complete; awaiting BASELINE (round-1)

## Baseline
- Time: 2026-05-21 07:02
- Backend: HK (HipKittens `hk_grouped_rrr_fp8`)
- GPU: chi2811 MI355X, HIP_VISIBLE_DEVICES=4
- Commit: HK `b7655e5a` + PT `42e63483` (last HK bump `720eb9fc`)
- Validation level: full (30 target shapes)
- primary_score (geomean ratio) = **1.2467**
- min_ratio = **0.8223** (dsv3-up-B4-M4096)
- n_pass: 30/30 @ SNR≥25 (PT official) | 26/30 @ SNR≥28 (user-requested)
- 4 SNR border cases (26.4-27.7 dB) all on B=16+down+bn128 path — real numerics, not bug
- 7 of 30 worst (ratio <0.95) all dsv3-up/down @ large M → primary optimization target
- 14 of 30 grouped > dense (ratio >1.0) — gpt-up biggest (2.2×, dense=hipBLASLt fallback)
- Detailed data: rounds/round-1/summary.md
- Raw CSV: rounds/round-1/artifacts/benchmark.csv

## Optimization History
(append rounds here in order)

## Current Best
(filled after baseline)

## Directions to Try
- [ ] Profile baseline w/ rocprof-compute SoL on all 6 shapes
- [ ] S1: sched_group_barrier batched + LocalPrefetch hoist (only after SoL says MFMA-vmem stall)
- [ ] S2: BLK 128×128 + LDS halve + occ=2 (only after SoL says occupancy-bound)
- [ ] S3: single acc resident main loop (combined w/ S2)
- [ ] Chunked persistent + cross-group B reuse (algorithmic, 800-1500 LOC; only if S1/S2/S3 net 0)

## Verified Ineffective Directions
| Direction | Round | Failure Reason | Memory |
|---|---|---|---|
| H1 hoist B-load | pre-campaign | compiler already lowered | [[feedback_fp8_rrr_attempt_h1]] |
| H2 readfirstlane pin | pre-campaign | compiler uniformity analysis covers | [[feedback_fp8_rrr_attempt_h2]] |
| H4 H3-on-bn128 | pre-campaign | +6 VGPR no spill benefit | [[feedback_fp8_rrr_attempt_h4_bn128]] |
| H5 view-drop diagnostic | pre-campaign | structural spill, not in lambdas | [[feedback_fp8_rrr_attempt_h5_diag]] |
| H8 launch_bounds + 2× grid | pre-campaign | LDS > CU limit, HW still 1 wave/CU | [[feedback_fp8_rrr_attempt_h8]] |
| H9 11-条 sched_group_barrier | pre-campaign | s_barrier blocks compiler reorder | [[feedback_fp8_rrr_attempt_h9]] |
| H10/H11/H12 chunk_size + GRID_MUL sweep | pre-campaign | physics-bound at B=16 | [[feedback_fp8_rrr_attempt_h10]] [[feedback_fp8_rrr_attempt_h11_h12]] |
| H18 b1 mirror in main loop | pre-campaign | net-loss on large M_g, +VGPR pressure | [[feedback_fp8_rrr_h17_clean_baseline]] |
| 32×32 wrapper swap alone | pre-campaign | per-warp area not the lever | [[feedback_fp8_rrr_32x32_flawed_premise]] |
| BN=128 forced for worst shape | pre-campaign | gap 23.85% → 23.56% (no diff) | [[feedback_bn128_per_warp_area_not_lever]] |

## Final Report
(filled when campaign terminates)
