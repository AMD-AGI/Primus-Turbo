| Round | Status | Description | Fwd Avg TFLOPS | Fwd Peak TFLOPS | Bwd Avg TFLOPS | Bwd Peak TFLOPS | Combined Step TFLOPS | vs Baseline | Key Finding |
|-------|--------|-------------|----------------|-----------------|----------------|-----------------|----------------------|-------------|-------------|
| 1 | BASELINE | 8bbd8c89 (async A gather fix), EP4 GPUs 0-3 | 1001.774 | 1362.000 | 1059.866 | 1493.000 | 1039.651 | — | WG bid runs on XCC (bid+7)%8, strictly; row-major gemm_combine makes both AIDs read every weight block |
| 2 | ROLLBACK | gemm_combine AID N-half split (AID_GROUP_M 2 / 4) | — | — | — | — | 1039.734 / 1040.625 (acc 1039.681 same session) | +0.005% / +0.09% | gc2: L2 fwd gemm_combine +12.6%, L1 dgrad -3.9%; stage bench: fused 1.23x gemm_only, 48% of comm hidden |
| 3 | ROLLBACK | dispatch_gemm AID N-half split (always / adaptive) | — | — | — | — | 1034.632 / 1037.375 (acc 1040.287 same session) | -0.54% / -0.28% | slower on every shape; XGMI: one GPU injects ~1.58 TB/s, Mega dispatch at 64 CUs 519 GB/s, combine push 290 GB/s |
| 4 | ACCEPT | dW1 x re-dispatch by SDMA peer copies on a side stream, overlapped with L1 dgrad gemm_combine | 995.977 | 1364.000 | 1081.099 | 1512.000 | 1051.126 (A/B: 1050.843 vs acc 1038.774) | +1.10% (A/B +1.16%) | exposed dispatch_only (304 us) gone; copies contend with the combine push on XGMI (+65 us); +470 MB symm memory |
| 5 | ROLLBACK | gemm_combine fold chunk 32 / 64 rows (4-row push groups per wave) | — | — | — | — | 1046.542 / 1039.384 (acc 1051.490 same session) | -0.47% / -1.15% | slower on every shape, t4096 fwd most; per-row latency is not the combine's bound |
| 6 | ACCEPT | gemm_combine without lead combine workgroups (num_lead_combine 16 -> 0); **EP2, GPUs 2,3** | 1241.340 | 1494.000 | 1309.268 | 1612.000 | 1285.765 (A/B: 1282.964 vs acc 1274.214) | EP2, not comparable with rows 1-5; A/B +0.687% (l4 +0.450%, l8 +0.045%) | lead WGs slow the GEMM by more than their early push saves; fewer leads monotonically better |
| 1-EP2 | BASELINE (EP2) | 8bbd8c89 re-measured at EP2 on GPUs 2,3 (same-session A/B with 4-EP2 and 6-EP2) | 1231.511 | 1486.500 | 1275.220 | 1564.500 | 1260.143 | — | EP2 baseline for round 6 on |
| 4-EP2 | ACCEPT (re-measured) | round-4 tree 3ef91841 at EP2 | 1230.670 | 1481.500 | 1294.141 | 1586.500 | 1272.241 | step +0.960%, fwd -0.068%, bwd +1.484% | SDMA x re-dispatch holds at EP2 (t4096 +1.48%, t8192 +0.82%, skew1 +0.58%) |
| 6-EP2 | ACCEPT (re-measured) | round-6 tree d97bd80c at EP2 | 1242.294 | 1490.000 | 1303.622 | 1600.000 | 1282.641 | step +1.785%, fwd +0.876%, bwd +2.227% | faster than base on every shape (t4096 +2.04%, t8192 +1.57%, skew1 +1.75%) |
| 7 | STOPPED | gemm_combine GROUP_M super-tile order (16 / 8 / 4 / 2) vs row-major; not validated, not in the tree | — | — | — | — | — | 1294.769 for GROUP_M 2 (acc 1284.332 same session) | A/B vs round 6: GROUP_M 16 +0.104%, 8 +0.356%, 4 +0.544% / +0.446%, 2 +0.813% | smaller super-tiles better inside the fused kernel; stopped for submission, GROUP_M 2 is the next candidate |

From round 6 (2026-10-27 15:30) GPUs 0,1 belong to another user: rounds run EP2 on GPUs 2,3 and the primary
metric is the Combined Step TFLOPS geomean over the same 3 DSV3 shapes at EP2. The EP2 reference of the round-4
tree (3ef91841) is 1274.214 TF (round-6 A/B, acc).

EP2 baseline (2026-10-27 16:00): from round 6 on, `vs Baseline` is against row 1-EP2 (8bbd8c89 at EP2, 1260.143 TF,
same-session A/B `ab_ep2base`, artifacts in `rounds/round-6/artifacts/ab_ep2base*.txt`). Row 6's `vs Baseline` cell
compares with the round-4 tree only; the EP2 cumulative number for round 6 is row 6-EP2 (+1.785%).

Campaign stopped 2026-10-27 16:15 for submission: committed code = round 6 (d97bd80c), EP2 +1.785% vs the EP2
baseline; round 7 (GROUP_M 2, +0.813% A/B) is left as a patch in `rounds/round-7/`.
