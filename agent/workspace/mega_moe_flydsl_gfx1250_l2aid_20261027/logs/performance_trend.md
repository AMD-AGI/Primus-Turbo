| Round | Status | Description | Fwd Avg TFLOPS | Fwd Peak TFLOPS | Bwd Avg TFLOPS | Bwd Peak TFLOPS | Combined Step TFLOPS | vs Baseline | Key Finding |
|-------|--------|-------------|----------------|-----------------|----------------|-----------------|----------------------|-------------|-------------|
| 1 | BASELINE | 8bbd8c89 (async A gather fix), EP4 GPUs 0-3 | 1001.774 | 1362.000 | 1059.866 | 1493.000 | 1039.651 | — | WG bid runs on XCC (bid+7)%8, strictly; row-major gemm_combine makes both AIDs read every weight block |
| 2 | ROLLBACK | gemm_combine AID N-half split (AID_GROUP_M 2 / 4) | — | — | — | — | 1039.734 / 1040.625 (acc 1039.681 same session) | +0.005% / +0.09% | gc2: L2 fwd gemm_combine +12.6%, L1 dgrad -3.9%; stage bench: fused 1.23x gemm_only, 48% of comm hidden |
| 3 | ROLLBACK | dispatch_gemm AID N-half split (always / adaptive) | — | — | — | — | 1034.632 / 1037.375 (acc 1040.287 same session) | -0.54% / -0.28% | slower on every shape; XGMI: one GPU injects ~1.58 TB/s, Mega dispatch at 64 CUs 519 GB/s, combine push 290 GB/s |
| 4 | ACCEPT | dW1 x re-dispatch by SDMA peer copies on a side stream, overlapped with L1 dgrad gemm_combine | 995.977 | 1364.000 | 1081.099 | 1512.000 | 1051.126 (A/B: 1050.843 vs acc 1038.774) | +1.10% (A/B +1.16%) | exposed dispatch_only (304 us) gone; copies contend with the combine push on XGMI (+65 us); +470 MB symm memory |
| 5 | ROLLBACK | gemm_combine fold chunk 32 / 64 rows (4-row push groups per wave) | — | — | — | — | 1046.542 / 1039.384 (acc 1051.490 same session) | -0.47% / -1.15% | slower on every shape, t4096 fwd most; per-row latency is not the combine's bound |
| 6 | ACCEPT | gemm_combine without lead combine workgroups (num_lead_combine 16 -> 0); **EP2, GPUs 2,3** | 1241.340 | 1494.000 | 1309.268 | 1612.000 | 1285.765 (A/B: 1282.964 vs acc 1274.214) | EP2, not comparable with rows 1-5; A/B +0.687% (l4 +0.450%, l8 +0.045%) | lead WGs slow the GEMM by more than their early push saves; fewer leads monotonically better |

From round 6 (2026-10-27 15:30) GPUs 0,1 belong to another user: rounds run EP2 on GPUs 2,3 and the primary
metric is the Combined Step TFLOPS geomean over the same 3 DSV3 shapes at EP2. The EP2 reference of the round-4
tree (3ef91841) is 1274.214 TF (round-6 A/B, acc).
