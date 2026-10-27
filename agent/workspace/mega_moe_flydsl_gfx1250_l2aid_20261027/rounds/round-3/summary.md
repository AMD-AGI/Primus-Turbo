## Hypothesis

`dispatch_gemm` (L1 fwd NT, L2 dgrad NN) decodes 16-row super-tiles m fastest, so the AID of a tile is set by
`m_local % 8`: each AID reads half the pool rows and every weight block. Splitting each super-tile by N halves
(one per AID) makes each AID read all of the super-tile's rows and half the weight: fewer GL2 fills when the pool
has fewer blocks than this rank's weights (`tools/aid_model.py`: L1 fwd at T=8192 random 1302 -> 1062 blocks),
more when it has more (skew rank 0). Mode 2 picks per launch on the device: split iff
`real_tiles < experts_per_rank * N_BLOCKS_N`.

## Single change

`dispatch_gemm_kernel.py`: constexpr `AID_SPLIT` (env `MEGA_AID_SPLIT_DG`): 0 shipped order, 1 always split,
2 adaptive (the candidate); uses `gemm_nt_tile.aid_half_index`.

## Results

### Validation
- Time: 2026-10-27 06:52
- Validation level: quick (EP1 dispatch_gemm + fused tests per mode: 15 passed each; the A/B's SNR check, all PASS)
- Raw artifacts: `artifacts/test_ep1_dg1.log`, `artifacts/test_ep1_dg2.log`, `artifacts/ab_r3.txt`,
  `artifacts/ab_r3_summary.txt`, `artifacts/xgmi_sdma1.log`, `artifacts/xgmi_sdma0.log`

### Same-session A/B (`tools/ab.sh r3 2`, fwd+bwd ms, mean of 2 reps)

| Variant | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs acc |
|---|---|---|---|---|---|
| acc (8bbd8c89) | 6.857 | 11.976 | 28.088 | 1040.287 | — |
| dg1 always | 6.896 (-0.57%) | 12.022 (-0.38%) | 28.279 (-0.68%) | 1034.632 | -0.54% |
| dg2 adaptive | 6.899 (-0.62%) | 11.992 (-0.14%) | 28.111 (-0.08%) | 1037.375 | -0.28% |

Both variants are slower on every shape. dg2 at skew1 runs the shipped order on rank 0 (624 > 512 blocks) and the
split on the cold ranks; at t4096 (A 138 < B 512 / 256 blocks) it splits both launches and loses 0.62%.

### XGMI ceiling (`tools/xgmi_peak.py`, measured between rounds)

| Pattern | Per-GPU egress |
|---|---|
| one GPU -> 1 peer (SDMA) | 1558 GB/s |
| one GPU -> 3 peers at once (SDMA) | 1581 GB/s (527 per link) |
| 3 peers -> one GPU (SDMA) | ingress 1654 GB/s |
| 0 <-> 1 both ways (SDMA) | 1511 each way (full duplex) |
| all 4 GPUs all-to-all, SDMA copies | 793 GB/s |
| all-to-all, RCCL all_to_all_single 320 / 1024 MiB per rank | 657 / 755 GB/s |
| all-to-all, Mega dispatch_only, 255 CUs, 320 MB (t8192 random) | 1116-1129 GB/s |

One GPU's injection (~1.58 TB/s) is the cap whether it uses one link or three. Against it, the Mega comm legs at
T=8192 random (320 MB per rank per transfer): dispatch at the fused kernel's 64 CUs 519 GB/s (33%), the same at
255 CUs 1129 GB/s (71%), the combine push baseline 290 GB/s (18%).

## Decision

`ROLLBACK`. `dispatch_gemm_kernel.py` and `gemm_nt_tile.py` restored to 8bbd8c89.

## Attribution

As in round 2: the GEMMs alone are at the dense roofline, so fewer GL2 fills buy nothing, and the N-half order
costs something — each AID's workgroups now wait on the same super-tile rows (all of them) and its half of the
output columns land on one AID's HBM write path. No GL2 placement round is left with a positive model margin
large enough to beat the ~0.3% A/B noise.

## Next direction

The overlap (user decision after round 2): the comm legs run at 18-33% of the GPU's XGMI injection inside the
fused kernels' CU budget, and the fusion hides 48% of the comm. Start with the combine push (290 GB/s, weakest
leg, L2 fwd fused/gemm_only 149%).

## Real-training transfer check
- Bucket: c (no change kept)
- Decision: ROLLBACK
- Skew: dg1 -0.68%, dg2 -0.08% at skew1
