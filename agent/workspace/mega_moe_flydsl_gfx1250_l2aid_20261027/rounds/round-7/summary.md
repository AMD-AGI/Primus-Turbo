## Status

`STOPPED` (2026-10-27 16:15): the campaign was stopped for submission before this round's full validation. The
change is not in the tree (kept as `gemm_combine_group_m.patch`, kernel snapshot in `kernel_snapshot/`); the
committed code stays at round 6 (d97bd80c). The data below makes GROUP_M 2 the first candidate for the next round.

## Hypothesis

gemm_combine walks its GEMM tiles row-major (`block_m = idx // N_BLOCKS_N`). One GPU, the pool GEMMs of the EP2
t8192 random step (`tools/gemm_tile_ab.py`, 128 experts, 271 pool blocks): Mega's tile in row-major order is
5.0% slower than in GROUP_M 16 super-tiles for L2 fwd (NT, K=2048: 1.190 vs 1.133 ms) and 3.0% for L1 dgrad (NN,
K=4096: 2.302 vs 2.233 ms). With N_BLOCKS_N = 28 = 4 mod 8, the two pool blocks of one expert land on XCCs 4
apart (the other AID), so each weight block is fetched once per AID; super-tiles put them on neighbouring
workgroups. dispatch_gemm already uses GROUP_M 16.

## Single change

`gemm_combine_kernel.py`: GROUP_M super-tile decode of the GEMM tile index over the live pool blocks (the
dispatch_gemm code), `group_m` constexpr argument, default 16; env `MEGA_GC_GROUP_M` for the A/B only.

## Results

### Validation
- Time: 2026-10-27 15:55 (default GROUP_M 16): EP1 gemm_combine + fused MoE 15 passed, EP2 5 passed. Not run:
  the full `run_round.sh 7 full` with the GROUP_M 2 default.
- Raw artifacts: `artifacts/gemm_tile_ab_{nt,nn}.log`, `artifacts/r7_ep{1,2}.log`, `artifacts/ab_r7*.txt`

### Same-session A/B, EP2 GPUs 2,3 (fwd+bwd ms, mean of 2 reps)

Session 1 (`tools/ab.sh r7 2`):

| Variant | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs acc |
|---|---|---|---|---|---|
| acc (d97bd80c, row-major) | 6.403 | 11.044 | 17.276 | 1285.712 | — |
| GROUP_M 16 | 6.410 | 11.024 | 17.235 | 1287.052 | +0.104% |
| GROUP_M 8 | 6.388 | 10.999 | 17.203 | 1290.290 | +0.356% |
| GROUP_M 4 | 6.367 | 10.970 | 17.208 | 1292.712 | +0.544% |

Session 2 (`tools/ab.sh r7b 2`):

| Variant | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs acc |
|---|---|---|---|---|---|
| acc (d97bd80c, row-major) | 6.430 | 11.037 | 17.268 | 1284.332 | — |
| GROUP_M 4 | 6.390 | 11.005 | 17.196 | 1290.060 | +0.446% |
| GROUP_M 2 | 6.363 | 10.943 | 17.178 | 1294.769 | +0.813% |

GROUP_M 2 is faster than acc on every shape in both reps. Against the EP2 baseline (1260.143 TF, round-6 session)
GROUP_M 2 would be about +2.7%.

## Decision

Not decided (stopped). Next round: GROUP_M 2 as the default, `run_round.sh 7 full`, then accept or roll back.

## Attribution

Standalone the large super-tile wins (GROUP_M 16), fused the small one does. Likely cause (not traced): inside
gemm_combine the combine chunks
wait on pool blocks in pool order, and a GROUP_M super-tile finishes its pool blocks together, so a large one
holds the last pushable rows back until the end of the GEMM. GROUP_M 2 matches the ~2 pool blocks per expert:
the two blocks of an expert's N column run on neighbouring workgroups (same AID) while pool blocks still finish
nearly in order.

## Real-training transfer check
- Bucket: K1 (tile order)
- Cache key (if any): no cache
- Workload distribution robustness: holds under skew (skew1 +0.53% for GROUP_M 2)
- Decision: pending (not accepted)
