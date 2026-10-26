## Hypothesis

The fused dispatch+GEMM launch has the same worst-case grid as round-6's gemm_combine
(`NUM_DISPATCH_CU + MAX_TILES * N_BLOCKS_N`, 544 tiles at EP2 T=8192 vs ~273 filled): its ~4300 (L1, N=4096)
/ ~2200 (L2 dgrad, N=2048) masked tiles drain after the live ones and extend the launch.

## Single change

Kernel launch: `dispatch_gemm_nt_gfx1250` / `dispatch_only_gfx1250` pass the grid as a runtime `fx.Int32`:
the comm workgroups plus this call's tiles (`pool rows // BLOCK_M * n_blocks`), or the one dispatch-only waiter.
The now unused `MAX_TILES` constexpr is dropped.

## Results

### Validation
- Time: 2026-10-23 06:25 (UTC)
- Validation level: full (EP2 pytest 5 passed, EP1 pytest 19 passed, quick bench all PASS), bench repeated
- Raw artifacts: `artifacts/test_ep.log`, `artifacts/test_ep1.log`, `artifacts/quick_validation.log`,
  `artifacts/bench.csv`, `artifacts/quick_rep2.log`

| Shape | R6 step TF (run 1; rep 2) | R7 step TF (run 1; rep 2) | R6 fwd ms (rep 2) | R7 fwd ms (run 1; rep 2) | Delta (run 1 vs run 1) |
|---|---|---|---|---|---|
| dsv3_t8192_random | 1498.3; 1495.6 | 1509.5; 1509.8 | 4.143 | 4.032; 4.042 | +0.75% |
| dsv3_t8192_skew1 | 976.2; 977.8 | 979.3; 978.7 | 6.300 | 6.280; 6.315 | +0.32% |
| dsv3_t4096_random | 1295.1; 1287.1 | 1289.8; 1287.7 | 2.206 | 2.193; 2.191 | -0.41% |
| Aggregate (geomean) | 1237.297; 1234.672 | 1239.985; 1239.146 | | | +0.217% (mean of reps +0.29%) |

Min SNR 48.5 dB.

## Decision

`ACCEPTED` — small (+0.22%, both reps above both round-6 reps), no shape regresses beyond noise; the forward
is consistently faster on the T=8192 random shape (-0.1 ms).

## Attribution

Masked tiles at the end of a launch cost little (~0.1 ms on the L1 forward): unlike round-6, no role waits
behind them. Most of round-6's gain was the tail combine's start, not the masked tiles' own run time.

## Real-training transfer check
- Bucket: K2 (launch geometry)
- Production path: stage1_forward (dispatch + L1), stage2_backward (dy dispatch + L2 dgrad), stage1_backward (dispatch-only)
- Skewed routing: dsv3_t8192_skew1 +0.32%
- Decision: ACCEPTED

## Next direction

Re-time the gemm_combine at EP2 (round-6 grid) to see what bounds the combine tail now.
