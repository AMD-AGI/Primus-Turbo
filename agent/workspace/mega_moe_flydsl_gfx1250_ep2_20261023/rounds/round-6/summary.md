## Hypothesis

The gemm_combine grid holds the worst-case pool's GEMM tiles (`MAX_TILES = 544` at EP2 T=8192, from
`num_max_pool_tokens`) while a call fills ~273, so ~7600 masked GEMM workgroups (more than the live ones)
sit in launch order between the live tiles and the tail combine workgroups. A per-workgroup timeline
(`/home/xiaompen/mega_scratch/timeline_combine_ep.py` on a stamped copy of the kernel, L2 fwd at EP2):
the last live GEMM tile ends at ~1020 us but the tail combine workgroups start at 1155-1510 us. The prologue
already syncs the pool row count to the host (`rows = num_tile_blocks * BLOCK_M`), so the grid can hold just
this call's tiles.

## Single change

Kernel launch: `gemm_combine_nt_gfx1250` passes the grid size as a runtime `fx.Int32`
(`rows // BLOCK_M * n_blocks + num_lead + num_tail`); the kernel's tail combine start follows the device
tile count (`num_tiles_m * N_BLOCKS_N + NUM_LEAD_COMBINE`) instead of the static worst case. No recompile per
tile count (the grid is not a constexpr).

## Results

### Validation
- Time: 2026-10-23 06:15 (UTC)
- Validation level: full (EP2 pytest 5 passed, EP1 pytest 19 passed, quick bench all PASS), bench repeated
- Raw artifacts: `artifacts/test_ep.log`, `artifacts/test_ep1.log`, `artifacts/quick_validation.log`,
  `artifacts/bench.csv`, `artifacts/quick_rep2.log`, `artifacts/bench_rep2.csv`

| Shape | R5 fwd / fwd+bwd ms | R6 fwd / fwd+bwd ms (run 1; rep 2) | R5 step TF | R6 step TF (run 1; rep 2) | Delta (run 1) |
|---|---|---|---|---|---|
| dsv3_t8192_random | 4.397 / 11.997 | 4.689 / 11.558; 4.143 / 11.579 | 1443.4 | 1498.3; 1495.6 | +3.80% |
| dsv3_t8192_skew1 | 6.413 / 17.970 | 6.328 / 17.740; 6.300 / 17.711 | 963.7 | 976.2; 977.8 | +1.30% |
| dsv3_t4096_random | 2.314 / 6.911 | 2.212 / 6.686; 2.206 / 6.727 | 1252.8 | 1295.1; 1287.1 | +3.37% |
| Aggregate (geomean) | | | 1203.385 | 1237.297; 1234.672 | +2.818% |

Run 1's random forward (4.689 ms) is a one-off: rep 2 measures 4.143 ms, below round-5. Min SNR 48.5 dB.

## Decision

`ACCEPTED` — aggregate +2.818% (rep 2: +2.60%), every shape improves, no regression.

## Attribution

The gemm_combine launches (L2 forward and L1 dgrad) no longer queue ~7600 masked tiles ahead of the tail
combine: fwd+bwd -0.44 / -0.23 / -0.23 ms. The skewed shape gains least: its hot rank fills more of the
worst-case pool, so it had fewer masked tiles.

## Real-training transfer check
- Bucket: K2 (launch geometry)
- Production path: both gemm_combine calls of every MoE layer (stage2_forward, stage1_backward) use the
  prologue's pool rows, which training always has.
- Skewed routing: dsv3_t8192_skew1 +1.30%
- Decision: ACCEPTED

## Next direction

The fused dispatch+GEMM launch has the same worst-case grid (`NUM_DISPATCH_CU + MAX_TILES * N_BLOCKS_N`): its
masked tiles drain after the live ones and extend both the L1 forward and the L2 dgrad.
