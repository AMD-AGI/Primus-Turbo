## Hypothesis

The fused kernels' data-movement roles (dispatch push, combine fold + push, reduce) run inside the GEMM's
launch, which holds one workgroup per CU (LDS), so each CU moves data with only the workgroup's 4 waves.
Each wave is latency-bound (EP1 dispatch: ~5 GB/s per wave = one 14 KB row in flight), so doubling the
waves per workgroup should roughly double those phases' throughput while the GEMM keeps its rate.
EP1 probe (`/home/xiaompen/mega_scratch/probe_waves*.log`): 4x2 waves vs 2x2 -> L2+combine 1.480 -> 1.341 ms,
L1 dgrad+combine 2.530 -> 2.352, dy dispatch+L2 dgrad 1.078 -> 1.026, dispatch+L1 2.036 -> 2.019, bitwise-equal
outputs; 4x4 spills (6.9 ms).

## Single change

Config only: default `m_warp, n_warp` of the gfx1250 fused kernels 2x2 -> 4x2 (8 wave32 waves per workgroup)
in `dispatch_gemm_nt_gfx1250`, `dispatch_only_gfx1250` and `gemm_combine_nt_gfx1250`.

## Results

### Validation
- Time: 2026-10-23 05:09 (UTC)
- Validation level: full
- Tests: EP2 5 passed, EP1 19 passed
- Raw artifacts: `artifacts/`

| Shape | Check (min SNR dB) | Fwd ms | Fwd+bwd ms | Fwd TF | Bwd TF | Step TF | Step vs R1 |
|---|---|---|---|---|---|---|---|
| dsv3_t8192_random | PASS (48.5) | 4.435 | 12.315 | 1302 | 1465 | 1406.2 | +4.91% |
| dsv3_t8192_skew1 | PASS (48.8) | 6.397 | 18.292 | 902 | 971 | 946.7 | +3.82% |
| dsv3_t4096_random | PASS (48.5) | 2.306 | 7.036 | 1252 | 1220 | 1230.5 | +2.18% |

### Aggregate
- Combined Step TFLOPS (geomean): 1178.823 (R1 1137.514, +3.632%)
- Fwd aggregate 1137.045 (+4.321%), Bwd aggregate 1201.630 (+3.299%)

## Decision

`ACCEPTED` — aggregate up, every shape up.

## Attribution

More waves per CU in the comm/combine/reduce roles; the 8-wave GEMM tile (64x128 per wave) keeps the
2x2 tile's rate, so the gain is the data-movement phases' latency hiding.

## Next direction

The combine chunk is still 16 fold rows (2 per wave now, below the 4-row unroll of `combine_bf16_tile`):
size the chunk to the workgroup (4 rows per wave). Then the x re-dispatch for dW1 and the pool gathers.

## Real-training transfer check
- Bucket: K1 (launch config)
- Cache key: no cache
- id(...) audit: n/a
- Workload distribution robustness: holds under skew (skew1 +3.82%)
- Benchmark gain on this round: +3.632% combined-step
- Estimated real-training gain: +3.632% of the MoE layer's fwd+bwd (kernel config, transfers 1:1)
- Decision: ACCEPT-as-real
