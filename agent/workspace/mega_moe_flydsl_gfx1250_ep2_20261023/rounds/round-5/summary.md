## Hypothesis

After round-4 the dispatch-only launch (x re-dispatch for dW1) is bounded by its data movement from 64 comm
CUs (~390 us for 117 MB remote at EP2 T=8192). The launch holds only comm workgroups (plus one waiter), so it can
take the whole GPU: remote push rate grows with comm CUs (sublinearly).

## Single change

Config: `dispatch_only_gfx1250`'s default `num_dispatch_cu` 64 -> every CU but the waiter's
(`multi_processor_count - 1`, 255 on MI455X).

Probe (`artifacts/probe_dispatch_cu.log`, dispatch-only + pool gather, EP2 T=8192 H=7168 32 experts/rank topk 8):

| comm CUs | 64 | 96 | 128 | 160 | 192 | 224 | 254 |
|---|---|---|---|---|---|---|---|
| us | 602 / 600 | 504 | 469 / 470 | 441 | 425 / 424 | 416 | 413 / 413 |

All received rows exact.

## Results

### Validation
- Time: 2026-10-23 05:50 (UTC)
- Validation level: full (EP2 pytest 5 passed, EP1 pytest 19 passed, quick bench all PASS)
- Raw artifacts: `artifacts/test_ep.log`, `artifacts/test_ep1.log`, `artifacts/quick_validation.log`, `artifacts/bench.csv`

| Shape | R4 fwd / fwd+bwd ms | R5 fwd / fwd+bwd ms | R4 step TF | R5 step TF | Delta |
|---|---|---|---|---|---|
| dsv3_t8192_random | 4.364 / 12.106 | 4.397 / 11.997 | 1430.4 | 1443.4 | +0.91% |
| dsv3_t8192_skew1 | 6.393 / 18.001 | 6.413 / 17.970 | 962.0 | 963.7 | +0.18% |
| dsv3_t4096_random | 2.317 / 6.994 | 2.314 / 6.911 | 1238.1 | 1252.8 | +1.19% |
| Aggregate (geomean) | | | 1194.345 | 1203.385 | +0.757% |

Min SNR 48.5 dB, unchanged.

## Decision

`ACCEPTED` — aggregate +0.757%, every shape improves (skew1 within noise), no regression.

## Attribution

The backward's x re-dispatch: -110 / -30 / -80 us of fwd+bwd; the fwd differences (+-0.03 ms) are noise (the
forward does not run dispatch-only). Skew1 gains least: its re-dispatch is bounded by the hot rank's receive side.

## Real-training transfer check
- Bucket: K1 (config)
- Production path: `fused_mega_moe_gfx1250_impl.stage1_backward` -> `dispatch_only_gfx1250(saved_x, handle)` with the default
- Skewed routing: dsv3_t8192_skew1 +0.18% (no regression)
- Decision: ACCEPTED

## Next direction

The x re-dispatch (~0.2 ms of push + ~0.2 ms gather) and the dy pool gather are still serial in the backward:
remove the torch pool gathers (row-index gather inside the variable-K GEMM) or overlap the re-dispatch with dW2.
Largest remaining EP2 cost is the combine (L2 fwd ~+0.77 ms, L1 dgrad ~+0.64 ms over the GEMM alone).
