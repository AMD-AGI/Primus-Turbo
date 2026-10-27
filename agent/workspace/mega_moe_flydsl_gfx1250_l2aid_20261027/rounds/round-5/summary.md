## Hypothesis

The combine stages carry the largest fused - gemm_only gap (stage bench after round 4, t8192 random: L2 fwd 0.452
ms, L1 dgrad 0.487 ms; bench_dsv3 L2 fwd 0.725 ms). A 16-row fold chunk gives each of the 8 waves two rows, below
`_ROW_UNROLL = 4`, so every row pays its own three-level index chain and its own remote store-retire wait before
its flag. 32 (or 64) rows per chunk make each wave push one (two) 4-row groups: one index chain and one retire wait
per 4 rows, a higher per-workgroup push rate, more pushed by the 16 lead workgroups while the GEMM runs.

## Single change

`gemm_combine_kernel.py`: `_FOLD_CHUNK_ROWS` from env `MEGA_FOLD_CHUNK_ROWS` (16 shipped, 32, 64); the A/B ran
each variant in its own FlyDSL cache.

## Results

### Validation
- Time: 2026-10-27 15:45
- Validation level: quick (the A/B's SNR check, all PASS)
- Raw artifacts: `artifacts/ab_r5.txt`, `artifacts/ab_r5_summary.txt`; stage bench of the round-4 tree:
  `../round-4/artifacts/stage_bench_acc/`

### Same-session A/B (`tools/ab.sh r5 2`, fwd+bwd ms, mean of 2 reps)

| Variant | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs acc |
|---|---|---|---|---|---|
| acc (3ef91841, 16 rows) | 6.748 | 11.860 | 27.905 | 1051.490 | — |
| c32 | 6.796 (-0.71%) | 11.884 (-0.20%) | 28.047 (-0.50%) | 1046.542 | -0.47% |
| c64 | 6.921 (-2.49%) | 11.906 (-0.39%) | 28.065 (-0.57%) | 1039.384 | -1.15% |

Slower on every shape, monotonically with the chunk size; the t4096 forward (the L2 fwd combine) loses most
(2.30 -> 2.43 ms at c64).

## Decision

`ROLLBACK`. `gemm_combine_kernel.py` restored to 3ef91841.

## Attribution

The per-row latency is not what bounds the combine: bigger chunks wait for a longer pool prefix before the first
push and leave fewer, coarser work items for the tail's 256 workgroups (t4096 has half the fold rows, so it loses
most). Before the next combine change the kernel needs a timeline — when GEMM tiles finish, when each chunk is
ready and pushed, how long the reduce runs — instead of a model.

## Next direction

Instrument `gemm_combine` with realtime-clock stamps (debug constexpr, not shipped) and attribute the L2 fwd / L1
dgrad gap: GEMM slowdown vs push during the GEMM vs tail push vs reduce.

## Real-training transfer check
- Bucket: c (no change kept)
- Decision: ROLLBACK
- Skew: c32 -0.50%, c64 -0.57% at skew1
