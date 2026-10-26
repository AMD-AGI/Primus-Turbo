## Hypothesis

With 8 waves per workgroup (round-2) a 16-row combine chunk leaves 2 fold rows per wave, below
`combine_bf16_tile`'s 4-row unroll, so each fold row is pushed alone (its own wait + flag). A chunk sized to
4 rows per wave (32) should restore the unroll and raise the combine rate.

## Single change

Config only: `_FOLD_CHUNK_ROWS` in `bf16_gfx1250/gemm_combine_kernel.py` 16 -> 32 (and 64 as the second sweep point).

## Results

### Validation
- Time: 2026-10-23 05:13 (UTC)
- Validation level: quick (per-shape SNR + timing)
- Raw artifacts: `artifacts/quick_chunk32.log`, `artifacts/quick_chunk64.log`

| Shape | R2 step TF | chunk 32 step TF | chunk 64 step TF |
|---|---|---|---|
| dsv3_t8192_random | 1406.2 | 1409.3 (+0.22%) | 1406.2 (0.00%) |
| dsv3_t8192_skew1 | 946.7 | 946.0 (-0.07%) | 952.4 (+0.60%) |
| dsv3_t4096_random | 1230.5 | 1222.0 (-0.69%) | 1219.2 (-0.92%) |
| Aggregate | 1178.823 | 1176.666 (-0.183%) | 1177.542 (-0.109%) |

All PASS (min SNR 48.5 dB).

## Decision

`ROLLED BACK` — no aggregate gain; within noise. Code restored to round-2 (`git checkout` of gemm_combine_kernel.py).

## Attribution

Fold rows per wave is not what bounds the combine; the per-row unroll does not change the time.

## Rollback Analysis
- Change made: combine chunk 16 -> 32 / 64 fold rows
- Expected effect: 4-row unroll per wave, fewer per-row waits
- Actual effect: +-0.2% (noise)
- Root cause: the combine/dispatch roles are bounded by the remote-store path, not per-wave issue. A follow-up
  probe (`/home/xiaompen/mega_scratch/probe_cpol_*.log`) shows the dispatch kernel's remote bandwidth is the same
  for every store cache policy (238-240 GB/s at 64 CUs) and grows sublinearly with CUs (364 GB/s at 192), and
  `dispatch_bf16_tile` waits for every store to retire plus a workgroup barrier and a flag after each of the 32
  experts, with only ~2 rows per copy warp per expert: a serialized remote round trip per expert.
- Lesson learned: attack the per-expert publish serialization in the dispatch (and the per-row publish in the combine).

## Next direction

Publish dispatch flags once per group of experts (one wait + barrier per group) on gfx1250.

## Real-training transfer check
- Bucket: K1 (config)
- Decision: REJECTED (no gain)
