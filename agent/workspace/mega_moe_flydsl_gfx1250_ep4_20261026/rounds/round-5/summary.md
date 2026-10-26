## Hypothesis

Round 2 left the post-GEMM push backlog in gemm_combine as the largest non-GEMM tail (GEMM done ~1020 us, tail
push ~1290 us median at T=8192). If each pushing wave keeps more fold rows in flight, the tail shrinks. Before
touching code, screen the push constants of `ep_intranode.py` that set rows per wave and loads in flight.

Two corrections to earlier rounds came out of the analysis for this round:
- Round 4's "the step is power-capped, so idle removal is absorbed by clocks" does not hold. A host delay of
  100 / 200 / 400 us inserted after the prologue lengthens the step by ~157 / 295 / 550 us, so idle is on the
  critical path and is not absorbed.
- Round 2's attribution ("fold rows become pushable too late") is incomplete. Per-chunk stamps show one remote
  chunk push takes 32 us in the lead phase and 35 us in the tail, so the push is limited by each WG's own rate
  (~7 GB/s). On gfx1250 each wave gets only 2 rows, so the `_ROW_UNROLL=4` grouped path is never taken and every
  row waits for its stores to retire on its own.

## Single change

Monkeypatched constants (no source edit), one at a time: `_ROW_UNROLL=2`, `FOLD_CHUNK_ROWS=32`,
`_MAX_LOADS_IN_FLIGHT=128`, each with its own FlyDSL cache dir.

## Results

### Validation
- Time: 2026-10-26 03:49 (UTC)
- Validation level: quick screen (dsv3_t8192_random, 2 alternating reps per config) + per-chunk timeline
- Raw artifacts: `/home/xiaompen/mega_scratch/ep4c/logs/push_sweep.txt`, `push_r{1,2}_dsv3_t8192_random.log`,
  `tl5_tl_*.log`

All 8 runs PASS (bench SNR check).

| config | fwd+bwd ms (rep 1 / rep 2) | step TF (rep 1 / rep 2) |
|---|---|---|
| default | 11.568 / 11.535 | 1497.0 / 1501.3 |
| `ROW_UNROLL=2` | 11.558 / 11.577 | 1498.3 / 1495.8 |
| `FOLD_CHUNK_ROWS=32` | 11.585 / 11.580 | 1494.9 / 1495.4 |
| `MAX_LIF=128` | 11.543 / 11.525 | 1500.2 / 1502.6 |

The timeline with `ROW_UNROLL=2` and `MAX_LIF=128` shows the same per-chunk push time (remote ~32-35 us).

### Aggregate
- Not run on the full set: every config is within ±0.2% of the default on the screen shape.

## Decision

`REJECTED`: the constants stay at their defaults (no code change to roll back).

## Attribution

The constants do not change how many rows a wave has in flight on gfx1250 (2 rows per wave), so the per-row
store-retire wait still serializes the push. A real gain needs a code change in the push loop: batch the
retire wait over more rows per wave, or give the tail push more waves.

## Rollback Analysis
- Change made: push constants ROW_UNROLL / FOLD_CHUNK_ROWS / MAX_LIF
- Expected effect: shorter post-GEMM push tail in gemm_combine
- Actual effect: ±0.2% (noise); per-chunk push time unchanged
- Root cause: with 2 rows per wave the knobs never reach the code path they control
- Lesson learned: check which code path a constant actually reaches at the target shape before screening it

## Next direction

1. Restructure the push loop so a wave keeps several rows' stores in flight before one retire wait.
2. The campaign was paused after this round: an intermittent data corruption in the fused dispatch_gemm
   (gfx1250 TDM gather) was found during the e2e check. See `bug_tdm_gather_corruption.md`.

## Real-training transfer check
- Bucket: K1 (config)
- Decision: REJECTED (no gain)
