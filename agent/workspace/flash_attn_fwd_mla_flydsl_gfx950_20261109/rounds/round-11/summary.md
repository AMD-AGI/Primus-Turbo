# Round 11 — ROLLED BACK

- Time: 2026-11-09 09:40 (UTC+8)
- Hypothesis: round 10 showed per-SIMD priority asymmetry is not neutral but did not separate
  the direction from the cost of the scalar branch. Same branch structure, opposite direction:
  P*V `s_setprio(1)` only for the first-dispatched half (wave_id < 4).
- Change (single variable vs round 7): loop C3/C7 `s_setprio(1)` only for
  `wave_id < NUM_WAVES / 2` when `NUM_WAVES > 4`. Snapshot: `kernel_snapshot/`.

## Result

Correctness: all 7 PASS; bitwise equal to round 7 (`artifacts/bitwise_vs_r7.log`).

| shape | round-11 | round-7 | delta |
|---|---|---|---|
| b1_s4k | 993.8 | 1004.8 | -1.1 % |
| b1_s8k | 1043.7 | 1056.4 | -1.2 % |
| b1_s16k | 1071.6 | 1088.5 | -1.6 % |
| b1_s32k | 1082.2 | 1095.2 | -1.2 % |
| b1_s64k | 1092.6 | 1110.1 | -1.6 % |
| b2_s4k | 976.8 | 987.9 | -1.1 % |
| b2_s16k | 1057.0 | 1074.1 | -1.6 % |

Geomean 1044.56 vs 1058.67 (-1.33 %); aiter 1140.90; ratio 0.9156.

Decision: ROLLBACK (every shape down). Kernel restored from round-7 snapshot (verified with
`cmp`).

## Real-training transfer check

Bucket K1 (priority hint only). Rolled back, nothing to transfer.

## Lesson

Both asymmetric assignments lose to the uniform bump (-2.29 % later half, -1.33 % earlier
half). Within one barrier-locked WG, priority cannot create the phase offset that the
block_m 128 two-WG layout got for free.

## Reference read (aiter opus d192, `profiles/aiter_ref/`, read-only)

Same tiling as this kernel (32 q rows per wave, 8 waves, KV tile 64, K and V double-buffered,
LDS = 2·K + max(Q, 2·V)). The difference is the pipeline: 8 stages per tile, one workgroup
barrier per stage, memory and compute stages alternating; GEMM0 is split along KV into two
super units whose K (48 VGPRs) is read completely in the preceding memory stage; one wave
group executes one extra barrier in the prologue and the other one in the epilogue, so the
groups sit one stage apart and one group's MFMA stage overlaps the other's LDS stage. This
kernel has one barrier per tile (round 5 folded the rest) and near-empty memory clusters,
and its double-buffer protocol has no slack for a one-barrier offset.
