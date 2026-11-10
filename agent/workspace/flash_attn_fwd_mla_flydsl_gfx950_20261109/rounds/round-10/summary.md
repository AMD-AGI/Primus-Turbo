# Round 10 — ROLLED BACK

- Time: 2026-11-09 09:37 (UTC+8)
- Hypothesis: at block_m 256 the two waves on one SIMD belong to the same WG and are
  barrier-locked into the same cluster, so the uniform `s_setprio(1)` around the P*V clusters
  (designed for block_m 128, where the SIMD partner is another, unsynchronized WG) ranks
  nothing. ATT wave launch order (round 7 trace): a WG's waves start on SIMD 0, 2, 1, 3 slot 0,
  then SIMD 0, 2, 1, 3 slot 1, so wave w and w + 4 share a SIMD (assuming launch order = wave id).
  Raising P*V priority on the later-dispatched half (w >= 4), which otherwise loses
  equal-priority arbitration, should let the two waves trade the lead and shrink the skew seen
  at the two loop barriers (13.9 % of wave latency at B1 S16K).
- Change (single variable): in loop C3/C7, `s_setprio(1)` only for `wave_id >= NUM_WAVES / 2`
  when `NUM_WAVES > 4` (wave-uniform scalar branch). Epilogue unchanged.
  Snapshot: `kernel_snapshot/`.

## Result

ISA: spill 0; the loop `s_setprio 1` sits behind `s_cbranch_scc1`.
Correctness: all 7 PASS; bitwise equal to round 7 (`artifacts/bitwise_vs_r7.log`).

| shape | round-10 | round-7 | delta |
|---|---|---|---|
| b1_s4k | 984.4 | 1004.8 | -2.0 % |
| b1_s8k | 1034.3 | 1056.4 | -2.1 % |
| b1_s16k | 1059.2 | 1088.5 | -2.7 % |
| b1_s32k | 1067.2 | 1095.2 | -2.6 % |
| b1_s64k | 1080.6 | 1110.1 | -2.7 % |
| b2_s4k | 973.4 | 987.9 | -1.5 % |
| b2_s16k | 1047.0 | 1074.1 | -2.5 % |

Geomean 1034.43 vs 1058.67 (-2.29 %); aiter 1140.02; ratio 0.9074.

Decision: ROLLBACK (every shape down). Kernel restored from round-7 snapshot.

## Real-training transfer check

Bucket K1 (priority hint only). Rolled back, nothing to transfer.

## Lesson

Per-SIMD priority asymmetry is not neutral at block_m 256 (a uniform bump was the baseline):
giving the later wave the P*V lead costs 2–3 % on every shape. Not yet separated: the direction
of the asymmetry versus the cost of the extra scalar branch in the loop. Next round flips the
direction with the same branch structure.
