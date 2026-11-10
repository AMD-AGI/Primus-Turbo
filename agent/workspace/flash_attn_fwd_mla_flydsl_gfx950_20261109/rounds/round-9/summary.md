# Round 9 — ROLLED BACK (neutral)

- Time: 2026-11-09 09:32 (UTC+8)
- Hypothesis: in the loop QK clusters (C1/C5) a `_sched_barrier(0)` sits between
  `cast_p_and_sum` and the QK MFMAs, so the dependent 4-MFMA row-sum chain (one accumulator)
  runs serialized in front of the 24 QK MFMAs (ATT round 7: largest single MFMA stalls are the
  row-sum chain and the first QK MFMA). `cast_p_and_sum` already fences the f32 P before the
  row-sum MFMAs, so dropping the outer fence cannot make the scores coexist with f32 P. The
  epilogue C5 site already has no such fence.
- Change (single variable): skip that fence in loop C1/C5 when `K_STEPS_QK > 8` (row-sum MFMA
  path). Snapshot: `kernel_snapshot/`.

## Result

ISA: spill 0. The scheduler moved the row-sum MFMAs to the end of the QK sequence
(`q x22, r r q r q r`), the same order the epilogue already had.

Correctness: all 7 PASS; bitwise equal to round 7 (5 repeats x 4 shapes,
`artifacts/bitwise_vs_r7.log`).

| shape | round-9 | round-7 | delta |
|---|---|---|---|
| b1_s4k | 1005.2 | 1004.8 | +0.0 % |
| b1_s8k | 1056.3 | 1056.4 | -0.0 % |
| b1_s16k | 1094.8 | 1088.5 | +0.6 % |
| b1_s32k | 1096.7 | 1095.2 | +0.1 % |
| b1_s64k | 1102.8 | 1110.1 | -0.7 % |
| b2_s4k | 985.1 | 987.9 | -0.3 % |
| b2_s16k | 1074.2 | 1074.1 | +0.0 % |

Geomean 1058.40 vs 1058.67 (-0.03 %); aiter 1140.32; ratio 0.9282.

Decision: ROLLBACK (aggregate did not improve; within noise). Kernel restored from round-7
snapshot (verified with `cmp`).

## Real-training transfer check

Bucket K1 (scheduling fence only). Rolled back, nothing to transfer.

## Lesson

Moving the row-sum chain behind the QK MFMAs only relocates the dependency bubble to the end of
the cluster, in front of the barrier. Rounds 8 and 9 both show the per-wave instruction order
inside a cluster is not the limiter any more; the cost sits in the two loop barriers (wave
skew) with both waves of a SIMD in the same phase.
