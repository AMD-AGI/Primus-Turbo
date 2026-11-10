# Round 8 — ROLLED BACK

- Time: 2026-11-09 09:29 (UTC+8)
- Hypothesis: with the spill gone (round 7), the k_head trade-off measured in rounds 3–4 (fewer
  head K packs to cut spills) may flip. ATT on round 7 (B1 S16K, one CU) shows the QK clusters
  paced by just-in-time tail K reads: `s_waitcnt lgkmcnt(0|1|2)` before nearly every QK MFMA
  (~18–50 cycles each), tail reads rotating through 2–3 register quads.
- Change (single variable): `ops.py` default `k_head` 1 -> 2. ISA: spill 0, scratch 0 B
  (k_head 3 -> spill 1, k_head 4 -> spill 11, so 2 is the largest spill-free value).
  Snapshot: `kernel_snapshot/`.

## Evidence before the change (ATT, round 7, B1 S16K)

- Wave latency split: MFMA 30.5 %, s_waitcnt 14.5 %, s_barrier 13.9 %, ds_read 12.3 %,
  v_exp 9.9 %, VALU 8.1 %, vmem load 7.7 %. Main loop segments ~93 % of latency.
- The two loop barriers (end of C1 and C5) are 6.0 % and 6.4 %; the full drains in front of them
  (`s_waitcnt vmcnt(0) lgkmcnt(0)`) only 0.7 % each, so DMA latency is hidden and the barrier
  cost is wave skew.
- The 4-MFMA row-sum chain (16x16x32, one accumulator) right after cast_p, then the first QK MFMA,
  carry the largest single MFMA stalls (first QK MFMA ~134 cycles per execution).

## Result

Correctness: all 7 PASS (O rel 0.0016–0.0020, LSE max 0.0046–0.0051).

| shape | round-8 | round-7 | delta |
|---|---|---|---|
| b1_s4k | 1003.4 | 1004.8 | -0.1 % |
| b1_s8k | 1054.5 | 1056.4 | -0.2 % |
| b1_s16k | 1084.9 | 1088.5 | -0.3 % |
| b1_s32k | 1094.0 | 1095.2 | -0.1 % |
| b1_s64k | 1100.3 | 1110.1 | -0.9 % |
| b2_s4k | 983.4 | 987.9 | -0.5 % |
| b2_s16k | 1070.0 | 1074.1 | -0.4 % |

Geomean 1054.93 vs 1058.67 (-0.35 %); aiter 1140.19; ratio 0.9252.

Decision: ROLLBACK (aggregate down, all shapes slightly down). Kernel restored from round-7
snapshot (verified with `cmp`).

## Real-training transfer check

Bucket K1 (launch default only). Rolled back, nothing to transfer.

## Lesson

Even spill-free, one more head K k-step does not help: the QK cluster is not limited by how far
ahead the first K packs are read. The remaining loop costs are wave skew at the two barriers and
the serialized row-sum chain in front of the QK MFMAs.
