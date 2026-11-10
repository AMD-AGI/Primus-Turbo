# Round 5 — ROLLED BACK (rowsum MFMAs moved from s3 to s5)

- Time: 2026-11-09 11:25 (UTC+8)
- Base: round 4.
- Hypothesis (round-4 ATT, B1 S16K): s3 is MFMA-bound at 526/532 per visit (76 units of
  seed/cvt VALU prefix + 12 QK MFMA x 32 + 4 rowsum MFMA 16x16x32 at 32/16/16/16); aiter's
  s3 is 413 with no rowsum MFMA. s5 waits 42/111 at its barrier on the trailing group's
  s4 (~400) with only 8 PV MFMA (256), so 64 units of rowsum MFMA should fit there.
- Single change: `l_row = ctx.rowsum(l_row, p_packs)` moved from the end of s3 to the end of
  s5 (after the sched hints; rowsum starts with a sched_barrier(0) fence). The l_row update
  order relative to the s7 rebase is unchanged. SWP8_REV 8.
- Resources: 256 VGPR, spill 31 -> 35 (private 144 B), 176 MFMA, 37 barriers.

## Correctness

- `swp8_check.py`: bitwise equal to round 7 on 5 shapes x 3 reps. All 7 PASS vs fp32.

## Performance (full set, GPU 0)

| shape | round 5 | round 4 | delta |
|---|---|---|---|
| b1_s4k | 817.8 | 875.3 | -6.57 % |
| b1_s8k | 926.0 | 985.5 | -6.04 % |
| b1_s16k | 985.3 | 1034.8 | -4.78 % |
| b1_s32k | 1006.1 | 1068.1 | -5.80 % |
| b1_s64k | 1007.0 | 1083.1 | -7.03 % |
| b2_s4k | 821.4 | 865.7 | -5.12 % |
| b2_s16k | 963.4 | 1026.7 | -6.17 % |

Geomean 929.23 vs 987.84 (-5.93 %); aiter 1139.22; ratio 0.8157.

Decision: ROLLBACK. Working files restored from round-4 snapshot.

## Rollback Analysis

- Change made: rowsum from s3 to s5.
- Expected effect: s3 -64 per visit, s5 absorbs 64 in its barrier slack.
- Actual effect (ATT B1 S16K, round 4 -> round 5): s1 lost all 32 v_exp (seg9: mfma 12 +
  valu 23 only); s4 gained them plus the cast (seg12/seg20: v_exp 32, valu 82/91), 399/416 ->
  601/657; s5 296/395 -> 476/633 waiting on it. s3 itself did shrink (MFMA 16 -> 12).
- Root cause: once rowsum left s3, the first use of P is PV pair 0 in s5. The exp + cast chain
  was moved out of s1/s3 to the join block of the s4 causal-mask branch, which is a memory
  stage. Sinking to the first-use block across the branch (MachineSink) is the suspected
  mechanism; the move itself is measured.
- Lesson learned: values computed in an MFMA stage but first used after a stage containing
  a branch must be pinned (an opaque use) or the compiler may sink them into a memory stage.

## Real-training transfer check

- Bucket: K1; benchmark gain -5.93 % -> REJECT.

## Next direction

Round 6: the same move with the P packs pinned at the end of s3 by an empty tied inline asm
(`"=v,0"`, side effects), so the exp/cast chain stays in s1/s3.
