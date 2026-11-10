# Round 1 — ROLLED BACK (QK from a zero accumulator, reference subtracted in stage 5)

- Time: 2026-11-09 10:15 (UTC+8)
- Hypothesis: the -reference seed puts ~34 v_mov per tile on the critical MFMA stages (s1, s3,
  s7 join). Starting QK from srcC = 0 (inline constant, as aiter does) and subtracting
  seed_offset(reference) once in s5 (which has barrier slack) should shorten s1/s3.
- Single change: `qk_half(..., c_zero_v16f32)` in s1 and s3; new `Swp8Context.shift_by_ref`
  applied to S(t) at the start of s5 before the max trigger; `fresh_seed` removed. SWP8_REV 3.
- Resources: 256 VGPR, spill 31 (private 128 B), 176 MFMA, 37 barriers.

## Correctness

- No longer bitwise vs round 7 (fp32 rounding order of S differs). `swp8_check.py --approx`:
  deterministic over 3 reps on all 5 check shapes; max|dO| 3.906e-3 (one bf16 ulp in [0.5, 1)),
  max|dLSE| 4.7e-4 .. 1.7e-3. LSE moves come from the bf16 MFMA row sum: ~1e-5 S perturbations
  flip bf16 roundings of P.
- All 7 PASS vs fp32: O rel 0.0016–0.0020, LSE max abs 0.0046–0.0051 (same as round 0).

## Performance (full set, GPU 0)

| shape | round 1 | round 0 | delta |
|---|---|---|---|
| b1_s4k | 858.1 | 846.6 | +1.36 % |
| b1_s8k | 963.2 | 963.7 | -0.05 % |
| b1_s16k | 1010.9 | 1013.4 | -0.25 % |
| b1_s32k | 1039.5 | 1047.8 | -0.79 % |
| b1_s64k | 1043.7 | 1062.2 | -1.74 % |
| b2_s4k | 851.8 | 846.5 | +0.63 % |
| b2_s16k | 1002.9 | 1005.2 | -0.23 % |

Geomean 964.14 vs 965.66 (-0.16 %); aiter 1139.58; ratio 0.8461.

Decision: ROLLBACK (aggregate did not improve). Working files restored from round-0 snapshot.

## Rollback Analysis

- Change made: zero-seeded QK + one explicit shift of S(t) in s5.
- Expected effect: s1/s3 shorter by the seed movs (~140 + ~70 units per phase A tile).
- Actual effect (ATT B1 S16K, per visit): s3 525/538 -> 458/461, s4 425/441 -> 387/427,
  s2 499/457 -> 459/424 (as intended); but phase-A s0 469 -> 577 and s1 521 -> 637 (barrier wait
  in s1 17 -> 196), and s5 phase B 424 -> 460.
- Root cause:
  1. The loop-exit test is a 64-bit compare of the index induction variable, which gfx9 can
     only do in VALU (`v_mov_b64 v[64:65], s[2:3]` + `v_cmp_lt_u64`). In round 1 that v_mov_b64,
     first VALU of the trailing group's s0, stalled 211 units per visit (round 0: 20), and the
     leading group then waited ~180 more at the s1 barrier. The same instruction in phase-B s4
     stalls 6. Measured: the stall coincides with the partner wave's exp-dense s1 now starting
     straight with MFMA+exp instead of seed movs. Mechanism not established (hypothesis: VALU
     issue contention / structural interaction with the partner's exp+MFMA stream).
  2. The s5 shift compiled to 32 scalar `v_add_f32_e64 ..., -v176` instead of 16 v_pk_add_f32,
     adding VALU to a stage that already waits on s4.
- Lesson learned: removing VALU from compute stages can move the bottleneck into VALU left in
  the partner's mem stage; loop control should be scalar before re-trying the seed removal.

## Real-training transfer check

- Bucket: K1 (kernel-internal arithmetic order and scheduling); no cache; no id() keys.
- 4-step trace hit rate: n/a (no cache).
- Benchmark gain: -0.16 %; estimated real-training gain: none (rolled back).
- Decision: REJECT (no gain), not an over-fit issue.

## Next direction

Round 2 (from round 0): scalar loop control — i32 induction variable so the loop-exit test is
s_cmp in SALU; then re-try the zero seed with a packed shift.
