# flash_attn_fwd_mla (192/128) FlyDSL gfx950 — swp8 (aiter 8-stage port) optimize log

Campaign: forward-only tuning of the 8-stage port `flash_attn_fwd_mla_swp8.py`.
Target: beat round-7 dualwave (1057.20 this session), then same-GPU ratio vs Turbo AITER >= 1.0.
GPU 0 pinned. Previous campaign: `flash_attn_fwd_mla_flydsl_gfx950_20261109` (round 7 accepted).

## Baseline (round-0) — 2026-11-09 10:10

- swp8 v2 (SWP8_REV 2): bitwise equal to round 7 on 5 check shapes; all 7 PASS vs fp32.
- Full geomean 965.66 / aiter 1140.55, ratio 0.8467 (round 7 same session 1057.20).
- ISA: 256 VGPR, spill 33, 176 MFMA, 37 barriers, no scratch in the loop.
- ATT B1 S16K vs aiter: per tile ~3416 vs ~2783 units; MFMA-instruction time 2.0x, barrier
  1.65x, waitcnt 2.4x aiter. Seed movs on s1/s3, cvt bunching in s3, K DMA x3 in s4.

## Round 1 — 2026-11-09 10:15 — ROLLBACK

- Change: QK from srcC = 0 and S(t) -= seed_offset(ref) at the start of s5 (removes ~34
  seed v_mov per tile).
- Result: correct (deterministic; fp32 gates unchanged); full geomean 964.14 (-0.16 %).
- ATT: s3/s4/s2 shorter as intended, but the 64-bit loop-exit VALU compare in phase-A s0
  stalled 211 units (round 0: 20) and the s5 shift compiled to 32 scalar v_add. Net zero.
- Next: scalar (i32) loop control on round-0 code, then re-try the zero seed.

## Round 2 — 2026-11-09 10:35 — ACCEPT (trade-off)

- Change: main loop as `while j < t_end - 1` on an i32 counter (FlyDSL scf_range always uses
  index) and the K(t+2) clamp in i32; loop exit and clamp now SALU. Arithmetic unchanged.
- Result: bitwise equal to round 7 (5 shapes x 3 reps); full geomean 978.41 vs 965.66
  (+1.32 %), b1_s64k -0.87 %, others +0.54 .. +2.92 %. Ratio 0.8576. Spill 33 -> 31.

## Round 3 — 2026-11-09 10:50 — ROLLBACK

- Change: zero QK seed + S shift as inline-asm v_pk_add_f32 (16 per tile) in s5.
- Result: deterministic, fp32 gates unchanged; full geomean 949.89 vs 978.41 (-2.92 %), all
  shapes down. ATT: each v_pk_add_f32 issued behind an in-flight MFMA stalls ~40 units;
  s5 336/397 -> 522/510. Packed FP32 VALU serializes behind MFMA, scalar VALU does not.
- Note: an ATT of "round 2" taken while the working source was round 3 traced round 3 (the
  cache key follows SWP8_REV of the current source); re-taken after restoring round 2.
- Correction (appended 11:00): the stalled loop-head instruction in round 3 is the 32-bit
  `v_add_u32` mask IV (206 units vs 21 in round 2), so the round-1 loop-head stall was a
  position effect (first VALU of the trailing group's s0 while the partner's s1 runs MFMA +
  exp), not a 64-bit-VALU effect. Round 2's gain stands as measured.

## Round 4 — 2026-11-09 11:10 — ACCEPT

- Change: K(t+2) DMA split as 1 instruction in s4 + 2 in s6 (aiter's split); SWP8_REV 7.
- Result: bitwise equal to round 7; full geomean 987.84 vs 978.41 (+0.96 %), every shape up
  (+0.30 % b2_s4k .. +2.86 % b1_s64k); ratio vs aiter 0.8669. New best.

## Round 5 — 2026-11-09 11:25 — ROLLBACK

- Change: rowsum (4 MFMA 16x16x32) moved from s3 to the end of s5; SWP8_REV 8.
- Result: bitwise equal to round 7; spill 31 -> 35; full geomean 929.23 vs 987.84 (-5.93 %),
  all shapes down. ATT: the 32 v_exp and the cast left s1/s3 and landed in s4 (601/657 per
  visit): with the rowsum gone, P's first use is in s5, and the exp/cast chain was sunk past
  the s4 causal-mask branch into its join block (MachineSink suspected, not confirmed).

## Round 6 — 2026-11-09 11:40 — ACCEPT

- Change: rowsum moved to the end of s5 with the P packs pinned in s3 (empty tied inline asm
  per pack) so the exp/cast chain stays in s1/s3; SWP8_REV 9.
- Result: bitwise equal; full geomean 992.33 vs 987.84 (+0.45 %); ratio 0.8732. ATT loop pair
  -3.2 %, whole traced CU -2.7 % cycles.
- Finding: sustained runs sit at the ~1.4 kW power cap (amd-smi GPU 3). Round 6 clocks ~1 %
  below round 4; aiter clocks ~1378 MHz vs swp8 ~1600 MHz and still wins 14.5 %. Under the
  cap the objective is energy per tile, so removing instructions matters more than removing
  idle cycles. Next: PMC instruction totals swp8 vs aiter.

## Round 7 candidate — 2026-11-09 11:05 — NOT ACCEPTED (race), campaign paused

- Change: V buffer-1 LDS reads from one opaque base VGPR (its offset exceeds the 16-bit ds
  immediate, so LLVM materialized one address VGPR per read). Spill 32 -> 0, v_add_u32 63 -> 10.
- Correctness: nondeterministic, b1_s16k 8/64 reps differ from round 7 (max|dO| ~2e-3..4e-3,
  LSE also moves); round 6 is 88/88 clean. With --no-stagger 64/64 clean. Opaque base only in
  the loop or only in the epilogue (spill 2 each) both still race. A long-lived base instead
  of a per-stage copy lowered but did not remove it. Not benchmarked.
- Hypothesis (unverified): an end-of-kernel ordering hole between the two wave groups that the
  slow serial spill reloads in the epilogue used to hide.
- Campaign paused at the user's request (back to the dualwave round-7 lineage, look at the
  HipKittens attention design). Working files restored to the round-6 snapshot; the candidate
  is in rounds/round-7-candidate/kernel_snapshot/.
