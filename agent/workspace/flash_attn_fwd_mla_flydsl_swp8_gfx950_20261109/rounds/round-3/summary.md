# Round 3 — ROLLED BACK (zero QK seed with a packed v_pk_add_f32 shift in stage 5)

- Time: 2026-11-09 10:50 (UTC+8)
- Hypothesis: round 1's zero seed failed on two counts, the 64-bit loop compare (fixed in
  round 2) and a scalar S shift (32 v_add_f32 in s5). Re-apply the zero seed on round 2 with
  the shift as 16 v_pk_add_f32 per tile.
- Single change: `qk_half(..., c_zero_v16f32)` in s1/s3; `shift_by_ref` at the start of s5 as
  8 inline-asm `v_pk_add_f32` per S half (a vector-level add was still scalarized by LLVM,
  SWP8_REV 5, not benchmarked). SWP8_REV 6.
- Resources: 256 VGPR, spill 30 (private 124 B), 176 MFMA, 37 barriers.

## Correctness

- `swp8_check.py --approx`: deterministic 3/3 on all 5 shapes; max|dO| 3.906e-3,
  max|dLSE| 4.7e-4 .. 1.7e-3 vs round 7 (same as round 1; zero-seed rounding order).
- All 7 PASS vs fp32: O rel 0.0016–0.0020, LSE max abs 0.0046–0.0051.

## Performance (full set, GPU 0)

| shape | round 3 | round 2 | delta |
|---|---|---|---|
| b1_s4k | 854.0 | 871.3 | -1.99 % |
| b1_s8k | 950.3 | 980.7 | -3.10 % |
| b1_s16k | 988.5 | 1028.1 | -3.85 % |
| b1_s32k | 1020.8 | 1053.5 | -3.10 % |
| b1_s64k | 1035.8 | 1053.0 | -1.63 % |
| b2_s4k | 841.0 | 863.1 | -2.56 % |
| b2_s16k | 978.1 | 1020.5 | -4.15 % |

Geomean 949.89 vs 978.41 (-2.92 %); aiter 1139.39; ratio 0.8337.

Decision: ROLLBACK. Working files restored from round-2 snapshot.

## Rollback Analysis

- Change made: zero seed + packed shift.
- Expected effect: s1/s3 lose the seed movs; s5 gains only 16 VALU.
- Actual effect (ATT B1 S16K, per visit, round 2 -> round 3): s3 526/530 -> 458/460 and s2
  495/456 -> 459/424 as intended, but s5 336/397 -> 522/510 (VALU 160/179 -> 452/461), and
  phase-A s0 444 -> 553 with its first VALU stalling 210, leading s1 to wait 168 at the barrier.
- Root cause (measured): in s5 every `v_pk_add_f32` issued right after a `v_mfma_f32_32x32x16`
  stalls ~40 units (MFMA issue 4 + ~40 ~= the MFMA's 32-unit occupancy); the second pk_add
  and the scalar v_max3 that follow do not stall. Packed-FP32 VALU waits for the in-flight
  MFMA on the same SIMD; scalar VALU does not.
- Second cause (measured): the first VALU after the loop head in the trailing group's s0 is
  a 32-bit `v_add_u32 v242, -128, v242` (causal-mask column IV) and stalls 206 units per
  visit (round 2: 21). So the round-1 loop-head stall was not specific to 64-bit VALU: it
  appears whenever the partner wave's s1 starts directly with MFMA + exp (zero seed) instead
  of seed movs, and the in-order s0 then issues its 12 K reads ~200 late. Mechanism (VALU
  issue starvation behind the partner's MFMA/exp stream) is a hypothesis.
- Lesson learned: in MFMA-interleaved code on gfx950, keep VALU scalar (v_add_f32 / v_max3);
  v_pk_* f32 ops serialize behind MFMAs. Mem-stage VALU ahead of LDS reads can be starved by
  the partner's exp+MFMA stage; the zero seed needs that VALU out of s0 first.

## Real-training transfer check

- Bucket: K1; no cache; no id() keys. Benchmark gain -2.92 % -> REJECT (no gain).

## Next direction

The zero seed needs the trailing group's s0 free of VALU ahead of the K reads; park it for
now. Round 4: split the K(t+2) DMA as aiter does (1 instruction in s4, 2 in s6); s5 waits
58/107 at its barrier on the trailing group's s4 (3 K DMA + 16 V reads).
