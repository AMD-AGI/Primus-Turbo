# Round 0 — BASELINE (swp8 v2, aiter 8-stage port)

- Time: 2026-11-09 10:10 (UTC+8)
- Kernel: `flash_attn_fwd_mla_swp8.py` SWP8_REV 2 — 8 barrier-delimited stages per KV tile
  (GEMM0 split along KV into two 12-MFMA halves, GEMM1 split along D_V into two 8-MFMA pairs),
  waves 4-7 trail waves 0-3 by one stage, softmax arithmetic identical to round-7 dualwave.
- Resources (ISA): 256 VGPR, spill 33 (private 136 B, no scratch in the main loop), 176 MFMA,
  37 s_barrier.

## Correctness

- `swp8_check.py`: bitwise equal to round 7 on b1_s1k/4k/16k/64k, b2_s4k (3/3 reps each).
- All 7 PASS vs fp32: O rel 0.0016–0.0020, LSE max abs 0.0046–0.0051.

## Performance (full set, GPU 0, same session)

| shape | swp8 r0 | round 7 | aiter | ratio vs aiter |
|---|---|---|---|---|
| b1_s4k | 846.6 | 1005.6 | 1022.0 | 0.828 |
| b1_s8k | 963.7 | 1050.6 | 1145.4 | 0.841 |
| b1_s16k | 1013.4 | 1086.6 | 1189.5 | 0.852 |
| b1_s32k | 1047.8 | 1096.4 | 1214.1 | 0.863 |
| b1_s64k | 1062.2 | 1108.1 | 1224.4 | 0.868 |
| b2_s4k | 846.5 | 984.3 | 1013.8 | 0.835 |
| b2_s16k | 1005.2 | 1075.3 | 1196.4 | 0.840 |

Geomean 965.66 (round 7 1057.20, aiter 1140.55); ratio 0.8467.
Quick-set A/B (earlier run): stagger off 863.58 vs 946.76 (stagger worth +9.6 %), sched hints off
921.66 (+2.7 %).

## Profile (ATT, B1 S16K, traced CU 1; units = ATT time units)

Stage visits on the traced CU match aiter (barrier hits 534528 vs 535040), so totals compare
directly. Per tile per wave: ~3416 (phase A) vs aiter ~2783, consistent with the 0.85 ratio.

| stage | work | swp8 A | swp8 B | aiter |
|---|---|---|---|---|
| s0 mem | V DMA, K half-0 reads | 469 | 357 | 376 |
| s1 mma | QK half 0, 32 exp | 521 | 380 | 373 |
| s2 mem | K half-1 reads | 499 | 457 | 408 |
| s3 mma | QK half 1, cast, rowsum | 525 | 538 | 413 |
| s4 mem | K DMA x3, V pair-0 reads, mask | 425 | 441 | 301 |
| s5 mma | PV pair 0, max trigger | 337 | 424 | 304 |
| s6 mem | V pair-1 reads | 328 | 281 | 316 |
| s7 mma | PV pair 1, rebase | 312 | - | 290 |

By class (whole kernel): MFMA latency 26.7M vs aiter 13.0M, barrier 27.4M vs 16.3M, waitcnt
12.9M vs 5.1M; ds_read / VALU / exp comparable.

Findings:
- The -reference QK seed costs ~34 v_mov per tile (s1 phase A: 15 v_mov_b32 + 8 v_mov_b64
  copy + s_nop; s3: 17 v_mov; phase-B seed built at the s7 join). aiter uses srcC = 0.
- s3: 16 cvt + 17 seed movs sit before the third MFMA, then 9 back-to-back MFMAs at 32 each.
- s4 issues all 3 K DMA instructions (aiter: 1 in s4, 2 in s6).
- V buffer 1 sits beyond the 16-bit ds offset: phase-B s6 needs one v_add_u32 per V read.
- exp issues at 8 units, a 32x32x16 MFMA occupies 32: 3 exp per MFMA hide in its shadow.
