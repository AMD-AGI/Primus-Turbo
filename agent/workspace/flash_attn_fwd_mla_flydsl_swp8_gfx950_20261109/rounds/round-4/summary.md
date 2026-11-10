# Round 4 — ACCEPTED (K(t+2) DMA split 1 in s4 + 2 in s6)

- Time: 2026-11-09 11:10 (UTC+8)
- Base: round 2 (round 3 rolled back).
- Hypothesis: in round 2, s5 waits 58/107 units at its barrier on the trailing group's s4,
  which issues all 3 K(t+2) DMA plus the 16 V pair-0 reads; s6 phase B has slack (276).
  aiter issues K(t+2) as 1 + 2 over its stages 4 and 6.
- Single change: `Swp8Context.load_k_tile_part(tile, buf, d_lo, d_hi)`; s4 issues d = 0,
  s6 issues d = 1, 2 ahead of the V pair-1 reads and before its vmcnt(KEEP_VM) wait. The wait
  still lands K(t+1) (the five newest ops are K(t+2) x3 and V(t) x2) and the s2 wait still
  lands V(t-1). SWP8_REV 7.
- Resources: 256 VGPR, spill 31 (private 128 B), 176 MFMA, 37 barriers (unchanged).

## Correctness

- `swp8_check.py`: bitwise equal to round 7 on 5 shapes x 3 reps.
- All 7 PASS vs fp32: O rel 0.0016–0.0020, LSE max abs 0.0046–0.0051.

## Performance (full set, GPU 0)

| shape | round 4 | round 2 | delta |
|---|---|---|---|
| b1_s4k | 875.3 | 871.3 | +0.46 % |
| b1_s8k | 985.5 | 980.7 | +0.49 % |
| b1_s16k | 1034.8 | 1028.1 | +0.65 % |
| b1_s32k | 1068.1 | 1053.5 | +1.39 % |
| b1_s64k | 1083.1 | 1053.0 | +2.86 % |
| b2_s4k | 865.7 | 863.1 | +0.30 % |
| b2_s16k | 1026.7 | 1020.5 | +0.61 % |

Geomean 987.84 vs 978.41 (+0.96 %); aiter 1139.57; ratio 0.8669 (round 2: 0.8576).
vs round 0 baseline 965.66: +2.30 %.

Decision: ACCEPT (all shapes up, no regression).

## Real-training transfer check

- Bucket: K1 (kernel-internal schedule; no cache, no id() keys, no host work). The kernel is
  the same callable and output is bitwise unchanged, so the gain applies to every forward
  call at these shapes.
