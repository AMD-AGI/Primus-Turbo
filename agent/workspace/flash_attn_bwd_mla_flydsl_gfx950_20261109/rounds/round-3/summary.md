# Round 3 — ACCEPTED — 2026-11-09 23:20 — q_split 1 by default

## Hypothesis
With q_split 2 each kv band's dK/dV is written as two bf16 slots and folded by two torch reduce
kernels (579 us = 9.3 % of kernel time at b2_s4k, 1143 us = 2.8 % at b1_s16k, round 1). A probe
showed the main kernel no slower at q_split 1 (5151 vs 5221 us at b2_s4k): at block_kv 128 the grid
already holds Skv/128 * Hkv * B >= 4096 work-groups, so splitting the long bands buys no balance.
Expected: about +9 % at the 4K shapes, +2-3 % at 16K, 0 at b1_s64k (already q_split 1), and
+1.1-1.5 dB dK/dV SNR (no bf16 slot round-trip).

## Single change
`ops_bwd.mla_bwd_sbhd` default `q_split` 2 -> 1 (the int32-length fallback becomes an assert).
Kernel source unchanged (snapshot diff: ops_bwd.py only).

## Results (n02-29 GPU 0, full set, rotated vs aiter a16) — 7 / 7 PASS

| shape | Step TFLOPS | delta vs R2 | Bwd TFLOPS | Bwd ms | step ratio vs aiter | SNR dq/dk/dv |
|---|---|---|---|---|---|---|
| b1_s4k | 715.09 | +8.624 % | 637.35 | 2.804 | 0.828 | 46.43/49.57/51.21 |
| b1_s8k | 774.49 | +3.609 % | 684.33 | 10.445 | 0.793 | 45.18/49.54/51.27 |
| b1_s16k | 808.18 | +1.927 % | 710.43 | 40.242 | 0.781 | 43.99/49.66/51.08 |
| b1_s32k | 831.45 | +1.804 % | 728.65 | 156.937 | 0.769 | 44.07/49.62/51.16 |
| b1_s64k | 839.47 | -0.024 % | 733.82 | 623.318 | 0.762 | 44.39/49.60/51.16 |
| b2_s4k | 721.13 | +9.207 % | 642.96 | 5.559 | 0.811 | 46.39/49.66/51.38 |
| b2_s16k | 806.82 | +3.607 % | 708.68 | 80.682 | 0.768 | 43.83/49.54/51.31 |

Geomean: Step 783.820 (+4.058 % vs R2; aiter 996.03, ratio 0.7869), Bwd 691.357 (ratio 0.7417),
Fwd 1202.494 (unchanged kernel; +0.137 % noise).

## Decision
ACCEPTED: aggregate +4.058 %, no shape regressed (worst -0.024 %, the shape that already ran q_split 1).

## Attribution
The gain tracks the removed fold's share of each shape's time (largest at 4K, where the fold is
~9 % of the backward), and dK/dV SNR rose 1.1-1.6 dB as the slot round-trip predicted.

## Real-training transfer check
- Bucket: K1 (launch configuration)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +4.058 % combined-step
- Estimated real-training gain: +4.058 % (no dependence on tensor reuse)
- Decision: ACCEPT-as-real

## Next direction
The main kernel is now the whole gap (0.73-0.75 of aiter's main kernel). Its MLA cost scales worse
than FLOPs vs the D128 build (5151 vs 3346 us = 1.54x for 1.30x FLOPs at b2_s4k, aiter 1.33x). Profile
the three main kernels (MLA ours, aiter MLA, Turbo D128) with counters before the next change.
