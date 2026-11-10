# Round 13 — ACCEPTED

- Time: 2026-11-09 11:19 (UTC+8). Base: round 12 (1104.55).
- Hypothesis (from round 12's S4K counters: hit rate up, cycles +25 %, clock +17 %): the S4K loss
  is a load-balance tail. Round 12 groups 8 q ranks head-major to the very end, so each XCD's last
  dispatches hold jobs of 8 different causal lengths (up to half the longest at S4K) and CUs idle
  behind the longest. Dispatching the last XCD_TAIL_Q = 8 q ranks one rank at a time across all
  of the XCD's heads (the round-7 longest-first order) should shrink the tail; those ranks carry
  little of the total work at S >= 16K, so the L2 gain should mostly stay.
- Change (single variable): in the XCD mapping, q ranks [q_split, NQ) with
  q_split = floor(max(NQ - 8, 0) / 8) * 8 use rank-major / head-minor order; ranks below q_split
  keep round 12's groups (now always full, so every divisor is a constant). Bijective for every
  NQ in 1..69 and 256 (checked by enumeration). `CODEGEN_REV` 2 -> 3. Snapshot: `kernel_snapshot/`.

## Correctness

Bitwise equal to round 12 (10 repeats x B1 S4K / B1 S16K / B2 S4K / B1 S64K,
`artifacts/bitwise_vs_r12.log`). Full set 7/7 PASS (O rel 0.0016-0.0020, LSE max 0.0046-0.0051).

## Result (`artifacts/benchmark_full.csv`)

| shape | round-13 | round-12 | delta | round-7 | vs r7 | aiter | ratio |
|---|---|---|---|---|---|---|---|
| b1_s4k | 998.8 | 965.3 | +3.47 % | 1004.8 | -0.60 % | 1022.5 | 0.977 |
| b1_s8k | 1112.8 | 1088.1 | +2.27 % | 1056.4 | +5.34 % | 1142.1 | 0.974 |
| b1_s16k | 1170.8 | 1162.5 | +0.71 % | 1088.5 | +7.56 % | 1190.7 | 0.983 |
| b1_s32k | 1200.7 | 1195.2 | +0.46 % | 1095.2 | +9.63 % | 1217.6 | 0.986 |
| b1_s64k | 1216.6 | 1216.5 | +0.01 % | 1110.1 | +9.59 % | 1228.0 | 0.991 |
| b2_s4k | 998.8 | 969.7 | +3.00 % | 987.9 | +1.10 % | 1012.5 | 0.986 |
| b2_s16k | 1173.1 | 1165.1 | +0.69 % | 1074.1 | +9.22 % | 1197.4 | 0.980 |

Geomean 1121.20 vs 1104.55 (+1.507 %; +5.906 % vs round 7); aiter 1141.21; ratio 0.9825.
Quick-set geomean (full-run values) 1091.82, ratio 0.9843.

Decision: ACCEPT (aggregate up, every shape up vs round 12, outputs bitwise unchanged).

## Attribution (measured, B1, `profiles/pmc_r12/r13_*`, sustained 12 s loop, amd-smi GPU 3)

| | r7 S4K | r12 S4K | r13 S4K | r7 S16K | r12 S16K | r13 S16K |
|---|---|---|---|---|---|---|
| TCC hit rate | 0.367 | 0.746 | 0.613 | 0.372 | 0.802 | 0.792 |
| GRBM_GUI_ACTIVE / dispatch | 9.515e6 | 1.194e7 | 1.007e7 | 1.223e8 | 1.291e8 | 1.270e8 |
| sustained TFLOPS | 999.8 | 958.0 | 1000.5 | 1086.6 | 1171.2 | 1168.6 |
| GFX clock (MHz) | 1639-1643 | 1919-1935 | 1748 | 1578-1594 | 1647-1673 | 1630-1631 |

- S4K: cycles fall 15.7 % from round 12 and the clock drops back toward round 7's, consistent with
  less idle tail; the hit rate gives back part of round 12's gain (half the 16 ranks are the tail).
  Sustained S4K is now level with round 7 (1000.5 vs 999.8).
- S16K: hit rate and sustained throughput within noise of round 12 (the tail is 8 of 64 ranks).
- The remaining S4K cycle excess over round 7 (+5.8 %) is not explained yet.

## Real-training transfer check
- Bucket: K1 (kernel-internal workgroup mapping; no launch-config change, no wrapper cache)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +1.507 % forward geomean
- Estimated real-training gain: same as the benchmark per shape. A 4K training step is now within
  -0.6 % (B1) / +1.1 % (B2) of round 7; 8K and longer keep +5.3 % to +9.6 %.
- Decision: ACCEPT-as-real

## Next direction

The XCD order is close to aiter at S >= 16K (0.98-0.99). Remaining: S4K/S8K (0.974-0.986) and the
energy-per-FLOP gap. Candidates: the XCD_TAIL_Q / XCD_Q_GROUP sweep (one config round), then
HipKittens' 16x16x32 MFMA atom for matrix-pipe register traffic under the power cap.
