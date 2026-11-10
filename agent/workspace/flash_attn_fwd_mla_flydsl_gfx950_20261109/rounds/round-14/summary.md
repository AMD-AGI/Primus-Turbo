# Round 14 — ACCEPTED (trade-off) — config sweep XCD_Q_GROUP x XCD_TAIL_Q

- Time: 2026-11-09 13:51 (UTC+8). Base: round 13.
- Machine change: rounds 1-13 ran on smci355-ccs-aus-n04-25; from this round on the campaign runs on
  smci355-ccs-aus-n02-25 (HIP device 0 = amd-smi GPU 3 on both, PWR_CAP 1400 W). Absolute numbers
  differ between the machines, so round 7 and round 13 were re-measured on n02-25 in the same
  session (`rebase_n02/`, with aiter) and round 14 is judged against the n02-25 round 13.
- Hypothesis: round 12 picked group 8 / tail 8 without a sweep. A larger group gives each K/V
  stream more co-resident readers per XCD; the tail length trades L2 reuse against the final-wave
  imbalance. Sweep G in {4, 8, 16} x T in {4, 8, 16} with the round-13 code (config only, one round).
- Change: `XCD_Q_GROUP` 8 -> 16 (`XCD_TAIL_Q` stays 8). `CODEGEN_REV` 3 -> 4.
  Snapshot: `kernel_snapshot/`.

## Sweep (full set, ours only, `sweep/G*_T*/bench*.log`)

| config | geomean n04-25 | geomean n02-25 |
|---|---|---|
| G4 T4 | 1094.32 | 1149.33 |
| G4 T8 | 1095.75 | 1153.85 |
| G4 T16 | 1096.54 | 1150.05 |
| G8 T4 | 1120.30 | 1179.54 |
| G8 T8 (round 13) | 1122.54 | 1180.42 |
| G8 T16 | 1117.47 | 1173.02 |
| G16 T4 | 1126.90 | 1198.22 |
| G16 T8 | 1127.21 | 1199.79 |
| G16 T16 | 1126.27 | 1199.85 |

G16 wins on both machines (+0.42 % on n04-25, +1.64 % on n02-25 over G8 T8); T is within noise at
G16, so T stays 8. G4 is ~2.5 % below G8 everywhere.

## Correctness

Bitwise equal to round 13 (10 repeats x B1 S4K / B1 S16K / B2 S4K / B1 S64K,
`artifacts/bitwise_vs_r13.log`). Full set 7/7 PASS (O rel 0.0016-0.0020, LSE max 0.0046-0.0051).

## Result on n02-25 (`artifacts/benchmark_full.csv`, `rebase_n02/*.csv`)

| shape | round-14 | round-13 | delta | round-7 | aiter | ratio |
|---|---|---|---|---|---|---|
| b1_s4k | 1063.8 | 1079.8 | -1.48 % | 1066.0 | 1078.1 | 0.987 |
| b1_s8k | 1191.8 | 1187.5 | +0.36 % | 1109.1 | 1206.6 | 0.988 |
| b1_s16k | 1285.2 | 1207.3 | +6.45 % | 1128.9 | 1268.3 | 1.013 |
| b1_s32k | 1310.2 | 1247.5 | +5.03 % | 1083.1 | 1291.8 | 1.014 |
| b1_s64k | 1326.0 | 1281.7 | +3.45 % | 1113.3 | 1300.8 | 1.019 |
| b2_s4k | 1035.5 | 1064.8 | -2.75 % | 1037.8 | 1044.2 | 0.992 |
| b2_s16k | 1250.7 | 1205.8 | +3.73 % | 1086.8 | 1268.0 | 0.986 |

Geomean 1203.93 vs 1179.59 (+2.063 %); aiter 1204.14; ratio 0.9998. Quick-set 1170.51, ratio 1.0027.
On n02-25 round 7 is 1088.92 (ratio 0.9041), so rounds 12-14 together give +10.56 % there.

Decision: ACCEPT WITH NOTE. Aggregate up; b1_s4k -1.48 % and b2_s4k -2.75 % (< 5 %).

## Attribution

- S >= 16K: a 16-rank group doubles the co-resident readers per K/V stream on each XCD
  (hypothesis from round 12's mechanism; this round did not re-profile L2).
- S4K: with G 16 and T 8, q_split = floor((16 - 8) / 16) * 16 = 0, so every rank is in the tail
  and S4K runs the round-7 order (round 7 on n02-25: b1_s4k 1066.0, round 14: 1063.8). The loss vs
  round 13 is the L2 reuse round 13 had from one 8-rank group at S4K.

## Real-training transfer check
- Bucket: K1 (kernel-internal workgroup mapping constant)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +2.063 % forward geomean (n02-25)
- Estimated real-training gain: same per shape; a 4K training step loses 1.5-2.8 % vs round 13
  and is level with round 7, S >= 16K gains 3.5-6.5 %.
- Decision: ACCEPT-as-real (with the 4K trade-off recorded)

## Next direction

Round 15: pick the group per launch so short sequences still get one group: G 16 when it leaves a
non-empty grouped section, else 8 (S4K then maps exactly like round 13). Expected to restore
b1_s4k / b2_s4k to round-13 levels without touching S >= 8K.
