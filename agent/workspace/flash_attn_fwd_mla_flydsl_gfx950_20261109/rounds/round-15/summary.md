# Round 15 — ACCEPTED

- Time: 2026-11-09 13:53 (UTC+8). Base: round 14 (1203.93, n02-25). Machine: smci355-ccs-aus-n02-25.
- Hypothesis (from round 14's attribution): with group 16 and tail 8, S4K (16 q ranks) has no
  grouped section and falls back to the round-7 order, losing the one 8-rank group round 13 had
  there. Choosing the group per launch (16 when at least 16 ranks precede the tail, else 8) should
  restore S4K to round-13 levels and leave every S >= 8K launch mapped exactly as in round 14.
- Change (single variable): `_xcd_map(group)` evaluates the round-14 mapping for group 16 and for
  `XCD_Q_GROUP_SHORT = 8`; `_nq_head >= 16` selects between them. Checked by enumeration:
  bijective for NQ 1..79 and 256, identical to round 14 for NQ >= 24, identical to round 13 at
  NQ = 16. `CODEGEN_REV` 4 -> 5. Snapshot: `kernel_snapshot/`.

## Correctness

Bitwise equal to round 14 (10 repeats x B1 S4K / B1 S16K / B2 S4K / B1 S64K,
`artifacts/bitwise_vs_r14.log`). Full set 7/7 PASS (O rel 0.0016-0.0020, LSE max 0.0046-0.0051).

## Result (n02-25, `artifacts/benchmark_full.csv`)

| shape | round-15 | round-14 | delta | round-13 (n02) | round-7 (n02) | aiter | ratio |
|---|---|---|---|---|---|---|---|
| b1_s4k | 1084.6 | 1063.8 | +1.96 % | 1079.8 | 1066.0 | 1078.6 | 1.006 |
| b1_s8k | 1191.1 | 1191.8 | -0.06 % | 1187.5 | 1109.1 | 1208.0 | 0.986 |
| b1_s16k | 1284.2 | 1285.2 | -0.08 % | 1207.3 | 1128.9 | 1270.0 | 1.011 |
| b1_s32k | 1312.9 | 1310.2 | +0.20 % | 1247.5 | 1083.1 | 1294.2 | 1.014 |
| b1_s64k | 1329.0 | 1326.0 | +0.22 % | 1281.7 | 1113.3 | 1298.8 | 1.023 |
| b2_s4k | 1066.7 | 1035.5 | +3.01 % | 1064.8 | 1037.8 | 1071.1 | 0.996 |
| b2_s16k | 1249.3 | 1250.7 | -0.11 % | 1205.8 | 1086.8 | 1266.8 | 0.986 |

Geomean 1212.71 vs 1203.93 (+0.729 %); aiter 1208.92; ratio 1.0031. Quick-set 1185.40, ratio 1.0089.
Against round 7 on the same machine: +11.368 %.

Decision: ACCEPT (aggregate up; only S4K launches changed and both went up; the S >= 8K deltas
are noise on bitwise-identical, identically mapped launches).

## Attribution

Only NQ in [9, 23] changes mapping; in this set that is S4K. Both S4K shapes return to round-13
levels (+0.45 % / +0.18 % vs round 13 on n02-25). S >= 8K launches are unchanged by construction.

## Real-training transfer check
- Bucket: K1 (kernel-internal workgroup mapping)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +0.729 % forward geomean
- Estimated real-training gain: +2.0 % (B1) / +3.0 % (B2) for 4K-sequence steps vs round 14;
  none elsewhere.
- Decision: ACCEPT-as-real

## Next direction

Campaign target (geomean ratio >= 1.0 vs aiter on the same GPU) is met: 1.0031. Below parity:
b1_s8k (0.986), b2_s16k (0.986), b2_s4k (0.996). Further levers: the batch axis is the slowest
grid axis, so B2 runs batch 0 then batch 1 (LPT across batches, XCD grouping across batches), and
the energy-per-FLOP gap (HipKittens' 16x16x32 MFMA atom).
