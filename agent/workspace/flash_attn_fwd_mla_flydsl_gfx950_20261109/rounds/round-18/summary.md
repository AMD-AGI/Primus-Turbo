# Round 18 — ACCEPTED (trade-off)

- Time: 2026-11-09 17:01 (UTC+8). Base: round 17. Machine: smci355-ccs-aus-n02-25, HIP device 0
  (amd-smi GPU 3).
- Hypothesis: K/V reads that miss the XCD's L2 still cost time at the power limit. A wrong-result
  probe where every CTA reads kv head 0 (`profiles/variants/fake_kv_head0`, K/V almost always in
  L2 / MALL) ran B1 S16K at 8.043 ms/call vs 8.339 (2 x 14 s sustained each,
  `profiles/r18_analysis/clock_power_kvh0.txt`): -3.5 % time, clock ~2146 -> ~2169 MHz at the same
  ~1360 W. Round 17 groups 16 q ranks of one head per XCD (8 when short) and puts the ranks after the
  groups into a rank-major tail of 8 or more (16 at S8K-S64K). With 32 CUs per XCD and 1 CTA per CU,
  a 32-rank group would keep one head's K/V streaming through each XCD's L2 instead of two, and
  grouping the remainder above the 8-rank tail would cover ranks that round 17 left ungrouped.
  Expected: S >= 8K faster; S4K unchanged (its mapping does not change).
- Change (single variable: the XCD q-group levels; config sweep {(16, 8), (32, 16, 8)} on the same
  code):
  - `XCD_Q_GROUP` / `XCD_Q_GROUP_SHORT` replaced by `XCD_Q_GROUPS`, largest first. Ranks ahead of
    the 8-rank tail are cut into as many full groups of the first size as fit, the remainder into
    the next size, and so on; what is left joins the rank-major tail (still 8 when nq >= 16).
  - Sections per launch (BLOCK_M 256, `artifacts/xcd_map_check.txt`):

    | S | round 17 | (16, 8) | (32, 16, 8) |
    |---|---|---|---|
    | 4K (nq 16) | G8 x1, tail 8 | G8 x1, tail 8 | G8 x1, tail 8 |
    | 8K (nq 32) | G16 x1, tail 16 | G16, G8, tail 8 | G16, G8, tail 8 |
    | 16K (nq 64) | G16 x3, tail 16 | G16 x3, G8, tail 8 | G32, G16, G8, tail 8 |
    | 32K (nq 128) | G16 x7, tail 16 | G16 x7, G8, tail 8 | G32 x3, G16, G8, tail 8 |
    | 64K (nq 256) | G16 x15, tail 16 | G16 x15, G8, tail 8 | G32 x7, G16, G8, tail 8 |

  - `CODEGEN_REV` 8 -> 9 (the (16, 8) package used 109 so the two configs cannot share a cache
    entry). `flash_attn_fwd_mla.py` and `ops.py` unchanged. Snapshot: `kernel_snapshot/` = (32, 16, 8).

## Correctness

- Mapping: a host replica (`artifacts/xcd_map_check.py`) is a bijection onto (head, q rank) per
  XCD for nq 1-400 and 1, 2, 8, 16 heads per XCD, for both configs.
- O / LSE bitwise equal to round 17 (5 / 5 calls each at B1 S4K, B1 S16K, B2 S4K, B1 S64K;
  `artifacts/bitwise_vs_r17.log`); both kernels were compiled fresh in that run.
- Full set 7 / 7 PASS in every full run of both configs (O rel vs fp32 0.0016-0.0020, LSE max
  0.0046-0.0051, same as round 17); quick set PASS in all 9 rotation runs.

## Result (full set with aiter, same session; order r17, g32, g16 then g16, g32, r17)

`artifacts/ab_full_{r17,g32,g16}_{1,2}.csv`, table in `artifacts/ab_full_table.txt`. g32 = (32, 16, 8),
g16 = (16, 8). TFLOPS are 2-run means.

| shape | round 17 | g32 | g32 vs r17 | g16 | g16 vs r17 | aiter (g32 runs) | g32 / aiter |
|---|---|---|---|---|---|---|---|
| b1_s4k | 1094.2 | 1082.6 | -1.06 % | 1095.9 | +0.16 % | 1066.0 | 1.0156 |
| b1_s8k | 1204.2 | 1223.8 | +1.63 % | 1239.3 | +2.91 % | 1204.8 | 1.0158 |
| b1_s16k | 1317.4 | 1323.7 | +0.48 % | 1305.2 | -0.93 % | 1267.2 | 1.0446 |
| b1_s32k | 1350.7 | 1366.6 | +1.18 % | 1345.3 | -0.40 % | 1295.8 | 1.0546 |
| b1_s64k | 1377.4 | 1400.5 | +1.67 % | 1382.5 | +0.37 % | 1305.2 | 1.0730 |
| b2_s4k | 1080.3 | 1074.6 | -0.53 % | 1083.9 | +0.33 % | 1043.8 | 1.0295 |
| b2_s16k | 1290.6 | 1305.3 | +1.14 % | 1299.4 | +0.69 % | 1267.6 | 1.0297 |

Geomean: r17 1244.12 / 1235.49 (mean 1239.81), g32 1251.24 / 1244.22 (mean 1247.73, +0.64 %),
g16 1246.90 / 1243.63 (mean 1245.26, +0.44 %). g32 is above r17 in both rotations (+0.57 %,
+0.71 %). Ratio vs aiter: r17 1.0272, g32 1.0374, g16 1.0321 (2-run means; aiter itself varied
between runs, e.g. b2_s4k 1043.8 in the g32 runs vs 1057.7 in the r17 runs, so use the same-session
r17 column for the decision).

Quick set, 3 rotations, no aiter, each package once in each position (`artifacts/ab_quick.txt`):

| shape | round 17 | g32 | g16 |
|---|---|---|---|
| b1_s4k | 1095.1 | 1085.0 (-0.92 %) | 1099.6 (+0.42 %) |
| b1_s16k | 1316.6 | 1326.2 (+0.73 %) | 1315.1 (-0.12 %) |
| b1_s64k | 1383.2 | 1410.5 (+1.97 %) | 1389.3 (+0.44 %) |
| b2_s4k | 1083.7 | 1078.6 (-0.47 %) | 1087.2 (+0.32 %) |
| geomean | 1212.50 | 1216.37 (+0.32 %) | 1215.69 (+0.26 %) |

Decision: ACCEPT (trade-off), config (32, 16, 8): best aggregate of the sweep (+0.64 % full,
+0.32 % quick, above r17 in all 5 runs); b1_s4k -1.06 % and b2_s4k -0.53 % regress, both < 5 %.

## Attribution

Sustained runs (`artifacts/clock_power_raw.txt`, 14 s each, order r17, g32, g16, g16, g32, r17):

| | B1 S64K ms / call | median clock (MHz) | B1 S16K ms / call | median clock (MHz) | PPT active |
|---|---|---|---|---|---|
| round 17 | 126.77 / 127.64 | ~2100 | 8.3722 / 8.3772 | ~2138 | 0.62-0.68 |
| g32 | 124.55 / 125.06 (-1.9 %) | ~2110 | 8.3205 / 8.3196 (-0.65 %) | ~2152 | 0.69-0.72 |
| g16 | 126.66 / 126.85 (-0.35 %) | ~2087 | 8.4197 / 8.4078 (+0.47 %) | ~2121 | 0.62-0.70 |

rocprofv3, one dispatch each (`artifacts/pmc_raw.txt`, logs `profiles/pmc_r18/`):

| | S16K L2 hit | S16K L2 miss | S16K EA read req | S64K L2 hit | S64K L2 miss | S64K EA read req | S16K / S64K GRBM_GUI_ACTIVE |
|---|---|---|---|---|---|---|---|
| round 17 | 0.786 | 7.92e7 | 7.28e7 | 0.860 | 7.68e8 | 7.43e8 | 1.392e8 / 2.048e9 |
| g32 | 0.812 | 6.98e7 (-12 %) | 6.33e7 (-13 %) | 0.856 | 7.96e8 (+3.6 %) | 7.70e8 (+3.7 %) | 1.388e8 / 2.026e9 |
| g16 | 0.821 | 6.62e7 (-16 %) | 5.98e7 (-18 %) | 0.862 | 7.59e8 (-1.2 %) | 7.34e8 (-1.2 %) | 1.367e8 / 2.015e9 |

Measured:
- At S4K the three packages run the same CTA order, yet g32 is 0.9-1.1 % (b1) and 0.5 % (b2) below
  r17 in all 5 runs while g16 is 0.2-0.4 % above it. At S8K g32 and g16 again share one order and
  g32 is 1.25 % below g16. So the g32 build pays about 1.3 % from code generation alone, unrelated
  to the order.
- ISA (`profiles/r18_isa/{r17,g32,g16}`): VGPR 256 and SGPR 62 in all three; VGPR spill 13 / 11 / 13;
  main loop 793 / 791 / 794 instructions with the same mix, its opcode order differing in 27 (r17 vs
  g16), 36 (r17 vs g32) and 13 (g16 vs g32) lines (a few `ds_read_b128`, `s_waitcnt`, and two
  MFMAs moved). Whole kernel 3669 / 3696 / 3651 instructions; g32 has +42 `v_mov_b32` and 5 fewer
  scratch loads outside the loop.
- L2 misses do not track speed. At S16K g16 has the fewest misses and fewest cycles but the lowest
  clock and the slowest sustained time; at S64K g32 has the most misses and is the fastest, with the
  highest clock. Every package is at the socket power limit 62-72 % of the time.

Hypothesis, not measured: g32's gain at S >= 16K is the 32-rank groups (one head's K/V per XCD at a
time), and it would be larger without the ~1.3 % code-generation cost seen at S4K / S8K; the
counters above cannot separate the two because the code-generation change also moves the clock.
The reverse reading (g32's long-S gain is its code generation, not its order) is not ruled out by
this data.

## Real-training transfer check
- Bucket: K1 (kernel-internal: CTA-to-(head, q-block) order)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +0.64 % forward geomean (full set, same-session rotation), +0.32 %
  quick set
- Estimated real-training gain: shape-dependent, the per-shape numbers above for the same (B, S):
  +0.5 % to +1.7 % at S 8K-64K, -1.1 % (B1) and -0.5 % (B2) at S4K. A training step at S4K would
  see the regression. The CTA order depends only on (B, H, S), which a training step has as well.
  The gain at long S came with a higher clock at the power limit; in a training step the forward
  shares the power budget with other kernels, and that clock was not measured (sustained-loop
  numbers only).
- Decision: ACCEPT-as-real (with the S4K trade-off above)

## Next direction

- Recover the S4K / S8K code-generation cost: the g16 build shows the same order can run 1.3 %
  faster. Candidates: compute the section bounds in a form closer to round 17's (two full maps and
  a select), or keep the per-level selects out of the loop's live range; measure with the same
  rotation, and compare the main-loop schedule against g16's.
- The kv-head0 probe bounds what L2 locality alone can give at B1 S16K: about 3.5 % of time. g32
  recovered 0.65 % of it there.
- Still open from round 17: PV sched hints for 32 shorter MFMAs; sharing the rebase-branch row-max
  trees.
