# Round 17 — ACCEPTED

- Time: 2026-11-09 15:25 (UTC+8). Base: round 16. Machine: smci355-ccs-aus-n02-25, HIP device 0.
- Hypothesis: round 16 left two things on the table. (1) The 16 `v_permlane16_swap` per tile that
  convert the raw 16x16 QK accumulators to the 32x32 score layout: a wrong-result variant without
  them measured +2.8 % quick (round 16 `artifacts/ab_quick_upper_bounds.txt`). (2) PV (40 % of the
  MFMA FLOPs at d_qk 192 / d_v 128) still ran on v_mfma_f32_32x32x16_bf16, the hotter atom. Running
  PV on v_mfma_f32_16x16x32_bf16 straight off the raw QK layout removes (1) and moves (2).
- Change (single variable: the PV MFMA variant, plus the P / V / O layouts it needs):
  - Trait `PV_MFMA16` (defaults to `QK_MFMA16`; needs the MFMA row sum and d_v 128). `qk_scores`
    becomes the identity: the raw half (32 kv) holds element 4 (2 kt + qt) + e at
    kv = 32 h + 16 kt + [0, 8, 4, 12][lane // 16] + e, q = lane % 16 + 16 qt, which is the PV
    16x16x32 B operand as is (pack for (h, qt) = elements [4 qt .. 4 qt + 3, 8 + 4 qt .. 8 + 4 qt + 3]).
  - O^T = V^T P^T: 4 k-steps per tile (one per kv half x 2 dv groups), 8 dv tiles x 2 q tiles,
    32 PV MFMAs per tile; `v_o[dc]` holds (dc, qt0), (dc, qt1), (dc + 4, qt0), (dc + 4, qt1).
  - Row scalars stay one q per lane (q = lane % 32): the row max is a 16-value tree per q tile,
    one `permlane16_swap` to pair (qt0, qt1), then the existing permlane32 pair reduce; scale and
    rescale vectors are rebuilt per q tile with one swap. Row sum: ones-MFMA per q tile into an
    8-wide l (`finish_row_sum` picks element 0 or 4 by lane // 16 parity).
  - V: `ds_read_b64_tr_b16` reads the A operand from the existing V LDS rows; odd LDS rows store
    16-byte chunk m at m ^ 2 (DMA per-lane address, `_v_pv16_chunk`) so the two k groups of a
    read land on different banks; the read uses two lane bases (even / odd dv tile).
  - Causal / sequence-pad masks rewritten for the raw layout; O store: 2 `permlane16_swap` per
    tile pair, then one 16-byte store per lane.
  - `CODEGEN_REV` 7 -> 8. Snapshot: `kernel_snapshot/`. The swp8 kernel is untouched (trait off).

## Correctness

Full set 7/7 PASS in both full runs: O rel vs fp32 0.0016-0.0020, LSE max 0.0046-0.0051, same as
round 16 to 4 digits. Quick set PASS on every rotation run (12 / 12).

## Result (full set, same session, r16 / r17 alternated twice; `artifacts/ab_full_*.csv`)

| shape | round-17 (2-run mean) | round-16 same session | delta | aiter (r17 runs) | ratio vs aiter |
|---|---|---|---|---|---|
| b1_s4k | 1090.8 | 1085.9 | +0.46 % | 1066.0 | 1.0233 |
| b1_s8k | 1202.4 | 1193.0 | +0.79 % | 1205.2 | 0.9977 |
| b1_s16k | 1311.3 | 1290.0 | +1.65 % | 1270.6 | 1.0320 |
| b1_s32k | 1345.3 | 1324.8 | +1.55 % | 1297.2 | 1.0371 |
| b1_s64k | 1376.7 | 1358.9 | +1.31 % | 1307.7 | 1.0528 |
| b2_s4k | 1081.3 | 1069.8 | +1.07 % | 1058.2 | 1.0218 |
| b2_s16k | 1291.6 | 1269.7 | +1.73 % | 1268.3 | 1.0184 |

Geomean 1238.67 / 1236.69 (mean 1237.68) vs round 16 1225.46 / 1220.01 (mean 1222.74): +1.22 %.
Ratio vs aiter 1.0254 / 1.0267 (round 16 in the same session: 1.0124 / 1.0109). Against round 16's
recorded 1217.15 (earlier session): +1.69 %. Quick set, 3 rotations (`artifacts/ab_quick_r16_r17.txt`):
1212.4 vs 1196.5 (+1.33 %; b1_s4k +1.0 %, b1_s16k +2.0 %, b1_s64k +1.3 %, b2_s4k +1.0 %).
`benchmark_full.csv` / `full.log` are the first round-17 run.

Decision: ACCEPT (aggregate up in both rotations; every shape up, smallest +0.46 %).

## Attribution (B1 S16K)

| | round 16 | round 17 |
|---|---|---|
| sustained GFX clock (MHz, 2 x 14 s) | 2003-2048 (median 2041) | 2127-2141 (median 2135) |
| socket power (W) | 1326-1363 | 1347-1363 |
| ms / call (sustained) | 8.5412 / 8.5258 | 8.3750 / 8.3859 |
| GRBM_GUI_ACTIVE (rocprofv3) | 1.330e8 | 1.386e8 (+4.2 %) |
| SQ_LDS_BANK_CONFLICT | 0 | 0 |
| SQ_INSTS_LDS | 4.741e8 | 4.741e8 |
| SQ_VALU_MFMA_BUSY_CYCLES | 1.136e10 | 1.136e10 |
| SQ_INSTS_VALU | 1.351e9 | 1.402e9 |

Measured: the same pattern as round 16, once more. The clock rose ~4.6 % at the same socket power
(1.33-1.36 kW sampled; the socket power limit PPT is active 68 % of the time, measured after the
round in `logs/optimize.md`, "Post-round-17 headroom analysis"), time per call fell 1.8 %, and
cycles per call rose (clock x time +2.7 %, GRBM +4.2 %). MFMA busy cycles are identical, as expected for the same FLOPs: 64 x 16x16x32 replace
32 x 32x32x16 per 2 tiles. LDS instruction count is identical and the V swizzle keeps conflicts at 0.

Main loop per 2 tiles vs round 16 (`artifacts/loop_hist_r1{6,7}.txt`): +64 16x16x32 / -32 32x32x16
MFMA, -26 `v_permlane16_swap` (34 -> 8; the 8 left sit in the rarely taken rebase branch), +30
v_max / v_max3, +15 v_add_u32, 835 -> 892 instructions. The v_max growth is mostly the rebase
branch: without the score bound (head_dim 192) the per-tile trigger is still a 32-value tree that
the new owned-layout row max does not share, so the branch builds the two 16-value trees again.
Whole kernel: `v_permlane16_swap` 101 -> 39, `v_mov_b32` 420 -> 349, VGPR spills 9 -> 13, all
outside the main loop.

Hypothesis, not measured: the extra cycles come from issuing twice as many (shorter) PV MFMAs,
whose 8 per k-step are independent and reuse an accumulator only two k-steps later, so the
dependency chain is not the suspect; the scheduler interleaving around them and the address adds
of the two V read bases are. The removed swaps did not cover them, and the net gain is the clock.

## Real-training transfer check
- Bucket: K1 (kernel-internal: MFMA variant, register layouts of P / V / O, LDS swizzle)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +1.22 % forward geomean (same-session rotation)
- Estimated real-training gain: the per-shape numbers above for the same (B, S): +1.3 % to +1.7 %
  at S >= 16K, +0.5 % to +1.1 % at S 4K-8K. The gain is clock at the same power; in a training
  step the forward shares the power / clock budget with neighbouring kernels, so the clock it gets
  there was not measured (sustained-loop numbers only). Calls are 8.4 ms (B1 S16K) and longer at larger S.
- Decision: ACCEPT-as-real

## Next direction

- Both GEMMs are now on 16x16x32. Chaining the two rounds' same-session GRBM ratios (+2.9 %,
  +4.2 %) gives ~7 % more cycles per call than round 15, paid for by ~10 % clock (1927-1933 MHz in
  the round-16 session, 2127-2141 MHz here); reclaiming cycles without losing the clock is the next
  lever.
- PV sched hints were left as written for 32x32x16 (16 MFMAs per tile); now 32 shorter MFMAs.
  Retune them (the round-16 QK retune lost 0.3 %, so measure rather than assume).
- The rebase branch duplicates the row-max trees (static, rarely executed); sharing the two
  16-value trees with the trigger would cost one v_max on the hot path and shrink the branch.
- `ds_read_b64_tr_b16` LDS phase rule not measured; the V swizzle assumed ds_read_b64-like
  32-lane phases. Conflicts are 0 either way.
- Still just below aiter: b1_s8k (0.998).
