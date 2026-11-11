# Round 16 — ACCEPTED (trade-off) — 2026-11-11 11:05 (UTC+8)

## Hypothesis
`_store_final_o16x4` (QT4, PV 16x16x32) leaves every lane holding all 128 dv of its own q row, so
each of its 16 `buffer_store_dwordx4` per lane covers 64 q rows x 16 B: 64 cache lines per
instruction, 8x the lines a row-contiguous store touches. The wrong-result `nostore` fake (no O
stores at all) measured dsv3 1107.1 -> 1163.3 TFLOPS (+5.1 %) and B1 S16K +1.1 %
(`profiles/fakes/fakes_probe_r15.log`), so at 4K the store pattern clogs the vector-memory path
ahead of the next job's Q loads. Staging the tile through LDS and storing four whole rows (16 lanes
x 16 B each) per instruction keeps the same 16 stores per lane with 8x fewer lines per store.

## Single change
`CODEGEN_REV` 1054, delta `artifacts/r16_delta_vs_r15.patch`:
- Traits: `O_LDS_STAGE` (PV16 with 64 rows per wave), `O_STAGE_ROW_ELEMS` = 136 (272 B rows, 16 B
  off the 64-bank period), `LDS_O_STAGE_SIZE` = 4 waves x 64 rows x 136; `SharedStorage.o`.
- `_store_final_o16x4`: after the 4x4 permlane transpose each lane `ds_write_b128`s its row into the
  wave's own LDS tile; the wave then reads back 16 B per lane (lane = row r % 4 of 4, 16 B chunk of
  16) and issues 16 `buffer_store`s of 4 rows each. No barrier: each wave reads only its own tile,
  and a wave's DS ops complete in order. The store count per lane is unchanged (16), so round 15's
  job-top `vmcnt(o_stores_per_lane)` still holds.
- The staging lane terms come from a fresh `_job_tid()`, and the global row offset steps as a running
  sum; the "lds_o" alias scope is one-sided (the K / V accesses keep their round 15 metadata).
- ISA at H=128 (`artifacts/r16_persistent_h128.s`): 16 `ds_write_b128` + 16 `ds_read_b128` (+1088 B
  steps) + 16 `buffer_store_dwordx4`; VGPRs 497 -> 488, 0 spills, LDS 84736 -> 154368 B (one CTA
  per CU either way). The loop's LLVM IR is identical to round 15's; its QK halves (.LBB0_8 /
  .LBB0_12) are scheduled differently (42 / 43 differing lines), the P*V halves are identical.

Iterations inside the round (none benchmarked):
- Lane terms from the job's `self.lane` / `self.wave_id`: CSE with the job's own copies kept them
  live across the KV loop: VGPRs 511, 4 AGPR reads back in the loop, SGPR spill lanes in v255.
- `row_base + i * 4 * stride` for i < 16: LLVM hoisted the 16 products out of the job loop into
  SGPRs (the v255 spill lanes); a running sum leaves one hoisted step.
- "lds_o" in every K / V access's noalias list vs one-sided: ISA identical; kept one-sided.

## Results (full set, 2 rotations, base = round-15 snapshot; two separate processes)

| shape | cand | base (R15) | cand / base (run 1) | cand / base (run 2) |
|---|---|---|---|---|
| dsv3 (B2 S4K H128) | 1130.2 | 1107.9 | 1.0201 | 1.0220 |
| dsv2lite (B8 S4K H16) | 1097.2 | 1077.3 | 1.0185 | 1.0194 |
| kimik2 (B4 S4K H64) | 1128.4 | 1108.9 | 1.0176 | 1.0197 |
| kimik3 (B2 S7K H96) | 1235.8 | 1220.5 | 1.0125 | 1.0164 |
| B1 S4K | 1119.4 | 1095.0 | 1.0223 | 1.0233 |
| B1 S8K | 1251.9 | 1245.2 | 1.0054 | 1.0088 |
| B1 S16K | 1294.5 | 1294.8 | 0.9998 | 0.9994 |
| B1 S32K | 1324.1 | 1329.7 | 0.9958 | 0.9954 |
| B1 S64K | 1323.1 | 1338.6 | 0.9884 | 0.9866 |
| B2 S4K | 1129.7 | 1106.1 | 1.0213 | 1.0212 |
| B2 S16K | 1304.3 | 1304.6 | 0.9998 | 1.0004 |

Geomean 1209.42 vs 1198.45 (+0.92 %, `artifacts/r16.csv`); repeat 1210.00 vs 1197.83 (+1.02 %,
`artifacts/r16_rep2.csv`). 11/11 PASS, bitwise identical to round 15 in both runs. Edge smoke
(`artifacts/r16_smoke.log`, incl. partial q blocks S=300 / 777 / 1000) 8/8 bitwise identical.
Two earlier runs at `CODEGEN_REV` 1053 (`artifacts/rev1053_*`; same ISA as 1054) gave +0.91 % /
+0.98 % with the same per-shape pattern.

Kept one-job-per-CTA path (`persistent=False`, `wip/np16` vs `wip/np15`; this path changes too):
smoke 6/6 bitwise, full set +1.55 % geomean (1173.52 vs 1155.61, `artifacts/np16.csv`), every shape
up: 4K +1.7-2.8 %, S8K +1.5 %, S16K-64K +0.4-0.7 %. VGPRs 484 -> 486 at H=8.

## Decision
ACCEPTED (trade-off) per Rule 3: geomean up in both runs (+0.92 %, +1.02 %), no shape regressed by
5 % or more; B1 S64K -1.2 % / -1.3 % and B1 S32K -0.4 % / -0.5 % in both runs. 1.203x aiter
(1005.24).

## Attribution
- Short jobs: +1.8-2.3 % on every 4K shape, about 40 % of the `nostore` bound (+5.1 % on dsv3). The
  rest is the stores themselves (16 KB of O per wave) plus the LDS round trip (~0.3 us per job).
- Long S, persistent only: the one-job-per-CTA kernel, with the same epilogue, gains +0.4 % at S64K,
  so the staged store is not a long-job cost. The persistent H=128 binary's loop changed schedule
  although its LLVM IR is identical: in the QK halves the V head `ds_read_b128`s after the cluster
  sync moved behind two MFMAs and an exp moved up, with one more `lgkmcnt` wait (wait 22 -> 23 in
  .LBB0_8). The backend's IGroupLP fill of the `sched_group_barrier` groups depends on the
  live-through register pressure, which dropped 497 -> 488.
- That schedule is not clearly slower when the GPU is power-limited. Sustained 10 s loops
  (`profiles/power_r16.log`, R15 / R16 alternated twice): B1 S64K 1307.3 / 1311.2 then 1325.0 /
  1320.7 TFLOPS (+0.3 %, -0.3 %), with R16 drawing 15-25 W less (1301-1308 vs 1322-1331 W) at the same
  clock and more TFLOPS per MHz (0.738 / 0.735 vs 0.735 / 0.720); B1 S16K equal (1291.2 / 1290.4,
  1291.0 / 1291.3). `ab_bench` times S64K in ~1.7 s windows (3 calls of 133 ms per rep), where the
  GPU is not yet at its power steady state, and that is where the -1.3 % reproduces. Treat it as a
  cycle-regime cost of the new loop schedule, not a loss in sustained runs. Not measured with
  counters.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +0.92 % / +1.02 % combined-step (forward-only: combined step =
  forward).
- Estimated real-training gain: ~+2 % on 4K-7K training shapes (dsv3 / dsv2lite / kimik2 / kimik3);
  B1 S64K between -1.2 % (short windows) and 0 % (sustained, power-limited). Every forward writes
  its O tile once, so the saving is per job and does not depend on the benchmark loop.
  Bitwise-neutral.
- Decision: ACCEPT-as-real.

## Next direction
The S64K short-window cost does not show up in sustained runs, so pinning the QK-half schedule is
not worth a round yet. The 4K shapes still trail long S by ~15 %, and per job the LPT model charges
~8 tiles of fixed cost. The largest identified part is the causal diagonal: waves own contiguous
64-row bands, so in the last diagonal tiles wave 3 does all the work while waves 0-2 wait at the
cluster barriers (~4.5 % of dsv3 in the R15 ATT). Interleaving the 16-row q tiles across waves
(wave w gets q tiles 4j + w) gives every wave 4, 3, 2, 1 live q tiles in diagonal tiles 0-3,
known at trace time. LDS is now at 154368 / 163840 B: any later LDS growth must alias the O tile
onto the K / V buffers.
