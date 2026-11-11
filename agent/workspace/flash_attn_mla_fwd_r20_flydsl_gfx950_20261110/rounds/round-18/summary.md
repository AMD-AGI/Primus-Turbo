# Round 18 — ACCEPTED — 2026-11-11 12:15 (UTC+8)

## Hypothesis
Before the round, ATT on dsv3 (`profiles/att_r17/stats_r17_dsv3.csv`, per-wave JSONs, tools
`wip/att_regions.py` / `wip/att_wave_jobs.py`) put the R17 drain at ~31K cycles per job per wave
(22.7 % of dsv3 time). Its largest single block is C10 (`.LBB0_60`): 5.8-7K cycles, almost all of it
issuing the next job's 24 Q gathers. Each `buffer_load_dwordx4` there has 16 lanes per quarter-wave
reading 16 B from 16 different Q rows (~98 KB row stride), so one load costs 64 line accesses and the
issue is TA-throughput bound. HipKittens' gfx950 attention (`dev/mi455x_v4`,
`hk_attn_fwd_d128_gfx950.cu`) stages all global tiles with coalesced global->LDS loads and reads
the MFMA layout back from LDS. Fakes (`profiles/fakes/fakes_probe_r18.log`, wrong-result, quick set)
bound the Q fetch pattern: `coalq` (same 24 loads, 1 KB contiguous each) +2.8 % dsv3, +1.9 %
dsv2lite, +1.5 % geomean; `warmq` (L2-hot Q) +0.8 %, `warmqh` (MALL-hot) +0.9 %. So the cost is the
access pattern, not the miss.

The paired k-steps 0-3 read d blocks 0 and 2 (lines 0 and 2 of every 384 B row); the d mapping is
fixed by K's conflict-free `ds_read_b128`. Those 16 of 24 packs (256 B per row, 16 KB per wave) fit
the wave's O stage tile (64 x 272 B), which is idle from the previous epilogue's read back to this
job's O staging. LDS DMA writes 1 KB contiguous per instruction, so 16 DMAs per wave fetch them with
8 line accesses each instead of 64.

## Single change
`CODEGEN_REV` 1056, delta `artifacts/r18_delta_vs_r17.patch`:
- `traits.Q_LDS_STAGE` = persistent, O stage, 16x16x32 QK and D_QK 192.
- Helper `prefetch_q_staged(h, q_block, batch)` (replaces `prefetch_q_raw` at C10 when the flag
  is set): 16 `buffer_load_dwordx4 ... lds` per wave into its O stage tile, one stage row per Q row
  (256 B), DMA i filling rows 4i..4i+3. Rotation swizzle: chunk p of row r holds part-A chunk
  (p - r) % 16, so the read back has 16 distinct chunks per 16 lanes (conflict-free, plain
  arithmetic, no XOR). Row offsets as a running sum. The 8 middle-block packs (k-steps 4, 5) stay
  gathers (`load_q_raw(..., keep=)`), the 16 staged ones are None in the carried list.
- Helper `read_q_stage(q_raw)`: 16 `ds_read_b128` (4 base addresses, qt as 4096 B immediates),
  bitcast to bf16 packs, filling the None slots.
- Kernel: in the epilogue after `_s_barrier()`, before the LSE / O stores, `vmcnt(8)` (the DMAs
  precede the 8 gathers; vmcnt retires in order), then `read_q_stage`. No barrier: each wave reads
  only its own tile, and its O stage writes come after these reads in DS order. The first job's Q is
  still the 24 gathers before the loop; job-top `vmcnt(16)` unchanged (8 gathers + LSE).
- `_q_job_view` factored out of `prefetch_q_raw`; `lds_o_base_idx` kept for the M0 base.
- ISA at H=128 (`artifacts/r18_persistent_h128.s`): the prefetch block has 12 K + 16 Q LDS DMAs and
  8 register gathers (R17: 12 + 0 and 24). Epilogue: `s_barrier`, `s_waitcnt vmcnt(8)`, 16
  `ds_read_b128`, then the O stage. VGPRs 490 -> 492, AGPRs 234 -> 236, 0 spills; AGPR moves overall
  slightly fewer. Non-persistent (`wip/np18`, H=8): instruction-for-instruction identical to R17.

## Results (full set, base = round-17 snapshot; separate processes)

| shape | cand | base (R17) | cand / base (run 1, rot 1) | cand / base (run 2, rot 2) | run 3 (quick, rot 3) |
|---|---|---|---|---|---|
| dsv3 (B2 S4K H128) | 1142.1 | 1133.7 | 1.0074 | 1.0051 | 1.0053 |
| dsv2lite (B8 S4K H16) | 1104.0 | 1101.1 | 1.0026 | 1.0023 | 1.0029 |
| kimik2 (B4 S4K H64) | 1146.6 | 1132.6 | 1.0124 | 1.0105 | - |
| kimik3 (B2 S7K H96) | 1242.2 | 1235.4 | 1.0055 | 1.0064 | 1.0058 |
| B1 S4K | 1130.6 | 1123.8 | 1.0061 | 1.0050 | - |
| B1 S8K | 1249.3 | 1247.1 | 1.0018 | 1.0022 | - |
| B1 S16K | 1293.1 | 1289.7 | 1.0026 | 1.0023 | 1.0025 |
| B1 S32K | 1321.7 | 1319.9 | 1.0014 | 1.0062 | - |
| B1 S64K | 1327.0 | 1324.4 | 1.0020 | 1.0273 | 1.0279 |
| B2 S4K | 1143.5 | 1138.1 | 1.0047 | 1.0051 | - |
| B2 S16K | 1307.2 | 1302.5 | 1.0036 | 1.0031 | - |

Geomean 1216.05 vs 1210.55 (+0.45 %, `artifacts/r18.csv`); repeat 1224.92 vs 1216.60 (+0.68 %,
`artifacts/r18_rep2.csv`); quick run 3 +0.88 % (`artifacts/ab_quick_rep3.log`). 11/11 PASS,
bitwise identical to round 17 in every run, no shape below 1.0. Edge smoke
(`artifacts/r18_smoke.log`, incl. S=300, S=777 partial blocks) 8/8 bitwise identical.

## Decision
ACCEPTED per Rule 3: geomean up in both full runs (+0.45 %, +0.68 %), no shape regressed. 1.210x
aiter (1005.24).

## Attribution
- 4K shapes +0.2-1.2 %, kimik2 the most (+1.0-1.2 %): C10 issues 8 scattered gathers instead of
  24 (the R17 ATT put that issue at ~5.8-7K cycles per job); line accesses per job 1536 -> 640
  (16 x 8 + 8 x 64).
- That is ~1/4 of the `coalq` bound on dsv3 (+0.5-0.7 % vs +2.8 %) for ~2/3 of its line accesses
  removed. Not measured (no ATT run this round); candidates: the 8 middle-block gathers are the last
  and slowest loads of C10, and the new epilogue `vmcnt(8)` can expose the DMA's tail before the O
  store.
- B1 S64K: +0.2 % / +2.7 % / +2.8 %, with base steady at 1324-1331 and cand at 1327 or 1366. All
  three Q fakes also showed S64K +1.6-2.4 %. S64K has been bistable on binary changes before
  (rounds 14, 16); not attributed.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +0.45 % / +0.68 % combined-step (forward-only: combined step =
  forward).
- Estimated real-training gain: +0.3-1.0 % on 4K training shapes, ~+0.2 % at 7K+. Every persistent
  job after the first fetches its Q this way, and the access pattern is the same for any Q tensor of
  this layout. Bitwise-neutral.
- Decision: ACCEPT-as-real.

## Next direction
The staged packs need no registers until the epilogue read back, and the O stage tile is free for
the whole job, so the Q DMA does not have to wait for C10 (R17's placement was set by the current
Q's register liveness). Issuing it at the drain start, or in the last KV iterations, should hide its
latency and the epilogue `vmcnt(8)` wait, and spread the TA work away from C10. The 8 middle
gathers stay at C10 (they need registers). Check after: ATT of C10 and the epilogue on dsv3.
