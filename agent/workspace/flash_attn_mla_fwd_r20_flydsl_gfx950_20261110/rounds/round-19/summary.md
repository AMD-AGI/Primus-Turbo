# Round 19 — ACCEPTED — 2026-11-11 12:32 (UTC+8)

## Hypothesis
ATT of round 18 on dsv3 (`profiles/att_r18/`, `wip/att_regions.py CSV ISA .LBBx_y` per-instruction
view): drain C10 (`.LBB0_61`) still takes 4.5-5.1K cycles per job on every wave. It issues the K
prefetch (12 DMAs), the 16 Q stage DMAs at 45-190 cycles each (VMEM queue backpressure) and the 8
middle gathers at 66-390 each. The epilogue's new `vmcnt(8)` costs < 12 cycles. Fakes against R18
(`profiles/fakes/fakes_probe_r19.log`, quick set): `nostage` (the 16 DMAs dropped) dsv3 +2.6 %,
dsv2lite +1.5 %, geomean +0.9 %; `noqc10` (gathers dropped too) dsv3 +3.3 %, geomean +1.9 %. Both
read stale LDS as Q, which can shift the lazy-rescale branch, so they may overstate.

The staged packs hold no registers until the epilogue, so the DMAs need not wait for C10. In C9
only the last wave has live QK work (the last KV tile is keys 192-255 of the block); the R18 trace
shows the other three waiting ~2.7K cycles at C9's barrier. If those waves issue every wave's Q
stage DMAs at the start of C9, C10 is left with the K prefetch and the gathers, and C10's existing
`vmcnt(0)` + barrier publish the stage to its owners.

## Single change
`CODEGEN_REV` 1057, delta `artifacts/r19_delta_vs_r18.patch`:
- Helper `stage_q_dma(h, q_block, batch)`: wave w < 3 DMAs its own stage tile (16 DMAs); waves 0
  and 1 also split the last wave's tile, 8 DMAs each (`Q_STAGE_LAST_SPLIT`). Wave-uniform
  `scf.if`s on `wave_id_uni`. Same rows, swizzle and per-DMA chunk pattern as round 18 (a DMA
  starts at a multiple of 4 rows, so its lanes' chunks depend only on its index mod 4).
  `prefetch_q_staged` keeps only the 8 middle gathers.
- Kernel: `stage_q_dma(next_job)` at the start of drain C9. The epilogue `vmcnt(8)` is gone (C10's
  `vmcnt(0)` and sync now cover the stage, including the cross-wave tile). `_job_coords` factored
  out of `_prefetch_q`.
- ISA at H=128 (`artifacts/r19_persistent_h128.s`): C8/C9 block 4 V + 16 Q DMAs, the shared-tile
  branch 8; C10 12 K DMAs + 8 gathers; no epilogue wait. VGPRs 492 -> 508, 0 spills; KV loop blocks
  rescheduled (±6 instructions each). Non-persistent (`wip/np19`, H=8): instruction-for-instruction
  identical to round 18.

## Results (full set, base = round-18 snapshot; two separate processes)

| shape | cand | base (R18) | cand / base (run 1) | cand / base (run 2) |
|---|---|---|---|---|
| dsv3 (B2 S4K H128) | 1162.8 | 1144.7 | 1.0158 | 1.0161 |
| dsv2lite (B8 S4K H16) | 1116.0 | 1102.4 | 1.0123 | 1.0130 |
| kimik2 (B4 S4K H64) | 1156.7 | 1146.1 | 1.0092 | 1.0101 |
| kimik3 (B2 S7K H96) | 1244.9 | 1240.4 | 1.0036 | 1.0042 |
| B1 S4K | 1147.7 | 1130.1 | 1.0156 | 1.0160 |
| B1 S8K | 1257.5 | 1249.3 | 1.0066 | 1.0070 |
| B1 S16K | 1298.2 | 1291.6 | 1.0051 | 1.0051 |
| B1 S32K | 1327.4 | 1323.3 | 1.0031 | 1.0052 |
| B1 S64K | 1354.7 | 1324.2 | 1.0230 | 1.0274 |
| B2 S4K | 1162.8 | 1144.4 | 1.0161 | 1.0155 |
| B2 S16K | 1315.9 | 1308.8 | 1.0054 | 1.0061 |

Geomean 1228.67 vs 1215.88 (+1.05 %, `artifacts/r19.csv`); repeat 1234.20 vs 1220.30 (+1.14 %,
`artifacts/r19_rep2.csv`). 11/11 PASS, bitwise identical to round 18 in both runs, every shape up
in both. Edge smoke (`artifacts/r19_smoke.log`) 8/8 bitwise identical.

## Decision
ACCEPTED per Rule 3: geomean up in both runs (+1.05 %, +1.14 %), no shape regressed. 1.222x aiter
(1005.24).

## Attribution
- ATT on dsv3 (`profiles/att_r19/`, per-wave `att_wave_jobs.py`): drain 477K -> 443K cycles per
  wave (~2.1K per job). C10 block 4.5-5.1K -> 2.2-2.8K on every wave; the DMA issue moved into the
  C9 waits of the three off-diagonal waves (their C9 barrier wait 2.7K -> 1.1-2.2K), the wave with
  the live QK issues none. Epilogue block ~3.25K -> ~3.0K (no wait).
- 4K shapes +0.9-1.6 %: that drain saving. dsv3 +1.6 % is ~60 % of the `nostage` bound.
- Long S +0.3-0.7 % (S8K-S32K) is more than the drain share there. The KV loop blocks were
  rescheduled at the new register allocation (loop ATT on dsv3 ~1 % fewer cycles); not separated.
- B1 S64K +2.3 % / +2.7 %: bistable as in rounds 14, 16 and 18; not attributed.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +1.05 % / +1.14 % combined-step (forward-only: combined step =
  forward).
- Estimated real-training gain: +0.9-1.6 % on 4K training shapes, +0.3-0.5 % at 7K-32K. Every
  persistent job after the first stages its Q this way regardless of the data; the off-diagonal
  waves' C9 slack exists in every full causal q block. Bitwise-neutral.
- Decision: ACCEPT-as-real.

## Next direction
C10 still issues 12 K DMAs and 8 gathers on every wave (2.2-2.8K cycles). K buffer 0 is last read
in C5, so the next job's K tile 0 could go out in an off-diagonal cluster too (C9 has room on waves
0-2; C5's QK is live on waves 2 and 3 only), with C10's `vmcnt(0)` covering it. The 8 middle
gathers need registers and stay. Separately, the C9 / C13 barrier imbalance (one live wave) is the
diagonal interleave noted in round 17.
