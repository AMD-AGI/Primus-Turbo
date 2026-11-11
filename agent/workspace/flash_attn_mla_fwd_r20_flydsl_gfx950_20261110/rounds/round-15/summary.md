# Round 15 — ACCEPTED (trade-off) — 2026-11-11 10:15 (UTC+8)

## Hypothesis
The persistent job top waits `s_waitcnt vmcnt(0)` for the K prefetch, and with it for the previous
job's 16 O stores (and the LSE store), which were issued after the prefetch. vmcnt retires in order,
so `vmcnt(<O stores per lane>)` retires the prefetch and the LSE store and leaves the O stores in
flight while this job's Q loads issue. The first Q use still waits for the stores, but the Q load
latency then overlaps the store drain instead of following it: up to one memory round trip per job,
~+0.5 % at 4K if the stores are on the critical path.

## Single change
`CODEGEN_REV` 1052, delta `artifacts/r15_delta_vs_r14.patch`:
- Helper: `o_stores_per_lane`, the number of 16-byte `buffer_store`s `store_final_o` issues per lane
  on each path (`D_CHUNKS * 4` for QT4 PV16, else `D_CHUNKS * 2`; 16 here).
- Kernel: the persistent job top waits `lgkmcnt(0)` + `vmcnt(o_stores_per_lane)` instead of
  `s_waitcnt 0`; the first job, whose prefetch has no stores behind it, gets `s_waitcnt 0` before
  the job loop. The one-job-per-CTA path keeps `s_waitcnt 0`.
- ISA at H=128 (`artifacts/r15_persistent_h128.s`): `s_waitcnt vmcnt(16) lgkmcnt(0)` at the job
  top, `vmcnt(0)` before the loop, 16 O stores in the loop latch; the KV loop is identical to round
  14's. The only other difference is the order of a few Q-scale VALU instructions in the prologue
  (37 differing lines in total). VGPRs 497, 0 spills.

## Results (full set, 2 rotations, base = round-14 snapshot; two separate processes)

| shape | cand | base (R14) | cand / base (run 1) | cand / base (run 2) |
|---|---|---|---|---|
| dsv3 (B2 S4K H128) | 1109.1 | 1110.2 | 0.9990 | 0.9986 |
| dsv2lite (B8 S4K H16) | 1077.4 | 1075.0 | 1.0022 | 1.0023 |
| kimik2 (B4 S4K H64) | 1108.8 | 1103.7 | 1.0046 | 1.0043 |
| kimik3 (B2 S7K H96) | 1222.5 | 1222.6 | 0.9999 | 0.9998 |
| B1 S4K | 1095.8 | 1097.2 | 0.9987 | 0.9985 |
| B1 S8K | 1245.6 | 1246.1 | 0.9996 | 0.9994 |
| B1 S16K | 1293.6 | 1291.0 | 1.0020 | 1.0020 |
| B1 S32K | 1328.7 | 1326.9 | 1.0014 | 1.0011 |
| B1 S64K | 1335.2 | 1329.7 | 1.0041 | 1.0079 |
| B2 S4K | 1105.9 | 1107.9 | 0.9982 | 0.9980 |
| B2 S16K | 1303.5 | 1303.0 | 1.0004 | 0.9995 |

Geomean 1198.28 vs 1197.18 (+0.09 %, `artifacts/r15.csv`); repeat 1199.16 vs 1197.92 (+0.10 %,
`artifacts/r15_rep2.csv`). 11/11 PASS, bitwise identical to round 14 in both runs; every shape
except B1 S64K repeats within 0.1 % between runs. Edge smoke (`artifacts/r15_smoke.log`) 8/8 bitwise
identical.

Kept one-job-per-CTA path (`persistent=False`, `wip/np15` vs `wip/np14`): H=8 ISA identical
instruction for instruction, so no benchmark needed.

## Decision
ACCEPTED (trade-off) per Rule 3: geomean up in both runs (+0.09 %, +0.10 %), no shape regressed by
5 % or more; the H=128 4K shapes (dsv3, B1 S4K, B2 S4K) are -0.10 to -0.20 % in both runs.
1.192x aiter (1005.24).

## Attribution
- The gain is far below the estimate, and its sign depends on the binary: H=16 +0.2 %, H=64 +0.4 %,
  H=96 0 %, H=128 -0.15 % at 4K but +0.1-0.2 % at 16K-32K. The loops are identical, so this is the
  job boundary. With ~80 us jobs at 4K, a full store round trip would be ~1 %, so the O stores were
  mostly not on the critical path: by the time a wave reaches the job top (after the LSE math and
  the schedule load), most of its stores have completed.
- Why H=128 at 4K loses 0.15 % is not measured. Candidates: the Q loads now enter the memory
  pipeline behind the stores of all four waves at once (more contention on the first Q packs), or
  the prologue's VALU reorder. No counter run; the effect is ~100 ns per job.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +0.09 % / +0.10 % combined-step (forward-only: combined step =
  forward).
- Estimated real-training gain: ~+0.1 % on the validation mix; dsv3-like H=128 4K shapes -0.15 %.
  Bitwise-neutral, independent of tensor identity and data.
- Decision: ACCEPT-as-real.

## Next direction
The job boundary is now worth ~0.1-0.3 % per change; the remaining prologue cost is the Q round
trip itself (24 loads, first use after the barrier). The next job's Q could be loaded in the drain
after the last QK (Q's AGPRs are free from then on) with the K prefetch, ahead of the O stores. Bigger
levers are in the KV loop: with 15 VGPRs free, revisit what round 8 lost to register pressure
(carrying the C1 K head over the back edge), and profile the loop's issue mix (MFMA vs exp vs
ds_read) at long S to see what holds it at ~53 % of MFMA peak.
