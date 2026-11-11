# Round 17 — ACCEPTED — 2026-11-11 11:25 (UTC+8)

## Hypothesis
Each persistent job still starts with its 24 Q loads (16 B per lane each) at the job top and waits
for them before QK0, so one Q round trip per job is on the critical path. The previous job's last QK
is in drain C9, after which its Q (in AGPRs) is dead, and the next job is already known there (round
11 prefetches its K in C10). Loading the next job's Q in C10, next to that K prefetch, and carrying
the raw packs over the job back edge puts the round trip under C11-C13, the normalization and the O
store. Probe before the round (`profiles/fakes/fakes_probe_r16.log`, wrong-result `noq`: Q loads
replaced by a constant): dsv3 +8.0 %, dsv2lite +6.2 %, kimik3 +4.5 %, B1 S16K / S64K +3.2-3.5 %. That
is an upper bound: a constant Q also removes the scale VALU and lowers MFMA switching power (the
long-S part of it).

## Single change
`CODEGEN_REV` 1055, delta `artifacts/r17_delta_vs_r16.patch`:
- Helper: `load_all` is split into `bind_q_rows` (the lane's q row / the wave's first row),
  `load_q_raw(q_div, offset, wave_q_offset)` (the 24 loads) and `assemble_q`; `load_all` composes
  them unchanged. `prefetch_q_raw(h, q_block, batch)` builds the next job's Q batch view and offset
  from a fresh opaque tid, like `prefetch_k_tile` (persistent only, one q head per WG).
- Kernel: the persistent job loop carries the 24 raw Q packs as iter_args. The first job's Q is
  loaded before the loop with its K prefetch, under the existing `s_waitcnt 0`. `_main_body` takes
  `q_raw` (skipping the loads) and returns the next job's packs, loaded in drain C10 right after
  `_prefetch_k(next_job)`. The last job re-loads its own Q (unused), as for K.
- vmcnt: the issue order at the end of a job is K prefetch, Q loads, LSE store, O stores, so round
  15's job-top `vmcnt(16)` now also retires the Q loads.
- ISA at H=128 (`artifacts/r17_persistent_h128.s`): no `vmcnt` wait between the C10 Q loads and the
  job-top `vmcnt(16)`. The loads land half in VGPR quads, half directly in AGPR quads (a[130:169]),
  untouched until the scale reads them (43 `v_accvgpr_read`). Job-top block: 28 -> 4 VMEM
  issues, waits 23 -> 15. VGPRs 488 -> 490, 0 spills. KV loop: same instruction counts per block,
  16-32 lines reordered. The one-job-per-CTA path (`wip/np17`, H=8) is instruction-for-instruction
  identical to round 16's.

## Results (full set, 2 rotations, base = round-16 snapshot; two separate processes)

| shape | cand | base (R16) | cand / base (run 1) | cand / base (run 2) |
|---|---|---|---|---|
| dsv3 (B2 S4K H128) | 1135.4 | 1127.8 | 1.0067 | 1.0072 |
| dsv2lite (B8 S4K H16) | 1100.9 | 1095.3 | 1.0051 | 1.0055 |
| kimik2 (B4 S4K H64) | 1135.1 | 1127.8 | 1.0065 | 1.0063 |
| kimik3 (B2 S7K H96) | 1237.4 | 1236.6 | 1.0006 | 1.0010 |
| B1 S4K | 1126.9 | 1121.7 | 1.0046 | 1.0043 |
| B1 S8K | 1251.8 | 1251.5 | 1.0002 | 1.0006 |
| B1 S16K | 1296.3 | 1293.8 | 1.0019 | 1.0020 |
| B1 S32K | 1327.6 | 1324.7 | 1.0022 | 1.0024 |
| B1 S64K | 1329.8 | 1327.0 | 1.0021 | 1.0001 |
| B2 S4K | 1137.6 | 1130.4 | 1.0064 | 1.0058 |
| B2 S16K | 1309.1 | 1304.7 | 1.0034 | 1.0030 |

Geomean 1213.99 vs 1209.62 (+0.36 %, `artifacts/r17.csv`); repeat 1215.42 vs 1211.24 (+0.35 %,
`artifacts/r17_rep2.csv`). 11/11 PASS, bitwise identical to round 16 in both runs, no shape below
1.0 in either run. Edge smoke (`artifacts/r17_smoke.log`) 8/8 bitwise identical.

## Decision
ACCEPTED per Rule 3: geomean up in both runs (+0.36 %, +0.35 %), no shape regressed. 1.208x aiter
(1005.24).

## Attribution
- 4K shapes +0.4-0.7 %: ~0.5 us per dsv3 job (~77 us per job, 16 jobs per CU), about 1K cycles.
  That is the Q round trip that was still exposed at the job top; most of it was already overlapped
  by the job-top DMA issue and the barrier.
- The `noq` bound (+8 % on dsv3, +3.2 % at S64K, where the per-job share is ~0.2 %) is mostly not
  latency. A constant Q also drops the 96-element scale per job and makes every QK MFMA operand
  constant (less switching energy on a power-limited part). Wrong-result fakes that change operand
  data are not latency bounds.
- KV loop moves (+0.2 % at S16K-64K in both runs) are within the per-block reorder seen in every
  round; not attributed.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +0.36 % / +0.35 % combined-step (forward-only: combined step =
  forward).
- Estimated real-training gain: +0.4-0.7 % on 4K training shapes, ~0 at 7K+. Every job loads its Q
  once, so this applies to every forward call; Q is read in the previous job's drain, which in a
  training step sees the same tensors. Bitwise-neutral.
- Decision: ACCEPT-as-real.

## Next direction
Per-job costs left at 4K: the diagonal barrier imbalance (balanced-work fakes: +1.8 % with every
wave at wave 2's work, +3.7 % at wave 1's; an interleave of the 16-row q tiles would sit between),
and the prologue's K1 / V0 waits. The interleave needs per-q-tile wave-uniform liveness in the drain
QK / P*V (dead q tiles are trace-time known only for full q blocks), so it is a larger change. Check
first how much of the drain is barrier wait now (ATT on dsv3).
