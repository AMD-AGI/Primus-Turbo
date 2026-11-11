# Round 11 — ACCEPTED — 2026-11-11 09:30 (UTC+8)

## Hypothesis
Each persistent job still starts with three serialized memory round trips: K tile 0 (overlapping the
previous job's O-store drain), the Q load, then K tile 1 before the first QK. In the persistent loop
split_t0 is always 0, so the next job's first two K tiles depend only on its (head, batch). Issuing
them in the current job's drain, once both K buffers are free, should take K1 (and the K0 part not
under the store drain) off the prologue: ~+1 % at 4K, ~0 at long S.

## Single change
`CODEGEN_REV` 1048, delta `artifacts/r11_delta_vs_r10.patch`:
- Kernel: drain C10, right after `_waitcnt_vm_n(0)` (K buf0 last read in C5, buf1 in C9), issues the
  next job's K tiles 0 / 1 into buf0 / buf1; the prologue skips its own two K DMAs. Job 0's tiles are
  issued before the loop; the last job re-fetches its own tiles (`select(k + 1 < n_jobs)`) instead of
  reading past the schedule row; `s_waitcnt 0` after the loop.
- Helper: `prefetch_k_tile(h, b, tile, buf)`; the lane terms move to `_bind_lanes()` (fresh opaque
  tid, called by `bind_item` and the prefetch); `_view` / `_batch_view` become methods taking
  `batch_idx` explicitly.
- ISA (`artifacts/r11_persistent.s`, local only): 12 K DMAs in C10 after `vmcnt(0)`, no vm wait until
  the next prologue's `s_waitcnt vmcnt(0)` (which now covers the O stores and the prefetch together).
  Spill slots 9, scratch 40 B (same as round 10).

## Results (full set, 2 rotations, base = round-10 snapshot; two separate processes)

| shape | cand | base (R10) | cand / base (run 1) | cand / base (run 2) |
|---|---|---|---|---|
| dsv3 | 1072.8 | 1071.5 | 1.0012 | 1.0019 |
| dsv2lite | 1048.1 | 1049.1 | 0.9990 | 0.9987 |
| kimik2 | 1072.3 | 1070.1 | 1.0021 | 1.0033 |
| kimik3 | 1198.6 | 1184.4 | 1.0120 | 1.0111 |
| B1 S4K | 1059.0 | 1057.1 | 1.0018 | 1.0015 |
| B1 S8K | 1210.3 | 1212.7 | 0.9980 | 0.9983 |
| B1 S16K | 1277.5 | 1273.8 | 1.0029 | 1.0019 |
| B1 S32K | 1316.5 | 1312.4 | 1.0031 | 1.0028 |
| B1 S64K | 1342.0 | 1318.6 | 1.0177 | 1.0158 |
| B2 S4K | 1074.3 | 1073.3 | 1.0009 | 1.0011 |
| B2 S16K | 1279.2 | 1283.6 | 0.9966 | 0.9963 |

Geomean 1172.27 vs 1168.53 (+0.32 %, `artifacts/r11.csv`); repeat 1172.39 vs 1168.95 (+0.29 %,
`artifacts/r11_rep2.csv`). 11/11 PASS, bitwise identical to round 10 in both runs; per-shape deltas
repeat within 0.15 %. Edge smoke (`artifacts/r11_smoke.log`) 8/8 bitwise identical.

Kept one-job-per-CTA path (`persistent=False`, `wip/np11` vs `wip/np10`): ISA identical instruction
for instruction (7905 instructions, 8 spill slots both), so no benchmark needed.

## Decision
ACCEPTED (geomean +0.3 %, reproduced in a second process). 1.166x aiter (1005.24).

## Analysis
- The prefetch itself converts far less than estimated: 4K shapes +0.1-0.2 % (estimate ~+1 %).
  Likely reasons (not measured separately): K tiles 0 / 1 of a head are L2 hits for most jobs
  (neighbouring CTAs on the XCD run the same heads), and the prologue's `vmcnt(0)` still waits for
  the previous job's O stores, issued after the prefetch, so the store drain stays exposed.
- The larger, deterministic per-shape moves (B1 S64K +1.6-1.8 %, kimik3 +1.1-1.2 %, B2 S16K -0.35 %,
  B1 S8K -0.2 %) do not scale with the per-job prologue share; they come from the KV loop's register
  allocation shifting once the prologue no longer holds K DMA voffsets: one loop block reads 17 AGPRs
  instead of 20 and a block splits in two (`wip/isa_blocks.py` diff). They are real for this binary
  but are a side effect, likely to move again with the next unrelated change.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no new cache. The schedule table and its shape-only key are those of round 10, and the
  drain prefetch reads only the job ids already in that table.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no new cache).
- Benchmark gain on this round: +0.32 % combined-step (forward-only: combined step = forward).
- Estimated real-training gain: +0.3 % at these shapes. Same work per job, same numerics (bitwise),
  the K DMAs moved one job earlier, plus a loop register-allocation shift that is a property of the
  compiled kernel, not of the benchmark loop.
- Decision: ACCEPT-as-real.

## Next direction
The prologue's `s_waitcnt vmcnt(0)` now waits for the previous job's O stores. With the prefetch older
than the stores and vmcnt in order, waiting with `vmcnt(<O store count>)` would retire only the
prefetch and leave the stores in flight (the next job's O registers are written much later). After
that: the next job's Q load in the drain (needs AGPR-destination loads to fit in 512 registers).
