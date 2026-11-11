# Round 10 — ACCEPTED — 2026-11-11 09:20 (UTC+8)

## Hypothesis
Round 9's persistent loop lost its launch saving to spills: job-invariant lane address terms were
hoisted out of the job loop and spilled across it (8 -> 77 spill slots). Re-deriving those terms per
job from an opaque copy of the thread id keeps them next to their per-job uses, so the persistent
kernel should keep the register footprint of round 7 and show the fixed-cost saving (+2-4 % at 4K).

## Single change
On top of the round-9 loop (`CODEGEN_REV` 1047, delta `artifacts/r10_delta_vs_r9.patch`):
`bind_item` passes `tid` through an empty side-effecting asm (`_pin_regs`) when `PERSISTENT`, and
derives the global-offset lane terms from it: `job_lane` (Q-load rows / d terms in `load_all`, moved
there from `_setup`), `dma_wave`, `n_in_warp`, `d_bucket`, `v16_dma_d_bucket` (K / V DMA voffsets).
LDS addresses stay in `_setup` (job-independent, as before). One-job-per-CTA path: same IR.
ISA (`artifacts/r10_persistent.s`, local only): spill slots 77 -> 9 (round 7: 8), scratch 312 -> 40 B, per-job
prologue reloads ~40 -> 3.

## Results (full set, 2 rotations, `artifacts/r10.csv`, base = round-7 snapshot)

| shape | cand | base (R7) | cand / base |
|---|---|---|---|
| dsv3 | 1069.4 | 1026.7 | 1.0416 |
| dsv2lite | 1048.6 | 1003.1 | 1.0454 |
| kimik2 | 1069.7 | 1023.4 | 1.0452 |
| kimik3 | 1183.6 | 1159.8 | 1.0205 |
| B1 S4K | 1058.1 | 1015.6 | 1.0418 |
| B1 S8K | 1213.6 | 1173.8 | 1.0339 |
| B1 S16K | 1275.5 | 1265.0 | 1.0083 |
| B1 S32K | 1311.9 | 1306.5 | 1.0041 |
| B1 S64K | 1318.2 | 1322.7 | 0.9966 |
| B2 S4K | 1069.9 | 1027.3 | 1.0415 |
| B2 S16K | 1283.4 | 1269.2 | 1.0112 |

Geomean 1168.05 vs 1138.20 (+2.62 %), 11/11 PASS, bitwise identical to round 7. Edge smoke
(`artifacts/r10_smoke.log`: S 300 / 777 / 1000, B3 H24, B4 H64, B2 H128) 8/8 bitwise identical.

Neutrality of the kept one-job-per-CTA path (`persistent=False`, `artifacts/r10_nonpersistent.*`):
geomean 1138.71 vs 1138.76 (1.0000), 11/11 bitwise, every shape within ±0.1 %. The gain is the
persistent schedule, not a side effect of the restructure.

## Decision
ACCEPTED (geomean +2.62 % vs the last accepted round 7). 1.162x aiter (1005.24).

## Analysis
- 4K shapes +4.2-4.5 %: matches the ~25 % per-job fixed cost estimate minus the part that is the
  kernel's own prologue / drain (still serialized inside a job). kimik3 (+2.1 %), 8K (+3.4 %) and
  16K (+0.8-1.1 %) scale with the job count per CU. B1 S64K -0.34 %: 129 long jobs per CTA; the
  static cost model (`JOB_FIXED_TILES` = 8) leaves a slightly worse tail than hardware dispatch.
- The loop body is unchanged, so prologue Q load (~7.3K cycles), drain barrier stall and the O store
  are still exposed per job; the persistent loop is now the frame to overlap them across jobs.

## Real-training transfer check
- Bucket: K1 (fixed-cost / dispatch). Cache key: `ops._schedule` lru_cache on (module, batch,
  seq_len, device), i.e. shape only; tensors and their addresses never enter it, so a training step
  with fixed shapes hits it from the second call. Host cost of a cold build: 1.9 ms (B1 S4K H128),
  2.7 ms (B4 S4K H64), 31 ms (B1 S64K H128), once per shape (`wip/sched_time.py`).
- Gain source: fewer CTA launches and per-CTA setup, the same work per job, the same numerics
  (bitwise). Transfers to any step that runs these shapes. Decision: ACCEPT, not over-fit.

## Next direction
Cross-job overlap inside the persistent loop, one piece per round: issue the next job's K0 DMA in the
drain once K buffer 0 is free (after C5), then K1 / V0, then the Q prefetch (needs AGPR-destination
loads or the scale folded into softmax to stay within 512 registers).
