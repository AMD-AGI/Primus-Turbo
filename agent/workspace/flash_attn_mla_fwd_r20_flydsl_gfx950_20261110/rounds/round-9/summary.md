# Round 9 — ROLLED BACK — 2026-11-11 08:45 (UTC+8)

## Hypothesis
At S = 4K about a quarter of each job's time is fixed cost (prologue Q load, drain, O store), and a
fresh CTA per q block pays launch / LDS setup on top. A persistent kernel (one CTA per CU running a
list of jobs) removes the per-CTA launch cost and is the frame later rounds need to overlap one job's
drain with the next job's loads.

## Single change
Persistent job loop (`persistent` builder option, `CODEGEN_REV` 1045):
- `_setup` keeps per-CTA state only; per-job values move to `bind_item(h, q_block, batch)`;
  `block_item()` is the old block-index mapping, so the one-job-per-CTA path is unchanged.
- Host schedule (`launcher.schedule`, cached per shape in `ops._schedule`): replays the hardware
  dispatch (CTA c on XCD c % 8, job i to queue i % 8, `host_job_order` = device `block_item`),
  LPT list scheduling with cost = `JOB_FIXED_TILES` (8) + KV tiles. Row c = job count + packed jobs
  (`h | qb << 9 | b << 24`), read with `s_buffer_load` one job ahead.
- Grid = `min(num CUs, jobs)` CTAs; dense, non-windowed layouts only (asserted).

## Results (full set, 2 rotations, `artifacts/r9.csv`, base = round-7 snapshot)

| shape | cand | base (R7) | cand / base |
|---|---|---|---|
| dsv3 | 1030.3 | 1032.4 | 0.9980 |
| dsv2lite | 1010.2 | 1004.9 | 1.0053 |
| kimik2 | 999.5 | 1025.4 | 0.9747 |
| kimik3 | 1160.8 | 1161.4 | 0.9995 |
| B1 S4K | 1013.3 | 1020.0 | 0.9934 |
| B1 S8K | 1187.0 | 1176.7 | 1.0088 |
| B1 S16K | 1263.2 | 1267.1 | 0.9969 |
| B1 S32K | 1306.3 | 1307.5 | 0.9991 |
| B1 S64K | 1317.6 | 1322.6 | 0.9962 |
| B2 S4K | 1026.4 | 1030.2 | 0.9963 |
| B2 S16K | 1272.8 | 1268.4 | 1.0035 |

Geomean 1137.50 vs 1140.48 (-0.26 %), 11/11 PASS, bitwise identical. GPU smoke (odd S, B3 H24,
H128) bitwise identical to round 7.

## Decision
ROLLED BACK (geomean down). Kept as the base of round 10 (same hypothesis, spill fixed).

## Rollback Analysis
- Change made: one CTA per CU running a host-scheduled job list.
- Expected effect: launch / setup cost per q block removed, +2-4 % at 4K.
- Actual effect: -0.26 % (kimik2 -2.5 %).
- Root cause: ISA (`artifacts/r9_persistent.s`, local only): VGPR spill slots 8 -> 77, scratch 36 -> 312 B,
  SGPR 74 -> 106. The lane-only parts of the global offsets (36 x i64 Q-load offsets, the K / V DMA
  voffsets) do not depend on the job, so LICM hoists them out of the job loop; they then live across
  the whole KV loop next to the per-job sums and spill. Each job reloads ~40 dwordx2 in its prologue,
  several behind `s_waitcnt vmcnt(0)` ahead of the first K / V DMA. This ate the launch saving.
- Lesson learned: in a persistent loop, lane-only address terms need a per-iteration opaque root
  (or they are hoisted and spilled); check the spill count before benchmarking.

## Real-training transfer check
- Bucket: K1 (fixed-cost / dispatch). Cache key: schedule table keyed on (module, batch, seq_len,
  device): shape only, never tensor identity; rebuilt for any new shape. Decision: REJECT
  (performance), not over-fit.

## Next direction
Same loop with the lane terms re-derived per job from an opaque copy of the thread id (round 10).
