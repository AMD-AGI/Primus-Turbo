# Round 23 — ROLLBACK — 2026-11-11 (UTC+8)

## Hypothesis
The dense causal drain (the job's last four KV tiles, the q block's diagonal band) is the
largest remaining per-job cost: 27.8K cycles per dsv3 job in round 19, against a mean wave busy
time of 21.9K. Wave w owns block rows 64w..64w+63, so on diagonal tile t only waves >= t have
live rows. Wave 3 does the full work of all four tiles while the others wait at the chunk
barriers (2-8.8K cycles per wave per job). Interleaving rows across waves (q tile qt of wave w =
block rows 64 qt + 16 w) gives every wave the same share of the diagonal. On a full band, tile
t's dead q tiles are then qt < t on every wave, a static set, so the drain can skip their MFMAs
with no branch.

## Single change
`CODEGEN_REV` 1062, delta `artifacts/r23_delta_vs_r19.patch`. `traits.ROW_INTERLEAVE` (persistent
dense causal, QT4, 16x16x32 QK / PV): `QT_ROW_STRIDE` 64, `WAVE_ROW_STRIDE` 16.
- Row mapping: `bind_q_rows`, `load_q_raw`, `prefetch_q_*`, `stage_q_dma` (DMA rows wrap per
  16-row q tile), `chunk_live`, the PV16 causal mask, `_store_final_o16x4` readback.
- `qts=` on `qk` / `_qk16_halves` / `pv_step_k` and the kernel's `_qk` / `_pv`.
  `_qk_diag` / `_pv_diag` compute q tiles `range(4 - k, 4)` of the k-th last tile.
- `block_row0`: a block running past the sequence end (rounded up to 128) is pulled back to end
  there, so every band is full. The shared rows are computed by both blocks with identical
  per-row math, so the double write is bitwise identical. A first version kept the band clipped
  and added the dead q tiles back in a branch. The branch merge of the accumulators cost 24
  scratch instructions (64 spilled VGPRs), so it was replaced.

ISA (`artifacts/r23_persistent_h128.s`): MFMAs 1008 -> 768, exactly the 240 dead-q-tile MFMAs.
VGPRs 508 -> 502, 0 spills. KV loop: 2-3 QK slots of the two hot blocks lost the
`RR w (M e)x4 MMMM` order (`ddWMMMMneWMee...`, `nddWMWMMMMMMMneeee`).

## Results (full set, base = round-19 snapshot; two separate processes)

| shape | cand / base (run 1) | cand / base (run 2) |
|---|---|---|
| dsv3 | 1.0096 | 1.0091 |
| dsv2lite | 1.0032 | 1.0046 |
| kimik2 | 1.0077 | 1.0086 |
| kimik3 | 0.9997 | 1.0002 |
| B1 S4K | 1.0063 | 1.0082 |
| B1 S8K | 0.9950 | 0.9953 |
| B1 S16K | 0.9936 | 0.9924 |
| B1 S32K | 0.9888 | 0.9870 |
| B1 S64K | 0.9689 | 0.9624 |
| B2 S4K | 1.0086 | 1.0103 |
| B2 S16K | 0.9961 | 0.9964 |

Geomean 1226.61 vs 1229.21 (-0.21 %, `artifacts/r23.csv`); repeat 1231.73 vs 1234.71
(-0.24 %, `artifacts/r23_rep2.csv`). 11/11 PASS, bitwise. Smoke 10/10 bitwise
(`artifacts/r23_smoke.log`), including the pulled-back blocks of S = 300 / 640 / 777 / 129.

ATT dsv3 CU 0 (`profiles/att_r23`, `wip/att_drain_busy.py`): drain 27817 -> 23137 cycles per job
(-4.7K, -17 %). The per-wave barrier wait fell from 2-8.8K to 0.2-3.3K. Mean wave busy is
unchanged (21913 -> 21932), so the gain is the balance, not the skipped MFMAs.

## Decision
ROLLBACK per Rule 3: the aggregate did not improve (-0.21 % / -0.24 %). Round 19 stays the best.
The working tree continues as round 24 (this delta plus round 22's slot pin), so the sources are
not reverted by hand. The round-19 snapshot remains the A/B base.

## Rollback Analysis
- Change made: wave-interleaved q rows and static dead-q-tile skip on the drain's diagonal
  tiles, with sequence-end blocks pulled back to a full band.
- Expected effect: about 3.8K cycles off each job's drain, so +2-4 % on 4K shapes and a
  smaller gain at long sequences.
- Actual effect: the drain fell by the expected 4.7K cycles per job, and the 4K shapes gained
  +0.6-1.0 %. Long sequences regressed in proportion to their KV-loop share: S8K -0.5 %,
  S64K -3.1 / -3.8 %.
- Root cause: IGroupLP re-chose 2-3 QK slot schedules of the KV loop. The pairs-only
  `_interleave` pin lets the free MFMAs and exps move with unrelated IR (here the changed row and
  lane arithmetic and the loop's live-in set), as in round 21. The loop slowed by about 3.5 %,
  which costs dsv3 about as much as the drain saves minus 1 %.
- Lesson learned: any change that alters the loop's surrounding IR needs the whole-slot pin
  (round 22's `_qk16_slot`, measured neutral on round 19). Otherwise its gain is masked by a loop
  schedule lottery. Measure the drain with ATT separately from the end-to-end number.

## Attribution
Drain balance: -4.7K cycles per dsv3 job (ATT). KV loop: about -3.5 % at S64K from the slot
schedule, unrelated to the drain logic (the loop's source is unchanged).

## Real-training transfer check
- Bucket: K1 (per-job fixed cost in the persistent kernel; every causal fwd call in training
  runs the same drain). Would transfer once the loop regression is removed.

## Next direction
Round 24: this delta plus round 22's whole-slot `_qk16_slot` pin, aiming at round 19's loop
schedule with round 23's drain. Then use the now-balanced drain's freed barrier time (every wave
idles under 3.3K) for the next job's K prefetch.
