# Round 12 — ACCEPTED (trade-off) — 2026-11-11 09:45 (UTC+8)

## Hypothesis
After round 11 the persistent prologue still reloads three spilled values from scratch, each followed
by `s_waitcnt vmcnt(0)`: `wave_q_offset` right after the prologue barrier (before the Q loads), a
causal-mask lane term before the QK0 mask, and an LDS read base before the loop. They are the
lane / wave layout values `_setup` computes once per CTA; in the job loop they live across every KV
loop and spill, which is round 9's LICM problem for the values round 10 did not cover. Re-deriving the
whole lane layout per job from the opaque tid should take those serialized scratch round trips (and the
full `vmcnt(0)` drain they force) out of every job: a fixed per-job saving, largest at 4K.

## Single change
`CODEGEN_REV` 1049, delta `artifacts/r12_delta_vs_r11.patch` (helper only):
- The lane / wave layout block of `_setup` (wave_id, lane, lane_mod/div_32, wave_row_group,
  wave_q_offset, K / V LDS read bases, PV16 lane terms) moves into `_bind_lane_layout(tid)`.
  `_setup` calls it with the real tid (one-job-per-CTA path: same values, same place);
  `bind_item` calls it again with the per-job opaque tid when `PERSISTENT`.
- `_job_tid()` returns that opaque tid; `_bind_lanes(tid)` (global-offset lane terms) and the K
  prefetch share it. The prefetch rebinds only the DMA lane terms, not the layout.
- ISA at H=8 (`artifacts/r12_persistent_h8.s`): spill slots 9 -> 3, scratch 40 -> 16 B, scratch
  reloads 10 -> 2 (both in the drain, none in the prologue). H=128 (`artifacts/r12_persistent_h128.s`):
  spill slots 11 -> 3 (round 11 at H=128: 11, 48 B).

## Results (full set, 2 rotations, `artifacts/r12.csv`, base = round-11 snapshot)

| shape | cand | base (R11) | cand / base |
|---|---|---|---|
| dsv3 (B2 S4K H128) | 1092.5 | 1071.9 | 1.0192 |
| dsv2lite (B8 S4K H16) | 1057.5 | 1045.2 | 1.0118 |
| kimik2 (B4 S4K H64) | 1066.6 | 1072.0 | 0.9950 |
| kimik3 (B2 S7K H96) | 1184.9 | 1196.3 | 0.9905 |
| B1 S4K | 1079.9 | 1060.0 | 1.0188 |
| B1 S8K | 1231.1 | 1209.3 | 1.0180 |
| B1 S16K | 1274.5 | 1278.3 | 0.9970 |
| B1 S32K | 1305.9 | 1315.5 | 0.9927 |
| B1 S64K | 1310.3 | 1342.7 | 0.9759 |
| B2 S4K | 1093.2 | 1076.3 | 1.0157 |
| B2 S16K | 1285.4 | 1278.2 | 1.0056 |

Geomean 1175.99 vs 1171.82 (+0.36 %), 11/11 PASS, bitwise identical to round 11. Edge smoke
(`artifacts/r12_smoke.log`) 8/8 bitwise identical. Largest regression B1 S64K -2.4 % (< 5 %).

Kept one-job-per-CTA path (`persistent=False`, `wip/np12` vs `wip/np11`): ISA identical instruction
for instruction (7905 instructions, 8 spill slots both).

## Decision
ACCEPTED (trade-off) per Rule 3: geomean +0.36 %, no shape regressed by 5 % or more; B1 S64K -2.4 %,
kimik3 -1.0 %, B1 S32K -0.7 %, kimik2 -0.5 %. 1.170x aiter (1005.24).

## Attribution
- Head count is a compile-time trait, so each H is its own binary. In the H=128 binary (dsv3, B1 / B2
  shapes) the gain falls with sequence length: S4K +1.6-1.9 %, S8K +1.8 %, S16K -0.3 / +0.6 %,
  S32K -0.7 %, S64K -2.4 %. S64K is almost pure KV loop, so that binary's loop got ~2.4 % slower;
  the per-job prologue saving is then ~+4 % gross at 4K.
- The loop slowdown is register allocation, not work: per iteration the H=128 loop has 881 (R10),
  896 (R11) and 878 (R12) instructions; R11 is the fastest. What moves is where the 16-20
  `v_accvgpr_read` copies land: loop-invariant values parked in AGPRs a236-a255 and read back in
  the loop, all in one block (R10, R11: 20 / 6+14) or split 8 + 8 (R12), inside MFMA shadows that
  are already at the issue-slot limit. Correction (round 13): those values are the V buffer-1
  `ds_read_b64_tr_b16` addresses (buffer 1 starts at the 64 KiB DS offset limit, so no read there
  folds its offset), not the K / V DMA voffsets, which stay in VGPRs.
- kimik2 (H64) and kimik3 (H96) are other binaries whose loops moved the same way; at 4K the H64
  loop loss outweighs the prologue saving.

## Real-training transfer check
- Bucket: K1 (kernel-internal register / instruction placement).
- Cache key: no cache (the schedule table and its shape-only key are unchanged from round 10).
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +0.36 % combined-step (forward-only: combined step = forward).
- Estimated real-training gain: +0.36 % at these shapes. The change removes per-job scratch reloads
  and their full drains; it is shape- and data-independent and bitwise-neutral, so a training step
  running these shapes sees the same per-kernel times.
- Decision: ACCEPT-as-real.

## Next direction
1. Make the loop independent of that lottery by removing the parked values themselves (done in
   round 13: an opaque base for the V buffer-1 reads, so their offsets fold into the DS immediate).
2. The prologue's pre-QK0 `vmcnt(VM_DRAIN_V)` now waits for the first half of V0 (K1 is already
   resident since round 11); V0 is first read in C3, so leave the whole V0 tile in flight there.
3. Then wait only for the K prefetch in the prologue (`vmcnt(<O stores>)`), not the previous job's
   O stores.
