# Round 20 — ROLLBACK — 2026-11-11 12:53 (UTC+8)

## Hypothesis
ATT of round 19 on dsv3 (`profiles/att_r19/`, hit-weighted per-instruction view): the persistent
job top (`.LBB0_3`) takes 4.3K cycles per job. QK0's 96 MFMAs are 1.4K of it; scaling the job's Q
(bf16 unpack `v_and` / `v_lshlrev`, `v_pk_mul_f32`, `v_cvt_pk_bf16_f32`, 100
`v_accvgpr_write`, ~500 VALU) is ~2.0K, and QK0 waits on it. The previous job's O store
(`.LBB0_2`, 2.7K) is LDS / VMEM bound with the VALU idle, and the next job's raw Q is in registers
there (stage read back, gathers from C10). Scaling it there and carrying the scaled packs over the
job back edge should take ~2K cycles off the job top for little cost in the store.

## Single change
`CODEGEN_REV` 1058, delta `artifacts/r20_delta_vs_r19.patch`:
- Helper: `scale_all` split into `scale_q_packs` (scaled, AGPR-pinned packs) + `assemble_q`;
  `scale_q_raw(q_raw)` for raw packs (asserts no score bound, whose q norm cannot cross jobs).
- Kernel (`PRESCALE_Q` = persistent and no score bound): `scale_q_raw(q_next)` right after
  `store_final_o`, in the same basic block, left to the scheduler; the first job's Q is scaled
  before the loop; the job top only joins the packs.
- ISA (`artifacts/r20_persistent_h128.s`): job top 868 -> 384 instructions (no scale VALU), O
  store block 314 -> 801. VGPRs 508 -> 488, 0 spills.

## Results (full set, base = round-19 snapshot; two separate processes)

| shape | cand / base (run 1) | cand / base (run 2) |
|---|---|---|
| dsv3 | 0.9983 | 0.9972 |
| dsv2lite | 0.9967 | 0.9964 |
| kimik2 | 1.0021 | 1.0012 |
| kimik3 | 0.9986 | 0.9994 |
| B1 S4K | 0.9956 | 0.9962 |
| B1 S8K | 0.9974 | 0.9970 |
| B1 S16K | 0.9983 | 0.9981 |
| B1 S32K | 0.9962 | 0.9966 |
| B1 S64K | 0.9825 | 0.9877 |
| B2 S4K | 0.9968 | 0.9978 |
| B2 S16K | 0.9983 | 0.9979 |

Geomean 1223.96 vs 1228.35 (-0.36 %, `artifacts/r20.csv`); repeat 1231.33 vs 1235.22 (-0.32 %,
`artifacts/r20_rep2.csv`). 11/11 PASS, bitwise identical to round 19; smoke 8/8 bitwise.

## Decision
ROLLBACK per Rule 3: geomean down in both runs. Sources restored to the round-19 snapshot
(`CODEGEN_REV` 1057).

## Attribution
- ATT (`profiles/att_r20/`): job top 4286 -> 2223 cycles per job, O store block 2742 -> 4697.
  The scale VALU moved without overlapping anything (net ~-0.1K per job).
- One wave per SIMD issues in order: when a `ds_write_b128` / `buffer_store` stalls on a full
  queue, the VALU behind it stalls too, and no other wave fills the slot. The store block's
  "idle VALU" is only usable if the VALU sits between the memory ops at a rate the queues drain.
- The remaining -0.2-0.4 % on every shape (S64K -1.2 / -1.8 %) comes with the new register
  allocation (KV loop rescheduled at 488 VGPRs); not separated.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key.
- 4-step pen-and-paper trace hit rate: n/a.
- Benchmark gain on this round: -0.36 % / -0.32 %.
- Estimated real-training gain: none (rolled back).
- Decision: ROLLBACK.

## Next direction
Keep the scale at the job top but overlap it with QK0 instead: QK0 needs the k-step ks packs
only, so (MFMA, VALU) `sched_group_barrier` pairs can run the scale of later k-steps under the
earlier k-steps' MFMAs (1.4K of MFMA against ~2.0K of VALU: ~2.3K instead of 4.3K at best).

## Rollback Analysis
- Change made: next job's Q scale moved from the job top to right after the previous job's O
  store.
- Expected effect: ~-2K cycles per job (VALU hidden in the LDS / VMEM bound store).
- Actual effect: job top -2.06K, O store +1.96K per job; geomean -0.3 %.
- Root cause: one wave per SIMD and in-order issue; the store's stalls block the VALU behind them.
- Lesson learned: only MFMA shadows (or explicit slotting at the queue drain rate) can hide VALU
  here.
