# Round 21 — ROLLBACK — 2026-11-11 (UTC+8)

## Hypothesis
Round 20 showed the job top's Q scale (~500 VALU, ~2.0K cycles per job) cannot hide behind the
LDS / VMEM bound O store. At the job top it runs entirely before QK0 (a `_sched_barrier(0)`
separates the two regions), while QK0's 96 MFMAs (1.4K cycles of pipe) wait on it. QK0 k-step ks
reads only that k-step's four Q packs, so the scale of k-step ks+1 can run under k-step ks's
MFMAs: ~4.3K -> ~2.4-2.6K cycles per job.

## Single change
`CODEGEN_REV` 1059, delta `artifacts/r21_delta_vs_r19.patch`:
- Helper: `scale_q_packs(q_raw, idx, scaled)` scales and AGPR-pins selected raw packs (same
  per-element ops as `scale_all`); `_qk16_halves` takes a pack list as well as the joined Q.
- Kernel (`LAZY_Q_SCALE`: persistent, QK16, SOLO_WAVE, split K reads, no score bound): the job top
  scales k-step 0 only; `_qk0_scaling` runs QK0 one k-slot at a time (as `_qk_finish_p`), each
  slot scaling half the q tiles of the next k-step under its 8 MFMAs with
  `_interleave(8 MFMA, 5 VALU)`, then joins the packs for the KV loop.
- ISA (`artifacts/r21_persistent_h128.s`): QK0 region is (MFMA, 5 VALU) throughout as intended.
  VGPRs 508 -> 490, 0 spills.

## Results (full set, base = round-19 snapshot; two separate processes)

| shape | cand / base (run 1) | cand / base (run 2) |
|---|---|---|
| dsv3 | 1.0015 | 1.0004 |
| dsv2lite | 0.9999 | 0.9987 |
| kimik2 | 0.9994 | 0.9997 |
| kimik3 | 0.9996 | 0.9998 |
| B1 S4K | 1.0001 | 0.9992 |
| B1 S8K | 0.9976 | 0.9981 |
| B1 S16K | 0.9997 | 0.9998 |
| B1 S32K | 0.9976 | 0.9953 |
| B1 S64K | 0.9759 | 0.9714 |
| B2 S4K | 1.0002 | 1.0005 |
| B2 S16K | 0.9989 | 0.9986 |

Geomean 1225.35 vs 1228.70 (-0.27 %, `artifacts/r21.csv`); repeat 1229.09 vs 1233.44 (-0.35 %,
`artifacts/r21_rep2.csv`). 11/11 PASS, bitwise identical to round 19; smoke 8/8 bitwise.

## Decision
ROLLBACK per Rule 3 (geomean down in both runs). Sources restored to the round-19 snapshot.

## Attribution
- ATT dsv3 (`profiles/att_r21/`): job top 4286 -> 4036 cycles per job only. QK0's MFMAs are now
  hidden (1378 -> 587 cycles), but its VALU grew 2225 -> 2834.
  - Every instruction issued right after an MFMA costs 8 cycles.
  - `v_pk_mul_f32` within 3 instructions of an MFMA waits 15-20 cycles: packed FP32 does not
    overlap an MFMA in flight. One pk_mul per (MFMA, 5 VALU) gap lands there about half the time.
  - The compiler split some per-pack products into scalar `v_mul_f32`.
- KV loop: dsv3's loop blocks are unchanged in ATT (1524.0K -> 1521.2K cycles per wave), but
  S64K (loop-only) loses 2.4-2.9 %. The per-k-slot loop schedule changed although the loop code
  did not:
  - Round 19 has every QK slot of both hot blocks (.LBB0_8 / .LBB0_12) as
    `RR w (M e)x4 MMMM`, the only round so far that does.
  - Rounds 17/18 read one K pack a slot late; rounds 20/21 put the four free MFMAs ahead of the
    (MFMA, exp) pairs in two slots per block.
  - S64K tracks this: round 19 ~1360-1370 TFLOPS, rounds 18/20/21 ~1325-1335.
  - The slot pattern pins only 4 (MFMA, EXP) pairs. The other 4 MFMAs and the K reads are left
    to IGroupLP, whose choice shifts with any change to the surrounding IR.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key.
- 4-step pen-and-paper trace hit rate: n/a.
- Benchmark gain on this round: -0.27 % / -0.35 %.
- Estimated real-training gain: none (rolled back).
- Decision: ROLLBACK.

## Next direction
Pin the whole KV-loop QK slot (K reads, the four pairs, the remaining MFMAs) so the loop keeps
round 19's schedule whatever changes around it. Then retry the lazy scale on top, with the pk_mul
placed away from the MFMAs.

## Rollback Analysis
- Change made: job-top Q scale interleaved under QK0's MFMAs one k-step ahead.
- Expected effect: job top 4.3K -> ~2.4-2.6K cycles per job.
- Actual effect: 4.3K -> 4.0K; S64K -2.4 / -2.9 % from a perturbed KV-loop slot schedule.
- Root cause: the QK0 region turned VALU-issue bound (8 cycles after each MFMA, pk_mul stalls
  behind MFMAs). Separately, the loop's partially pinned slots were re-solved by IGroupLP.
- Lesson learned: count issue cycles, not pipe cycles, when hiding VALU under MFMA on a single
  wave; pin full slots.
