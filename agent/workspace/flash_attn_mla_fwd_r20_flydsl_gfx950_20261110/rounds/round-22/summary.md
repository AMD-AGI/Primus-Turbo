# Round 22 — ROLLBACK — 2026-11-11 (UTC+8)

## Hypothesis
Round 21 found the KV loop's QK slot schedule is chosen by IGroupLP afresh whenever unrelated IR
changes. Only round 19 has every slot of both hot blocks as `RR w (M e)x4 MMMM`, and the
loop-bound S64K tracks it (~1365 vs ~1330 TFLOPS). Pinning the whole slot (its two K reads, the
four (MFMA, EXP) pairs, then the other four MFMAs) should keep that schedule regardless of other
edits. On round 19 itself it may also regularize the two slots that are still irregular.

## Single change
`CODEGEN_REV` 1060, delta `artifacts/r22_delta_vs_r19.patch`: helper `_qk16_slot(traits,
n_reads, n_pairs, group)` emits `sched_group_barrier(DS_READ 0x100, n_reads)`, the
(MFMA, EXP) x n_pairs pairs, then `sched_group_barrier(MFMA, 2 * Q_TILES - n_pairs)`. It is used
in `_qk_finish_p`'s SOLO_WAVE k-slots 0-7 in place of the pairs-only `_interleave`.

ISA (`artifacts/r22_persistent_h128.s`): slots 0-6 unchanged. The last exp slot of both
`.LBB0_8` and `.LBB0_12` is now regular as well (it was `MeMeMMneMeMMM` / `MeMMMMeMeM`).
VGPRs 508, 0 spills.

## Results (full set, base = round-19 snapshot; two separate processes)

| shape | cand / base (run 1) | cand / base (run 2) |
|---|---|---|
| dsv3 | 1.0005 | 0.9997 |
| dsv2lite | 1.0009 | 0.9996 |
| kimik2 | 0.9994 | 0.9997 |
| kimik3 | 0.9993 | 0.9996 |
| B1 S4K | 0.9991 | 0.9989 |
| B1 S8K | 1.0006 | 1.0002 |
| B1 S16K | 1.0002 | 1.0000 |
| B1 S32K | 1.0002 | 0.9996 |
| B1 S64K | 1.0006 | 1.0004 |
| B2 S4K | 0.9992 | 0.9991 |
| B2 S16K | 1.0005 | 1.0002 |

Geomean 1229.45 vs 1229.41 (+0.003 %, `artifacts/r22.csv`); repeat 1233.56 vs 1233.91
(-0.03 %, `artifacts/r22_rep2.csv`). 11/11 PASS, bitwise; smoke 8/8 bitwise
(`artifacts/r22_smoke.log`).

## Decision
ROLLBACK per Rule 3: the aggregate did not improve (+0.003 % / -0.03 %, noise). Sources restored to
the round-19 snapshot. The pin is measured neutral on round 19 and is available to later rounds
whose change would otherwise perturb the loop (stated there if reused).

## Rollback Analysis
- Change made: whole-slot sched_group_barrier pin for the KV loop's QK k-slots.
- Expected effect: neutral on round 19 (its schedule already matches) or a small gain from the
  two irregular slots; robustness for later rounds.
- Actual effect: neutral within noise, bitwise.
- Root cause: the two regularized slots were not on the critical path (the last exp slot is
  followed by the cast/row-sum slot, which absorbs the difference).
- Lesson learned: round 19's loop is already at its pinned schedule. The S64K gap between rounds
  is entirely the perturbation of slots 3 / 5, not something the pin can improve further.

## Attribution
Pure schedule constraint. It changes only the last exp slot's order on round 19, and that slot
has slack.

## Real-training transfer check
- Bucket: K1 (fixed-cost schedule).
- Cache key: no cache.
- id(...) audit: no key.
- 4-step pen-and-paper trace hit rate: n/a.
- Benchmark gain on this round: +0.003 % / -0.03 %.
- Estimated real-training gain: none.
- Decision: ROLLBACK.

## Next direction
Third consecutive rollback (R20-R22), so per Rule 7 escalate to an ISA-level comparison against a
reference. Per-job fixed cost is still ~26 % of dsv3. The largest piece of it not yet attacked is
the causal diagonal's wave imbalance in the drain: wave w owns rows 64w..64w+63, so wave 3 does
four diagonal KV tiles while wave 0 does one and waits at the barriers.
