# Round 8 — ROLLED BACK — 2026-11-10 20:40 (UTC+8)

## Hypothesis
Round 7 moved C5's K-head read into the C3 P*V site (+0.34 %). The same for C1: read K buffer 1's
head in the C7 P*V site (buffer 1 is visible after C5's sync; C7's DMAs fill buffer 0), carry it over
the loop back edge, and read the first iteration's head at the end of the prologue.

## Single change
`_pv_spec_site(..., k_head_buf=1)` in C7, the packs appended to the loop's yield / init args, C0 takes
them from `loop_args` instead of `_load_k_head(1)`. `CODEGEN_REV` 1042. ISA: VGPR 512, 8 spill slots,
C7 block `v_accvgpr_read` 13 -> 17, C1 block `s_nop` 4 -> 8.

## Results (full set, 2 rotations, `artifacts/r8.csv`, base = round-7 snapshot)

| shape | cand | base (R7) | cand / base |
|---|---|---|---|
| dsv3 | 1028.4 | 1028.6 | 0.9998 |
| dsv2lite | 1002.4 | 1003.1 | 0.9993 |
| kimik2 | 1027.4 | 1021.6 | 1.0057 |
| kimik3 | 1158.8 | 1160.2 | 0.9988 |
| B1 S4K | 1016.9 | 1016.7 | 1.0002 |
| B1 S8K | 1173.6 | 1174.7 | 0.9991 |
| B1 S16K | 1256.1 | 1264.8 | 0.9931 |
| B1 S32K | 1292.2 | 1306.1 | 0.9894 |
| B1 S64K | 1311.2 | 1321.5 | 0.9922 |
| B2 S4K | 1027.6 | 1028.3 | 0.9993 |
| B2 S16K | 1263.2 | 1268.1 | 0.9961 |

Geomean 1135.50 vs 1138.31 (-0.25 %), 11/11 PASS, bitwise identical.

## Decision
ROLLED BACK (geomean down). Code restored to round 7 (`CODEGEN_REV` 1043).

## Rollback Analysis
- Change made: C1's K head read in the C7 P*V site, carried over the loop back edge.
- Expected effect: like round 7, one exposed LDS round trip per iteration removed (+0.3 %).
- Actual effect: -0.25 %, long S -0.4 to -1.1 %.
- Root cause: the C1-side wait was small (~20 cycles in the round-6 ATT vs 48 at C5: the loop
  back-edge SALU already covers part of it), while 8 more VGPRs live across the C7 rebase branch and
  the back edge push the full (512) register file into more AGPR spill traffic in the hot blocks.
- Lesson learned: with the register file full, a latency fix only pays where the exposed wait is
  large; carry values over the back edge only when the saving clearly exceeds a few AGPR moves.

## Real-training transfer check
- Bucket: K1. Cache key: no cache. Decision: REJECT (performance), not over-fit.

## Next direction
Register pressure is the common limiter (rounds 5, 6, 8 all added spill traffic). Find what holds the
spilled values in the hot loop (the `v_accvgpr_read` of spilled LDS addresses in the C7 block) and
free those registers, e.g. by recomputing LDS read addresses from a base + immediate offsets.
