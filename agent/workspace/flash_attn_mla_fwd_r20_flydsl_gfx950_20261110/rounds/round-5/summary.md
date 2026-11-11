# Round 5 — ROLLED BACK — 2026-11-10 19:35 (UTC+8)

## Hypothesis
Each P*V site emits `s_mov m0` -> asm P*V MFMA -> `s_nop 0` -> `buffer_load ... lds` for its 7 DMA
slots (14 `s_nop 0` per KV tile): the hazard recognizer counts the inline-asm MFMA as no wait state,
so the M0 -> LDS-DMA hazard gets a nop (in the prologue a real instruction always sits between). Putting
the next softmax piece into the DMA slot should let the scheduler fill the hazard slot with real work.

## Single change
`_pv_spec_site`: a DMA slot also runs the next softmax piece (`lambda d, p: (d(), p())`). `CODEGEN_REV`
1037. ISA: PV-site `s_nop` 7 -> 1 per site; VGPR 512, spill slots 4 -> 8 (scratch 36 B, cold blocks).

## Results (full set, 2 rotations, `artifacts/r5.csv`, base = round-4 snapshot)

| shape | cand | base (R4) | cand / base |
|---|---|---|---|
| dsv3 | 1003.8 | 1021.2 | 0.9830 |
| dsv2lite | 984.8 | 1002.9 | 0.9820 |
| kimik2 | 1003.7 | 1015.9 | 0.9880 |
| kimik3 | 1128.5 | 1148.4 | 0.9827 |
| B1 S4K | 991.5 | 1008.1 | 0.9835 |
| B1 S8K | 1138.1 | 1161.4 | 0.9799 |
| B1 S16K | 1230.6 | 1259.5 | 0.9771 |
| B1 S32K | 1284.3 | 1305.6 | 0.9837 |
| B1 S64K | 1307.2 | 1322.7 | 0.9883 |
| B2 S4K | 1004.9 | 1020.7 | 0.9845 |
| B2 S16K | 1234.0 | 1262.8 | 0.9772 |

Geomean 1112.69 vs 1132.28 (-1.73 %), 11/11 PASS, bitwise identical.

## Decision
ROLLED BACK (every shape slower). Code restored to round 4 (`CODEGEN_REV` 1038).

## Rollback Analysis
- Change made: merge each P*V-site DMA slot with the following softmax piece.
- Expected effect: -14 `s_nop 0` per tile (~70 cycles, ~1 %).
- Actual effect: -1.7 % on all shapes.
- Root cause: a 16-cycle 16x16x32 MFMA leaves ~12 issue cycles; a DMA slot (`s_mov m0` + ~11-cycle
  `buffer_load ... lds`) already fills it, so adding an 8-12 cycle max3 / exp piece overflows the slot
  and the MFMA pipe idles ~10-15 cycles per DMA slot (14 per tile), far more than the nops cost. The
  nop sat in a slot that was full anyway.
- Lesson learned: in the solo-wave loop the unit of cost is per-slot overflow, not total instruction
  count. Find the slots that overflow (MFMA issued with no pipe stall after a long gap) before moving
  work; removing an instruction only helps in an overflowing slot.

## Real-training transfer check
- Bucket: K1. Cache key: no cache. Decision: REJECT (performance), not over-fit.

## Next direction
Per-slot bubble map from ATT (gap between consecutive MFMA issues minus 16 cycles) on round-4 code,
then move work out of the overflowing slots into the ones with spare capacity.
