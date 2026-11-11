# Round 25 — ACCEPTED WITH NOTE — 2026-11-11 (UTC+8)

## Hypothesis
Round 24 found that the QK slot pins act post-RA only. The pre-RA schedule keeps source order
(`dd M x8 e x4`), so the RA may give an exp a register freed by an MFMA of the same slot, which
makes the post-RA pin unsatisfiable (round 24's slot 5). Emitting each slot in the pinned order
in the IR (exp i right after MFMA i) makes every register reuse inside the slot agree with the
pin. The loop schedule then no longer depends on unrelated code, and round 23's drain gain
should come through.

## Single change
On top of round 24 (round 23's drain balance plus round 22's pin): `CODEGEN_REV` 1064, delta
`artifacts/r25_delta_vs_r24.patch` (cumulative vs round 19: `artifacts/r25_delta_vs_r19.patch`).
- `qk` / `_qk16_both` / `_qk16_halves` take `after_mfma(i)`, emitted after the call's i-th MFMA.
- `finish_p_pieces`: an exp2 piece called with `i` emits only its i-th exp2.
- `_qk_finish_p` SOLO_WAVE pinned k-slots (QK16): `ctx.qk(..., after_mfma=piece(i) for i < 4)`
  then `_qk16_slot` (`QK_SLOT_EXPS` = 4). The other slots are unchanged.

ISA (`artifacts/r25_persistent_h128.s`, `artifacts/loop_pattern_r25.txt`): VGPRs 499, 0 spills,
MFMAs 768. The first hot block now has all 8 pinned slots `dd W M e W M e M e M e MMMM`; round 19
had 7, round 24 had 6. The second block has 7, its last exp slot slightly off as in round 19.
Non-persistent (`wip/np25`, `artifacts/r25_nonpersistent_h8.s`): VGPRs 486, 0 spills, MFMAs
1008 (as np19), bitwise vs np19 on 6 shapes (`artifacts/np25_smoke.log`).

## Results (full set, base = round-19 snapshot; two separate processes)

| shape | cand / base (run 1) | cand / base (run 2) |
|---|---|---|
| dsv3 | 1.0145 | 1.0156 |
| dsv2lite | 1.0071 | 1.0103 |
| kimik2 | 1.0128 | 1.0143 |
| kimik3 | 1.0043 | 1.0042 |
| B1 S4K | 1.0116 | 1.0139 |
| B1 S8K | 0.9980 | 0.9977 |
| B1 S16K | 0.9975 | 0.9968 |
| B1 S32K | 0.9928 | 0.9898 |
| B1 S64K | 0.9671 | 0.9697 |
| B2 S4K | 1.0149 | 1.0175 |
| B2 S16K | 1.0017 | 1.0005 |

Geomean 1232.09 vs 1229.71 (+0.19 %, `artifacts/r25.csv`); repeat 1237.52 vs 1234.24
(+0.27 %, `artifacts/r25_rep2.csv`). 11/11 PASS, bitwise; smoke 10/10 bitwise
(`artifacts/r25_smoke.log`). 1.226x aiter (1005.24).

S64K sustained (`artifacts/power_s64k.log`): R25 1786 / 1824 MHz, 1312 / 1307 W, PPT throttle
0.34 / 0.40. R19 1865 / 1862 MHz, 1349 / 1352 W, 0.91 / 0.90. TFLOPS/MHz is the same
(0.722-0.738 vs 0.727-0.730).

## Decision
ACCEPTED WITH NOTE per Rule 3: the geomean is up in both runs (+0.19 %, +0.27 %). Every regressed
shape is under 5 %: B1 S64K -3.3 / -3.0 %, S32K -0.7 / -1.0 %, S8K / S16K -0.2 / -0.3 %.
Trade-off: the 4K training shapes gain +1.2-1.8 %, and the long single-batch sequences lose by
running at a 2-3 % lower sustained clock. Round 25 is the new best and the A/B base from round 26.

## Attribution
- 4K shapes +1.2-1.8 %: round 23's drain balance (ATT dsv3: drain 27.8K -> 23.1K cycles per
  job), now that the KV loop keeps its schedule. Round 24 got +0.7-1.1 % with one irregular
  slot.
- Long S: not cycles (round 24 ATT +0.2 % at S64K) but clock. The QK slots are now fully regular,
  yet S64K stays at ~1800 MHz / ~1310 W off the PPT limit. This corrects round 24: the clock cap
  does not follow the QK slot pattern. Round 22 (round 19 + pin) runs at the high clock; rounds
  21 and 23-25 do not. What triggers the unexposed limiter is open; candidates are the PV / softmax
  blocks' VALU order (`.LBB0_10` / `.LBB0_14` differ from round 19) and the register assignment
  itself.

## Real-training transfer check
- Bucket: K1 (per-job fixed cost: the causal drain runs in every persistent causal q block).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +0.19 % / +0.27 % combined-step (forward-only: combined step =
  forward).
- Estimated real-training gain: +1.2-1.8 % on the 4K multi-head training shapes (dsv3,
  kimik2, dsv2lite, B2 S4K). About -0.3 % at 8K-16K. -3 % only for single-sequence 32K-64K
  causal, where the sustained clock is lower. Bitwise-neutral; the pulled-back last q block
  writes identical rows twice.
- Decision: ACCEPT-as-real (with the long-sequence clock note).

## Next direction
Find what caps the sustained clock at S64K since round 21, now that QK slot order is ruled out.
- Bisect round 19 -> round 25 on S64K `power_probe.py` (MHz / W / PPT share): round 19 +
  `ROW_INTERLEAVE` row mapping only (no drain skip), then + `block_row0`, then + drain skip.
- Diff `.LBB0_10` / `.LBB0_14` (PV + softmax VALU) order against round 19, and apply the same
  IR-order discipline to their pinned sites if they differ.
Recovering round 19's clock at S64K / S32K is worth +3 % / +1 % there, about +0.4 % geomean on
top of this round.
