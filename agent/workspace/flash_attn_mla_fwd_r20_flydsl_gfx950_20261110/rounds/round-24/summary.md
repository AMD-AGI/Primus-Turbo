# Round 24 — ROLLBACK — 2026-11-11 (UTC+8)

## Hypothesis
Round 23's drain balance saved 4.7K cycles per job, but IGroupLP re-chose 2-3 KV-loop QK slots
and S64K fell 3-4 %. Round 22's whole-slot pin (`_qk16_slot`: K reads, (MFMA, EXP) x4, other
MFMAs), measured neutral on round 19, should hold the loop at round 19's schedule and keep the
drain gain.

## Single change
On top of round 23: round 22's `_qk16_slot` pin in `_qk_finish_p`'s SOLO_WAVE k-slots
(`SCHED_DS_READ_MASK` 0x100). `CODEGEN_REV` 1063, deltas `artifacts/r24_delta_vs_r23.patch`
(this round) and `artifacts/r24_delta_vs_r19.patch` (cumulative).

ISA (`artifacts/r24_persistent_h128.s`): VGPRs 502, 0 spills, MFMAs 768. The second hot block
is back to round 19's seven regular slots. The first keeps one irregular slot (slot 5:
`ddWMMMMneWMeeMeMM`, as in round 23).

## Results (full set, base = round-19 snapshot; two separate processes)

| shape | cand / base (run 1) | cand / base (run 2) |
|---|---|---|
| dsv3 | 1.0090 | 1.0088 |
| dsv2lite | 1.0029 | 1.0045 |
| kimik2 | 1.0077 | 1.0083 |
| kimik3 | 0.9998 | 1.0000 |
| B1 S4K | 1.0082 | 1.0073 |
| B1 S8K | 0.9950 | 0.9944 |
| B1 S16K | 0.9942 | 0.9931 |
| B1 S32K | 0.9892 | 0.9863 |
| B1 S64K | 0.9685 | 0.9642 |
| B2 S4K | 1.0109 | 1.0098 |
| B2 S16K | 0.9967 | 0.9961 |

Geomean 1227.20 vs 1229.27 (-0.17 %, `artifacts/r24.csv`); repeat 1229.87 vs 1233.03
(-0.26 %, `artifacts/r24_rep2.csv`). 11/11 PASS, bitwise; smoke 10/10 bitwise.

## Where the S64K loss is (new measurements)
- ATT S64K, CU 0 (`profiles/att_r24`): total cycles per wave 73.03M (R19) vs 73.20M (R24),
  +0.2 %. Per KV-loop iteration +0.9 %. That is not the 3.5 %.
- Sustained clock / power (`artifacts/power_s64k.log`, `power_probe.py`, 10 s each,
  alternating):

| build | MHz | W | PPT-throttle share | TFLOPS/MHz |
|---|---|---|---|---|
| R19 (x5) | 1819-1862 | 1327-1348 | 0.67-0.90 | 0.722-0.730 |
| R22 | 1858 | 1336 | 0.88 | 0.720 |
| R21 | 1747 | 1300 | 0.21 | 0.746 |
| R23 | 1841 | 1300 | 0.32 | 0.712 |
| R24 (x5) | 1794-1809 | 1299-1310 | 0.27-0.33 | 0.726-0.730 |

  The loss is clock, not cycles. R19 and R22 run at the power limit (~1345 W, PPT throttle
  ~0.9) at ~1860 MHz. R21, R23 and R24 sit ~45 W lower, mostly not at the PPT limit, yet at a
  2-3 % lower clock. Another limiter that amd-smi does not expose (current / di/dt class) caps
  them, and which builds hit it tracks the KV-loop slot schedule.
- Why the schedule moves (`wip/ir_loop_cmp.py`): the two hot blocks' LLVM IR is identical
  between R19 and R23 after SSA renaming. The variation is in codegen. With
  `enable-post-misched` off (`artifacts/r24_nopostmisched_h128.s`), every pre-RA slot is in
  source order `dd W M W MMMMMMM eeee`. The pre-RA scheduler does not apply the pin (likely
  reverted at ~500 / 512 VGPRs), so only the post-RA scheduler interleaves. That is possible only
  if the RA did not give an exp a register freed by an MFMA of the same slot. In R24's slot 5 the
  accumulators rotate (MFMA dest = previous MFMA's srcC). The four exps got the chain's freed
  `v[158:161]`, so they must follow all four MFMAs. The exact IGroupLP solver (capped) gives the
  same slot.

## Decision
ROLLBACK per Rule 3: the aggregate did not improve (-0.17 % / -0.26 %). Round 19 stays the best.
The working tree continues as round 25, so the sources are not reverted by hand.

## Rollback Analysis
- Change made: whole-slot QK pin on top of round 23.
- Expected effect: round 19's loop schedule with round 23's drain, so S64K back to round 19
  and +0.5-1 % from the 4K shapes.
- Actual effect: one hot-block slot stays irregular, S64K -3.2 / -3.6 %, the geomean is the
  same as round 23.
- Root cause: a sched_group_barrier pin only takes effect post-RA here, where the RA's register
  reuse inside a slot can make it unsatisfiable. The 3 % is a clock cap that follows the
  schedule, not lost cycles.
- Lesson learned: make the source (IR) order of each slot the pinned order. Then any register
  the RA reuses within the slot is freed before its new user in the target order too, and the
  post-RA pin always fits. Measure clock and power (`power_probe.py`) whenever a long-sequence
  shape moves while its ATT cycles do not.

## Attribution
Drain: as round 23. Loop: same IR, different RA, so a different post-RA schedule and a 2-3 %
lower sustained clock at S64K. The cycles are unchanged.

## Real-training transfer check
- Bucket: K1. Not transferable until the loop schedule is context-independent.

## Next direction
Round 25: emit each SOLO_WAVE QK slot in pin order in the IR. `_qk16_halves` gets an
`after_mfma` hook, and the slot's piece is split per exp (exp j right after MFMA j). This keeps
round 24's code otherwise.
