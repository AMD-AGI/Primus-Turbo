# Round 46 — GateUP-B4-M2048 dgrad-via-H4 (gm, xcds) drift re-audit METHODOLOGY-CORRECTED — SHIP +1-2% wmin_beats_lmax

**TL;DR**: R45 forward-pointer (PRIMARY) executed. With production R10
(slots=200) + R15 (chunk_size=24) + R16 (gm=1) levers FIXED, the (gm,
xcds) sweep on the GateUP-B4-M2048 dgrad-via-H4 RCR rule
(config.py:3022-3029) shows that **(gm=1, xcds=4) wins +1.0..+1.95 %
over the production (gm=1, xcds=None=8)** in two independent 5-seed
× 2000-iter p20 runs. wmin_beats_lmax holds in BOTH runs. Drift since
R34 (which defended xcds=8 with R10/R15/R16 NOT yet stacked) is
empirically real and ships.

## Hypothesis (R45 forward-pointer, PRIMARY)

> "GateUP-B4-M2048 dgrad-via-H4 RCR rule (gm, xcds) drift audit,
> METHODOLOGY-CORRECTED. The R45 dgrad column (uninterpretable in R45
> at slots=0/cs=0) hinted that (1, 8) and (1, 2) cells beat (1, 4)
> at slots=0/cs=0. With the R10/R15/R16 levers held FIXED at
> production (slots=200, cs=24, gm=1), the (gm, xcd) optimum may have
> shifted. Probe should clone _probe_round_45 → swap to time_dgrad-only,
> FIX slots=200 + cs=24, sweep cells {(1, 8)*current, (1, 4), (1, 2),
> (1, 1), (2, 8), (4, 8), (8, 8)} where (1,8) is the current production.
> Same SHIP gate: best ≥ 1.0% lift over (1,8) AND signal > spread."

## Methodology

`scripts/_probe_round_46_gateup_b4_m2048_dgradH4_drift.py` — direct
`grouped_rcr_dscale` invocation (kernel-only timing) with the
binding's `(group_m, num_xcds, num_slots, chunk_size)` quadruple
overridden per-call. **Held FIXED to production**:
- `num_slots = 200` (R10 winner)
- `chunk_size = 24` (R15 winner)

Swept `(gm, xcds)`:
- `(1, 8)` — current production rule (baseline; xcds=None==8 default)
- `(1, 4)` — alternative chiplet partition (R34 defended xcds=8 BEFORE
  R10/R15/R16 stacked)
- `(1, 2)` — narrower partition probe
- `(1, 1)` — single-XCD control (catastrophic-loss expected)
- `(2, 8)`, `(4, 8)`, `(8, 8)` — gm-axis controls at production xcds

Per-cell: 5 seeds × 2000-iter p20 each, kernel-only; total wall ~11s.
Two independent runs (run-1 build cycle + run-2 NO_BUILD reuse).

## Results

### Run 1 (with HK rebuild at start of probe)

| cell      | med ms  | min ms  | max ms  | spread% | TFLOPS  | Δ% vs base |
|-----------|---------|---------|---------|---------|---------|------------|
| (1, 8)*   | 0.1312  | 0.1311  | 0.1315  | 0.30 %  | 2070.9  | baseline   |
| (1, 4)    | 0.1299  | 0.1296  | 0.1300  | 0.25 %  | 2092.0  | **+1.01 %** |
| (1, 2)    | 0.1320  | 0.1319  | 0.1323  | 0.30 %  | 2058.4  | -0.61 %    |
| (1, 1)    | 0.1364  | 0.1361  | 0.1365  | 0.29 %  | 1992.6  | -3.93 %    |
| (2, 8)    | 0.1345  | 0.1344  | 0.1346  | 0.12 %  | 2021.1  | -2.47 %    |
| (4, 8)    | 0.1346  | 0.1344  | 0.1346  | 0.15 %  | 2019.8  | -2.53 %    |
| (8, 8)    | 0.1352  | 0.1350  | 0.1353  | 0.27 %  | 2010.9  | -2.99 %    |

**wmin_beats_lmax**: max((1,4)) = 0.1300 < min((1,8)) = 0.1311 ✓

### Run 2 (NO_BUILD reuse of same .so)

| cell      | med ms  | min ms  | max ms  | spread% | TFLOPS  | Δ% vs base |
|-----------|---------|---------|---------|---------|---------|------------|
| (1, 8)*   | 0.1296  | 0.1294  | 0.1300  | 0.46 %  | 2097.8  | baseline   |
| (1, 4)    | 0.1270  | 0.1268  | 0.1273  | 0.38 %  | 2139.4  | **+1.95 %** |
| (1, 2)    | 0.1297  | 0.1296  | 0.1298  | 0.15 %  | 2095.2  | -0.12 %    |
| (1, 1)    | 0.1342  | 0.1340  | 0.1342  | 0.15 %  | 2025.9  | -3.55 %    |
| (2, 8)    | 0.1329  | 0.1328  | 0.1330  | 0.12 %  | 2045.4  | -2.56 %    |
| (4, 8)    | 0.1327  | 0.1326  | 0.1328  | 0.12 %  | 2048.5  | -2.41 %    |
| (8, 8)    | 0.1334  | 0.1333  | 0.1336  | 0.24 %  | 2038.0  | -2.93 %    |

**wmin_beats_lmax**: max((1,4)) = 0.1273 < min((1,8)) = 0.1294 ✓

### Cross-run consistency

- (1, 4) is the unique winner in BOTH runs.
- Lift consistent direction across runs: +1.01 % then +1.95 %.
- All other cells uniformly lose in BOTH runs. No "noise alias"
  interpretation possible — every single non-baseline non-(1,4) cell
  is consistently worse, meaning the (1,4) win is structural rather
  than a one-shot lucky seed.
- Adjacent neighbour (1, 2) sits at -0.12 % to -0.61 % (slight loss,
  small spread) → confirms xcds=4 is the unique optimum, not a tie
  with xcds=2 the way (1, 8) and (1, 4) might have been if R34's
  argument still held.

## Decision

**SHIP**. Edit config.py:3022-3029 from `num_xcds=None` (= kernel
default 8) to `num_xcds=4`. Comment block below the existing R8/R10/
R15/R16 stack documents R46's mechanism + probe data. group_m,
num_slots, chunk_size unchanged. Bit-equivalent (num_xcds is a pure
chiplet-swizzle scheduling knob — same property documented for every
other (gm, xcds) RCR/RRR/var-K rule in this file).

## Why the drift is mechanistic — chunk_size=24 swizzle math

R34's 2026-05-08 comment block at config.py:2700-2710 defended xcds=8
for tiles_m=8 + tiles_n=11 + k=5760 + B=4 on the grounds that "the
small persistent grid needs WIDER chiplet spread (xcds=8) to amortise
launch + drain costs across all 8 XCDs in a single wave-step." That
argument was made BEFORE R15 added chunk_size=24. Post-R15:

- `xcds=8` + `slots=200` + `cs=24`: block = 8 × 24 = 192, num_chunks
  = limit/block = 1, limit = 192. **24 PIDs/XCD per chunk × 1 chunk
  = 24 PIDs/XCD**. 8 trailing PIDs round-robin.
- `xcds=4` + `slots=200` + `cs=24`: block = 4 × 24 = 96, num_chunks
  = 2, limit = 192. **24 PIDs/XCD per chunk × 2 chunks = 48 PIDs/XCD**.
  Same 8 trailing PIDs round-robin, same 192/200 chiplet coverage,
  but **2× deeper per-XCD slice**.

The doubled per-XCD slice (48 vs 24 PIDs) lets each XCD's L2 absorb
a longer K-tile-column reuse window on the deep k=5760 main loop
(45 K-tile-loads × consecutive N-tile rows on the same A-pack column).
Same per-K B-pack L2 reuse mechanism R16 documented for gm=1 on the
M-axis, now compounded on the XCD-partition axis via xcds=4. R34's
"xcds=8 for wider spread" is moot because cs=24's 1-chunk (xcds=8) →
2-chunk (xcds=4) flip preserves full chiplet coverage.

## Why this audit was correct to perform

R45 sat at the same (1, 4) cell with slots=0/cs=0 (R34's pre-R15
config) and found (1, 4) was -2.07 % vs (1, 8) — i.e. R34's xcds=8
defence held under the OLD pre-R15 wiring. R46 establishes that the
R10/R15/R16 lever stack flipped the optimum, vindicating R44's
"compounded lever stacks need re-audits" hypothesis on the cell with
the largest stack.

The R45 forward-pointer call to fix slots=200/cs=24 was the critical
methodology correction. Without it, R45's (gm, xcds) sweep was
uninterpretable.

## Single-sample metric

Post-edit canonical metric runs: `score=692-693` (2 samples).
Single-sample metric noise floor on this score band is ±5 (per task
spec); +0.7 expected score lift is below the floor. The probe
data is the authoritative signal — wmin_beats_lmax holds at the
cell, the metric will resolve the lift over multi-sample medians
(same R10/R11/R15/R16 sub-noise-but-real ship pattern).

## Project queue status post-R46

R39's enumeration of multi-round projects:
- (a) 2-CTA-per-tile             — FALSIFIED at R40
- (b) var-K RCR template         — FALSIFIED at R42
- (d) 4w grouped port            — FALSIFIED at R39b
- (MFMA 32x32x64 main loop)      — FALSIFIED at R41
- R44-dm cross-port var-K        — FALSIFIED at R43

R44/R45 single-round drift audit chain:
- Down-B4-M2048 (gm,xcd) drift fwd       — FALSIFIED at R44
- GateUP-B4-M2048 (gm,xcd) drift fwd     — FALSIFIED at R45
- GateUP-B4-M2048 (gm,xcd) drift dgrad-H4 — **SHIPPED at R46 (this round)**

The R45 BACKUP plan ("if R46 also returns NEUTRAL, write closure doc")
is OBSOLETE — R46 found a real signal and shipped. The drift-audit
class is NOT exhausted; the methodology lesson is "audit each
compounded-lever rule with the levers FIXED at production".

## R47 forward-pointer

Two viable next-round directions (single-hypothesis, ranked):

1. **PRIMARY — extend R46's drift-audit-with-production-levers-FIXED
   methodology to the next-most-stacked rule.** Candidates ranked by
   lever-stack depth (more compounded levers → more likely drift):

   a. Down-B4-M2048 fwd RCR rule at config.py:2335-2342 (`gm=16,
      xcds=2, slots=196, chunk_size=96`) — 4-lever stack from R2/R11/
      R14. R44 audited (gm, xcds) at slots=196/cs=96 FIXED and
      FALSIFIED. R46 method applied here would test xcds drift more
      narrowly; **already done as R44, mostly negative**. SKIP.
   b. Down-B4-M4096 fwd RCR rule (likely also slots/cs/gm/xcds
      stacked) — locate the rule at config.py and apply R46 method
      with production slots/cs FIXED. NEW CANDIDATE.
   c. GateUP-B4-M4096 dgrad-via-H4 RCR rule at config.py:2432-2536
      (`gm=1, xcds=4, slots=NUM_CUS=256, cs=64 default`, R8 anchor +
      R16 fuse_ktail layer). Note: only a 1-lever stack on the dispatcher
      axis (R8 gm/xcd, no slots/cs override) — drift unlikely.
      LOWER PRIORITY.
   d. Down-B4-M2048 wgrad var-K rule at config.py around line 2440-
      2620 (R3/R8/R10 cumulative lever stack). var-K levers are
      separate (`vk_num_slots`, `vk_num_xcds`, `vk_chunk_size`) —
      audit method symmetric. NEW CANDIDATE.

   Pick (d) for R47 — wgrad section is the gating section (1815 T
   in this metric run vs 1910 fwd / 2090 dgrad), and var-K rules
   have rebuilt many times since R3.

2. **BACKUP — measure R46's metric-side compound impact via a 5-sample
   median A/B**. R46 will appear in the daemon canonical metric run;
   if the 5-round-following-R46 median lifts >= +1 score, validates
   the dispatcher-drift methodology end-to-end.

## Files touched this round

- `Primus-Turbo/scripts/_probe_round_46_gateup_b4_m2048_dgradH4_drift.py`:
  this round's empirical probe (kept; useful as a template for R47).
- `Primus-Turbo/primus_turbo/pytorch/kernels/hipkitten/config.py`:
  edit at lines 3022-3105 — `num_xcds=None` → `num_xcds=4`, R46
  comment block added below the existing R8/R10/R15/R16 stack.
- `Primus-Turbo/analysis/_notes/round-46-gateup-b4-m2048-dgradH4-xcds-drift-SHIP-1pct-wmin-beats-lmax.md`:
  this file.

No HK kernel edits.
