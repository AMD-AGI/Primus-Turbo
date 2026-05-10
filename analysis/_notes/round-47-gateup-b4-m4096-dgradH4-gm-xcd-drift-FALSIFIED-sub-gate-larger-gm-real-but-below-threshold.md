# Round 47 — GateUP-B4-M4096 dgrad-via-H4 (gm, xcds) drift FALSIFIED (sub-gate)

## TL;DR

R47 PRIMARY (per R46 forward pointer): methodology-corrected (gm, xcds)
drift audit on the **GateUP-B4-M4096 dgrad-via-H4 RCR rule** at
`config.py:2603-2617` (`tiles_m == 16` branch, currently
`group_m=1, num_xcds=4, fuse_ktail_off=0`, no slots/cs override → kernel
defaults NUM_CUS=256, chunk=64). This rule had NEVER been (gm, xcds)
swept under fixed production levers — only chunk_size was probed (R15-4).

**Verdict: FALSIFIED.** Two independent 5-seed × 2000-iter p20 runs.
Best non-baseline cells (`(8, 4)` and `(16, 4)`) consistently beat
baseline `(1, 4)` by **+0.39% to +0.57%** on dgrad-via-H4 with
`wmin_strict=YES` — the **direction** (gm: 1 → ~8/16) is a real but
sub-gate signal. Below the R47 PRIMARY ship gate of ≥1.0%.

## Probe data (5 seeds × 2000-iter p20, GPU 3 MI355X)

### Run 1: GateUP-B4-M4096 dgrad-via-H4 (PRIMARY)

| cell | med ms | min ms | max ms | TFLOPS | Δ% | wmin>lmax |
|---|---|---|---|---|---|---|
| (1, 4)* | 0.2168 | 0.2164 | 0.2180 | 2507.3 | +0.00 | base |
| (1, 8) | 0.2244 | 0.2242 | 0.2248 | 2421.9 | -3.52 | no |
| (1, 2) | 0.2174 | 0.2171 | 0.2175 | 2500.8 | -0.26 | YES |
| (1, 1) | 0.2247 | 0.2244 | 0.2250 | 2418.9 | -3.65 | no |
| (2, 4) | 0.2192 | 0.2188 | 0.2194 | 2480.3 | -1.09 | no |
| (4, 4) | 0.2192 | 0.2189 | 0.2194 | 2479.8 | -1.11 | no |
| **(8, 4)** | **0.2157** | 0.2153 | 0.2159 | **2519.8** | **+0.50** | **YES** |
| **(16, 4)** | **0.2158** | 0.2156 | 0.2159 | **2518.4** | **+0.44** | **YES** |

### Run 2: GateUP-B4-M4096 dgrad-via-H4 (PRIMARY) — confirmation

| cell | med ms | min ms | max ms | TFLOPS | Δ% | wmin>lmax |
|---|---|---|---|---|---|---|
| (1, 4)* | 0.2179 | 0.2177 | 0.2186 | 2494.9 | +0.00 | base |
| (1, 8) | 0.2256 | 0.2254 | 0.2258 | 2409.9 | -3.52 | no |
| (1, 2) | 0.2186 | 0.2184 | 0.2188 | 2486.6 | -0.33 | YES |
| (1, 1) | 0.2258 | 0.2255 | 0.2260 | 2407.3 | -3.64 | no |
| (2, 4) | 0.2203 | 0.2202 | 0.2204 | 2467.2 | -1.12 | no |
| (4, 4) | 0.2198 | 0.2196 | 0.2200 | 2472.6 | -0.90 | no |
| **(8, 4)** | **0.2170** | 0.2169 | 0.2171 | **2504.5** | **+0.39** | **YES** |
| **(16, 4)** | **0.2166** | 0.2165 | 0.2168 | **2509.1** | **+0.57** | **YES** |

**Cross-run consistency**: both `(8, 4)` and `(16, 4)` produce wmin_strict
positive lifts on both runs; the magnitude band is ~0.4-0.6%; the best
flips between the two (within probe noise). Direction (gm: 1 → 8/16,
xcds=4 unchanged) is a robust real signal but **magnitude < 1.0% gate**.

### Fwd column (diversity, NOT a drift test of the dgrad rule)

The fwd path under (B=4, M=4096, N=5760, K=2880) hits the GateUP-B4-M4096
**fwd RCR rule** at `config.py:1495-1623` (`tiles_n==22, tiles_m==16,
k==2880, m_total<=16384`) which is **already shipped at `(gm=8, xcds=4)`**
per Round-3 of current run (commit history Round-3 GateUP-B4-M4096 fwd).
The probe baseline `(1, 4)` was the wrong reference for fwd; (8, 4)
"winning" by +2.78% / +2.74% is the rule already at production.
Non-informative as a drift test.

## Decision: FALSIFIED, no rule change

R47 PRIMARY ship gate per R46 forward pointer: best cell wins ≥1.0% on
dgrad-via-H4 with wmin_beats_lmax. Best cell across two runs lifts
~0.45% — sub-gate. Per the conservative ship protocol (R43 shipped at
+1.01% / +1.95%), this falls below threshold. Hold rule at
`(group_m=1, num_xcds=4, fuse_ktail_off=0)`.

Sub-gate signal characterization:
- Score impact if shipped: dgrad section_avg moves +12T/8 = +1.5T →
  +0.5/2800 score weight ≈ **+0.18 score**, well below daemon ±2-3
  noise floor. No detectable lift.
- Direction (gm: 1 → 8/16) opposes the dgrad-via-H4 R43 sibling rule
  (GateUP-B4-M2048: gm=1, xcds=4) — the smaller cell prefers gm=1, the
  larger M=4096 cell shows weak preference for larger gm. Mechanism:
  larger m_total = larger persistent grid, and gm=8 batches more
  M-tiles/wave on the now-deeper grid. Same direction as the GateUP-B4
  fwd sibling above (gm=8 winner at M=4096 fwd vs gm=1 at M=2048 fwd).
- Cross-shape coherence: GateUP-B4-M4096 fwd (gm=8) + dgrad sub-gate
  preference for gm=8/16 hints that the M=4096 cells generally prefer
  larger gm than the M=2048 siblings on the current binding. But the
  signal is too weak to ship on its own.

## R44-R47 closure: B=4 RCR (gm, xcds) lever class definitively saturated

Per R46's plan: "If R47 PRIMARY also returns NEUTRAL/FALSIFIED, the
4-round chain (R44-R47) on B=4 RCR (gm, xcds) constitutes definitive
closure."

| Round | Rule cell | Verdict |
|---|---|---|
| R43 | GateUP-B4-M2048 dgrad-via-H4 | **xcds None→4 SHIP** (+1.0–1.95%) |
| R44 | Down-B4-M2048 fwd RCR | drift FALSIFIED ((16,2) at local opt) |
| R45 | GateUP-B4-M2048 fwd RCR | drift FALSIFIED ((1,4) at local opt) |
| R47 | GateUP-B4-M4096 dgrad-via-H4 | drift FALSIFIED (sub-gate +0.45%) |

All four B=4 RCR (gm, xcds) cells in the gpt_oss FP8 metric have been
swept under methodology-corrected production levers. One ship, three
falsifications (one with sub-gate real signal). The (gm, xcds)
dispatcher lever class on B=4 RCR cells is **empirically exhausted**
on the current binding.

## R48 forward pointer (substantive, NOT closure)

The (gm, xcds) lever exhausted on B=4. The next genuine attempt should
**transition to a new lever class**. Per SKILL.md NEW DIRECTIONS, the
recommended priority list:

### PRIMARY — NEW DIRECTION D (SALU coord-decode for var-K wgrad)

PMC evidence (SKILL.md table): `Down-B4-M2048 wgrad SALU=85.4% of
SQ_busy` (vs fwd 39.8%). The var-K persistent loop recomputes
`group-index + cumsum binary-search` every iter; hoisting this out
(rematerialize in regs, precompute CTA-static) frees SALU cycles for
MFMA issue. Estimated 2-3 rounds. Risk: register pressure (already 256
VGPR / 37 spill on var-K wgrad path).

This is the **highest-EV direction** identified in SKILL.md, attacks the
PMC-confirmed 85% SALU bottleneck on var-K wgrad (worst section, gap
2800 - 1748 = 1052 T = 37.6% remaining). Wgrad is the worst section
(progress 0.645) — biggest score lever.

Concrete first probe (R48): `_probe_round_48_vark_salu_decode_audit.py`
that disasms the var-K wgrad kernel via `roc-obj-ls` + `llvm-objdump
-d` and counts SALU ops in the main loop body, identifying which SALU
ops are loop-invariant. Output: a SALU-density map from which the
hoist target is selected for R49 implementation.

### BACKUP-1 — NEW DIRECTION F (larger tiles for M=4096 cells)

256x384 (1.5× N) or 512x256 (2× M) on the M=4096 cells where
register/LDS budget allows. M=2048 cells excluded (m_per_g=2048; 512-M
tile = only 4 tiles/group → too few). Investigate via PMC headroom
analysis on Down-B4-M4096 / GateUP-B4-M4096 first (need to verify
register/LDS budget before any kernel edit). Estimated 3-5 rounds.

### BACKUP-2 — Cross-shape co-optimization (NEW DIRECTION G)

A rule that loses -0.5% on shape A but gains +1.5% on shape B can be
falsified per-shape but win on the metric mean. The R47 sub-gate +0.45%
on GateUP-B4-M4096 dgrad-via-H4 is exactly this kind of micro-signal —
not enough alone, but if 4-6 such micro-signals can be batched into a
single rule edit, the cumulative score impact can clear the noise floor.
Probe: sweep all 8 cells × dgrad-via-H4 simultaneously and fit a
single (gm, xcds) per cell, then count how many micro-wins compound.

## Files touched this round

- `analysis/_notes/round-47-gateup-b4-m4096-dgradH4-gm-xcd-drift-FALSIFIED-sub-gate-larger-gm-real-but-below-threshold.md`:
  this note.
- `scripts/_probe_round_47_gateup_b4_m4096_drift.py`: probe script (5 seeds
  × 2000-iter × 8 cells × 2 sections, ~42s remote wall).

No `config.py` edits. No HK kernel edits. No `_metric_*` edits.

## Streak status

Pre-R47 streak: 5 rounds since last improved (R41-R46 not improved on
score). R47 expected NEUTRAL/sub-noise on score → streak 6. Patience
cap: 40. Plenty of budget for the R48 NEW DIRECTION D (SALU coord-
decode) audit to be the next genuine substantive attempt. Closure
documentation per R46 plan is now formally in place; daemon rotation
to a new lever class is recommended for R48.
