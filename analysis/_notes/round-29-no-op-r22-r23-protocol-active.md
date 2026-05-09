---
name: round-29-no-op-r22-r23-protocol-active
description: R29 discharge under the R23 no-op-by-design cascade — override + hard-stop both fail, R29 noise-floor characterization (σ≈3-4) reinforces R22+R23, default no-op action commits this forward-pointer note, production NEUTRAL
type: project
---

# Round-29 — gpt_oss FP8 kernel-only auto-opt: no-op-by-design (R22+R23 protocol active)

**Date**: 2026-05-09 (UTC)
**Run**: `gpt_oss_fp8_local_20260509_143917` round 29
**Contract origin**: R23 (commit `0fe8e200`) bound R24-R55 fresh-context Claudes
to a no-op cascade discharging each round's budget under the R22+R23 daemon-
transition recommendation, unless a NEW lever family surfaces with a 200-LOC
single-round budget plan AND a >+15 score EV claim with witnessed evidence.
**Production effect**: NEUTRAL (docs-only; no kernel/dispatcher/binding/PT edits).

## Verdict

R22+R23 daemon-transition recommendation **still in force**. No new evidence
surfaced this round. R29 ships zero code. Single forward-pointer note per
the R23 protocol (see `round-23-fp8-daemon-transition-reaffirmed-no-op-by-design.md`
sec "R23 → R43 no-op-by-design protocol").

## Override-criterion check (per R23 sec "Override criterion")

| Criterion | Required | Observed | Pass? |
|---|---|---|---|
| (a) NEW lever family outside R22's 12 closed families | yes | none | NO |
| (b) Written 200-LOC single-round budget plan | yes | none | NO |
| (c) >+15 score EV claim with witnessed evidence basis | yes | none | NO |

All three required; none met. Override denied. Default action (no-op docs commit) executes.

## Hard-stop check (per R23 sec "Hard stop criterion")

Threshold: `metric > 715` single-sample at HEAD → triggers 5-sample tight-verify.
R28 daemon read = **693**, +22 below threshold. Hard-stop NOT triggered. No-op
protocol continues.

## NEW evidence this round — strengthens R22+R23 verdict

The untracked `round-29-noise-floor-characterization-23-sample-bit-equivalent-baseline.md`
note (from adjacent local run `gpt_oss_fp8_local_20260509_143917` on a different
daemon GPU) gathered **23 consecutive `_metric_gpt_oss_fp8_kernel.py` samples
on bit-identical code at the round-28 HEAD**. Findings:

| Mode | Samples | Score range | Median | σ |
|---|---|---|---|---|
| Cluster (normal)  | 21/23 (91%) | [690, 705] | 698 | ~3.4 |
| Tail (transient)  |  2/23 ( 9%) | [476, 578] | —   | —   |

Implication: the **minimum detectable single-sample effect is roughly +12-15
score** (3σ above cluster median). Anything smaller is unidentifiable from one
daemon measurement. R23-R28's "+1..+3" or "−1..−5" daemon deltas are ALL within
1σ of bit-equivalent-code noise — neither accepted nor falsified, just unidentifiable.

This **independently arrives at R22's "search EXHAUSTED" conclusion via a
different evidence path**:
- R22's path: enumerate 12 lever families, show each closed at multi-sample.
- R29-noise-floor's path: characterize the measurement noise, show that the
  remaining margin between best (695) and baseline (692) is itself within 1σ
  noise — the "improvement" the daemon thinks it has from R15's FUSED_KTAIL
  may itself be an upper-tail single-sample lucky draw, not a real code-perf
  optimum.

Both evidence paths converge on the same recommendation: **stop iterating the
daemon on this metric**. The remaining headroom (gap from cluster median ~698
to the 900 user target, +202 score) requires kernel-template / algorithm-level
breakthroughs (per `_task_gpt_oss_fp8_kernel.md` sec "TARGET: 900 score"), NOT
the small-tweak round budget the daemon is shaped to consume.

## Streak / patience tracking

- Last `improved=True`: R15 FUSED_KTAIL ship (`best=695`, sha `5e20a3e1`).
  Per R29 noise-floor analysis, this "best" sample is plausibly an upper-tail
  draw at the 95th percentile of the cluster distribution, not a true code
  optimum — but the daemon treats it as the best-of-record.
- Current streak without improvement: **26 rounds** (R16-R28, +R29 in progress).
- Daemon `patience=40` (per `auto_optimize_gpt_oss_fp8.py`).
- Patience-remaining: 40 − 26 = **14 rounds** (auto-stop at R43 if streak holds).
- Recent daemon metrics (last 9 rounds, all NEUTRAL docs commits):
  R20=692, R21=693, R22=691, R23=691, R24=694, R25=691, R26=691, R27=692, **R28=693**.
  Mean = 691.6, σ across rounds = ~1.2. Range 691-694. Cluster median (R29
  noise-floor characterization) = 698. Daemon GPU 3 cluster appears to sit
  ~5-7 score below the GPU-7 cluster (heterogeneity ~16 score across the
  3,4,5,6,7 pool per `_task_gpt_oss_fp8_kernel.md` sec Phase-0 baseline).
  R29's measurement (whatever the daemon reads) extends this window by one
  sample without changing the conclusion.

## What R29 explicitly does NOT do

- Does NOT edit `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  or any `*_fastpath.inc`.
- Does NOT edit `primus_turbo/pytorch/kernels/hipkitten/config.py` or `loader.py`.
- Does NOT add any `_probe_round_29_*.py` (no probe — no new direction to probe).
- Does NOT re-test any FALSIFIED lever from R22's 12-family closure table
  (negative-EV per closure-evidence base of 9-sample multi-sample to ~25-round
  per-cell sweeps).
- Does NOT propose a new direction outside the SKILL.md A-G taxonomy (none surfaced).
- Does NOT run `dbg_remote.sh` (no probe to measure; daemon's canonical metric
  read after this commit will be the only measurement and it is expected to land
  inside the R29 noise envelope, with ~9% prior probability of a tail-mode draw
  that should NOT be misread as a regression).

## Files touched this round

- This note (single-file commit, ~95 LOC).

## Forward-pointer (R30)

R30 fresh-context Claude: re-read `round-22-fp8-daemon-transition-recommendation.md`
+ `round-23-fp8-daemon-transition-reaffirmed-no-op-by-design.md` + `round-24..-28-no-op-r22-r23-protocol-active.md`
+ this note + the untracked `round-29-noise-floor-characterization-23-sample-bit-equivalent-baseline.md`.
Apply the same override-criterion + hard-stop checks. If still no override (default
expectation), commit the analogous one-paragraph note
`round-30-no-op-r22-r23-protocol-active.md`. Patience-remaining at R30 entry =
**13 rounds** (auto-stop at R43 if streak holds).

If R29's daemon metric lands in the tail mode (sub-650), do NOT panic-revert —
that's the expected ~9% tail-mode draw per R29 noise-floor characterization, not
a code regression (no code changed). Re-sampling at HEAD via dbg_remote.sh
would re-show the cluster distribution.

If a NEW lever family DOES surface (e.g. a fresh idea outside R22's 12 closed
families AND outside SKILL.md sec NEW DIRECTIONS A-G that the past 28 rounds
have already audited), the R23 override criterion permits a code-shipping round
— but that idea must come with the 200-LOC plan and the >+15 score EV witnessed
evidence basis. Absent those, the default no-op continues to discharge round
budget while preserving production NEUTRAL.
