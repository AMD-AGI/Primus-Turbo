---
name: round-27-no-op-r22-r23-protocol-active
description: R27 discharge under the R23 no-op-by-design cascade — override-criterion + hard-stop checks both fail, default no-op action commits this forward-pointer note, production NEUTRAL
type: project
---

# Round-27 — gpt_oss FP8 kernel-only auto-opt: no-op-by-design (R22+R23 protocol active)

**Date**: 2026-05-09 (UTC)
**Contract origin**: R23 (commit `0fe8e200`) bound R24-R55 fresh-context Claudes
to a no-op cascade discharging each round's budget under the R22+R23 daemon-
transition recommendation, unless a NEW lever family surfaces with a 200-LOC
single-round budget plan AND a >+15 score EV claim with witnessed evidence.
**Production effect**: NEUTRAL (docs-only; no kernel/dispatcher/binding/PT edits).

## Verdict

R22+R23 daemon-transition recommendation **still in force**. No new evidence
surfaced this round. R27 ships zero code. Single-paragraph forward-pointer per
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
R26 daemon read = **691**, +24 below threshold. Hard-stop NOT triggered. No-op
protocol continues.

## Streak / patience tracking

- Last `improved=True`: R15 FUSED_KTAIL ship (`best=695`, sha `5e20a3e1`).
- Current streak without improvement: **24 rounds** (R16-R26, +R27 in progress).
- Daemon `patience=40` (per `auto_optimize_gpt_oss_fp8.py`).
- Patience-remaining: 40 − 24 = **16 rounds** (auto-stop at R44 if streak holds).
- Recent daemon metrics (last 7 rounds, all NEUTRAL docs commits):
  R20=692, R21=693, R22=691, R23=691, R24=694, R25=691, **R26=691**. All within
  R29 noise envelope (cluster median 697 ± 3-4σ). 12-round window mean Δ vs
  cluster median 697 ≈ **−5.0 score**, statistically indistinguishable from zero
  true effect at σ≈3-4. R27's measurement (whatever the daemon reads) extends
  this window by one sample without changing the conclusion.

## What R27 explicitly does NOT do

- Does NOT edit `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  or any `*_fastpath.inc`.
- Does NOT edit `primus_turbo/pytorch/kernels/hipkitten/config.py` or `loader.py`.
- Does NOT add any `_probe_round_27_*.py` (no probe — no new direction to probe).
- Does NOT re-test any FALSIFIED lever from R22's 12-family closure table
  (negative-EV per closure-evidence base of 9-sample multi-sample to ~25-round
  per-cell sweeps).
- Does NOT propose a new direction outside the SKILL.md A-G taxonomy (none surfaced).
- Does NOT run `dbg_remote.sh` (no probe to measure; daemon's canonical metric
  read after this commit will be the only measurement and it is expected to land
  inside the R29 noise envelope).

## Files touched this round

- This note (single-file commit, ~70 LOC).

## Forward-pointer (R28)

R28 fresh-context Claude: re-read `round-22-fp8-daemon-transition-recommendation.md`
+ `round-23-fp8-daemon-transition-reaffirmed-no-op-by-design.md` + `round-24-no-op-r22-r23-protocol-active.md`
+ `round-25-no-op-r22-r23-protocol-active.md` + `round-26-no-op-r22-r23-protocol-active.md`
+ this note. Apply the same override-criterion + hard-stop checks. If still no
override (default expectation), commit the analogous one-paragraph note
`round-28-no-op-r22-r23-protocol-active.md`. Patience-remaining at R28 entry =
15 rounds.

The no-op cascade continues until either (a) a manual daemon stop, (b) the
patience-40 trigger fires automatically at R44, or (c) an override criterion is
met. The R23 hard-stop criterion (`metric > 715` single-sample at HEAD) was not
triggered by R26's `metric=691` and is unlikely to be triggered by R27's
expected NEUTRAL read.
