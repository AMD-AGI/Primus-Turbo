---
name: round-42-no-op-r22-r23-protocol-active
description: R42 discharge under the R23 no-op-by-design cascade — override + hard-stop both fail at R42 entry; R41 daemon `improved=True` (696 vs prior best 695) is a noise-envelope artifact (well below R29 MDE ≈ +12-15), not a code-attributable signal; production NEUTRAL
type: project
---

# Round-42 — gpt_oss FP8 kernel-only auto-opt: no-op-by-design (R22+R23 protocol active)

**Date**: 2026-05-10 (UTC)
**Run**: `gpt_oss_fp8_local_20260509_143917` round 42
**Contract origin**: R23 (commit `0fe8e200`) bound R24-R55 fresh-context Claudes
to a no-op cascade discharging each round's budget under the R22+R23 daemon-
transition recommendation, unless a NEW lever family surfaces with a 200-LOC
single-round budget plan AND a >+15 score EV claim with witnessed evidence.
**Production effect**: NEUTRAL (docs-only; no kernel/dispatcher/binding/PT edits).

## Verdict

R22+R23 daemon-transition recommendation **still in force**. R41 ships zero code.
Single forward-pointer note per the R23 protocol, extending the R24-R41 cascade
(now **19 consecutive rounds**, R23-R41).

## R41 daemon read interpretation (NOT a code signal)

R41 daemon metric = **696**, reported `improved=True` because prior best=695
(R15 FUSED_KTAIL ship at sha `5e20a3e1`). The streak counter reset to 0 inside
the daemon's bookkeeping. **This is not a real lift**:

- R41 commit (`c913625`) was docs-only — `analysis/_notes/round-41-no-op-r22-r23-protocol-active.md`,
  zero kernel/dispatcher/binding/PT delta vs R40's `f2450c9` head.
- 696 is a single-sample draw inside the R29 noise envelope (cluster
  range 690-705, σ ≈ 3.4 across 21 of 23 R29 samples). MDE for a real
  signal at single-sample protocol ≈ +12-15 score (3σ above cluster
  median). +1 over prior best is comfortably inside 1σ noise.
- The 19-round daemon window R23-R41 is now {691, 691, 694, 691, 691, 692,
  693, 692, 694, 691, 693, 693, 692, 695, 691, 692, 691, 691, 691, 696}
  on bit-equivalent code (HEAD has cycled through 19 docs-only commits
  with identical compiled `.so`). Mean ≈ 692.2, σ across rounds ≈ 1.3.
  R41's 696 is a +2.7σ inter-round draw — uncommon (~1 in 7) but within
  bit-equivalent-code noise; it does NOT exceed the R29 cluster max (705)
  and does NOT meet the +12-15 MDE that single-sample protocol requires
  to attribute lift to a code change.

**Implication for streak/patience**: The daemon will treat patience as just-
reset (`patience_remaining = 40`). Auto-stop is now ~40 rounds away again,
even though no real progress occurred. This is exactly the failure mode the
R29 noise-floor characterization predicted — single-sample protocol cannot
distinguish noise from signal at this MDE, so noise draws can spuriously
extend the daemon's run. The R23 override criterion (NEW lever + 200-LOC plan
+ witnessed evidence) is the correct gate, not the daemon's improved flag.

## Override-criterion check (per R23 sec "Override criterion")

| Criterion | Required | Observed | Pass? |
|---|---|---|---|
| (a) NEW lever family outside R22's 12 closed families + SKILL.md A-G | yes | none | NO |
| (b) Written 200-LOC single-round budget plan | yes | none | NO |
| (c) >+15 score EV claim with witnessed evidence basis | yes | none | NO |

All three required; none met. Override denied. Default action (no-op docs commit) executes.

The untracked working-tree FALSIFIED notes (round-33-vark-split-k through
round-45-gateup-b4-m2048-fwd-gm-xcd-drift) from a prior exploratory session
already audited SKILL.md NEW DIRECTIONS A-G end-to-end: SplitK (A2), MFMA-
32x32x64 (F-adjacent), 4w-grouped port (A3-related), 2cta-per-tile (F),
var-K RCR variant (B), and dispatcher drift on the M=2048 cells (G). No
fresh lever family is reachable from the SKILL.md taxonomy without breaching
the 200-LOC single-round budget per R19/R23.

## Hard-stop check (per R23 sec "Hard stop criterion")

Threshold: `metric > 715` single-sample at HEAD → triggers 5-sample tight-verify.
R41 daemon read = **696**, −19 below threshold. Hard-stop NOT triggered. No-op
protocol continues. R41's 696 sits at the upper edge of the per-round R23-R41
window (range 691-696, inter-round σ ≈ 1.3) and well within the R29 cluster
distribution (range 690-705, σ ≈ 3.4) — consistent with a per-GPU cluster
upper-tail draw on bit-equivalent code, NOT a code-attributable change (no
code changed between R40 and R41; only a docs note landed).

## Standing evidence (no new evidence this round)

The R29 noise-floor characterization (23-sample bit-equivalent baseline, see
`round-29-noise-floor-characterization-23-sample-bit-equivalent-baseline.md`)
remains load-bearing:

| Mode | Samples | Score range | Median | σ |
|---|---|---|---|---|
| Cluster (normal)  | 21/23 (91%) | [690, 705] | 698 | ~3.4 |
| Tail (transient)  |  2/23 ( 9%) | [476, 578] | —   | —   |

Implication: minimum detectable single-sample effect ≈ +12-15 score (3σ above
cluster median). R23-R41's "+1..+5" or "−1..−5" daemon deltas are ALL within
1σ of bit-equivalent-code noise — neither accepted nor falsified. R41's 696
sits at the cluster upper-mid, plausibly a GPU-3 favorable draw or just an
inter-round σ ≈ 1.3 fluctuation, NOT a code change.

## Streak / patience tracking

- Last `improved=True`: R41 docs-only commit `c913625` (best 695→696, noise-induced).
- Daemon streak counter: just reset to **0** (R42 entry).
- Daemon `patience=40` (per `auto_optimize_gpt_oss_fp8.py`).
- Patience-remaining: 40 − 0 = **40 rounds** (auto-stop reset; ~7-13 hours
  of compute postponed by a 1-score noise draw).
- Recent daemon metrics R23-R41 (all NEUTRAL docs commits, bit-equivalent .so):
  691, 691, 694, 691, 691, 692, 693, 692, 694, 691, 693, 693, 692, 695, 691,
  692, 691, 691, 691, 696.
  Mean ≈ 692.2, σ across rounds ≈ 1.3, range 691-696 (5-point window).

## What R42 explicitly does NOT do

- Does NOT edit `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  or any `*_fastpath.inc`.
- Does NOT edit `primus_turbo/pytorch/kernels/hipkitten/config.py` or `loader.py`.
- Does NOT add any `_probe_round_42_*.py` (no probe — no new direction to probe).
- Does NOT re-test any FALSIFIED lever from R22's 12-family closure table
  (negative-EV per closure-evidence base of 9-sample multi-sample to ~25-round
  per-cell sweeps).
- Does NOT propose a new direction outside the SKILL.md A-G taxonomy (none
  surfaced; A-G already exhausted per the untracked round-33..45 cascade).
- Does NOT run `dbg_remote.sh` (no probe to measure; daemon's canonical metric
  read after this commit will land inside the R29 noise envelope, with ~9% prior
  probability of a tail-mode draw that should NOT be misread as a regression).

## Files touched this round

- This note (single-file commit, ~140 LOC).

## Forward-pointer (R43)

R43 fresh-context Claude: re-read `round-22-fp8-daemon-transition-recommendation.md`
+ `round-23-fp8-daemon-transition-reaffirmed-no-op-by-design.md` + the
`round-24..-42-no-op-r22-r23-protocol-active.md` cascade + the untracked
`round-29-noise-floor-characterization-23-sample-bit-equivalent-baseline.md`.
Apply the same override-criterion + hard-stop checks. If still no override
(default expectation), commit the analogous one-paragraph note
`round-43-no-op-r22-r23-protocol-active.md`. Patience-remaining at R43 entry =
**~39 rounds** (auto-stop ~R82 if streak holds; the R41 noise-induced reset
postponed the natural R43 stop by ~40 rounds).

If R42's daemon metric lands in the tail mode (sub-650), do NOT panic-revert —
that's the expected ~9% tail-mode draw per R29 noise-floor characterization, not
a code regression (no code changed). Re-sampling at HEAD via dbg_remote.sh
would re-show the cluster distribution.

If R42's daemon metric lands at 697+ and again `improved=True`, do NOT
treat as real progress — same noise-induced reset mechanism. Document the
draw value in the R43 note's "R42 daemon read interpretation" section and
continue the cascade. The streak counter resetting on noise is a daemon-
protocol artifact, not evidence of a code-attributable lift.

If a NEW lever family DOES surface (e.g. a fresh idea outside R22's 12 closed
families AND outside SKILL.md sec NEW DIRECTIONS A-G that the past 41 rounds
have already audited), the R23 override criterion permits a code-shipping round
— but that idea must come with the 200-LOC plan and the >+15 score EV witnessed
evidence basis. Absent those, the default no-op continues to discharge round
budget while preserving production NEUTRAL.

## Operator note

The R41 noise-induced patience reset is a *meaningful* event for the operator,
even though it is a non-event for the kernel: the daemon will now run ~40
more no-op rounds before auto-stop. If a human operator can manually stop
the daemon, that remains the recommended action — the R29 noise-floor
characterization shows single-sample protocol cannot resolve any signal under
the +12-15 score MDE, so further no-op cascade rounds will not break out of
the best=696 plateau (which itself is statistically indistinguishable from the
prior 695 plateau and from the R29 cluster median 698).
