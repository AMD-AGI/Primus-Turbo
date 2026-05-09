# Round-22 — gpt_oss FP8 kernel-only auto-opt: daemon-transition recommendation

**Date**: 2026-05-09 (UTC)
**Contract origin**: R19 (commit 40a2822d) bound R20+R21 to close direction G's
two single-knob axes (chunk_size, num_slots); R20+R21 both NEUTRAL/LOSS;
R21 closing note (commit 6627256c) forward-pointed R22 as the
daemon-transition recommendation per the R44 closure pattern. This note
discharges that contract.
**Production effect**: NEUTRAL (docs-only; no kernel/dispatcher edits).

## TL;DR

The single-round auto-opt search space for `gpt_oss_fp8_kernel_score` is
**empirically exhausted**. All seven SKILL.md NEW DIRECTIONS (A, A3, B,
C, D, E, F, G) are FALSIFIED or out-of-scope; the per-cell dispatcher
and the kernel macro flags are likewise exhausted. The score has sat at
median **697 ± 3** for 19 consecutive rounds (R03 → R21), with `best=695`
reflecting upper-tail noise rather than a real optimum.

**Recommendation**: terminate the auto-opt daemon for this metric. The
remaining 6.6 pp gap to the 55.6 % MFMA-peak target is gated by
kernel-template / producer-consumer rewrites that explicitly exceed the
single-round budget (R19 binding constraint). Continued daemon spend
yields zero positive EV per round and consumes Claude tokens at ~5–15
minutes / round.

## What is closed (and where it was closed)

Cross-cited from R21 closure summary (`round-21-direction-G-cross-shape-co-opt-FALSIFIED-on-both-axes.md`)
and reverified against the auto-opt-logs index for this run:

| Direction | Status | Closing round(s) |
|---|---|---|
| A  (Stream-K)                              | FALSIFIED (round-budget vs impl-cost) | R7 variant-1, R12–R19 variant-2 |
| A3 (decoupled-warps)                       | PREFLIGHT FALSIFIED                    | R6 |
| B  (cross-stream parallelism)              | A-PRIORI FALSIFIED                     | R5 (metric times sections serially) |
| C  (activation cache reuse)                | OUT OF METRIC SCOPE                    | R5 (metric pre-quantizes) |
| D  (SALU coord-decode)                     | FALSIFIED                              | R9–R10 (PMC re-attribution) |
| E  (alt barrier scheme — single drops)     | FALSIFIED                              | R26–R28 multi-sample |
| F  (larger tiles)                          | PREFLIGHT FALSIFIED                    | R8 (AGPR threshold × 4-acc/FUSED_KTAIL coupling) |
| G  (cross-shape co-opt single-knob)        | FALSIFIED both axes                    | R20 chunk_size, R21 num_slots |
| Per-cell dispatcher (gm/xcds/slots/cs)     | EXHAUSTED                              | R1–R45 (~25 rounds) |
| Macro flags (UNROLL/HOIST/DRAIN/BARRIER)   | EXHAUSTED multi-sample                  | R22–R28 |
| Kernel-template inventory swap (existing)  | FALSIFIED                              | R19 inventory probe |
| Stream-K caller-allocated workspace path   | INFRA SHIPPED, dormant @ sk_split_n=0   | R12–R17 |

Total directions closed: 12 distinct lever families. No untried lever
remains in the SKILL.md taxonomy that fits the single-round auto-opt
budget envelope (≤15 min wall / ≤200 LOC / single build).

## Plateau characterisation (evidence basis)

Score history, this auto-opt run (`gpt_oss_fp8_local_20260509_143917`):

| Round | metric | best  | Δ vs best | improved | Notes |
|---|---|---|---|---|---|
| 14  | (not in window) | 695 | — | — | — |
| 15  | n/a | 695 | — | — | FUSED_KTAIL ship |
| 16  | 693 | 695 | −2  | False | docs |
| 17  | 692 | 695 | −3  | False | infra (workspace cache scaffold) |
| 18  | 692 | 695 | −3  | False | docs (A1' EV re-anchor) |
| 19  | 694 | 695 | −1  | False | docs (A1' FALSIFIED) |
| 20  | 692 | 695 | −3  | False | docs (G chunk_size FALSIFIED) |
| 21  | 693 | 695 | −2  | False | docs (G num_slots FALSIFIED) |

**19 consecutive rounds without `improved=True`.** The R29 noise-floor
characterisation (23-sample bit-equivalent baseline) puts the
single-sample minimum-detectable Δ at **+12–15 score** at GPU-3 on this
host (cluster median 697-700, σ≈3-4). The `best=695` historic anchor
is below the current cluster median because it pre-dates several
score-positive landings (R15 FUSED_KTAIL gating, R18 dispatcher fast-path
extension); the auto-opt `best` book-keeping is sticky-low.

Headroom to target (per task md):

| Section | current avg (T) | target avg (T) | gap (T) | gap (% peak) |
|---|---:|---:|---:|---:|
| fwd   | ~1898 | 2800 | 902  | 17.9 % |
| dgrad | ~2097 | 2800 | 703  | 14.0 % |
| wgrad | ~1807 | 2800 | 993  | 19.7 % |

Score 697 → 900 = **+203 score = ~+30 % per-section TFLOPS uplift on
average**. The PMC bottleneck (R21 IDENTIFIED, retained in R29) is
**MFMA-pipe under-feed at the CTA-barrier-per-iter schedule pin**:
60-70 % cycles idle, MfmaUtil 31-49 %, MemStall < 1 %. No flag, no
dispatcher, no existing template, and no single-knob global override
moves this — multi-sample falsification covered all of the above.

## Why the remaining levers are out of single-round budget

1. **Producer-consumer warp decoupling** (unified A3+E rewrite). 4–6
   rounds: kernel restructure to dedicate 1–2 warps to HBM→LDS load,
   replace CTA-wide `s_barrier()` with intra-warp-group sync. New register
   layout, new prologue/epilogue, new SNR matrix, ~400 LOC. Each round
   risks build break + correctness regression that aborts the auto-opt
   metric harness.
2. **New 256×384 / 512×256 tile template variant** (direction F revisited).
   3–5 rounds: register/AGPR re-budgeting, LDS-tile reshape, new fastpath
   .inc, register the binding in `kernel_fp8_layouts.cpp`. R8 PREFLIGHT
   identified the AGPR-threshold trap that single-round budget cannot
   safely defuse.
3. **MFMA 32×32×64 main-loop port** (R41 re-examination). 4+ rounds:
   only a 2 % EV ceiling per R41, and that ceiling falls below the noise
   floor of this metric — net not worth even with infinite budget unless
   bundled with (1) or (2).

R19 codified the single-round budget cutoff:
> "200-LOC MFMA-pipeline surgery + atomicAdd path + separate reduction
> kernel + multi-build verify cycle exceeds single-round auto-opt wall
> budget; 5 prior infra deferrals R12-R17 confirm empirically."

The same logic binds (1)–(3): each requires 4+ consecutive rounds of
cooperative scaffold→build→verify with an intermediate state that does
NOT improve `best`, which the patience-40 daemon will tolerate but
provides no constructive guidance to a fresh-context Claude per round.
A multi-day **human kernel-engineer** track is the right venue.

## Recommendation

### Primary (recommended)

**Stop the daemon for `gpt_oss_fp8_kernel`.** Rationale:
- 19/19 rounds without `improved=True` since R03 (FUSED_KTAIL ship was last positive).
- All taxonomy directions closed; no candidate lever passes the
  EV-vs-budget filter.
- Per-round token cost (≈30k input + 5k output × Opus 4.7) buys zero EV.

If the daemon must continue running for orchestration reasons, the
patience-40 cap will auto-trigger at round 43 (current streak 19 →
patience exhausted at round +21). **Recommend manually stopping at the
end of this round (R22)** rather than waiting for patience exhaustion.

### Secondary (if a different metric is wanted)

Pivot the daemon to a different scoring target where single-round levers
remain open. Candidates the bf16-grouped track left untouched
(per `round-96-bf16-grouped-dispatcher-exhaustion-qwen-down-and-dsv3-gateup-var-k-falsified.md`):

- `_metric_grouped_only.py` (24-shape kernel-only) — currently ~1000
  cap; per task md, must stay ≥ 990. Actively defended, low room to
  improve, but a valid baseline-defence regression-monitor target.
- A new fp8 metric anchored on the **GateUP-B32-M4096 dgrad** cell (the
  best per-shape current absolute T) — could measure ceiling pressure
  on a single shape rather than the 8-shape mean. Outside this auto-opt
  task contract; would require human review before harness change.

### Tertiary (long-haul human-engineer track — out of auto-opt scope)

Bundle the three over-budget levers into a project-track PRD:
1. Decoupled-warps producer-consumer kernel rewrite (unified A3+E).
2. 256×384 large-tile template (direction F redo with AGPR-aware design).
3. MFMA 32×32×64 main-loop port (folded in with #1 to avoid the R41
   stand-alone 2 % ceiling).

Estimated combined upside if all three land: **+150 to +200 score**
(reaching the 850–900 target band). Estimated combined work:
**6–10 engineer-weeks** with full PMC + bit-eq verification per ship.
Not auto-opt-able.

## Files

- This note.

## Production effect

NEUTRAL — no kernel/dispatcher/binding/PT edits. Pure documentation
discharging the R19 → R21 → R22 contract chain.
