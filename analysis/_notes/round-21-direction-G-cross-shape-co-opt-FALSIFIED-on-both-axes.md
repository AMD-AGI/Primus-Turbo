# Round-21 — Direction G (cross-shape co-optimization) FORMALLY FALSIFIED on both single-knob axes

## TL;DR

Per R19 binding contract (commit 40a2822d), direction G's "single-knob
global override across all 8 gpt_oss cells with default values" sub-direction
was the last un-falsified structural lever. R20 closed the **chunk_size**
axis empirically; R21 closes the **num_slots** axis. Both end NEUTRAL or
LOSS at the canonical metric — no global override of these dispatcher
fields nets ≥+1 score (let alone the +12-15 noise-floor threshold per R29).

**Direction G = CLOSED.** Per R19's R22 contract:
> "If R20+R21 both NEUTRAL, R22 writes daemon-transition recommendation
> per R44 closure pattern."

## R20 chunk_size axis (commit 40a2822d HEAD, GPU 3, this session)

3-sample medians via `_probe_round_20_cross_shape_chunk_size.py`:

| chunk_size override | s1 | s2 | s3 | median | Δ vs baseline | verdict |
|---:|---:|---:|---:|---:|---:|---|
| baseline       | 698 | 697 | 696 | **697** | — | — |
| 16             | 684 | 684 | 682 | 684 | -13 | LOSS |
| 24             | 681 | 686 | 686 | 686 | -11 | LOSS |
| 32             | 693 | 688 | 692 | 692 | -5  | LOSS (within noise but consistent) |
| 48             | 688 | 685 | 684 | 685 | -12 | LOSS |

All values lose. cs=32 closest to baseline at -5 but still on the loss
side; consistent with R30 historical falsification on B=32 cells (-3% to
-38% per-cell on cs=16/24/32/48 over 5 seeds).

## R21 num_slots axis (this round, commit 40a2822d HEAD, GPU 3)

3-sample medians via `_probe_round_21_cross_shape_slots.py`:

| num_slots override | s1 | s2 | s3 | median | Δ vs baseline | verdict |
|---:|---:|---:|---:|---:|---:|---|
| baseline (kernel default, slots=0) | 698 | 697 | 696 | **697** | — | — |
| 64             | 428 | 427 | 428 | 428 | **-269** | CATASTROPHIC LOSS |
| 128            | 544 | 543 | 542 | 543 | **-154** | SEVERE LOSS |
| 256            | 691 | 690 | 695 | 691 | -6 | LOSS (within R29 noise) |
| 512            | 692 | 691 | 695 | 692 | -5 | LOSS (within R29 noise) |

slots=64/128 collapse the score because the chunked-partition fraction
shrinks below the kernel's L2-locality break-even (consistent with R30
falsified note's "block = xcds*cs, limit = (slots/block)*block"
arithmetic — at slots=64 with xcds=4, only 4 chunks fit before
round-robin tail dominates the schedule). slots=256/512 are at-baseline
within noise but show no positive signal.

The kernel's internal default at `cfg.num_slots == 0` already routes to
~slots=512-equivalent partitioning for these grids; the override merely
re-confirms that default. No global value beats the per-cell defaults.

## Why this closes direction G's "global override" sub-direction

Direction G as written in `_task_gpt_oss_fp8_kernel.md` describes
"Cross-shape co-optimization" — a rule that loses on shape A but gains
more on shape B. The empirical metric data above shows that ANY single
chunk_size or num_slots override globally REDUCES the score relative to
the per-cell-tuned baseline. The Pareto-trade hypothesis is empirically
falsified at the metric level for these two single-knob axes.

A higher-order direction-G variant — e.g. a *family-aware* rule (one
override for B=4 cells, a different one for B=32 cells, branching on
`m_total`) — is mechanically equivalent to a per-cell rule and therefore
already covered by the exhausted dispatcher rounds (R1-R5, R10-R13,
R23, R30, R34-R45 — all FALSIFIED at the noise floor).

## Combined R44/R45 + R19/R20/R21 closure summary

| Direction | Status | Closing round(s) |
|---|---|---|
| A (Stream-K) | FALSIFIED | R19 (variant-1 R7, variant-2 R12-R18) |
| A3 (decoupled-warps) | PREFLIGHT FALSIFIED | R6 |
| B (cross-stream parallelism) | A-PRIORI FALSIFIED | R5 (metric times sections serially) |
| C (activation cache reuse) | OUT OF METRIC SCOPE | R5 (metric pre-quantizes) |
| D (SALU coord-decode) | FALSIFIED | R9-R10 (PMC re-attribution) |
| E (alt barrier scheme) | sub-FALSIFIED via R26-R28 | R26-R28 (single-barrier drops all FALSIFIED) |
| F (larger tiles) | PREFLIGHT FALSIFIED | R8 (AGPR threshold × 4-acc/FUSED_KTAIL coupling) |
| G (cross-shape co-opt) | FALSIFIED both single-knob axes | R20-R21 (this round) |
| Per-cell dispatcher (gm/xcds/slots/cs) | EXHAUSTED | R1-R45 (~25 rounds, see R44/R45) |
| Macro flags (UNROLL/HOIST/DRAIN/BARRIER) | EXHAUSTED | R22-R28 (multi-sample) |

**Every direction in the SKILL.md NEW DIRECTIONS list (A-G) plus the
exhaustively-audited dispatcher and macro spaces are closed.** The
~700-score plateau is structural under the current 8-wave / RBN=64 /
256×256 production kernel template family.

## R22 forward pointer — daemon-transition recommendation

R19 specified the closure path: "R22 writes daemon-transition
recommendation per R44 closure pattern." Concretely R22 should:

1. **Document the structural ceiling** with the R29 noise model
   (cluster median ~697-700, σ≈3-4, single-sample +12-15 minimum
   detectable). The "best=695" tracking number reflects upper-tail noise,
   not a real optimum.
2. **Recommend daemon termination** for the gpt_oss FP8 kernel-only
   metric. The per-shape kernel-only TFLOPS is at 49% of MFMA peak
   (vs target 55.6%), and the remaining 6.6pp gap is gated by either
   (a) a new kernel template (256×384 / 512×256 / 4-wave variant),
   or (b) MFMA-pipe under-feed from CTA-barrier-per-iter restructure
   — both 4-6+ round investments and explicitly out of the
   single-round auto-opt budget per R19.
3. **Forward-point to follow-up tasks** the human kernel-engineer
   might pursue manually (multi-week timeline, not auto-opt-suitable):
   * Direction A3 + E re-imagined as a unified producer-consumer
     warp-decoupling kernel rewrite, replacing the CTA-wide barrier
     with intra-warp-group sync.
   * Direction F via a **new** kernel template variant (256×384 with
     explicit 4-acc layout that sidesteps the AGPR-threshold trap R8
     identified) — would require building the new template first
     before any dispatcher rule can route to it.

## Files

- `scripts/_probe_round_21_cross_shape_slots.py` — sister probe to R20.
- This note.

## Production effect

NEUTRAL — no kernel/dispatcher edits this round. Probe-only.
