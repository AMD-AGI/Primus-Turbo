# Round 5 plan — selected r5.i1.g13 (L6 cross-tile softmax/QK pipeline, ported onto the round-4 champion)

## Where the round stands
Prod runs at 1083.33 TF/s against beat's 1410.87 (0.768) (`../1-profiling/benchmark-results.md`). It is
latency-bound and core-clock-scaled; HBM is excluded by the data swap: sclk 988 → 1053 MHz, time −7.2%,
Mcycles/call 1.964 → 1.941 (`../1-profiling/5-power-wall-analysis/analysis.md`). The remaining gap is per-wave
execution. Beat hides the adjacent tile's softmax VALU inside its WMMA gaps; current does not
(`job_context/profiling/beat/4-thread-trace/analysis.md`, route h4). Missing data: 3-kernel-metrics and
4-thread-trace of current were skipped (`../1-profiling/profiling.yaml` coverage). Stall/issue attribution is
closed on this chip (correction 5), so every price in this round has to come from the card.

## Decision
**r5.i1.g13**: port the h21 pipeline diff (not the proto files, h27) onto op/current. It compiled in plan at
512 VGPR / 97 SGPR, 0 spill, 0 scratch, with 103/330 v_exp inside WMMA gaps (champion 0/264)
(`synthesis_notes.md` S1, `raw/vgpr_probe/`).

**Prediction:** prod Mcycles/call at randn 1.964 → ≤1.905, prod vs_champion ≥1.03 over 3 slot-rotated
sessions, sclk within ±2%, output bitwise equal. The reviewer predicts ≥1.05 for the same arm. L6 closes only
below the session floor (~0.4%).

**Route row 1 is h22 (must) → r5.i2.g14.** Act builds it as a **separate arm** from op/current, in the same
sessions. The two are not bundled and are not merged this round, because g13 has no VGPR headroom. The arm with
the higher prod result ships if it clears the floor (correction 4). If g14 wins alone, g13's prediction is
recorded as falsified. g15 is an optional third arm that closes L13 with a predicted null.

## What was argued ([dialogue.md](dialogue.md))
- [00_analyse](dialogue.md#00_analyse----planner) / [reviewer](dialogue.md#00_analyse----reviewer): both agents
  independently proposed the same two mechanisms, with crossed ids
  (`synthesis_notes.md` S2).
- [01_synthesis](dialogue.md#01_synthesis----planner): the order [g13, g14, g15]. g13 first, because its VGPR
  risk was measured away in plan.
- [02_review](dialogue.md#02_review----reviewer) → [03_response](dialogue.md#03_response----planner):
  - Static nop/msb counts are withdrawn as a price (correction 3).
  - "Prediction false" and "mechanism closed" are split onto separate lines.
  - g14's inconsistent band is fixed.
  - g15 takes the reviewer's null prediction.
  - g04 goes back to still_valid / deferred.
- [04_close](dialogue.md#04_close----reviewer): g15 and g04 confirmed. The g13 and g14 magnitudes remain
  **unresolved between the agents** (`ended_by: unresolved`).

## Open (settled next by act's numbers)
| id | planner | reviewer | settled by |
|---|---|---|---|
| r5.i1.g13 | ≥3% | ≥5% | ≥5% → reviewer; 3–5% → planner; floor–3% → both wrong, still ships; < floor → L6 closed at n_block 64 |
| r5.i2.g14 | 1–2% | ≥2% | ≥2% → reviewer; 1–2% → planner; floor–1% → both wrong, still ships; < floor → per-tile cross-lane sum off the critical path |

## Instrument taken in this round (not filed to the pool)
The agreed g04 line (fast, warm vs flushed, current and beat) was measured in plan:
[`raw/warm_flush/summary.md`](raw/warm_flush/summary.md).
- **The agreed line settles neither way.** The back-to-back ratio is 1.19–1.24, between the thresholds
  (≤1.15 stale, ≥1.3 stays).
- **L2 state is irrelevant** for both arms (qwarm ≈ qflush), and host launch is equal (16.4 vs 16.6 µs).
- **Most of the scored fast gap is a per-dispatch penalty.** About 36 of ~41 µs is paid only by current, and it
  grows with the idle time before the call: 38 µs back to back, 71 µs after a 1.8 ms sleep. Beat stays at
  32–35 µs throughout.
- **This is r1.i3.g03's penalty**, now shown to be neither L2 nor host-side.

For reflect: g04's justification (underfill) explains at most the ~6–7 µs warm gap. g03's untaken lever
(`s_prefetch_inst` / warm-up, from its round-2 note) targets the ~36 µs. It is fast-only; prod exposure is
~22 µs in ~2 ms. It never ships against a prod loss.
