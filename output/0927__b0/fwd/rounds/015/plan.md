# Round 15 plan

**Selected: `r15.i5.g59`.** It defers the per-tile cross-lane row-sum to the epilogue on the m32x8 body.

**Prediction:** prod x1.004-1.010 vs op/current, over 3 rotated sessions, each with a champ2 A/A.
- The gain should arrive through the clock: cycles/XCD about flat (x0.99-1.005), ticks/ns up.
- Correctness: SQNR >= 49 dB and det 200/200. The result is not bitwise equal to op/current.
- If prod falls below the A/A floor in any session, it is not shipped (correction 4).
- The full prediction is in [plan.yaml](plan.yaml), and the candidate in [candidates.yaml](candidates.yaml).

## What the dialogue did

The full exchange is in [dialogue.md](dialogue.md).
- Two agents proposed five candidates. The step-1 renumbering is in [synthesis_notes.md](synthesis_notes.md) §1.
- The reviewer opposed `r15.i2.g56` (3-slot ring). At a power-bound shape a cycles-only saving comes back as clock,
  as g24 showed. The planner withdrew it.
- The reviewer conceded the planner's ceiling for `r15.i3.g57` (coalesced TDM, <= ~0.85%;
  [synthesis_notes.md](synthesis_notes.md) §2). g57 moved behind g59.
- The planner accepted both of the reviewer's instrument fixes:
  - g55 is decided on the cycles x clock product, with an explicit mixed outcome.
  - g58 has a committed prediction and uses W=128 as a control.
- The one open fact was why K and V blocks are floored at 64 KB.
- The dialogue closed with nothing held in dispute (`converged_at_close`).

## What the plan step measured

The plan depended on two instruments and one open fact, so this step took all three rather than filing them to the pool.
- Container: fa-g2, physical GPU 0003, idle before and after, with no new amdgpu dmesg lines.
- Raw data: [measure/](measure/). Summaries: [pmc_summary.txt](measure/pmc_summary.txt),
  [wall_summary.txt](measure/wall_summary.txt).

| instrument | result | verdict |
| --- | --- | --- |
| `r15.i1.g55`: cycles vs clock | l2hit: cycles x1.013/x1.006, ticks/ns x1.078/x1.069. halfkv: cycles x1.027/x1.025, ticks/ns x1.063/x1.064. The pmc product matches benchmark wall (x1.063-1.068, x1.036-1.037) within 0.5% | **Energy branch.** All of the prod gain is clock, and cycles went slightly up. Latency hiding (g56) is dead at prod |
| `r15.i4.g58`: K/V window W | W=8 x1.003-1.009; W=32 x1.003-1.008; W=128 (control) x0.998-1.004 against champ2 A/A x0.999-1.002; W=2 (l2hit) x1.063-1.068 | **Capacity.** The reviewer's ">= x1.04 at W=32" is falsified. Cutting the footprint 16x buys <= 0.9%. Only a 2-tile working set pays, and no correct schedule has one. Locality levers are closed at prod |
| MIN_KV_BLK_BYTES = 0 | bitwise = op/current at fast, proxy and prod; ut PASS 16/16; prod x0.997-1.000 against A/A x0.999-1.002 | **The floor is vestigial.** Removing it frees 128 KB of LDS at no cost, which unblocks g57's slot budget |

## Why g59

After these measurements, prod pays only for **removed energy**: fewer bytes past L2, or fewer issued ops.
- halfkv priced bytes at about 3.6% for half the K/V traffic. But the correct-output way to drop bytes, multicast
  (r14.i3.g54), lost its gain to the cluster pairing.
- g59 removes issued work on every tile, with one edit and no LDS change. It is the family that has paid at prod
  (g09, g10, g14).
- g57 (fewer requests, same bytes) has a lower ceiling. It waits in the pool for a probe that halves requests while
  keeping the bytes.

## What remains open

- **g57's premise:** is request count priced at all when the bytes are kept? The settle is a probe that halves the
  requests with the same bytes, read at prod (>= +0.9% to take g57).
- **Byte removal with correct output:** multicast lost its x1.023 merge gain to a 1.7% pairing cost (r14). A cheaper
  pairing is the only open road to the ~3.6% that halfkv priced. It has no candidate id yet. The route notes it for
  next round.
- **The stall-vs-issue split stays closed** (correction 5). No counter was taken for it.
- **Not covered by this round's profiling:** 3-kernel-metrics, 4-thread-trace and 5-power-wall were skipped
  (profiling.yaml coverage). There are no byte, WMMA or power counters. The energy conclusion rests on GRBM
  ticks/ns, which is an inferred clock (facts.md).
