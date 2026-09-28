# Round 10 plan

## Chosen: r10.i2.g24, the 4-wave × 2 WG/CU (occ2, all-LO) body for prod-sized grids

This ports the GPU-2 lab's `r6_occ2_lo` onto op/current (h39) behind a shape gate. Grids with at least 4 WGs/CU on the 8-wave grid take the new body. Of the benchmark shapes, that is prod (16/CU). Proxy (2/CU) and fast (m16x8 path) keep today's kernels, byte for byte.

The occ2 body:
- 4 waves, 2 rows each, BLOCK_M 128;
- 160 KB of LDS, so 2 WGs fit on a CU;
- waves_per_eu=2;
- every wave takes the LO order (K load before prefetch).

**Prediction:**
- Prod cycles: GRBM_GUI_ACTIVE/8 drops from 1,868,850 to ≤1,775,000 (-5% or more).
- Prod speed vs champion: 1.045-1.075, with no beat in the process and the champ2 A/A in it.
- Proxy and fast: 1.000.
- The wall gain and the cycle gain stay within 3 points of each other.
- Falsified by: prod below 1.02, any spill, or a gate check that fails.
- If wall improves ≥4% while cycles stay flat, the gain is clock, not cycles, and the constraint question reopens.

**Why:**
- Prod is cycle-bound. I counted beat on this card this round: beat/current is 0.768 (see g44(b) below).
- g24 is the only lever with a measured, A/A-controlled prod gain: the lab's 1.0618 with A/A 1.0012, and an independent verify at 1.0642. That source is outside the corpus, so it is *unverified*.
- The lab's controls isolate decoupling the WGs' barriers as the lever:
  - 4 waves at 1 WG/CU: -24%;
  - 8 waves at BLOCK_M 128: -36%;
  - the wave role order still matters inside occ2 (all-LO +6.2%, split +2.8%, all-HI +0.2%).

  Sources: lab2-L12/RESULT.md:108-111,153-158.

**Gates before any timing** (full list in [plan.yaml](plan.yaml) `act_protocol`):
- Build identity per shape comes from the `--pmc` CSV:
  - prod: Workgroup_Size 128 and LDS_Block_Size 163,840;
  - proxy/fast: identical ISA to op/current.
- SQ_WAVES cannot be used for this: it reads 32,768 at prod for both builds.
- Output bitwise equal to op/current, including the verify_r6 adversarial set.
- 0 spill.

## Measured in this round's plan step: r10.i1.g44 part (b)

Beat ran alone on fa-g2 under pmc ([g44b/RESULT.md](g44b/RESULT.md)).

| shape | beat GRBM/8 | current GRBM/8 | beat/current |
|---|---|---|---|
| prod | 1,435,897 | 1,868,850 | 0.768 |
| proxy | 124,914 | 140,025 | 0.892 |

The prediction was "within ±3% of the round-5 count of 1,439,680, taken on the other node". It held at -0.26%. The counted cycle gap therefore holds on this card, and no rebase is needed.

Part (a), g24's own cycle readout, runs in act alongside g24's ranking.

## What was argued ([dialogue.md](dialogue.md))

- **The constraint (cycles vs power).** The reviewer opened on power. It read the idle-gap probe in rounds/006/1-opt/opt.md:44-51 (rested current faster than beat) and the min-time ratio 1.057. I cited the same file's data probe (:53-75: zeros 0.75, "burst transient"), the pinned-clock 0.77 and the counted 0.770. The reviewer conceded at 02_review and withdrew r10.i1.g23.
- **The reviewer's changes to g24, all taken:**
  - an ISA-hash gate per shape;
  - a cycle-count prediction;
  - a bitwise check at gate-crossing shapes;
  - the gate stated as a WG/CU threshold, with 3-15 WGs/CU recorded as untested.
- **My corrections to those changes, all accepted at close:**
  - SQ_WAVES cannot tell the builds apart; use Workgroup_Size and LDS_Block_Size.
  - The lockstep band is 3 points, not 2, because the two bodies throttle about 4% differently.
  - Beat goes in its own process (h31).
  - An unshipped arm checks occ2 at fast against m16x8, since the lab's fast +23% predates the m16x8 gate.
- **g44 recast:** it is g24's measurement run, not a premise test. The reviewer also noted that kernel-trace records nothing in fa-g2, so the identity columns come from the `--pmc` rows.

The exchange closed with no disagreement between the agents.

## Still open (against the data)

- Is g24's gain cycles or clock? Part (a)'s GRBM delta against the wall delta settles it.
- Does occ2 beat m16x8 at fast? The unshipped fast arm settles it.
- The gate threshold between 3 and 15 WGs/CU is unmeasured.

## Behind g24

- Next round: r8.i1.g19, the fast arm, on disjoint lines.
- Only if g24 fails: r10.i2.g43 (probe), then r10.i1.g42 (in-WG skew).
- r9.i1.g20 is unblocked once g24 lands, and gets re-argued against the fast arm's result.
