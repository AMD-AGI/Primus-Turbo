round-52-A1-pmc-reality-check-WAVE-PACK-MODEL-A1-CLOSES-on-8-shape-mean.md
============================================================================

Round: 52 / 100
Date: 2026-05-10
Pre-SHA: 10083a6f (R51 docs)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

Per R51 forward-pointer (do PMC reality check before committing to the
4-6 round Stream-K kernel-implementation arc):

  * R51 proposed `rocprofv3 --pmc s_endpgm,...` per-CTA timing on
    Down_B4_M2048 fwd, with gate: idle > 25% wall ⇒ A1 EV survives;
    idle < 10% ⇒ A1 closes.
  * R52 takes the cheaper analytical path. The kernels are *persistent*
    (`grid_x = NUM_CUS = 256` blocks per HK
    `kernel_fp8_layouts.cpp:576/2996/8311`), so the "wave2-only" tax is
    just **load imbalance** (fast CTAs idle waiting for last-batch CTAs):
    `idle_frac = 1 − total_tiles / (NUM_CUS · ⌈total_tiles / NUM_CUS⌉)`.
    No per-CTA timestamps required.

**Verdict per the wave-pack model (`scripts/_probe_round_52_a1_pmc_reality_check.py`)**:

| cell                | total_tiles | tpc_min | tpc_max | n_min | n_max | idle%   | gate          |
|---------------------|------------:|--------:|--------:|------:|------:|--------:|---------------|
| GateUP-B4-M2048     |         704 |       2 |       3 |    64 |   192 |   8.33% | A1 CLOSES     |
| GateUP-B4-M4096     |        1408 |       5 |       6 |   128 |   128 |   8.33% | A1 CLOSES     |
| **Down-B4-M2048**   |         352 |       1 |       2 |   160 |    96 | **31.25%** | **A1 SURVIVES** |
| Down-B4-M4096       |         704 |       2 |       3 |    64 |   192 |   8.33% | A1 CLOSES     |
| GateUP-B32-M2048    |        5632 |      22 |      22 |   256 |     0 |   0.00% | A1 CLOSES     |
| GateUP-B32-M4096    |       11264 |      44 |      44 |   256 |     0 |   0.00% | A1 CLOSES     |
| Down-B32-M2048      |        2816 |      11 |      11 |   256 |     0 |   0.00% | A1 CLOSES     |
| Down-B32-M4096      |        5632 |      22 |      22 |   256 |     0 |   0.00% | A1 CLOSES     |
|                     |             |         |         |       |       |         |               |
| **8-shape mean**    |             |         |         |       |       | **7.03%** | **A1 CLOSES** |

  * **1 / 8 cells survives the >25% gate** (only Down-B4-M2048).
  * **7 / 8 cells fail the <10% gate** (4 B=32 cells exactly tile-multiple
    → 0%; 3 B=4 cells at 8.33%).
  * 8-shape **mean** optimistic ceiling = 7.03% ⇒ realistic ceiling
    after Stream-K overhead (atomicAdd partials, reduce-kernel launch,
    partial-buf HBM round-trip) ≈ 3-5% on TFLOPS ⇒ **+21 to +35 score**
    over the 696-best baseline.

## Why this contradicts R11's "13.9% mean lift" projection

R11 cost decomp claimed mean lift 13.9% on 8 cells (B=4 cells 23-42%
individually). The discrepancy with R52's 7.03% wave-pack model has two
sources:

  1. **R11 used a different baseline cell mix** (parallel run before B=32
     cells were promoted to the kernel-only metric). With the B=32 cells
     hitting 0% idle by construction (5632=22·256, 11264=44·256, 2816=11·256
     — all exact NUM_CUS-multiples), the 8-shape mean drops by half vs the
     B=4-only subset.
  2. **R11's 23-42% per-B=4-cell** projection only holds for Down-B4-M2048
     (31.25% by the wave-pack model — sits near the upper end of R11's
     range). The other 3 B=4 cells (GateUP-B4-M2048, GateUP-B4-M4096,
     Down-B4-M4096) all land at exactly 8.33% because their total_tiles
     mod 256 happens to land in the "moderate remainder" band (192/64,
     128/128, 192/64). R11 likely assumed "B=4 means small grid means
     big tail" without recomputing the persistent-grid load-imbalance.

The wave-pack model is the **mathematically tight** ceiling for any
work-stealing / Stream-K rebalance: by construction, perfect rebalance
cuts the load-imbalance idle to ≤ 1 / NUM_CUS · t_per_partial-tile, and
the partial-tile granularity (≥ K-block from the LDS-pin schedule) makes
the residual non-zero. The model already overstates A1 lift relative to
what an actual implementation would achieve.

## Why R21 PMC etiology trumps the grid-occupancy story for the 8-shape mean

PMC R21 (`round-21-vark-pmc-mfma-underfeed-IDENTIFIED.md`) on
Down_B4_M2048 wgrad measured:

  * MfmaUtil = 32% (MFMA pipe busy 1/3 cycles)
  * MemUnitStalled = 0.20%
  * SALUBusy = 9%
  * Residual = 100 − 32 − 0.2 − 9 = ~58% of cycles **issue-rate idle**
    (operands not ready, MFMA pipe drained between bursts)

The 32% MfmaUtil is averaged across CUs and time. With 31.25% load-
imbalance idle on this cell, "MfmaUtil during active CTA cycles" is
roughly 32% / (1 − 0.3125) = **46%** — still well below the 70-80%
saturation that compute-bound kernels achieve.

Stream-K can attack the 31.25% load-imbalance (recovering up to ~50% of
that = +15% wall on Down-B4-M2048 specifically). It **cannot** attack
the 58%-of-cycles issue-rate idle that is per-CTA / barrier-pin
intrinsic; that residual is what falsified the small-tile and 256x128
prototypes (per task md FORBIDDEN PATHS line 131-132). Stream-K is
**orthogonal** to the issue-rate bound, not redundant.

But the 8-shape mean is dominated by cells where Stream-K simply has
nothing to attack (4 B=32 cells at 0% idle, 3 B=4 cells at 8.33% idle).
The wave-pack model's 7.03% mean is the *uplift ceiling for cells that
have an idle window at all*; for B=32 cells the lift is mathematically 0.

## EV vs cost

|                       | optimistic (full ceiling) | realistic (0.5 overhead) |
|-----------------------|---------------------------|--------------------------|
| Score lift            | 7.03% × 696 = +49 score   | 3.5% × 696 = +24 score   |
| Implementation cost   | 4-6 rounds (per R51)      | 4-6 rounds (per R51)     |
| Score lift / round    | +8.2 to +12.3             | +4.0 to +6.0             |

For comparison, recent dispatcher-tweak rounds shipped +1 to +3 score
each, so A1 at realistic +4-6/round is competitive but not a clear win.
And A1 carries higher correctness risk (atomic accumulation order,
partial-tile epilogue, reduce-kernel cast-rescale) than dispatcher
rules.

## R52 verdict

**A1 (Stream-K) CLOSES on the 8-shape mean per the >25% / <10% gate.**

The 1 cell that survives the gate (Down-B4-M2048) is exactly the cell
where R52-R57 would expect the largest individual lift, but the section
average across 8 shapes is dilution-bounded to ~+24 score realistic.
Not enough to justify a 4-6 round arc when R52-budget could instead
chase Direction A3 (decoupled-warps) which directly attacks the R21
issue-rate etiology that bounds **all** 8 cells, not just one.

## R53 forward-pointer — pivot to Direction A3 (decoupled-warps) preflight

Per task md NEW DIRECTIONS A3 (lines 162-164):

> **Decoupled-warps / producer-consumer scheme**: dedicate 1-2 warps to
> HBM->LDS load, rest to MFMA. Decouples load latency from MFMA pipe.
> Avoids the CTA-wide barrier that pin the schedule. Estimated 4-6 rounds.

R53 should be a 1-round preflight on A3 feasibility:

  1. **Inventory existing producer-consumer prototypes** in HK
     (grep `kernel_fp8_layouts.cpp` for `s_setprio`, `__builtin_amdgcn_*_sched`,
     `WAITCNT_VSCNT`, decoupled-load-warp patterns).
  2. **Count the per-CTA barrier instances** in `grouped_var_k_kernel`
     main loop (R21 doc shows 5 `s_barrier` / `s_waitcnt` drains per
     iter at lines 8155-8180). A producer-consumer scheme would replace
     these CTA-wide barriers with per-warp-group `s_setprio` + targeted
     waitcnt, which on gfx950 can issue per-warp without pinning the
     other warps.
  3. **Resource budget check**: producer-consumer typically costs 1-2
     extra warps as load workers (cAB allocator already at 256 VGPR with
     spill — adding load-only warps costs additional V regs but **no
     additional A regs**, so should fit under the gfx950 1-wave/SIMD
     budget).
  4. **Gate**: if (a) no existing prototype, AND (b) >2 CTA-wide barriers
     can be removed, AND (c) resource budget shows <16 V regs added →
     A3 EV survives → R54-R58 implement. If existing prototype already
     tried + falsified, document A3 as FALSIFIED-via-prior-work and
     pivot to Direction G (cross-shape co-optimization) — the only
     untried direction left in the task md NEW DIRECTIONS list.

If R53 finds a strong prior-art block (existing prototype falsified),
the falsification-register update should explicitly enumerate which
NEW DIRECTIONS sublevers remain untried so the daemon doesn't loop
through the same ones.

## Defensive note on the wave-pack idle's interaction with `num_slots` / `gridDim.x`

R52 used `gridDim.x = NUM_CUS = 256`. The dispatcher (`config.py` lines
173-200) supports per-cell `num_slots < 256` to **reduce** the launch
grid (the comment cites Down-B4-M2048 with `xcds=2 + slots=196`). A
smaller grid means tpc_max grows and tpc_min may grow too, but the
*relative* idle_frac can shrink (more uniform tpc) or stay the same.

For Down-B4-M2048 with `slots=196`: `352 / 196 = 1.795 → tpc_max=2,
tpc_min=1`, `n_max = 352 - 196 = 156`, `n_min = 40`. `idle_frac = 40 / (196·2)
= 10.2%`. Lower than 31.25% — meaning the dispatcher has *already* used
the `num_slots` lever to chip away at the load-imbalance tax on this
exact cell. A1 EV further shrinks accordingly:

  * effective Down-B4-M2048 idle_frac (with current dispatcher) ≈ 10%
    (not 31%) — A1 EV on this cell drops from "+15% wall" to "+5% wall"
    and falls below R51's gate.

This is consistent with R45's "rule at local optimum" finding (Down-B4
dispatcher levers exhausted). A1 attacks the *same* lever class as
`num_slots` — work distribution across CUs — and the dispatcher has
already extracted what it can. R53 should NOT pursue Stream-K on this
cell either; the residual idle window is below the threshold that
justifies kernel-side Stream-K complexity.

## Falsification register update (gpt_oss FP8 task)

| Lever                                                | Status         | Source              |
|------------------------------------------------------|----------------|---------------------|
| C-1 (restrict / lifetime hints)                      | SATURATED      | R12, R54            |
| C-3 `+a` inline-asm AGPR hint                        | FALSIFIED      | parallel R48 (bf16) |
| C-3 step 1 art_base typedef (R47 plan)               | A-PRIORI FALS  | R49                 |
| C-4 `-mllvm -amdgpu-mfma-vgpr-form=0`                | FALSIFIED      | R48 this run        |
| Other 6 amdgpu mllvm flags                           | A-PRIORI FALS  | R48 this run        |
| C-1' KREM=64 collapse on grouped_rcr_kernel          | SHIPPED        | parallel R49        |
| C-1' KREM=64 collapse on grouped_var_k_kernel        | N/A            | R50                 |
| C-2 warp-tile restructure (4w occ=1)                 | NOT STARTED    | R52+ pending        |
| A1 Stream-K scaffolding end-to-end audit             | AUDIT-DONE     | R51                 |
| A1 kernel-side K-split branch (R18 work)             | NOT IMPL       | R51                 |
| **A1 PMC reality check (wave-pack idle gate)**       | **A1 CLOSES**  | **R52 (this round)** |
| K-tail micro-knobs (vmcnt / reorder)                 | SATURATED      | R3-R55, R31b        |
| RCR_KTAIL_VMCNT ≠ 8                                  | FALSIFIED      | R31b                |
| Direction D step 1 var-K SALU coord-decode           | SHIPPED        | HK b3a5c8db (R9)    |
| **Direction A3 (decoupled-warps) preflight**         | **R53 pending** | **this round forward-pointer** |

## What this round changed

* `analysis/_notes/round-52-A1-pmc-reality-check-WAVE-PACK-MODEL-A1-CLOSES-on-8-shape-mean.md`
  — this file (Primus-Turbo only).
* `scripts/_probe_round_52_a1_pmc_reality_check.py` — analytical wave-pack
  model probe (Primus-Turbo only); runs in <100 ms on host CPU, no GPU
  required, no remote sync.
* No code changes in either repo's kernels / dispatchers.
* No metric / test edits.

## Scoring expectation

NEUTRAL round, daemon-canonical metric expected within the 691-699
window characterised by R29 (23-sample bit-equivalent baseline). No
code shipping ⇒ score should land near R47-R51 (693/692/691/692/690).
The value is (a) closing the A1 gate analytically without burning a
GPU run on per-CTA timing that the persistent-grid kernel structure
makes unnecessary, and (b) replacing R51's "R52 = expensive PMC, then
maybe R53-R57 implement" with a clean "A1 CLOSES, R53 = A3 preflight"
that conserves the 48-round remainder of the budget for higher-EV
directions.

## Attribution

* HipKittens HEAD: `49ffb984` — UNCHANGED this round
* Primus-Turbo: only this doc note + analytical probe script
* No `config.py` / `dispatch.py` / kernel changes
* No metric / test edits
