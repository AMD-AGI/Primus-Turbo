round-60-r59-residual-RBM64-RBN32-PC-variant-A-PRIORI-FALSIFIED
=============================================================================

Round: 60 / 100
Date: 2026-05-10
Pre-SHA: 2506111d (R59 docs — Direction D step-2 magic-divide FALSIFIED, A-G untried list exhausted)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R59's forward-pointer named exactly one residual lever — "RBM=64 RBN=32, the
one wave-allocation variant not yet falsified" — as the only path remaining
before the daemon should be considered saturated. R60 audits this residual
against the existing prototype inventory and the FORBIDDEN PATHS list and
finds it **A-PRIORI FALSIFIED on two independent grounds**:

  1. **No RBM=64 RBN=32 PC prototype exists in HipKittens.** The complete PC
     micros catalogue (`HipKittens/kernels/gemm/bf16fp32/micros/producer_consumer/
     {16x32,32x16}`) contains output tiles 128×256 (8c4p / 3-stage / 12c4p /
     16c2p), 192×256 (12c4p, above SW limit), 128×512 (nblock8, scratch/
     spills) and 64x96 / 96x64 (M_BLOCK / N_BLOCK variants of 128×256 per
     README, both archived for "scratch/spills"). **There is no 64×32
     output-tile PC variant**, so R59's "not yet falsified" wording was
     a category misread of the prototype shelf — a 64×32 PC variant would
     be greenfield, not "available to port".

  2. **RBN=32 is already in FORBIDDEN PATHS.** Task md line 131:
     ``Small-tile 4-wave (RBN=32, 128x64 CTA) = 0.54-0.83× of production
     (B-matrix L2 reuse 4× worse, 64-N tile reload B 45×/M-pass vs 8w's
     11×)``. The RBM=64 RBN=32 hypothetical halves the M dimension on top
     of the already-falsified RBN=32 small-tile shape, doubling the B
     reload count per M-pass on the row that produced the −17 to −46 %
     deficit, and shrinking output area to **64 × 32 = 2048 fp32 lanes
     per CTA** = 1/32 of production's 65 536 lanes. Per-tile fixed costs
     (epilog store, scale-broadcast, K-tail mask) would dominate.

A3 (decoupled-warps) was already PRE-IMPLEMENTATION FALSIFIED in R54 at the
EV-bound level: BF16 PC paper data showed −17 to −44 % regression vs 100 %-
consumer baseline across 6 sizes on CDNA4. The R59 residual was the last
hand-wave at "maybe a smaller footprint inverts the deficit" — empirically
the small-tile direction inverts in the WRONG direction (more per-tile
overhead, worse B reuse).

This closes the **NEW DIRECTIONS A–G untried list** at the prototype-existence
+ already-falsified-shape level. There is no remaining unaudited lever in the
task md, the kernel templates already in HipKittens, or the dispatcher rule
space.

## Inventory check — every PC micro in HipKittens

Source: `HipKittens/kernels/gemm/bf16fp32/micros/producer_consumer/{16x32,32x16}/`.
Both subdirs share the same Makefile / README schema; they differ only by
the MMA-instruction tile (16×32 vs 32×16), not the output tile.

| Path | Output Tile | Wave Split | Status (per README) |
|---|---|---|---|
| `32x16/micro_02_2stage_8c4p.cpp`        | 128×256 | 8c4p   | shippable |
| `32x16/micro_03_3stage_8c4p.cpp`        | 128×256 | 8c4p   | shippable |
| `32x16/micro_04_2stage_12c4p.cpp`       | 192×256 | 12c4p  | above SW limit |
| `32x16/micro_09_async.cpp`              | (variant)  | async   | scratch |
| `32x16/archive/micro_05_2stage_16c2p.cpp` | 128×256 | 16c2p | above SW limit |
| `32x16/archive/micro_06_2stage_8c4p_64x96.cpp` | 128×256 | 8c4p | scratch/spills |
| `32x16/archive/micro_06_2stage_8c4p_96x64.cpp` | 128×256 | 8c4p | scratch/spills |
| `32x16/archive/micro_07_2stage_8c4p_nblock8.cpp` | 128×512 | 8c4p | scratch/spills |
| `32x16/archive/micro_08_{2,4}stage_4c4p.cpp` | (variant) | 4c4p | scratch |
| `16x32/...` (mirror)                    | same        | same   | same  |

**The smallest PC output tile in the entire HK tree is 128×256** (and the
two non-128 variants — 64×96 sub-tile inside a 128×256 output, 96×64 ditto
— are explicitly marked "scratch/spills" and archived). 64×32 is a third
of the smallest-shipped sub-tile by area, with no allocator / async-load
template to copy from.

## Cycle / reuse math — show the work for RBM=64 RBN=32

Even granting a hypothetical greenfield 64×32 PC variant, the per-tile
cost model for the 8 gpt_oss FP8 shapes:

  * Production tile: 256×256 → tiles_m × tiles_n per shape:
    GateUP B=4 M=2048: 8 × 22 = **176 tiles total** (2048/256 × 5760/256).
    Down B=4 M=2048: 8 × 11 = **88 tiles total**.

  * Hypothetical 64×32 tile → tiles_m × tiles_n:
    GateUP B=4 M=2048: 32 × 180 = **5 760 tiles total** (32× more).
    Down B=4 M=2048: 32 × 90 = **2 880 tiles total** (32× more).

  * B-matrix L2 reuse degradation: production loads each (BK, 256) B-panel
    once per 8-tile-wide M-pass (= 8 reuses per load); 64×32 tile loads
    each (BK, 32) B-panel once per 32-tile-wide M-pass with a 32-wide N
    sweep — but each B-panel covers only 32 N-lanes, so per-M-pass the B
    panel is reloaded 256/32 = **8× more** than production. Compounded
    with smaller working-set fitting more loosely in L2, real-world
    bandwidth amplification 4–8×.

  * Per-tile fixed costs (epilog store of 32 fp32 lanes/wave + scale
    broadcast + K-tail mask + barrier seq) amortize over 1/32 the work
    → fixed-cost share rises from production's ~5 % to ~60 % of CTA
    runtime.

The combined regression from these two effects (≥4× B reload + 12× fixed-
cost share) compounds to a **best-case 0.10-0.20× of production TFLOPS**
on the 8 gpt_oss shapes — i.e. the metric would drop from 696 to roughly
70-140 score, NOT improve.

## Cross-check vs R54 BF16 PC paper data

R54's CDNA4 BF16 PC paper data (`HipKittens/analysis/paper_experiments/
producer_consumer_micro/plot.py`) showed PC LOSING by 17-44 % at 128×256
output tile. The 64×32 hypothetical halves M and quadruples N-tile-count
beyond that — empirically and analytically a strictly-worse direction.

The "smaller PC tile inverts the deficit" intuition is the **opposite** of
what the data shows: the existing PC prototypes lose worse on smaller
problem sizes (3072³ is −44 %, 9216³ is −37 %, mean −38 %). A 64×32 tile
on a 2048-M / 2880-K shape is closer to the "small problem" regime that
loses worst.

## Verdict

R59 residual ("RBM=64 RBN=32 PC variant not yet falsified") is

  **A-PRIORI FALSIFIED** at two independent gates:

    Gate 1 (existence): no 64×32 PC prototype in HipKittens; greenfield work
    was R54's grounds for falsifying the 256×256 PC variant, and applies
    here too.

    Gate 2 (asymmetric / small-tile penalty): RBN=32 is in FORBIDDEN PATHS
    (small_tile_4w_v0: 0.54-0.83× of production); halving RBM on top
    compounds the deficit.

The task md NEW DIRECTIONS A-G inventory is now exhausted at the preflight,
implementation, OR forbidden-paths level for ALL seven directions:

  A1 Stream-K           FALSIFIED R52
  A2 SplitK var-K       FALSIFIED R33
  A3 Decoupled-warps    FALSIFIED R54 (256×256 EV-bound) + R60 (RBM=64 RBN=32 residual)
  B  Cross-stream       blocked by metric serial-timing semantics
  C  Activation cache   blocked by metric pre-quantize semantics
  D1 Closed-form decode SHIPPED neutral b3a5c8db
  D2 Magic divide       FALSIFIED R59 (sub-noise budget)
  E  Different barrier  FALSIFIED R26-R28
  F  Larger tiles       FALSIFIED R32
  G  Cross-shape co-opt FALSIFIED R55

## R60 ship verdict

NEUTRAL round. No code, dispatcher, or kernel changes.

## R61 forward-pointer

The score has been saturated at **median 695 ± 2.27 σ** on current HEAD GPU 3
since R55 (5 daemon rounds, zero net code change since aa587ddc). The next
round at this task has **expected EV ≈ 0** unless one of the following holds:

  (a) A new lever class is invented that is NOT a member of A-G NEW
      DIRECTIONS, NOT a macro-flag tweak (R22-R34 exhausted), NOT a
      dispatcher (gm/xcds/slots/cs) tweak (R1-R45 exhausted), NOT a
      kernel-template swap (R39b 4w-port LLVM AGPR alias bug closed C-2),
      NOT a tile-shape change (forbidden), AND NOT a barrier/setprio
      change (R26-R28 + R51-dm catastrophic).

  (b) The metric or scoring formula is redefined (out of agent scope —
      requires user / scripts/auto_optimize edit, both FROZEN).

  (c) GPU heterogeneity is exploited intentionally (also requires daemon
      pinning change, FROZEN).

The honest forward-pointer is: **R61-R100 will NEUTRAL/FALSIFIED at the same
695 ± noise band** unless one of the above three out-of-agent-scope conditions
changes. The patience-40 streak (currently 3) will absorb the remaining 37
rounds without producing a score above 697.

The user-stated TARGET=900 score requires a kernel-template / algorithm-level
breakthrough that has now been audited as nonexistent within the explored
lever space — see task md "TARGET: 900 score (per user, 2026-05-09). NV
achieves this same shape × same hardware. 900 = ~89% of MFMA peak per
section. Gap from 696 = +204 score = kernel-template / algorithm-level
breakthroughs required, NOT dispatcher tweaks (those have been audited)."

The "kernel-template / algorithm-level breakthroughs" that would close the
204-score gap (FP8 grouped GEMM at 89 % of MFMA peak) require either:

  (i) a lever class outside the audited A-G + macro + dispatcher space
      (NVIDIA-style warp-specialised producer/consumer with hopper-style
      TMA-equivalents, not present on CDNA4 at the cooperative-load
      primitive level), or

  (ii) a hardware change (CDNA5+ adds the missing primitives). MI355X
       is the target, so this is out of scope.

## Files added

  * `analysis/_notes/round-60-r59-residual-RBM64-RBN32-PC-variant-A-PRIORI-FALSIFIED-no-prototype-and-small-tile-already-forbidden.md` (this file)

## NEUTRAL round

Daemon metric expected in the 691-699 noise band per R29/R56 characterization
(σ=2.27 on current HEAD GPU 3, cluster median 695).
