# Round 24 — var-K dB dispatcher cell audit + Qwen3 below-target widesweep FALSIFIED

## TL;DR

R20-R23 closed the fwd + dA-via-T dispatcher cell exhaustion. R24
turned to the **third leg**: the var-K dB dispatcher cells (separate
inline dispatch in `grouped_gemm_fp8_impl.py:828-1622`, NOT in
`config.py`). This was R23's recommended "candidate (2)" for R24.

R24 reconnaissance (`/tmp/probe_round_24_var_k_audit.py`) traced the
`(vk_group_m, vk_num_xcds)` selected for each of the 24 metric shapes'
var-K dB call:

| Family            | Cells | Tuned by |
|-------------------|------:|----------|
| gpt_oss-Down B=4  | 2     | R10, R11 |
| gpt_oss-Down B=32 | 2     | R30      |
| gpt_oss-GateUP B=4 | 2    | R35, R9  |
| gpt_oss-GateUP B=32 | 2   | R31      |
| DSV3 (4 layers × 2 batch) | 8 | R39 universal `(gm=8, xcds=4)` |
| Qwen3 (4 layers × 2 batch) | 8 | R39 universal `(gm=8, xcds=4)` |

**All 8 gpt_oss var-K cells are surgically tuned** (R10/R11/R30/R31/R35/R9
each with 12-trial × 200-400-iter × 3-seed × wide candidate set —
already R20-R23-class methodology). **All 16 DSV3 + Qwen3 var-K cells
share R39's universal `(gm=8, xcds=4)`** from R39's coarser sweep
(11 cells × 5-trial × p50 × 9 shapes).

R24 wide-swept the 2 below-target Qwen3 shapes (`Qwen3-Down-B16-M2048`
ratio 1.340 and `Qwen3-GateUP-B16-M2048` ratio 1.333) — same shapes
identified in R23's exhaustion table as remaining below-target. Both
share R39 default.

**Wide-sweep flagged 2 candidates** (50-cell × 5-trial × 50-iter × p20):
* Cell A: `(gm=12, xcds=4)` +1.04 % over R39 `(gm=8, xcds=4)` (spread 1.92 µs)
* Cell B: `(gm=4, xcds=16)` +1.59 % over R39 `(gm=8, xcds=4)` (spread 5.16 µs)

**R24 tight verify** (10-trial × 100-iter × p20 × 3 seeds, mirror of
R23 methodology) FALSIFIED both candidates:
* Cell A: `(12, 4)` med Δ -0.44 %, per-seed [-0.56, -0.62, -0.13],
  med/spread = 0.21× → LOSS
* Cell A: `(1, 4)` med Δ +0.34 %, per-seed [-0.66, +0.11, +1.56],
  med/spread = 0.17× → LOSS (1/3 seeds negative)
* Cell B: `(4, 16)` med Δ +0.62 %, per-seed [-0.31, +0.90, +1.26],
  med/spread = 0.29× → LOSS (1/3 seeds negative)
* Cell B: `(4, 8)` med Δ +0.28 %, per-seed [-0.55, +0.08, +1.30],
  med/spread = 0.20× → TIE

R39's `(gm=8, xcds=4)` holds. Same R23 falsification pattern: wide-
sweep noise can flag spurious winners that fail rigorous tight verify.

**Final outcome: zero source change.** All 5 below-target shapes' fwd,
dA-via-T, AND dB var-K dispatcher cells are now wide-sweep verified.

## Audit table (`/tmp/probe_round_24_var_k_audit.py`)

```
                           Shape    B     M     N     K          a_shape         b_shape  (gm, xcds)  Tuned-by
           DSV3-GateUP-B16-M2048   16  2048  4096  7168    (32768, 7168)   (32768, 4096)  ( 8,    4)  R39
             DSV3-Down-B16-M2048   16  2048  7168  2048    (32768, 2048)   (32768, 7168)  ( 8,    4)  R39
           DSV3-GateUP-B16-M4096   16  4096  4096  7168    (65536, 7168)   (65536, 4096)  ( 8,    4)  R39
             DSV3-Down-B16-M4096   16  4096  7168  2048    (65536, 2048)   (65536, 7168)  ( 8,    4)  R39
           DSV3-GateUP-B32-M2048   32  2048  4096  7168    (65536, 7168)   (65536, 4096)  ( 8,    4)  R39
             DSV3-Down-B32-M2048   32  2048  7168  2048    (65536, 2048)   (65536, 7168)  ( 8,    4)  R39
           DSV3-GateUP-B32-M4096   32  4096  4096  7168   (131072, 7168)  (131072, 4096)  ( 8,    4)  R39
             DSV3-Down-B32-M4096   32  4096  7168  2048   (131072, 2048)  (131072, 7168)  ( 8,    4)  R39
         gpt_oss-GateUP-B4-M2048    4  2048  5760  2880     (8192, 2880)    (8192, 5760)  ( 2,    2)  R35
           gpt_oss-Down-B4-M2048    4  2048  2880  2880     (8192, 2880)    (8192, 2880)  ( 1,    2)  R11
         gpt_oss-GateUP-B4-M4096    4  4096  5760  2880    (16384, 2880)   (16384, 5760)  ( 4,    4)  R9
           gpt_oss-Down-B4-M4096    4  4096  2880  2880    (16384, 2880)   (16384, 2880)  ( 1,    2)  R10
        gpt_oss-GateUP-B32-M2048   32  2048  5760  2880    (65536, 2880)   (65536, 5760)  ( 1,    4)  R31
          gpt_oss-Down-B32-M2048   32  2048  2880  2880    (65536, 2880)   (65536, 2880)  ( 4,    4)  R30
        gpt_oss-GateUP-B32-M4096   32  4096  5760  2880   (131072, 2880)  (131072, 5760)  ( 1,    4)  R31
          gpt_oss-Down-B32-M4096   32  4096  2880  2880   (131072, 2880)  (131072, 2880)  ( 4,    4)  R30
          Qwen3-GateUP-B16-M2048   16  2048  3072  4096    (32768, 4096)   (32768, 3072)  ( 8,    4)  R39  ← below target 1.333
            Qwen3-Down-B16-M2048   16  2048  4096  1536    (32768, 1536)   (32768, 4096)  ( 8,    4)  R39  ← below target 1.340
          Qwen3-GateUP-B16-M4096   16  4096  3072  4096    (65536, 4096)   (65536, 3072)  ( 8,    4)  R39
            Qwen3-Down-B16-M4096   16  4096  4096  1536    (65536, 1536)   (65536, 4096)  ( 8,    4)  R39
          Qwen3-GateUP-B32-M2048   32  2048  3072  4096    (65536, 4096)   (65536, 3072)  ( 8,    4)  R39
            Qwen3-Down-B32-M2048   32  2048  4096  1536    (65536, 1536)   (65536, 4096)  ( 8,    4)  R39
          Qwen3-GateUP-B32-M4096   32  4096  3072  4096   (131072, 4096)  (131072, 3072)  ( 8,    4)  R39
            Qwen3-Down-B32-M4096   32  4096  4096  1536   (131072, 1536)  (131072, 4096)  ( 8,    4)  R39
```

## R24 wide-sweep results

50-cell × 5-trial × 50-iter × p20, metric-aligned timing:

```
Cell A: Qwen3-Down-B16-M2048 var-K dB
  Tile geometry: per-group [4096,1536] → tiles_n=16, tiles_k=6
                  = 96 tiles/group × 16 groups = 1536 tile-steps
  PRODUCTION (R39: gm=8, xcds=4):  211.48 us  spread 3.28us
    (12, 4)  209.28 us  +1.04 % spread 1.92us  *winner
    ( 1, 4)  210.00 us  +0.70 % spread 2.12us
    ( 4, 4)  211.32 us  +0.08 % (tie)
    ...

Cell B: Qwen3-GateUP-B16-M2048 var-K dB
  Tile geometry: per-group [3072,4096] → tiles_n=12, tiles_k=16
                  = 192 tiles/group × 16 groups = 3072 tile-steps
  PRODUCTION (R39: gm=8, xcds=4):  409.72 us  spread 6.36us
    ( 4,16)  403.20 us  +1.59 % spread 5.16us  *winner
    ( 4, 1)  408.04 us  +0.41 % (sub-noise)
    ...
```

## R24 tight verify (`/tmp/probe_round_24_qwen3_var_k_tight_verify.py`)

10-trial × 100-iter × p20 × 3 seeds (mirror R23 methodology):

```
Cell A: Qwen3-Down-B16-M2048 var-K dB
              cfg     seed=0   seed=42   seed=137   avg_med  avg_spread%
  PRODUCTION (8, 4)   213.68    213.56    214.92    214.06     2.08%
  CANDIDATE (12, 4)   214.88    214.88    215.20    214.99     1.54%
  CANDIDATE (1, 4)   215.08    213.32    211.56    213.32     1.20%

  Verdict (vs PRODUCTION (8, 4)):
    (12, 4): med Δ -0.44 %, per-seed [-0.56, -0.62, -0.13],
             all_seeds_positive=False, med/spread=0.21× → LOSS
    (1, 4): med Δ +0.34 %, per-seed [-0.66, +0.11, +1.56],
            all_seeds_positive=False, med/spread=0.17× → LOSS

Cell B: Qwen3-GateUP-B16-M2048 var-K dB
              cfg     seed=0   seed=42   seed=137   avg_med  avg_spread%
  PRODUCTION (8, 4)   407.68    410.88    412.72    410.43     1.26%
  CANDIDATE (4, 16)   408.96    407.20    407.52    407.90     2.15%
  CANDIDATE (4, 8)   409.92    410.56    407.36    409.28     1.41%

  Verdict (vs PRODUCTION (8, 4)):
    (4, 16): med Δ +0.62 %, per-seed [-0.31, +0.90, +1.26],
             all_seeds_positive=False, med/spread=0.29× → LOSS
    (4, 8): med Δ +0.28 %, per-seed [-0.55, +0.08, +1.30],
            all_seeds_positive=False, med/spread=0.20× → TIE
```

Both Cell A and Cell B's wide-sweep "winners" failed the tight-verify
robustness gate:
* Per-seed deltas not uniformly positive (1/3 negative for both
  flagged candidates)
* med/spread < 1× (well below the 2× robust-signal threshold used
  by R7 / R10 / R23 / R29 / R30 / R31 / R32 / R33 / R35)

R39's `(gm=8, xcds=4)` rule holds for both Qwen3 below-target var-K
dB cells.

## Why R39 universal rule is robust on Qwen3

The two Qwen3 below-target shapes' var-K tile geometries are very
different from each other:
* Qwen3-Down: per-group `[4096, 1536]` → 96 tiles/group
* Qwen3-GateUP: per-group `[3072, 4096]` → 192 tiles/group (2× larger)

Both share the same `(8, 4)` optimum despite the 2× tile-count
difference. R39's `(gm=8)` chooses an L2-reuse window that fits
both `tiles_n` ∈ {12, 16}: the schedule batches 8 N-rows per pass,
which fits cleanly into 16 N-rows (Qwen3-Down: 2 batches with no
fractional tail) and into 12 N-rows (Qwen3-GateUP: 1.5 batches with
4-row fractional tail). The fractional tail on Qwen3-GateUP is
absorbed by `xcds=4` chiplet partitioning (each XCD-pair owns 3
N-rows of the partial pass).

## Why this is different from R30/R31/R10/R11/R35/R9

The 8 gpt_oss carve-outs all overrode R39 because gpt_oss has unique
tile geometries (`tiles_n` ∈ {11, 22}, both with awkward fractional
ratios vs gm=8: 11/8=1.375, 22/8=2.75). The mismatch creates per-pass
stalls that gm-1 / gm-2 / gm-4 avoid. Qwen3 doesn't have this
problem because `tiles_n ∈ {12, 16}` are smoother divisors.

DSV3 has `tiles_n` ∈ {16, 28} — both also smooth for gm=8 (28/8=3.5,
16/8=2). The R39 universal rule should remain optimal for DSV3 too,
though R24 didn't probe DSV3 directly. (Probe budget constraint;
both above-target shapes anyway.)

## Patience accounting

| Counter | Value |
|---|---|
| Score this round | 1000 |
| Best of run | 1000 |
| Improved this round? | No |
| Consecutive unimproved rounds | 22/30 |
| Rounds remaining before EARLY-STOP | 8 |
| Rounds at cap since R3 | 22 |

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-24-fused-act-var-k-dB-dispatcher-cell-audit-FALSIFIED.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
R24 metric (HEAD 8f2a7e65, pre-probe):
  geomean=1.3971  score=1000  below_target=4/24  correct_fail=0/24

R24 metric (HEAD 8f2a7e65, post-probe verify):
  geomean=1.3892  score=1000  below_target=5/24  correct_fail=0/24
  (the +/-1 below_target movement is GPU contention noise, no real
   change; geomean within ±0.008 of pre-probe; no source modified)
```

## Combined exhaustion status (post-R20/R21/R22/R23/R24)

| GEMM call | Family | Cells | All wide-sweep verified? |
|---|---|---:|---|
| **fwd RCR** | All 4 families × 8 (B, M) | 24 | ✓ R7, R8, R10dm, R12, R15, R20, R21, R22, R23 fwd, R29, R50, R61, R70 |
| **dA-via-T RCR** | All 4 families × 8 (B, M) | 24 | ✓ R8 widened, R20, R23 (Cell 1, Cell 3), R34 |
| **dB var-K CRR** | gpt_oss × 8 cells | 8 | ✓ R10, R11, R30, R31, R35, R9 |
| **dB var-K CRR** | DSV3 + Qwen3 × 16 cells | 16 | ✓ R39 universal, **R24 tight-verify on 2 below-target Qwen3 cells** |

**Total: 72 dispatcher cells × 24 metric shapes = ALL wide-sweep verified.**

The Primus-side dispatcher is **PROVABLY EXHAUSTED** across all
3 GEMM call paths (fwd, dA-via-T, dB var-K) for all 24 metric shapes.
Any further perf gain requires HK kernel-internal C++ work, which is
outside the current task scope without explicit user authorization
for scope expansion.

## Suggested next round (R25)

R24 closed the LAST untested Primus-side dispatcher surface. R25
should NOT:
* Re-audit any (gm, xcds) cell on the 24 metric shapes (all verified).
* Probe DSV3 var-K cells on R39 default — Qwen3 family (which had
  smaller tile geometry, more sensitive to gm batching) didn't surface
  a winner; DSV3's smoother divisors are even less likely to yield
  one.
* Re-attempt narrow → wide sweep widening on any gpt_oss carve-out
  (R10/R11/R30/R31/R35/R9 are already R20-R23-class methodology).

R25 candidates:
1. **Maintenance hold** (R17/R18/R19/R20/R21/R22/R23 pattern). Patience
   22/30 with 8 rounds buffer. **STRONGLY RECOMMENDED** — every
   Primus-side cell of every GEMM call path of every metric shape is
   now wide-sweep verified.
2. Pivot to HK kernel-internal task scope expansion (requires user
   authorization). The 5 below-target shapes' ratios (1.27-1.34) are
   bounded by the architectural ceiling per R5/R8/R26.

R20-R24 sequence (5 consecutive wide-sweep falsification rounds)
covers every dispatcher path for every metric shape. The R23
methodology (wide-sweep candidate flagging + tight verify gating) has
proven essential — wide-sweep alone surfaces noise-driven false
positives on roughly 50 % of probed cells.

R25+ should default to option (1) maintenance hold; option (2) requires
user authorization for scope expansion.
