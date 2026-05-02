round-47-fp8-grouped-AGPR-vs-VGPR-discovery-and-attribute-falsifications.md
=========================================================================

Round: 47 / 100
Date: 2026-05-02
SHA: 60b976a4 (pre) → TBD (post)
Task: grouped FP8 tensorwise (fwd + var-K bwd + dA bwd) on the 24-shape suite

## TL;DR — major discovery, 2 falsifications, R48+ path

Built kernel with `-Rpass-analysis=kernel-resource-usage` (already enabled
in Makefile). Compared resource reports for **rcr_4w** (dense, never
optimized for grouped) vs **grouped_rcr_kernel** (target):

```
Kernel                            VGPRs  AGPRs  Scratch  Spill  Occ
rcr_4w::kernel                     198   256      0       0      1
grouped_rcr_kernel<F,F>            256     0    220      54      2
grouped_rcr_kernel<T,F>            256     0    156      38      2
grouped_rcr_kernel<F,T>            256     0    140      34      2
grouped_rcr_kernel<T,T> (gpt_oss)  256     0    152      37      2
grouped_rrr_kernel                 256     0    264      65      2
grouped_var_k_kernel_fp8           256     0    152      37      2
```

**rcr_4w uses 256 AGPR for accumulators → 0 spill.**
**Every grouped_*_kernel uses 0 AGPR → 34-65 VGPR spills, 140-264 scratch
bytes/lane** (= VMEM round-trips for spill load/store, ~50-100 cy each).

This is a NEW INSIGHT not previously documented. R8-dm-fp8-rcr4w-port-plan
spotted the rcr_4w 256 AGPR (in passing) but never investigated WHY grouped
has 0 AGPR or pursued AGPR migration. The R8-dm plan for "AGPR accumulator
migration on grouped_rcr_kernel" was deferred and never implemented.

## 2 attribute experiments — both BIT-IDENTICAL → falsified

Tested two LLVM attribute variants on grouped_rcr_kernel:

### Experiment 1: `__attribute__((amdgpu_waves_per_eu(2,2)))`

Theory: forcing min=max=2 waves/EU might trigger AGPR allocation by
constraining VGPR budget more strictly than `__launch_bounds__(NT, 1)`.

```
Before:  VGPR 256, AGPR 0, Scratch 220/156/140/152, Spill 54/38/34/37
After:   VGPR 256, AGPR 0, Scratch 220/156/140/152, Spill 54/38/34/37
```

**Bit-identical** — same as R5-dm/R7-dm finding for `__launch_bounds__(_, 2)`.
LLVM heuristic decides VGPR-only allocation regardless of waves/EU hint.
**Reverted; falsified.**

### Experiment 2: Remove `__launch_bounds__(_, 1)` entirely

Theory: explicit `(_, 1)` tells LLVM "you have a whole CU per block" →
might disincentivize AGPR allocation. Removing might let LLVM pick
AGPR if it's beneficial.

```
Before (with launch_bounds):  Spill 54/38/34/37, AGPR 0
After (without):              Spill 54/38/34/37, AGPR 0
```

**Bit-identical**. Compiler arrives at same allocation decision regardless
of MIN_BLOCKS_PER_CU hint. Confirms R5-dm/R7-dm general finding: occupancy
hint is NOT the lever for AGPR allocation. **Reverted; falsified.**

## Why grouped_rcr_kernel doesn't get AGPR (root-cause hypothesis)

Comparing accumulator footprint:

| Kernel        | RC type                                  | VGPRs/lane (acc only) |
|---------------|------------------------------------------|-----------------------|
| rcr_4w        | rt_fl<HB/2, HB/2, col_l, rt_16x16_s>     | 4 acc × 64 = **256** (whole budget) |
| grouped       | rt_fl<RBM=64, RBN=32, col_l, rt_16x16_s> | 4 acc × 32 = **128** |

**rcr_4w's accumulator alone consumes the full 256 VGPR/lane budget.**
With ANY other live state (a/b regs, offsets, SRDs, etc.), LLVM is forced
to spill SOMETHING — and choosing AGPR for accumulators is the obvious
choice (free 256 AGPR available, accumulators are MFMA-only consumed in
the main loop, perfect AGPR candidates). Hence: 256 AGPR, 0 spill.

**grouped_rcr_kernel's 128 VGPR accumulator footprint fits comfortably in
VGPR.** LLVM heuristic: "no spill needed → keep in VGPR, AGPR copy
overhead not worth it". So 0 AGPR. Then a/b/offsets/SRDs/loop control
add up to ~250+ VGPR, and the K-tail block adds ~25 more → exceeds 256
budget → 34-54 VGPR spills to scratch.

This is a **structural property** of how the kernel is parameterised
(WARPS_M=2, WARPS_N=4, RBN=32) — NOT something a hint can fix.

## How to actually trigger AGPR allocation (R48+ candidates)

### Option C-2 (data-flow restructure, 2-3 rounds)

**Match rcr_4w's per-warp accumulator size: 256 VGPR.** Two paths:

  - **C-2a**: WARPS_N: 4 → 2, RBN: 32 → 64. Each warp covers 128×128
    cells (same as rcr_4w). RC = rt_fl<64, 64> = 4 acc × 64 VGPR = 256.
    LLVM heuristic should switch to AGPR (matches rcr_4w precedent).
    But: total warps drops from 8 → 4 → BLOCK_SIZE_N stays 256 only if
    each warp covers 128 N → fewer warps means fewer HBM loads in
    parallel → potentially slower memory ILP. RBN=64 also requires
    the load_b helper to handle taller B tiles.

  - **C-2b**: keep WARPS_N=4, but raise RBM: 64 → 128 (per-warp covers
    128 M instead of 64). Each warp does 4 acc × (128×32 / 16x16 = 16
    base tiles × 4 fp32) = **256 VGPR per warp** for accumulator. But
    BLOCK_SIZE_M would double to 512 unless WARPS_M halves to 1 — and
    WARPS_M=1 disables the M-slab swap pattern.

**Risk**: this is a structural change to the kernel's warp tile mapping.
Requires re-tuning load patterns, prefetch depth, possibly K-tail/N-tail
helpers. Estimate 2-3 rounds.

### Option C-3 (inline-asm AGPR migration, 1-2 rounds)

**Direct LLVM IR / inline asm to force AGPR via art_base wrapper.**

Existing infra in `include/types/register/art_base.cuh` provides the
`art_base<float, layout, shape, register_range>` type which uses register
RANGES instead of an array — designed for assembly-mode register
management. Could declare:

```cpp
using ART_RC = art_base<float, col_l, rt_16x16_s, ducks::art::range<0,4>>;
// 4 of these for cA/cB/cC/cD
```

Then call mma_AB_base directly with `acc.tiles[h][w].registers` and
manually emit `v_accvgpr_read_b32` / `v_accvgpr_write_b32` at the
mul/store epilog boundary.

R8-dm planned this: "rcr_mma(acc, a, b) → mma_ABt_base(acc.tiles[..],
a.tiles[..], b.tiles[..], acc.tiles[..]); insert v_accvgpr_read before
mul(cA, cA, scale) and store(g.c, cA, ...)". Never implemented.

**Risk**: art_base / mma_AB_base path is less battle-tested than the
rt_fl / rcr_mma path. Possible numerical or IR-validation issues. But
the scope is narrow (4 accumulator decl sites, 4 mfma call sites,
1 epilog block). 1-2 rounds.

### Option C-4 (compile-flag, 0 rounds — try first)

Test `-mllvm -amdgpu-mfma-vgpr-form=0` (already default, but explicit
might tickle different heuristic path). Add to Makefile, build, check
resource report. If AGPR appears → run probe → potentially commit.
If bit-identical → falsify and move to C-2 / C-3.

This is 1 build + 1 probe = 1 round, lowest risk. **R48 should try this first.**

## Why this is HIGHER LEVERAGE than past micro-knob attempts

Per `_fp8_grouped_nogate_probe.py` baseline this round:

```
GEOMEAN(24 cases) = 1.1603 (Run 5; range across runs 4-5 = 1.149-1.203)
```

Need +2-5pp on geomean to hit 1.20 target. The 5 worst (all gpt_oss
K=2880) sit at 1.05-1.08; their `grouped_rcr_kernel<T,T>` spec has
**37 VGPR spills + 152 scratch bytes/lane**. Each tile makes ~22 K-iters,
so the cumulative spill traffic across a tile is ~37 spills × ~22 reloads
× 64 lanes × 50 cy/spill = ~2.6M cycles = ~1.3 ms across the GPU,
distributed over many tiles. On gpt_oss-Down-B32-M4096 with kernel wall
~1 ms, even partial spill elimination is a percent-scale win.

If C-2 / C-3 can reduce grouped_rcr_kernel<T,T> spill from 37 → 0
(matching rcr_4w's 0 spill on dense), expected gain ≈ 5-10 pp on the
gpt_oss cluster, lifting their 1.06-1.08 ratios to 1.11-1.16. **That
single change alone could close the geomean gap to target.**

## What this round actually changed

- **No kernel changes** (both experiments reverted).
- **No Primus-Turbo Python changes**.
- **Only this round note** ⇒ Primus-Turbo doc-only commit.

HipKittens HEAD unchanged at `92407889`. Working tree clean after revert.

## Authoritative metric is STILL GPU-blocked (20th consecutive round)

Same `card3,309220868096,20924743680` zombie KFD VRAM (19.49 GB) — no
KFD process listed but VRAM held. 20 rounds counting. User intervention
still outstanding (`sudo rmmod amdkfd && sudo modprobe amdkfd`).

Round 47 baseline probe (single-trial):

```
GEOMEAN(24 cases) = 1.1603 (run 5)
Worst 5 (consistent across 5 runs):
  1.048-1.084  gpt_oss-GateUP-B32-M2048 / B4-M4096 / B32-M4096
  1.061-1.077  gpt_oss-Down-B32-M2048 / B32-M4096
```

Score extrapolation (assuming bf16 plateau 1.187): **~977**, similar to
last round's 977 (run 5 of probe series). No regression, no progress —
matches the plateau the kernel has been at for 5+ rounds.

## Next round (R48) action ladder — concrete

1. **First experiment (cheapest)**: Add `-mllvm -amdgpu-mfma-vgpr-form=0`
   to Makefile HIPFLAGS. Build. Compare resource report. If AGPR > 0 on
   any grouped_*_kernel spec → run probe. If bit-identical → revert and
   move to step 2. **Time: 10 minutes.**

2. **Second experiment**: Try `-mllvm -amdgpu-aggressively-redistribute`
   or similar IR-level flag to encourage AGPR. Several flags listed in
   `/opt/rocm/llvm/bin/llc --help-hidden | grep amdgpu | grep -i agpr`
   — none directly "force AGPR" but some may shift the heuristic.
   **Time: 30 minutes.**

3. **Option C-3 implementation step 1** (if 1-2 are bit-identical):
   declare `art_base`-wrapped accumulator type in the kernel. Just the
   typedef; no rcr_mma replacement yet. Build, verify it compiles,
   commit infra-only. **Time: 1 round.**

4. **Option C-3 step 2**: replace rcr_mma calls with mma_AB_base calls
   targeting art_base accumulators. This is the actual migration.
   Build, verify resource report shows AGPR > 0, run probe.
   **Time: 1 round.**

5. **Option C-3 step 3**: handle epilog (mul + store) with explicit
   `v_accvgpr_read` to bring values back to VGPR for store instructions.
   Verify correctness via probe. **Time: 1 round.**

Total expected: 1 quick falsification round (R48) + 3 implementation
rounds (R49-R51) = 4 rounds to reach a metric-validated AGPR migration
or formal falsification of the entire C-3 path. Patience budget allows
this (currently 19/30 streak, 11 rounds remaining before EARLY-STOP).

## Falsification register (updated)

| Lever / approach                              | Status         | Round   |
|-----------------------------------------------|----------------|---------|
| Lever A (async g→LDS) — base shipped          | SHIPPED        | R54-dm  |
| Lever B (dual LDS) — base shipped             | SHIPPED        | early   |
| Lever C-1 (restrict / lifetime hints)         | SATURATED      | R12,R54 |
| **Lever C-4** (mfma-vgpr-form mllvm flag)     | **NEXT R48**   | —       |
| **Lever C-3** (art_base AGPR migration)       | **R49+ if C-4 fail** | — |
| **Lever C-2** (warp-tile restructure to 4w)   | **R50+ fallback** | —    |
| Lever D K-tail-only port                      | FALSIFIED      | R62-dm  |
| Lever D full main-loop port (R-B 5+)          | NOT STARTED    | —       |
| Lever E (ASM main-loop)                       | NOT STARTED    | —       |
| Lever F (Qwen3 K=1536 short-K variant)        | FALSIFIED      | R35-grp |
| `amdgpu_waves_per_eu(2,2)` attribute          | **FALSIFIED**  | **R47** |
| Drop `__launch_bounds__(_, 1)` entirely       | **FALSIFIED**  | **R47** |
| sched_barrier / LICM / anti-CSE class         | FALSIFIED      | R31-32  |
| K-tail micro-knobs (vmcnt / reorder)          | SATURATED      | R3-R55  |

## Probe data files

```
/tmp/probe_round_47_baseline.log     (run 5, geomean 1.1603)
/tmp/build_round_47_baseline.log     (current HEAD resource report)
/tmp/build_round_47_wpe22.log        (with amdgpu_waves_per_eu(2,2): identical)
/tmp/build_round_47_no_lb.log        (without launch_bounds: identical)
```

## Attribution

- HipKittens HEAD: `92407889` — UNCHANGED this round (both experiments reverted)
- Primus-Turbo: only this doc note
- No `config.py` / `dispatch.py` / kernel changes
- No metric / test edits
