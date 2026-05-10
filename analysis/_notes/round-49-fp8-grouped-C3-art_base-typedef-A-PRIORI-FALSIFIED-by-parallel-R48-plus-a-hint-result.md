round-49-fp8-grouped-C3-art_base-typedef-A-PRIORI-FALSIFIED-by-parallel-R48-plus-a-hint-result.md
==================================================================================================

Round: 49 / 100
Date: 2026-05-10
Pre-SHA: d3323863 (R48 docs commit)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R48 (this run) handed off "Lever C-3 step 1 — declare `art_base`-wrapped
accumulator typedef on `grouped_rcr_kernel<T,T>`, verify resource remark
shows AGPR > 0" as the next experiment. **A-PRIORI FALSIFIED this round
by cross-referencing the parallel R48 (bf16 24-shape series) result on
the same source file** (`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`).

That parallel R48 implemented the cheaper sibling of C-3 — a `+a` inline-asm
constraint hint inside `rcr_mma` / `rrr_mma` / `crr_mma` on the same
kernel — and got AGPR allocation working AT THE COST of a 5.3× spill
regression on the `grouped_rcr_kernel<T,T>` (gpt_oss target) instantiation:

```
                              before (R47/R48 baseline)   after +a AGPR hint
Spec <F,F>                    VGPR 256 / AGPR  0 / 54sp  → V 128 / A 128 / 129sp
Spec <T,F>                    VGPR 256 / AGPR  0 / 38sp  → V 128 / A 128 / 129sp
Spec <F,T>                    VGPR 256 / AGPR  0 / 34sp  → V 128 / A 128 / 188sp
Spec <T,T> (gpt_oss target)   VGPR 256 / AGPR  0 / 37sp  → V 128 / A 128 / 197sp  ← 5.3× worse
grouped_rrr_kernel            VGPR 256 / AGPR  0 / 65sp  → V 128 / A 128 / 129sp
grouped_var_k_kernel_fp8      VGPR 256 / AGPR  0 / 37sp  → V 128 / A 128 / 143sp
```

Source: `analysis/_notes/round-48-fp8-grouped-AGPR-hint-FALSIFIED-and-fallthrough-found.md`

The art_base typedef path (R47/R48-recommended C-3 step 1) **inherits the
same structural blocker** as the +a hint and is therefore A-PRIORI
falsified. Reasoning below.

## Why art_base typedef inherits the same regression

The parallel-R48 root-cause analysis identified the structural blocker:

> on gfx950 the per-lane register file is shared between V and A — using
> 128 AGPR forces VGPR limit down to 128 (occupancy 2 wave/SIMD
> constraint). The non-acc working set (a/b/offsets/SRDs/loop control
> ≈ 120 VGPR) no longer fits in 128 VGPR → cascading spill of MORE
> values to scratch (197 vs 37 = +160).

This is the **CDNA shared occupancy-budget model**: the SUM of VGPR + AGPR
per lane determines wave/SIMD count. For the production target of 2
wave/SIMD (the existing `__launch_bounds__(NT, 1)` translates to 256
combined reg slots per lane). Currently the compiler picks 256 V / 0 A
because the 128-VGPR-equivalent accumulator + ~120-VGPR working set fits
cleanly with 8 spills (= 37 spill ÷ ~4 spill-per-VGPR rounding).

Forcing AGPR for the accumulator — whether by `+a` inline-asm hint
(parallel R48) OR by `art_base` typedef declaration (R47/R48 plan) — both
move 128 V worth of state into the A-half of the shared budget. The
allocator must then re-pack the working set into the remaining ~128 V.
Two outcomes possible:

  1. **Keep occupancy 2**: 128A + 128V budget. Working set (~120 VGPR live)
     barely fits BUT every K-tail / N-tail / mul / store epilog needs short
     temporaries that no longer fit → cascading 197 spill (parallel R48
     measured exactly this).

  2. **Drop occupancy to 1**: 256A + 256V available (matches `rcr_4w`).
     Requires editing `__launch_bounds__(NT, 1)` to `__launch_bounds__(NT)`
     (or removing the `2` blocks/CU hint). This is **structurally
     equivalent to the C-2 path** — porting the kernel to the rcr_4w
     occ=1 register profile. EV depends on whether the doubled per-block
     VGPR budget outweighs the halved memory ILP from occ=2 → 1; this
     trade-off is exactly what the R8-dm / R47 ladder identified as
     "C-2 warp-tile restructure (multi-round, R50+ fallback)".

So C-3 step 1 (typedef-only, no occupancy / warp-tile change) cannot
land at an AGPR allocation that doesn't regress spill. The two viable
end-states are:
  * Keep occ 2 → +160 spill regression (parallel R48 confirmed)
  * Drop to occ 1 → C-2 territory (multi-round restructure)

Either way, the "step 1" (typedef-only, infra-only commit) does not
buy any incremental signal: building it would either reproduce the
parallel-R48 spill regression (if `art_base` is honored) or be
bit-identical (if LLVM's allocator ignores the storage hint, same as
the C-4 mfma-vgpr-form flag tickle). Neither outcome advances the
ladder beyond what we already know from parallel R48 + R48-this-round.

## Falsification register update (gpt_oss FP8 task)

| Lever                                            | Status        | Source              |
|--------------------------------------------------|---------------|---------------------|
| C-1 (restrict / lifetime hints)                  | SATURATED     | R12, R54            |
| C-3 `+a` inline-asm AGPR hint                    | FALSIFIED     | parallel R48 (bf16) |
| C-4 `-mllvm -amdgpu-mfma-vgpr-form=0`            | FALSIFIED     | R48 this run        |
| Other 6 amdgpu mllvm flags                       | A-PRIORI FALS | R48 this run        |
| **C-3 step 1 art_base typedef (R47/R48 plan)**   | **A-PRIORI FALS** | **R49 this round** |
| C-2 warp-tile restructure (4w occ=1)             | NOT STARTED   | R50+ fallback       |
| **C-1' per-spec KREM_HINT=64 constexpr**         | **NEXT R50**  | parallel R48 hand-off |
| K-tail micro-knobs (vmcnt / reorder)             | SATURATED     | R3-R55              |

## R50 forward-pointer — C-1' per-spec KREM_HINT=64 constexpr template

The next concrete experiment is **independent of the V+A budget blocker**
and borrows the parallel R48 (bf16) "Best ROI (1-2 rounds)" recommendation.
It targets the K-tail (K%128==64) which every gpt_oss K=2880 shape pays:

1. Add a 4th template parameter `int KREM_HINT = 0` to:
   * `grouped_rcr_kernel<int FUSE_ACT, bool TRANS_A, bool TRANS_B, int KREM_HINT = 0>`
     (line 3024 in `kernel_fp8_layouts.cpp`)
   * `grouped_var_k_kernel_fp8` (line 8418-ish — confirm line number at
     edit time, file has shifted since the parallel-R48 snapshot)

2. Inside the K-tail block (search for `b128_lo_valid` / `K_REM` /
   `FUSED_KTAIL` blocks): when `KREM_HINT == 64`, replace the runtime
   per-lane validity check
   ```cpp
   bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;  // runtime, K_REM in regs
   ```
   with a constexpr conditional based on lane id
   ```cpp
   constexpr bool b128_lo_valid = (laneid < 32);  // K_REM=64 → first 32 lanes valid
   ```
   (or equivalent — the exact form depends on the 16/32-lane mapping; verify
   by inspection of the existing K-tail block).

3. Add explicit template instantiations at the bottom of the .cpp:
   ```cpp
   template __global__ void grouped_rcr_kernel<0, true, true, 64>(...);
   template __global__ void grouped_var_k_kernel_fp8</*existing*/, 64>(...);
   ```

4. Update `dispatch_grouped_rcr` / `dispatch_grouped_var_k_fp8` (in the
   pybind layer at file bottom) to pass `KREM_HINT=64` when the binding
   is called with `K=2880` (the universal gpt_oss K). Default `0` for
   all other callers — backwards compatible.

5. Build, read the `-Rpass-analysis=kernel-resource-usage` remark for
   the new `<*,*,*,64>` instantiation. Goal: VGPR spill 37 → ~25-30,
   scratch 152 → ~100-120. (~5-10 VGPR freed from removed sentinel
   compute + scratch reduction proportional.)

6. If resource report shows the predicted reduction → run probe (3
   gpt_oss shapes × 3 sections × 7 trials × 200 iter p20). If TFLOPS
   > +1% on at least 4 of 8 shapes with no SNR regression → ship rule
   in `select_default_config` returning `HipKittenConfig(kernel="grouped_rcr_kernel_krem64")`
   for the gpt_oss family (`tiles_n in {11, 22} and k == 2880`).

7. EV: parallel R48 (bf16) estimated 5-10 VGPRs freed × ~5 K-tail-pass
   per kernel + a few cy/lane saved. On the gpt_oss FP8 cluster (8 cells
   averaging 1851/2029/1748 T fwd/dgrad/wgrad, target 2800), even 2-3pp
   per cell across 8 cells × 3 sections ≈ +6-12 score. Sub-5pp target
   honest for a 1-2 round investment.

## Why this is worth doing instead of giving up

The patience streak is now 7 / 40, and the score has plateaued at
691-693 for 5+ rounds within the 6-pt noise window. The gpt_oss task's
remaining structural levers (C-1', C-2, Stream-K) are exactly the kind
of multi-round investments the patience budget exists for. C-3 has now
been formally retired from the search space (this round) with a
concrete forward pointer to C-1' that is independent of the V+A
register budget that killed C-3.

## What this round changed

* `analysis/_notes/round-49-fp8-grouped-C3-art_base-typedef-A-PRIORI-FALSIFIED-by-parallel-R48-plus-a-hint-result.md`
  — this file (Primus-Turbo only; HK working tree clean).
* No code changes in either repo (Primus-Turbo Python untouched, HK
  source untouched).
* No metric / test edits.

## Scoring expectation

NEUTRAL round, daemon-canonical metric expected within the 691-699
window characterised by R29 (23-sample bit-equivalent baseline). No
code shipping = score should land near R47/R48 (693/692). This is
explicitly a planning + falsification round; the value is removing
C-3 from the search space + handing R50 a concrete experiment that
is structurally independent of the falsified path.

## Attribution

* HipKittens HEAD: `49ffb984` — UNCHANGED this round
* Primus-Turbo: only this doc note
* No `config.py` / `dispatch.py` / kernel changes
* No metric / test edits
