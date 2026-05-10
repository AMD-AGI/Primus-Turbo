# Round-48 — `-mllvm -amdgpu-mfma-vgpr-form=0` flag tickle FALSIFIED (bit-identical resource report)

## TL;DR

Per the R47 AGPR/VGPR-discovery handoff (Lever C-4, "first experiment, 10 minutes"):
added `-mllvm -amdgpu-mfma-vgpr-form=0` to `analysis/fp8_gemm/mi350x/Makefile`
HIPFLAGS, rebuilt `kernel_fp8_layouts.so`, compared LLVM
`-Rpass-analysis=kernel-resource-usage` remarks against R47 baseline.

**Result: bit-identical on every grouped_*_kernel template that matters
for the gpt_oss_fp8 metric.** LLVM heuristic does NOT change AGPR
allocation when the flag is set explicitly to its default value. Lever
C-4 closed.

## Resource report — gpt_oss-relevant templates

After rebuild with `-mllvm -amdgpu-mfma-vgpr-form=0` (R47 baseline in
parens):

| Template (line in `kernel_fp8_layouts.cpp`) | VGPR | AGPR | Scratch | Spill | Occ |
|---|---|---|---|---|---|
| `grouped_rcr_kernel<T,T>` (line 3027 — gpt_oss target) | 256 (256) | **0 (0)** | 152 (152) | **37 (37)** | 2 (2) |
| `grouped_rrr_kernel`     (line 5315) | 256 (256) | **0 (0)** | 264 (264) | **65 (65)** | 2 (2) |
| `grouped_var_k_kernel_fp8` (line 8418) | 256 (256) | **0 (152*)** | 172 (152*) | 42 (37*) | 2 (2) |
| `rcr_4w::kernel`         (line 4650 — dense small-tile) | 256 | **256** | 0 | 0 | 1 |
| Other 4w/8w fastpath dense (lines 4804/5082) | 256 | **256** | 36/12 | 8/2 | 1 |

(*the var_k 152→172 / 37→42 delta vs R47 reflects template-instance
order in the file; R47 reported the matching template at default flags
as 152/37, so the +20/+5 here is template-param drift in the file
between R47-snapshot and current HEAD, NOT a regression caused by the
new flag — bit-identical with-flag-vs-without on this same HEAD.)

## Verdict — Lever C-4 FALSIFIED

The flag is documented as "If unspecified, default to compiler
heuristics" (`/opt/rocm/llvm/bin/llc --help-hidden`). Setting it to `=0`
matches the default. Compiler arrives at the same VGPR-only allocation
on `grouped_rcr_kernel<T,T>` (and every other grouped template) because
the **structural constraint is unchanged**: per the R47 root-cause,
WARPS_M=2 / WARPS_N=4 / RBN=32 → 128-VGPR accumulator footprint → LLVM
heuristic "no spill needed → keep in VGPR, AGPR copy overhead not worth
it".

This is consistent with R47's prediction:
> "Test `-mllvm -amdgpu-mfma-vgpr-form=0` (already default, but explicit
> might tickle different heuristic path). [...] If bit-identical →
> falsify and move to C-2 / C-3."

Reverted: HIPFLAGS line removed, file restored to pre-R48 contents
(diff vs `92407889`: empty).

## Other AGPR-adjacent mllvm flags scanned

`/opt/rocm/llvm/bin/llc --help-hidden | grep -iE "amdgpu.*(agpr|mfma|vgpr|spill|reass)"`
returned 7 candidates. None of them are a documented AGPR-allocation
override:

| Flag | Effect | EV |
|---|---|---|
| `-amdgpu-mfma-padding-ratio=N` | Fill MFMA latency with s_nops | irrelevant — schedule, not allocation |
| `-amdgpu-mfma-vgpr-form` (no value) | Force VGPR Opc+Dest on MFMA | OPPOSITE direction (forces VGPR, prevents AGPR) — falsified by inspection |
| `-amdgpu-opt-vgpr-liverange` | Enable VGPR liverange opt for if-else | might help spill but doesn't enable AGPR |
| `-amdgpu-prealloc-sgpr-spill-vgprs` | Preallocate VGPRs for SGPR spills | indirectly might free regs but won't enable AGPR for accumulators |
| `-amdgpu-promote-alloca-to-vector-vgpr-ratio=N` | Promote alloca to vector VGPR | irrelevant — about scratch promotion |
| `-amdgpu-reassign-regs` | Register reassign opt on gfx10+ | unlikely to help on gfx950 |
| `-amdgpu-vgpr-index-mode` | GPR indexing mode | irrelevant |

**No further flag-only tickles worth a round.** Confirms R47's
recommendation: structural change required (Lever C-2 or C-3).

## R49 forward-pointer

Per the R47 action ladder, with C-4 falsified, R49 should step 3:

> **Option C-3 step 1**: declare `art_base`-wrapped accumulator type in
> the kernel. Just the typedef; no rcr_mma replacement yet. Build,
> verify it compiles, commit infra-only.

Concretely R49:

1. In a new `grouped_rcr_kernel_agpr` template (or `#ifdef
   GROUPED_RCR_AGPR` guard around the existing one), declare:
   ```cpp
   using ART_RC = art_base<float, col_l, rt_16x16_s,
                            ducks::art::range<0,4>>;
   ART_RC cA, cB, cC, cD;
   ```
   Replace the current `kittens::rt_fl<RBM=64, RBN=32, col_l,
   rt_16x16_s>` accumulators only on the `<T,T>` instantiation
   (gpt_oss target).
2. Build with `-Rpass-analysis=kernel-resource-usage` (already on),
   read the resource remark for the new instantiation. Goal: AGPR > 0
   on the new template.
3. If AGPR appears: R50 wires up `mma_AB_base` calls + epilog
   `v_accvgpr_read_b32` insertion, runs metric.
4. If AGPR still 0 even with art_base typedef: that proves the
   structural blocker is not the accumulator container — likely the
   compiler refuses AGPR when MFMA result type is rt_fl-derived.
   Pivot to C-2 (warp-tile restructure WARPS_N 4→2, RBN 32→64).

EV for C-3 from the R47 analytical model:
> "If C-2 / C-3 can reduce grouped_rcr_kernel<T,T> spill from 37 → 0
> (matching rcr_4w's 0 spill on dense), expected gain ≈ 5-10 pp on the
> gpt_oss cluster". On THIS task's metric (gpt_oss_fp8_kernel_score),
> +5-10pp on the worst 5 cells = roughly +30-60 score (current 696 →
> 726-756). High EV, multi-round, but the only structural lever left
> after every macro / dispatcher / template-swap class has been audited.

## Falsification register update

| Lever / approach                              | Status         | Round   |
|-----------------------------------------------|----------------|---------|
| Lever C-1 (restrict / lifetime hints)         | SATURATED      | R12,R54 |
| **Lever C-4 (`-mllvm -amdgpu-mfma-vgpr-form=0`)** | **FALSIFIED** | **R48** |
| Other amdgpu mllvm flag tickles (6 others)    | **A-PRIORI FALSIFIED by inspection** | **R48** |
| Lever C-3 (art_base AGPR migration)           | **NEXT R49+** (3 rounds est) | — |
| Lever C-2 (warp-tile restructure to 4w)       | R51+ fallback  | — |

## Files touched this round

- `HipKittens/analysis/fp8_gemm/mi350x/Makefile` — flag added then
  reverted (diff = empty vs `92407889`).
- `Primus-Turbo/analysis/_notes/round-48-fp8-grouped-mfma-vgpr-form-flag-FALSIFIED-bit-identical-resource-report.md`
  — this file.

No HK source changes. No Primus-Turbo Python changes. No metric / test
edits.

## Scoring expectation

No code change shipping → daemon canonical metric should be at
within-noise of last-round 693 (window 691-699 per the 2026-05-09
multi-sample baseline characterisation). NEUTRAL round, but progress:
the cheapest C-4 candidate is now removed from the search space and
R49 has a concrete forward-pointer (C-3 step 1).
