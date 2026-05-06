# Round 89 — bf16 grouped GEMM weighted wall (auto_optimize round 12/100)

> **Context:** auto_optimize round 12/100. R88 closed Lever B5 at the
> +5-score gate (build-gate passed but metric flat) and pivoted R12 to
> Lever B6 step 1 — VGPR live-range profile of `grouped_var_k_kernel`,
> the kernel R87 PMC identified as the actual 50% LDS-BC bottleneck and
> the source of the 256-VGPR/Occ-2 ceiling that's blocked R83-R88 (5
> falsified swizzle/prefetch rounds + 1 diagnostic). Goal: produce a
> per-helper VGPR-trim payoff signal for R13's recompute-instead-of-store
> attack.

**Status:** R89 = **profile + structural-prep round**. var_k production
.so is bit-identical to baseline (256 VGPR / 0 scratch / Occ 2 / 0 spill).
Two helper functions extracted with a build-toggleable inline/noinline
attribute — production builds default to `__attribute__((always_inline))`
(zero codegen change), a `-DPROFILE_VAR_K_NOINLINE` flag flips them to
`__attribute__((noinline))` for offline VGPR-cost measurement.

| run                                            | n | mean | std | gpt_oss / DSV3 / Qwen3 geomean |
|------------------------------------------------|--:|-----:|----:|-------------------------------:|
| R88 baseline (commit c599537, post-revert)     | 5 | 883.4 | 1.0 | 1.094 / 1.122 / 1.111         |
| R89 helpers extracted (Build A, always_inline) | 5 | 882.6 | 0.5 | 1.095 / 1.119 / 1.111         |

Δ-mean = −0.8 (within 1 σ). All 24 shapes PASS correctness. Backward
bench (24-shape grouped fwd+bwd, untracked CSV):
fwd 1267.47→1268.37 TFLOPS (+0.07 %), bwd 1024.15→1027.00 TFLOPS (+0.28 %)
— both within run-to-run noise.

## Implementation

`analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

* Inserted before `grouped_var_k_kernel` (≈ line 4760):
  - `struct var_k_coord_result_t { int pid_m, pid_n, valid; }`
  - `struct var_k_group_lookup_t { int group_idx, local_tile, m_start_g, ki_g, valid; }`
  - `compute_var_k_coords(local_tile, num_pid_m, num_pid_n, group_m)`:
    extracted persistent-loop coord swizzle (originally lines 4856-4874,
    the most complex inline block — 2-way layout branch + 6-7
    intermediate locals per branch + skip-on-OOB). Returns the new
    `var_k_coord_result_t` POD.
  - `compute_var_k_group_lookup(gt, tiles_per_group, s_offs)`:
    extracted group-lookup + ki_g <2 skip (originally lines 4842-4847).
    Returns `var_k_group_lookup_t` POD with `group_idx`, `local_tile`,
    `m_start_g`, `ki_g`, and validity flag.
* `grouped_var_k_kernel` body now calls both helpers in order:
  ```cpp
  const auto gl = compute_var_k_group_lookup(gt, tiles_per_group, s_offs);
  if (!gl.valid) continue;
  ...
  const auto coords = compute_var_k_coords(local_tile, num_pid_m, num_pid_n, g.group_m);
  if (!coords.valid) continue;
  ```
* Build-time attribute:
  ```cpp
  #ifdef PROFILE_VAR_K_NOINLINE
  #define VAR_K_HELPER_ATTR __attribute__((noinline))
  #else
  #define VAR_K_HELPER_ATTR __attribute__((always_inline))
  #endif
  ```
  Production build (no flag) → `always_inline` → byte-identical codegen
  to baseline. Profile build (`-DPROFILE_VAR_K_NOINLINE`) → `noinline`
  → forces ABI-clean call boundaries that surface the helper's
  caller-save spill cost in `-Rpass-analysis`.

ST_A / ST_B / register tile types unchanged. Kernel logic unchanged.
Forward grouped + dense `gemm_kernel` paths untouched. RCR/RRR/CRR
all verified byte-identical.

## VGPR profile data (R12 deliverable)

Build-A (production, `always_inline`):

| kernel                              | VGPR | Scratch B/lane | Occ | VGPR Spill |
|-------------------------------------|-----:|---------------:|----:|-----------:|
| `grouped_var_k_kernel<0>`           | 256  | 0              | 2   | 0          |

Bit-identical to baseline (R87 commit 4bbc00e: 256 / 0 / Occ 2 / 0).
The helpers fully fold into the kernel; no codegen difference.

Build-B (`-DPROFILE_VAR_K_NOINLINE`, both helpers `noinline`):

| kernel                              | VGPR | Scratch B/lane | Occ | VGPR Spill |
|-------------------------------------|-----:|---------------:|----:|-----------:|
| `grouped_var_k_kernel<0>`           | 256  | **160**        | 2   | **38**     |

Δ vs Build-A:
* VGPR ceiling unchanged (still 256 — both helpers' computation lives
  inside the function call and is rematerialised per-call).
* +160 byte/lane scratch + 38 VGPR spill — ABI-mandated caller-save
  state at the two `noinline` call boundaries.

The +38 VGPR spill is the **spill cost the always-inline build avoids
by inlining**, i.e. the always-inline build has roughly 38 VGPR-
equivalent of cross-call live-range that must coexist with the helper
locals during the inlined sequence. Saying it inversely: **inlining
both helpers fits ~38 VGPR worth of helper-local state into the
already-256-VGPR kernel without exceeding the ceiling.**

LLVM `-Rpass-analysis=kernel-resource-usage` only emits per-kernel
(global) reports, not per-`__device__`-helper, so the helpers'
standalone VGPR is not directly visible. The Build-A vs Build-B delta
(scratch + spill) is the available proxy and is consistent with a
combined helper live-range of ~30-40 VGPR.

## What this means for R13 (recompute-instead-of-store trim)

The 38 VGPR spill in Build-B is **not** an unavoidable kernel cost —
it's the cost of forcing ABI calls. The always-inline build at 256
VGPR / 0 spill / 0 scratch shows the kernel CAN absorb the helpers'
combined live-range without any ceiling penalty. So R13's actual
trim payoff isn't "save 38 VGPR by reducing helper count" — it's:

> Of the helper outputs that flow into the persistent loop body
> (`group_idx, local_tile, m_start_g, ki_g, pid_m, pid_n`), which can
> R13 RECOMPUTE inside `device_gemm_tile_body` from already-live state
> (e.g. derive `m_start_g` from `s_offs[group_idx]` at body entry
> instead of materialising it before the call) so the corresponding
> SSA value drops out of the persistent-loop live-range?

Concrete R13 candidates (ordered by speculative payoff):

1. **`m_start_g` rematerialisation**: pass only `group_idx` (1 int)
   to `device_gemm_tile_body`; recompute `m_start_g = s_offs[group_idx]`
   at body entry from the LDS-cached `s_offs` array. Saves 1 VGPR.
2. **`pid_m, pid_n` row/col fold**: already aliased to `row, col`
   immediately at lines 4885-4886. The aliasing is free; removing
   the aliased pair would touch the body's signature → riskier.
3. **`ki_g` materialisation**: live across the entire main loop;
   used in `device_gemm_tile_body` as the K-loop bound. Cannot trim
   without changing body signature.

The 1-VGPR `m_start_g` win alone is too small to unblock A1 (forward
K-tail dual-A prefetch needs +8 VGPR slack). Multi-helper combined
trim plausibly buys 4-6 VGPR (= 1 m_start_g + 1 ki_g cached as derived
from group_idx + 2-4 from the persistent-loop's own scalar bookkeeping).
**R13's expected output: 4-6 VGPR margin restored**, falling short of
B6's 8-12 VGPR target — extends the unblock sequence to R13-R15 (with
R14 attacking the body's accumulator-side live-range or persistent-
loop chiplet/bookkeeping locals if R13's prologue trim isn't enough).

## What does NOT work / Cumulative falsification streak (R83-R89)

| Round | Lever                                               | Verdict | Cause |
|-----:|-----------------------------------------------------|---------|-------|
|   83 | RCR FUSE=true KI=88 dual-A prefetch                 | -3 | grouped<RCR,88,1> +9 spill drops Occ 2→1 |
|   84 | RCR FUSE=true KI=44 LDS pre-shuffle                 | -7 | grouped<RCR,44,1> +28 spill drops Occ 2→1 |
|   85 | var-K KI=32/64 inner-loop unroll                    | -4 | var_k +14-18 spill |
|   86 | st_32x16_v2 within-half swizzle (var_k + grouped CRR) | -5 | var_k +17 spill; grouped CRR +8 spill |
|   87 | (diagnostic / pivot only — PMC + plan)              | -  | (no code change) |
|   88 | st_32x16_v2 RRR ST_B (pad-only)                     | flat | Build-gate passes; +8-24 scratch on hot RRR KIs cancels any 1-BC/Inst gain |
|   89 | (profile-only — extract var_k helpers; no metric ask) | flat | Same .so as baseline; profile data committed for R13 |

Round 89 is the first **structural-prep** round. No metric ask, but
the source change is committed because (a) it's bit-equivalent to
baseline in the production build and (b) it leaves R13 a clean unit
of trim (the helper signature) without re-extracting next round.

## Files touched

* HipKittens repo: `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  — added 2 POD structs, 2 helpers, build-toggle macro; rewrote 12
  inline lines in `grouped_var_k_kernel` to call them.
* Primus-Turbo repo:
  `analysis/_notes/round-89-bf16-grouped-var-k-helper-extraction-r12-profile.md`
  (this file).
* `/tmp/build_round12_A2.log` (production build, `always_inline`,
  baseline-identical resource report).
* `/tmp/build_round12_B2.log` (`-DPROFILE_VAR_K_NOINLINE` build, +160
  scratch / +38 VGPR spill, profile-only, NOT shipped).

## Metric / numbers

* R88 baseline (5 runs):     883, 885, 883, 882, 884 → mean 883.4, std 1.0.
* R89 production (5 runs):   882, 882, 883, 883, 883 → mean 882.6, std 0.5.
* Δ-mean = −0.8 (within 1 σ; bench shows fwd +0.07 %, bwd +0.28 %).
* All 24 shapes PASS correctness:
  - DSV3-GateUP-B16-M2048: y=0.42 % / dA=0.53 % / dB=0.73 % rel-err.
  - gpt_oss-Down-B4-M2048: y=0.71 % / dA=0.66 % / dB=0.41 % rel-err.
  - Qwen3-Down-B16-M4096: y=0.51 % / dA=0.39 % / dB=0.50 % rel-err.

## Recommendation for round 13

Execute Lever B6 step 2 — `m_start_g` rematerialisation: drop
`m_start_g` from `device_gemm_tile_body`'s argument list, instead
read `s_offs[group_idx]` once at body entry. Combined with passing
`ki_g` derived from `s_offs[group_idx+1] - s_offs[group_idx]` at
body entry, this removes 2 ints from the persistent-loop live-range.
Expected outcome: var_k 256 → 252-254 VGPR (2-4 VGPR margin), Occ 2
preserved, scratch 0. If the trim is more than 4 VGPR, retry the
B6-step-3 swizzle landing on var_k CRR ST_A + ST_B in R14. Otherwise
extend B6 with one more prologue trim (e.g., aliasing `row, col` in
the body signature instead of via the persistent loop) in R14 and
defer the swizzle attack to R15.
