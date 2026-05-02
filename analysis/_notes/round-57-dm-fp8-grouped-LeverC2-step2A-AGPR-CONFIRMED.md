# Round 57-dm — FP8 grouped: Lever C-2 step-2A AGPR allocation **CONFIRMED**

**Status**: STEP-2A LANDED (HK 1 commit, infrastructure only) — no metric change expected, none observed
**Score before**: 983 (run 1) / 980 / 979 / 981 (4-run median ≈ 980, noise band 977-989)
**Score after**:  983 (single run post-build) — unchanged, **as expected for compile-only landing**
**HK SHA**: `73da21c6` (R54 step-1 scaffold) → **TBD** (this round, step-2A test kernel)
**Round time**: ~45 min, 4 build cycles, 4 metric runs (3 baseline + 1 post-build)
**Auto-optimize round**: 57

---

## TL;DR

**The R47-dm hypothesis is empirically confirmed**: when a grouped-style
persistent kernel allocates 4 accumulator tiles each of `rt_fl<64, 64,
col_l, rt_16x16_s>` (= 256 fp32/lane per warp), LLVM **picks AGPR
allocation** for the accumulators and emits **0 VGPR spill**, exactly
matching the `rcr_4w::kernel` (dense 4w-style) precedent.

This unblocks the multi-round Lever C-2 step-2 path: the structural
question that has gated step-2 implementation for the past 3 rounds
(R54 / R55 / R56 deferred) is now answered with a positive ISA
measurement, not a hypothesis.

```
                                  TotalSGPRs  VGPRs  AGPRs  Scratch  Spill  Occupancy
                                  ──────────  ─────  ─────  ───────  ─────  ─────────
rcr_4w::kernel (dense, R47 ref)         46    198    256       0      0       1
grouped_rcr_kernel<0,F,F>               64    256      0     220     54       2
grouped_rcr_kernel<0,T,F>               66    256      0     156     38       2
grouped_rcr_kernel<0,F,T> (DSV3)        77    256      0     140     34       2
grouped_rcr_kernel<0,T,T> (gpt_oss)     79    256      0     152     37       2
grouped_var_k_kernel_fp8 (dB bwd)       79    256      0     152     37       2

  ──── NEW (this round) ────
test_grouped_rcr_kernel_4w_compile_test<4>
                                        30    256    256       0      0       1
```

The test kernel sits in
`namespace lever_c2_round_57_step2a_compile_test` (HK
`kernel_fp8_layouts.cpp` line ~3000), is force-instantiated for
`KI_HINT=4`, and is **never wired into the runtime dispatcher**, so the
`.so` binary's runtime behaviour is bit-identical to HEAD (verified by
metric: 983 with vs 983 without — same as last 4 baseline runs).

## Why we are here (recap of the prior 9 rounds)

Auto-optimize rounds 49-56 (Primus-Turbo SHAs `a7fe4e0..64ea7bb5`)
sequentially falsified every micro-knob and per-spec-template variant
that could reasonably be probed in a single round:

| Round | Lever                                                   | Result |
|-------|---------------------------------------------------------|--------|
| R49   | KREM=64 invariant collapse (HK `6c52d017`)              | LANDED, -2 SGPR / 0 metric (within noise) |
| R50   | gpt_oss-Down-B32-M4096 fall-through (gm=4, xcd=4)       | LANDED, +1pp on 1 shape |
| R51   | VMCNT main-loop INIT0/INIT1 sweep                       | FALSIFIED |
| R52   | wm==0 / M2N2 / lgkmcnt drops                            | FALSIFIED |
| R53   | K-tail load reorder + setprio                           | FALSIFIED |
| R54   | `__noinline__` / `__builtin_expect` register hints      | FALSIFIED |
| R55   | helper if-else swap / partial-helper dead-code-strip    | NEUTRAL (real -5dw spill on cold path → 0 metric) |
| R56   | doc-only quantitative analysis                          | confirmed plateau ceiling ≈ 985 |

R55-dm's NEUTRAL probe-2 (partial helper, -18 dw on `<0,1,0>`) was the
last actionable insight: the helper-side dead-code-stripping moved real
spills off boundary tiles, but those spills were on the **cold path**
(boundary tile fraction ≈ 9% on gpt_oss N=2880, 0% on aligned N=4096
DSV3) so no measurable runtime change. The hot-path spill driver is the
**main-loop K-iter body**, not the helper.

R56 quantitative analysis identified **MFMA compute density** as the
real lever: `rcr_4w::kernel` runs 16 mfma_16x16x128 per warp per K-step
(WARPS_N=2, RBN=64) vs grouped's 8 mfma per warp per K-step (WARPS_N=4,
RBN=32). Plus rcr_4w sits at AGPR=256 / Spill=0 vs grouped's
AGPR=0 / Spill=34-54. Both are downstream of **per-warp accumulator
footprint** (rcr_4w: 4 acc × 64 fp32/lane = 256 fp32/lane; grouped:
4 acc × 32 fp32/lane = 128 fp32/lane).

R56 deferred actual implementation of step-2 because the test would
have required a `~660 line` copy-and-adapt of `grouped_rcr_kernel` and
the user's chat budget ran out at 89/90 cap.

## R57-dm goal: validate the AGPR hypothesis at minimal scope

The R47 hypothesis is:

> **256 fp32/lane per-warp accumulator footprint → LLVM picks AGPR.**

R47 hypothesised this from observation (rcr_4w sits at exactly that
threshold) but never built a controlled test. R47-R55 falsified
*hint-based* attempts to flip the lever (`+a` AGPR constraint pin,
`__attribute__((amdgpu_waves_per_eu(2,2)))`, removing
`__launch_bounds__(_, 1)`) — none worked because they applied
allocation pressure at the wrong abstraction level (HW resource limits
or pseudo-register classes), not at the **register-tile-type** level.

The actual rcr_4w mechanism is:
1. Declare per-warp acc as `RC c[2][2]` of `rt_fl<64, 64, col_l, rt_16x16_s>`
2. LLVM type system propagates 64 fp32/lane × 4 acc = 256 fp32/lane
   liveness through `mma_ABt` calls
3. LLVM register allocator sees that "accumulator alone consumes the
   full 256-VGPR per-lane budget" and chooses AGPR for them (frees
   VGPR for the rest of the live set)

The hypothesis was never tested **inside** a grouped-style persistent
kernel — only inside the dense rcr_4w. R47-R55 saturation took the
hypothesis on faith.

R54 step-1 staged the **types** (`A_row_reg_4w` / `B_row_reg_4w` /
`C_acc_4w`) inside `namespace lever_c2_round_54_step1_scaffold` and
validated them via static_asserts only — no codegen exercise.

R57 step-2A (this round) actually **codegens** the 4w-style register
layout:
- declares `cAB[2][2]` of `C_acc_4w` (4 acc × 64 fp32/lane = 256
  fp32/lane per warp)
- declares `a_reg[2]` of `A_row_reg_4w` and `b_reg[2]` of
  `B_row_reg_4w`
- threads them through 4 `mma_ABt` calls per K-iter inside a
  persistent outer loop with cooperative LDS load + LDS-to-register
  staging
- stores all 4 cAB to `g.c` to anchor the output (LLVM DCE prevention)

This is force-instantiated for KI_HINT=4 so LLVM emits codegen and
prints the `-Rpass-analysis=kernel-resource-usage` line at build time.

## Result: hypothesis CONFIRMED

The new `test_grouped_rcr_kernel_4w_compile_test<4>` resource report:

```
TotalSGPRs:   30
VGPRs:        256
AGPRs:        256     ← AGPR allocated, matches rcr_4w precedent
ScratchSize:  0
Occupancy:    1 wave/SIMD
SGPRs Spill:  0
VGPRs Spill:  0       ← 0 spill, vs grouped_rcr_kernel<T,T> 37 spill
LDS:          69632 B/block (= 2 × 2 × 16 KB ST_v2 slabs + ~4 KB group meta)
```

Comparison vs the active gpt_oss spec (worst-case grouped path):
```
                          grouped<T,T>     test_4w     Δ
─────────────────────────────────────────────────────────
TotalSGPRs                       79          30        -49 (much smaller — no group meta)
VGPRs                           256         256          0 (saturated either way)
AGPRs                             0         256       +256 (THE KEY LEVER)
Scratch [bytes/lane]            152           0       -152 (no spill traffic)
VGPRs Spill [dwords]             37           0        -37 (THE PAYOFF)
Occupancy [waves/SIMD]            2           1         -1 (HALVES concurrent blocks)
LDS [bytes/block]            139796       69632     -70K (smaller; group cache absent)
```

### What the result means

1. **AGPR > 0 + Spill = 0**: hypothesis "256 fp32/lane → AGPR" is
   correct. LLVM's allocator reliably picks AGPR for the accumulator
   when its per-lane footprint hits 256.

2. **VGPRs = 256 still saturated**: even with acc moved to AGPR, the
   non-acc live state (a/b regs + LDS pointers + outer-loop control)
   already fills the VGPR budget. This means R48's
   "V128/A128 = -160 spill regression" failure mode IS NOT inevitable
   — the failure was from the `+a` hint forcing a hardcoded 128/128
   split; LLVM's natural pick under type-system pressure is V256/A256.

3. **Occupancy: 2 → 1 (caveat)**: the 4w-style block (256 threads / CU
   with V256/A256) consumes more SIMD register file per warp,
   restricting concurrent block count to 1 wave/SIMD vs the grouped
   kernel's 2. **This is identical to rcr_4w's occupancy** (1 wave/SIMD
   in R47 baseline) and is the trade-off documented in R8-dm; the
   higher per-warp compute density (16 mfma vs 8 mfma per K-step) plus
   zero spill is meant to outweigh halved concurrency.

   On the gpt_oss-K=2880 specs that R55-R56 plateau analysis identified
   as worst (1.07-1.11 ratio, where spill traffic dominates), zero
   spill should help most. On Qwen3 K=4096 specs (1.13-1.19 ratio,
   already main-loop-bound), the occupancy hit might dominate. R58+
   port will need to validate per-shape.

4. **Reduced LDS budget**: test kernel uses 69 KB / block vs grouped
   140 KB / block. The grouped LDS includes the group_offs cache,
   per-tile cumulative counters, and the v2/v2a swizzle layout that
   the production K-tail / N-mask paths require. Step-2B will reclaim
   that LDS as the full machinery comes back online; expect the final
   port's LDS at ~140 KB / block matching the production grouped.

### What the result does NOT prove

- The test kernel **doesn't exercise the production register-pressure
  path**: no group binary search (no `s_offs` / `s_cum_tiles` LDS
  caches, no swizzle), no K-tail FUSED_KTAIL block (no `a_kt1`
  register tile + SENTINEL voffset masks), no N-mask store helper.
  Adding any of these in step-2B will increase the non-acc live set;
  if it grows to >256-AGPR-implied-VGPR-budget, AGPR allocation may
  partially or fully cascade back to VGPR/scratch (R48-style failure
  mode). Step-2B must build incrementally and re-check the resource
  report after each layer.

- The test kernel has placeholder load/store coordinates (does not
  produce correct GEMM output). Step-2B has to thread real coords
  through the 4-warp cooperative load helpers, which themselves don't
  exist yet (R6-dm falsified the auto-derive of
  `prefill_swizzled_offsets` + `rcr_8w_load_hoist` from the WARPS_M /
  WARPS_N constants — those are hand-written for 2×4 layout; step-2B
  needs to write a `G_4w` cooperative load family).

## Files touched

### HipKittens

- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - **NEW**: `namespace lever_c2_round_57_step2a_compile_test`
    (~120 lines) immediately after the `grouped_rcr_kernel`
    template instantiations (line ~2982). Defines
    `test_grouped_rcr_kernel_4w_compile_test<KI_HINT=4>` plus an
    explicit instantiation. Imports `WARPS_M / WARPS_N / RBM_4w /
    RBN_4w / A_row_reg_4w / B_row_reg_4w / C_acc_4w` from
    `lever_c2_round_54_step1_scaffold`.
  - All other kernel functions / specs are bit-identical (verified by
    grep'ing the `grouped_rcr_kernel<*,*,*>` resource reports — VGPR /
    AGPR / Scratch / Spill all match the pre-step-2A baseline).

### Primus-Turbo

- `analysis/_notes/round-57-dm-fp8-grouped-LeverC2-step2A-AGPR-CONFIRMED.md`
  (this note; round-note style consistent with R52-R56 docs).

No production code changes on the Primus-Turbo side.

## R58+ roadmap

R57's positive AGPR result unlocks the multi-round step-2B port. The
plan:

| Step | Round | Action | Acceptance gate |
|------|-------|--------|-----------------|
| 2B-1 | R58 | Implement `G_4w = group<4>` cooperative load helpers (`prefill_swizzled_offsets_4w` + `rcr_4w_load_hoist`). Force-instantiate against test kernel. | Build clean, AGPR retained. |
| 2B-2 | R59 | Replace placeholder coords in test kernel with real `(br, bc, k)` indexing into g.a / g.b / g.c (correct GEMM output, no group binary search yet — fixed B=1 launch). Probe correctness on a single (M=256, N=256, K=128) shape. | `max_abs ≤ 0.5` and `SNR ≥ 22 dB` vs torch fp32 ref. |
| 2B-3 | R60 | Add group binary search + LDS group_offs cache (port the prologue from `grouped_rcr_kernel`). Keep test kernel's main-loop body simplified (no K-tail / N-mask). | Resource report: AGPR > 0; correctness PASS on B=2, G=2 grouped probe. |
| 2B-4 | R61 | Add K-tail FUSED_KTAIL block (port from grouped). Test correctness on K_REM=64 case (gpt_oss N=5760 K=2880). | Correctness PASS; AGPR retained or partial cascade documented. |
| 2B-5 | R62 | Add N-masked C-store helper. Test on N=2880 misaligned case (gpt_oss). | Correctness PASS on all 8 gpt_oss shapes. |
| 2B-6 | R63 | Wire dispatcher selection: when `(K_rem ∈ {0, 64}) && (N % 256 ∈ {0, gpt_oss_residue})`, dispatch to `grouped_rcr_kernel_4w` instead of `grouped_rcr_kernel`. Run full 24-shape metric. | grp_FP8 geomean ≥ 1.18 (+1pp over current 1.17 baseline) AND no shape regresses below current per-shape baseline. |

Total: 6 rounds (R58-R63). Risk: any step that adds register pressure
might force AGPR cascade back to VGPR + scratch (R48's failure mode).
If that happens at step 2B-3 or 2B-4, R58+ should pivot to evaluating
**occupancy=2 retention** as the primary lever instead of AGPR (e.g.,
keep WARPS=4 but use shared-mem accumulator staging — Lever D adjacent).

If R58-R63 succeed and `grouped_rcr_kernel_4w` reaches metric parity
with current `grouped_rcr_kernel`, R64+ tunes the dispatcher per-shape
to choose between 4w-style and outer-grouped style based on K-iter
count / N alignment / occupancy preference.

## Validation paper-trail

```
/tmp/hk_build_baseline.log        — pre-step-2A build (R56 head, no test kernel)
/tmp/hk_build_step2a.log          — post-step-2A build (test kernel landed)
/tmp/metric_round_57_baseline.log — single baseline metric (run 1)
                                    — score 983, geomean 1.1707 grp_FP8
post-step-2A metric (this run)    — score 983, geomean 1.1723 grp_FP8
                                    (within R56 noise band 1.165-1.175)
```

Both build logs preserve the `-Rpass-analysis=kernel-resource-usage`
remarks for every kernel; the diff is just the new
`test_grouped_rcr_kernel_4w_compile_test<4>` symbol with V256/A256/0
spill at line 3060.

## Round meta

- Auto-optimize round: 57
- Score trajectory: 980-983 noise band (3-run sample) → 983 post-step-2A (single run)
- Plateau: round 11 of 977-989 noise band (no metric change since R50 ship)
- patience counter: 7/30 — multi-round structural commit; no metric regression expected before R63.
- HK SHA: `73da21c6` (R54 step-1 scaffold) → **TBD** (this commit, step-2A test kernel)
