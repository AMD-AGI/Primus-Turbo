# Round 90 — bf16 grouped GEMM weighted wall (auto_optimize round 13/100)

> **Context:** auto_optimize round 13/100. R89 (round 12) committed
> structural prep (helper extraction with build-toggleable
> inline/noinline) for `grouped_var_k_kernel`, plus a R13 plan to
> execute Lever B6 step 2: `k_offset_tiles` + `ki_g` rematerialisation
> at `device_gemm_tile_body` entry to free 1-2 VGPR cross-call live-range
> margin for R14's swizzle re-attack.
>
> R13 falsifies that plan analytically (k_offset_tiles is body-wide
> live, not cross-call live) and tries a fallback lever (explicit
> readfirstlane on derived a_lds_xx / b_lds_xx LDS pointers) which
> turns out bit-identical to baseline.

**Status:** R90 = **analytical-falsification + secondary-lever-flat
round**. HK + Primus working trees clean (post-revert). Score in noise
band (881 single-run open, 5×883.0 close).

| run                                                        | n | mean | std |
|------------------------------------------------------------|--:|-----:|----:|
| R89 baseline (commit af70d1a, helpers extracted, post-R12) | 5 | 882.6 | 0.5 |
| R13 open metric (single run, idle GPU contention noise)    | 1 | 881   | -   |
| R13 readfirstlane (5 runs)                                 | 5 | 883.0 | 0.7 |

Δ-mean (R89 → R13 readfirstlane) = +0.4 (well within noise). Reverted
since the source change adds 8 lines of explicit readfirstlane noise
without a measurable codegen/runtime delta — the AMDGPU SI Convergent
Annotator pass was already auto-promoting the derived a_lds_00..11 /
b_lds_00..11 to SGPR, so the explicit annotation is redundant.

## Part A: R89 plan (k_offset_tiles + ki_g rematerialisation) — analytically falsified

R89 estimated saving 2 VGPRs by rematerialising `k_offset_tiles`
(= `m_start_g / K_STEP`) and `ki_g` (= `M_g / K_STEP`) inside
`device_gemm_tile_body` instead of passing them as arguments. Closer
inspection of `device_gemm_tile_body`'s lambda captures (line 526-537
of `kernel_bf16_dynamic.cpp`):

```cpp
auto a_coord = [&](int spatial, int k) {
    if constexpr (L == Layout::CRR)
        return coord<ST_A_T>{0, 0, k_offset_tiles + k, m_subtile_A + spatial};
    ...
};
auto b_coord = [&](int spatial, int k) {
    if constexpr (L == Layout::RCR)
        return coord<ST_B_T>{0, group_idx, spatial, k_offset_tiles + k};
    ...
};
```

The lambdas capture `k_offset_tiles` BY REFERENCE, and they are called
during the prologue (4 `G::load`s), every main_loop_iter, and both
epilogs. Effective live-range = **the entire body** (≈ 200+ VGPRs of
peak pressure including C_accum + A/B reg tiles + bookkeeping). Same
analysis applies to `ki_g`: it's the `num_tiles_dyn` loop bound for
KI_HINT == 0 (var_k's only specialization), used throughout main_loop.

Net VGPR saving from rematerialisation = **0**. Not 1-2. The
"cross-call live-range" framing in R89 was incorrect: there's no
distinct call boundary because both helpers and `device_gemm_tile_body`
are `__forceinline__`. The compiler sees one large SSA flow with
k_offset_tiles alive throughout the body's expansion.

R89's other speculative trims (row/col aliasing, ki_g recompute) hit
the same wall: they're all alive across the body, not just at call
boundaries. **Lever B6 step 2 is a dead lever.**

## Part B: R13 fallback (explicit readfirstlane on derived LDS pointers) — bit-identical

Hypothesis: explicitly applying `__builtin_amdgcn_readfirstlane` to
the 8 derived per-buffer LDS pointers (`a_lds_00..11`, `b_lds_00..11`,
each = `a_lds + k * sizeof(ST)` for k ∈ {0,1,2,3}) might force them
to SGPR if the AMDGPU compiler's uniformity propagation through
constant-add expressions wasn't already promoting them. Precedent:
line 1086-1089 (RRR FUSE Path B `b_arr_base_10/11`) explicitly
readfirstlanes derived `&Bs[1][n].data[0]` addresses for the same
reason.

Result: `grouped_var_k_kernel<0>` resource report is **byte-identical**
to baseline (94 SGPR / 256 VGPR / 0 scratch / 0 spill / Occ 2). The
SI Convergent Annotator was already promoting these — the explicit
annotation is redundant. Reverted; no source bloat shipped.

5-run metric: 883, 882, 884, 883, 883 → mean 883.0, std 0.7 (vs R89
baseline 882.6 ± 0.5). Within noise.

## Part C: Cumulative falsification streak (R83-R90)

| Round | Lever                                                                  | Verdict | Cause |
|-----:|------------------------------------------------------------------------|---------|-------|
|   83 | RCR FUSE=true KI=88 dual-A prefetch                                    | -3 | grouped<RCR,88,1> +9 spill drops Occ 2→1 |
|   84 | RCR FUSE=true KI=44 LDS pre-shuffle                                    | -7 | grouped<RCR,44,1> +28 spill drops Occ 2→1 |
|   85 | var-K KI=32/64 inner-loop unroll                                       | -4 | var_k +14-18 spill |
|   86 | st_32x16_v2 within-half swizzle (var_k + grouped CRR)                  | -5 | var_k +17 spill; grouped CRR +8 spill |
|   87 | (diagnostic / pivot only — PMC + plan)                                 | -  | (no code change) |
|   88 | st_32x16_v2 RRR ST_B (pad-only)                                        | flat | Build-gate passes; no LDS-BC reduction; +8-24 scratch on hot RRR KIs |
|   89 | (var_k helper extraction, profile-only — no metric ask)                | flat | Bit-identical production .so; profile data committed |
|   90 | k_offset_tiles/ki_g rematerialisation + LDS-ptr readfirstlane          | flat | Both levers are no-ops post-compiler analysis |

**6 perf rounds + 3 structural/diagnostic rounds**, all flat or
regressed. The pattern is now overdetermined: the compiler is using
all 256 VGPR effectively; the 50% LDS-BC bottleneck in var_k cannot
be attacked with any swizzle/permutation that fits inside the
existing register budget; and the helper-extraction / rematerialisation
levers don't free margin because every inlined value is body-wide live.

## Part D: R14 plan — pivot to PMC of the worst shape, then attack a NEW lever

The R10/R87 PMC was on `gpt_oss-Down-B4-M2048` (ratio 1.060 currently,
not the lowest-progress shape today). Today's worst shape is
**`gpt_oss-Down-B32-M2048`** at ratio **1.044** (8× more groups, but
same N=2880 K=2880 K-tail family). Before pursuing more VGPR-budget
levers, R14 should re-PMC the actual worst-progress shape and check:

1. Does the per-kernel wall split look the same as B4-M2048? Or does
   the fwd path dominate more / less at B32?
2. Is var_k still 50% LDS BC at B=32? (Larger persistent loops →
   more cache pressure on `s_offs`?)
3. What's the kernel mix? Does B32 hit different KI specializations
   in `grouped_kernel<RRR/RCR, KI, FUSE>`?

If the answer is "same pattern" → confirm var_k is the unblock target;
the next attack lever should be a **pure scheduling change** (no VGPR
add) — for example, moving the s_barrier in the persistent loop's
epilog to overlap with the next iteration's prologue HBM load.

If B32 looks DIFFERENT (e.g. fwd-dominated), pivot to a fwd-side lever
(K-tail HBM dual-buffer, RRR ceil_div extension to KI=64 specialization
already shipped in HK commit 757360a8).

## Files touched

* HipKittens repo: working tree clean (analytical-only round, source
  reverted post-experiment).
* Primus-Turbo repo:
  `analysis/_notes/round-90-bf16-grouped-var-k-r13-rematerialisation-falsified.md`
  (this file).
* `/tmp/build_round13.log` (R13 readfirstlane build, baseline-identical
  resource report — bit-confirmed redundant).

## Metric / numbers

* R89 baseline (5 runs):       882, 882, 883, 883, 883 → mean 882.6, std 0.5.
* R13 readfirstlane (5 runs):  883, 882, 884, 883, 883 → mean 883.0, std 0.7.
* Δ-mean = +0.4 (within 1 σ).
* var_k_kernel<0> resource report bit-identical to baseline:
  - 94 SGPR / 256 VGPR / 0 scratch / 0 SGPR spill / 0 VGPR spill / Occ 2.
* All 24 metric shapes PASS correctness (verified at R89 baseline; no
  source change shipped this round).

## Recommendation for round 14

1. Run rocprofv3 PMC on `gpt_oss-Down-B32-M2048` (the actual lowest-
   progress shape today, not R87's B4-M2048). Reuse
   `/tmp/rocprof_round10_fwd_bwd.py` with `B=32, M=2048` substitution.
   Capture fwd/dA/transpose/dB wall split + LDS BC% per kernel.
2. Compare vs R87 PMC (B4-M2048). If similar split, **var_k is still
   the unblock target** but VGPR-budget levers are exhausted; consider
   schedule-only changes (e.g., main_loop_iter sched_barrier density
   adjustment, similar to HK commit 2bdfca7a/237ca6b1 that yielded +5
   to +9 score with 0 VGPR cost).
3. If B32 is fwd-dominated instead, pivot to forward K-tail HBM dual-
   buffer (Lever A1) which was previously gated on var_k VGPR margin
   — the gate was R89's 8-12 VGPR target, now confirmed unreachable.
