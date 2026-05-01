# Round-35-dm · FP8 grouped — collapsing `<0,*,true>` fused spec into single `N_MASKED_STORE=true` variant FALSIFIED by runtime branch overhead on DSV3

**Status**: FALSIFIED. Score 935 → 932 (−3 pts). Correctness all PASS.
Reverted; no HipKittens commit this round.

## Hypothesis (derived from Round-34-dm ISA tooling)

Post-R34, the FP8 grouped dispatcher has 4 template instances in the cache:

| Spec (`<PID, NMASK, FUSED>`) | Interleaved spills | Total scratch | Current consumers (post-R34) |
|---|---|---|---|
| `<0, false, false>` | 62 | 741 | (none — K_REM ≠ 0 AND ≠ 64 has no MoE shapes) |
| `<0, true,  false>` | 44 | 661 | (none — same, vestigial) |
| `<0, false, true>`  | **28** | **559** | **8× DSV3 (N-aligned, K_REM=0)** |
| `<0, true,  true>`  | **22** | **444** | **8× gpt_oss (N-masked, K_REM=64)** |

Both `<0, *, true>` specs have the extra `A_row_reg a_kt1;` declaration (from
R34 fuse_ktail win). But `<0, true, true>` has an additional codegen
advantage: **20% fewer TOTAL scratch ops AND fewer interleaved
scratch/mfma pairs** than `<0, false, true>`.

This suggests LLVM's register allocator takes a **fundamentally better path**
when the N-masked helper body is in scope. The runtime branch
`if ((bc + 1) * BLOCK_SIZE <= g.n)` inside `if constexpr (N_MASKED_STORE)`
takes the fast-path `store(...)` for N-aligned shapes (byte-identical to
the `!N_MASKED_STORE` codepath).

**Predicted win:** +2-3 pp per DSV3 shape (22 vs 28 interleaved ≈ 21% fewer
hot-path stalls).

## Change applied

```cpp
// Before (R34 state):
if (fuse_ktail_eligible) {
    if (n_aligned) {
        grouped_rcr_kernel<0, false, true><<<...>>>(g);   // DSV3
    } else {
        grouped_rcr_kernel<0, true , true><<<...>>>(g);   // gpt_oss
    }
}

// After (R35 probe — REVERTED):
if (fuse_ktail_eligible) {
    grouped_rcr_kernel<0, true , true><<<...>>>(g);       // ALL fused shapes
}
```

## Observation

| Shape | R34 ratio | R35 ratio | Δ pp | Notes |
|---|---|---|---|---|
| DSV3-GateUP-B16-M2048 | 1.090 | 1.083 | −0.7 | |
| DSV3-Down-B16-M2048 | 1.089 | **1.055** | **−3.4** | regress |
| DSV3-GateUP-B16-M4096 | 1.128 | 1.132 | +0.4 | |
| DSV3-Down-B16-M4096 | 1.101 | **1.070** | **−3.1** | regress |
| DSV3-GateUP-B32-M2048 | 1.116 | 1.127 | +1.1 | |
| DSV3-Down-B32-M2048 | 1.133 | **1.104** | **−2.9** | regress |
| DSV3-GateUP-B32-M4096 | 1.132 | 1.121 | −1.1 | |
| DSV3-Down-B32-M4096 | 1.136 | **1.048** | **−8.8** | regress |
| gpt_oss-GateUP-B4-M2048 | 1.041 | 1.038 | −0.3 | (no spec change) |
| gpt_oss-Down-B4-M2048 | 1.085 | 1.111 | +2.6 | (no spec change, noise) |
| gpt_oss-GateUP-B4-M4096 | 0.999 | 1.003 | +0.4 | (no spec change) |
| gpt_oss-Down-B4-M4096 | 1.028 | 1.042 | +1.4 | (no spec change, noise) |
| gpt_oss-GateUP-B32-M2048 | 0.985 | 0.983 | −0.2 | (no spec change) |
| gpt_oss-Down-B32-M2048 | 1.015 | 1.026 | +1.1 | (no spec change, noise) |
| gpt_oss-GateUP-B32-M4096 | 0.971 | 0.972 | +0.1 | (no spec change) |
| gpt_oss-Down-B32-M4096 | 1.003 | 1.035 | +3.2 | (no spec change, noise) |
| **geomean grp_FP8** | **1.0642** | **1.0584** | **−0.58 pp** | |

**DSV3-Down family is the casualty**: all 4 shapes regressed by 2.9 to 8.8 pp.
DSV3-Down has N=7168 (largest N of any shape), meaning the most N-tiles per
CU, meaning the highest per-tile runtime-branch overhead accumulation.

## Post-mortem: why ISA interleaved-spill was necessary but not sufficient

The ISA proxy (interleaved scratch/mfma pairs) is derived from **static
assembly**. It correctly predicts **codegen register-pressure behavior**.
It does NOT model:

1. **Per-tile runtime-branch cost**: the `if ((bc + 1) * BLOCK_SIZE <= g.n)`
   check fires once per C-tile per persistent-kernel iteration. For
   DSV3-Down (N=7168 → bpc=28, so 28 C-tiles per row) × (many M-rows), the
   branch fires thousands of times per group. Even 1-2 cycles per fire
   times thousands of fires matters more than the ~6 saved interleaved-spill
   pairs in the epilog.
2. **Dead-code instruction-cache pollution**: spec `<0, true, true>` has the
   `store_c_tile_n_masked` helper body compiled into the ELF text section.
   For N-aligned runs, that code is cold but still evicts other code from
   L1i, potentially causing extra icache misses in the main loop on
   subsequent tile-group iterations.
3. **LLVM's allocation changes non-linearly with shape geometry**: the
   "better" spec `<0, true, true>` may only be better WHEN the N-masked code
   path is actually hot (gpt_oss shapes). For N-aligned DSV3 workloads,
   the allocator's choices are optimized for a runtime distribution that
   doesn't match the actual call pattern.

## What this means going forward

The **interleaved-spill metric is still useful but requires pairing
with runtime-branch-cost estimation** before acting on it. A better proxy:

```
perceived_cost = interleaved_spill_count * spill_latency
               + runtime_branch_count_per_tile * ~1.5cycles
               + icache_pressure_from_dead_code (hard to quantify)
```

For the **non-fused path** (`<0, *, false>`, 62 vs 44 interleaved), the
delta is much bigger (18 interleaved-spill saves) AND both n_aligned
and n_masked shapes currently bypass it (R34 put K_REM=0 AND K_REM=64
into fused). So this path is vestigial and not attackable.

## Remaining search space after R35 falsification

Still on the table:
- **Lever D** (piecewise 32×32×64 MFMA cell-shape for epilog 1 only) —
  ISA-unverified, but structural change that could reduce register
  tiles/MFMA for short-K shapes.
- **Explicit `asm volatile` lifetime fences** on accumulator VGPRs
  (cA/cB/cC/cD) to guide LLVM's live-range analysis away from the
  interleaved-spill pattern.
- **Main-loop software-pipelining in ASM** — high-risk final lever.
- **K-tail prologue optimization for K_REM=64** — gpt_oss pays 1 full K-tail
  iter per tile on top of main loop; if we can eliminate or overlap that
  iter, it's a direct +5-10 pp for all gpt_oss shapes. Needs careful
  correctness analysis (LDS layout, ds_write ordering).

Next round should pick the **K-tail overlap** lever: inspect whether the
K-tail iter can be overlapped with the prologue of the NEXT tile-group
(not within the same tile). The chain is:

```
prologue(g=0) → main(g=0) → K-tail(g=0) → epilog(g=0) →
prologue(g=1) → main(g=1) → K-tail(g=1) → epilog(g=1) → ...
```

If K-tail loads can be moved INTO prologue(g+1)'s load stages, the K-tail
MFMA-only code in the tile boundary gets shorter, saving wall time on
every tile-group transition — ~56 tile-groups per CU for gpt_oss.

Alternatively: continue ISA-guided codegen probes with a CO-DESIGNED
runtime-branch-cost model, so future predictions don't get blindsided by
per-tile overhead like this round did.
