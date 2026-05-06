# Round 86 — bf16 grouped GEMM weighted wall

> **Context:** auto_optimize round 9 / 100, MI355X. Continuation of
> R85's plan: prototype the custom `st_32x16_s` swizzle that breaks
> the within-half 4-way bank-cycle alias identified by R85's PMC walk.

**Status:** R86 `st_32x16_v2` peer shape with within-half XOR
**FALSIFIED** (-5 score, var-K kernel grew 17 VGPR spills + CRR grouped
grew 8 VGPR spills, swallowing the LDS-bank-conflict reduction).
This makes the 4th consecutive round (R83-R86) where a register-state-
adding optimization on the BF16 grouped/var-K bodies got swallowed by
VGPR spills.

| run                                                            | weighted score | gpt_oss geomean |
|----------------------------------------------------------------|---------------:|-----------------|
| baseline (R85 commit 5675d8b)                                  | 882-886 (±3 noise) | 1.093-1.096 |
| R86 prototype 1: directly modify `st_32x16` swizzle            | crash (dA/dB max_abs ≈ 280-414)| n/a (correctness FAIL) |
| R86 prototype 2: peer shape `st_32x16_v2`, generic next_addr   | 879            | 1.0876 (-0.87 pp) |
| post-revert (HK + Primus, file-clean)                          | 882            | 1.093           |

`st_32x16_v2` peer shape on grouped/var-K paths: 17 / 15 VGPR spills, MfmaU
unmeasured (metric regressed without further investigation). **REVERT.**

## Part A: The lever — within-half 4-way bank-cycle alias breaker

Per R85 PMC walk, `st_32x16_s` emits ~1 LDS bank conflict per LDS
instruction on the bf16 32x16-tile stride-128 access pattern (lanes
0/8/16/24 hit offsets 0/128/256/384 → all bank-cycle 0). The current
swizzle `((offset % 1024) >> 9) << 4` only XORs offset bit 9 (= row
bit 4) into bit 4 (= column bit 3), which breaks the upper-half ↔
lower-half alias but leaves the within-half 4-way alias intact.

Lever: layer in a second XOR that maps offset bits 7-8 (= row bits
2-3) into bits 4-5 (= column bit 3 + row bit 0). After the new
swizzle, lanes 0/8/16/24 land at offsets 0/144/288/432 → banks
0/4/8/12 — distinct cycles, no within-half conflict.

Math verification (involution check):
* B'(r, c) = B(r, c) XOR ((r bit 4 XOR r bit 2) << 4) XOR ((r bit 3) << 5)
* Decomposition: r' = r XOR ((r bit 3) << 0), c' = c XOR ((r bit 4 XOR r bit 2) << 3)
* Apply swizzle again: r'' = r' XOR (r' bit 3) << 0 = r ✓, c'' = c' XOR (...) = c ✓

So the swizzle is a coordinate-level involution → `prefill_swizzled_offsets`
round-trip stays correct (logic at
`include/ops/warp/memory/tile/global_to_shared.cuh:147-152`).

## Part B: Prototype 1 — directly modify `st_32x16::swizzle()`

First attempt: just XOR the new bits in the existing struct's swizzle
function. Build clean, but the correctness probe on 6 shapes
(downsized BF16 fwd+bwd vs Triton) showed:

| shape                                          | out max_abs | dA max_abs | dB max_abs |
|------------------------------------------------|-------------|------------|------------|
| BF16_gpt_oss_20B-Down-B4-M2048_DOWN            | 2.0  ✓      | 2.0  ✓     | 96.25 ✗    |
| BF16_gpt_oss_20B-GateUP-B4-M2048_DOWN          | 2.0  ✓      | 372.0 ✗    | 88.75 ✗    |
| BF16_DeepSeek-V3-Down-B16-M2048_DOWN           | 1.0  ✓      | 414.0 ✗    | 88.25 ✗    |
| BF16_DeepSeek-V3-GateUP-B16-M2048_DOWN         | 2.0  ✓      | 334.0 ✗    | 99.0  ✗    |
| BF16_Qwen3-235B-A22B-Down-B16-M2048_DOWN       | 1.0  ✓      | 340.0 ✗    | 85.75 ✗    |
| BF16_Qwen3-235B-A22B-GateUP-B16-M2048_DOWN     | 2.0  ✓      | 280.5 ✗    | 89.5  ✗    |

Forward (RCR layout, ST_A=ST_B=`st_16x32_s`) was always clean — the
swizzle change doesn't affect it. dA / dB (which use `st_32x16_s` via
RRR / CRR layouts) failed catastrophically.

**Root cause:** `shared_to_register.cuh` has TWO `st_32x16_s`-specific
fast paths:

1. Line 254-262: row_l ds_read_b64 pair (uses `addr` and `next_addr`,
   actually correct under any swizzle since both are computed via
   `src.swizzle()`).
2. **Line 663-672: col_l ds_read_b64_tr_b16 pair** with `addr` and
   IMMEDIATE `offset + 4 * underlying_subtile_row_bytes` (i.e.,
   hardcoded "row+4 is at +128 bytes in LDS"):

   ```cpp
   asm volatile(
       "ds_read_b64_tr_b16 %0, %2 offset:%3\n"
       "ds_read_b64_tr_b16 %1, %2 offset:%4\n"
       : ...
       : "v"(addr), "i"(offset), "i"(offset + 4 * ST::underlying_subtile_row_bytes)
       : "memory"
   );
   ```

   The "+128" assumption only holds when `swizzle(r+4, c) - swizzle(r, c)`
   is a constant. Under the OLD swizzle, the upper-half-row XOR
   (bit 9 → bit 4) is consistent for r and r+4 in the col_l lane
   mapping (row_offset ∈ {0..3, 8..11, 16..19, 24..27}; row+4 ∈
   {4..7, 12..15, 20..23, 28..31} — both halves of each pair share
   the same `r >= 16` bit). Under the NEW swizzle, the within-half
   XOR (bit 7 → bit 4) toggles whenever r bit 2 changes, which DOES
   happen for every row vs row+4 pair (e.g., r=0 has bit 7 = 0, r+4=4
   has bit 7 = 1). The hardcoded `+128` lands at the wrong physical
   LDS slot for these pairs ⇒ the second ds_read_b64_tr_b16 pulls
   the wrong bytes ⇒ dA / dB are corrupted.

## Part C: Prototype 2 — peer shape `st_32x16_v2_s`

Spawned a peer struct `st_32x16_v2` in `include/types/shared/st_shape.cuh`
+ alias `st_32x16_v2_s` in `include/types/types.cuh` + concept-list
addition. Wired only the BF16 grouped CRR ST_A / RRR ST_B / var-K
ST_A / ST_B paths (`kernel_bf16_dynamic.cpp:3731-3736 + 4768-4769`)
through the new shape. Dense `gemm_kernel` left on the original
`st_32x16_s` to preserve the b64-pair-immediate fast path on
DoD-smoke-covered dense BF16 backward.

The new shape doesn't match the special-case templates at
`shared_to_register.cuh:254` (`std::is_same_v<ST::shape, st_32x16_s>`)
or 663 (same), so it falls through to the generic
`ds_read_b64_tr_b16 (addr, next_addr)` path that re-issues
`src.swizzle({row + 4, col})` for `next_addr`. Correctness is
preserved — verified on the same 6-shape probe:

| shape                                      | out | dA  | dB  |
|--------------------------------------------|-----|-----|-----|
| BF16_gpt_oss_20B-Down-B4-M2048_DOWN        | ✓   | ✓   | ✓ (max=0.5) |
| BF16_gpt_oss_20B-GateUP-B4-M2048_DOWN      | ✓   | ✓   | ✓   |
| BF16_DeepSeek-V3-Down-B16-M2048_DOWN       | ✓   | ✓   | ✓   |
| BF16_DeepSeek-V3-GateUP-B16-M2048_DOWN     | ✓   | ✓   | ✓   |
| BF16_Qwen3-235B-A22B-Down-B16-M2048_DOWN   | ✓   | ✓   | ✓   |
| BF16_Qwen3-235B-A22B-GateUP-B16-M2048_DOWN | ✓   | ✓   | ✓   |

**Build report (kernel_bf16_dynamic.cpp resource usage delta):**

| kernel                              | baseline VGPR spill | v2 VGPR spill | Δ spill |
|-------------------------------------|---------------------|---------------|---------|
| `grouped_kernel<RCR, 832, FUSE=0>`  | 1                   | 1             | 0 (RCR uses st_16x32, untouched) |
| `grouped_kernel<RRR, 832, FUSE=0>`  | 4                   | 6             | +2      |
| `grouped_kernel<CRR, 832, FUSE=0>`  | 7                   | **15** (+SGPR 12) | **+8** |
| `grouped_var_k_kernel<0>`           | 0                   | **17**        | **+17** |

The CRR `grouped_kernel` and `grouped_var_k_kernel` (both use
`st_32x16` for ST_A AND ST_B → 2× the per-iter LDS reads × 2 register
addresses for next_addr) take the brunt of the cost.

**Why the spill increase:** the generic `next_addr` path needs an extra
register per concurrent LDS access vs the fast path's single-`addr` +
immediate-offset format. With the unrolled K-loop in the var-K kernel,
the compiler can't keep all (`addr`, `next_addr`) pairs live within
the 256-VGPR per-thread budget, forcing 17 VGPRs of scratch traffic
per iteration.

**Metric:** 879 (vs baseline 882-886). gpt_oss geomean dropped from
1.093-1.096 to 1.0876 (-0.87 pp on the worst-case family); all 8
gpt_oss shapes regressed (Down by -0.001 to -0.015, GateUP by -0.007
to -0.010). DSV3 / Qwen3 mostly flat (some +0.005, some -0.005). Net
weighted score: -5.

## Part D: 4-round VGPR-pressure pattern is now decisive

| round | lever                              | spill | metric | direction |
|-------|------------------------------------|-------|--------|-----------|
| R83   | RCR FUSE KI=88 (dead — never hit)  |  9    | flat   | -         |
| R84   | RCR FUSE KI=44                     | 28    | -40    | regressed |
| R85   | var-K KI=32 / KI=64                | 14-18 | -30    | regressed |
| R86   | st_32x16_v2 (within-half swizzle)  |  6-17 |  -5    | regressed |

The pattern: any single-round optimization on the BF16 grouped /
var-K kernels that adds register state (full unroll for KI specs;
extra `next_addr` register for swizzle path; live-state for FUSE
epilog) gets dominated by spill traffic. The 256 VGPR ceiling is
the binding budget; the kernel bodies' baseline VGPR pressure leaves
≤8 VGPRs of headroom before scratch traffic costs more than the
optimization saves.

## Part E: revert + clean state

Both HK kernel and Primus changes fully reverted; `git diff` clean
on both files. Final 1-run metric post-revert: **882** (back at
baseline within noise). HK kernel diff vs R85 commit: 0 lines (only
the NFS .nfs* lock files remain). var-K kernel back to 0 VGPR spill.

## Part F: direction for R87+

Both viable BF16 grouped levers identified at the end of R85 are now
known to require **multi-round VGPR-pressure-trimming PROLOGUE work**
before they can land:

1. **Custom st_32x16 swizzle (R86 falsified):** the cost was the
   variable `next_addr`. Two paths to make it land:
   * (a) Modify `shared_to_register.cuh` to support `st_32x16_v2_s`
     in the existing fast paths by deriving the `+128 → +144` /
     `+128 → +96` immediate-offset variation from the swizzle's
     bit pattern at compile time. Since the variation depends on
     a row-bit of the call-site coord (which is laneid-dependent),
     this would need either an extra immediate offset selected at
     runtime per lane (a vsel between two ds_read_b64_tr_b16 with
     different immediates — cheap but still adds 1 VGPR of variation
     state) OR a change to the swizzle to make the variation
     orthogonal to the inner-loop coord (e.g., XOR based on a
     thread-block-uniform parameter rather than per-lane row bits).
     1-2 round redesign.
   * (b) Find a different swizzle that satisfies BOTH (i) breaks the
     within-half 4-way alias AND (ii) preserves
     `swizzle(r+4, c) − swizzle(r, c) = constant`. Math: (ii)
     constrains the XOR amount to be invariant under r += 4, i.e.,
     XOR can only depend on (r mod 4, c, ...) or on r-block-uniform
     bits. Lanes 0/8/16/24 hitting bank cycle 0 all have r mod 4 = 0
     → same XOR amount → can't distinguish. So (i) and (ii) are
     mutually exclusive on this lane mapping. Falsified by
     construction.

2. **VGPR live-range compression in the var-K body** (R85 plan B):
   trim the baseline 256-VGPR footprint to make room for the
   `next_addr` register OR for a future KI spec. Per-pass register
   usage profile + hot-path coord / group-bookkeeping live-range
   audit is a 2-3 round prologue.

3. **A different conflict-reduction lever entirely** (new for R87):
   modify the var-K LDS access pattern itself rather than the
   storage layout. E.g., interleave the two MFMA C-tile halves
   (col_l rt_32x16) such that lanes 0/8/16/24 don't all hit bank
   cycle 0 simultaneously — a lane permutation in the inner-loop
   `row_offset` / `col_offset` derivation. Would need to re-derive
   the lane → MFMA-input mapping; might be a single-round lever if
   the permutation can be expressed as a one-line tweak in
   `shared_to_register.cuh:322-323` for the col_l rt_32x16_s case
   without breaking the other col_l shapes (rt_16x16_s / rt_8x32_s).

R87 candidate: **profile var-K's per-pass VGPR live-range** with a
`-Rpass-analysis=kernel-resource-usage` + careful spill-source
attribution (which SGPR / VGPR is causing the high-water mark, by
splitting the kernel body into named sub-passes via attribute
markers). Multi-round prologue toward both Lever A1 / A2 above; if
the profile points at a clear hot-path coord that can be
recomputed-instead-of-stored, that's a single-round trim worth
trying immediately.

## Files touched

* `/workspace/code/HipKittens/include/types/shared/st_shape.cuh`
  — added `struct st_32x16_v2` peer shape, added to concept all.
  **REVERTED**.
* `/workspace/code/HipKittens/include/types/types.cuh`
  — added `using st_32x16_v2_s` alias. **REVERTED**.
* `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  — wired ST_A (CRR) / ST_B (RRR or CRR) of `grouped_kernel` and
  `grouped_var_k_kernel` to use `st_32x16_v2_s`. **REVERTED**.
* This round note (Primus-Turbo).
* `/tmp/r9_correctness_probe.py` — 6-shape downsized correctness
  probe (kept offline, not committed).
* `/tmp/metric_round_9.log`, `/tmp/metric_round_9_v2.log` — metric
  artifacts (offline, not committed).
