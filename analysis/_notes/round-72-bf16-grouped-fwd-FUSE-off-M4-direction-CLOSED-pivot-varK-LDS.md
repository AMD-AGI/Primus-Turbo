# Round 72 — BF16 grouped fwd FUSE-off + M4 K-tail direction CLOSED; pivot to R68 Priority 1 (var-K CRR LDS swizzle)

**Status:** CLOSED (diagnostic + direction decision, no code change).
**Score:** baseline 876 (unchanged).

## Summary of R69-R71 investigation arc on M4 K-tail direction

| Round | Hypothesis | Result | Falsifier |
|---|---|---|---|
| R69 | FUSE-off for g.ki<48 routes fwd to non-fuse main + M4 → recovers 11 pp MfmaUtil gap | 8/8 gpt_oss fwd-allclose FAIL | M4 kernel dead code for fwd globals, latent bug |
| R70 | var-K chiplet chunk_size sweep (16/32/64/128) | +3 score (in noise) | Baseline chunk_size=64 near-optimal for var-K dispatch |
| R71 | M4 bug = uninit C memory (main skips some cells in [0,M_total)×[0,g.n)) → use torch.zeros init | Same 8/8 FAIL | Main kernel's store_c_tile_n_masked (L303-363) DOES cover every cell via predicated per-lane path |
| R72 | Pinpoint M4 axis/lane bug | **Closed — M4 is dead code, not worth fixing** | See analysis below |

## R72 analysis — why M4 bug-fix is low-value

### 1. M4 kernel path trace (review)

Dispatch site: `kernel_bf16_dynamic.cpp:4524` calls
`grouped_ktail_kernel_mfma32x32_M4<Layout::RCR, 64>` ONLY when:
- `need_tail_run && !fuse_ktail_eligible` (non-fuse path)
- `L == Layout::RCR`
- `K_rem == 64` (i.e., `g.k % K_STEP == K_STEP` → **g.k % 128 == 64**)
- `lds_k_tail_safe` (m_per_group >= TAIL_BLOCK_M && % TAIL_BLOCK_M == 0)
- `mfma32_m4_safe` (m_per_group >= 128 && % 128 == 0)

### 2. When does metric hit this path?

**Fwd (forward-pass globals, trans_b=True, L=RCR):**
- `fast_k = (g.k / 128) * 128`, `K_rem = g.k - fast_k`
- gpt_oss K=2880: `fast_k = 22*128 = 2816`, `K_rem = 64` ✓
- DSV3 K ∈ {4096, 7168}: K%128 = 0, `K_rem = 0` → `need_tail_run = false` → M4 SKIPPED
- Qwen3 K=4096: same, `K_rem = 0` → M4 SKIPPED
- gpt_oss DOES want K_rem=64 path — BUT `fuse_ktail_eligible = true` for g.ki=44>=2
  → FUSE path taken → **M4 never called in fwd**

**dA backward (after H4 reroute, L=RCR, MFMA_K = N_fwd):**
- gpt_oss N_fwd=2880: K_rem = 2880%128 = 64 ✓, g.ki=44>=2 → FUSE path → **M4 skipped**
- DSV3 N_fwd ∈ {2048, 4096, 7168}: all %128==0 → K_rem=0 → M4 SKIPPED
- Qwen3 N_fwd ∈ {1536, 2880, 4096, 5760}: 1536%128=0, 2880%128=64, 4096%128=0, 5760%128=0
  - Qwen3 2880 → K_rem=64 ✓ but g.ki=22>=2 → FUSE → M4 SKIPPED

**Net: M4 kernel is NEVER dispatched in the 24-shape baseline metric.** It
is compiled-but-unused code; the R69 experiment was the first real
invocation attempt since its R54 commit.

### 3. Latent bug hypothesis candidates (eliminated H1, deferred H2*)

- **H1 (uninit C):** ELIMINATED by R71 (torch.zeros init did not fix).
- **H2a (lane→row/col wrong for fwd globals):** Unlikely —
  `mfma_f32_32x32x16_bf16` lane layout is layout-agnostic. Fwd/dA differ
  only in the *semantic* meaning of A/B/C, not the raw byte layout that
  the mfma instruction consumes.
- **H2b (A-side SRD wrong):** Unlikely — `coord<>(row, k)` on `g.a` uses
  the same row-major stride for fwd and dA.
- **H2c (= vs +=):** ELIMINATED — L3289-3291 reads bf16 existing, adds
  `acc[i]`, writes bf16. Correct RMW polarity.

The actual bug is likely in the M4 **dispatcher-global interaction** or a
compiler codegen issue specific to fwd `g` construction that doesn't
manifest when the kernel was tested (R54) against dA-shaped globals.
Root-causing requires a full device-side printf or SASS-level diff of the
kernel's first instructions under fwd vs dA g inputs, which is a 2-4 hour
investigation for at most +4-8 score potential (the 11 pp MfmaUtil gap in
FUSE translates to ~4-8% fwd speedup if entirely recovered).

### 4. Opportunity-cost rationale for CLOSING

- **Best-case ROI:** M4 fix + R69 gate → gpt_oss fwd +5-8% → score +8-12.
- **Risk:** Latent fwd correctness regression on the now-live M4 path; M4 for
  dA path (untested) could also regress if the fix applies globally.
- **Time cost:** 2-4 rounds of SASS-level debug.
- **Alternative levers with higher ROI:**
  - R68 Priority 1 — var-K CRR LDS swizzle (unpadded `st_32x16_s` →
    padded `st_64x32_padded_b128_s`). 217M LDS bank conflicts observed →
    potential dB var-K speedup 10-15% → affects ALL 24 shapes
    (backward dB is in every shape) → score +15-25 potential.
  - B1 (MFMA pipeline scheduling) — FUSE MfmaUtil 61% → target 80%+
    via prefetch/reorder in `kernel_bf16_dynamic.cpp:808-946` K-tail
    epilog. Affects gpt_oss fwd + dA (after H4) which is ~60% of the
    weighted-score headroom.
  - B2 (register pressure / __launch_bounds__) — DSV3 K=7168 / Qwen3
    K=4096 are large enough K-loops that occupancy tuning may pay off.

## R73 direction recommendation

**Priority 1 (R68 main lever):** Change `grouped_var_k_kernel`'s LDS tile
shape from `st_32x16_s` to a padded variant. Candidate:
`st_64x32_padded_b128_s` (same shape that RCR uses with 0 conflicts).
Requires:

1. Update `ST_A` / `ST_B` definitions in `grouped_var_k_kernel` (~line 4700)
2. Verify `shared_to_register.cuh` has a working `rt_16x32` / `rt_32x16`
   load path for the padded variant (R69 note flagged Path B for
   `st_64x32_padded_b128_s` as "compiles only, not semantically verified")
3. If Path B is not verified, stage via Path A (HBM→LDS→register) first,
   accept the extra LDS bounce, re-measure bank conflicts.

Expected:
- LDS bank conflicts 217M → <50M
- dB var-K wall ~1.05x-1.1x faster
- Score +10-15 (dB var-K is ~34% of bwd wall × 24 shapes weighted)

**Priority 2 (B1 FUSE pipeline):** Prefetch slab-1 A-load during slab-0
MMAs in `kernel_bf16_dynamic.cpp:808-946`. Requires adding a second
A-tile register or restructuring the `vmcnt(0)` barrier. Risk: VGPR
spill on the already-tight FUSE path (KI=44 is close to the spill
threshold per R67).

**Priority 3 (DoD smoke):** Next DoD checkpoint at R75. If DoD 608
degrades, pause and triage.

## No code change this round

HK + Primus working trees clean. Baseline 876 verified.
