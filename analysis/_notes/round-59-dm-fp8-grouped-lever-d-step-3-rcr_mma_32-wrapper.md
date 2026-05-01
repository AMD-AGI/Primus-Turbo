# Round 59-dm — FP8 grouped: Lever D R-B step 3 — `rcr_mma_32` wrapper + force-instantiate

**Status**: INFRASTRUCTURE-only commit / no metric impact expected
**Score before**: 958 (R31 baseline)
**Score after**:  959 (within 947-963 plateau noise)
**HK SHA**: `75e30a5f` → `addaf23e`
**PT SHA**: this commit
**Round time**: ~14 min, 1 build cycle, 2 metric runs
**Auto-optimize round**: 31

---

## TL;DR

1. Added `rcr_mma_32` wrapper in `kernel_fp8_layouts.cpp` (RBM=64, RBN=32
   accumulator + rt_32x64_s A and B operands).
2. Added a force-instantiate stub (`__attribute__((used))` +
   `[[maybe_unused]]`) so the build VALIDATES the dispatch resolves
   to `mfma_323264` AT COMPILE TIME instead of waiting for R32+.
3. Build verified: grouped_rcr_kernel spill UNCHANGED (39/43 dw).
4. Metric 958 → 959 (within plateau).
5. R31 was a 16-min chat-window slot; full Lever D K-tail port is too
   large to fit in a single round. This commit is the smallest useful
   step that creates persistent infrastructure for R32+.

---

## Discovered while implementing — why `rt_32x64_s` for B (NOT `rt_64x32_s`)

When wiring `rcr_mma_32`, I hit a layout decision: for RCR
(A row × B row → MFMA computes C = A·Bᵀ via `mma_ABt`), what shape
does B operand take?

Looking at `mma_ABt_base` in `mma.cuh:202-241`:

```cpp
template<…> static inline void mma_ABt_base(
    rt_base<float,    col,  D_shape>& d,
    const rt_base<MM_T, row, A_shape>& a,
    const rt_base<MM_T, row, B_shape>& b,  // in row-major mode
    const rt_base<float,    col,  C_shape>& c)
{
  // …
  } else if constexpr (D_shape == rt_32x32 &&
                A_rows == 32 && A_cols == 64 &&
                B_rows == 32 && B_cols == 64 &&  // ← BOTH 32×64
                C_shape == rt_32x32) {
      mfma323264(d.data, a.data, b.data, c.data);
  }
}
```

Both A and B are `rt_32x64` (NOT `rt_64x32`).

**Why?** The mfma_ABt operation already does an *implicit transpose* on
B during the contraction. So B is provided in the same row-major
layout as A — it's the K-axis of B that gets reduced against A's K-axis.
Hardware lane partition for B in `mfma_323264_f8f6f4` (per AMD CDNA4 ISA):
each lane provides `B[l mod 32, k=(l div 32)*32 + 0..31]` — same as A's
lane mapping but indexing the N-dim instead of M-dim.

**`rt_64x32_s` was the wrong alias for RCR.** Where `rt_64x32_s` is
needed: CRR/CCR layouts (`mma_AB_base` with B col-major). Search
`mma_AB_base` for `rt_32x32, A 32x64, B 64x32` (mma.cuh:172-185 — added
in HK SHA c2abba21 / R57-dm).

This was a useful clarification — `rt_64x32_s` infrastructure I added
in R29 is for the `mma_AB`/CRR/CCR family, NOT for K-tail port. R32's
K-tail port uses ONLY `rt_32x64_s` (for both A and B).

## Updated build infrastructure

### From R30-R31 cumulative

| Component | Where | Status |
|--|--|--|
| `rt_32x64_s` / `rt_64x32_s` aliases | `types.cuh` (HK c2abba21) | ✓ R29 |
| static_assert namespace `lever_d_round_b_step1_compile_test` | `kernel_fp8_layouts.cpp` | ✓ R30 (75e30a5f) |
| `rcr_mma_32` wrapper | `kernel_fp8_layouts.cpp:241+` | ✓ R31 (addaf23e) |
| Force-instantiate stub | `kernel_fp8_layouts.cpp:249+` | ✓ R31 (addaf23e) |
| K-tail rt_32x64 loader | TODO | R32 |
| K-tail block wiring | TODO | R32-R33 |
| LDS roundtrip merge | TODO | R32-R33 |
| Correctness verify | TODO | R33 |
| Metric verify | TODO | R33 |

### What `rcr_mma_32`'s force-instantiate validates

The stub at `__lever_d_round_b_force_instantiate_rcr_mma_32`:

```cpp
__attribute__((used)) [[maybe_unused]] static __device__ void
__lever_d_round_b_force_instantiate_rcr_mma_32() {
    rt_fl<RBM, RBN, col_l, rt_32x32_s> dummy_acc{};
    rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s> dummy_a{};
    rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s> dummy_b{};
    rcr_mma_32(dummy_acc, dummy_a, dummy_b);
}
```

`__attribute__((used))` keeps the symbol in the binary even though it's
never called from any kernel; this forces clang to type-check ALL the
way through the dispatch:

1. `rcr_mma_32(acc, a, b)` — wrapper signature accepts 3 register tile
   types — type-check ✓
2. `mma_ABt(acc, a, b, acc)` — wrapper body — looks up template
   overload — instantiates `mma_ABt_base` for the 32x32x64 fp8 RCR
   tuple — type-check ✓
3. `mma_ABt_base` dispatch — selects `else if` branch at mma.cuh:234,
   reaches `mfma323264(d.data, a.data, b.data, c.data)` — intrinsic
   call — type-check ✓
4. `mfma323264` definition — eventually `__builtin_amdgcn_mfma_…` —
   ABI-level argument check — ✓ if all dword sizes match.

Build pass = full pipeline confirmed working.

## Side observation — `rcr_mma_32` is unreferenced but emits a symbol

The stub emits a symbol `_Z53__lever_d_round_b_force_instantiate_rcr_mma_32v`
into the .so. This adds a few hundred bytes to the binary but is
unreachable. Confirmed via `objdump -d`:

* Spill profile UNCHANGED for grouped_rcr_kernel (39/43 dw).
* No template instantiation pollution of cache.
* Stub is dead code from runtime perspective.

## Recommendation for R32

R32 should:

1. **Read the existing K-tail loader** at `kernel_fp8_layouts.cpp:2455-2501`
   (the `load_a_kt` / `load_b_kt` lambdas for rt_16x128).
2. **Adapt for rt_32x64**:
   - `row_lane = laneid % 16` → `row_lane = laneid % 32` (rows 0-31)
   - `k_lane_byte = (laneid / 16) * 32` → `k_lane_byte = (laneid / 32) * 32`
     (each chunk of 32 lanes covers a 32-K span)
   - For K_REM=64: lane 0..31 covers K=[0..32), lane 32..63 covers K=[32..64).
     ALL LANES VALID (no SENTINEL needed) — vs current rt_16x128 path
     which has lanes 32..63 SENTINEL because K=[64..128) is K-OOB.
3. **Write a new `rcr_mma_32`-based K-tail block** that:
   - Computes per cAB_32 tile (4 of them per warp, RBM=64, RBN=32)
   - After all 4 mfmas, run LDS roundtrip merge: ds_write cAB_32 to
     LDS in 32x32 layout, s_barrier, ds_read into cA-cD as 16x16
     additive accumulate. ~16 cy per cAB_32 (8 ds_write + 8 ds_read
     b128, plus 1 barrier amortized = ~64 cy + 10 cy = ~74 cy).
4. **Validate correctness** (allclose vs the existing 16x16x128 K-tail
   path on gpt_oss B16/B32 cases). SNR ≥ 25 dB target.
5. **Measure metric**. Compare 16x16x128 vs 32x32x64 K-tail.

Estimated R32 effort: 30-60 min focused work (one full round, possibly
two if correctness debugging is needed).

Risk factors:
- LLVM register allocator may not keep cAB_32 dead between iterations;
  if it stays alive in batched mode, peak hits 320 dw and spill increases.
- LDS bank conflict: if rt_32x32 → rt_16x16 layouts have
  different bank distributions, the ds_write may serialize.

## Round summary

| Item | Value |
|--|--|
| Goal | Add Lever D R-B `rcr_mma_32` wrapper + force-instantiate type-check |
| Change | 36 lines added to `kernel_fp8_layouts.cpp` (HK SHA `addaf23e`) |
| Lines added | 36 (HK) + this note (PT) |
| Spill profile change | NONE (39/43 dw — UNCHANGED) |
| Metric before | 958 |
| Metric after | 959 (plateau noise — change is compile-time only) |
| HK commit | `addaf23e` |
| Side discovery | `rt_64x32_s` is for CRR/CCR (`mma_AB`), `rt_32x64_s` is for RCR (`mma_ABt`) — both — A and B use same shape in mma_ABt 32x32x64 |
| Next round suggestion | R32 — wire K-tail loader + block + LDS merge using `rcr_mma_32` |
