# Round 61-dm — FP8 grouped: Lever D R-B step 4 — K-tail rt_32x64 loaders LANDED

**Status**: INFRASTRUCTURE-only commit / no metric impact expected
**Score before**: 960 (R34 baseline)
**Score after**:  958 (within plateau noise 947-963)
**HK SHA**: `addaf23e` → `78415fb0`
**PT SHA**: this commit
**Round time**: ~7 min, 1 build cycle (passed first try), 2 metric runs
**Auto-optimize round**: 34

---

## TL;DR

R32-R33 deferred this implementation due to chat window pressure.
R34 had enough budget (~7 min) to actually paste the loader skeleton
from R32's handoff doc into HK. Build passed first try, types
validated, no codegen impact.

This shifts the next-chat-session R35 task from "implement loaders +
K-tail block + LDS merge" to just "implement K-tail block + LDS merge"
— roughly 30-50% time reduction.

---

## What was committed

HK `78415fb0` — added 2 templates and extended force-instantiate stub:

```cpp
template<typename A_RT_32x64>
__device__ __forceinline__ void load_a_kt_32x64(
    A_RT_32x64& A_tile,
    i32x4 a_srsrc_kt,
    int M_warp_base,
    int row_lane,           // = laneid % 32
    int k_lane_byte,        // = (laneid / 32) * 32
    int a_row_stride_bytes,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    // 2 × buffer_load_b128 per lane per cell:
    //   data[0..3] = K=[k_lane_byte, k_lane_byte + 16) → b128 #1
    //   data[4..7] = K=[k_lane_byte + 16, k_lane_byte + 32) → b128 #2
    // …
}

template<typename B_RT_32x64>
__device__ __forceinline__ void load_b_kt_32x64(/* mirror of A */);
```

Force-instantiate stub now also calls these loaders with dummy operands
to validate types at HK build time.

## Build outcome

* First-try build pass — no template / type / dispatch errors.
* `grouped_rcr_kernel<0,0,0>` spill: 39 dw → 39 dw (UNCHANGED)
* `grouped_rcr_kernel<0,1,0>` spill: 43 dw → 43 dw (UNCHANGED)
* Force-instantiate symbols are unreachable from any kernel; zero
  runtime / codegen impact.

## Metric

* Before R34: 960
* After R34: 958 (within 947-963 plateau noise band)
* Bit-equivalent change confirmed; metric variation is run-to-run noise.

## Cumulative Lever D R-B infrastructure (R29-R34)

| Round | HK SHA | What | Lines |
|--|--|--|--|
| R29 | c2abba21 | `_s` aliases for rt_32x64 / rt_64x32 | 17 |
| R30 | 75e30a5f | static_assert namespace validating geometry | 78 |
| R31 | addaf23e | `rcr_mma_32` wrapper + force-instantiate stub | 36 |
| R32 | (PT only) | doc handoff with paste-ready code | 307 |
| R33 | (none) | acknowledge plateau | 0 |
| R34 | 78415fb0 | K-tail rt_32x64 loaders + extended stub | 102 |
| **Total** | — | — | **~540 lines** |

All HK commits bit-equivalent at runtime. ~233 lines of compile-time
validated kernel infrastructure across R29-R34.

## What's left for R35+ (next chat session)

Per R32's handoff doc — roughly halved scope now that loaders are landed:

1. **R35 (~25 min)**: K-tail block rewrite using `rcr_mma_32` and the
   new loaders. Replace the `if constexpr (FUSED_KTAIL)` branch at
   `kernel_fp8_layouts.cpp:~2363-2470` with the rt_32x64 path. Build,
   verify spill profile (target: ≤ 32 dw per spec, NOT increase).

2. **R36 (~25 min)**: LDS roundtrip merge — cAB_32 (rt_32x32) → cA-cD
   (rt_16x16) cell-shape conversion. Allocate scratch in shared mem,
   ds_write the 4 cAB_32 tiles, s_barrier, ds_read into cA-cD with
   fp32 additive. Run correctness (allclose vs golden) on gpt_oss
   B16/B32 cases. SNR ≥ 25 dB target.

3. **R37 (~10 min)**: metric measurement. If +1 pp or better, ship.
   Else revert (kernel state is recoverable: revert just the K-tail
   block change; loaders + wrapper + types stay as harmless dead code).

Total next-chat-session budget: ~60 min for R35-R37. Should fit
comfortably in one chat window with debug headroom.

## Design notes for R35+ implementer

### Slab/tile geometry

For the K-tail block, the warp covers 4 cA-cD output tiles each of
shape rt_fl<RBM=64, RBN=32, col_l, rt_16x16> = 64x32 region.

In the rt_32x64 K-tail port, each cAB_32 tile is rt_fl<RBM=64, RBN=32,
col_l, rt_32x32> = also 64x32 region. So 4 cAB_32 tiles cover the
same M×N region as cA-cD.

The mapping:
* cA → cAB_32[0]: M slab 0, N strip 0 (M=br*4+0, N=bc*2+0 in unit_coord space)
* cB → cAB_32[1]: M slab 0, N strip 1
* cC → cAB_32[2]: M slab 1 (now M=br*4+1 due to RBM=64 / 32-row cell)
* cD → cAB_32[3]: M slab 1, N strip 1

Wait — careful. RBM=64 / cell_rows=32 = height(2) for rt_32x64.
So one cAB_32 tile already covers 64 M-rows (= height 2 cells of 32 rows
each). To cover 4 cA-cD tiles (which cover M=128 rows total via 2 M
slabs at 64 rows each), we need 2 cAB_32 tiles per N strip × 2 N strips
= 4 cAB_32 total. ✓

Each cAB_32 tile mfma count: D::height(2) × D::width(1) × A::width(1)
= 2 mfma_323264. Total 4 cAB_32 × 2 mfma = 8 mfma_323264.
Compare current 16x16x128: 4 cA-cD × mma_AB(height 4 × width 2) × 1
= 32 mfma_16x16x128 (50% K-SENTINEL waste).

### M_warp_base for the new path

Current rt_16x128 path:
```
const int M_warp_base = (m_subtile_A + br*2 + slab) * HB + wm * RBM;
// where slab ∈ {0, 1}, RBM=64, HB=128
```

For rt_32x64 path, per-cAB_32 M base needs to handle 32-row cells:
```
const int M_warp_base = (m_subtile_A + br*4 + slab) * (HB/2) + wm * RBM;
// where slab ∈ {0, 1, 2, 3} now? Or just {0, 1} with height(2) folded in?
```

Actually since rt_32x64 has height=2 (= 64 M-rows per A tile, same as
existing A_row_reg height=4 × 16 = 64 rows), the loader handles the 2
cells internally via the `for h` loop. So M_warp_base mirrors the
existing pattern: 2 M slabs per warp output region, NOT 4.

Re-derive carefully when writing R35.

### LDS merge specifics

For the cAB_32 → cA-cD merge, the simplest path is via a kittens st
staging tile:

```cpp
__shared__ st_fl<RBM, RBN, st_32x32_s> staging[4]; // 4 cAB_32 worth
// (st_32x32_s alias may not exist; use raw shared buffer if so)
```

If kittens doesn't have st_32x32_s, allocate raw `__shared__ float buf[4 * 64 * 32]`
and use `ds_write_b128` / `ds_read_b128` directly. Either way, ~16 cy
write + 10 cy barrier + 16 cy read + 8 cy fp32 add = ~50 cy per cAB_32.

For 4 cAB_32: ~200 cy total (with batched merge, single barrier).

### Risk: register pressure spike

If LLVM keeps all 4 cAB_32 alive simultaneously (~128 dw added) on
top of cA-cD (128 dw) + a/b loaders (~40 dw), peak is ~296 dw — over
budget by 40 dw → ~320 cy spill cost.

Mitigation: do per-cAB_32 store + merge serially (one-at-a-time).
After ds_write of cAB_32[i] to LDS, mark cAB_32[i] dead and reuse
the registers for cAB_32[i+1]. This is the "one-at-a-time" pattern
from R58-dm cost model.

If LLVM is smart enough to do this automatically, peak is just
+32 dw — no spill. If not, manual `__builtin_amdgcn_s_setprio` /
inline asm hints may be needed (be VERY careful — R53 found
`__builtin_amdgcn_s_setprio(1)` was -120 pts catastrophic in the
main loop; impact in K-tail might differ).

## Round summary

| Item | Value |
|--|--|
| Goal | Land R32-handoff loaders before chat session ends |
| Change | Added load_a_kt_32x64 / load_b_kt_32x64 templates + extended force-instantiate |
| Lines added | 102 (HK 78415fb0) + this note |
| Spill profile change | NONE (39/43 dw — UNCHANGED) |
| Metric before | 960 |
| Metric after | 958 (within plateau noise band) |
| HK commit | `78415fb0` |
| Build outcome | Pass first try — types and dispatch fully validated |
| Next round suggestion | R35 in next chat: K-tail block rewrite using rcr_mma_32 + loaders |
