# Round 60-dm — FP8 grouped: Lever D R-B step 4 — chat-session handoff with loader skeleton

**Status**: HANDOFF — chat session ~90 min limit reached, deferring K-tail port wiring to next chat
**Score before**: 957 (R32 baseline)
**Score after**:  957 (no changes; doc-only commit)
**HK SHA**: `addaf23e` (unchanged)
**PT SHA**: this commit
**Round time**: ~8 min, 1 metric run, 0 build cycles
**Auto-optimize round**: 32

---

## Why doc-only this round

R32 was the round where Lever D R-B step 4 (K-tail loader + block wiring +
LDS roundtrip merge) was scheduled. Honest accounting:

| | minutes |
|--|--|
| Loader implementation + code-review for lane mapping correctness | ~10 |
| Build + iterate on type errors | ~5 |
| K-tail block rewrite using rcr_mma_32 | ~15 |
| LDS roundtrip merge implementation | ~10 |
| Build + iterate on dispatch / type errors | ~5 |
| Correctness validation (allclose, SNR) on gpt_oss B16/B32 | ~10 |
| Metric measurement | ~3 |
| Possible debug iteration if correctness fails | ~10-30 |
| **Total** | **~60-90 min** |

The chat session has run 82 min into a 90-min window. R32 cannot fit.
A forced partial implementation would leave the kernel in a broken
state (uncalled types defined but inconsistent loader) — net
regression risk.

This round commits ONE concrete artifact: the loader code skeleton
that the next chat agent can paste directly.

---

## State at end of R32 (Primus-Turbo SHA at this commit)

**Cumulative Lever D R-B infrastructure (R29-R31)** — bit-equivalent to baseline,
all compile-time validated:

| Round | HK SHA | What |
|--|--|--|
| R29 | `c2abba21` | `rt_32x64_s` / `rt_64x32_s` public aliases in `types.cuh` |
| R30 | `75e30a5f` | `static_assert` namespace `lever_d_round_b_step1_compile_test` validating shape geometry |
| R31 | `addaf23e` | `rcr_mma_32` wrapper + force-instantiate stub validating dispatch chain |
| **R32** | (none) | (this doc-only commit) |

**Critical R31 discovery** (in case the next agent forgets):
> For RCR layout (mma_ABt 32x32x64), **B uses `rt_32x64_s`, NOT `rt_64x32_s`**.
> `rt_64x32_s` is for CRR/CCR layouts (mma_AB with B col-major).

---

## Paste-ready loader skeleton for next chat agent

Copy this into `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` as
free `__device__` functions just below the existing `rcr_mma_32`
wrapper (around line 280). Then add force-instantiate calls inside
`__lever_d_round_b_force_instantiate_rcr_mma_32`.

```cpp
// Round-26-dm (auto-optimize R33+ / Lever D Round-B step 4):
// K-tail loaders for rt_32x64 fp8 layout. Each lane reads 2 × b128
// from HBM into rt_base.data[]. Lane mapping per AMD CDNA4 mfma_323264
// (verified by mma_ABt_base dispatch in mma.cuh:234-238):
//   row_lane    = laneid % 32      (rows 0..31 within 32-row cell)
//   k_lane_byte = (laneid / 32)*32 (chunk 0 = K[0..31], chunk 1 = K[32..63])
//   data[0..3]  = K=[k_lane_byte,        k_lane_byte + 16) → b128 #1
//   data[4..7]  = K=[k_lane_byte + 16,   k_lane_byte + 32) → b128 #2
//
// For K_REM=64 (gpt_oss K=2880 = 22*128 + 64): ALL LANES VALID, both
// b128 reads in-bounds. No SENTINEL lanes (vs rt_16x128 loader which
// had lanes 32..63 SENTINEL because K=64..127 out of range).

template<typename A_RT_32x64>
__device__ __forceinline__ void load_a_kt_32x64(
    A_RT_32x64& A_tile,
    ::kittens::i32x4 a_srsrc_kt,
    int M_warp_base,
    int row_lane,           // = laneid % 32
    int k_lane_byte,        // = (laneid / 32) * 32
    int a_row_stride_bytes,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    static_assert(A_RT_32x64::base_tile_rows == 32);
    static_assert(A_RT_32x64::base_tile_cols == 64);
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h = 0; h < A_RT_32x64::height; ++h) {
        const int A_row_idx = M_warp_base + h * 32 + row_lane;
        const uint32_t v_base = static_cast<uint32_t>(
            A_row_idx * a_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_hi, 0, 0);
        // tiles[h][0].data is dtype[8] = fp8e4m3_4[8] = 32 bytes/cell/lane
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
    }
}

template<typename B_RT_32x64>
__device__ __forceinline__ void load_b_kt_32x64(
    B_RT_32x64& B_tile,
    ::kittens::i32x4 b_srsrc_kt,
    int N_warp_base,
    int row_lane,
    int k_lane_byte,
    int b_row_stride_bytes,
    uint32_t b_group_byte_base,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    static_assert(B_RT_32x64::base_tile_rows == 32);
    static_assert(B_RT_32x64::base_tile_cols == 64);
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h_b = 0; h_b < B_RT_32x64::height; ++h_b) {
        const int B_row_idx_in_group = N_warp_base + h_b * 32 + row_lane;
        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
            B_row_idx_in_group * b_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_hi, 0, 0);
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
    }
}
```

Force-instantiate stub addendum (paste into existing
`__lever_d_round_b_force_instantiate_rcr_mma_32` body):

```cpp
    // Force-instantiate the loaders for the canonical 32x32x64 K-tail
    // shapes. Build will type-check the b128 buffer_load chain.
    rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s> dummy_A_kt;
    rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s> dummy_B_kt;
    ::kittens::i32x4 dummy_srsrc{};
    load_a_kt_32x64(dummy_A_kt, dummy_srsrc, /*M_warp_base=*/0,
                    /*row_lane=*/0, /*k_lane_byte=*/0,
                    /*a_row_stride_bytes=*/0, /*K_tail_base_bytes=*/0u,
                    /*b128_lo_valid=*/true, /*b128_hi_valid=*/true);
    load_b_kt_32x64(dummy_B_kt, dummy_srsrc, 0, 0, 0, 0, 0u, 0u,
                    true, true);
```

---

## Skeleton K-tail block (paste into `if constexpr (FUSED_KTAIL)` branch)

This replaces the existing K-tail block at `kernel_fp8_layouts.cpp:2363-2470`:

```cpp
if constexpr (FUSED_KTAIL) {
    using A_kt_32x64 = rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s>;
    using B_kt_32x64 = rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s>;
    using cAB_32     = rt_fl<RBM, RBN, col_l, rt_32x32_s>;

    if (g.fast_k < g.k) {
        const int laneid = kittens::laneid();
        const int row_lane    = laneid % 32;        // ← NEW lane mapping
        const int k_lane_byte = (laneid / 32) * 32; // ← NEW lane mapping
        const int K_REM = g.k - g.fast_k;
        const bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;
        const bool b128_hi_valid = (k_lane_byte + 32) <= K_REM;

        // (existing SRD setup — same as current loader)
        // …compute a_srsrc_kt, b_srsrc_kt, K_tail_base_bytes, b_group_byte_base…

        // M slab base: 1 cAB_32 covers 32 M-rows (rt_32x32 cell), so
        // M slabs [0..3] map to a_kt[0..3] using the SAME M dimension
        // pattern as existing loader, just with stride 32 (not 16).
        A_kt_32x64 a_kt[4];
        B_kt_32x64 b_kt[2];

        // Issue all loads up front (same hoist pattern as current path):
        #pragma unroll
        for (int n_strip = 0; n_strip < 2; ++n_strip) {
            const int N_warp_base = (bc * 2 + n_strip) * HB + wn * RBN;
            load_b_kt_32x64(b_kt[n_strip], b_srsrc_kt, N_warp_base,
                            row_lane, k_lane_byte, b_row_stride_bytes,
                            b_group_byte_base, K_tail_base_bytes,
                            b128_lo_valid, b128_hi_valid);
        }
        #pragma unroll
        for (int slab = 0; slab < 4; ++slab) {
            // M_warp_base for slab s: (m_subtile_A + br*4 + s) * (HB/2)
            // — note RBM=64 means each cAB_32 covers 32 rows, so 4 slabs
            // cover the 128-row tile (vs current 2 slabs at 64-row each).
            // BR/M-pattern needs careful re-derivation here.
            const int M_warp_base =
                (m_subtile_A + br * 4 + slab) * (HB / 2) + wm * RBM;
            load_a_kt_32x64(a_kt[slab], a_srsrc_kt, M_warp_base,
                            row_lane, k_lane_byte, a_row_stride_bytes,
                            K_tail_base_bytes,
                            b128_lo_valid, b128_hi_valid);
        }

        asm volatile("s_waitcnt vmcnt(0)");

        // 4 cAB_32 tiles cover the 4 cA-cD output tiles (M slab × N strip).
        cAB_32 c_kt[4];
        zero(c_kt[0]); zero(c_kt[1]); zero(c_kt[2]); zero(c_kt[3]);
        rcr_mma_32(c_kt[0], a_kt[0], b_kt[0]);  // M slab 0, N strip 0
        rcr_mma_32(c_kt[1], a_kt[0], b_kt[1]);  // M slab 0, N strip 1
        // (etc — wire up all 4×2 = 8 mfma_323264 calls)

        // LDS roundtrip merge: cAB_32 (rt_32x32) → cA-cD (rt_16x16).
        // Use existing kittens st_fl<RBM, RBN, st_32x32_s> staging tile
        // OR allocate scratch in shared mem manually. Each cAB_32 takes
        // 64×32 fp32 = 8 KB shared = 8 KB × 4 = 32 KB total scratch.
        // Per cAB_32:
        //   ds_write_b128 × 8  (32 dw write)
        //   s_barrier
        //   ds_read_b128 × 8 + accumulate into cA-cD
        // Estimated ~74 cy per cAB_32, ~300 cy total merge.

        // TODO: implement the merge using kittens store/load helpers
        // OR raw ds_* intrinsics. The kittens path is cleaner but may
        // not have a "load additive" overload — manual fp32 add likely.
    }
}
```

**Open question for next agent**: Does kittens have a built-in
`load(rt_fl<...,rt_16x16_s>, st_fl<...,rt_32x32_s>)` cross-layout
overload? If yes, use it. If no, implement raw `ds_read_b128` +
manual fp32 add into cA-cD's data[].

---

## Cost-model recap (from R58-dm and R59-dm)

| Component | Existing 16x16x128 K-tail | Lever D 32x32x64 | Δ |
|--|--|--|--|
| mfma issue | 256 cy (32 mfma × 8 cy, 50% K-SENTINEL waste) | 128 cy (8 mfma × 16 cy, no waste) | **-128 cy** |
| K-tail spill | ~256 cy (cA-cD + a + a_kt1 + b = 288 dw, over budget by 32) | 0 cy (192 dw peak, under budget by 64) | **-256 cy** |
| Cell-shape merge | 0 cy | ~300 cy (4 cAB_32 LDS roundtrip) | **+300 cy** |
| **Total** | **~512 cy** | **~428 cy** | **-84 cy savings** |

Revised estimate (more conservative than R58-dm's -284 cy):
* Per K-tail tile savings: ~84 cy
* If tile cycle count is ~1500 cy: savings = 5.6% per K-tile
* On K_REM>0 shapes (4 of 16): per-shape +5.6 pp = **+1.4 pp on
  grp_FP8 geomean** (instead of the optimistic +4.5 pp from R58-dm).

**Worst case** (LDS bank conflicts double the merge cost):
* Merge = ~600 cy
* Net = 256 + 128 - 600 = **-216 cy LOSS** = -1.4 pp regression.

So the empirical probe in R33+ has both upside and downside. Without
the actual measurement, the cost model is too uncertain to be sure.

---

## Recommendation for next chat session

1. **R33** (~30 min): paste the loader skeleton, build, verify
   force-instantiate type-checks. ~10 min.
2. **R33** continued (~20 min): paste the K-tail block skeleton,
   build, verify it compiles AND get spill profile. Don't worry
   about correctness yet.
3. **R34** (~30 min): implement the LDS merge. Run correctness
   (allclose vs golden) on gpt_oss B16/B32 cases. Iterate until
   SNR ≥ 25 dB.
4. **R35** (~10 min): metric measurement. Compare 16x16x128 vs
   32x32x64 K-tail. If +1 pp or better, ship. If 0 or negative,
   revert and document.

Total: 3 rounds of 30 min each. Should fit in one chat session.

If the next chat agent has fresh patience budget (which they will),
this is a reasonable bet.

---

## Round summary

| Item | Value |
|--|--|
| Goal | Honest handoff — chat window closing, defer Lever D wiring |
| Change | Doc-only (this round note); no code changes |
| Lines added | this note |
| Spill profile change | NONE (no code change) |
| Metric before | 957 |
| Metric after | 957 (no change) |
| HK commit | none |
| PT commit | this commit |
| Next round suggestion | R33+ in next chat: paste loader skeleton, wire K-tail block, validate correctness, measure metric |
