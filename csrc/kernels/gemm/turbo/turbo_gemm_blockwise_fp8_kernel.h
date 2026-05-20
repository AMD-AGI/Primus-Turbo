// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Blockwise FP8 GEMM kernel for GFX950 (MI350/MI355).
//
// DeepSeek-V3 style 1×128 (A rowwise) + 128×128 (B 2D) FP8 quantization with
// FP32 scales. Forward NT layout only.
//
// Tile: 128M × 128N × 128K, 4 warps as 2M × 2N (each warp 64M × 64N output).
// MFMA: v_mfma_f32_16x16x128_f8f6f4 (unscaled). Scale path uses a software
// promotion accumulator: each MFMA produces an inner FP32 partial in AGPR;
// per-fragment promotion folds (a_scale × b_scale) into a separate outer FP32
// accumulator that lives in 64 AGPR. This matches the design described in
// docs/kernel_optimize/primus_turbo_blockwise_fp8_gemm_design.md (candidate F').

#pragma once

#include "primus_turbo/device/memory.cuh"
#include "primus_turbo/device/mfma.cuh"
#include "primus_turbo/device/register.cuh"
#include <cassert>
#include <cstdint>

namespace primus_turbo {
namespace turbo {

using dtype::float32x4;

// from device/register.cuh
using device::clobber_agpr_one;
using device::clobber_vgpr_one;
using device::read_agpr;
using device::reserve_agpr_range;
using device::reserve_vgpr_range;
using device::zero_agpr;
using device::zero_agpr_range;

// from device/memory.cuh
using device::BufferSRD;
using device::ds_read_pinned;
using device::load_gmem_to_smem_srd;
using device::wait_lgkmcnt;
using device::wait_vmcnt;

// from device/mfma.cuh
using device::apply_scale_promotion_16x16;
using device::zero_vgpr_4;

// ── Tile-index swizzle (XCD-aware, identical to mxfp8) ──
template <uint32_t BLOCK_SIZE_M, uint32_t BLOCK_SIZE_N>
__device__ __forceinline__ void
swizzle_pid_m_n_blockwise(const int m, const int n, int &pid_m, int &pid_n) {
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;
    const int ntiles_n = (n + static_cast<int>(BLOCK_SIZE_N) - 1) / static_cast<int>(BLOCK_SIZE_N);
    const int WGM      = (ntiles_n > 32) ? 4 : 8;

    const int pid = static_cast<int>(blockIdx.x * gridDim.y + blockIdx.y);

    if (NUM_WGS < NUM_XCDS) {
        pid_m = static_cast<int>(blockIdx.x) * static_cast<int>(BLOCK_SIZE_M);
        pid_n = static_cast<int>(blockIdx.y) * static_cast<int>(BLOCK_SIZE_N);
        return;
    }

    const int q        = NUM_WGS / NUM_XCDS;
    const int r        = NUM_WGS % NUM_XCDS;
    const int xcd_id   = pid % NUM_XCDS;
    const int local_id = pid / NUM_XCDS;
    const int wgid     = xcd_id * q + local_id + min(xcd_id, r);

    const int num_pid_m = (m + static_cast<int>(BLOCK_SIZE_M) - 1) / static_cast<int>(BLOCK_SIZE_M);
    const int num_pid_n = (n + static_cast<int>(BLOCK_SIZE_N) - 1) / static_cast<int>(BLOCK_SIZE_N);
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id          = int(wgid / num_wgid_in_group);
    const int first_pid_m       = group_id * WGM;
    const int group_size_m      = min(num_pid_m - first_pid_m, WGM);
    pid_m                       = first_pid_m + int((wgid % num_wgid_in_group) % group_size_m);
    pid_n                       = int((wgid % num_wgid_in_group) / group_size_m);
    pid_m *= static_cast<int>(BLOCK_SIZE_M);
    pid_n *= static_cast<int>(BLOCK_SIZE_N);
}

// ── GemmTile: tile-level GEMM operations for 128x128x128 Blockwise FP8 ──
template <typename AType, typename BType, typename CType, typename AccType>
struct GEMM_Tile_BlockwiseFP8_NT_128x128x128_16x16x128_4_WAVE_GFX950 {
public:
    static constexpr uint32_t WARP_SIZE = 64;
    static constexpr uint32_t NUM_WARPS = 4;

    static constexpr uint32_t BLOCK_SIZE_M = 128;
    static constexpr uint32_t BLOCK_SIZE_N = 128;
    static constexpr uint32_t BLOCK_SIZE_K = 128;

    static constexpr uint32_t MFMA_SIZE_M = 16;
    static constexpr uint32_t MFMA_SIZE_N = 16;
    static constexpr uint32_t MFMA_SIZE_K = 128;

    // Blockwise scaling block size: 1×128 for A, 128×128 for B.
    static constexpr uint32_t SCALE_BLOCK = 128;

    // Per warp: 64M × 64N output, organized as 4 × 4 fragments of 16×16.
    static constexpr uint32_t WARP_M_FRAGS = 4;
    static constexpr uint32_t WARP_N_FRAGS = 4;
    static constexpr uint32_t NUM_FRAGS    = WARP_M_FRAGS * WARP_N_FRAGS;

    using Mfma = device::mfma_f32_16x16x128_f8f6f4<AType, BType>;

    // Pinned register layout.
    //
    // VGPR:
    //   v[0:95]    compiler-managed (96) — hosts the 4 outer-AGPR shuttle
    //              VGPRs declared as inline-asm operands inside
    //              ``apply_scale_promotion_16x16`` (so the compiler
    //              tracks def/use and does not clobber them between
    //              read/fma/write).
    //   v[96:99]   inner accumulator A (4) — VGPR-resident MFMA output
    //              for even-index fragments (frag 0, 2, 4, …, 14)
    //   v[100:103] inner accumulator B (4) — VGPR-resident MFMA output
    //              for odd-index fragments (frag 1, 3, 5, …, 15). The
    //              two buffers enable a 2-deep software pipeline that
    //              hides the MFMA 32-cycle drain behind the next MFMA's
    //              MAI window (see ``compute_one_k_iter``).
    //   v[104:119] combined scale buffer (16) — 4 frag_m × 4 scales/lane
    //   v[120:123] b_scale: broadcast B scale (4 — only [0] used, padded)
    //   v[124:155] A data buffer (32) — 64M × 128K FP8
    //   v[156:187] B data buffer (32) — 64N × 128K FP8
    //
    // AGPR:
    //   a[0:63]    outer accumulator (16 fragments × 4 AGPR = 64)
    static constexpr int PIN_INNER_A   = 96;  // 4 VGPR for even-frag inner partial
    static constexpr int PIN_INNER_B   = 100; // 4 VGPR for odd-frag inner partial
    static constexpr int PIN_S         = 104; // 16 VGPR for combined A*B scales
    static constexpr int PIN_B_SCALE   = 120; // 4 VGPR; only PIN_B_SCALE[0] live
    static constexpr int PIN_A         = 124; // 32 VGPR for A data (64M × 128K)
    static constexpr int PIN_B         = 156; // 32 VGPR for B data (64N × 128K)

    static constexpr int AGPR_OUTER = 0; // 64 AGPR

    // SMEM tiles ────────────────────────────────────────────────────────────
    template <typename T, uint32_t N> struct SmemTile {
        T                   data[N];
        __device__ uint32_t u32_ptr() { return reinterpret_cast<uintptr_t>(data); }
    };

    // 64-row sub-tiles to match the 4-warp loader pattern (warp w covers 16 rows
    // of every sub-tile; 4 warps × 16 rows = 64 rows per sub-tile).
    using ASmemSubtile = SmemTile<AType, 64 * BLOCK_SIZE_K>; // 64 × 128 FP8
    using BSmemSubtile = SmemTile<BType, 64 * BLOCK_SIZE_K>;
    // A scales: 128 FP32 (one per M row, K-block constant within tile).
    // The loader is dispatched as 2 warps × 64 lanes × 4 B = 256 B/warp; warp 0
    // covers the lower 64 rows, warp 2 covers the upper 64 rows.
    using AScaleSmemTile = SmemTile<float, BLOCK_SIZE_M>; // 128 FP32 (512 B)
    // B scale: 1 FP32 broadcast across the full 128×128 tile. buffer_load_lds<4>
    // with 64 lanes occupies 256 B even when only the first slot is consumed.
    using BScaleSmemTile = SmemTile<float, WARP_SIZE>; // 64 FP32 (256 B)

    const uint32_t lane_id;
    const uint32_t warp_id;
    const uint32_t warp_m, warp_n;
    const uint32_t m, n, k;

public:
    __device__ __forceinline__ GEMM_Tile_BlockwiseFP8_NT_128x128x128_16x16x128_4_WAVE_GFX950(
        uint32_t tid, uint32_t m, uint32_t n, uint32_t k)
        : lane_id(tid % WARP_SIZE), warp_id(tid / WARP_SIZE), warp_m(tid / WARP_SIZE / 2),
          warp_n(tid / WARP_SIZE % 2), m(m), n(n), k(k) {}

    // ── GMEM → SMEM data loads ──
    // Each warp writes 16 rows to each of the 2 sub-tiles. 2 ldg/sts pairs per
    // sub-tile cover the 16M × 128K rectangle (16 rows × 128 bytes/row = 2048
    // bytes; 64 lanes × 16 bytes/lane × 2 = 2048 bytes ✓).
    __device__ __forceinline__ void
    load_a_gmem_to_smem(const BufferSRD &a_srd, const uint32_t (&ldg_offsets)[2],
                        ASmemSubtile (&a_smem_tile)[2], const uint32_t (&sts_offsets)[2],
                        int32_t extra_soffset = 0) {
        const uint32_t sts_warp_base = warp_id * MFMA_SIZE_M * MFMA_SIZE_K;
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            int32_t soff = __builtin_amdgcn_readfirstlane(
                (int32_t) ((i * 64 + warp_id * MFMA_SIZE_M) * k) + extra_soffset);
            load_gmem_to_smem_srd<16>(a_srd, ldg_offsets[0],
                                      a_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[0],
                                      soff);
            load_gmem_to_smem_srd<16>(a_srd, ldg_offsets[1],
                                      a_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[1],
                                      soff);
        }
    }

    __device__ __forceinline__ void
    load_b_gmem_to_smem(const BufferSRD &b_srd, const uint32_t (&ldg_offsets)[2],
                        BSmemSubtile (&b_smem_tile)[2], const uint32_t (&sts_offsets)[2],
                        int32_t extra_soffset = 0) {
        const uint32_t sts_warp_base = warp_id * MFMA_SIZE_N * MFMA_SIZE_K;
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            int32_t soff = __builtin_amdgcn_readfirstlane(
                (int32_t) ((i * 64 + warp_id * MFMA_SIZE_N) * k) + extra_soffset);
            load_gmem_to_smem_srd<16>(b_srd, ldg_offsets[0],
                                      b_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[0],
                                      soff);
            load_gmem_to_smem_srd<16>(b_srd, ldg_offsets[1],
                                      b_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[1],
                                      soff);
        }
    }

    // ── GMEM → SMEM scale loads ──
    // A scale: 128 FP32 = 512 B per K-iter. Two of the 4 warps (id%2 == 0)
    // each load 64 contiguous M rows; warps with id%2 == 1 share data with
    // their pair so the load is redundant and skipped. 64 lanes × 4 B/lane =
    // 256 B per loader-warp ⇒ 1 ldg/loader.
    //
    // M0 (uniform LDS dest base) = a_s_smem.u32_ptr() + warp_m * 256.
    // Per-lane ldg_offset      = (warp_m * 64 + lane_id) * scale_cols * 4.
    __device__ __forceinline__ void
    load_a_scale_gmem_to_smem(const BufferSRD &a_s_srd, AScaleSmemTile &a_s_smem,
                              const uint32_t scale_cols, int32_t extra_soffset = 0) {
        if (warp_n == 0) {
            const uint32_t row              = warp_m * 64 + lane_id;
            const uint32_t ldg_offset_bytes = row * scale_cols * (uint32_t) sizeof(float);
            const uint32_t lds_base = a_s_smem.u32_ptr() + warp_m * 256;
            const int32_t  soff = __builtin_amdgcn_readfirstlane(extra_soffset);
            load_gmem_to_smem_srd<4>(a_s_srd, ldg_offset_bytes, lds_base, soff);
        }
    }

    // B scale: 1 FP32 per K-iter for the entire 128×128 tile. Issued by warp 0;
    // 64 lanes redundantly load the same 4 B (one ldg returns 256 B at the
    // same uniform GMEM offset, of which only the first 4 B is consumed).
    __device__ __forceinline__ void
    load_b_scale_gmem_to_smem(const BufferSRD &b_s_srd, BScaleSmemTile &b_s_smem,
                              const uint32_t b_scale_block_offset, int32_t extra_soffset = 0) {
        if (warp_id == 0) {
            const int32_t soff = __builtin_amdgcn_readfirstlane(extra_soffset);
            load_gmem_to_smem_srd<4>(b_s_srd, b_scale_block_offset, b_s_smem.u32_ptr(), soff);
        }
    }

    // ── SMEM → VGPR LDS reads ──
    // 64M × 128K data subtile: 8 ds_read_b128 (4 fragments × 2 K-halves).
    template <int VSTART>
    __device__ __forceinline__ static void load_data_subtile_pinned(uint32_t subtile_addr,
                                                                    uint32_t (&lds_offsets)[2]) {
        uint32_t addr0 = subtile_addr + lds_offsets[0];
        uint32_t addr1 = subtile_addr + lds_offsets[1];
        ds_read_pinned<16, VSTART + 0, 0>(addr0);
        ds_read_pinned<16, VSTART + 4, 0>(addr1);
        ds_read_pinned<16, VSTART + 8, 2048>(addr0);
        ds_read_pinned<16, VSTART + 12, 2048>(addr1);
        ds_read_pinned<16, VSTART + 16, 4096>(addr0);
        ds_read_pinned<16, VSTART + 20, 4096>(addr1);
        ds_read_pinned<16, VSTART + 24, 6144>(addr0);
        ds_read_pinned<16, VSTART + 28, 6144>(addr1);
    }

    // Per warp A scale read: 4 ds_read_b128 (one per M-fragment). Each lane gets
    // 4 FP32 scales for its 4 lane-local rows of the 16×16 fragment output.
    //
    // Lane (lane_id) holds fragment partials at row = (lane_id/16)*4 + i (i=0..3),
    // col = lane_id%16. Reading 16 B at (lane_id/16)*16 inside the fragment's
    // 16-row scale block delivers the 4 needed scales.
    template <int VSTART>
    __device__ __forceinline__ void
    load_a_scale_per_warp_pinned(uint32_t a_scale_smem_base) const {
        // base: 4 B/scale × 64 scales/warp_m = 256 B/warp_m
        const uint32_t warp_base = warp_m * 256;
        const uint32_t lane_base = (lane_id / 16) * 16;
        const uint32_t addr      = a_scale_smem_base + warp_base + lane_base;
        ds_read_pinned<16, VSTART + 0, 0>(addr);   // M-frag 0 (rows 0..15)
        ds_read_pinned<16, VSTART + 4, 64>(addr);  // M-frag 1 (rows 16..31)
        ds_read_pinned<16, VSTART + 8, 128>(addr); // M-frag 2 (rows 32..47)
        ds_read_pinned<16, VSTART + 12, 192>(addr); // M-frag 3 (rows 48..63)
    }

    // B scale broadcast: all lanes load the same 4 B (one FP32). Pinned VGPR
    // slot holds the value lane-locally for use in the combined-scale multiply.
    template <int VSTART>
    __device__ __forceinline__ void
    load_b_scale_pinned(uint32_t b_scale_smem_base) const {
        ds_read_pinned<4, VSTART, 0>(b_scale_smem_base);
    }

    __device__ __forceinline__ void zero_c_agpr() { zero_agpr_range<AGPR_OUTER, AGPR_OUTER + 63>(); }

    __device__ __forceinline__ void reserve_pinned_regs() {
        // Reserve VGPRs that are owned by inline asm. The outer-accumulator
        // shuttle (formerly PIN_TMP_OUTER, v[100..103]) is now declared as
        // ``uint32_t`` inline-asm operands in ``apply_scale_promotion_16x16``,
        // so the compiler tracks it directly and no explicit reservation is
        // needed here.
        reserve_vgpr_range<PIN_INNER_A, PIN_INNER_A + 3>();
        reserve_vgpr_range<PIN_INNER_B, PIN_INNER_B + 3>();
        reserve_vgpr_range<PIN_S, PIN_S + 15>();
        reserve_vgpr_range<PIN_B_SCALE, PIN_B_SCALE + 3>();
        reserve_vgpr_range<PIN_A, PIN_A + 31>();
        reserve_vgpr_range<PIN_B, PIN_B + 31>();
        reserve_agpr_range<AGPR_OUTER, AGPR_OUTER + 63>();
    }

    // ── C output read from outer AGPR ──
    // Layout: AGPR[(frag_m * WARP_N_FRAGS + frag_n) * 4 + i] for i ∈ [0..3].
    template <int FRAG_M, int FRAG_N>
    __device__ __forceinline__ void read_c_fragment_from_agpr(float32x4 &c_out) {
        constexpr int B = (FRAG_M * WARP_N_FRAGS + FRAG_N) * 4;
        c_out           = read_agpr<float32x4, AGPR_OUTER + B>();
    }

    // Store one 16×16 fragment back to GMEM.
    __device__ __forceinline__ void store_c_fragment(CType *c_stg_base_ptr, const int32_t n_stride,
                                                     const float32x4 &c_frag,
                                                     const uint32_t (&c_stg_offsets)[4],
                                                     const int32_t valid_rows = 16,
                                                     const int32_t valid_cols = 16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int32_t row = lane_id / 16 * 4 + i;
            int32_t col = lane_id % 16;
            if (row < valid_rows && col < valid_cols)
                c_stg_base_ptr[c_stg_offsets[i]] = CType(c_frag[i]);
        }
    }

    // ── Address computation ──
    __device__ __forceinline__ void compute_ldg_offsets(uint32_t (&ldg_offsets)[2],
                                                        const uint32_t stride) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t ldg_row = i * 8 + lane_id / 8;
            uint32_t ldg_col = swizzle_col_(ldg_row, lane_id % 8);
            ldg_offsets[i]   = ldg_row * stride + ldg_col * 16;
        }
    }

    __device__ __forceinline__ void compute_sts_offsets(uint32_t (&sts_offsets)[2]) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sts_offsets[i] = i * 1024 + lane_id * 16;
        }
    }

    __device__ __forceinline__ void compute_lds_offsets(uint32_t (&lds_offsets)[2]) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t lds_row = lane_id % 16;
            uint32_t lds_col = lane_id / 16 + i * 4;
            uint32_t swz_col = swizzle_col_(lds_row, lds_col);
            lds_offsets[i]   = lds_row * 128 + swz_col * 16;
        }
    }

    __device__ __forceinline__ void compute_stg_offsets(uint32_t (&c_stg_offsets)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            c_stg_offsets[i] = (lane_id / 16 * 4 + i) * n + lane_id % 16;
        }
    }

    __device__ __forceinline__ uint32_t swizzle_col_(const uint32_t row, const uint32_t col) {
        return col ^ (row >> 1);
    }

    // ── Combined-scale prologue ──
    // Multiplies pre-loaded raw A scales (16 VGPR) by the broadcast B scale,
    // overwriting the same VGPR with combined values used by promotion.
    __device__ __forceinline__ static void multiply_combined_scale() {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            asm volatile("v_mul_f32_e32 v[%0], v[%1], v[%2]"
                         :
                         : "n"(PIN_S + i), "n"(PIN_B_SCALE), "n"(PIN_S + i));
        }
    }

    // ── Pipelined MFMA + per-fragment promotion ──
    // 16 fragments are walked in row-major order (FRAG_IDX = 0..15) and split
    // across two inner-accumulator buffers (even FRAG_IDX → PIN_INNER_A, odd
    // → PIN_INNER_B). A 2-deep software pipeline issues the MFMA for
    // fragment I in the same step as the promotion for fragment I-2, so each
    // promotion's first ``v_fmac`` lands ≥ 32 cycles after the matching
    // MFMA's MAI write and no explicit ``s_nop 15`` drain is required.
    //
    // Per-step layout (steady state, FRAG_IDX = 2..15):
    //   promote_(FRAG_IDX-2)            (12 cycles VALU; reads INNER_(FI-2))
    //   zero buf_(FRAG_IDX%2)           (6 cycles VALU; overwrites INNER)
    //   mfma_(FRAG_IDX) -> INNER_(FI%2) (1 cycle issue; MAI runs 32 cycles)
    //
    // MFMA throughput (one per 32 cycles on a wave's MAI pipeline) dominates,
    // so steady-state cadence is 32 cycles per fragment vs. the baseline ~51
    // cycles per fragment (which serialised drain + promote after every MFMA).

    template <int FRAG_IDX>
    __device__ __forceinline__ static void pipeline_steady() {
        constexpr int FRAG_M  = FRAG_IDX / WARP_N_FRAGS;
        constexpr int FRAG_N  = FRAG_IDX % WARP_N_FRAGS;
        constexpr int PROMOTE_IDX   = FRAG_IDX - 2;
        constexpr int PROMOTE_M     = PROMOTE_IDX / WARP_N_FRAGS;
        constexpr int PROMOTE_N     = PROMOTE_IDX % WARP_N_FRAGS;
        constexpr int CUR_INNER     = (FRAG_IDX & 1) ? PIN_INNER_B : PIN_INNER_A;
        constexpr int PROMOTE_INNER = (PROMOTE_IDX & 1) ? PIN_INNER_B : PIN_INNER_A;
        constexpr int OUTER_PROMOTE = AGPR_OUTER + (PROMOTE_M * WARP_N_FRAGS + PROMOTE_N) * 4;
        constexpr int SCALE_PROMOTE = PIN_S + PROMOTE_M * 4;

        // Promote frag (FRAG_IDX-2): reads INNER buf last written by
        // mfma_(FRAG_IDX-2). With a 2-deep pipeline that MFMA was launched
        // 2 × 32 cycles ago, so its 32-cycle MAI write is long drained.
        apply_scale_promotion_16x16<OUTER_PROMOTE, PROMOTE_INNER, SCALE_PROMOTE,
                                    /*DRAIN_MFMA=*/false>();
        // Zero the same buf in preparation for mfma_(FRAG_IDX). The
        // (PROMOTE_IDX & 1) == (FRAG_IDX & 1) parity match means we
        // overwrite the buf that promote just consumed (read-then-overwrite,
        // so safe).
        zero_vgpr_4<CUR_INNER>();
        Mfma::template run_pinned_acc_vgpr<PIN_A + FRAG_M * 8, PIN_B + FRAG_N * 8, CUR_INNER>();
    }

    template <int FRAG_IDX, bool DRAIN>
    __device__ __forceinline__ static void pipeline_epilogue() {
        constexpr int FRAG_M = FRAG_IDX / WARP_N_FRAGS;
        constexpr int FRAG_N = FRAG_IDX % WARP_N_FRAGS;
        constexpr int INNER  = (FRAG_IDX & 1) ? PIN_INNER_B : PIN_INNER_A;
        constexpr int OUTER  = AGPR_OUTER + (FRAG_M * WARP_N_FRAGS + FRAG_N) * 4;
        constexpr int SCALE  = PIN_S + FRAG_M * 4;
        apply_scale_promotion_16x16<OUTER, INNER, SCALE, DRAIN>();
    }

    __device__ __forceinline__ static void compute_one_k_iter() {
        // ── Prologue (interleaved with multiply_combined_scale) ──
        //
        // 2-deep MFMA prologue gives every steady-state promote a ≥ 32
        // cycle natural gap from its matching MFMA write so the drain
        // NOPs inside ``apply_scale_promotion_16x16`` can be skipped.
        //
        // ``multiply_combined_scale`` is hoisted INSIDE mfma_0's MAI
        // window (16 VALU cycles fit in the 32-cycle MFMA pipeline
        // latency). Its inputs (PIN_S / PIN_B_SCALE) were loaded by the
        // outer-loop LDS reads and are valid here; its outputs are
        // first consumed by promote_0 in step 2 (after the whole
        // prologue), well after mul completes.
        zero_vgpr_4<PIN_INNER_A>();
        Mfma::template run_pinned_acc_vgpr<PIN_A + 0 * 8, PIN_B + 0 * 8, PIN_INNER_A>();
        multiply_combined_scale();
        zero_vgpr_4<PIN_INNER_B>();
        Mfma::template run_pinned_acc_vgpr<PIN_A + 0 * 8, PIN_B + 1 * 8, PIN_INNER_B>();

        // ── Steady state: 14 paired (promote + mfma) steps ──
        pipeline_steady<2>();
        pipeline_steady<3>();
        pipeline_steady<4>();
        pipeline_steady<5>();
        pipeline_steady<6>();
        pipeline_steady<7>();
        pipeline_steady<8>();
        pipeline_steady<9>();
        pipeline_steady<10>();
        pipeline_steady<11>();
        pipeline_steady<12>();
        pipeline_steady<13>();
        pipeline_steady<14>();
        pipeline_steady<15>();

        // ── Epilogue: promote frag 14 and frag 15 ──
        // - promote_14 follows mfma_14 by ~64 cycles (steady step gap) → safe.
        // - promote_15 follows mfma_15 by ~30 cycles (step 15 steady cadence
        //   32 + epilogue<14> VALU 12 + 4 cycle accvgpr_read prefix, minus
        //   mfma_15's 18-cycle in-step offset). 30 ≥ 11 → in the
        //   16x16x128_f8f6f4 "11+ OK" drain window, so no explicit drain is
        //   needed either. Saves 32 cycle/K-iter vs. the conservative
        //   ``DRAIN=true``.
        pipeline_epilogue<14, /*DRAIN=*/false>();
        pipeline_epilogue<15, /*DRAIN=*/false>();
    }
};

// ── Blockwise FP8 NT GEMM Kernel (128x128x128, 4-warp, GFX950) ──
//
// Forward layout: C[M,N] = A[M,K] · B[N,K]^T.
//   A scales: rowwise 1×128, shape [M, K/128], FP32.
//   B scales: 2D 128×128, shape [N/128, K/128], FP32.
//
// Pipeline (single-buffer A/B, 1 barrier per K-iter):
//   prologue:   issue LDG K0 → SMEM
//   main loop:  issue LDS for K_i → VGPR
//               wait for LDS, run MFMA + promotion for K_i
//               issue LDG K_{i+1} → SMEM (overlaps with compute since SMEM is
//               released after LDS)
//               s_barrier
template <typename AType, typename BType, typename CType, typename AccType = float>
__global__ __launch_bounds__(256, 2) void turbo_gemm_blockwise_fp8_128x128x128_16x16x128_4wave_kernel(
    const AType *a_ptr, const BType *b_ptr, const float *a_s_ptr, const float *b_s_ptr,
    CType *c_ptr, const uint32_t m, const uint32_t n, const uint32_t k) {
#if !defined(__gfx950__)
    assert(false && "turbo_gemm_blockwise_fp8 kernel requires gfx950");
    return;
#else
    using GemmTile =
        GEMM_Tile_BlockwiseFP8_NT_128x128x128_16x16x128_4_WAVE_GFX950<AType, BType, CType, AccType>;
    GemmTile tile(threadIdx.x, m, n, k);
    tile.reserve_pinned_regs();

    const uint32_t lane_id = tile.lane_id;
    const uint32_t warp_id = tile.warp_id;
    const uint32_t warp_m  = tile.warp_m;
    const uint32_t warp_n  = tile.warp_n;

    using ASmem                       = typename GemmTile::ASmemSubtile;
    using BSmem                       = typename GemmTile::BSmemSubtile;
    using ASSmem                      = typename GemmTile::AScaleSmemTile;
    using BSSmem                      = typename GemmTile::BScaleSmemTile;
    constexpr size_t SMEM_DATA_BYTES  = sizeof(ASmem) * 2 + sizeof(BSmem) * 2;
    constexpr size_t SMEM_SCALE_BYTES = sizeof(ASSmem) + sizeof(BSSmem);
    __shared__ char  smem_buf[SMEM_DATA_BYTES + SMEM_SCALE_BYTES];

    auto &a_smem_tile = *reinterpret_cast<ASmem(*)[2]>(smem_buf);
    auto &b_smem_tile = *reinterpret_cast<BSmem(*)[2]>(smem_buf + sizeof(ASmem) * 2);
    auto &a_s_smem    = *reinterpret_cast<ASSmem *>(smem_buf + SMEM_DATA_BYTES);
    auto &b_s_smem    = *reinterpret_cast<BSSmem *>(smem_buf + SMEM_DATA_BYTES + sizeof(ASSmem));

    int32_t pid_m, pid_n;
    swizzle_pid_m_n_blockwise<GemmTile::BLOCK_SIZE_M, GemmTile::BLOCK_SIZE_N>(m, n, pid_m, pid_n);
    if (pid_m >= (int32_t) m || pid_n >= (int32_t) n)
        return;

    const AType *a_base_ptr = a_ptr + (int64_t) pid_m * k;
    const BType *b_base_ptr = b_ptr + (int64_t) pid_n * k;
    const uint32_t scale_cols = (k + GemmTile::SCALE_BLOCK - 1) / GemmTile::SCALE_BLOCK;
    const float   *a_s_base_ptr = a_s_ptr + (int64_t) pid_m * scale_cols;
    // B scale: row index = pid_n / 128.
    const float *b_s_base_ptr = b_s_ptr;
    const uint32_t b_scale_row = pid_n / GemmTile::SCALE_BLOCK;
    const uint32_t b_scale_row_offset_bytes =
        b_scale_row * scale_cols * (uint32_t) sizeof(float);

    uint32_t ldg_offsets[2];
    tile.compute_ldg_offsets(ldg_offsets, k);
    uint32_t sts_offsets[2];
    tile.compute_sts_offsets(sts_offsets);
    uint32_t lds_offsets[2];
    tile.compute_lds_offsets(lds_offsets);

    const uint32_t  a_remaining  = (m - pid_m) * k * sizeof(AType);
    const uint32_t  b_remaining  = (n - pid_n) * k * sizeof(BType);
    const uint32_t  as_remaining = (m - pid_m) * scale_cols * sizeof(float);
    // B-scale SRD base is the full ``b_s_ptr`` (not offset by ``b_scale_row``)
    // because ``load_b_scale_gmem_to_smem`` already passes
    // ``b_scale_row * scale_cols * sizeof(float)`` as the ldg-offset. NUM_RECORDS
    // therefore has to span the full buffer in bytes; otherwise the second
    // N-tile (b_scale_row >= 1) reads at an offset >= NUM_RECORDS and the
    // hardware OOB-select returns 0, silently dropping the B scale for that
    // tile.
    const uint32_t  bs_remaining =
        ((n + GemmTile::SCALE_BLOCK - 1) / GemmTile::SCALE_BLOCK) * scale_cols * sizeof(float);
    const BufferSRD a_srd(a_base_ptr, a_remaining);
    const BufferSRD b_srd(b_base_ptr, b_remaining);
    const BufferSRD a_s_srd(a_s_base_ptr, as_remaining);
    const BufferSRD b_s_srd(b_s_base_ptr, bs_remaining);

    constexpr int32_t DATA_STRIDE  = (int32_t) GemmTile::BLOCK_SIZE_K;
    constexpr int32_t SCALE_STRIDE = (int32_t) sizeof(float);

    // ── Prologue: issue LDG for K-iter 0 ──
    tile.load_a_gmem_to_smem(a_srd, ldg_offsets, a_smem_tile, sts_offsets);
    tile.load_b_gmem_to_smem(b_srd, ldg_offsets, b_smem_tile, sts_offsets);
    tile.load_a_scale_gmem_to_smem(a_s_srd, a_s_smem, scale_cols, 0);
    tile.load_b_scale_gmem_to_smem(b_s_srd, b_s_smem, b_scale_row_offset_bytes, 0);

    tile.zero_c_agpr();
    wait_vmcnt<0>();
    __builtin_amdgcn_s_barrier();

    const uint32_t k_iters = (k + GemmTile::BLOCK_SIZE_K - 1) / GemmTile::BLOCK_SIZE_K;

    // ── Main loop ──
    int32_t data_off  = DATA_STRIDE;  // offset for the *next* K-iter's LDG
    int32_t scale_off = SCALE_STRIDE;
    for (uint32_t ki = 0; ki < k_iters; ++ki, data_off += DATA_STRIDE, scale_off += SCALE_STRIDE) {
        // 1) Issue LDS reads for current K-iter (data + raw A scale + B scale).
        GemmTile::template load_data_subtile_pinned<GemmTile::PIN_A>(
            a_smem_tile[warp_m].u32_ptr(), lds_offsets);
        GemmTile::template load_data_subtile_pinned<GemmTile::PIN_B>(
            b_smem_tile[warp_n].u32_ptr(), lds_offsets);
        tile.template load_a_scale_per_warp_pinned<GemmTile::PIN_S>(a_s_smem.u32_ptr());
        tile.template load_b_scale_pinned<GemmTile::PIN_B_SCALE>(b_s_smem.u32_ptr());

        // 2) Wait for LDS reads.
        wait_lgkmcnt<0>();

        // 3) Issue LDG for next K-iter (after LDS reads consumed SMEM).
        if (ki + 1 < k_iters) {
            tile.load_a_gmem_to_smem(a_srd, ldg_offsets, a_smem_tile, sts_offsets, data_off);
            tile.load_b_gmem_to_smem(b_srd, ldg_offsets, b_smem_tile, sts_offsets, data_off);
            tile.load_a_scale_gmem_to_smem(a_s_srd, a_s_smem, scale_cols, scale_off);
            tile.load_b_scale_gmem_to_smem(b_s_srd, b_s_smem, b_scale_row_offset_bytes, scale_off);
        }

        // 4) Compute: 16 MFMA + 16 promotion.
        GemmTile::compute_one_k_iter();

        // 5) Sync: wait for next K-iter's LDG before reusing SMEM.
        if (ki + 1 < k_iters) {
            wait_vmcnt<0>();
            __builtin_amdgcn_s_barrier();
        }
    }

    // ── Store C ──
    __builtin_amdgcn_sched_barrier(0);
    uint32_t c_stg_offsets[4];
    tile.compute_stg_offsets(c_stg_offsets);

    CType *c_stg_base_ptr =
        c_ptr + (int64_t) pid_m * n + pid_n + warp_m * 64 * n + warp_n * 64;
    const bool is_boundary_tile = (pid_m + (int32_t) GemmTile::BLOCK_SIZE_M > (int32_t) m) ||
                                  (pid_n + (int32_t) GemmTile::BLOCK_SIZE_N > (int32_t) n);

    auto store_one_fragment = [&](int frag_m, int frag_n, const float32x4 &c, int32_t vrow,
                                  int32_t vcol) {
        CType *p = c_stg_base_ptr + frag_m * GemmTile::MFMA_SIZE_M * n +
                   frag_n * GemmTile::MFMA_SIZE_N;
        tile.store_c_fragment(p, n, c, c_stg_offsets, vrow, vcol);
    };

    if (!is_boundary_tile) {
#pragma unroll
        for (int frag_m = 0; frag_m < (int) GemmTile::WARP_M_FRAGS; ++frag_m) {
#pragma unroll
            for (int frag_n = 0; frag_n < (int) GemmTile::WARP_N_FRAGS; ++frag_n) {
                float32x4 c_frag;
                // Switch on (frag_m, frag_n) at compile time.
                if (frag_m == 0 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<0, 0>(c_frag);
                else if (frag_m == 0 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<0, 1>(c_frag);
                else if (frag_m == 0 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<0, 2>(c_frag);
                else if (frag_m == 0 && frag_n == 3)
                    tile.template read_c_fragment_from_agpr<0, 3>(c_frag);
                else if (frag_m == 1 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<1, 0>(c_frag);
                else if (frag_m == 1 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<1, 1>(c_frag);
                else if (frag_m == 1 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<1, 2>(c_frag);
                else if (frag_m == 1 && frag_n == 3)
                    tile.template read_c_fragment_from_agpr<1, 3>(c_frag);
                else if (frag_m == 2 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<2, 0>(c_frag);
                else if (frag_m == 2 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<2, 1>(c_frag);
                else if (frag_m == 2 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<2, 2>(c_frag);
                else if (frag_m == 2 && frag_n == 3)
                    tile.template read_c_fragment_from_agpr<2, 3>(c_frag);
                else if (frag_m == 3 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<3, 0>(c_frag);
                else if (frag_m == 3 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<3, 1>(c_frag);
                else if (frag_m == 3 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<3, 2>(c_frag);
                else
                    tile.template read_c_fragment_from_agpr<3, 3>(c_frag);
                store_one_fragment(frag_m, frag_n, c_frag, 16, 16);
            }
        }
    } else {
        const int32_t warp_base_m = warp_m * 64;
        const int32_t warp_base_n = warp_n * 64;
        const int32_t tile_m_lim  = min((int32_t) m - pid_m, (int32_t) GemmTile::BLOCK_SIZE_M);
        const int32_t tile_n_lim  = min((int32_t) n - pid_n, (int32_t) GemmTile::BLOCK_SIZE_N);
#pragma unroll
        for (int frag_m = 0; frag_m < (int) GemmTile::WARP_M_FRAGS; ++frag_m) {
#pragma unroll
            for (int frag_n = 0; frag_n < (int) GemmTile::WARP_N_FRAGS; ++frag_n) {
                float32x4 c_frag;
                if (frag_m == 0 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<0, 0>(c_frag);
                else if (frag_m == 0 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<0, 1>(c_frag);
                else if (frag_m == 0 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<0, 2>(c_frag);
                else if (frag_m == 0 && frag_n == 3)
                    tile.template read_c_fragment_from_agpr<0, 3>(c_frag);
                else if (frag_m == 1 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<1, 0>(c_frag);
                else if (frag_m == 1 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<1, 1>(c_frag);
                else if (frag_m == 1 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<1, 2>(c_frag);
                else if (frag_m == 1 && frag_n == 3)
                    tile.template read_c_fragment_from_agpr<1, 3>(c_frag);
                else if (frag_m == 2 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<2, 0>(c_frag);
                else if (frag_m == 2 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<2, 1>(c_frag);
                else if (frag_m == 2 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<2, 2>(c_frag);
                else if (frag_m == 2 && frag_n == 3)
                    tile.template read_c_fragment_from_agpr<2, 3>(c_frag);
                else if (frag_m == 3 && frag_n == 0)
                    tile.template read_c_fragment_from_agpr<3, 0>(c_frag);
                else if (frag_m == 3 && frag_n == 1)
                    tile.template read_c_fragment_from_agpr<3, 1>(c_frag);
                else if (frag_m == 3 && frag_n == 2)
                    tile.template read_c_fragment_from_agpr<3, 2>(c_frag);
                else
                    tile.template read_c_fragment_from_agpr<3, 3>(c_frag);
                int32_t vrow =
                    min(16, tile_m_lim - warp_base_m - frag_m * (int32_t) GemmTile::MFMA_SIZE_M);
                int32_t vcol =
                    min(16, tile_n_lim - warp_base_n - frag_n * (int32_t) GemmTile::MFMA_SIZE_N);
                if (vrow > 0 && vcol > 0)
                    store_one_fragment(frag_m, frag_n, c_frag, vrow, vcol);
            }
        }
    }
#endif // __gfx950__
}

} // namespace turbo
} // namespace primus_turbo
