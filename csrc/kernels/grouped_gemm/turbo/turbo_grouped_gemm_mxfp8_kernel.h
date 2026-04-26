// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

// Reuse GEMM_Tile_MXFP8_NT_*, preshuffle_scale_16x4_kernel, and BufferSRD helpers
// from the single-GEMM kernel.
#include "../../gemm/turbo/turbo_gemm_mxfp8_kernel.h"

namespace primus_turbo {
namespace turbo {

// ── Per-block group lookup ──
//
// Walk the prefix sum of per-group tile counts (M_g + 255) / 256 to find which
// group this tile belongs to and its row index within that group.  Returns
// false if this is a padding tile (flat_tile_idx >= total active tiles).
__device__ __forceinline__ bool resolve_grouped_tile(const int64_t *group_lens_ptr,
                                                     const int32_t  group_num,
                                                     const int32_t pid_m_flat, int32_t &group_id,
                                                     int32_t &pid_m_local, int32_t &M_g) {
    int32_t cum_tiles = 0;
#pragma unroll 1
    for (int32_t g = 0; g < group_num; ++g) {
        const int32_t M_iter =
            __builtin_amdgcn_readfirstlane(static_cast<int32_t>(group_lens_ptr[g]));
        const int32_t tiles_g = (M_iter + 255) / 256;
        if (pid_m_flat < cum_tiles + tiles_g) {
            group_id    = g;
            pid_m_local = pid_m_flat - cum_tiles;
            M_g         = M_iter;
            return true;
        }
        cum_tiles += tiles_g;
    }
    return false; // padding tile
}

// ── Per-tile compute body (shared between flat and persistent kernels) ──
//
// Given a resolved tile (group_id, pid_m_local in tiles, pid_n_idx in tiles)
// computes one 256x256 output tile.  Caller is responsible for all smem
// allocation and resolution of group_id from the persistent loop.
template <typename GemmTile, typename AType, typename BType, typename CType>
__device__ __forceinline__ void turbo_grouped_gemm_mxfp8_compute_tile(
    GemmTile &tile, typename GemmTile::ASmemSubtile (*a_smem_tile)[4],
    typename GemmTile::BSmemSubtile (*b_smem_tile)[4],
    typename GemmTile::AScaleSmemSubtile (*a_s_smem_tile)[4],
    typename GemmTile::BScaleSmemSubtile (*b_s_smem_tile)[4], const AType *a_ptr,
    const BType *b_ptr, const uint32_t *a_s_ptr, const uint32_t *b_s_ptr, CType *c_ptr,
    const int64_t *group_offs_ptr, const int32_t group_id, const int32_t pid_m_local,
    const int32_t pid_n_idx, const int32_t M_g, const uint32_t n, const uint32_t k) {
    const int32_t pid_m = pid_m_local * 256;
    const int32_t pid_n = pid_n_idx * 256;
    if (pid_m >= M_g || pid_n >= (int32_t) n)
        return;

    const uint32_t lane_id = tile.lane_id;
    const uint32_t warp_id = tile.warp_id;
    const uint32_t warp_m  = tile.warp_m;
    const uint32_t warp_n  = tile.warp_n;

    const uint32_t scale_cols = (k + GemmTile::MX_BLOCK_SIZE - 1) / GemmTile::MX_BLOCK_SIZE;
    // Force scalar (SGPR) load of group_offs_ptr[group_id]: group_id is uniform
    // across the block but compiler may emit a vector load (vmcnt-tracked) if
    // it can't prove it.  readfirstlane forces the s_load path (lgkmcnt-tracked,
    // matches the wait sequence below).
    const uint32_t group_offset_lo = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(group_offs_ptr[group_id] & 0xffffffffULL));
    const uint32_t group_offset_hi = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(static_cast<uint64_t>(group_offs_ptr[group_id]) >> 32));
    const int64_t group_offset =
        (static_cast<int64_t>(group_offset_hi) << 32) | static_cast<int64_t>(group_offset_lo);
    const int64_t m_global     = group_offset + (int64_t) pid_m;
    const int64_t b_group_off  = (int64_t) group_id * (int64_t) n * (int64_t) k;
    const int64_t bs_group_off = (int64_t) group_id * (int64_t) n * (int64_t) scale_cols;

    const AType    *a_base_ptr   = a_ptr + m_global * k;
    const BType    *b_base_ptr   = b_ptr + b_group_off + (int64_t) pid_n * k;
    const uint32_t *a_s_base_ptr = a_s_ptr + m_global * scale_cols;
    const uint32_t *b_s_base_ptr = b_s_ptr + bs_group_off + (int64_t) pid_n * scale_cols;

    uint32_t ldg_offsets[2];
    tile.compute_ldg_offsets(ldg_offsets, k);
    uint32_t sts_offsets[2];
    tile.compute_sts_offsets(sts_offsets);
    uint32_t lds_offsets[2];
    tile.compute_lds_offsets(lds_offsets);
    const uint32_t scale_ldg_offset = lane_id;
    const uint32_t scale_sts_offset = lane_id;
    const uint32_t scale_lds_offset = lane_id;

    const uint32_t  a_remaining  = ((uint32_t) M_g - pid_m) * k * sizeof(AType);
    const uint32_t  b_remaining  = (n - pid_n) * k * sizeof(BType);
    const uint32_t  as_remaining = ((uint32_t) M_g - pid_m) * scale_cols * sizeof(uint32_t);
    const uint32_t  bs_remaining = (n - pid_n) * scale_cols * sizeof(uint32_t);
    const BufferSRD a_srd(a_base_ptr, a_remaining);
    const BufferSRD b_srd(b_base_ptr, b_remaining);
    const BufferSRD a_s_srd(a_s_base_ptr, as_remaining);
    const BufferSRD b_s_srd(b_s_base_ptr, bs_remaining);

    constexpr int32_t DATA_STRIDE  = GemmTile::BLOCK_SIZE_K;
    constexpr int32_t SCALE_STRIDE = GemmTile::SCALE_FRAG_SIZE * sizeof(uint32_t);

    // ── Load tile 0 → smem[0], tile 1 → smem[1] ──
    tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_a_gmem_to_smem_half_srd<1>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd<1>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);
    tile.template load_a_scale_gmem_to_smem_half_srd<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);
    tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);
    tile.template load_b_scale_gmem_to_smem_half_srd<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);

    tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_a_gmem_to_smem_half_srd<1>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<1>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);

    tile.zero_c_agpr();
    wait_vmcnt<0>();
    __builtin_amdgcn_s_barrier();

    uint32_t       cur     = 0;
    uint32_t       next    = 1;
    const uint32_t k_iters = (k + GemmTile::BLOCK_SIZE_K - 1) / GemmTile::BLOCK_SIZE_K;

    // ── Prologue: issue LDS for A0/B0 ──
    GemmTile::template load_data_subtile_pinned<GemmTile::PIN_A0>(
        a_smem_tile[cur][warp_m].u32_ptr(), lds_offsets);
    GemmTile::template load_scale_subtile_pinned<GemmTile::PIN_AS0>(
        a_s_smem_tile[cur][warp_m].u32_ptr(), scale_lds_offset);
    GemmTile::template load_data_subtile_pinned<GemmTile::PIN_B0>(
        b_smem_tile[cur][warp_n].u32_ptr(), lds_offsets);
    GemmTile::template load_scale_subtile_pinned<GemmTile::PIN_BS0>(
        b_s_smem_tile[cur][warp_n].u32_ptr(), scale_lds_offset);

    if (k_iters > 2) {
        tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[cur],
                                                      sts_offsets, 2 * DATA_STRIDE);
        tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset,
                                                            a_s_smem_tile[cur], scale_sts_offset,
                                                            scale_cols, 2 * SCALE_STRIDE);
        tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[cur],
                                                      sts_offsets, 2 * DATA_STRIDE);
        tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset,
                                                            b_s_smem_tile[cur], scale_sts_offset,
                                                            scale_cols, 2 * SCALE_STRIDE);
    }
    wait_lgkmcnt<0>();

    int32_t base_data_soff[4], base_scale_soff[4];
    tile.precompute_base_soff(base_data_soff, base_scale_soff, scale_cols);

    const uint32_t sts_wb =
        __builtin_amdgcn_readfirstlane(warp_id * GemmTile::MFMA_SIZE_M * GemmTile::MFMA_SIZE_K);
    const uint32_t s_smem_off = __builtin_amdgcn_readfirstlane((warp_id * 64 + scale_sts_offset) *
                                                               (uint32_t) sizeof(uint32_t));
    const uint32_t scale_gmem_byte_off = scale_ldg_offset * (uint32_t) sizeof(uint32_t);

    // ── Main loop ──
    const uint32_t main_iters = k_iters > 3 ? k_iters - 3 : 0;
    int32_t        data_off = 2 * DATA_STRIDE, scale_off = 2 * SCALE_STRIDE;
    for (uint32_t ki = 0; ki < main_iters;
         ++ki, data_off += DATA_STRIDE, scale_off += SCALE_STRIDE) {
        const int32_t next_data_off  = data_off + DATA_STRIDE;
        const int32_t next_scale_off = scale_off + SCALE_STRIDE;

        // Phase 1: MFMA A0×B0, LDG B1→cur[2,3]
        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0, 0, 0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1>(
                b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
                b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset, b_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, b_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        // RACE FIX (lite): partial vmcnt drain + s_barrier between Phase 1 and
        // Phase 2. 5000-iter grouped stress still leaked at vmcnt<4>, so use
        // the next tighter drain without changing the rest of the pipeline.
        wait_vmcnt<3>();
        __builtin_amdgcn_s_barrier();

        // Phase 2: MFMA A0×B1, LDG A1→cur[2,3]
        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1, 0, 1,
                                                  GemmTile::PIN_A1, GemmTile::PIN_AS1>(
                a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
                a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset, a_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, a_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        // Phase 3: MFMA A1×B0, LDG A0→next[0,1]
        {
            uint32_t dm0_0 =
                __builtin_amdgcn_readfirstlane(a_smem_tile[next][0].u32_ptr()) + sts_wb;
            uint32_t dm0_1 =
                __builtin_amdgcn_readfirstlane(a_smem_tile[next][1].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[next][0].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[next][1].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A1, GemmTile::PIN_AS1,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0, 1, 0,
                                                  GemmTile::PIN_A0, GemmTile::PIN_AS0>(
                a_smem_tile[next][warp_m].u32_ptr(), lds_offsets,
                a_s_smem_tile[next][warp_m].u32_ptr(), scale_lds_offset, a_srd, ldg_offsets, dm0_0,
                dm0_1, base_data_soff[0] + next_data_off, base_data_soff[1] + next_data_off,
                a_s_srd, scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[0] + next_scale_off,
                base_scale_soff[1] + next_scale_off);
        }

        // Phase 4: MFMA A1×B1, LDG B0→next[0,1]
        {
            uint32_t dm0_0 =
                __builtin_amdgcn_readfirstlane(b_smem_tile[next][0].u32_ptr()) + sts_wb;
            uint32_t dm0_1 =
                __builtin_amdgcn_readfirstlane(b_smem_tile[next][1].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[next][0].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[next][1].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A1, GemmTile::PIN_AS1,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1, 1, 1,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0>(
                b_smem_tile[next][warp_n].u32_ptr(), lds_offsets,
                b_s_smem_tile[next][warp_n].u32_ptr(), scale_lds_offset, b_srd, ldg_offsets, dm0_0,
                dm0_1, base_data_soff[0] + next_data_off, base_data_soff[1] + next_data_off,
                b_s_srd, scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[0] + next_scale_off,
                base_scale_soff[1] + next_scale_off);
        }

        // End-of-loop barrier: original vmcnt<12>+s_barrier (race-fix sweep
        // showed full drain here is redundant once Phase 1→2 is barriered).
        wait_vmcnt<0>();
        __builtin_amdgcn_s_barrier();
        cur ^= 1;
        next ^= 1;
    }

    // ── Epilogue 1: last LDG tile — Phase 1+2 prefetch B1+A1, Phase 3+4 compute only ──
    {
        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0, 0, 0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1>(
                b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
                b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset, b_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, b_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1, 0, 1,
                                                  GemmTile::PIN_A1, GemmTile::PIN_AS1>(
                a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
                a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset, a_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, a_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 1, 0, GemmTile::PIN_A0,
                                          GemmTile::PIN_AS0>(
            a_smem_tile[next][warp_m].u32_ptr(), lds_offsets, a_s_smem_tile[next][warp_m].u32_ptr(),
            scale_lds_offset);
        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 1, 1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0>(
            b_smem_tile[next][warp_n].u32_ptr(), lds_offsets, b_s_smem_tile[next][warp_n].u32_ptr(),
            scale_lds_offset);

        // RACE FIX: drain Epi1 buffer_load_lds (B1+A1 prefetch) and pending
        // ds_reads before flipping the double buffer.
        wait_vmcnt<0>();
        wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();
        cur ^= 1;
        next ^= 1;
    }

    // ── Epilogue 2: no LDG, LDS from both buffers ──
    {
        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 0, 0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1>(
            b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
            b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset);

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 0, 1, GemmTile::PIN_A1,
                                          GemmTile::PIN_AS1>(
            a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
            a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset);

        wait_vmcnt<0>();
        wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 1, 0, GemmTile::PIN_A0,
                                          GemmTile::PIN_AS0>(
            a_smem_tile[next][warp_m].u32_ptr(), lds_offsets, a_s_smem_tile[next][warp_m].u32_ptr(),
            scale_lds_offset);

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 1, 1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0>(
            b_smem_tile[next][warp_n].u32_ptr(), lds_offsets, b_s_smem_tile[next][warp_n].u32_ptr(),
            scale_lds_offset);

        cur ^= 1;
        next ^= 1;
    }

    // ── Epilogue 3: no LDG, no LDS from next ──
    {
        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 0, 0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1>(
            b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
            b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset);

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 0, 1, GemmTile::PIN_A1,
                                          GemmTile::PIN_AS1>(
            a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
            a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset);

        GemmTile::template phase_mfma_only<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B0,
                                           GemmTile::PIN_BS0, 1, 0>();
        GemmTile::template phase_mfma_only<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B1,
                                           GemmTile::PIN_BS1, 1, 1>();
    }

    // ── Store C ──
    asm volatile("s_nop 7\ns_nop 7\ns_nop 7\ns_nop 7" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
    uint32_t c_stg_offsets[4];
    tile.compute_stg_offsets(c_stg_offsets);
    CType *c_stg_base_ptr =
        c_ptr + m_global * (int64_t) n + pid_n + warp_id / 2 * 64 * (int64_t) n + warp_id % 2 * 64;
    const bool is_boundary_tile = (pid_m + 256 > M_g) || (pid_n + 256 > (int32_t) n);

    if (!is_boundary_tile) {
        float32x4 c_tmp[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) n + 0 * 128, n, c_tmp,
                             c_stg_offsets);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) n + 1 * 128, n, c_tmp,
                             c_stg_offsets);
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) n + 0 * 128, n, c_tmp,
                             c_stg_offsets);
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) n + 1 * 128, n, c_tmp,
                             c_stg_offsets);
    } else {
        const int32_t warp_base_m  = warp_id / 2 * 64;
        const int32_t warp_base_n  = warp_id % 2 * 64;
        const int32_t tile_valid_m = min(M_g - pid_m, 256) - warp_base_m;
        const int32_t tile_valid_n = min((int32_t) n - pid_n, 256) - warp_base_n;
        float32x4     c_tmp[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) n + 0 * 128, n, c_tmp,
                             c_stg_offsets, tile_valid_m, tile_valid_n);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) n + 1 * 128, n, c_tmp,
                             c_stg_offsets, tile_valid_m, tile_valid_n - 128);
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) n + 0 * 128, n, c_tmp,
                             c_stg_offsets, tile_valid_m - 128, tile_valid_n);
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) n + 1 * 128, n, c_tmp,
                             c_stg_offsets, tile_valid_m - 128, tile_valid_n - 128);
    }
    // End-of-tile drain.  Originally for persistent-kernel reuse of smem/VGPRs;
    // persistent kernel was dropped (3ac1a42), but empirically removing this
    // makes the flat-kernel race rate ~2x worse, so it is load-bearing for
    // wave-retire ordering of in-flight stores.
    wait_vmcnt<0>();
    wait_lgkmcnt<0>();
    __builtin_amdgcn_s_barrier();
}

// ── Grouped MXFP8 NT GEMM Kernel — flat grid (256x256x128, 4-warp, GFX950) ──
//
//   A: [total_M, K] FP8           (groups concatenated along M)
//   B: [group_num, N, K] FP8      (per-group weight)
//   C: [total_M, N] FP16/BF16
//   group_lens: [group_num] int64 (M_g per group, sum = total_M)
//   group_offs: [group_num+1] int64
//
// Grid: (max_g ceil(M_g/256), ceil(N/256), group_num).
// Each block processes ONE 256x256 tile.  Per-group padding tiles early-exit.
template <typename AType, typename BType, typename CType, typename AccType = float>
__global__
__launch_bounds__(256, 1) void turbo_grouped_gemm_mxfp8_256x256x128_16x16x128_4wave_kernel(
    const AType *a_ptr, const BType *b_ptr, const uint32_t *a_s_ptr, const uint32_t *b_s_ptr,
    CType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const int32_t group_num, const uint32_t n, const uint32_t k) {
#if !defined(__gfx950__)
    assert(false && "turbo_grouped_gemm_mxfp8 kernel requires gfx950");
    return;
#else
    const int32_t group_id = (int32_t) blockIdx.z;
    if (group_id >= group_num) {
        return;
    }
    const int32_t M_g =
        __builtin_amdgcn_readfirstlane(static_cast<int32_t>(group_lens_ptr[group_id]));
    const int32_t pid_m_local = (int32_t) blockIdx.x;
    if (M_g <= 0 || pid_m_local * 256 >= M_g)
        return;

    using GemmTile =
        GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950<AType, BType, CType, AccType>;
    GemmTile tile(threadIdx.x, (uint32_t) M_g, n, k);
    tile.reserve_pinned_regs();

    using ASmem                       = typename GemmTile::ASmemSubtile;
    using BSmem                       = typename GemmTile::BSmemSubtile;
    using ASSmem                      = typename GemmTile::AScaleSmemSubtile;
    using BSSmem                      = typename GemmTile::BScaleSmemSubtile;
    constexpr size_t SMEM_DATA_BYTES  = sizeof(ASmem) * 2 * 4 + sizeof(BSmem) * 2 * 4;
    constexpr size_t SMEM_SCALE_BYTES = sizeof(ASSmem) * 2 * 4 + sizeof(BSSmem) * 2 * 4;
    __shared__ char  smem_buf[SMEM_DATA_BYTES + SMEM_SCALE_BYTES];
    auto            *a_smem_tile = reinterpret_cast<ASmem(*)[4]>(smem_buf);
    auto            *b_smem_tile = reinterpret_cast<BSmem(*)[4]>(smem_buf + sizeof(ASmem) * 2 * 4);
    auto            *a_s_smem_tile = reinterpret_cast<ASSmem(*)[4]>(smem_buf + SMEM_DATA_BYTES);
    auto            *b_s_smem_tile =
        reinterpret_cast<BSSmem(*)[4]>(smem_buf + SMEM_DATA_BYTES + sizeof(ASSmem) * 2 * 4);

    turbo_grouped_gemm_mxfp8_compute_tile<GemmTile, AType, BType, CType>(
        tile, a_smem_tile, b_smem_tile, a_s_smem_tile, b_s_smem_tile, a_ptr, b_ptr, a_s_ptr,
        b_s_ptr, c_ptr, group_offs_ptr, group_id, pid_m_local, (int32_t) blockIdx.y, M_g, n, k);
#endif
}

} // namespace turbo
} // namespace primus_turbo
