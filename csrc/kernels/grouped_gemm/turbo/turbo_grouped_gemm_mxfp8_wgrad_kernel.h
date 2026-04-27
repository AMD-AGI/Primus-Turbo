// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Variable-K MXFP8 grouped GEMM kernel for the wgrad (dB) path.
//
// Reduction dimension is M_g (variable per group); N and K are fixed across
// groups (they are the weight matrix dims).  Each workgroup computes one
// (group, n_tile, k_tile) 256×256 tile of dB.
//
// Forward computes:   C[g] = A[g] @ B[g]^T   shape (M_g, N)
// Wgrad computes:     dB[g] = dC[g]^T @ A[g] shape (N, K)
//
// We rephrase wgrad as an NT GEMM to reuse the forward kernel's tile primitives:
//   dB[g] = NT(dC[g]^T, A[g]^T)  where NT(L, R) = L @ R^T
//   - LHS (dC[g]^T): shape (N, M_g)
//   - RHS (A[g]^T):  shape (K, M_g)
//   - reduction is along M_g (the last/contiguous dim of both inputs)
//
// The flat memory layout of the inputs (after col-wise MX quantization of the
// original dC and A):
//   dC^T_full: shape (N, total_M)             — col-quant of dC (total_M, N)
//   A^T_full:  shape (K, total_M)             — col-quant of A (total_M, K)
// For group g, slice columns [group_offs[g], group_offs[g+1]).

#pragma once

// Reuse GemmTile + preshuffle_scale_16x4_kernel + BufferSRD helpers.
#include "../../gemm/turbo/turbo_gemm_mxfp8_kernel.h"

namespace primus_turbo {
namespace turbo {

// ── Per-tile compute body for wgrad ──
//
// LHS rows (N) are the "M" axis of the underlying tile primitives.
// RHS rows (K) are the "N" axis of the underlying tile primitives.
// Reduction dim (M_g) is the "K" axis.  Row stride of both LHS and RHS in
// gmem is `total_m` (cols of the flat (N, total_M) and (K, total_M) tensors).
template <typename GemmTile, typename AType, typename BType, typename CType>
__device__ __forceinline__ void turbo_grouped_gemm_mxfp8_wgrad_compute_tile(
    GemmTile &tile, typename GemmTile::ASmemSubtile (*a_smem_tile)[4],
    typename GemmTile::BSmemSubtile (*b_smem_tile)[4],
    typename GemmTile::AScaleSmemSubtile (*a_s_smem_tile)[4],
    typename GemmTile::BScaleSmemSubtile (*b_s_smem_tile)[4], const AType *lhs_ptr,
    const BType *rhs_ptr, const uint32_t *lhs_s_ptr, const uint32_t *rhs_s_ptr, CType *db_ptr,
    const int64_t *group_offs_ptr, const int32_t group_id, const int32_t pid_n_local,
    const int32_t pid_k_local, const int32_t M_g, const uint32_t n, const uint32_t k,
    const uint32_t total_m) {
    const int32_t pid_n = pid_n_local * 256;  // along N (output dim 0)
    const int32_t pid_k = pid_k_local * 256;  // along K (output dim 1)
    if (pid_n >= (int32_t) n || pid_k >= (int32_t) k)
        return;

    const uint32_t lane_id = tile.lane_id;
    const uint32_t warp_id = tile.warp_id;
    const uint32_t warp_m  = tile.warp_m;  // along output dim 0 (N here)
    const uint32_t warp_n  = tile.warp_n;  // along output dim 1 (K here)

    const uint32_t scale_cols_full = (total_m + GemmTile::MX_BLOCK_SIZE - 1) / GemmTile::MX_BLOCK_SIZE;

    // Force scalar load of group_offs_ptr[group_id]; mirrors the forward kernel.
    const uint32_t group_offset_lo = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(group_offs_ptr[group_id] & 0xffffffffULL));
    const uint32_t group_offset_hi = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(static_cast<uint64_t>(group_offs_ptr[group_id]) >> 32));
    const int64_t group_col_offset =
        (static_cast<int64_t>(group_offset_hi) << 32) | static_cast<int64_t>(group_offset_lo);
    // Scale tensors are PRE-SHUFFLED into (16-row × 4-col) blocks of 64 uint32
    // each.  Skipping `group_col_offset` raw FP8 cols means skipping
    // `group_col_offset / 128` col-blocks, each 64 uint32 elements:
    //   preshuffled_offset_elements = (group_col_offset / 128) * 64
    //                               = group_col_offset / 2
    // This requires `group_col_offset` (and hence per-group M_g) to be a
    // multiple of 128 — enforced by the wrapper.
    const int64_t group_scale_offset = group_col_offset / 2;

    // Base pointers (point to start of this workgroup's (256,256) row range
    // and this group's column range within the flat (rows, total_m) tensor).
    const AType    *lhs_base_ptr   = lhs_ptr + (int64_t) pid_n * total_m + group_col_offset;
    const BType    *rhs_base_ptr   = rhs_ptr + (int64_t) pid_k * total_m + group_col_offset;
    const uint32_t *lhs_s_base_ptr = lhs_s_ptr + (int64_t) pid_n * scale_cols_full + group_scale_offset;
    const uint32_t *rhs_s_base_ptr = rhs_s_ptr + (int64_t) pid_k * scale_cols_full + group_scale_offset;

    uint32_t ldg_offsets[2];
    tile.compute_ldg_offsets(ldg_offsets, total_m);  // stride = total_m
    uint32_t sts_offsets[2];
    tile.compute_sts_offsets(sts_offsets);
    uint32_t lds_offsets[2];
    tile.compute_lds_offsets(lds_offsets);
    const uint32_t scale_ldg_offset = lane_id;
    const uint32_t scale_sts_offset = lane_id;
    const uint32_t scale_lds_offset = lane_id;

    // SRD remaining bounds.  LHS valid bytes = (N - pid_n) rows × total_m cols × sizeof(AType).
    const uint32_t lhs_remaining =
        ((n - pid_n) * total_m - static_cast<uint32_t>(group_col_offset)) * sizeof(AType);
    const uint32_t rhs_remaining =
        ((k - pid_k) * total_m - static_cast<uint32_t>(group_col_offset)) * sizeof(BType);
    const uint32_t lhs_s_remaining =
        ((n - pid_n) * scale_cols_full - static_cast<uint32_t>(group_scale_offset)) *
        sizeof(uint32_t);
    const uint32_t rhs_s_remaining =
        ((k - pid_k) * scale_cols_full - static_cast<uint32_t>(group_scale_offset)) *
        sizeof(uint32_t);
    const BufferSRD a_srd(lhs_base_ptr, lhs_remaining);
    const BufferSRD b_srd(rhs_base_ptr, rhs_remaining);
    const BufferSRD a_s_srd(lhs_s_base_ptr, lhs_s_remaining);
    const BufferSRD b_s_srd(rhs_s_base_ptr, rhs_s_remaining);

    constexpr int32_t DATA_STRIDE  = GemmTile::BLOCK_SIZE_K;  // 128
    constexpr int32_t SCALE_STRIDE = GemmTile::SCALE_FRAG_SIZE * sizeof(uint32_t);

    // ── Load tile 0 → smem[0], tile 1 → smem[1] ──
    tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_a_gmem_to_smem_half_srd<1>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd<1>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols_full);
    tile.template load_a_scale_gmem_to_smem_half_srd<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols_full);
    tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols_full);
    tile.template load_b_scale_gmem_to_smem_half_srd<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols_full);

    tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_a_gmem_to_smem_half_srd<1>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<1>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols_full,
                                                        SCALE_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols_full,
                                                        SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols_full,
                                                        SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols_full,
                                                        SCALE_STRIDE);

    tile.zero_c_agpr();
    wait_vmcnt<0>();
    __builtin_amdgcn_s_barrier();

    uint32_t       cur     = 0;
    uint32_t       next    = 1;
    // K-loop iterates over the M_g reduction dimension, NOT over total_m.
    const uint32_t k_iters = (M_g + GemmTile::BLOCK_SIZE_K - 1) / GemmTile::BLOCK_SIZE_K;

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
                                                            scale_cols_full, 2 * SCALE_STRIDE);
        tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[cur],
                                                      sts_offsets, 2 * DATA_STRIDE);
        tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset,
                                                            b_s_smem_tile[cur], scale_sts_offset,
                                                            scale_cols_full, 2 * SCALE_STRIDE);
    }
    wait_lgkmcnt<0>();

    int32_t base_data_soff[4], base_scale_soff[4];
    tile.precompute_base_soff(base_data_soff, base_scale_soff, scale_cols_full);

    const uint32_t sts_wb =
        __builtin_amdgcn_readfirstlane(warp_id * GemmTile::MFMA_SIZE_M * GemmTile::MFMA_SIZE_K);
    const uint32_t s_smem_off = __builtin_amdgcn_readfirstlane((warp_id * 64 + scale_sts_offset) *
                                                               (uint32_t) sizeof(uint32_t));
    const uint32_t scale_gmem_byte_off = scale_ldg_offset * (uint32_t) sizeof(uint32_t);

    // ── Main loop ── (identical to forward; only base_ptr & strides differ)
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

        // RACE FIX: wgrad uses the same 4-phase LDS/LDG pipeline as forward.
        // The earlier lite vmcnt<4> drain was not enough for long-K single GEMM,
        // so keep wgrad on the conservative full vmcnt drain before Phase 2.
        wait_vmcnt<0>();
        wait_lgkmcnt<0>();
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

        // Wgrad's transposed access pattern showed rare drift after Phase 2.
        // Drain both memory counters before Phase 3 reuses the opposite buffer.
        wait_vmcnt<0>();
        wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();

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

        wait_vmcnt<12>();
        __builtin_amdgcn_s_barrier();
        cur ^= 1;
        next ^= 1;
    }

    // ── Epilogue 1: last LDG tile ──
    if (k_iters > 2) {
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

    // ── Store dB[group_id][pid_n:pid_n+256, pid_k:pid_k+256] ──
    __builtin_amdgcn_sched_barrier(0);
    uint32_t c_stg_offsets[4];
    tile.compute_stg_offsets(c_stg_offsets);
    // dB layout: (G, N, K) row-major.  Per-group offset = group_id * N * K.
    CType *db_group_ptr = db_ptr + (int64_t) group_id * n * k;
    CType *c_stg_base_ptr = db_group_ptr + (int64_t) pid_n * k + pid_k +
                            warp_id / 2 * 64 * (int64_t) k + warp_id % 2 * 64;
    const bool is_boundary_tile = (pid_n + 256 > (int32_t) n) || (pid_k + 256 > (int32_t) k);

    // Wgrad keeps the default volatile C-store path (see store_c_subtile in
    // turbo_gemm_mxfp8_kernel.h): switching to non-volatile gives marginal
    // perf here (the wgrad kernel processes many more tiles per CTA than FWD
    // so C-store is already amortized) and measurably increases pre-existing
    // dB stress-test races on small shapes.
    if (!is_boundary_tile) {
        float32x4 c_tmp[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) k + 0 * 128, k, c_tmp,
                             c_stg_offsets);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) k + 1 * 128, k, c_tmp,
                             c_stg_offsets);
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) k + 0 * 128, k, c_tmp,
                             c_stg_offsets);
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) k + 1 * 128, k, c_tmp,
                             c_stg_offsets);
    } else {
        const int32_t warp_base_m  = warp_id / 2 * 64;
        const int32_t warp_base_n  = warp_id % 2 * 64;
        const int32_t tile_valid_m = min((int32_t) n - pid_n, 256) - warp_base_m;
        const int32_t tile_valid_n = min((int32_t) k - pid_k, 256) - warp_base_n;
        float32x4     c_tmp[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) k + 0 * 128, k, c_tmp,
                             c_stg_offsets, tile_valid_m, tile_valid_n);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * (int64_t) k + 1 * 128, k, c_tmp,
                             c_stg_offsets, tile_valid_m, tile_valid_n - 128);
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) k + 0 * 128, k, c_tmp,
                             c_stg_offsets, tile_valid_m - 128, tile_valid_n);
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * (int64_t) k + 1 * 128, k, c_tmp,
                             c_stg_offsets, tile_valid_m - 128, tile_valid_n - 128);
    }
}

template <typename AType, typename BType, typename CType, typename AccType = float>
__global__ __launch_bounds__(256, 1) void
turbo_grouped_gemm_mxfp8_wgrad_256x256x128_16x16x128_4wave_persistent_kernel(
    const AType *lhs_ptr, const BType *rhs_ptr, const uint32_t *lhs_s_ptr,
    const uint32_t *rhs_s_ptr, CType *db_ptr, const int64_t *group_lens_ptr,
    const int64_t *group_offs_ptr, const int32_t group_num, const uint32_t total_m,
    const uint32_t n, const uint32_t k, const int32_t grid_n, const int32_t grid_k) {
#if !defined(__gfx950__)
    assert(false && "turbo_grouped_gemm_mxfp8_wgrad persistent kernel requires gfx950");
    return;
#else
    using GemmTile =
        GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950<AType, BType, CType, AccType>;

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

    GemmTile reserve_tile(threadIdx.x, n, k, total_m);
    reserve_tile.reserve_pinned_regs();

    const int32_t tiles_per_group = grid_n * grid_k;
    const int32_t total_tiles     = tiles_per_group * group_num;
    for (int32_t tile_id = (int32_t) blockIdx.x; tile_id < total_tiles; tile_id += (int32_t) gridDim.x) {
        const int32_t group_id = tile_id / tiles_per_group;
        const int32_t rem      = tile_id - group_id * tiles_per_group;
        const int32_t pid_k    = rem / grid_n;
        const int32_t pid_n    = rem - pid_k * grid_n;

        const int32_t M_g =
            __builtin_amdgcn_readfirstlane(static_cast<int32_t>(group_lens_ptr[group_id]));
        if (M_g <= 0) {
            continue;
        }

        GemmTile tile(threadIdx.x, n, k, total_m);
        turbo_grouped_gemm_mxfp8_wgrad_compute_tile<GemmTile, AType, BType, CType>(
            tile, a_smem_tile, b_smem_tile, a_s_smem_tile, b_s_smem_tile, lhs_ptr, rhs_ptr,
            lhs_s_ptr, rhs_s_ptr, db_ptr, group_offs_ptr, group_id, pid_n, pid_k, M_g, n, k,
            total_m);
        __builtin_amdgcn_s_barrier();
    }
#endif
}

} // namespace turbo
} // namespace primus_turbo
