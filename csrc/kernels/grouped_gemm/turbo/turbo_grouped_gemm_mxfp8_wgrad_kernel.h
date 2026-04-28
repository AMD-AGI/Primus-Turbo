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

    // CORRECTNESS FIX (variable-K wgrad with M_g == 128 groups):
    //
    // Tile 1's prologue LDG reads cols [DATA_STRIDE, 2*DATA_STRIDE) of the
    // current group via SRDs scoped to (rows, total_m).  When this group's
    // M_g == DATA_STRIDE (k_iters == 1), those cols belong to the *next*
    // group (or padding) and the loaded fp8 bytes / E8M0 scales are
    // semantically garbage.  Epilogue 2 phases 3-4 then issue ds_reads
    // from smem[1] into PIN_A0/PIN_B0, and Epilogue 3 MFMAs A0×B0,
    // A0×B1, A1×B0, A1×B1 entirely on those corrupted register values
    // — i.e., exactly half of the 8 MFMAs that contribute to dB are
    // bogus.  This corrupts dB hard enough that
    // tests/pytorch/ops/test_grouped_gemm_fp8.py::test_grouped_gemm_fp8_mx_blockwise
    // sees b_grad_snr collapse from ~30+ to ~6 on every M=256, balance=False
    // case (B in {2,4,8}) where the random padded distribution lands at
    // least one group with M_g == 128.  This explains the 48 pre-existing
    // failures observed since round 5.
    //
    // Mitigation: for tile 1 only, swap to a clamped SRD whose num_records
    // is 0 when k_iters < 2.  buffer_load to a 0-record SRD returns 0 for
    // every element, the LDS stores then write zeros into smem[1], and
    // every MFMA in epilogue 3 (and the cross-buffer ds_reads in epilogue
    // 2 phases 3-4) sources zero data — yielding zero MFMA contribution
    // regardless of the scale ROM.  We deliberately leave the scale SRDs
    // unclamped; the scale SMEM may hold next-group bytes but with data
    // forced to zero the MFMA result is zero independent of the scale
    // exponent.  For k_iters >= 2 the SRDs are unchanged so the existing
    // pipeline is bit-identical.
    // K-loop iterates over the M_g reduction dimension, NOT over total_m.
    const uint32_t k_iters = (M_g + GemmTile::BLOCK_SIZE_K - 1) / GemmTile::BLOCK_SIZE_K;
    const uint32_t a_t1_remaining = (k_iters >= 2) ? lhs_remaining : 0u;
    const uint32_t b_t1_remaining = (k_iters >= 2) ? rhs_remaining : 0u;
    const BufferSRD a_srd_t1(lhs_base_ptr, a_t1_remaining);
    const BufferSRD b_srd_t1(rhs_base_ptr, b_t1_remaining);

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

    tile.template load_a_gmem_to_smem_half_srd<0>(a_srd_t1, ldg_offsets, a_smem_tile[1],
                                                  sts_offsets, DATA_STRIDE);
    tile.template load_a_gmem_to_smem_half_srd<1>(a_srd_t1, ldg_offsets, a_smem_tile[1],
                                                  sts_offsets, DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<0>(b_srd_t1, ldg_offsets, b_smem_tile[1],
                                                  sts_offsets, DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<1>(b_srd_t1, ldg_offsets, b_smem_tile[1],
                                                  sts_offsets, DATA_STRIDE);
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

    // ── Prologue: issue LDS for A0/B0 ──
    GemmTile::template load_data_subtile_pinned<GemmTile::PIN_A0>(
        a_smem_tile[cur][warp_m].u32_ptr(), lds_offsets);
    GemmTile::template load_scale_subtile_pinned<GemmTile::PIN_AS0>(
        a_s_smem_tile[cur][warp_m].u32_ptr(), scale_lds_offset);
    GemmTile::template load_data_subtile_pinned<GemmTile::PIN_B0>(
        b_smem_tile[cur][warp_n].u32_ptr(), lds_offsets);
    GemmTile::template load_scale_subtile_pinned<GemmTile::PIN_BS0>(
        b_s_smem_tile[cur][warp_n].u32_ptr(), scale_lds_offset);

    // PERF: drain ONLY the prologue ds_reads here (24 LDS reads that fill
    // PIN_A0/AS0/B0/BS0 for the main loop's first MFMA).  The tile-2
    // prefetch `buffer_load_lds` below targets a_smem[cur][0,1] /
    // b_smem[cur][0,1] (warp-specific 2KB chunks) which the main loop's
    // Phase 1 does NOT read — Phase 1 consumes the PIN_* regs filled above
    // and issues new ds_reads from b_smem[cur][warp_n+2] (the OPPOSITE
    // half).  The tile-2 LDG's LDS-write side will be drained by Phase 1's
    // existing WAR barrier (`wait_lgkmcnt<0>; s_barrier` inside
    // `phase_mfma_lds_ldg`).
    //
    // Mirrors the same reorder applied to the FWD persistent kernel; lets
    // the ~150-cycle GMEM-to-LDS latency overlap with the 6 MFMAs that
    // Phase 1 fires before its WAR barrier (~96 cycles of compute), so the
    // wait_lgkmcnt at this point shrinks from a buffer_load_lds-bound
    // ~150 cycles to a ds_read-bound ~5-10 cycles per output tile.
    wait_lgkmcnt<0>();

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
        //
        // The previous wait_lgkmcnt<0> here is redundant and has been removed:
        // phase_mfma_lds_ldg (used for Phase 1) internally issues
        // `wait_lgkmcnt<0>+__builtin_amdgcn_s_barrier()` between its ds_reads
        // and its buffer_load_lds (the WAR barrier inside the phase, see
        // turbo_gemm_mxfp8_kernel.h ~L477), so by Phase 1's exit lgkmcnt is
        // already 0 with no pending ds_reads.  Dropping the redundant call
        // matches the FWD persistent kernel's Phase 1->2 barrier (which has
        // never used wait_lgkmcnt here) and removes a per-K-iter compiler
        // scheduler `"memory"`-clobber barrier on the variable-K wgrad hot
        // path, freeing the scheduler to fold the readfirstlane/SGPR work for
        // Phase 2's source operands into the vmcnt drain window.
        wait_vmcnt<0>();
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

        // No outer barrier between Phase 2 and Phase 3, matching the FWD main-loop
        // pipeline.  Justification:
        //   - phase_mfma_lds_ldg internally does wait_lgkmcnt<0>+s_barrier between
        //     its ds_reads and buffer_load_lds, so all waves are sync'd at exit;
        //   - Phase 2 writes a_smem[cur][2,3]; Phase 3 reads a_smem[next][warp_m]
        //     (different ping-pong buffer), no LDS WAR;
        //   - the prior Phase 1->2 wait_vmcnt<0> drained pre-Phase-2 LDGs, so
        //     vmcnt entering Phase 3 is bounded by Phase 2's 6 in-flight LDGs.
        // Removes one drain+barrier per K-iter on the variable-K wgrad hot path.

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

        // Drain Epi1 buffer_load_lds (B1+A1 prefetch) before flipping the
        // double buffer.  wait_lgkmcnt<0> previously here is redundant:
        // phase_mfma_lds (used for Phase 3+4 of this Epi1) ends with its own
        // `wait_lgkmcnt<0>` (see turbo_gemm_mxfp8_kernel.h ~L414), so by the
        // time we hit this drain lgkmcnt is already 0 with no pending
        // ds_reads.  Mirror the FWD inner-loop barrier idiom of vmcnt-only
        // drain + s_barrier.
        wait_vmcnt<0>();
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

        // PERF: drop the previously-here `wait_vmcnt<0>()` because it is
        // provably a no-op at this program point.  vmcnt is drained by the
        // `wait_vmcnt<0>()` at the end of Epi1 (~L395), and Epi2 phases 1+2
        // are both `phase_mfma_lds` (no `buffer_load_lds`, hence no vmem
        // ops issued in this region).  The `s_barrier()` is preserved so
        // waves stay in step before Phase 3 reads from the `next`
        // (cross-buffer) LDS region.  Mirrors the same change in the
        // forward persistent kernel; per-tile saving compounds across the
        // wgrad persistent loop's `grid_n * grid_k * group_num` tiles.
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

    // Wgrad C-store: mirror the FWD epilogue layout (cfd2616 + 2a1943c):
    //   * non-volatile stores (`store_c_subtile<false>`) so the compiler
    //     coalesces flat_store_short_d16_hi and skips the per-element
    //     `s_waitcnt vmcnt(0)` drains that volatile would emit.  Drain once
    //     at the end with an explicit `wait_vmcnt<0>()`.
    //   * two alternating `c_tmp_a` / `c_tmp_b` buffers so adjacent
    //     (read_c, store_c) pairs always land in disjoint VGPR slots and
    //     pair-(i+1)'s `v_accvgpr_read_b32` cannot clobber pair-i's
    //     in-flight non-volatile stores.  Without this split, register
    //     coalescing reuses the same 64 VGPRs for all four pairs and the
    //     `flat_store_short` from pair-i still latches its source registers
    //     while pair-(i+1)'s reads are already writing to them — the same
    //     LLVM-allocator behaviour that drove FWD `out` race down from
    //     2-6/100 to 0-2/100 once split (see 2a1943c).
    //   * `sched_barrier(0)` between adjacent pairs to keep them pinned in
    //     program order, matching FWD epilogue (df78f4a).
    //
    // The earlier comment ("switching to non-volatile gives marginal perf
    // and increases dB races") was measured BEFORE the alternating buffers
    // were introduced and is no longer accurate: with alternating buffers
    // the register-reuse race is closed, so wgrad can take the same
    // C-store throughput win as FWD without a stress regression.
    //
    // RACE FIX (mirror of FWD): insert wait_vmcnt<63> immediately before
    // each c_tmp reuse so the older pair's flat_store_short have committed
    // (and thus released their source VGPRs) before the new pair's
    // accvgpr_read overwrites them.  pair-1 issues 64 stores → vmcnt up
    // to 128 (with pair-0's 64 still pending); wait_vmcnt<63>() blocks
    // until vmcnt≤63, i.e., 65 commits = pair-0's 64 + 1 of pair-1's =
    // pair-0 fully drained, c_tmp_a safe to overwrite.  63 is the gfx950
    // assembler max for the vmcnt field.  Empirically near-zero cost
    // because pair-1's ~80-cycle read+store window already overlaps
    // most of pair-0's commit latency to L1.
    if (!is_boundary_tile) {
        float32x4 c_tmp_a[4][4];
        float32x4 c_tmp_b[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) k + 0 * 128, k,
                                             c_tmp_a, c_stg_offsets);
        __builtin_amdgcn_sched_barrier(0);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) k + 1 * 128, k,
                                             c_tmp_b, c_stg_offsets);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) k + 0 * 128, k,
                                             c_tmp_a, c_stg_offsets);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) k + 1 * 128, k,
                                             c_tmp_b, c_stg_offsets);
    } else {
        const int32_t warp_base_m  = warp_id / 2 * 64;
        const int32_t warp_base_n  = warp_id % 2 * 64;
        const int32_t tile_valid_m = min((int32_t) n - pid_n, 256) - warp_base_m;
        const int32_t tile_valid_n = min((int32_t) k - pid_k, 256) - warp_base_n;
        float32x4     c_tmp_a[4][4];
        float32x4     c_tmp_b[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) k + 0 * 128, k,
                                             c_tmp_a, c_stg_offsets, tile_valid_m, tile_valid_n);
        __builtin_amdgcn_sched_barrier(0);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) k + 1 * 128, k,
                                             c_tmp_b, c_stg_offsets, tile_valid_m,
                                             tile_valid_n - 128);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) k + 0 * 128, k,
                                             c_tmp_a, c_stg_offsets, tile_valid_m - 128,
                                             tile_valid_n);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) k + 1 * 128, k,
                                             c_tmp_b, c_stg_offsets, tile_valid_m - 128,
                                             tile_valid_n - 128);
    }
    // PERF: drop the previously-here trailing `wait_vmcnt<0>()`.  Same
    // reasoning as the FWD kernel's matching site (see
    // turbo_grouped_gemm_mxfp8_kernel.h, end of `compute_tile`):
    //   1. Next persistent-loop iteration's prologue `wait_vmcnt<0>();
    //      s_barrier();` (around L184) drains any in-flight dB stores
    //      before the new iteration's ds_reads consume the freshly issued
    //      buffer_load_lds writes — same vmem counter, same wave, so the
    //      drain cost is paid exactly once per tile boundary either way.
    //   2. The persistent-loop CTA-level `s_barrier()` carries an implicit
    //      `"memory"` clobber, preventing the compiler from sinking
    //      C-stores past the tile boundary.
    //   3. Kernel exit (`s_endpgm`) blocks until outstanding VMEM ops
    //      retire, so the last tile's dB writes are guaranteed visible
    //      after `hipStreamSynchronize` / `hipDeviceSynchronize` — no
    //      explicit fence needed at the very end of the persistent loop.
    // Each output tile's dB slice (slice `[g, k_global..+256,
    // m_global..+256]`) is exclusively owned by this CTA, so removing the
    // fence cannot create cross-tile aliasing.
    // The pre-existing dB sub-tile race (seen with the c_tmp_a/c_tmp_b
    // register-rename scheme on shape G=4 M=128 N=2048 K=2048) is gated
    // by the `wait_vmcnt<63>()` calls inside the c-subtile-store sequence
    // above and is not affected by this trailing-wait removal.
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
        // CRITICAL: do NOT remove this CTA-level s_barrier.  Same rationale
        // as the forward grouped MXFP8 persistent kernel: the per-tile
        // wgrad_compute_tile() ends without an explicit cross-tile sync,
        // and the next tile's prologue would otherwise start overwriting
        // a/b/scale SMEM ping-pong banks that the previous tile's epilogue
        // mfma_lds sequence may still be sourcing from.  Removing this
        // barrier (verified during the df78f4a follow-up) opens a wave-skew
        // SMEM race that surfaces as dB drift on top of the existing wgrad
        // C-store race.
        __builtin_amdgcn_s_barrier();
    }
#endif
}

} // namespace turbo
} // namespace primus_turbo
