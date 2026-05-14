// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

// Reuse GEMM_Tile_MXFP8_NT_*, preshuffle_scale_16x4_kernel, and BufferSRD helpers
// from the single-GEMM kernel.
#include "../../gemm/turbo/turbo_gemm_mxfp8_kernel.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {
namespace turbo {

// Per-tile compute body for the persistent kernel.  Computes one 256x256
// output tile for (group_id, pid_m_local, pid_n_idx); SMEM and tile dispatch
// are owned by the caller.
//
// ``a_group_offs_ptr[group_id]`` is the *input* row base (where group g's
// real rows start in the per-group-padded input layout); ``M_g`` is the
// number of *real* rows in group g (compute bound — padding rows are
// never visited).  When ``c_group_offs_ptr`` is non-null the kernel
// writes group g's output rows starting at ``c_group_offs_ptr[group_id]``
// (the original/unpadded layout) instead of ``a_group_offs_ptr[group_id]``,
// fusing the per-group-padding extract directly into the GEMM store.
// CK-style separation: compute_tile receives ALREADY group-resolved
// pointers (a_grp_ptr, b_grp_ptr, a_s_grp_ptr, b_s_grp_ptr, c_grp_ptr)
// + only per-tile bookkeeping.  All per-group readfirstlane chains and
// offset arithmetic live in the outer persistent kernel — narrows the
// SGPR live set in compute_tile to roughly the dense single-GEMM
// kernel's level, removing the SGPR pressure that drove scratch spill
// and the m0 chain race against buffer_load_lds.
template <typename GemmTile, typename AType, typename BType, typename CType>
__device__ __forceinline__ void turbo_grouped_gemm_mxfp8_compute_tile(
    GemmTile &tile, char *smem_buf,
    const AType *a_grp_ptr, const BType *b_grp_ptr,
    const uint32_t *a_s_grp_ptr, const uint32_t *b_s_grp_ptr, CType *c_grp_ptr,
    const int32_t pid_m_local, const int32_t pid_n_idx, const int32_t M_g, const uint32_t n,
    const uint32_t k) {
    using ASmem  = typename GemmTile::ASmemSubtile;
    using BSmem  = typename GemmTile::BSmemSubtile;
    using ASSmem = typename GemmTile::AScaleSmemSubtile;
    using BSSmem = typename GemmTile::BScaleSmemSubtile;
    constexpr size_t SMEM_DATA_BYTES_LOC = sizeof(ASmem) * 2 * 4 + sizeof(BSmem) * 2 * 4;
    auto *a_smem_tile   = reinterpret_cast<ASmem(*)[4]>(smem_buf);
    auto *b_smem_tile   = reinterpret_cast<BSmem(*)[4]>(smem_buf + sizeof(ASmem) * 2 * 4);
    auto *a_s_smem_tile = reinterpret_cast<ASSmem(*)[4]>(smem_buf + SMEM_DATA_BYTES_LOC);
    auto *b_s_smem_tile = reinterpret_cast<BSSmem(*)[4]>(
        smem_buf + SMEM_DATA_BYTES_LOC + sizeof(ASSmem) * 2 * 4);
    const int32_t pid_m = pid_m_local * 256;
    const int32_t pid_n = pid_n_idx * 256;
    if (pid_m >= M_g || pid_n >= (int32_t) n)
        return;

    // Re-assert pinned VGPR reservation per tile invocation: persistent
    // kernel reserves once at entry, but the compiler can repurpose
    // pinned VGPRs for spill across loop iterations.  Re-reserving here
    // refreshes the live-range marker before the LDS-direct prologue.
    // (Verified empirically: removing this gives 100% race rate.)
    tile.reserve_pinned_regs();

    const uint32_t lane_id = tile.lane_id;
    const uint32_t warp_id = tile.warp_id;
    const uint32_t warp_m  = tile.warp_m;
    const uint32_t warp_n  = tile.warp_n;

    const uint32_t scale_cols = (k + GemmTile::MX_BLOCK_SIZE - 1) / GemmTile::MX_BLOCK_SIZE;
    // M_g_in uses the padded length: HW SRD bounds make the over-read
    // safe (heap garbage for rows M_g..M_g_in-1) and store path's
    // is_boundary_tile mask drops those rows.  MFMA outer-product keeps
    // per-row accumulators independent — garbage rows can't contaminate
    // rows < M_g.
    const int32_t M_g_in =
        (M_g + detail::MXFP8_PADDING_ALIGN_SIZE - 1) & ~(detail::MXFP8_PADDING_ALIGN_SIZE - 1);

    const AType    *a_base_ptr   = a_grp_ptr + (int64_t) pid_m * k;
    const BType    *b_base_ptr   = b_grp_ptr + (int64_t) pid_n * k;
    const uint32_t *a_s_base_ptr = a_s_grp_ptr + (int64_t) pid_m * scale_cols;
    const uint32_t *b_s_base_ptr = b_s_grp_ptr + (int64_t) pid_n * scale_cols;

    uint32_t ldg_offsets[2];
    tile.compute_ldg_offsets(ldg_offsets, k);
    uint32_t sts_offsets[2];
    tile.compute_sts_offsets(sts_offsets);
    uint32_t lds_offsets[2];
    tile.compute_lds_offsets(lds_offsets);
    const uint32_t scale_ldg_offset = lane_id;
    const uint32_t scale_sts_offset = lane_id;
    const uint32_t scale_lds_offset = lane_id;

    const uint32_t  a_remaining  = ((uint32_t) M_g_in - pid_m) * k * sizeof(AType);
    const uint32_t  b_remaining  = (n - pid_n) * k * sizeof(BType);
    const uint32_t  as_remaining = ((uint32_t) M_g_in - pid_m) * scale_cols * sizeof(uint32_t);
    const uint32_t  bs_remaining = (n - pid_n) * scale_cols * sizeof(uint32_t);
    const BufferSRD a_srd(a_base_ptr, a_remaining);
    const BufferSRD b_srd(b_base_ptr, b_remaining);
    const BufferSRD a_s_srd(a_s_base_ptr, as_remaining);
    const BufferSRD b_s_srd(b_s_base_ptr, bs_remaining);

    constexpr int32_t DATA_STRIDE  = GemmTile::BLOCK_SIZE_K;
    constexpr int32_t SCALE_STRIDE = GemmTile::SCALE_FRAG_SIZE * sizeof(uint32_t);

    // ── Load tile 0 → smem[0], tile 1 → smem[1] ──
    // Two-step prologue (buffer_load_b128 → VGPR → ds_write_b128) instead
    // of the single-instruction LDS-direct buffer_load_lds.  ds_write is
    // tracked by lgkmcnt while compiler-emitted scratch_load (SGPR spill
    // consume) is on vmcnt — different counters break the spill→m0 race
    // window that gfx950's lack of vmcnt FIFO between scratch and buffer
    // ops would otherwise open.  See memory.cuh::load_gmem_to_smem_srd_two_step.
    tile.reserve_pinned_regs();
    tile.template load_a_gmem_to_smem_half_srd_two_step<0>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_a_gmem_to_smem_half_srd_two_step<1>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd_two_step<0>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd_two_step<1>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_a_scale_gmem_to_smem_half_srd_two_step<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                                  scale_sts_offset, scale_cols);
    tile.template load_a_scale_gmem_to_smem_half_srd_two_step<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                                  scale_sts_offset, scale_cols);
    tile.template load_b_scale_gmem_to_smem_half_srd_two_step<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                                  scale_sts_offset, scale_cols);
    tile.template load_b_scale_gmem_to_smem_half_srd_two_step<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                                  scale_sts_offset, scale_cols);

    tile.template load_a_gmem_to_smem_half_srd_two_step<0>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                            DATA_STRIDE);
    tile.template load_a_gmem_to_smem_half_srd_two_step<1>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                            DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd_two_step<0>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                            DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd_two_step<1>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                            DATA_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd_two_step<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                                  scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd_two_step<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                                  scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd_two_step<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                                  scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd_two_step<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                                  scale_sts_offset, scale_cols, SCALE_STRIDE);

    tile.zero_c_agpr();
    // Two-step prologue uses ds_write (lgkmcnt-tracked) for the LDS half.
    // Drain BOTH counters before the barrier so the LDS state is fully
    // committed before the first warp-coordinated ds_read in the main
    // loop.  vmcnt also needs draining for any not-yet-consumed buffer_load
    // results (the helper uses an implicit s_waitcnt internally, but the
    // last few may still be in flight).
    wait_vmcnt<0>();
    wait_lgkmcnt<0>();
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

    // Re-assert pinned reservation right before main MFMA loop entry.
    tile.reserve_pinned_regs();
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

        // End-of-K-iter drain.  `<12>` is the empirically loosest safe
        // value; `<16>` raced 100/100 on stress.
        wait_vmcnt<12>();
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

        // Single-GEMM-style split drain: keep the last 6 buffer_load_lds
        // in flight so Phase 3+4 mfma_lds overlap with the trailing GMEM→LDS DMAs.
        wait_vmcnt<6>();
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

    // ── Store C ──
    __builtin_amdgcn_sched_barrier(0);
    uint32_t c_stg_offsets[4];
    tile.compute_stg_offsets(c_stg_offsets);
    CType *c_stg_base_ptr = c_grp_ptr + (int64_t) pid_m * n + pid_n +
                            warp_id / 2 * 64 * (int64_t) n + warp_id % 2 * 64;
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
}

template <typename AType, typename BType, typename CType, typename AccType = float>
__global__
__launch_bounds__(256, 1) void turbo_grouped_gemm_mxfp8_256x256x128_16x16x128_4wave_persistent_kernel(
    const AType *a_ptr, const BType *b_ptr, const uint32_t *a_s_ptr, const uint32_t *b_s_ptr,
    CType *c_ptr, const int64_t *group_lens_ptr, const int64_t *a_group_offs_ptr,
    const int64_t *c_group_offs_ptr, const int32_t group_num, const uint32_t n, const uint32_t k,
    const int32_t grid_m, const int32_t grid_n) {
#if !defined(__gfx950__)
    assert(false && "turbo_grouped_gemm_mxfp8 persistent kernel requires gfx950");
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

    // The per-tile m field is unused inside compute_tile (boundary uses M_g
    // arg), so reuse one tile instance across the persistent loop.  Avoids
    // re-running its const-init (~lane_id/warp_id mods) per tile.
    GemmTile tile(threadIdx.x, 0, n, k);
    tile.reserve_pinned_regs();

    const int32_t tiles_per_group = grid_m * grid_n;
    const int32_t total_tiles     = tiles_per_group * group_num;
    for (int32_t tile_id = (int32_t) blockIdx.x; tile_id < total_tiles;
         tile_id += (int32_t) gridDim.x) {
        const int32_t group_id = tile_id / tiles_per_group;
        const int32_t rem      = tile_id - group_id * tiles_per_group;
        const int32_t pid_n    = rem / grid_m;
        const int32_t pid_m    = rem - pid_n * grid_m;

        const int32_t M_g =
            __builtin_amdgcn_readfirstlane(static_cast<int32_t>(group_lens_ptr[group_id]));
        if (M_g <= 0 || pid_m * 256 >= M_g) {
            continue;
        }

        // CK-style: resolve per-group pointers entirely in the outer loop.
        // compute_tile then sees only fully-offset base pointers — no
        // group_offs reads, no group-id arithmetic, no c_m_global.  This
        // shrinks compute_tile's SGPR live range to dense single-GEMM
        // kernel level, removing the LDS-direct prologue spill trigger.
        // Force scalar (s_load, lgkmcnt-tracked) loads via readfirstlane.
        const uint32_t scale_cols_outer =
            (k + GemmTile::MX_BLOCK_SIZE - 1) / GemmTile::MX_BLOCK_SIZE;
        const uint32_t a_off_lo = __builtin_amdgcn_readfirstlane(
            static_cast<uint32_t>(a_group_offs_ptr[group_id] & 0xffffffffULL));
        const uint32_t a_off_hi = __builtin_amdgcn_readfirstlane(
            static_cast<uint32_t>(static_cast<uint64_t>(a_group_offs_ptr[group_id]) >> 32));
        const int64_t group_offset =
            (static_cast<int64_t>(a_off_hi) << 32) | static_cast<int64_t>(a_off_lo);
        const uint32_t c_off_lo = __builtin_amdgcn_readfirstlane(
            static_cast<uint32_t>(c_group_offs_ptr[group_id] & 0xffffffffULL));
        const uint32_t c_off_hi = __builtin_amdgcn_readfirstlane(
            static_cast<uint32_t>(static_cast<uint64_t>(c_group_offs_ptr[group_id]) >> 32));
        const int64_t c_group_offset =
            (static_cast<int64_t>(c_off_hi) << 32) | static_cast<int64_t>(c_off_lo);

        const AType    *a_grp_ptr   = a_ptr + group_offset * (int64_t) k;
        const BType    *b_grp_ptr   = b_ptr + (int64_t) group_id * (int64_t) n * (int64_t) k;
        const uint32_t *a_s_grp_ptr = a_s_ptr + group_offset * (int64_t) scale_cols_outer;
        const uint32_t *b_s_grp_ptr =
            b_s_ptr + (int64_t) group_id * (int64_t) n * (int64_t) scale_cols_outer;
        CType          *c_grp_ptr   = c_ptr + c_group_offset * (int64_t) n;

        turbo_grouped_gemm_mxfp8_compute_tile<GemmTile, AType, BType, CType>(
            tile, smem_buf,
            a_grp_ptr, b_grp_ptr, a_s_grp_ptr, b_s_grp_ptr, c_grp_ptr,
            pid_m, pid_n, M_g, n, k);
    }
#endif
}

} // namespace turbo
} // namespace primus_turbo
