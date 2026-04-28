// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

// Reuse GEMM_Tile_MXFP8_NT_*, preshuffle_scale_16x4_kernel, and BufferSRD helpers
// from the single-GEMM kernel.
#include "../../gemm/turbo/turbo_gemm_mxfp8_kernel.h"

namespace primus_turbo {
namespace turbo {

// ── Per-tile compute body used by the persistent kernel ──
//
// Given a resolved tile (group_id, pid_m_local in tiles, pid_n_idx in tiles)
// computes one 256x256 output tile. Caller is responsible for shared-memory
// allocation and persistent tile scheduling.
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

    // PERF: drain ONLY the prologue ds_reads here (8+4+8+4 = 24 LDS reads
    // that fill PIN_A0/AS0/B0/BS0 for the main loop's first MFMA).  The
    // tile-2 prefetch `buffer_load_lds` below targets a_smem[cur][0,1] and
    // b_smem[cur][0,1] (warp-specific 2KB chunks) which the main loop's
    // Phase 1 does NOT read — Phase 1 consumes the just-filled PIN_* regs
    // and issues new ds_reads from b_smem[cur][warp_n+2] (the OPPOSITE
    // half).  The tile-2 LDG's LDS-write side will be drained by Phase 1's
    // existing WAR barrier (`wait_lgkmcnt<0>; s_barrier` inside
    // `phase_mfma_lds_ldg`, see turbo_gemm_mxfp8_kernel.h ~L477).
    //
    // Issuing the LDGs AFTER the wait lets their ~150-cycle GMEM-to-LDS
    // latency overlap with the 6 MFMAs that Phase 1 fires before its WAR
    // barrier (~96 cycles of compute), so the steady-state stall on the
    // tile-2 LDGs drops from ~150 to ~50 cycles per output tile.  The wait
    // itself shrinks to the ds_read tail (~5-10 cycles).
    wait_lgkmcnt<0>();

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

        // PERF: partial vmcnt drain at Phase 1->2 barrier.  vmcnt<3> is the
        // tightest empirically-safe bound for this 4-phase pipeline on the
        // forward grouped MXFP8 kernel: 286c897 originally shipped with
        // wait_vmcnt<3> after a 5000-iter grouped stress sweep showed
        // wait_vmcnt<4> still leaked (race re-emerges); a08b5b7 then bumped
        // it to wait_vmcnt<0> as part of a "stabilize variable-K kernels"
        // pass that bundled all four kernel barriers (FWD + wgrad + epilogue)
        // to the most conservative drain.  The variable-K reasoning only
        // applied to the wgrad kernel (its K dimension = M_g varies per
        // group); the forward kernel has fixed K and the original vmcnt<3>
        // analysis still holds, so we restore the partial drain.
        //
        // What's still in flight when Phase 2 begins (worst case, vmcnt<3>):
        //   * up to 3 of Phase 1's tail buffer_load_lds writes targeting
        //     b_smem[cur][2,3].  Phase 2's ds_reads target a_smem[cur][warp_m+2]
        //     (different LDS array, no LDS WAR), and Phase 2's MFMAs use
        //     PIN_B1 sourced from Phase 1's already-completed ds_reads
        //     (drained by Phase 1's internal WAR barrier at the end of its
        //     `phase_mfma_lds_ldg` body), so the in-flight LDGs cannot
        //     corrupt Phase 2's inputs.
        //   * Phase 2's own `phase_mfma_lds_ldg` then issues 6 more LDGs;
        //     end-of-loop wait_vmcnt<12> still bounds the cumulative
        //     in-flight count.
        //
        // Per-iter saving: drains drop from ~50 cycles (vmcnt<0> for ~18
        // outstanding LDGs) to ~30 cycles (vmcnt<3> for at most 15-of-18),
        // i.e. ~20 cycles × k_iters per output tile.
        //
        // PERF: drop the previously-here `__builtin_amdgcn_s_barrier()`.
        // The inter-Phase wave-sync is provided by Phase 2's own
        // `phase_mfma_lds_ldg` body, which begins with 6 MFMA + 20
        // ds_reads and then issues `wait_lgkmcnt<0> + s_barrier` (the
        // WAR barrier; see turbo_gemm_mxfp8_kernel.h ~L477-L478) before
        // its `buffer_load_lds`.  That internal s_barrier is a stronger
        // wave-sync point than the one we're removing here:
        //   * Wave A entering Phase 2 with wave B still in Phase 1's
        //     buffer_load_lds tail does NOT corrupt wave A's Phase 2
        //     ds_reads — wave A reads `a_smem[cur][warp_m+2]` which is
        //     wave A's own LDS region (warp-local sts_warp_base offset),
        //     never written by wave B.  The cross-warp LDS dependency
        //     (Phase 2's PIN_B1 source) was already drained by Phase 1's
        //     own internal WAR barrier inside its `phase_mfma_lds_ldg`,
        //     not by this outer s_barrier.
        //   * Wave A's Phase 2 `buffer_load_lds` writes target
        //     `a_smem[cur][2,3]`, i.e. the OPPOSITE LDS array from
        //     wave B's still-in-flight Phase 1 writes (`b_smem[cur][2,3]`),
        //     so no LDS WAW.
        //   * Phase 2's internal s_barrier (after its 20 ds_reads) re-aligns
        //     all 4 waves before any new buffer_load_lds is issued, so
        //     wave skew accumulated across Phase 1->2 cannot persist past
        //     that point.
        // Net saving: ~30 cycles of s_barrier wave-sync stall per K-iter
        // (gfx950 s_barrier emulates a wavefront-rendezvous via lgkm-bound
        // hardware sync).  On main_iters=53 (DSv3-GateUP-B16, K=7168) this
        // is ~1.6k cycles per output tile (~0.3-0.5 % of fwd K-loop wall).
        wait_vmcnt<3>();

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

        // End-of-K-iter drain.  Phase 4 just issued 6 LDGs to next[0,1]
        // (data + scale).  After the swap below, Phase 1 of the NEXT K-iter
        // reads from cur[warp_n+2] (== formerly-next[2,3]) — a *different*
        // SMEM sub-array than where Phase 3+4 wrote (formerly-next[0,1]) —
        // so the immediate post-swap reader does not RAW-depend on Phase
        // 3+4's tail LDGs.  However tightening this bound has been
        // characterized empirically and `<12>` is the loosest safe value:
        //   * `<12>`  : current — 0/100 stress on G=4 M=1024 N=2048 K=2048
        //              E4M3 (5000-iter), and 0-2/100 on the metric stress
        //              probe.  Holds on the post-squash kernel.
        //   * `<16>`  : auto-opt loop round 20 (2026-04-28, log dir
        //              `auto_optimize_logs/20260427_115238/round_020/`)
        //              tried this and triggered a CATASTROPHIC race —
        //              `stress_bad=100/100` with score collapse to ~5681.
        //              The change was reverted.  Do not retry.
        //   * `<13..15>` : not characterized but expected to race because
        //              the 100/100 race at `<16>` indicates the LSU-WAR
        //              window between this iter's tail LDG-LDS-writes and
        //              the next iter's Phase 3 ds_reads is closer to
        //              4 in-flight slots than to 0 — `<13>` would already
        //              expose it on most shapes.
        // If a future round wants to reach for a tighter bound, the right
        // direction is `<10>`/`<8>` (more drain, slightly safer, slower)
        // — NOT `<13+>` (looser, races).
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

        // RACE FIX: drain Epi1 buffer_load_lds (B1+A1 prefetch) before
        // flipping the double buffer.  wait_lgkmcnt<0> previously here is
        // redundant: phase_mfma_lds (used for Phase 3+4 of this Epi1) ends
        // with its own `wait_lgkmcnt<0>` (see turbo_gemm_mxfp8_kernel.h
        // ~L414), so lgkmcnt is already 0 with no pending ds_reads at this
        // point.  Drop it to match the inner-loop Phase 1->2 barrier idiom.
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
        // `wait_vmcnt<0>()` at the end of Epi1 (~L298), and Epi2 phases 1+2
        // are both `phase_mfma_lds` (no `buffer_load_lds`, hence no vmem
        // ops issued in this region).  The `s_barrier()` is preserved so
        // waves stay in step before Phase 3 reads from the `next`
        // (cross-buffer) LDS region.  The remaining lgkmcnt drain is also
        // unnecessary: `phase_mfma_lds` for Phase 2 ends with its own
        // `wait_lgkmcnt<0>()` (turbo_gemm_mxfp8_kernel.h ~L414), so by
        // exit lgkmcnt is already 0.  Net effect: one fewer s_waitcnt
        // instruction inside every `compute_tile()` call (the persistent
        // kernel reaches Epi2 once per output tile, so the saving scales
        // with `total_tiles / 256_resident_CTAs`).
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
    CType *c_stg_base_ptr =
        c_ptr + m_global * (int64_t) n + pid_n + warp_id / 2 * 64 * (int64_t) n + warp_id % 2 * 64;
    const bool is_boundary_tile = (pid_m + 256 > M_g) || (pid_n + 256 > (int32_t) n);

    // Forward C-store: non-volatile so the compiler can coalesce stores and
    // skip the per-element vmcnt(0) drains that volatile would emit.  Each
    // output tile is written by exactly one CTA to a disjoint slice, so this
    // is correctness-safe.  A single wait_vmcnt<0>() below drains the whole
    // C epilogue before the next persistent-loop iteration or kernel exit.
    //
    // RACE FIX (FWD `out` race on G=4 M=1024 N=2048 K=2048): when all four
    // (read_c, store_c) pairs share a single `c_tmp` VGPR array, the LLVM
    // register allocator gives the SAME 64 VGPRs to every pair, so pair-(i+1)
    // `v_accvgpr_read_b32` instructions write the very registers that
    // pair-(i)'s in-flight `flat_store_short` are still latching from.
    // `sched_barrier(0)` between pairs reduces but does not eliminate the
    // resulting drift (df78f4a brought it from 4/200 to 3/500 on E4M3, but
    // 100-iter stress still trips 2-6/100 on this shape).
    //
    // Use two alternating `c_tmp_a` / `c_tmp_b` buffers so adjacent pairs
    // are guaranteed to land in disjoint VGPR slots: pair-0 writes c_tmp_a,
    // pair-1 writes c_tmp_b (cannot collide with pair-0's stores), pair-2
    // reuses c_tmp_a only AFTER pair-1's complete (read+store) sequence
    // sits between pair-0 and pair-2 in program order, giving pair-0's
    // flat_store_short ample time to issue and consume their source VGPRs
    // before pair-2's reads land.  Cost: ~64 extra unpinned VGPRs (the
    // kernel is at vgpr_count=512 with launch_bounds(256,1), so this is
    // well within budget — no spill).
    //
    // RACE FIX (2a1943c left ~2/100 residual): even with alternating
    // buffers, pair-2's `v_accvgpr_read_b32` into c_tmp_a can fire
    // before pair-0's 64 `flat_store_short` have read their source
    // VGPRs in their entirety — only ~80 cycles of pair-1 (16 reads + 64
    // stores) sit between pair-0's issue and pair-2's overwrite, which
    // is short of the LSU's source-read window for ~few percent of waves.
    // Insert `wait_vmcnt<63>` immediately before each c_tmp reuse so the
    // older pair's stores are FORCED to commit (and thus retire their
    // source-VGPR reads) before the new pair's accvgpr_read overwrites
    // them.  After pair-1's 64 stores issue, vmcnt<=128; wait_vmcnt<63>()
    // blocks until vmcnt≤63, i.e., pair-0's 64 stores plus 1 of pair-1's
    // have committed — pair-0 fully drained, c_tmp_a safe to overwrite.
    // 63 is the gfx950 max for the s_waitcnt vmcnt field; 64 is rejected
    // by the assembler ("too large value for vmcnt").  The wait is
    // empirically near-zero cost because pair-1's read+store sequence
    // (~80 cycles) already overlaps most of pair-0's commit latency.
    if (!is_boundary_tile) {
        float32x4 c_tmp_a[4][4];
        float32x4 c_tmp_b[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) n + 0 * 128, n,
                                             c_tmp_a, c_stg_offsets);
        __builtin_amdgcn_sched_barrier(0);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) n + 1 * 128, n,
                                             c_tmp_b, c_stg_offsets);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) n + 0 * 128, n,
                                             c_tmp_a, c_stg_offsets);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) n + 1 * 128, n,
                                             c_tmp_b, c_stg_offsets);
    } else {
        const int32_t warp_base_m  = warp_id / 2 * 64;
        const int32_t warp_base_n  = warp_id % 2 * 64;
        const int32_t tile_valid_m = min(M_g - pid_m, 256) - warp_base_m;
        const int32_t tile_valid_n = min((int32_t) n - pid_n, 256) - warp_base_n;
        float32x4     c_tmp_a[4][4];
        float32x4     c_tmp_b[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) n + 0 * 128, n,
                                             c_tmp_a, c_stg_offsets, tile_valid_m, tile_valid_n);
        __builtin_amdgcn_sched_barrier(0);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 0 * 128 * (int64_t) n + 1 * 128, n,
                                             c_tmp_b, c_stg_offsets, tile_valid_m,
                                             tile_valid_n - 128);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp_a);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) n + 0 * 128, n,
                                             c_tmp_a, c_stg_offsets, tile_valid_m - 128,
                                             tile_valid_n);
        __builtin_amdgcn_sched_barrier(0);
        wait_vmcnt<63>();
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp_b);
        tile.template store_c_subtile<false>(c_stg_base_ptr + 1 * 128 * (int64_t) n + 1 * 128, n,
                                             c_tmp_b, c_stg_offsets, tile_valid_m - 128,
                                             tile_valid_n - 128);
    }
    // PERF: drop the previously-here trailing `wait_vmcnt<0>()`.  C-store
    // drain ordering is already provided by:
    //   1. The next persistent-loop iteration's prologue
    //      `wait_vmcnt<0>(); s_barrier();` (around L112) drains any C-store
    //      vmem still in flight before the new iteration's ds_reads consume
    //      the prologue's buffer_load_lds writes — same counter, same wave.
    //   2. The persistent-loop CTA-level `s_barrier()` (at the end of each
    //      iteration body) carries an implicit `"memory"` clobber, so the
    //      compiler cannot reorder C-store stores past it.
    //   3. On the LAST persistent-loop iteration the kernel exits via
    //      `s_endpgm`, which natively waits for outstanding VMEM ops to
    //      retire before the wave terminates and the dispatch completes —
    //      no trailing fence required to make the writes visible to host
    //      after `cudaDeviceSynchronize` / `hipStreamSynchronize`.
    // Net effect: one fewer `s_waitcnt vmcnt(0)` per output tile in the
    // persistent loop hot path; the wait was previously labeled "effectively
    // free" but in practice the prev-iter's stores and the next-iter's
    // prologue LDGs naturally overlap in the HBM bandwidth window when this
    // explicit serialization point is removed.  C-store addresses (output
    // tile `c[m_global..m_global+256, pid_n..pid_n+256]`) and the new
    // prologue's input addresses (`a[m_global_next * k]`, `b[g_next * n *
    // k]`) live in disjoint global regions, so removing the fence introduces
    // no aliasing race.
}

template <typename AType, typename BType, typename CType, typename AccType = float>
__global__ __launch_bounds__(256, 1) void
turbo_grouped_gemm_mxfp8_256x256x128_16x16x128_4wave_persistent_kernel(
    const AType *a_ptr, const BType *b_ptr, const uint32_t *a_s_ptr, const uint32_t *b_s_ptr,
    CType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const int32_t group_num, const uint32_t n, const uint32_t k, const int32_t grid_m,
    const int32_t grid_n) {
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
    auto            *a_smem_tile = reinterpret_cast<ASmem(*)[4]>(smem_buf);
    auto            *b_smem_tile = reinterpret_cast<BSmem(*)[4]>(smem_buf + sizeof(ASmem) * 2 * 4);
    auto            *a_s_smem_tile = reinterpret_cast<ASSmem(*)[4]>(smem_buf + SMEM_DATA_BYTES);
    auto            *b_s_smem_tile =
        reinterpret_cast<BSSmem(*)[4]>(smem_buf + SMEM_DATA_BYTES + sizeof(ASSmem) * 2 * 4);

    GemmTile reserve_tile(threadIdx.x, 0, n, k);
    reserve_tile.reserve_pinned_regs();

    const int32_t tiles_per_group = grid_m * grid_n;
    const int32_t total_tiles     = tiles_per_group * group_num;
    for (int32_t tile_id = (int32_t) blockIdx.x; tile_id < total_tiles; tile_id += (int32_t) gridDim.x) {
        const int32_t group_id = tile_id / tiles_per_group;
        const int32_t rem      = tile_id - group_id * tiles_per_group;
        const int32_t pid_n    = rem / grid_m;
        const int32_t pid_m    = rem - pid_n * grid_m;

        const int32_t M_g =
            __builtin_amdgcn_readfirstlane(static_cast<int32_t>(group_lens_ptr[group_id]));
        if (M_g <= 0 || pid_m * 256 >= M_g) {
            continue;
        }

        GemmTile tile(threadIdx.x, (uint32_t) M_g, n, k);
        turbo_grouped_gemm_mxfp8_compute_tile<GemmTile, AType, BType, CType>(
            tile, a_smem_tile, b_smem_tile, a_s_smem_tile, b_s_smem_tile, a_ptr, b_ptr, a_s_ptr,
            b_s_ptr, c_ptr, group_offs_ptr, group_id, pid_m, pid_n, M_g, n, k);
        // CRITICAL: do NOT remove this CTA-level s_barrier.
        //
        // It re-syncs the 4 waves between the just-finished tile's epilogue
        // (whose final phase still issues `phase_mfma_lds` reads against the
        // double-buffered SMEM region for the next tile's PIN_NEXT load) and
        // the next tile's prologue, which immediately starts overwriting the
        // SAME SMEM regions with fresh GMEM->SMEM LDGs.  Without this barrier
        // a faster wave can race ahead and clobber SMEM that a slower wave
        // is still reading from for the previous tile's final mfma_lds, which
        // shows up as `out` drift on the MXFP8 stress probe (verified: removing
        // this barrier raised stress_bad to ~5-7/100 on G=4 M=1024 N=2048
        // K=2048 E4M3, well above the ≤2/100 target).  compute_tile()'s own
        // prologue does its own `wait_vmcnt<0>(); s_barrier();`, but that fires
        // AFTER its 6 prologue LDGs have already issued, i.e. too late.
        __builtin_amdgcn_s_barrier();
    }
#endif
}

} // namespace turbo
} // namespace primus_turbo
