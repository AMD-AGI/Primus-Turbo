// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// gfx950 (MI355X) fused FP8 x FP4 mega-MoE kernel.
//
// Direct structural port of DeepGEMM's ``sm100_fp8_fp4_mega_moe_impl``
// (deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) to AMD
// CDNA4 (gfx950).  The high-level structure is preserved 1:1:
//
//   * Persistent warp-specialized CTA partitioned into
//       - dispatch warps   : pull remote tokens into the local L1 pool
//       - A/B loader warps : stage activations/weights + SFs into LDS
//       - MMA issue warp   : issue scaled MFMA ops over the LDS staged data
//       - epilogue warps   : SwiGLU + UE8M0 quant (L1)  and  BF16 NVLink write
//                            (L2) ; the same warps then drive the combine
//                            reduce-by-topk write-back path.
//   * The per-wave persistent scheduler from
//     ``primus_turbo::mega_moe::sched::MegaMoEScheduler`` (a 1:1 port of
//     ``deep_gemm::sched::MegaMoEScheduler``) walks the
//       Linear1 (L1 GEMM)  ->  Linear2 (L2 GEMM)
//     pair for every expert wave assigned to this CTA.  ``for_each_block``
//     is the same state machine on both backends - the Linear1 sub-wave
//     completes for every expert before any Linear2 block starts, so the
//     L1 -> SwiGLU -> L2 pipeline overlaps with the next wave's L1 fill.
//
// AMD-specific substitutions (kept as close to the SM100 line-by-line
// order as possible so a side-by-side diff stays readable):
//
//   * TMA descriptors  ->  raw global pointers + ``buffer_load_lds``.
//   * tcgen05 / TMEM  ->  in-register AGPR accumulators (4 floats per
//                          MFMA output) - reused across Linear1/Linear2
//                          via per-stage rotation through LDS.
//   * mbarrier / Barrier  -> atomic counters in LDS, polled with
//                            ``__atomic_load_n`` (acq) /
//                            ``__atomic_fetch_add`` (rel).
//   * Cluster / 2-CTA  ->  single CTA (UMMA_M == BLOCK_N, no leader/follower).
//                          The wave64 mfma operates over a BLOCK_M x BLOCK_N
//                          tile in a single workgroup, mirroring the SM100
//                          ``UMMA_N * 2`` accumulator layout but folded
//                          back into one CTA.
//   * UMMA scaled MMA  ->  ``v_mfma_scale_f32_16x16x128_f8f6f4`` (MXFP8/FP4).
//   * NVLink barrier  ->  HIP IPC + ``__atomic_*`` on the same symmetric
//                          buffer pad as DG.
//
// The host launcher ``launch_fp8_fp4_mega_moe_impl`` mirrors DG's
// public API signature so the existing JIT TU / binding layer can
// instantiate the template unchanged.  For the kNumRanks==1 path used
// by the JIT correctness/perf test, the NVLink barrier collapses to a
// no-op grid sync (no XGMI signalling needed).

#pragma once

#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

// ``__grid_constant__`` is a CUDA-only qualifier used by DG to bind
// large by-value kernel parameters (TMA descriptors, SymBuffer) into
// constant memory.  On HIP the same parameters are passed by value
// through the regular kernel ABI; alias the qualifier to a no-op so
// the DG kernel signature can be copied verbatim.
#ifndef __grid_constant__
#define __grid_constant__
#endif

#include "primus_turbo/device/lds_swizzle.cuh"
#include "primus_turbo/device/memory.cuh"
#include "primus_turbo/device/mfma.cuh"
#include "primus_turbo/dtype.h"

#include "../layout/mega_moe.cuh"
#include "../layout/sym_buffer.cuh"
#include "../scheduler/mega_moe.cuh"
#include "prims.cuh"

namespace primus_turbo {
namespace mega_moe {

// ---------------------------------------------------------------------
//  Compile-time descriptor of a single mega-MoE specialization.  Lifted
//  1:1 from DG's ``sm100_fp8_fp4_mega_moe_impl`` template parameter list
//  with the addition of an explicit ``arch`` tag so multiple AMD targets
//  (gfx942 / gfx950 / ...) can coexist.
// ---------------------------------------------------------------------
enum class MegaMoEArch : uint32_t {
    Unknown = 0,
    Gfx942  = 942,
    Gfx950  = 950,
};

// ---------------------------------------------------------------------
//  HIP-side equivalents of the SM100 ``ptx::`` / ``cute::`` primitives
//  the device kernel depends on live in the sibling ``prims.cuh`` under
//  ``primus_turbo::mega_moe::prims`` - names mirror
//  ``deep_gemm/ptx/{utils,ld_st}.cuh`` (e.g. ``prims::sync_aligned`` <->
//  ``ptx::sync_aligned``).  No alias is needed since the enclosing
//  ``primus_turbo::mega_moe`` namespace already exposes ``prims::`` at
//  the short name used throughout the device kernel.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
//  LDS mbarrier helpers - atomic counter + spin loop.  Each "barrier"
//  is just a ``uint32_t`` in LDS whose value encodes ``(phase, count)``;
//  arrive bumps ``count`` and may flip ``phase``, wait spins until the
//  phase matches.  Mirrors DG's ``cutlass::arch::ClusterTransactionBarrier``
//  API surface (init / arrive / wait) at a coarse granularity sufficient
//  for our producer-consumer pipeline.
// ---------------------------------------------------------------------
struct Mbarrier {
    uint32_t        state;
    __device__ void init(uint32_t arrive_count) { state = (arrive_count << 16) | 0u; }
    __device__ void arrive() { atomicAdd(&state, 1u); }
    __device__ void wait(uint32_t target_count) {
        while ((prims::ld_acq(&state) & 0xffffu) < target_count) {
        }
    }
    __device__ void reset(uint32_t arrive_count) {
        __atomic_store_n(&state, (arrive_count << 16) | 0u, __ATOMIC_RELEASE);
    }
};

// ---------------------------------------------------------------------
//  Cross-rank barriers.  For kNumRanks == 1 these are simple grid syncs;
//  for kNumRanks > 1 they additionally exchange signals over IPC.
// ---------------------------------------------------------------------
namespace comm {

// NOTES on the named-barrier arguments:
//   * ``named_bar`` is the workgroup-shared NamedBarrierWg; both grid
//     sync flanks below replace DG's PTX ``bar.sync`` with our
//     emulation.
//   * ``bar_id`` identifies the LDS counter to use - must be distinct
//     from any concurrently-active bar on the SAME wave set.
//   * ``num_threads`` is the *participating* thread count (the calling
//     role's wave count x warpSize).  This is also what bumps each
//     thread's ``expected[bar_id]``, so every thread in the role must
//     pass the same value.
//   * ``leader_thread_idx`` is the absolute ``threadIdx.x`` of the
//     first participating thread.  DG hard-codes ``thread_idx == 0``
//     because PTX bar.sync internally picks a leader; on AMD we need
//     an explicit per-role leader because ``threadIdx.x == 0`` belongs
//     to the dispatch role and is never executed by the epilogue path.
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0>
__device__ __forceinline__ void
grid_sync(const layout::Workspace &workspace, prims::NamedBarrierWg &named_bar, uint32_t bar_id,
          uint32_t num_threads, uint32_t leader_thread_idx, uint32_t sm_idx, uint32_t thread_idx) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    prims::sync_aligned(named_bar, num_threads, bar_id);
    if (thread_idx == leader_thread_idx) {
        auto          *count_ptr = workspace.get_grid_sync_count_ptr(kGridSyncIndex);
        const uint32_t old_value =
            prims::atomic_add_rel(count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1u)) : 1u);
        uint32_t new_value;
        do {
            new_value = prims::ld_acq(count_ptr);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0u);
    }
    prims::sync_aligned(named_bar, num_threads, bar_id);
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex,
          uint32_t kTag>
__device__ __forceinline__ void
nvlink_barrier(const layout::Workspace &workspace, const layout::SymBuffer<kNumRanks> &sym_buffer,
               prims::NamedBarrierWg &named_bar, uint32_t bar_id, uint32_t leader_thread_idx,
               uint32_t sm_idx, uint32_t thread_idx, bool sync_prologue = true,
               bool sync_epilogue = true) {
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, named_bar, bar_id, kNumThreads,
                                           leader_thread_idx, sm_idx, thread_idx);

    if constexpr (kNumRanks > 1) {
        if (sm_idx == 0) {
            auto *counter_ptr = workspace.get_nvl_barrier_counter_ptr();
            // NOTES: ``counter_ptr`` is shared across SMs and gets
            // incremented below via an atomic RMW.  A naked ``*counter_ptr``
            // load races with that RMW under ROCm's memory model; use
            // a relaxed atomic load so the value is well-defined.
            const auto status       = prims::ld_volatile(counter_ptr) & 3u;
            const auto signal_phase = status & 1u;
            const auto signal_sign  = status >> 1u;
            auto      *signal_ptr   = workspace.get_nvl_barrier_signal_ptr(signal_phase);

            // Role-relative predicate: each role has its own
            // ``leader_thread_idx`` (0 for dispatch, kEpiLeader for epi).
            // The cross-rank signal writers are the first ``kNumRanks``
            // threads of the calling role, not absolute threads
            // [0, kNumRanks).  The original absolute check excluded all
            // epi threads since kEpiLeader (128) > kNumRanks (2), so
            // the epi nvlink_barrier-epi never published its signal and
            // the leader thread spun forever on signal=0.
            if (thread_idx >= leader_thread_idx && thread_idx < leader_thread_idx + kNumRanks)
                prims::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx - leader_thread_idx),
                                       signal_sign ? -1 : 1);
            prims::sync_aligned(named_bar, kNumThreads, bar_id);

            if (thread_idx == leader_thread_idx) {
                prims::red_add(reinterpret_cast<int *>(counter_ptr), 1);
                const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
                while (prims::ld_acq_sys(signal_ptr) != target) {
                }
            }
        }
    } else {
        (void) sym_buffer;
    }

    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, named_bar, bar_id, kNumThreads,
                                           leader_thread_idx, sm_idx, thread_idx);
}

} // namespace comm

// ---------------------------------------------------------------------
//  Block-scaled MFMA helpers.  Each ``UMMA``-style call in the SM100
//  kernel corresponds to a ``v_mfma_scale_f32_16x16x128_f8f6f4`` MFMA
//  on AMD.  The accumulator lives in AGPRs (4 floats per MFMA tile per
//  lane), so the LDS-staged A/B tiles must be ``ds_read``ed into VGPRs
//  per inner-k loop.  For the "first step" alignment pass we keep the
//  accumulator and operand layouts in source form - a follow-up patch
//  will hand-tune the register pinning.
// ---------------------------------------------------------------------
template <typename Adtype, typename Bdtype>
__device__ __forceinline__ dtype::float32x4 mfma_scaled(dtype::int32x8 a, dtype::int32x8 b,
                                                        dtype::float32x4 c, uint32_t scale_a,
                                                        uint32_t scale_b) {
    return device::mfma_scale_f32_16x16x128_f8f6f4<Adtype, Bdtype>::run(a, b, c, scale_a, scale_b);
}

// ---------------------------------------------------------------------
//  Persistent warp-specialized device kernel.  Direct structural port of
//  ``deep_gemm::sm100_fp8_fp4_mega_moe_impl`` to gfx950 - see the file
//  header for the substitution map.
//
//  The template parameter list and ordering mirror SM100 exactly so the
//  host launcher in ``mega_moe.cu`` and the JIT TU in
//  ``tests/pytorch/ops/jit/mega_moe_jit_launch.cu`` can instantiate the
//  same shape tuples on both backends.
// ---------------------------------------------------------------------
template <uint32_t kNumMaxTokensPerRank, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kNumTopk, uint32_t kNumExpertsPerWave, uint32_t BLOCK_M,
          uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t STORE_BLOCK_M, uint32_t SF_BLOCK_M,
          uint32_t SF_BLOCK_N, uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens,
          uint32_t kNumStages, uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads, uint32_t kNumSMs, uint32_t kNumRanks,
          float kActivationClamp, bool kFastMath, uint32_t L1_SHAPE_N = kIntermediateHidden * 2u,
          uint32_t L1_SHAPE_K = kHidden, uint32_t L2_SHAPE_N = kHidden,
          uint32_t L2_SHAPE_K  = kIntermediateHidden,
          uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
          uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks>
__global__ __launch_bounds__(kNumThreads, 1) void gfx950_fp8_fp4_mega_moe_kernel(
    void *y, int *cumulative_local_expert_recv_stats, const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, void *l1_weights,
    void *l1_weights_sf, void *l2_weights, void *l2_weights_sf) {
#if defined(__gfx950__)
    using namespace prims;

    // ---- Wave / thread role indices (wave64 on AMD) ----
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = get_warp_idx(); // 0-based wave64 within CTA
    const uint32_t lane_idx   = get_lane_idx();

    // Warp grouping (wave64 units).  Total threads = 512 by default
    // (128 + 128 + 256), giving 8 wave64s arranged as
    //   [0..kNumDispatchWarps)               : dispatch
    //   [kNumDispatchWarps..+kNumMMANonEpilogueWarps)  : A loader / B loader / MMA issue
    //   [..rest..)                            : epilogue + combine
    constexpr uint32_t kNumDispatchWarps       = kNumDispatchThreads / kWarpSize;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / kWarpSize;
    constexpr uint32_t kNumEpilogueWarps       = kNumEpilogueThreads / kWarpSize;
    constexpr uint32_t kNumEpilogueWarpgroups  = (kNumEpilogueWarps + 3u) / 4u;
    (void) kNumEpilogueWarpgroups;

    // ---- Token / SF layout descriptors (mirrored from SM100) ----
    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(__hip_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32u);
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 32u);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // ---- Workspaces / per-pool buffers (1:1 from DG) ----
    //
    // Inter-buffer base pointers are 256-byte aligned so the device-side
    // chain matches the host launcher's ``bump`` cursor in
    // ``mega_moe.cu::get_symm_buffer_size_for_mega_moe`` (and the JIT
    // probe in ``mega_moe_jit_launch.cu::mega_moe_jit_compute_layout``).
    // Without this, the dispatch path reads ``input_topk_idx`` from a
    // shifted (mostly-zero) region, routing all tokens to expert 0.
    constexpr uint64_t kInterBufAlign = 256u;
    auto               align256       = [](void *p) -> void                     *{
        const auto v = reinterpret_cast<uintptr_t>(p);
        return reinterpret_cast<void *>((v + kInterBufAlign - 1u) &
                                                            ~uintptr_t(kInterBufAlign - 1u));
    };

    const auto workspace = layout::Workspace(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
                                             kNumMaxTokensPerRank, kNumTopk);

    const auto input_token_buffer    = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank,
                                                      align256(workspace.get_end_ptr()));
    const auto input_sf_buffer       = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank,
                                                      align256(input_token_buffer.get_end_ptr()));
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank, align256(input_sf_buffer.get_end_ptr()));
    const auto input_topk_weights_buffer =
        layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank,
                       align256(input_topk_idx_buffer.get_end_ptr()));

    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens,
                                                       align256(input_topk_weights_buffer.get_end_ptr()));
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
                                                       align256(l1_token_buffer.get_end_ptr()));
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens,
                                                       align256(l1_sf_buffer.get_end_ptr()));

    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
                                                align256(l1_topk_weights_buffer.get_end_ptr()));
    const auto l2_sf_buffer = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
                                             align256(l2_token_buffer.get_end_ptr()));
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, align256(l2_sf_buffer.get_end_ptr()));

    // ---- SF transpose helper (UTCCP 4x32 on SM100 - we keep the same
    //      index transform so the SF byte layout matches DG byte-for-byte).
    constexpr uint32_t kGranK                 = 32u;
    constexpr uint32_t kNumUTCCPAlignedElems  = 128u;
    auto               transform_sf_token_idx = [](const uint32_t &token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M + (idx & ~127u) + (idx & 31u) * 4u +
               ((idx >> 5) & 3u);
    };

    // ---- MMA tile geometry.  SM100 swaps A/B; we keep the same names ----
    constexpr uint32_t LAYOUT_AD_M  = 128u;
    constexpr uint32_t UMMA_M       = LAYOUT_AD_M; // single-CTA (no 2-CTA cluster)
    constexpr uint32_t UMMA_N       = BLOCK_M;     // swap AB
    constexpr uint32_t UMMA_K       = 32u;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2u; // multicast on A (DG-aligned)
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    static_assert(BLOCK_M % 16u == 0u, "Invalid block M");
    static_assert(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    static_assert(BLOCK_K == 128u, "Invalid block K");
    (void) UMMA_M;
    (void) UMMA_N;
    (void) UMMA_K;
    (void) LOAD_BLOCK_M;
    (void) LOAD_BLOCK_N;

    // ---- Per-stage LDS region sizes (mirrors SM100 byte-for-byte) ----
    constexpr uint32_t L1_OUT_BLOCK_N        = BLOCK_N / 2u;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * 1u; // FP8 (1B/elt)
    // NOTES: B is FP4 but stored UNPACKED in LDS (1B per element), mirroring
    // SM100's ``float_e2m1_unpacksmem_t`` (sizeof == 1).  The MFMA reads
    // packed FP4 from VGPRs, but staging in LDS keeps 1 element per byte.
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE   = LOAD_BLOCK_N * BLOCK_K * 1u; // FP4 unpacked-in-LDS
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    (void) SMEM_A_SIZE_PER_STAGE;
    (void) SMEM_B_SIZE_PER_STAGE;
    (void) SMEM_SFA_SIZE_PER_STAGE;
    (void) SMEM_SFB_SIZE_PER_STAGE;
    (void) L1_OUT_BLOCK_N;

    // ---- Shared memory layout.  ``extern __shared__`` block is sized by
    //      the host launcher; we only carve out the regions we need here.
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto      smem_expert_count = reinterpret_cast<uint32_t *>(smem_buffer);
    Mbarrier *smem_barriers =
        reinterpret_cast<Mbarrier *>(smem_buffer + sizeof(uint32_t) * kNumExperts + 16u);
    auto smem_send_buffer = reinterpret_cast<uint8_t *>(smem_barriers + 32u);
    (void) smem_send_buffer;

    // ---- Per-loader-wave LDS staging slot (single stage today).
    //
    // Real cooperative ``(A | B | SFA | SFB | sink)`` carve-out: each
    // loader wave owns
    //   A : kRowsPerLoaderWave × BLOCK_K bytes from l{1,2}_token_buffer
    //   B : kColsPerLoaderWave × BLOCK_K bytes from l{1,2}_weights
    //   SFA / SFB : ``kLoaderSFBytes`` each (placeholder until real
    //               UE8M0 SF loads land - TODO §B.2 deferred item 4).
    //   sink : ``kSubTilesPerWave`` × ``float32x16`` accumulators
    //          drained to LDS at end-of-role so the optimiser cannot
    //          collapse the MFMA chain (numerics still scaffold-only).
    //
    // The MFMA loop walks the *real* output partition the production
    // kernel needs:
    //   for k_block in num_k_blocks:               // outer GEMM K
    //     stage A + B for this (m,n,k) tile via buffer_load_lds
    //     for k_inner in BLOCK_K / 64:             // inner-K of one block
    //       for sub in (kSubTilesM × kSubTilesN):  // wave-owned 32×32 tiles
    //         mfma_scale_f32_32x32x64_f8f6f4(...)
    //
    // Lane→tile-coordinate mapping for the LDS read on the MFMA path
    // is still a scaffold (lane reads its consecutive int32x8 from
    // LDS, which does not match the MFMA's lane→data convention) -
    // tracked as TODO §B.2 deferred item 5.  Switching to
    // ``ds_read_pinned`` with the MFMA convention is a follow-up patch
    // that also unlocks correct numerics once the SFA/SFB loads
    // (deferred item 4) and the SwiGLU/quant/BF16 writeback (deferred
    // item 3) land alongside.
    constexpr uint32_t kMfmaM       = 32u;              // MFMA 32x32x64 output rows
    constexpr uint32_t kMfmaN       = 32u;              // MFMA 32x32x64 output cols
    constexpr uint32_t kMfmaK       = 64u;              // MFMA 32x32x64 K reduction
    constexpr uint32_t kInnerKIters = BLOCK_K / kMfmaK; // 2 for BLOCK_K=128
    static_assert(BLOCK_K % kMfmaK == 0u, "BLOCK_K must be a multiple of MFMA K");
    // Partition the (M, N) output across loader waves along the M axis
    // ONLY.  Per the §B.4 design note, splitting along both axes turns
    // the wave-private output into a checkerboard (wave 0 owns
    // (M0..M_half, N0..N_half); wave 1 owns (M_half..M, N_half..N)) and
    // leaves the off-diagonal quadrants un-computed.  Split-along-M
    // keeps each wave responsible for the full N width of its M slice,
    // which matches both the DG SM100 partition and the writeback the
    // epilogue role expects.
    static_assert(BLOCK_M % kNumMMANonEpilogueWarps == 0u,
                  "BLOCK_M must be evenly partitionable across loader waves");
    constexpr uint32_t kRowsPerLoaderWave = BLOCK_M / kNumMMANonEpilogueWarps; // 64 today
    constexpr uint32_t kColsPerLoaderWave = BLOCK_N;                           // 128 today
    static_assert(kRowsPerLoaderWave % kMfmaM == 0u,
                  "loader wave row partition must be a multiple of MFMA M");
    static_assert(kColsPerLoaderWave % kMfmaN == 0u,
                  "loader wave col partition must be a multiple of MFMA N");
    constexpr uint32_t kSubTilesM       = kRowsPerLoaderWave / kMfmaM;
    constexpr uint32_t kSubTilesN       = kColsPerLoaderWave / kMfmaN;
    constexpr uint32_t kSubTilesPerWave = kSubTilesM * kSubTilesN;

    // Per-tile size in LDS = (rows or cols owned by wave) × BLOCK_K bytes.
    // The cooperative ``buffer_load_lds<16>`` covers 64 lanes × 16 B = 1024 B
    // per call, so the number of loader calls per tile per K iteration is
    // ``kTileBytes / 1024``.  For 64 rows × 128 K bytes this is 8 calls.
    constexpr uint32_t kATileBytes       = kRowsPerLoaderWave * BLOCK_K;
    constexpr uint32_t kBTileBytes       = kColsPerLoaderWave * BLOCK_K;
    constexpr uint32_t kLaneLoadBytes    = 16u; // ds_read_b128 / buffer_load_lds width
    constexpr uint32_t kBytesPerLoadCall = kWarpSize * kLaneLoadBytes; // 1024 B
    static_assert(kATileBytes % kBytesPerLoadCall == 0u,
                  "A tile must be a multiple of one cooperative load call");
    static_assert(kBTileBytes % kBytesPerLoadCall == 0u,
                  "B tile must be a multiple of one cooperative load call");
    constexpr uint32_t kATileLoadsPerWave = kATileBytes / kBytesPerLoadCall;
    constexpr uint32_t kBTileLoadsPerWave = kBTileBytes / kBytesPerLoadCall;

    constexpr uint32_t kLoaderSFBytes = 64u; // placeholder SFA / SFB slot
    constexpr uint32_t kLoaderSinkBytes =
        kSubTilesPerWave * sizeof(dtype::float32x16); // one fragment per sub-tile

    // Per-stage carve-out for one (A | B | SFA | SFB) tile.  The sink
    // is NOT staged - it lives once per wave at the tail of the
    // wave-private carve-out so the persistent accumulator bank is not
    // duplicated across stages.
    constexpr uint32_t kStagedBytesPerStage = kATileBytes + kBTileBytes + kLoaderSFBytes * 2u;
    constexpr uint32_t kLoaderBaseBytes =
        ((sizeof(uint32_t) * kNumExperts + 16u + 32u * sizeof(Mbarrier) + 1023u) / 1024u) * 1024u;

    // kLoaderStages: bounded by both the host's kNumStages and the LDS
    // budget.  Each loader wave's carve-out is
    //   kLoaderStages * kStagedBytesPerStage + kLoaderSinkBytes
    // and the two waves together must fit under ``kSmemBytes -
    // kLoaderBaseBytes``.  For the smoke shape (BLOCK_M=BLOCK_N=BLOCK_K=128,
    // kNumMMANonEpilogueWarps=2): kStagedBytesPerStage = 24704 B, so the
    // 140 KiB budget supports up to 2 stages × 2 waves ≈ 100 KiB; raising
    // to 3 stages would overflow.  We let the host pin kNumStages = 4
    // for the rest of the kernel (scheduler, future writeback) and clamp
    // the loader's effective stage count locally.
    static constexpr uint32_t kSmemBytesBudget = 140u * 1024u;
    static_assert(kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderSinkBytes < kSmemBytesBudget,
                  "loader sink + base already exceed kSmemBytes budget");
    constexpr uint32_t kMaxLoaderStagesByLds =
        (kSmemBytesBudget - kLoaderBaseBytes - kNumMMANonEpilogueWarps * kLoaderSinkBytes) /
        (kNumMMANonEpilogueWarps * kStagedBytesPerStage);
    static_assert(kMaxLoaderStagesByLds >= 1u,
                  "loader staged carve-out too large for one stage even - shrink BLOCK_K?");
    constexpr uint32_t kLoaderStages =
        (kNumStages < kMaxLoaderStagesByLds) ? kNumStages : kMaxLoaderStagesByLds;

    // Per-wave carve-out: kLoaderStages staged (A | B | SFA | SFB) regions
    // followed by one sink region for the persistent accumulator bank.
    constexpr uint32_t kLoaderWaveBytes = kLoaderStages * kStagedBytesPerStage + kLoaderSinkBytes;
    static_assert(kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderWaveBytes <= kSmemBytesBudget,
                  "loader tile carve-out exceeds kSmemBytes budget");

    // Cooperative load count per stage (A loads + B loads), used to size
    // the s_waitcnt vmcnt window for double-buffered prefetch below.
    constexpr uint32_t kLoadsPerStage = kATileLoadsPerWave + kBTileLoadsPerWave;
    // vmcnt field is 6 bits on CDNA - the prefetch window must fit.
    static_assert(kLoadsPerStage < 63u,
                  "kLoadsPerStage exceeds vmcnt range - reduce per-stage loads");

    // Fast-exit path for ``num_tokens == 0``.  The JIT smoke test (see
    // ``tests/pytorch/ops/jit/test_mega_moe_jit_perf.py``) invokes the
    // launcher with zero tokens just to confirm the kernel symbol is
    // resolvable and ``hipLaunchKernelGGL`` actually dispatches.
    //
    // NOTES (TODO.md Section A item 13): the early-return must run
    // BEFORE ``named_bar.init`` (and before the bring-up
    // ``__syncthreads`` below).  ``init`` contains a ``__syncthreads``,
    // which deadlocks the moment a partial-role early exit is added
    // later.  Today every thread sees the same ``num_tokens``, so all
    // 512 threads return together either way; doing the exit here keeps
    // the code defensively correct for the eventual partial-WG case.
    if (num_tokens == 0u) {
        (void) l1_weights;
        (void) l1_weights_sf;
        (void) l2_weights;
        (void) l2_weights_sf;
        (void) sym_buffer;
        (void) cumulative_local_expert_recv_stats;
        (void) y;
        return;
    }

    // Named-barrier (software-emulated PTX ``bar.sync``).  The LDS
    // counter pad lives inside ``init()`` as a static ``__shared__`` —
    // see the per-kernel-singleton note on ``prims::NamedBarrierWg``.
    prims::NamedBarrierWg named_bar;
    named_bar.init(); // includes __syncthreads()

    // Per-role bar_id allocation.  Each ``bar_id`` couples a fixed
    // ``num_threads`` value (the calling role's wave count x warpSize),
    // so distinct ``(num_threads, role)`` tuples must use distinct ids.
    //   0 : dispatch-only internal sync  (num_threads = kNumDispatchThreads)
    //   1 : dispatch + loader + epilogue cross sync (full CTA = kNumThreads)
    //   2 : dispatch role grid_sync / nvlink_barrier internal
    //   3 : epilogue role grid_sync / nvlink_barrier internal
    //   4 : dispatch + epilogue serializer for nvl_barrier_counter access
    //       (disp+epi only - loader has already exited).  Without this,
    //       dispatch's nvlink_barrier-2 (L897) and epi's nvlink_barrier-epi
    //       (L1755) race on the shared nvl_barrier_counter + signal slots:
    //       kBarDispEpi at L868/L1753 forces both roles to arrive
    //       simultaneously, both read the same counter value, both compute
    //       (phase, sign) identically, and both red_add_rel_sys +1 to the
    //       SAME signal[phase] on each remote rank.  Each rank then
    //       receives 2*kNumRanks increments instead of kNumRanks, so the
    //       leader spin ``while(signal != kNumRanks)`` never matches.
    //       On SM100 the heavy MFMA/combine work between the equivalent
    //       cross-role sync and the two nvlink_barriers naturally
    //       desynchronizes them; our scaffold has no such work, so the
    //       race fires reliably under EP>=2.
    constexpr uint32_t kBarDispLocal = 0u;
    constexpr uint32_t kBarDispEpi   = 1u;
    constexpr uint32_t kBarDispGrid  = 2u;
    constexpr uint32_t kBarEpiGrid   = 3u;
    constexpr uint32_t kBarDispEpi2  = 4u;

    // Role leader threads (absolute ``threadIdx.x``).  PTX bar.sync
    // implicitly picks a leader; on AMD we have to name one explicitly
    // because ``thread_idx == 0`` belongs to the dispatch role.
    constexpr uint32_t kDispLeader = 0u;
    constexpr uint32_t kEpiLeader  = (kNumDispatchWarps + kNumMMANonEpilogueWarps) * kWarpSize;

    // Single bring-up sync so the dispatch/loader/MMA/epilogue waves all
    // see a consistent LDS state before they start.
    if (warp_idx == 0u && elect_one()) {
#pragma unroll
        for (uint32_t i = 0; i < kNumExperts; ++i)
            smem_expert_count[i] = 0u;
#pragma unroll
        for (uint32_t i = 0; i < 32u; ++i)
            smem_barriers[i].init(1u);
    }
    __syncthreads();

    // ---- Persistent scheduler instance (per CTA) ----
    auto scheduler = sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K,
                                             L2_SHAPE_N, L2_SHAPE_K, kNumExpertsPerRank,
                                             kNumExpertsPerWave, kNumSMs, kNumRanks>(workspace);

    // =================================================================
    //  Role: dispatch warps.  Pull remote tokens into the local L1 pool
    //  buffer.
    // =================================================================
    if (warp_idx < kNumDispatchWarps) {
        // NOTES: DG uses 32 because SM100 is wave32; on AMD wave64 we
        // pack twice as many (token, topk) pairs per wave so all 64 lanes
        // do useful work.  ``kNumActivateLanes = kNumTokensPerWarp *
        // kNumTopk`` must be <= warpSize.
        constexpr uint32_t kNumTokensPerWarp = kWarpSize / kNumTopk;
        constexpr uint32_t kNumGlobalWarps   = kNumSMs * kNumDispatchWarps;
        static_assert(kNumTokensPerWarp * kNumTopk <= kWarpSize,
                      "kNumTopk does not divide wave size");

        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto         read_topk_idx     = [&](const auto &process) {
// TODO: figure out better unrolling
// Now, `unroll` is better than `unroll 8`
#pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens; i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(__ldg(
                        input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t &token_topk_idx, const int &expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        sync_aligned(named_bar, kNumDispatchThreads, kBarDispLocal);

        // Pass 2: post the per-expert send count into the workspace.
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i]      = static_cast<uint32_t>(
                atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        sync_aligned(named_bar, kNumDispatchThreads, kBarDispLocal);

        // Pass 3: write source ``(token, topk)`` indices.
        for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
             i < num_tokens; i += kNumGlobalWarps * kNumTokensPerWarp) {
            const uint32_t slot = i + (lane_idx / kNumTopk);
            if (slot < num_tokens && lane_idx < kNumTokensPerWarp * kNumTopk) {
                const uint32_t topk_idx_in_token = lane_idx % kNumTopk;
                const int e = static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                                     slot * kNumTopk + topk_idx_in_token));
                if (e >= 0) {
                    const uint32_t dst_rank_idx = e / kNumExpertsPerRank;
                    const uint32_t dst_slot_idx = atomic_add_block(smem_expert_count + e, 1u);
                    auto          *dst_ptr      = workspace.get_src_token_topk_idx_ptr(
                        e % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = slot * kNumTopk + topk_idx_in_token;
                }
            }
            sync_warp();
        }

        // System-scope acq_rel fence: empirically required at EP=8 to
        // prevent a wedge in ``scheduler.fetch_expert_recv_count`` further
        // down.  Without it, the spin loop on the cross-rank
        // ``expert_recv_count_sum`` counter never observes the remote
        // RELEASE atomic_add even though the writer (other ranks' SM0)
        // completed it.  See the note on ``atomic_add_sys`` above.
        // Grid sync, then post per-rank recv counts (only SM0 touches them).
        comm::grid_sync<kNumSMs, 0>(workspace, named_bar, kBarDispGrid, kNumDispatchThreads,
                                    kDispLeader, sm_idx, thread_idx);

        if (sm_idx == 0u) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t dst_rank         = i / kNumExpertsPerRank;
                const uint32_t dst_local_expert = i % kNumExpertsPerRank;
                const uint64_t expert_status    = *workspace.get_expert_send_count_ptr(i);
                // System-scope RELEASE store: the consumer on ``dst_rank``
                // reads this via ``ld_acq_sys``-scope load after observing the
                // counter (below) bumped to ``kNumSMs * kNumRanks``.  A plain
                // store goes through L1/L2 with AGENT coherence only and is
                // not guaranteed visible to the remote agent (XGMI peer)
                // before the RELEASE atomic that follows can flush it.
                // Empirically: with a plain store the np=8 launch wedges in
                // ``scheduler.fetch_expert_recv_count`` because the remote
                // observes the counter complete but reads 0 for this src
                // rank's count.
                __hip_atomic_store(
                    reinterpret_cast<unsigned int *>(sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert),
                        dst_rank)),
                    static_cast<unsigned int>(expert_status & 0xffffffffull), __ATOMIC_RELEASE,
                    __HIP_MEMORY_SCOPE_SYSTEM);
                atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert),
                                   dst_rank),
                    expert_status);
            }
        }
        sync_aligned(named_bar, kNumDispatchThreads, kBarDispLocal);

        // NVLink barrier (collapses to grid sync for kNumRanks==1).
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads, 0, 1>(
            workspace, sym_buffer, named_bar, kBarDispGrid, kDispLeader, sm_idx, thread_idx,
            /*sync_prologue=*/false, /*sync_epilogue=*/true);

        // Pass 4: pull tokens + SFs into the local L1 pool.
        //
        // For each ``token_idx_in_expert`` we run DG's iterative
        // min-peeling round-robin (sm100 impl lines 491-539) to recover
        // the ``(src_rank, slot_in_rank)`` pair.  The algorithm peels
        // rounds of ``length = min(remaining_per_active_rank)`` tokens
        // distributed round-robin across the active ranks until the
        // target slot falls into the current round.  Wave32 primitives
        // (__ballot_sync / __popc / __fns / __reduce_*_sync) translate
        // 1:1 to AMD wave64 via the helpers above.
        //
        // For kNumRanks == 1 this trivially picks rank 0 and
        // slot_in_rank == token_idx_in_expert (one round, one active
        // rank, length == stored_rank_count[0]), so the fast-path
        // numerics are preserved.
        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = (kNumRanks + kWarpSize - 1u) / kWarpSize;

        int      current_expert_idx                  = -1;
        uint32_t expert_start_idx                    = 0u;
        uint32_t expert_end_idx                      = 0u;
        uint32_t expert_pool_block_offset            = 0u;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};

        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx;;
             token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= static_cast<int>(kNumExpertsPerRank))
                    break;
                expert_pool_block_offset +=
                    (expert_end_idx - expert_start_idx + BLOCK_M - 1u) / BLOCK_M;
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= static_cast<int>(kNumExpertsPerRank))
                break;

            // Re-load per-rank recv counts when the expert changes.
            // Lane ``lane_idx`` of iteration ``i`` holds rank
            // ``i * warpSize + lane_idx``'s count for ``current_expert_idx``.
            // The cross-rank writes were published by the prior
            // nvlink_barrier (SYSTEM scope), so a SYSTEM-scope relaxed
            // load is sufficient to observe them.
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    const uint32_t j = i * kWarpSize + lane_idx;
                    if (j < kNumRanks) {
                        const auto raw = __hip_atomic_load(
                            reinterpret_cast<const unsigned long long *>(
                                workspace.get_expert_recv_count_ptr(
                                    j, static_cast<uint32_t>(current_expert_idx))),
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                        stored_rank_count[i] = static_cast<uint32_t>(raw);
                    } else {
                        stored_rank_count[i] = 0u;
                    }
                }
            }

            // Round-robin rank selection via iterative min-peeling.
            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t       remaining[kNumRanksPerLane];
#pragma unroll
            for (uint32_t i = 0u; i < kNumRanksPerLane; ++i)
                remaining[i] = stored_rank_count[i];

            uint32_t current_rank_in_expert_idx = 0u;
            uint32_t token_idx_in_rank          = 0u;
            uint32_t slot_idx                   = token_idx_in_expert;
            uint32_t offset                     = 0u;
            while (true) {
                // Per-lane active count + min-of-remaining (skipping zeros).
                uint32_t num_actives_in_lane = 0u;
                uint32_t min_in_lane         = 0xffffffffu;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    if (remaining[i] > 0u) {
                        ++num_actives_in_lane;
                        if (remaining[i] < min_in_lane)
                            min_in_lane = remaining[i];
                    }
                }
                const uint32_t num_active_ranks = reduce_add(num_actives_in_lane);
                const uint32_t length           = reduce_min(min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    // ``slot_idx_in_round`` selects which active rank
                    // owns this token in the current round (round-robin
                    // ordering inside the round); the remaining
                    // ``slot_idx / num_active_ranks`` indexes into that
                    // rank's strip of the round.
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t       num_seen_ranks    = 0u;
#pragma unroll
                    for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                        const uint64_t mask             = ballot(remaining[i] > 0u);
                        const uint32_t num_active_lanes = popcnt(mask);
                        if (slot_idx_in_round >= num_seen_ranks &&
                            slot_idx_in_round < num_seen_ranks + num_active_lanes) {
                            current_rank_in_expert_idx =
                                i * kWarpSize +
                                nth_set_bit(mask, slot_idx_in_round - num_seen_ranks + 1u);
                        }
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Advance to the next round.
                slot_idx -= num_round_tokens;
                offset += length;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    if (remaining[i] >= length)
                        remaining[i] -= length;
                    else
                        remaining[i] = 0u;
                }
            }

            // The src-token-topk-idx slot was written by the source rank
            // into our workspace via Pass 3 (sym_buffer.map(...,
            // dst_rank=our_rank)), so this load is local (no
            // sym_buffer.map).
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                static_cast<uint32_t>(current_expert_idx), current_rank_in_expert_idx,
                token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            const uint32_t pool_token_idx =
                expert_pool_block_offset * BLOCK_M + token_idx_in_expert;

            // Copy the FP8 token + SF data into the local L1 pool.  The
            // source tensors live in the remote rank's symmetric
            // workspace, so we apply ``sym_buffer.map(...,
            // current_rank_in_expert_idx)`` to translate every load
            // pointer into the remote allocation.  The vectorized
            // ``uint4`` global copy then routes over XGMI transparently
            // (single-rank reduces to a self-map, i.e. a local copy).
            //
            // This is a GMEM -> GMEM transfer (the L1 pool lives in the
            // symmetric workspace, not LDS), so ``buffer_load_lds`` does
            // NOT apply here - that helper is for GMEM -> LDS and is
            // reserved for the A/B loader once its body lands (TODO.md
            // Section B item 2 / Section A item 6).
            auto *src_token_ptr = sym_buffer.map(
                input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint8_t>(),
                current_rank_in_expert_idx);
            auto *dst_token_ptr =
                l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr<uint8_t>();
            for (uint32_t k = lane_idx * 16u; k < kHidden; k += kWarpSize * 16u) {
                if (k + 16u <= kHidden) {
                    auto *src4 = reinterpret_cast<uint4 *>(src_token_ptr + k);
                    auto *dst4 = reinterpret_cast<uint4 *>(dst_token_ptr + k);
                    *dst4      = *src4;
                }
            }

            constexpr uint32_t kNumSFUint32  = kHidden / 128u;
            const auto         remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            auto      *local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            const auto sf_pool_token_idx =
                expert_pool_block_offset * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert);
#pragma unroll
            for (uint32_t i = 0u; i < (kNumSFUint32 + 31u) / 32u; ++i) {
                const uint32_t j = i * 32u + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            sync_warp();

            if (elect_one()) {
                const float weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() =
                    weight;

                *workspace.get_token_src_metadata_ptr(pool_token_idx) = {
                    current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                red_add_rel(workspace.get_l1_arrival_count_ptr(expert_pool_block_offset +
                                                               token_idx_in_expert / BLOCK_M),
                            1u);
            }
            sync_warp();
        }

        // Workspace cleanup for the next launch.  This is a cross-role
        // sync: it waits for the loader/MMA AND epilogue waves to
        // finish using the L1/L2 arrival counters/masks before we wipe
        // them.  Loader threads spin on ``l1_arrival_count`` /
        // ``l2_arrival_mask`` inside ``for_each_block`` (see L1004,
        // L1009); if they have not yet observed ``expected`` for a
        // later block when cleanup zeros the counter, the spin reads 0
        // forever - hipDeviceSynchronize hangs.  ALL three roles
        // (dispatch + loader/MMA + epilogue) must arrive, so the
        // participant count is the full CTA (``kNumThreads``).
        // The matching arrivals are below: loader at end of its
        // for_each_block, epi after its (empty) for_each_block body.
        sync_unaligned(named_bar, kNumThreads, kBarDispEpi);

        // Stats-only pass (no mutation of shared workspace state): the
        // per-expert ``red_add`` into ``cumulative_local_expert_recv_stats``
        // reads ``num_recv_tokens`` from ``recv_count_sum`` and writes to
        // a *separate* user-supplied buffer.  We have to run this
        // BEFORE the resets below, but it does not race with other SMs
        // reading ``recv_count_sum``.
        if (sm_idx != 0u && cumulative_local_expert_recv_stats != nullptr) {
            for (uint32_t i = sm_idx - 1u; i < kNumExpertsPerRank; i += kNumSMs - 1u) {
                if (warp_idx == 1u && elect_one()) {
                    const uint32_t num_recv_tokens =
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
                    red_add(cumulative_local_expert_recv_stats + i,
                            static_cast<int>(num_recv_tokens));
                }
            }
        }

        // Cross-SM/cross-rank barrier.  Once this returns, every rank's
        // every SM has finished all reads of ``expert_recv_count_sum`` /
        // ``expert_recv_count`` (epi side enforces this via its own
        // ``kBarDispEpi`` sync at the matching call site, which gates
        // each SM's dispatch from reaching this point until that SM's
        // epi has exited ``for_each_block``).  Workspace cleanup MUST
        // happen after this point - doing it inside the per-SM
        // ``kBarDispEpi`` window (the original DG SM100 layout) lets
        // SM1..N-1 race ahead and zero ``recv_count_sum[i]`` while SM0
        // is still spinning on it inside ``scheduler.fetch_expert_recv_count``.
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads, 0, 3>(
            workspace, sym_buffer, named_bar, kBarDispGrid, kDispLeader, sm_idx, thread_idx,
            /*sync_prologue=*/true, /*sync_epilogue=*/false);

        // Serializer with epi's nvlink_barrier-epi: forces epi to observe
        // the counter increment from nvlink_barrier-2 above, so the two
        // barriers land on different (phase, sign) pairs and write to
        // different signal slots.  See the bar-id allocation note on
        // ``kBarDispEpi2`` at the top of the kernel.  Participants are
        // disp + epi only (loader has already exited via its kBarDispEpi
        // arrival at L1717).
        sync_unaligned(named_bar, kNumDispatchThreads + kNumEpilogueThreads, kBarDispEpi2);

        // Workspace cleanup for the next launch.  Safe here: the
        // ``nvlink_barrier-2`` above guarantees all SMs (on all ranks)
        // are done reading the per-expert recv counts.
        if (sm_idx == 0u) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0u;
        } else {
            for (uint32_t i = sm_idx - 1u; i < kNumExpertsPerRank; i += kNumSMs - 1u) {
                const uint32_t num_recv_tokens =
                    static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
                const uint32_t num_recv_m_blocks = (num_recv_tokens + BLOCK_M - 1u) / BLOCK_M;
                expert_pool_block_offset         = scheduler.get_pool_block_offset(i);

                if (warp_idx == 0u && elect_one())
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0u;

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0u;
                sync_warp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0u;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j)  = 0ull;
                }
                sync_warp();
            }
        }

        return;
    }

    // =================================================================
    //  Role: GEMM loader / MMA waves.  Drive the per-wave persistent
    //  scheduler ``for_each_block`` and stage A/B/SF tiles for MFMA.
    //
    //  This block intentionally keeps the SM100 control flow intact -
    //  the scheduler walks every (phase, expert, m_block, n_block) tuple
    //  for the current CTA, completing all Linear1 (L1) blocks of a wave
    //  before any Linear2 (L2) block starts.  The MFMA accumulator lives
    //  in AGPRs for the duration of one block.
    // =================================================================
    if (warp_idx >= kNumDispatchWarps && warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Loader-role-local wave index (0-based within the loader band).
        const uint32_t loader_warp_local = warp_idx - kNumDispatchWarps;

        // Per-wave LDS slot offsets.  Lane-invariant uint32_t addresses
        // suitable for ``buffer_load_lds`` / ``ds_read``.
        //
        // Layout (per loader wave):
        //   wave_base
        //     stage 0 : [A | B | SFA | SFB]    (kStagedBytesPerStage)
        //     stage 1 : [A | B | SFA | SFB]
        //     ...
        //     stage S-1
        //     sink    : kSubTilesPerWave × float32x16
        //
        // The stage index walks (0 .. kLoaderStages-1) and wraps as the
        // outer K loop advances - the ring-buffer that the in-flight
        // ``buffer_load_lds`` prefetch uses to overlap stage N+1's GMEM
        // read with stage N's MFMA consumption.
        const uint32_t wave_base_byte = kLoaderBaseBytes + loader_warp_local * kLoaderWaveBytes;
        const uint32_t sink_off       = wave_base_byte + kLoaderStages * kStagedBytesPerStage;

        // LDS-segment base addresses for stage 0 (lane-invariant); other
        // stages are reached by adding ``stage * kStagedBytesPerStage``.
        // ``a_lds_base`` / ``b_lds_base`` are uint32 SMEM addresses
        // suitable for the ``raw.buffer.load.lds`` intrinsic.
        // HIP: ``extern __shared__`` lives in address space 3; the LDS
        // byte offset is just the low 32 bits of the pointer (no CUDA
        // ``__cvta_generic_to_shared`` builtin on AMDGPU).
        const uint32_t a_lds_stage0 =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_buffer + wave_base_byte));
        const uint32_t b_lds_stage0 = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(smem_buffer + wave_base_byte + kATileBytes));

        // BufferSRDs over both the activation pools and weight tensors.
        // ``l{1,2}_token_buffer.get_base_ptr()`` lives inside the
        // symmetric workspace (the dispatch role filled it via
        // ``sym_buffer.map``); ``l{1,2}_weights`` is the per-expert
        // weight tensor passed in by the host.  ``num_bytes`` defaults
        // to ``0xffffffffu`` (no hardware bounds check) — the loader
        // never issues an offset that escapes the corresponding
        // tensor's actual allocation because every per-tile address
        // below is anchored by the scheduler's pool / expert / block
        // indices.
        device::BufferSRD srd_l1_a(l1_token_buffer.get_base_ptr<void>());
        device::BufferSRD srd_l2_a(l2_token_buffer.get_base_ptr<void>());
        device::BufferSRD srd_l1_b(l1_weights);
        device::BufferSRD srd_l2_b(l2_weights);

        // Per-wave accumulator bank.  One float32x16 per output sub-tile
        // owned by this wave (``kSubTilesM × kSubTilesN``).  Lives in
        // registers/AGPRs across all ``for_each_block`` visits so the
        // entire MFMA chain stays observably alive.
        dtype::float32x16 acc[kSubTilesPerWave] = {};

        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            if (phase == sched::BlockPhase::Linear1) {
                // Wait for L1 token arrivals from the dispatch warps.
                const auto    *ptr      = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const uint32_t expected = scheduler.template get_valid_m<false>();
                while (ld_acq(ptr) != expected) {
                }
            } else {
                const auto    *ptr      = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                const uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1ull;
                while (ld_acq_gpu(ptr) != expected) {
                }
            }

            // ----------------------------------------------------------
            //  Real cooperative (A | B) tile staging + MFMA partition.
            //
            //  Per-phase tile geometry:
            //    A row stride (bytes)         = kHidden (L1) | kIntermediateHidden (L2)
            //    B row stride (bytes)         = L1_SHAPE_K (L1) | L2_SHAPE_K (L2)
            //    B per-expert stride (bytes)  = L1_SHAPE_N*L1_SHAPE_K (L1)
            //                                 | L2_SHAPE_N*L2_SHAPE_K (L2)
            //
            //  Each loader wave covers ``kRowsPerLoaderWave`` rows of A
            //  starting at ``pool_block_idx*BLOCK_M + loader_warp_local*
            //  kRowsPerLoaderWave`` and ``kColsPerLoaderWave`` cols of B
            //  starting at ``n_block_idx*BLOCK_N + loader_warp_local*
            //  kColsPerLoaderWave``.
            //
            //  Lane mapping for the cooperative ``buffer_load_lds<16>``:
            //    call C ∈ [0, kATileLoadsPerWave): lane L writes
            //      A row  = C*8 + L/8  at K byte (L%8)*16
            //    same shape for B.
            //  8 lanes/row × 16 B/lane = 128 B = BLOCK_K, exactly one
            //  full row per 8 lanes, 8 rows per 64-lane call, 8 calls
            //  per 64-row tile.
            //
            //  Outer K loop walks the ``num_k_blocks`` BLOCK_K-wide K
            //  iterations the scheduler hands us (kNumL1BlockKs for
            //  Linear1, kNumL2BlockKs for Linear2) and accumulates the
            //  MFMA result into the persistent ``acc[]`` bank.
            //
            //  B is FP4 packed-in-GMEM in production but for the
            //  scaffold pass we treat the weight tensor as 1 B/elt
            //  (matching what the JIT smoke test allocates); the
            //  GMEM→LDS transfer therefore mirrors A's 1 B/elt layout.
            //  FP4 unpacking is part of TODO §B.2 deferred item 1.
            // ----------------------------------------------------------
            const auto &srd_a = (phase == sched::BlockPhase::Linear1) ? srd_l1_a : srd_l2_a;
            const auto &srd_b = (phase == sched::BlockPhase::Linear1) ? srd_l1_b : srd_l2_b;

            const uint32_t a_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? kHidden : kIntermediateHidden;
            const uint32_t b_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? L1_SHAPE_K : L2_SHAPE_K;
            const uint32_t b_expert_stride_bytes = (phase == sched::BlockPhase::Linear1)
                                                       ? L1_SHAPE_N * L1_SHAPE_K
                                                       : L2_SHAPE_N * L2_SHAPE_K;

            // Tile base GMEM byte offsets (uniform across the wave -
            // only the lane's intra-tile address varies per call).
            const uint32_t a_wave_row0 =
                pool_block_idx * BLOCK_M + loader_warp_local * kRowsPerLoaderWave;
            const uint32_t a_tile_base_bytes = a_wave_row0 * a_row_stride_bytes;
            // Each loader wave covers the full N width of B for its M
            // slice (split-along-M partition); n_block_idx selects which
            // BLOCK_N stripe of the per-expert weight tensor this CTA is
            // working on, but both waves load the same B columns.
            // Mutable so the SwiGLU pairing below (Linear1 only) can
            // override the N-stripe between the gate and up passes.
            // Linear2 and the single-pass Linear1 fallback leave this
            // at its n_block_idx-derived value.
            uint32_t b_wave_col0 = n_block_idx * BLOCK_N;
            uint32_t b_tile_base_bytes =
                local_expert_idx * b_expert_stride_bytes + b_wave_col0 * b_row_stride_bytes;

            // E8M0 scales of 1.0 packed into a uint32_t (0x7f per byte).
            // Kept as a fall-back when an out-of-range SF address would
            // be queried (e.g. the m_in_block / n_global guards below).
            constexpr uint32_t kScaleOne = 0x7f7f7f7fu;

            // ------------------------------------------------------------
            //  Per-tile SFA / SFB pointers (TODO §B.2 deferred item 3).
            //
            //  The MFMA-issue role consumes one E8M0 scale byte per
            //  ``mfma_scale_f32_32x32x64_f8f6f4`` lane per 32 K elements.
            //  Per Triton's ``chooseScaledMfmaScaleLayout``
            //  (lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
            //  function body around the ``lanes = LinearLayout({{kLane,
            //  {{0,1},{0,2},{0,4},{0,8},{0,16},{1,0}}}, ...)``
            //  construction), the per-lane scale partition for fp8 single-
            //  rate at MFMA 32x32x64 is
            //
            //      m_scale = lane & 31
            //      k_scale = (lane >> 5) & 1       // 0..1 inside one
            //                                       // 64-wide K chunk
            //
            //  i.e. lanes [0..31] take the K[0:32] half of A's 32 rows
            //  and lanes [32..63] take K[32:64].  One scale byte per
            //  (lane, sub_m, k_inner) for A; same shape for B with the
            //  N axis replacing M.
            //
            //  Source pointers per phase:
            //    Linear1 A SF   :  ``l1_sf_buffer`` (pool buffer, written
            //                       by the dispatch role with the
            //                       ``transform_sf_token_idx`` transpose).
            //                       Layout per uint32: index by
            //                         (k_byte / 128) * kNumPaddedSFPoolTokens
            //                         + transform_sf_token_idx(token_in_block)
            //                       Each uint32 packs 4 E8M0 bytes
            //                       (one per 32 K elements).
            //    Linear1 B SF   :  ``l1_weights_sf`` (per-expert tensor).
            //                       Production layout: ``(E, N, K/32)``
            //                       of E8M0 bytes.  The smoke test
            //                       allocates ``float`` (4 B/scale) for
            //                       simplicity - the byte we read lands
            //                       inside that 4× larger buffer, so the
            //                       address is in-bounds even though the
            //                       scale value itself is FP32 mantissa
            //                       garbage.  Numerics-gate work that
            //                       reconciles the test's allocation
            //                       width with the production format is
            //                       tracked alongside the Python
            //                       reference GEMM (TODO §B.2 follow-up).
            //    Linear2 A SF   :  ``l2_sf_buffer`` (pool buffer, written
            //                       by the Linear1 writeback below).
            //                       Same layout shape as Linear1 SFA
            //                       with kIntermediateHidden replacing
            //                       kHidden.
            //    Linear2 B SF   :  ``l2_weights_sf`` (per-expert tensor).
            // ------------------------------------------------------------
            const auto    *sfa_pool_base = (phase == sched::BlockPhase::Linear1)
                                               ? l1_sf_buffer.get_base_ptr<uint32_t>()
                                               : l2_sf_buffer.get_base_ptr<uint32_t>();
            const uint8_t *sfb_weights_base =
                (phase == sched::BlockPhase::Linear1)
                    ? reinterpret_cast<const uint8_t *>(l1_weights_sf)
                    : reinterpret_cast<const uint8_t *>(l2_weights_sf);
            const uint32_t sfa_pool_token_idx_base = pool_block_idx * SF_BLOCK_M;
            const uint32_t sfb_n_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? L1_SHAPE_K / kGranK : L2_SHAPE_K / kGranK;
            const uint32_t sfb_expert_stride_bytes = (phase == sched::BlockPhase::Linear1)
                                                         ? (L1_SHAPE_N * L1_SHAPE_K) / kGranK
                                                         : (L2_SHAPE_N * L2_SHAPE_K) / kGranK;
            // Mutable for the same reason as ``b_tile_base_bytes``.
            uint32_t sfb_n_global_base = n_block_idx * BLOCK_N;

            // ------------------------------------------------------------
            //  Ring-buffered ``buffer_load_lds`` -> MFMA pipeline.
            //
            //  We carry ``kLoaderStages`` LDS slots per wave and prefetch
            //  stage N+1's GMEM into LDS while stage N's MFMA chain runs.
            //  Concretely, for each outer K iteration ``k_block`` we
            //
            //    1. wait until stage ``k_block``'s GMEM transfer is done
            //       AND its LDS write has retired (lgkmcnt drain),
            //    2. ds_read stage ``k_block``'s tile into VGPRs,
            //    3. issue stage ``k_block + kLoaderStages``'s GMEM read
            //       BEFORE running MFMA so the GMEM bandwidth overlaps
            //       the MFMA latency (vmcnt counter absorbs the
            //       in-flight prefetch),
            //    4. run the inner-K × sub-tile MFMA chain consuming the
            //       VGPRs from step 2.
            //
            //  The initial fill of stages [0 .. min(num_k_blocks,
            //  kLoaderStages) - 1] happens before the consumer loop
            //  starts.  Lane→tile-coordinate mapping for the MFMA
            //  operand is still scaffold-quality (the ``ds_read``
            //  ignores the MFMA's lane→data convention, TODO §B.2
            //  deferred item 4); switching to ``ds_read_pinned`` with
            //  the MFMA convention is the follow-up patch that also
            //  unlocks correct numerics.
            // ------------------------------------------------------------
            const auto issue_stage = [&](uint32_t stage, uint32_t k_block) {
                const uint32_t a_k_offset_bytes = k_block * BLOCK_K;
                const uint32_t b_k_offset_bytes = k_block * BLOCK_K;
                const uint32_t a_lds            = a_lds_stage0 + stage * kStagedBytesPerStage;
                const uint32_t b_lds            = b_lds_stage0 + stage * kStagedBytesPerStage;
#pragma unroll
                for (uint32_t c = 0u; c < kATileLoadsPerWave; ++c) {
                    const uint32_t m_in_wave      = c * 8u + lane_idx / 8u;
                    const uint32_t k_byte_in_tile = (lane_idx % 8u) * kLaneLoadBytes;
                    const uint32_t ldg_offset = a_tile_base_bytes + m_in_wave * a_row_stride_bytes +
                                                a_k_offset_bytes + k_byte_in_tile;
                    const uint32_t lds_offset =
                        device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                            0u, m_in_wave, k_byte_in_tile);
                    device::load_gmem_to_smem_srd<16>(srd_a, ldg_offset, a_lds + lds_offset,
                                                      /*soffset=*/0);
                }
#pragma unroll
                for (uint32_t c = 0u; c < kBTileLoadsPerWave; ++c) {
                    const uint32_t n_in_wave      = c * 8u + lane_idx / 8u;
                    const uint32_t k_byte_in_tile = (lane_idx % 8u) * kLaneLoadBytes;
                    const uint32_t ldg_offset = b_tile_base_bytes + n_in_wave * b_row_stride_bytes +
                                                b_k_offset_bytes + k_byte_in_tile;
                    const uint32_t lds_offset =
                        device::b_tile_smem_byte_offset<kColsPerLoaderWave, BLOCK_K>(
                            0u, n_in_wave, k_byte_in_tile);
                    device::load_gmem_to_smem_srd<16>(srd_b, ldg_offset, b_lds + lds_offset,
                                                      /*soffset=*/0);
                }
            };

            // ------------------------------------------------------------
            //  K-loop wrapped in a closure so the SwiGLU pairing path
            //  below (Linear1 only) can invoke it twice -- once for the
            //  gate N-stripe and once for the up N-stripe -- with the
            //  same SMEM ring buffer and the same persistent ``acc[]``
            //  bank.  The closure captures by reference, so callers can
            //  mutate ``b_tile_base_bytes`` / ``sfb_n_global_base``
            //  between invocations to retarget the B side.  Linear2
            //  invokes it once with the n_block_idx-derived defaults.
            // ------------------------------------------------------------
            auto run_k_loop = [&]() {
            // Reset the accumulator bank to zero for this pass.
            // ``acc[]`` is persistent across scheduler iterations so
            // the optimiser cannot collapse the MFMA chain, but we
            // need a fresh zero start every time we walk K --
            // whether that is a new (m_block, n_block) iteration or
            // the up-pass of the dual-pass SwiGLU pairing.
#pragma unroll
                for (uint32_t s = 0u; s < kSubTilesPerWave; ++s)
                    acc[s] = dtype::float32x16{};

                // Prime stage 0.  Subsequent stages are prefetched at
                // the tail of each consumer iteration so stage N+1's
                // GMEM read overlaps with stage N's MFMA chain.
                if (num_k_blocks > 0u)
                    issue_stage(0u, 0u);

                for (uint32_t k_block = 0u; k_block < num_k_blocks; ++k_block) {
                    const uint32_t this_stage = k_block % kLoaderStages;
                    const uint32_t a_stage_byte =
                        wave_base_byte + this_stage * kStagedBytesPerStage;
                    const uint32_t b_stage_byte = a_stage_byte + kATileBytes;

                    // Wait for THIS stage's prefetched tile to land in LDS.
                    // AMD ``buffer_load_lds`` decrements BOTH ``vmcnt`` and
                    // ``lgkmcnt`` at LDS-write completion (CDNA ISA: the
                    // GMEM-side completion is the vector-memory event AND
                    // the LDS-write completion is the lgkm event for the
                    // same load), so a single ``vmcnt(0)`` here is enough
                    // to guarantee the upcoming ds_reads see the freshly
                    // written bytes.  No ``lgkmcnt(0)`` drain is needed.
                    //
                    // The next-stage prefetch issued AFTER the ds_reads
                    // below stays in flight (its loads sit in vmcnt while
                    // the MFMA chain runs); the next iteration's
                    // ``vmcnt(0)`` waits only on it specifically.
                    //
                    // hipcc's AMDGPUInsertWaitcnts pass tracks the builtin
                    // ds_read precisely, so the ``lgkmcnt(N)`` it inserts
                    // before the first MFMA waits only on THIS stage's
                    // ds_reads -- the prefetch's lgkmcnt scoreboard slot is
                    // newer and stays pending across MFMA issue.  No
                    // ``sync_warp`` either: the loader role is single
                    // wave-per-role and wave64 is naturally lockstep.
                    device::wait_vmcnt<0>();

                    // Issue the next prefetch (stage ``k_block + 1``) into
                    // its ring slot BEFORE running MFMA.  The GMEM
                    // transaction queues behind the MFMA chain below; even
                    // with the pessimistic drain above, ``vmcnt`` lets the
                    // GMEM bus stay busy while MFMA runs.
                    if (k_block + 1u < num_k_blocks) {
                        issue_stage((k_block + 1u) % kLoaderStages, k_block + 1u);
                    }

                    // Inner-K + sub-tile MFMA partition with the
                    // ``mfma_scale_f32_32x32x64_f8f6f4`` operand convention.
                    //
                    // Per Triton's ``chooseScaledMfmaScaleLayout`` (the
                    // authoritative AMD MFMA scaled-fp8 layout reference, see
                    // ``Triton-distributed/3rdparty/triton/lib/Dialect/
                    // TritonGPU/IR/LinearLayoutConversions.cpp``):
                    //
                    //   for 32x32x64 f8f6f4, each lane takes 32 K elements
                    //   from A (and from B, treating its M as N).  Lanes
                    //   [0..31] collectively handle A[0..31][0..31]; lanes
                    //   [32..63] handle A[0..31][32..63].  Per-lane:
                    //     m_in_subtile = lane & 31
                    //     k_base       = ((lane >> 5) & 1) * 32
                    //   The lane then consumes 32 consecutive K bytes
                    //   starting at ``k_base``.
                    //
                    // Because the LDS is 128 B / 64-bank swizzled with a
                    // 16 B quantisation (``swizzle_offset_128b_64bank``), the
                    // 32 K bytes per lane straddle TWO 16 B swizzle chunks
                    // that are NOT guaranteed to land contiguously in LDS.
                    // Issue two ``ds_read_b128`` (one per chunk) per lane and
                    // concatenate them into the ``int32x8`` operand.  This is
                    // the gfx950 analogue of CUTLASS's two-half-tile read for
                    // SM100 swizzle.
                    //
                    // The output partition is (sub_m, sub_n) where sub_m ∈
                    // [0, kSubTilesM) and sub_n ∈ [0, kSubTilesN).  A is
                    // reused across sub_n (load once per sub_m); B is reused
                    // across sub_m (load once per sub_n); the MFMA grid then
                    // walks all kSubTilesM × kSubTilesN tiles.
                    auto read_int32x8 = [&](uint32_t base_byte, uint32_t off_lo, uint32_t off_hi) {
                        dtype::int32x8 v;
                        const auto     lo =
                            *reinterpret_cast<const uint4 *>(smem_buffer + base_byte + off_lo);
                        const auto hi =
                            *reinterpret_cast<const uint4 *>(smem_buffer + base_byte + off_hi);
                        reinterpret_cast<uint4 *>(&v)[0] = lo;
                        reinterpret_cast<uint4 *>(&v)[1] = hi;
                        return v;
                    };

#pragma unroll
                    for (uint32_t k_inner = 0u; k_inner < kInnerKIters; ++k_inner) {
                        const uint32_t k_base_in_subtile =
                            k_inner * kMfmaK + ((lane_idx >> 5u) & 1u) * 32u;
                        const uint32_t m_in_subtile = lane_idx & 31u;

                        dtype::int32x8 a_vec[kSubTilesM];
#pragma unroll
                        for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                            const uint32_t m_in_wave = sub_m * kMfmaM + m_in_subtile;
                            const uint32_t off_lo =
                                device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                    0u, m_in_wave, k_base_in_subtile);
                            const uint32_t off_hi =
                                device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                    0u, m_in_wave, k_base_in_subtile + 16u);
                            a_vec[sub_m] = read_int32x8(a_stage_byte, off_lo, off_hi);
                        }

                        dtype::int32x8 b_vec[kSubTilesN];
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t n_in_wave = sub_n * kMfmaN + m_in_subtile;
                            const uint32_t off_lo =
                                device::b_tile_smem_byte_offset<kColsPerLoaderWave, BLOCK_K>(
                                    0u, n_in_wave, k_base_in_subtile);
                            const uint32_t off_hi =
                                device::b_tile_smem_byte_offset<kColsPerLoaderWave, BLOCK_K>(
                                    0u, n_in_wave, k_base_in_subtile + 16u);
                            b_vec[sub_n] = read_int32x8(b_stage_byte, off_lo, off_hi);
                        }

                        // Per-lane SF bytes for this k_inner.  Each lane
                        // covers 32 K elements of A (and 32 K elements of B
                        // along its N axis), so we read ONE byte per
                        // (lane, sub_m, k_inner) for SFA and ONE per
                        // (lane, sub_n, k_inner) for SFB.  The MFMA only
                        // looks at the low byte of the uint32_t when both
                        // operand types are fp8 single-rate; the upper 3
                        // bytes are don't-care (set to 0x7f = E8M0 1.0 so
                        // any latent broadcast on different op-type combos
                        // stays well-defined).
                        //
                        // Outer K offset (in bytes) of this MFMA's K[0:32]
                        // half.  k_block * BLOCK_K + k_inner * 64 + 32 for
                        // lanes with (lane >> 5) == 1, +0 otherwise.
                        const uint32_t k_byte_outer =
                            k_block * BLOCK_K + k_inner * kMfmaK + ((lane_idx >> 5u) & 1u) * 32u;
                        // SFA pool dword index: each uint32_t covers 128
                        // K elements (4 E8M0 bytes); for BLOCK_K == 128 the
                        // outer k_block selects the dword index.  Byte
                        // inside the dword is the 32-K group inside the
                        // 128-K block: ``(k_byte_outer & 127) / 32``.
                        const uint32_t sfa_dword_idx     = k_byte_outer / 128u;
                        const uint32_t sfa_byte_in_dword = ((k_byte_outer & 127u) >> 5u);

                        uint32_t sfb_byte[kSubTilesN];
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t n_in_wave = sub_n * kMfmaN + m_in_subtile;
                            // Split-along-M partition: both loader waves
                            // cover the full N width of the BLOCK_N stripe,
                            // so n_in_block == n_in_wave (no per-wave N
                            // offset).
                            const uint32_t n_global = sfb_n_global_base + n_in_wave;
                            const uint32_t sfb_byte_offset =
                                local_expert_idx * sfb_expert_stride_bytes +
                                n_global * sfb_n_stride_bytes + k_byte_outer / kGranK;
                            const uint8_t raw = __ldg(sfb_weights_base + sfb_byte_offset);
                            sfb_byte[sub_n]   = 0x7f7f7f00u | static_cast<uint32_t>(raw);
                        }

#pragma unroll
                        for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                        sub_m * kMfmaM + m_in_subtile;
                            const uint32_t sf_token_idx =
                                sfa_pool_token_idx_base + transform_sf_token_idx(m_in_block);
                            const uint32_t sfa_dword =
                                __ldg(sfa_pool_base + sfa_dword_idx * kNumPaddedSFPoolTokens +
                                      sf_token_idx);
                            const uint32_t sfa_raw =
                                (sfa_dword >> (sfa_byte_in_dword * 8u)) & 0xffu;
                            const uint32_t scale_a = 0x7f7f7f00u | sfa_raw;
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const uint32_t sub = sub_m * kSubTilesN + sub_n;
                                acc[sub]           = device::mfma_scale_f32_32x32x64_f8f6f4<
                                              __hip_fp8_e4m3, __hip_fp8_e4m3>::run(a_vec[sub_m], b_vec[sub_n],
                                                                                   acc[sub], scale_a,
                                                                                   sfb_byte[sub_n]);
                            }
                        }
                    }
                } // end of for (k_block) inside run_k_loop
            }; // end of run_k_loop lambda

            // ------------------------------------------------------------
            //  SwiGLU pairing dispatch (TODO §B.2 deferred follow-up).
            //
            //  Linear1's output is gate||up over a 2*kIntermediateHidden
            //  N dimension; the L2 pool slot consumed by Linear2 is only
            //  kIntermediateHidden wide and holds ``silu(gate)*up`` per
            //  element.  Two scheduler n_block callbacks are needed to
            //  produce one L2 pool row -- the lower half holds gate, the
            //  upper half holds up.  These two callbacks land on
            //  DIFFERENT SMs (the scheduler hands every n_block_idx of
            //  the same (m_block, expert) to a separate SM via
            //  block_idx += kNumSMs), so we cannot pass partial sums
            //  between them in registers.
            //
            //  Solution: collapse both halves into the same callback by
            //  having the gate-half callback (n_block_idx in
            //  [0, kL1GateNBlocks)) walk the K loop twice -- once for
            //  the gate B-stripe, once for the up B-stripe -- and the
            //  up-half callback (n_block_idx >= kL1GateNBlocks) return
            //  early.  Total compute and GMEM traffic are unchanged
            //  (the original placeholder code computed gate then threw
            //  it away).  SM utilisation in the L1 phase drops to ~50 %
            //  because half the n_block callbacks no-op, but this is
            //  amortised over the many (m_block, expert) iterations
            //  each SM handles in a persistent CTA lifetime.
            //
            //  The L2 arrival mask is set for BOTH bits (gate and up)
            //  at the end of the writeback so the Linear2 wait
            //  ``mask == (1<<kNumL1BlockNs)-1`` continues to fire.
            // ------------------------------------------------------------
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            static_assert(kNumL1BlockNs >= 2u && kNumL1BlockNs % 2u == 0u,
                          "L1 N must split evenly into gate||up halves");
            static_assert(kIntermediateHidden % BLOCK_N == 0u,
                          "BLOCK_N must divide kIntermediateHidden for SwiGLU pairing");
            constexpr uint32_t kL1GateNBlocks = kNumL1BlockNs / 2u;

            dtype::float32x16 acc_gate[kSubTilesPerWave];
            bool              has_gate = false;
            if (phase == sched::BlockPhase::Linear1) {
                // Up-half callback is paired into the gate-half callback
                // -- skip without touching ``acc[]`` (the next iteration
                // of for_each_block will reset it).
                if (n_block_idx >= kL1GateNBlocks)
                    return;

                // Pass 1: gate stripe (current b_tile_base_bytes /
                // sfb_n_global_base already point at the gate N offset
                // because n_block_idx is in [0, kL1GateNBlocks)).
                run_k_loop();

#pragma unroll
                for (uint32_t s = 0u; s < kSubTilesPerWave; ++s)
                    acc_gate[s] = acc[s];
                has_gate = true;

                // Pass 2: up stripe.  Shift B by kIntermediateHidden
                // columns; same K, same A.
                b_wave_col0 += kIntermediateHidden;
                b_tile_base_bytes += kIntermediateHidden * b_row_stride_bytes;
                sfb_n_global_base += kIntermediateHidden;
                run_k_loop();
            } else {
                run_k_loop();
            }
            (void) has_gate;

            // ============================================================
            //  Writeback (TODO §B.2 deferred item 2).  Drain the
            //  ``acc[]`` AGPR bank to GMEM per the MFMA 32×32 output
            //  layout from Triton's ``mfmaToLinearLayout``
            //  (LinearLayoutConversions.cpp, non-transposed branch):
            //
            //      LinearLayout({{kRegister, {{0,1},{0,2},{0,8},{0,16}}},
            //                    {kLane,    {{1,0},{2,0},{4,0},{8,0},
            //                                {16,0},{0,4}}}},
            //                   {kOutM, kOutN});
            //
            //  Output dim order is (M, N).  Per lane ``l``, per register
            //  ``i ∈ [0, 16)``:
            //      M = lane & 31
            //      N = ((lane >> 5) & 1) * 4 + kNPattern[i]
            //      kNPattern[i] = expand 4-bit i with bits 0..3 mapping
            //                     to N offsets {1, 2, 8, 16} respectively
            //                  = {0, 1, 2, 3,
            //                     8, 9, 10, 11,
            //                     16,17,18,19,
            //                     24,25,26,27}
            //
            //  Per phase:
            //    Linear1  -> SwiGLU(gate, up) per element, FP8 quantise
            //                with E8M0 1.0 scale, store to the L2 pool
            //                slot (m_in_block × kIntermediateHidden).
            //                ``acc_gate[sub][i]`` holds the gate MFMA
            //                accumulator from the first run_k_loop pass
            //                above; ``acc[sub][i]`` holds the up MFMA
            //                accumulator from the second pass.  Compute
            //                ``silu(gate) * up`` per (sub, i), cast to
            //                ``__hip_fp8_e4m3``, and write to the pool
            //                column ``n_block_idx*BLOCK_N + n_in_wave``
            //                (every gate-half callback owns one stripe
            //                of width BLOCK_N).  Mirrors DG's SM100
            //                SwiGLU body (sm100_fp8_fp4_mega_moe.cuh)
            //                without the bf16 round trip -- the FP32
            //                silu suffices here because the bf16 cast
            //                is a precision detail bound to the
            //                production SM100 TMEM_LOAD layout, not a
            //                correctness requirement.
            //                Real UE8M0 per-block scale (vs the fixed
            //                1.0 placeholder) lands alongside the
            //                Python reference numerics gate, see TODO
            //                §B.2 follow-up.
            //
            //    Linear2  -> per-element BF16 cast, push into the
            //                combine buffer.  Destination is the source
            //                token's combine slot on the source rank:
            //                ``combine_token_buffer.get_rank_buffer(
            //                meta.topk_idx).get_data_buffer(
            //                meta.token_idx)`` translated through
            //                ``sym_buffer.map(..., meta.rank_idx)``.
            //                The epilogue's combine pass reads this
            //                buffer once the cross-rank push is
            //                complete.  No explicit loader→epilogue
            //                sync is required for the smoke test
            //                (numerics gate already deferred); a
            //                production patch will add a
            //                ``red_add_rel(workspace.get_l1_arrival
            //                _count_ptr, kFinishFlag)``-style signal
            //                so the combine read does not race.
            // ============================================================
            {
                constexpr uint32_t kNPattern[16] = {
                    0u, 1u, 2u, 3u, 8u, 9u, 10u, 11u, 16u, 17u, 18u, 19u, 24u, 25u, 26u, 27u,
                };
                const uint32_t m_lane      = lane_idx & 31u;
                const uint32_t n_lane_half = ((lane_idx >> 5u) & 1u) * 4u;
                const uint32_t valid_m     = scheduler.template get_valid_m<false>();

                if (phase == sched::BlockPhase::Linear1) {
                    // SwiGLU pairing: acc_gate holds gate, acc holds up.
                    // Write ``silu(gate)*up`` per element into the L2
                    // pool slot column ``n_block_idx*BLOCK_N+n_in_wave``
                    // (already in [0, kIntermediateHidden) because we
                    // early-returned for n_block_idx >= kL1GateNBlocks).
                    auto *l2_pool_base = l2_token_buffer.get_base_ptr<uint8_t>();

#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                        const uint32_t m_in_block =
                            loader_warp_local * kRowsPerLoaderWave + sub_m * kMfmaM + m_lane;
                        if (m_in_block >= valid_m)
                            continue;
                        const uint32_t pool_token_idx = pool_block_idx * BLOCK_M + m_in_block;
                        auto          *dst_row =
                            l2_pool_base + pool_token_idx * fp8_intermediate_token_layout.num_bytes;
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub = sub_m * kSubTilesN + sub_n;
#pragma unroll
                            for (uint32_t i = 0u; i < 16u; ++i) {
                                const uint32_t n_in_wave =
                                    sub_n * kMfmaN + n_lane_half + kNPattern[i];
                                const uint32_t intermediate_col = n_block_idx * BLOCK_N + n_in_wave;
                                if (intermediate_col >= kIntermediateHidden)
                                    continue;
                                const float gate = acc_gate[sub][i];
                                const float up   = acc[sub][i];
                                // silu(x) = x / (1 + exp(-x)).  Use the
                                // fast ``__expf`` HIP intrinsic; DG's
                                // SM100 path does the same.  No clamp
                                // here -- the production clamp lives in
                                // the Python wrapper alongside the
                                // UE8M0 scale.
                                const float          silu_gate = gate / (1.0f + __expf(-gate));
                                const float          swiglu    = silu_gate * up;
                                const __hip_fp8_e4m3 quant = static_cast<__hip_fp8_e4m3>(swiglu);
                                dst_row[intermediate_col] =
                                    reinterpret_cast<const uint8_t &>(quant);
                            }
                        }
                    }

                    // Mark BOTH the gate (n_block_idx) and up
                    // (n_block_idx + kL1GateNBlocks) stripes complete.
                    // The Linear2 K-loop waits for
                    // ``l2_arrival_mask == (1ull << kNumL1BlockNs) - 1``
                    // before consuming the L2 pool; we own both halves
                    // of this (m_block, intermediate_col_stripe)
                    // because the up-half callback short-circuits.
                    __threadfence();
                    if (elect_one()) {
                        constexpr uint64_t kGateBit  = 1ull;
                        const uint64_t     pair_mask = (kGateBit << n_block_idx) |
                                                   (kGateBit << (n_block_idx + kL1GateNBlocks));
                        red_or_rel_gpu(workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                                       pair_mask);
                    }
                } else {
                    // Linear2 -> combine buffer.  Per token, look up the
                    // source rank / topk / token_idx and push the BF16
                    // partial sum into the source rank's symmetric slot.
                    const auto write_combine = [&](uint32_t m_in_block, uint32_t n_global,
                                                   float val) {
                        const auto meta = *workspace.get_token_src_metadata_ptr(
                            pool_block_idx * BLOCK_M + m_in_block);
                        auto *dst_local = combine_token_buffer.get_rank_buffer(meta.topk_idx)
                                              .get_data_buffer(meta.token_idx)
                                              .get_base_ptr<uint8_t>();
                        auto                *dst_remote = sym_buffer.map(dst_local, meta.rank_idx);
                        const __hip_bfloat16 bf         = __float2bfloat16(val);
                        *reinterpret_cast<__hip_bfloat16 *>(dst_remote +
                                                            n_global * sizeof(__hip_bfloat16)) = bf;
                    };

#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                        const uint32_t m_in_block =
                            loader_warp_local * kRowsPerLoaderWave + sub_m * kMfmaM + m_lane;
                        if (m_in_block >= valid_m)
                            continue;
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub = sub_m * kSubTilesN + sub_n;
#pragma unroll
                            for (uint32_t i = 0u; i < 16u; ++i) {
                                const uint32_t n_in_wave =
                                    sub_n * kMfmaN + n_lane_half + kNPattern[i];
                                const uint32_t n_global = n_block_idx * BLOCK_N + n_in_wave;
                                if (n_global >= kHidden)
                                    continue;
                                write_combine(m_in_block, n_global, acc[sub][i]);
                            }
                        }
                    }
                }
            }
        });

        // Sink every sub-tile accumulator into LDS so the optimiser
        // cannot collapse the MFMA loop down to its first iteration.
        // Only lane 0 writes to keep the sink slot from being a hot
        // bank - the entire wave's accumulator bank is observable via
        // that single lane because the wave's MFMA chain depends on
        // all 64 lanes (each lane contributes its operand fragment to
        // every MFMA call).
        if (elect_one()) {
            auto *sink_ptr = reinterpret_cast<dtype::float32x16 *>(smem_buffer + sink_off);
#pragma unroll
            for (uint32_t sub = 0u; sub < kSubTilesPerWave; ++sub)
                sink_ptr[sub] = acc[sub];
        }

        // Cross-role gate paired with the dispatch cleanup at L865 and
        // the epi side at the bottom of its for_each_block.  Holding
        // the loader here until cleanup has fired is harmless (the
        // loader is exiting anyway); the load-bearing direction is
        // that the dispatch cleanup cannot zero ``l1_arrival_count`` /
        // ``l2_arrival_mask`` until every loader wave on this CTA has
        // observed its last ``expected`` value.  Without this arrive,
        // the loader's tight spin (L1004 / L1009) can outlive the
        // cleanup's plain-store zeros and read 0 forever.
        sync_unaligned(named_bar, kNumThreads, kBarDispEpi);
        return;
    }

    // =================================================================
    //  Role: epilogue + combine waves.  Drives the SwiGLU + UE8M0 quant
    //  (L1) and the BF16 NVLink write (L2) paths, then the cross-topk
    //  reduce + write-back loop.  Both paths are driven by the SAME
    //  ``scheduler.for_each_block`` so the per-wave scheduling holds.
    // =================================================================
    if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
            // Mirror the SM100 epilogue divergence: Linear1 -> SwiGLU
            // path (writes into l1/l2 pools), Linear2 -> BF16 push path
            // (writes into the combine buffer).  We only need to mark
            // arrivals on the workspace at this alignment stage; the
            // tile-level math will be filled in by the next patch.
            // L2 arrival mask used to be set here as a placeholder so
            // the loader's Linear2 K-loop wait would not deadlock; the
            // loader role now sets the bit at the end of its Linear1
            // writeback (after the actual L2 pool data is in place),
            // so the epilogue's per-block callback no longer needs to.
            (void) phase;
            (void) num_k_blocks;
            (void) local_expert_idx;
            (void) m_block_idx;
            (void) n_block_idx;
        });

        // Pair the dispatch cleanup cross-sync (``kBarDispEpi``) so the
        // dispatch waves can safely wipe the L1/L2 arrival counters
        // once every epilogue AND loader/MMA wave has exited
        // ``for_each_block``.  Participant count is the full CTA
        // (``kNumThreads``); see the dispatch-side comment above L865.
        sync_unaligned(named_bar, kNumThreads, kBarDispEpi);

        // Serializer with dispatch's nvlink_barrier-2: blocks epi until
        // dispatch's nvlink_barrier-2 has incremented the shared
        // nvl_barrier_counter, so this rank's nvlink_barrier-epi below
        // reads the post-increment value and lands on the next
        // (phase, sign) pair.  See kBarDispEpi2 allocation note.
        sync_unaligned(named_bar, kNumDispatchThreads + kNumEpilogueThreads, kBarDispEpi2);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads, 1, 2>(
            workspace, sym_buffer, named_bar, kBarEpiGrid, kEpiLeader, sm_idx, thread_idx,
            /*sync_prologue=*/true, /*sync_epilogue=*/true);

        // ----------------------------------------------------------------
        // Combine: per-token reduce across topk into the output buffer.
        //
        // DG's SM100 path strides the per-token uint4 work across all 32
        // wave32 lanes (``kNumUint4PerLane = kNumChunkUint4 / 32``).  The
        // earlier AMD port left the inner loop guarded by ``lane_idx == 0``,
        // which serialised all kNumUint4PerToken loads + FP32 adds onto
        // a single lane per wave64 - a 64x slowdown on hidden=7168 (896
        // uint4 elements/token).  We mirror DG's stride pattern with
        // wave64 granularity so every lane reduces its share of the
        // token's uint4 chunks.  ``mask`` is uniform across the wave
        // (computed via ``ballot``), so the per-lane while loop visits
        // the same set of expert ranks on every lane.
        // ----------------------------------------------------------------
        const uint32_t epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(__hip_bfloat162);
        constexpr uint32_t kNumUint4PerToken = (kHidden * sizeof(__hip_bfloat16)) / sizeof(uint4);
        static_assert(
            (kHidden * sizeof(__hip_bfloat16)) % sizeof(uint4) == 0u,
            "hidden * sizeof(bf16) must be a multiple of uint4 for combine vectorization");

        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens; token_idx += kNumSMs * kNumEpilogueWarps) {
            // Read the topk slot indices for this token.
            const int slot =
                lane_idx < kNumTopk
                    ? static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                             token_idx * kNumTopk + lane_idx))
                    : -1;
            const uint64_t mask = ballot(slot >= 0);

            // Per-lane stride over the token's uint4 chunks: lane L
            // handles uint4 indices {L, L + 64, L + 128, ...}.
            for (uint32_t i = lane_idx; i < kNumUint4PerToken; i += kWarpSize) {
                const uint32_t off                        = i * sizeof(uint4);
                float2         reduced[kNumElemsPerUint4] = {};

                uint64_t remaining = mask;
                while (remaining) {
                    const uint32_t b = ffs(remaining) - 1u;
                    remaining ^= 1ull << b;
                    auto *src_ptr = combine_token_buffer.get_rank_buffer(b)
                                        .get_data_buffer(token_idx)
                                        .get_base_ptr<uint8_t>();
                    const uint4 partial = *reinterpret_cast<const uint4 *>(src_ptr + off);
                    const auto *bf16    = reinterpret_cast<const __hip_bfloat162 *>(&partial);
#pragma unroll
                    for (uint32_t l = 0u; l < kNumElemsPerUint4; ++l) {
                        const float2 fp32 = __bfloat1622float2(bf16[l]);
                        reduced[l].x += fp32.x;
                        reduced[l].y += fp32.y;
                    }
                }

                uint4 out;
                auto *bf16_out = reinterpret_cast<__hip_bfloat162 *>(&out);
#pragma unroll
                for (uint32_t l = 0u; l < kNumElemsPerUint4; ++l)
                    bf16_out[l] = __float22bfloat162_rn(reduced[l]);
                auto *dst =
                    reinterpret_cast<uint8_t *>(y) + token_idx * kHidden * sizeof(__hip_bfloat16);
                *reinterpret_cast<uint4 *>(dst + off) = out;
            }
            sync_warp();
        }
        return;
    }

    // Silence "unused" warnings on weight/SF tensors - they are wired
    // into the loader path that the next optimization pass enables.
    (void) l1_weights;
    (void) l1_weights_sf;
    (void) l2_weights;
    (void) l2_weights_sf;
#else
    (void) y;
    (void) cumulative_local_expert_recv_stats;
    (void) num_tokens;
    (void) sym_buffer;
    (void) l1_weights;
    (void) l1_weights_sf;
    (void) l2_weights;
    (void) l2_weights_sf;
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        // The DG-aligned mega-MoE kernel requires gfx950 hardware.
    }
#endif
}

// ---------------------------------------------------------------------
//  Host-side launcher template.  Signature mirrors DG so the calling
//  host wrapper (mega_moe.cu) can dispatch to the AMD implementation by
//  simply changing the namespace / arch tag.
// ---------------------------------------------------------------------
template <MegaMoEArch kArch, uint32_t kNumMaxTokensPerRank, uint32_t kHidden,
          uint32_t kIntermediateHidden, uint32_t kNumExperts, uint32_t kNumTopk,
          uint32_t kNumExpertsPerWave, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t STORE_BLOCK_M, uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
          uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens, uint32_t kNumStages,
          uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads, uint32_t kNumSMs, uint32_t kNumRanks,
          float kActivationClamp, bool kFastMath>
hipError_t launch_fp8_fp4_mega_moe_impl(void *y, int *cumulative_local_expert_recv_stats,
                                        const uint32_t                      num_tokens,
                                        const layout::SymBuffer<kNumRanks> &sym_buffer,
                                        const void *l1_weights, const void *l1_weights_sf,
                                        const void *l2_weights, const void *l2_weights_sf,
                                        hipStream_t stream) {
    static_assert(kArch == MegaMoEArch::Gfx950, "Only gfx950 (MI355X) is supported for now");

    constexpr uint32_t kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    // LDS budget for the device kernel's ``extern __shared__`` carve-out.
    //
    // MI355X exposes 160 KiB of LDS per CU (CDNA4).  The loader now
    // ring-buffers ``kLoaderStages`` × per-stage (A | B | SFA | SFB)
    // tiles per wave, where ``kLoaderStages`` is the minimum of
    // ``kNumStages`` and the largest stage count that fits the budget:
    //
    //   per stage  : LOAD_BLOCK_M * BLOCK_K     (A, FP8)   = 8192 B
    //                LOAD_BLOCK_N * BLOCK_K     (B, FP4u)  = 16384 B
    //                SF_BLOCK_M  * sizeof(u32)             = 512 B
    //                SF_BLOCK_N  * sizeof(u32)             = 512 B
    //                                            ~= 25 KiB / stage
    //   2 stages × 2 waves   ~=  100 KiB    (smoke shape: kLoaderStages = 2)
    //   + accumulator sinks × 2 waves      ~=    1 KiB
    //   + scheduler / barrier / expert-count pad ~= 1 KiB
    //                                       ====================
    //                                                  ~=  102 KiB
    //
    // The device-side ``static_assert(kLoaderBaseBytes +
    // kNumMMANonEpilogueWarps * kLoaderWaveBytes <= kSmemBytesBudget)``
    // enforces this at compile time.  Stays comfortably under the
    // 160 KiB cap with headroom for the named-barrier state, scheduler
    // workspace and future epilogue staging buffers.  See TODO.md
    // Section B item 3.
    constexpr uint32_t kSmemBytes = 140u * 1024u;

    const dim3 grid(kNumSMs);
    const dim3 block(kNumThreads);

    auto kernel = gfx950_fp8_fp4_mega_moe_kernel<
        kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk,
        kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M, SF_BLOCK_M, SF_BLOCK_N,
        kNumMaxPoolTokens, kNumPaddedSFPoolTokens, kNumStages, kNumDispatchThreads,
        kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp,
        kFastMath>;

    // ROCm's default per-launch dynamic LDS cap is 64 KiB.  Opt in to the
    // CU's 160 KiB pool (MI355X) for the 96 KiB carve-out below; without
    // this attribute the launch fails with ``hipErrorInvalidValue``.
    if constexpr (kSmemBytes > 64u * 1024u) {
        const auto attr_err = hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(kSmemBytes));
        if (attr_err != hipSuccess)
            return attr_err;
    }

    hipLaunchKernelGGL(kernel, grid, block, kSmemBytes, stream, y,
                       cumulative_local_expert_recv_stats, num_tokens, sym_buffer,
                       const_cast<void *>(l1_weights), const_cast<void *>(l1_weights_sf),
                       const_cast<void *>(l2_weights), const_cast<void *>(l2_weights_sf));
    return hipGetLastError();
}

} // namespace mega_moe
} // namespace primus_turbo
