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
    const auto workspace = layout::Workspace(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
                                             kNumMaxTokensPerRank, kNumTopk);

    const auto input_token_buffer =
        layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer =
        layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    const auto l1_token_buffer = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens,
                                                input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer =
        layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer =
        layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
                                                l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
                                             l2_token_buffer.get_end_ptr());
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

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
    //   1 : dispatch + epilogue cross sync (kNumDispatchThreads + kNumEpilogueThreads)
    //   2 : dispatch role grid_sync / nvlink_barrier internal
    //   3 : epilogue role grid_sync / nvlink_barrier internal
    constexpr uint32_t kBarDispLocal = 0u;
    constexpr uint32_t kBarDispEpi   = 1u;
    constexpr uint32_t kBarDispGrid  = 2u;
    constexpr uint32_t kBarEpiGrid   = 3u;

    // Canonical (bar_id -> num_threads) registry (TODO.md Section A
    // item 11).  Every named-barrier sync below derives ``num_threads``
    // from this table - never spell the count out by hand at the call
    // site.  Centralising the binding means a future role-count change
    // updates exactly one place; the alternative (each callsite
    // independently spelling out e.g. ``kNumDispatchThreads +
    // kNumEpilogueThreads``) would silently produce a (counter,
    // expected) mismatch and deadlock on the first drift.
    constexpr uint32_t kBarThreads[prims::kNumMaxNamedBars] = {
        /*[kBarDispLocal] =*/kNumDispatchThreads,
        /*[kBarDispEpi]   =*/kNumDispatchThreads + kNumEpilogueThreads,
        /*[kBarDispGrid]  =*/kNumDispatchThreads,
        /*[kBarEpiGrid]   =*/kNumEpilogueThreads,
        0u,
        0u,
        0u,
        0u,
    };
    static_assert(prims::kNumMaxNamedBars >= 4u,
                  "Named-barrier registry needs at least 4 slots for the current role map");

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
            atomicAdd(smem_expert_count + expert_idx, 1);
        });
        sync_aligned(named_bar, kBarThreads[kBarDispLocal], kBarDispLocal);

        // Pass 2: post the per-expert send count into the workspace.
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i]      = static_cast<uint32_t>(
                atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        sync_aligned(named_bar, kBarThreads[kBarDispLocal], kBarDispLocal);

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
        comm::grid_sync<kNumSMs, 0>(workspace, named_bar, kBarDispGrid, kBarThreads[kBarDispGrid],
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
        sync_aligned(named_bar, kBarThreads[kBarDispLocal], kBarDispLocal);

        if (sm_idx == 0u && thread_idx == kDispLeader)
            printf("[r%u/disp] pre-nvlink_barrier-1\n", sym_buffer.rank_idx);
        // NVLink barrier (collapses to grid sync for kNumRanks==1).
        comm::nvlink_barrier<kNumRanks, kNumSMs, kBarThreads[kBarDispGrid], 0, 1>(
            workspace, sym_buffer, named_bar, kBarDispGrid, kDispLeader, sm_idx, thread_idx,
            /*sync_prologue=*/false, /*sync_epilogue=*/true);
        if (sm_idx == 0u && thread_idx == kDispLeader)
            printf("[r%u/disp] post-nvlink_barrier-1, pre-fetch_expert_recv_count\n",
                   sym_buffer.rank_idx);

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
        if (sm_idx == 0u && thread_idx == kDispLeader)
            printf("[r%u/disp] post-fetch_expert_recv_count, pre-pass4\n", sym_buffer.rank_idx);

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

        if (sm_idx == 0u && thread_idx == kDispLeader)
            printf("[r%u/disp] post-pass4, pre-sync_unaligned(kBarDispEpi)\n", sym_buffer.rank_idx);
        // Workspace cleanup for the next launch.  This is a cross-role
        // sync: it waits for the epilogue waves to finish spinning on
        // the L1/L2 arrival counters/masks before we wipe them.  The
        // epilogue side has a matching ``sync_unaligned(named_bar,
        // kBarThreads[kBarDispEpi], kBarDispEpi)`` call placed after its
        // ``for_each_block`` (see below in the epilogue branch).  Mirrors
        // DG ``ptx::sync_unaligned(..., kDispatchWithEpilogueBarrierIdx)``
        // - the two roles arrive from non-contiguous warp groups.
        sync_unaligned(named_bar, kBarThreads[kBarDispEpi], kBarDispEpi);
        if (sm_idx == 0u && thread_idx == kDispLeader)
            printf("[r%u/disp] post-sync_unaligned(kBarDispEpi)\n", sym_buffer.rank_idx);

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
        if (sm_idx == 0u && thread_idx == kDispLeader)
            printf("[r%u/disp] pre-nvlink_barrier-2\n", sym_buffer.rank_idx);
        comm::nvlink_barrier<kNumRanks, kNumSMs, kBarThreads[kBarDispGrid], 0, 3>(
            workspace, sym_buffer, named_bar, kBarDispGrid, kDispLeader, sm_idx, thread_idx,
            /*sync_prologue=*/true, /*sync_epilogue=*/false);
        if (sm_idx == 0u && thread_idx == kDispLeader)
            printf("[r%u/disp] post-nvlink_barrier-2\n", sym_buffer.rank_idx);

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
        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
            // For the alignment pass we only walk the iteration space -
            // the per-block tile staging + MFMA issue path lives in the
            // epilogue branch below so a single wave drives both load and
            // compute.  This keeps the kernel compilable while preserving
            // DG's exact for_each_block iteration order (the per-wave
            // scheduling property the user explicitly called out).
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
            (void) local_expert_idx;
            (void) n_block_idx;
        });
        return;
    }

    // =================================================================
    //  Role: epilogue + combine waves.  Drives the SwiGLU + UE8M0 quant
    //  (L1) and the BF16 NVLink write (L2) paths, then the cross-topk
    //  reduce + write-back loop.  Both paths are driven by the SAME
    //  ``scheduler.for_each_block`` so the per-wave scheduling holds.
    // =================================================================
    if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        if (sm_idx == 0u && thread_idx == kEpiLeader)
            printf("[r%u/epi] pre-for_each_block\n", sym_buffer.rank_idx);
        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
            // Mirror the SM100 epilogue divergence: Linear1 -> SwiGLU
            // path (writes into l1/l2 pools), Linear2 -> BF16 push path
            // (writes into the combine buffer).  We only need to mark
            // arrivals on the workspace at this alignment stage; the
            // tile-level math will be filled in by the next patch.
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            if (phase == sched::BlockPhase::Linear1) {
                if (elect_one()) {
                    red_add_rel(workspace.get_l1_arrival_count_ptr(pool_block_idx), 0u);
                    red_or_rel_gpu(workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                                   1ull << n_block_idx);
                }
            } else {
                (void) num_k_blocks;
                (void) local_expert_idx;
            }
        });

        if (sm_idx == 0u && thread_idx == kEpiLeader)
            printf("[r%u/epi] post-for_each_block\n", sym_buffer.rank_idx);
        // Pair the dispatch cleanup cross-sync (``kBarDispEpi``) so the
        // dispatch waves can safely wipe the L1/L2 arrival counters
        // once every epilogue wave has exited ``for_each_block``.
        // Mirrors DG ``ptx::sync_unaligned(..., kDispatchWithEpilogueBarrierIdx)``.
        sync_unaligned(named_bar, kBarThreads[kBarDispEpi], kBarDispEpi);
        if (sm_idx == 0u && thread_idx == kEpiLeader)
            printf("[r%u/epi] post-sync_unaligned(kBarDispEpi)\n", sym_buffer.rank_idx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kBarThreads[kBarEpiGrid], 1, 2>(
            workspace, sym_buffer, named_bar, kBarEpiGrid, kEpiLeader, sm_idx, thread_idx,
            /*sync_prologue=*/true, /*sync_epilogue=*/true);
        if (sm_idx == 0u && thread_idx == kEpiLeader)
            printf("[r%u/epi] post-nvlink_barrier-epi\n", sym_buffer.rank_idx);

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
    // MI355X exposes 160 KiB of LDS per CU (CDNA4).  Once the A/B loader
    // body lands, ``kNumStages`` ring-buffered stages of A (FP8) + B
    // (FP4-unpacked) + SFA + SFB tiles dominate the carve-out:
    //
    //   per stage  : LOAD_BLOCK_M * BLOCK_K     (A, FP8)   = 8192 B
    //                LOAD_BLOCK_N * BLOCK_K     (B, FP4u)  = 16384 B
    //                SF_BLOCK_M  * sizeof(u32)             = 512 B
    //                SF_BLOCK_N  * sizeof(u32)             = 512 B
    //                                            ~= 25 KiB / stage
    //   4 stages   ~= 100 KiB
    //   + scheduler / barrier / expert-count pad ~= 8 KiB
    //
    // 96 KiB (the previous setting matching SM100's TMEM-style budget)
    // overruns the loader carve-out by ~10 KiB.  Bump to 140 KiB - the
    // largest value that comfortably stays under the 160 KiB cap with
    // headroom for the named-barrier state, scheduler workspace and
    // future epilogue staging buffers.  See TODO.md Section B item 3.
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
