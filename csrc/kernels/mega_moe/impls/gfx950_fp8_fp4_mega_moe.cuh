
#pragma once

#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

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

enum class MegaMoEArch : uint32_t {
    Unknown = 0,
    Gfx942  = 942,
    Gfx950  = 950,
};

// ---------------------------------------------------------------------------
// Optional per-stage in-kernel profiler.
//
// Compile with ``-DMEGA_MOE_PROFILE=1`` to time each pipeline stage of the
// persistent mega-MoE kernel: dispatch / Linear1 MMA / Linear1 epilogue
// (SwiGLU) / Linear2 MMA / Linear2 epilogue (combine write) / final combine
// reduction, plus the whole-kernel total.  Timing uses ``wall_clock64()``
// (the device-global steady counter that ticks at the fixed frequency
// reported by ``hipDeviceAttributeWallClockRate`` and is consistent across
// all CUs/XCDs, so cross-block min/max spans are meaningful).
//
// Each stage records the EARLIEST start and LATEST end seen anywhere in the
// grid; the host reads the [start,end] pairs back and reports the span
// (end - start) in ticks, then converts to microseconds.  Default is 0 ->
// the whole profiler compiles away to nothing (no runtime overhead).
// ---------------------------------------------------------------------------
#ifndef MEGA_MOE_PROFILE
#define MEGA_MOE_PROFILE 0
#endif

#if MEGA_MOE_PROFILE
enum MegaMoEProfStage : uint32_t {
    kProfDispatch = 0u, // dispatch warps: token routing + pool fill
    kProfL1Mma,         // Linear1 grouped GEMM (gate + up)
    kProfL1Epi,         // Linear1 epilogue: SwiGLU + quant -> L2 pool
    kProfL2Mma,         // Linear2 grouped GEMM
    kProfL2Epi,         // Linear2 epilogue: write_combine
    kProfCombine,       // final combine warps: topk reduction -> y
    kProfTotal,         // whole kernel
    kProfNumStages,
};

// Layout: [2 * stage + 0] = earliest start tick, [2 * stage + 1] = latest
// end tick.  Host resets to {~0ull, 0ull} before every launch.
__device__ unsigned long long g_mega_moe_prof[2u * kProfNumStages];

__device__ __forceinline__ unsigned long long mega_moe_prof_now() {
    return static_cast<unsigned long long>(wall_clock64());
}

__device__ __forceinline__ void mega_moe_prof_commit(uint32_t stage, unsigned long long t0,
                                                     unsigned long long t1) {
    if (t0 == ~0ull && t1 == 0ull)
        return; // stage never entered by this thread
    atomicMin(&g_mega_moe_prof[2u * stage + 0u], t0);
    atomicMax(&g_mega_moe_prof[2u * stage + 1u], t1);
}

// Fold one [a,b] sample into a thread-local [lo,hi] running span (no atomics);
// used to accumulate across the many per-block iterations of the MMA warps
// before a single atomic commit at the end.
#define MEGA_MOE_PROF_ACC(lo, hi, a, b)                                                            \
    do {                                                                                           \
        const unsigned long long _pa = (a);                                                        \
        const unsigned long long _pb = (b);                                                        \
        if (_pa < (lo))                                                                            \
            (lo) = _pa;                                                                            \
        if (_pb > (hi))                                                                            \
            (hi) = _pb;                                                                            \
    } while (0)
#endif

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

namespace comm {

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

            const auto status       = prims::ld_volatile(counter_ptr) & 3u;
            const auto signal_phase = status & 1u;
            const auto signal_sign  = status >> 1u;
            auto      *signal_ptr   = workspace.get_nvl_barrier_signal_ptr(signal_phase);

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

template <typename Adtype, typename Bdtype>
__device__ __forceinline__ dtype::float32x4 mfma_scaled(dtype::int32x8 a, dtype::int32x8 b,
                                                        dtype::float32x4 c, uint32_t scale_a,
                                                        uint32_t scale_b) {
    return device::mfma_scale_f32_16x16x128_f8f6f4<Adtype, Bdtype>::run(a, b, c, scale_a, scale_b);
}

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

    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = get_warp_idx();
    const uint32_t lane_idx   = get_lane_idx();

#if MEGA_MOE_PROFILE
    const unsigned long long prof_kernel_start = mega_moe_prof_now();
#endif

    constexpr uint32_t kNumDispatchWarps       = kNumDispatchThreads / kWarpSize;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / kWarpSize;
    constexpr uint32_t kNumEpilogueWarps       = kNumEpilogueThreads / kWarpSize;
    constexpr uint32_t kNumEpilogueWarpgroups  = (kNumEpilogueWarps + 3u) / 4u;
    (void) kNumEpilogueWarpgroups;

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(__hip_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32u);
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 32u);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

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

    constexpr uint32_t kGranK                 = 32u;
    constexpr uint32_t kNumUTCCPAlignedElems  = 128u;
    auto               transform_sf_token_idx = [](const uint32_t &token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M + (idx & ~127u) + (idx & 31u) * 4u +
               ((idx >> 5) & 3u);
    };

    constexpr uint32_t LAYOUT_AD_M  = 128u;
    constexpr uint32_t UMMA_M       = LAYOUT_AD_M;
    constexpr uint32_t UMMA_N       = BLOCK_M;
    constexpr uint32_t UMMA_K       = 32u;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2u;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    static_assert(BLOCK_M % 16u == 0u, "Invalid block M");
    static_assert(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    static_assert(BLOCK_K == 128u, "Invalid block K");
    (void) UMMA_M;
    (void) UMMA_N;
    (void) UMMA_K;
    (void) LOAD_BLOCK_M;
    (void) LOAD_BLOCK_N;

    constexpr uint32_t L1_OUT_BLOCK_N        = BLOCK_N / 2u;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * 1u;

    constexpr uint32_t SMEM_B_SIZE_PER_STAGE   = LOAD_BLOCK_N * BLOCK_K * 1u;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    (void) SMEM_A_SIZE_PER_STAGE;
    (void) SMEM_B_SIZE_PER_STAGE;
    (void) SMEM_SFA_SIZE_PER_STAGE;
    (void) SMEM_SFB_SIZE_PER_STAGE;
    (void) L1_OUT_BLOCK_N;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto      smem_expert_count = reinterpret_cast<uint32_t *>(smem_buffer);
    Mbarrier *smem_barriers =
        reinterpret_cast<Mbarrier *>(smem_buffer + sizeof(uint32_t) * kNumExperts + 16u);
    auto smem_send_buffer = reinterpret_cast<uint8_t *>(smem_barriers + 32u);
    (void) smem_send_buffer;

    constexpr uint32_t kMfmaM       = 32u;
    constexpr uint32_t kMfmaN       = 32u;
    constexpr uint32_t kMfmaK       = 64u;
    constexpr uint32_t kInnerKIters = BLOCK_K / kMfmaK;
    static_assert(BLOCK_K % kMfmaK == 0u, "BLOCK_K must be a multiple of MFMA K");

    static_assert(BLOCK_M % kNumMMANonEpilogueWarps == 0u,
                  "BLOCK_M must be evenly partitionable across loader waves");
    constexpr uint32_t kRowsPerLoaderWave = BLOCK_M / kNumMMANonEpilogueWarps;
    constexpr uint32_t kColsPerLoaderWave = BLOCK_N;
    static_assert(kRowsPerLoaderWave % kMfmaM == 0u,
                  "loader wave row partition must be a multiple of MFMA M");
    static_assert(kColsPerLoaderWave % kMfmaN == 0u,
                  "loader wave col partition must be a multiple of MFMA N");
    constexpr uint32_t kSubTilesM       = kRowsPerLoaderWave / kMfmaM;
    constexpr uint32_t kSubTilesN       = kColsPerLoaderWave / kMfmaN;
    constexpr uint32_t kSubTilesPerWave = kSubTilesM * kSubTilesN;

    constexpr uint32_t kATileBytes       = kRowsPerLoaderWave * BLOCK_K;
    constexpr uint32_t kBTileBytes       = kColsPerLoaderWave * BLOCK_K;
    constexpr uint32_t kLaneLoadBytes    = 16u;
    constexpr uint32_t kBytesPerLoadCall = kWarpSize * kLaneLoadBytes;
    static_assert(kATileBytes % kBytesPerLoadCall == 0u,
                  "A tile must be a multiple of one cooperative load call");
    static_assert(kBTileBytes % kBytesPerLoadCall == 0u,
                  "B tile must be a multiple of one cooperative load call");
    constexpr uint32_t kATileLoadsPerWave = kATileBytes / kBytesPerLoadCall;
    constexpr uint32_t kBTileLoadsPerWave = kBTileBytes / kBytesPerLoadCall;

    constexpr uint32_t kLoaderSFBytes   = 64u;
    constexpr uint32_t kLoaderSinkBytes = kSubTilesPerWave * sizeof(dtype::float32x16);

    // N6: stage the B-tile ONCE for the whole MMA warpgroup. B is identical
    // across MMA warps (all share n_block_idx / local_expert_idx; only A's
    // M-rows differ per warp), so the old per-warp B duplication was pure waste
    // that starved the loader pipeline at higher MMA-warp counts. A (+ its SF)
    // stays per-warp; B becomes a single shared region after all warps' A.
    constexpr uint32_t kStagedABytesPerStage = kATileBytes + kLoaderSFBytes * 2u; // per-warp
    constexpr uint32_t kStagedBBytesPerStage = kBTileBytes;                       // shared
    constexpr uint32_t kLoaderBaseBytes =
        ((sizeof(uint32_t) * kNumExperts + 16u + 32u * sizeof(Mbarrier) + 1023u) / 1024u) * 1024u;

    static constexpr uint32_t kSmemBytesBudget = 140u * 1024u;
    static_assert(kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderSinkBytes < kSmemBytesBudget,
                  "loader sink + base already exceed kSmemBytes budget");
    constexpr uint32_t kMaxLoaderStagesByLds =
        (kSmemBytesBudget - kLoaderBaseBytes - kNumMMANonEpilogueWarps * kLoaderSinkBytes) /
        (kNumMMANonEpilogueWarps * kStagedABytesPerStage + kStagedBBytesPerStage);
    static_assert(kMaxLoaderStagesByLds >= 1u,
                  "loader staged carve-out too large for one stage even - shrink BLOCK_K?");
    constexpr uint32_t kLoaderStages =
        (kNumStages < kMaxLoaderStagesByLds) ? kNumStages : kMaxLoaderStagesByLds;

    // Per-warp A region (A + SF ping-pong stages, then the accumulator sink).
    constexpr uint32_t kLoaderAWaveBytes = kLoaderStages * kStagedABytesPerStage + kLoaderSinkBytes;
    // Single shared B region, placed after every warp's A region.
    constexpr uint32_t kLoaderBPoolBaseBytes =
        kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderAWaveBytes;
    constexpr uint32_t kLoaderBPoolBytes = kLoaderStages * kStagedBBytesPerStage;
    static_assert(kLoaderBPoolBaseBytes + kLoaderBPoolBytes <= kSmemBytesBudget,
                  "loader tile carve-out (per-warp A + shared B) exceeds kSmemBytes budget");

    constexpr uint32_t kLoadsPerStage = kATileLoadsPerWave + kBTileLoadsPerWave;

    static_assert(kLoadsPerStage < 63u,
                  "kLoadsPerStage exceeds vmcnt range - reduce per-stage loads");

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

    prims::NamedBarrierWg named_bar;
    named_bar.init();

    constexpr uint32_t kBarDispLocal = 0u;
    constexpr uint32_t kBarDispEpi   = 1u;
    constexpr uint32_t kBarDispGrid  = 2u;
    constexpr uint32_t kBarEpiGrid   = 3u;
    constexpr uint32_t kBarDispEpi2  = 4u;
    constexpr uint32_t kBarMmaB      = 5u; // N6: shared-B producer/consumer sync (MMA warpgroup)

    constexpr uint32_t kDispLeader = 0u;
    constexpr uint32_t kEpiLeader  = (kNumDispatchWarps + kNumMMANonEpilogueWarps) * kWarpSize;

    if (warp_idx == 0u && elect_one()) {
#pragma unroll
        for (uint32_t i = 0; i < kNumExperts; ++i)
            smem_expert_count[i] = 0u;
#pragma unroll
        for (uint32_t i = 0; i < 32u; ++i)
            smem_barriers[i].init(1u);
    }
    __syncthreads();

    auto scheduler = sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K,
                                             L2_SHAPE_N, L2_SHAPE_K, kNumExpertsPerRank,
                                             kNumExpertsPerWave, kNumSMs, kNumRanks>(workspace);

    if (warp_idx < kNumDispatchWarps) {

#if MEGA_MOE_PROFILE
        const unsigned long long prof_disp_t0 = mega_moe_prof_now();
#endif

        constexpr uint32_t kNumTokensPerWarp = kWarpSize / kNumTopk;
        constexpr uint32_t kNumGlobalWarps   = kNumSMs * kNumDispatchWarps;
        static_assert(kNumTokensPerWarp * kNumTopk <= kWarpSize,
                      "kNumTopk does not divide wave size");

        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto         read_topk_idx     = [&](const auto &process) {

#pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens; i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {

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

        read_topk_idx([&](const uint32_t &token_topk_idx, const int &expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        sync_aligned(named_bar, kNumDispatchThreads, kBarDispLocal);

        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i]      = static_cast<uint32_t>(
                atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        sync_aligned(named_bar, kNumDispatchThreads, kBarDispLocal);

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

        comm::grid_sync<kNumSMs, 0>(workspace, named_bar, kBarDispGrid, kNumDispatchThreads,
                                    kDispLeader, sm_idx, thread_idx);

        if (sm_idx == 0u) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t dst_rank         = i / kNumExpertsPerRank;
                const uint32_t dst_local_expert = i % kNumExpertsPerRank;
                const uint64_t expert_status    = *workspace.get_expert_send_count_ptr(i);

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

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads, 0, 1>(
            workspace, sym_buffer, named_bar, kBarDispGrid, kDispLeader, sm_idx, thread_idx, false,
            true);

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

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                static_cast<uint32_t>(current_expert_idx), current_rank_in_expert_idx,
                token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            const uint32_t pool_token_idx =
                expert_pool_block_offset * BLOCK_M + token_idx_in_expert;

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

#if MEGA_MOE_PROFILE
        // Capture the dispatch "work" end HERE, before the trailing grid /
        // nvlink barriers + counter resets.  Those tail barriers block until
        // the MMA/combine roles finish Linear2, so measuring to the role's
        // return would fold the whole pipeline's wall time into "dispatch".
        // This keeps kProfDispatch comparable to kProfL*Mma / kProfCombine,
        // which already exclude their own arrival/entry waits.
        const unsigned long long prof_disp_t1 = mega_moe_prof_now();
#endif

        sync_unaligned(named_bar, kNumThreads, kBarDispEpi);

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

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads, 0, 3>(
            workspace, sym_buffer, named_bar, kBarDispGrid, kDispLeader, sm_idx, thread_idx, true,
            false);

        sync_unaligned(named_bar, kNumDispatchThreads + kNumEpilogueThreads, kBarDispEpi2);

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

#if MEGA_MOE_PROFILE
        if (elect_one()) {
            mega_moe_prof_commit(kProfDispatch, prof_disp_t0, prof_disp_t1);
            // kProfTotal uses the real role exit (after the tail barriers) so
            // the whole-kernel envelope stays faithful.
            mega_moe_prof_commit(kProfTotal, prof_kernel_start, mega_moe_prof_now());
        }
#endif
        return;
    }

    if (warp_idx >= kNumDispatchWarps && warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {

        const uint32_t loader_warp_local = warp_idx - kNumDispatchWarps;

#if MEGA_MOE_PROFILE
        // Per-warp running [lo,hi] spans, folded across all block iterations
        // and atomically committed once after for_each_block().
        unsigned long long prof_l1mma_lo = ~0ull, prof_l1mma_hi = 0ull;
        unsigned long long prof_l1epi_lo = ~0ull, prof_l1epi_hi = 0ull;
        unsigned long long prof_l2mma_lo = ~0ull, prof_l2mma_hi = 0ull;
        unsigned long long prof_l2epi_lo = ~0ull, prof_l2epi_hi = 0ull;
#endif

        const uint32_t wave_base_byte = kLoaderBaseBytes + loader_warp_local * kLoaderAWaveBytes;
        const uint32_t sink_off       = wave_base_byte + kLoaderStages * kStagedABytesPerStage;

        const uint32_t a_lds_stage0 =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_buffer + wave_base_byte));
        // N6: shared B base is warpgroup-wide (independent of loader_warp_local).
        const uint32_t b_lds_stage0 =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_buffer + kLoaderBPoolBaseBytes));

        device::BufferSRD srd_l1_a(l1_token_buffer.get_base_ptr<void>());
        device::BufferSRD srd_l2_a(l2_token_buffer.get_base_ptr<void>());
        device::BufferSRD srd_l1_b(l1_weights);
        device::BufferSRD srd_l2_b(l2_weights);

        dtype::float32x16 acc[kSubTilesPerWave] = {};

        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            if (phase == sched::BlockPhase::Linear1) {

                const auto    *ptr      = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const uint32_t expected = scheduler.template get_valid_m<false>();
                while (ld_acq(ptr) != expected) {
                }
            } else {
                // L2 arrival is a COUNT, not an OR-mask.  An OR-mask would
                // reach `expected` as soon as the FIRST MMA wave finished its
                // gate-blocks (every wave OR's the same bits for its M-row
                // slice), letting Linear2 read a partially-written pool.
                // Counting one increment per (wave, gate-block) — expected =
                // waves * gate-blocks — requires every wave to finish first.
                const auto        *ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                constexpr uint32_t kL1GateNBlocksWait = (L1_SHAPE_N / BLOCK_N) / 2u;
                const uint64_t     expected = static_cast<uint64_t>(kNumMMANonEpilogueWarps) *
                                          static_cast<uint64_t>(kL1GateNBlocksWait);
                while (ld_acq_gpu(ptr) != expected) {
                }
            }

            const auto &srd_a = (phase == sched::BlockPhase::Linear1) ? srd_l1_a : srd_l2_a;
            const auto &srd_b = (phase == sched::BlockPhase::Linear1) ? srd_l1_b : srd_l2_b;

            const uint32_t a_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? kHidden : kIntermediateHidden;
            const uint32_t b_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? L1_SHAPE_K : L2_SHAPE_K;
            const uint32_t b_expert_stride_bytes = (phase == sched::BlockPhase::Linear1)
                                                       ? L1_SHAPE_N * L1_SHAPE_K
                                                       : L2_SHAPE_N * L2_SHAPE_K;

            const uint32_t a_wave_row0 =
                pool_block_idx * BLOCK_M + loader_warp_local * kRowsPerLoaderWave;
            const uint32_t a_tile_base_bytes = a_wave_row0 * a_row_stride_bytes;

            uint32_t b_wave_col0 = n_block_idx * BLOCK_N;
            uint32_t b_tile_base_bytes =
                local_expert_idx * b_expert_stride_bytes + b_wave_col0 * b_row_stride_bytes;

            constexpr uint32_t kScaleOne = 0x7f7f7f7fu;

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

            uint32_t sfb_n_global_base = n_block_idx * BLOCK_N;

            const auto issue_stage = [&](uint32_t stage, uint32_t k_block) {
                const uint32_t a_k_offset_bytes = k_block * BLOCK_K;
                const uint32_t b_k_offset_bytes = k_block * BLOCK_K;
                const uint32_t a_lds            = a_lds_stage0 + stage * kStagedABytesPerStage;
                const uint32_t b_lds            = b_lds_stage0 + stage * kStagedBBytesPerStage;
#pragma unroll
                for (uint32_t c = 0u; c < kATileLoadsPerWave; ++c) {
                    const uint32_t m_in_wave = c * 8u + lane_idx / 8u;
                    // R145: buffer_load_lds collapses lds_addr to readfirstlane,
                    // so all lanes write linearly from lane 0's base. Pre-XOR
                    // each lane's k-chunk by (m & 7) so the linear LDS write
                    // lands at the position the reader's swizzle expects.
                    const uint32_t k_chunk_in_lane = (lane_idx % 8u) ^ ((lane_idx / 8u) & 7u);
                    const uint32_t k_byte_in_tile  = k_chunk_in_lane * kLaneLoadBytes;
                    const uint32_t ldg_offset = a_tile_base_bytes + m_in_wave * a_row_stride_bytes +
                                                a_k_offset_bytes + k_byte_in_tile;
                    const uint32_t lds_offset =
                        device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                            0u, m_in_wave, k_byte_in_tile);
                    device::load_gmem_to_smem_srd<16>(srd_a, ldg_offset, a_lds + lds_offset, 0);
                }
                // N6: B is shared across the MMA warpgroup, so only warp 0 issues
                // the B load (else N warps redundantly buffer_load_lds to the same
                // shared address). Consumers are released by the kBarMmaB barrier
                // in the K-loop after the matching wait_vmcnt.
                if (loader_warp_local == 0u) {
#pragma unroll
                    for (uint32_t c = 0u; c < kBTileLoadsPerWave; ++c) {
                        const uint32_t n_in_wave       = c * 8u + lane_idx / 8u;
                        const uint32_t k_chunk_in_lane = (lane_idx % 8u) ^ ((lane_idx / 8u) & 7u);
                        const uint32_t k_byte_in_tile  = k_chunk_in_lane * kLaneLoadBytes;
                        const uint32_t ldg_offset      = b_tile_base_bytes +
                                                    n_in_wave * b_row_stride_bytes +
                                                    b_k_offset_bytes + k_byte_in_tile;
                        const uint32_t lds_offset =
                            device::b_tile_smem_byte_offset<kColsPerLoaderWave, BLOCK_K>(
                                0u, n_in_wave, k_byte_in_tile);
                        device::load_gmem_to_smem_srd<16>(srd_b, ldg_offset, b_lds + lds_offset, 0);
                    }
                }
            };

            auto run_k_loop = [&]() {

#pragma unroll
                for (uint32_t s = 0u; s < kSubTilesPerWave; ++s)
                    acc[s] = dtype::float32x16{};

                if (num_k_blocks > 0u)
                    issue_stage(0u, 0u);

                for (uint32_t k_block = 0u; k_block < num_k_blocks; ++k_block) {
                    const uint32_t this_stage = k_block % kLoaderStages;
                    const uint32_t a_stage_byte =
                        wave_base_byte + this_stage * kStagedABytesPerStage;
                    const uint32_t b_stage_byte =
                        kLoaderBPoolBaseBytes + this_stage * kStagedBBytesPerStage;

                    device::wait_vmcnt<0>();

                    // N6: shared-B producer/consumer sync across the MMA warpgroup.
                    // After this barrier: (a) warp 0's B[this_stage] load (waited
                    // above) is published to the consumer warps before the MFMA read
                    // below, and (b) every warp has finished reading the stage buffer
                    // that warp 0 is about to overwrite in the prefetch below (WAR).
                    sync_aligned(named_bar, kNumNonEpilogueThreads, kBarMmaB);

                    if (k_block + 1u < num_k_blocks) {
                        issue_stage((k_block + 1u) % kLoaderStages, k_block + 1u);
                    }

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

                        // Lane-independent K base for the MFMA's two 32-elem
                        // K-blocks (block_lo at K offset 0..31, block_hi at
                        // 32..63).  The block-scaled MFMA HW picks byte
                        // (lane/32) from scale_a/scale_b, so all 64 lanes carry
                        // the same packed dword: block_lo at byte 0, block_hi
                        // at byte 1.
                        const uint32_t k_byte_base          = k_block * BLOCK_K + k_inner * kMfmaK;
                        const uint32_t sfa_dword_idx_shared = k_byte_base / 128u;
                        const uint32_t sfa_byte_lo_pos      = (k_byte_base & 127u) >> 5u;
                        const uint32_t sfa_byte_hi_pos      = sfa_byte_lo_pos + 1u;

                        // Per-K-block SFB (block_lo / block_hi) for each sub_n,
                        // consumed by the K-split MFMA below.
                        uint32_t sfb_raw_lo_arr[kSubTilesN];
                        uint32_t sfb_raw_hi_arr[kSubTilesN];
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t n_in_wave = sub_n * kMfmaN + m_in_subtile;
                            const uint32_t n_global  = sfb_n_global_base + n_in_wave;
                            const uint32_t sfb_byte_offset_lo =
                                local_expert_idx * sfb_expert_stride_bytes +
                                n_global * sfb_n_stride_bytes + k_byte_base / kGranK;
                            const uint32_t sfb_byte_offset_hi =
                                local_expert_idx * sfb_expert_stride_bytes +
                                n_global * sfb_n_stride_bytes + (k_byte_base + 32u) / kGranK;
                            sfb_raw_lo_arr[sub_n] =
                                static_cast<uint32_t>(__ldg(sfb_weights_base + sfb_byte_offset_lo));
                            sfb_raw_hi_arr[sub_n] =
                                static_cast<uint32_t>(__ldg(sfb_weights_base + sfb_byte_offset_hi));
                        }

#pragma unroll
                        for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                        sub_m * kMfmaM + m_in_subtile;
                            const uint32_t sf_token_idx =
                                sfa_pool_token_idx_base + transform_sf_token_idx(m_in_block);
                            // Per-K-block SFA for this lane's M row: block_lo at
                            // byte position lo, block_hi at lo+1.  Consumed by
                            // the K-split MFMA below.
                            const uint32_t sfa_dword_db =
                                __ldg(sfa_pool_base +
                                      sfa_dword_idx_shared * kNumPaddedSFPoolTokens + sf_token_idx);
                            const uint32_t sfa_byte_lo_db =
                                (sfa_dword_db >> (sfa_byte_lo_pos * 8u)) & 0xffu;
                            const uint32_t sfa_byte_hi_db =
                                (sfa_dword_db >> (sfa_byte_hi_pos * 8u)) & 0xffu;
                            // K-split (FUNDAMENTAL, see C1-K rejected): the block-
                            // scaled MFMA broadcasts byte 0 of scale from one lane,
                            // so a single K=64 call can't carry both 32-elem MX
                            // scale groups. Issue two K=32 sub-calls, masking only
                            // A's wrong K-half (0*B=0 zeroes that half regardless of
                            // B). Linear2 SFA is UE8M0(1.0); Linear1 uses per-K SFA.
                            //
                            // C1-S: the mask + A-scales depend only on sub_m, so
                            // hoist them out of the sub_n loop, and software-pipeline
                            // the split -- issue all kSubTilesN lo-half MFMAs (into
                            // independent accumulators) before the hi-half ones, so
                            // each hi call finds its lo retired instead of stalling
                            // the 16-pass MAI on the immediate acc[sub] RAW. Numerics
                            // identical: each acc[sub] still gets lo then hi.
                            const uint32_t kHalf    = ((lane_idx >> 5u) & 1u);
                            const uint32_t maskLo   = (kHalf == 0u) ? 0xFFFFFFFFu : 0u;
                            const uint32_t maskHi   = (kHalf == 1u) ? 0xFFFFFFFFu : 0u;
                            const uint32_t sa_lo_db = (phase == sched::BlockPhase::Linear2)
                                                          ? 0x7f7f7f7fu
                                                          : (0x7f7f7f00u | sfa_byte_lo_db);
                            const uint32_t sa_hi_db = (phase == sched::BlockPhase::Linear2)
                                                          ? 0x7f7f7f7fu
                                                          : (0x7f7f7f00u | sfa_byte_hi_db);
                            dtype::int32x8 a_lo_v, a_hi_v;
                            {
                                const uint32_t *a_p =
                                    reinterpret_cast<const uint32_t *>(&a_vec[sub_m]);
                                uint32_t *a_lo_p = reinterpret_cast<uint32_t *>(&a_lo_v);
                                uint32_t *a_hi_p = reinterpret_cast<uint32_t *>(&a_hi_v);
#pragma unroll
                                for (uint32_t i = 0u; i < 8u; ++i) {
                                    a_lo_p[i] = a_p[i] & maskLo;
                                    a_hi_p[i] = a_p[i] & maskHi;
                                }
                            }
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const uint32_t sub      = sub_m * kSubTilesN + sub_n;
                                const uint32_t sb_lo_db = 0x7f7f7f00u | sfb_raw_lo_arr[sub_n];
                                acc[sub]                = device::mfma_scale_f32_32x32x64_f8f6f4<
                                                   __hip_fp8_e4m3, __hip_fp8_e4m3>::run(a_lo_v, b_vec[sub_n],
                                                                                        acc[sub], sa_lo_db,
                                                                                        sb_lo_db);
                            }
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const uint32_t sub      = sub_m * kSubTilesN + sub_n;
                                const uint32_t sb_hi_db = 0x7f7f7f00u | sfb_raw_hi_arr[sub_n];
                                acc[sub]                = device::mfma_scale_f32_32x32x64_f8f6f4<
                                                   __hip_fp8_e4m3, __hip_fp8_e4m3>::run(a_hi_v, b_vec[sub_n],
                                                                                        acc[sub], sa_hi_db,
                                                                                        sb_hi_db);
                            }
                        }
                    }
                }
            };

            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            static_assert(kNumL1BlockNs >= 2u && kNumL1BlockNs % 2u == 0u,
                          "L1 N must split evenly into gate||up halves");
            static_assert(kIntermediateHidden % BLOCK_N == 0u,
                          "BLOCK_N must divide kIntermediateHidden for SwiGLU pairing");
            constexpr uint32_t kL1GateNBlocks = kNumL1BlockNs / 2u;

            dtype::float32x16 acc_gate[kSubTilesPerWave];
            bool              has_gate = false;
#if MEGA_MOE_PROFILE
            const unsigned long long prof_mma_t0 = mega_moe_prof_now();
#endif
            if (phase == sched::BlockPhase::Linear1) {

                if (n_block_idx >= kL1GateNBlocks)
                    return;

                run_k_loop();

#pragma unroll
                for (uint32_t s = 0u; s < kSubTilesPerWave; ++s)
                    acc_gate[s] = acc[s];
                has_gate = true;

                b_wave_col0 += kIntermediateHidden;
                b_tile_base_bytes += kIntermediateHidden * b_row_stride_bytes;
                sfb_n_global_base += kIntermediateHidden;
                run_k_loop();
            } else {
                run_k_loop();
            }
            (void) has_gate;

#if MEGA_MOE_PROFILE
            {
                const unsigned long long prof_mma_t1 = mega_moe_prof_now();
                if (phase == sched::BlockPhase::Linear1)
                    MEGA_MOE_PROF_ACC(prof_l1mma_lo, prof_l1mma_hi, prof_mma_t0, prof_mma_t1);
                else
                    MEGA_MOE_PROF_ACC(prof_l2mma_lo, prof_l2mma_hi, prof_mma_t0, prof_mma_t1);
            }
            const unsigned long long prof_epi_t0 = mega_moe_prof_now();
#endif

            {
                constexpr uint32_t kNPattern[16] = {
                    0u, 1u, 2u, 3u, 8u, 9u, 10u, 11u, 16u, 17u, 18u, 19u, 24u, 25u, 26u, 27u,
                };
                const uint32_t m_lane      = lane_idx & 31u;
                const uint32_t n_lane_half = ((lane_idx >> 5u) & 1u) * 4u;
                const uint32_t valid_m     = scheduler.template get_valid_m<false>();

                if (phase == sched::BlockPhase::Linear1) {

                    auto *l2_pool_base = l2_token_buffer.get_base_ptr<uint8_t>();

                    // R30: v_mfma_scale_f32_32x32x64_f8f6f4 output register layout
                    // is M-varying within reg-i and N-constant per lane (the
                    // non-transposed CDNA mapping).  Earlier code used the
                    // transposed convention; R28 STAGE_A_ROWID proved it wrong.
                    // So: m_in_block varies with reg i (via n_lane_half +
                    // kNPattern[i]), and n_in_wave is lane-constant per
                    // subtile (via m_lane = lane & 31).
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub = sub_m * kSubTilesN + sub_n;
                            // MFMA output: M varies with reg i (n_lane_half +
                            // kNPattern[i]); N is lane-constant per subtile
                            // (m_lane = lane & 31).
                            const uint32_t n_in_wave        = sub_n * kMfmaN + m_lane;
                            const uint32_t intermediate_col = n_block_idx * BLOCK_N + n_in_wave;
                            if (intermediate_col >= kIntermediateHidden)
                                continue;
#pragma unroll
                            for (uint32_t i = 0u; i < 16u; ++i) {
                                const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                            sub_m * kMfmaM + n_lane_half +
                                                            kNPattern[i];
                                if (m_in_block >= valid_m)
                                    continue;
                                const uint32_t pool_token_idx =
                                    pool_block_idx * BLOCK_M + m_in_block;
                                auto *dst_row =
                                    l2_pool_base +
                                    pool_token_idx * fp8_intermediate_token_layout.num_bytes;

                                const float gate_raw = acc_gate[sub][i];
                                const float up_raw   = acc[sub][i];
                                const float gate =
                                    fmaxf(-kActivationClamp, fminf(kActivationClamp, gate_raw));
                                const float up =
                                    fmaxf(-kActivationClamp, fminf(kActivationClamp, up_raw));

                                const float          silu_gate = gate / (1.0f + __expf(-gate));
                                const float          swiglu    = silu_gate * up;
                                const __hip_fp8_e4m3 quant = static_cast<__hip_fp8_e4m3>(swiglu);
                                dst_row[intermediate_col] =
                                    reinterpret_cast<const uint8_t &>(quant);
                            }
                        }
                    }

                    __threadfence();
                    if (elect_one()) {
                        // L2 arrival as a COUNT: one increment per (wave,
                        // gate-block).  The __threadfence above (run by all
                        // lanes of this wave) orders this wave's pool writes
                        // before the bump, so Linear2 only proceeds once ALL
                        // waves' M-row slices are visible.
                        red_add_rel_gpu(workspace.get_l2_arrival_mask_ptr(pool_block_idx), 1ull);
                    }
                } else {

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

                    // R30: same M/N output-register swap as Linear1 writeback
                    // above.  m_in_block varies with reg i; n_global is
                    // lane-constant per subtile.
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub       = sub_m * kSubTilesN + sub_n;
                            const uint32_t n_in_wave = sub_n * kMfmaN + m_lane;
                            const uint32_t n_global  = n_block_idx * BLOCK_N + n_in_wave;
                            if (n_global >= kHidden)
                                continue;
#pragma unroll
                            for (uint32_t i = 0u; i < 16u; ++i) {
                                const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                            sub_m * kMfmaM + n_lane_half +
                                                            kNPattern[i];
                                if (m_in_block >= valid_m)
                                    continue;
                                write_combine(m_in_block, n_global, acc[sub][i]);
                            }
                        }
                    }
                }
            }
#if MEGA_MOE_PROFILE
            {
                const unsigned long long prof_epi_t1 = mega_moe_prof_now();
                if (phase == sched::BlockPhase::Linear1)
                    MEGA_MOE_PROF_ACC(prof_l1epi_lo, prof_l1epi_hi, prof_epi_t0, prof_epi_t1);
                else
                    MEGA_MOE_PROF_ACC(prof_l2epi_lo, prof_l2epi_hi, prof_epi_t0, prof_epi_t1);
            }
#endif
        });

        if (elect_one()) {
            auto *sink_ptr = reinterpret_cast<dtype::float32x16 *>(smem_buffer + sink_off);
#pragma unroll
            for (uint32_t sub = 0u; sub < kSubTilesPerWave; ++sub)
                sink_ptr[sub] = acc[sub];
        }

        sync_unaligned(named_bar, kNumThreads, kBarDispEpi);
#if MEGA_MOE_PROFILE
        if (elect_one()) {
            mega_moe_prof_commit(kProfL1Mma, prof_l1mma_lo, prof_l1mma_hi);
            mega_moe_prof_commit(kProfL1Epi, prof_l1epi_lo, prof_l1epi_hi);
            mega_moe_prof_commit(kProfL2Mma, prof_l2mma_lo, prof_l2mma_hi);
            mega_moe_prof_commit(kProfL2Epi, prof_l2epi_lo, prof_l2epi_hi);
            mega_moe_prof_commit(kProfTotal, prof_kernel_start, mega_moe_prof_now());
        }
#endif
        return;
    }

    if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
            (void) phase;
            (void) num_k_blocks;
            (void) local_expert_idx;
            (void) m_block_idx;
            (void) n_block_idx;
        });

        sync_unaligned(named_bar, kNumThreads, kBarDispEpi);

        sync_unaligned(named_bar, kNumDispatchThreads + kNumEpilogueThreads, kBarDispEpi2);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads, 1, 2>(
            workspace, sym_buffer, named_bar, kBarEpiGrid, kEpiLeader, sm_idx, thread_idx, true,
            true);

        const uint32_t epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(__hip_bfloat162);
        constexpr uint32_t kNumUint4PerToken = (kHidden * sizeof(__hip_bfloat16)) / sizeof(uint4);
        static_assert(
            (kHidden * sizeof(__hip_bfloat16)) % sizeof(uint4) == 0u,
            "hidden * sizeof(bf16) must be a multiple of uint4 for combine vectorization");

#if MEGA_MOE_PROFILE
        const unsigned long long prof_comb_t0 = mega_moe_prof_now();
#endif
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens; token_idx += kNumSMs * kNumEpilogueWarps) {

            const int slot =
                lane_idx < kNumTopk
                    ? static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                             token_idx * kNumTopk + lane_idx))
                    : -1;
            const uint64_t mask = ballot(slot >= 0);

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
#if MEGA_MOE_PROFILE
        if (elect_one()) {
            const unsigned long long prof_comb_t1 = mega_moe_prof_now();
            mega_moe_prof_commit(kProfCombine, prof_comb_t0, prof_comb_t1);
            mega_moe_prof_commit(kProfTotal, prof_kernel_start, prof_comb_t1);
        }
#endif
        return;
    }

#else
    PRIMUS_TURBO_DEVICE_CHECK(false);
#endif
}

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

    constexpr uint32_t kSmemBytes = 140u * 1024u;

    const dim3 grid(kNumSMs);
    const dim3 block(kNumThreads);

    auto kernel = gfx950_fp8_fp4_mega_moe_kernel<
        kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk,
        kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M, SF_BLOCK_M, SF_BLOCK_N,
        kNumMaxPoolTokens, kNumPaddedSFPoolTokens, kNumStages, kNumDispatchThreads,
        kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp,
        kFastMath>;

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
