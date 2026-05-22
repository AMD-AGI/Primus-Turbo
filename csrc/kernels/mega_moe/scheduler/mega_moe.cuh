// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Mega-MoE per-CTA scheduler.  Direct port of DeepGEMM's
// ``deep_gemm::sched::MegaMoEScheduler`` (deep_gemm/scheduler/mega_moe.cuh).
//
// The scheduler walks the per-expert L1 and L2 GEMM blocks assigned
// to this CTA via the state machine
//
//      [Linear1 → fetch_next_l1_block]    ─ for each expert in wave ─┐
//                                                                     │
//      [Linear2 → fetch_next_l2_block]    ─ for each expert in wave ──┴── repeat next wave
//
// Compile-time block / shape / wave parameters are mirrored 1:1 from
// DG so callers can swap backends without touching template
// arguments.  The device-only method bodies (``fetch_expert_recv_count``,
// ``for_each_block``) deliberately leave the AMD wave64 intrinsics
// unimplemented — only the C++ API surface is brought across; the
// ``__global__`` content sits in the impls/ device kernel and will
// fill these in later.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

#include "../layout/mega_moe.cuh"

namespace primus_turbo {
namespace mega_moe {
namespace sched {

// Computation phase for the current block.  Mirrors DG's
// ``deep_gemm::sched::BlockPhase``.
enum class BlockPhase {
    None    = 0,
    Linear1 = 1,
    Linear2 = 2,
};

namespace detail {

template <typename T> __host__ __device__ constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T> __host__ __device__ constexpr T align_up(T value, T alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace detail

template <
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
    uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K, uint32_t kNumExpertsPerRank,
    uint32_t kNumExpertsPerWave, uint32_t kNumSMs, uint32_t kNumRanks,
    uint32_t kNumExpertsPerLane = detail::constexpr_ceil_div(kNumExpertsPerRank, 32u),
    uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N, uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
    uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K, uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
    static_assert(L1_SHAPE_N % BLOCK_N == 0, "Invalid L1 N shape");
    static_assert(L2_SHAPE_N % BLOCK_N == 0, "Invalid L2 N shape");
    static_assert(L1_SHAPE_K % BLOCK_K == 0, "Invalid L1 K shape");
    static_assert(L2_SHAPE_K % BLOCK_K == 0, "Invalid L2 K shape");
    static_assert(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");

    // NOTES: 2-CTA cluster requires even N block counts so that two
    // adjacent CTAs always land on the same ``m_block_idx`` with
    // ``n_block_idx`` differing by 1.
    static_assert(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    static_assert(kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    static_assert(kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // Arrival counts owned by the kernel-level workspace.
    const layout::Workspace &workspace;

    // Scheduler state.
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices.
    uint32_t current_local_expert_idx  = 0;
    uint32_t current_num_tokens        = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx                 = 0;
    uint32_t m_block_idx               = 0;
    uint32_t n_block_idx               = 0;

    // Pre-cached per-expert token counts (filled during ``for_each_block`` init).
    // Layout: ``stored_num_tokens_per_expert[i]`` holds expert
    // ``(i * 32 + lane_idx)``'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    __device__ explicit MegaMoEScheduler(const layout::Workspace &workspace)
        : workspace(workspace) {
        block_idx = blockIdx.x;
    }

    __device__ uint32_t get_wave_expert_end_idx() const {
        return detail::align_up(current_local_expert_idx + 1, kNumExpertsPerWave);
    }

    // Per-lane expert -> token count exchange.  Device-only; left as a
    // declaration here because the AMD wave64 implementation lives in
    // the gfx950 device kernel (see impls/).
    __device__ uint32_t get_num_tokens(const uint32_t &expert_idx) const;

    // Per-lane expert -> pool block offset reduction.  See note above.
    __device__ uint32_t get_pool_block_offset(const uint32_t &expert_idx);

    __device__ void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    __device__ void set_expert_idx(const uint32_t &expert_idx) {
        current_local_expert_idx  = expert_idx;
        current_num_tokens        = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    __device__ uint32_t get_current_pool_block_offset() const { return current_pool_block_offset; }

    __device__ uint32_t get_current_num_m_blocks() const {
        return detail::constexpr_ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false> __device__ uint32_t get_valid_m() const {
        const uint32_t remain = current_num_tokens - m_block_idx * BLOCK_M;
        const uint32_t m      = remain < BLOCK_M ? remain : BLOCK_M;
        return kDoUMMAAligned ? detail::align_up<uint32_t>(m, 16u) : m;
    }

    __device__ bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx             = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;

            // Current expert is fully assigned, move to the next.
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    __device__ bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }

            // Current expert is fully assigned, move to the next.
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // Core state machine: assigns the next block to this CTA.  Returns
    // the tuple ``(phase, expert_idx, m_block_idx, n_block_idx)``.
    struct NextBlock {
        BlockPhase phase;
        uint32_t   expert_idx;
        uint32_t   m_block_idx;
        uint32_t   n_block_idx;
    };

    __device__ NextBlock get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx,
                            n_block_idx};
                }
                // L1 for the current wave is complete, transition to L2.
                next_phase = BlockPhase::Linear2;
                const uint32_t wave_start =
                    (current_local_expert_idx - 1) / kNumExpertsPerWave * kNumExpertsPerWave;
                set_expert_idx(wave_start);
            } else {
                if (fetch_next_l2_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx,
                            n_block_idx};
                }
                // Move to L1 of the next wave.
                next_phase = BlockPhase::Linear1;
            }
        }

        // All waves and experts are fully processed.
        return {BlockPhase::None, 0u, 0u, 0u};
    }

    // Wait for all expert counters to be finalized; device-only.  See
    // note above ``get_num_tokens``.
    __device__ void fetch_expert_recv_count();

    template <typename Func> __device__ void for_each_block(Func &&func) {
        fetch_expert_recv_count();

        set_expert_idx(0);

        while (true) {
            const NextBlock nb = get_next_block();
            if (nb.phase == BlockPhase::None)
                break;
            func(nb.phase, nb.expert_idx,
                 nb.phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs, nb.m_block_idx,
                 nb.n_block_idx);
        }
    }
};

} // namespace sched
} // namespace mega_moe
} // namespace primus_turbo
