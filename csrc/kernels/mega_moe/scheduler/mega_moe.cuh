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

#include "../impls/prims.cuh"
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

    // NOTES: SM100 ran in a 2-CTA cluster where adjacent CTAs shared a
    // single ``m_block_idx`` with ``n_block_idx`` differing by 1, which
    // required ``kNumSMs`` and ``kNum{L1,L2}BlockNs`` to all be even.
    // The AMD port collapses the cluster to a single CTA per tile, so
    // the iteration walks every (m, n) block independently — there is
    // no pairing constraint to maintain. The asserts were dropped to
    // unlock e.g. ``BLOCK_N=192`` with ``hidden=7168`` (an odd N-block
    // count). See TODO.md Section A item 10.

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

    // Per-lane expert -> token count exchange.  All call sites pass a
    // wave-uniform ``expert_idx`` (driven by the scheduler's persistent
    // state), so the per-lane equality scan in DG's original wave32
    // pattern collapses to a direct array lookup + ``__shfl`` broadcast.
    // The lane that holds ``expert_idx``'s count is ``expert_idx %
    // kWarpSize``; the slot inside its per-lane cache is
    // ``expert_idx / kWarpSize``.  See TODO.md Section A item 12.
    __device__ uint32_t get_num_tokens(const uint32_t &expert_idx) const {
        return __shfl(stored_num_tokens_per_expert[expert_idx / prims::kWarpSize],
                      static_cast<int>(expert_idx & (prims::kWarpSize - 1u)));
    }

    // Per-lane expert -> pool block offset reduction.  AMD wave64 port
    // using ``__reduce_add_sync`` semantics emulated via ``ds_swizzle`` -
    // backed cross-lane reductions (HIP exposes ``__reduce_add`` only on
    // newer toolchains; we fall back to a manual butterfly).
    __device__ uint32_t get_pool_block_offset(const uint32_t &expert_idx) {
        const uint32_t lane       = prims::get_lane_idx();
        uint32_t       num_blocks = 0;
#pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            if (i * prims::kWarpSize + lane < expert_idx)
                num_blocks += detail::constexpr_ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        // Wave64 butterfly sum (matches ``__reduce_add_sync(0xffffffff,..)``).
#pragma unroll
        for (int offset = prims::kWarpSize / 2; offset > 0; offset >>= 1)
            num_blocks += __shfl_xor(num_blocks, offset);
        return num_blocks;
    }

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

    // Wait for all expert counters to be finalized.  Direct AMD wave64
    // port of DG's wave32 spin loop (``ptx::ld_volatile`` -> HIP
    // ``__atomic_load_n`` with ``__ATOMIC_RELAXED``; the high 32 bits
    // of the counter track how many SMs × Ranks have arrived).
    //
    // Memory-scope rationale (mirrors the policy in
    // ``impls/gfx950_fp8_fp4_mega_moe.cuh`` head comment):
    //   * Single-rank (``kNumRanks == 1``): all writers are local SMs
    //     on the same agent (see the ``atomic_add_sys`` at the dispatch
    //     write site - in single-rank, ``sym_buffer.map(ptr, 0)``
    //     resolves to a local pointer).  AGENT scope is sufficient and
    //     avoids a system-wide cache invalidate on every poll, which
    //     would otherwise serialize all CTAs through Infinity Fabric
    //     and inflate the spin from microseconds to seconds.
    //   * Multi-rank (``kNumRanks > 1``): the counter is incremented
    //     via XGMI by remote ranks, so SYSTEM scope is required for
    //     the reader to observe those remote writes.
    __device__ void fetch_expert_recv_count() {
        const uint32_t lane = prims::get_lane_idx();
#pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            const uint32_t expert_idx = i * prims::kWarpSize + lane;
            uint64_t       value      = 0;
            if (expert_idx < kNumExpertsPerRank) {
                auto *ptr = workspace.get_expert_recv_count_sum_ptr(expert_idx);
                do {
                    if constexpr (kNumRanks == 1u) {
                        value = __hip_atomic_load(reinterpret_cast<unsigned long long *>(ptr),
                                                  __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                    } else {
                        value = __hip_atomic_load(reinterpret_cast<unsigned long long *>(ptr),
                                                  __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
                    }
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __builtin_amdgcn_wave_barrier();
    }

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
