// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Mega-MoE layout helpers — direct port of DeepGEMM's
// ``deep_gemm/include/deep_gemm/layout/mega_moe.cuh``.  These types
// describe how the symmetric memory buffer is sliced into a workspace
// region (barriers + per-expert counters + arrival masks + dispatch
// pulling state + combine source metadata) followed by the per-token
// data buffers (FP8 activations + UE8M0 SF + topk indices/weights +
// L1/L2 pools + BF16 combine buffer).
//
// Naming, field order and method signatures intentionally mirror DG
// so kernel code can read like a 1:1 translation; the only deltas are
// (1) namespace path (``primus_turbo::mega_moe::layout``), (2) AMD/HIP
// portable macros for device qualifiers, and (3) a direct
// ``align``/``align_up`` helper since we don't import ``cute::math``.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

#include "primus_turbo/mega_moe.h"

namespace primus_turbo {
namespace mega_moe {
namespace layout {

// ---------------------------------------------------------------------
//  Pool-token helpers (host + device).  Mirror DG's templated
//  ``get_num_max_pool_tokens`` / ``get_num_padded_sf_pool_tokens``.
// ---------------------------------------------------------------------

namespace detail {

template <typename T> __host__ __device__ constexpr T align_up(T value, T alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

template <typename T> __host__ __device__ constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

} // namespace detail

// Pool capacity for shared expert token pool: worst-case total tokens
// + per-expert BLOCK_M alignment padding, among all possible BLOCK_M.
template <typename T>
__host__ __device__ constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank,
                                                        T num_topk, T num_experts_per_rank) {
    const auto num_max_recv_tokens       = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = detail::constexpr_min(num_topk, num_experts_per_rank);
    return detail::align_up(num_max_recv_tokens * num_max_experts_per_token +
                                num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
                            static_cast<T>(kTokenAlignment));
}

// SF pool capacity: all experts share a contiguous SF region, sized by
// pool blocks × SF_BLOCK_M.
template <typename T>
__host__ __device__ constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
    return (num_max_pool_tokens / block_m) *
           detail::align_up(block_m, static_cast<T>(kScaleBlockMN));
}

// ---------------------------------------------------------------------
//  Per-token source metadata for combine write-back.
// ---------------------------------------------------------------------

struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

// ---------------------------------------------------------------------
//  Workspace region: leading sub-allocation inside ``sym_buffer`` that
//  holds barriers, expert counters, arrival masks, dispatch pulling
//  state and combine source metadata.  Layout matches DG exactly so
//  the device kernel can use the same pointer arithmetic.
// ---------------------------------------------------------------------

struct Workspace {
    void    *base;
    uint32_t num_ranks;
    uint32_t num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool capacity: all local experts share a contiguous token pool.
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // Combined grid-sync + NVLink-barrier signal pad.
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    // Sub-regions encoded inside ``kNumBarrierSignalBytes``:
    //   [ 0..15]: 4 × ``uint32_t`` grid sync counters
    //   [16..20]: ``uint32_t`` NVLink barrier counter
    //   [20..27]: 2 × ``int`` NVLink barrier signals (phase 0 / 1)
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    __host__ __device__ Workspace(void *base, const uint32_t &num_ranks,
                                  const uint32_t &num_experts,
                                  const uint32_t &num_max_tokens_per_rank, const uint32_t &num_topk)
        : base(base), num_ranks(num_ranks), num_experts(num_experts),
          num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank           = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk,
                                                      num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    __host__ __device__ uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier signal pad.
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv counts (one ``uint64_t`` each for send and recv).
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Per-local-expert recv count sum.
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count (padded to even entry count for ``uint64_t``
        // alignment of the L2 mask that follows).
        num_bytes += detail::align_up<uint32_t>(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask.
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Dispatch pulling source ``(token, topk)`` indices.
        num_bytes += static_cast<uint64_t>(num_experts_per_rank) * num_ranks *
                     num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices.
        num_bytes += static_cast<uint64_t>(num_max_pool_tokens) * sizeof(TokenSrcMetadata);

        // Round up to TMA descriptor alignment (16 B).
        return detail::align_up<uint64_t>(num_bytes, 16ull);
    }

    __host__ __device__ void *get_end_ptr() const {
        return static_cast<uint8_t *>(base) + get_num_bytes();
    }

    // ----- Device-only pointer accessors -----

    __device__ uint32_t *get_grid_sync_count_ptr(uint32_t index = 0) const {
        return static_cast<uint32_t *>(base) + index;
    }

    __device__ uint32_t *get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t *>(base) + kNumMaxGridSyncCounters;
    }

    __device__ int *get_nvl_barrier_signal_ptr(const uint32_t &phase) const {
        // NOTES: the signal is signed (we may subtract).
        return reinterpret_cast<int *>(static_cast<uint8_t *>(base) +
                                       (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) +
                                       phase * sizeof(int));
    }

    __device__ uint64_t *get_expert_send_count_ptr(const uint32_t &expert_idx = 0) const {
        return reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(base) + kNumBarrierSignalBytes) +
               expert_idx;
    }

    __device__ uint64_t *get_expert_recv_count_ptr(const uint32_t &rank_idx   = 0,
                                                   const uint32_t &expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank +
               expert_idx;
    }

    __device__ uint64_t *get_expert_recv_count_sum_ptr(const uint32_t &expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    __device__ uint32_t *get_l1_arrival_count_ptr(const uint32_t &pool_block_idx = 0) const {
        const auto base_after = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t *>(base_after) + pool_block_idx;
    }

    __device__ uint64_t *get_l2_arrival_mask_ptr(const uint32_t &pool_block_idx = 0) const {
        // Pad L1 entry count to even so that ``l2_arrival_mask`` is 8 B aligned.
        const auto base_after =
            get_l1_arrival_count_ptr(detail::align_up<uint32_t>(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t *>(base_after) + pool_block_idx;
    }

    __device__ uint32_t *get_src_token_topk_idx_ptr(const uint32_t &expert_idx = 0,
                                                    const uint32_t &rank_idx   = 0,
                                                    const uint32_t &token_idx  = 0) const {
        const auto base_after = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t *>(base_after) +
               expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
               rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    __device__ TokenSrcMetadata *
    get_token_src_metadata_ptr(const uint32_t &pool_token_idx = 0) const {
        const auto base_after =
            reinterpret_cast<TokenSrcMetadata *>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base_after + pool_token_idx;
    }
};

// ---------------------------------------------------------------------
//  Per-token data layout descriptor (host + device usable POD).
//  Mirrors DG's ``layout::Data``.
// ---------------------------------------------------------------------

struct Data {
    uint32_t num_bytes;
    bool     require_tma_alignment;
    void    *base;

    __host__ __device__ constexpr explicit Data(const uint32_t &num_bytes,
                                                const bool     &require_tma_alignment = true,
                                                void           *base                  = nullptr)
        : num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {}

    template <typename dtype_t = uint32_t>
    __host__ __device__ constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void> __host__ __device__ dtype_t *get_base_ptr() const {
        return static_cast<dtype_t *>(base);
    }

    __host__ __device__ void set_base_ptr(void *ptr) { base = ptr; }
};

// ---------------------------------------------------------------------
//  Multi-rank, multi-token buffer over a ``Data`` payload.  Mirrors
//  DG's ``layout::Buffer``.
// ---------------------------------------------------------------------

struct Buffer {
    Data     data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    void    *base;

    __host__ __device__ Buffer(const Data &data_layout, const uint32_t &num_ranks,
                               const uint32_t &num_max_tokens_per_rank, void *base = nullptr)
        : data_layout(data_layout), num_ranks(num_ranks),
          num_max_tokens_per_rank(num_max_tokens_per_rank), base(base) {}

    __host__ __device__ uint64_t get_num_bytes_per_rank() const {
        return static_cast<uint64_t>(num_max_tokens_per_rank) *
               data_layout.template get_num_bytes<uint64_t>();
    }

    __host__ __device__ uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void> __host__ __device__ dtype_t *get_base_ptr() const {
        return static_cast<dtype_t *>(base);
    }

    __host__ __device__ void *get_end_ptr() const {
        return static_cast<uint8_t *>(base) + get_num_bytes();
    }

    __host__ __device__ Buffer get_rank_buffer(const uint32_t &rank_idx) const {
        return Buffer(data_layout, 1, num_max_tokens_per_rank,
                      static_cast<uint8_t *>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    __host__ __device__ Data get_data_buffer(const uint32_t &token_idx,
                                             const bool     &global = false) const {
        // NOTE: assumes single rank unless ``global`` is set.
        (void) global;
        return Data(data_layout.num_bytes, data_layout.require_tma_alignment,
                    static_cast<uint8_t *>(base) +
                        data_layout.template get_num_bytes<uint64_t>() * token_idx);
    }
};

} // namespace layout
} // namespace mega_moe
} // namespace primus_turbo
