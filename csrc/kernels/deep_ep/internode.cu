/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#include "buffer.h"
#include "configs.h"
#include "launch.h"
#include "primus_turbo/macros.h"
#include "utils.h"
#include <hip/hip_runtime.h>

namespace primus_turbo::deep_ep {

namespace internode {

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    PRIMUS_TURBO_STATIC_CHECK(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool *is_token_in_nvl_ranks) {
        src_rdma_rank             = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
#pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
        return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
    }
};

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

void notify_dispatch(const int *num_tokens_per_rank, int *moe_recv_counter_mapped, int num_ranks,
                     const int *num_tokens_per_rdma_rank, int *moe_recv_rdma_counter_mapped,
                     const int *num_tokens_per_expert, int *moe_recv_expert_counter_mapped,
                     int num_experts, const bool *is_token_in_rank, int num_tokens,
                     int num_channels, int hidden_int4, int num_scales, int num_topk,
                     int expert_alignment, int *rdma_channel_prefix_matrix,
                     int *recv_rdma_rank_prefix_sum, int *gbl_channel_prefix_matrix,
                     int *recv_gbl_rank_prefix_sum, void *rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens, void **buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens, int **task_fifo_ptrs, int head, int rank,
                     hipStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                     bool low_latency_mode) {
    PRIMUS_TURBO_CHECK(false, "not support");
}

void dispatch(void *recv_x, float *recv_x_scales, int64_t *recv_topk_idx, float *recv_topk_weights,
              void *recv_src_meta, const void *x, const float *x_scales, const int64_t *topk_idx,
              const float *topk_weights, int *send_rdma_head, int *send_nvl_head,
              int *recv_rdma_channel_prefix_matrix, int *recv_gbl_channel_prefix_matrix,
              const int *rdma_channel_prefix_matrix, const int *recv_rdma_rank_prefix_sum,
              const int *gbl_channel_prefix_matrix, const int *recv_gbl_rank_prefix_sum,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              const bool *is_token_in_rank, void *rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void **buffer_ptrs, int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens, int rank, int num_ranks, bool is_cached_dispatch,
              hipStream_t stream, int num_channels, bool low_latency_mode) {

    PRIMUS_TURBO_CHECK(false, "not support");
}

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_channels, int num_combined_tokens,
                   int *combined_rdma_head, const int *rdma_channel_prefix_matrix,
                   const int *rdma_rank_prefix_sum, int *combined_nvl_head, void *rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens, void **buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens, int **task_fifo_ptrs, int head, int rank,
                   hipStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode) {
    PRIMUS_TURBO_CHECK(false, "not support");
}

void combine(hipDataType type, void *combined_x, float *combined_topk_weights,
             const bool *is_combined_token_in_rank, const void *x, const float *topk_weights,
             const int *combined_rdma_head, const int *combined_nvl_head, const void *src_meta,
             const int *rdma_channel_prefix_matrix, const int *rdma_rank_prefix_sum,
             const int *gbl_channel_prefix_matrix, int num_tokens, int num_combined_tokens,
             int hidden, int num_topk, void *rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens, void **buffer_ptrs,
             int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens, int rank,
             int num_ranks, hipStream_t stream, int num_channels, bool low_latency_mode) {
    PRIMUS_TURBO_CHECK(false, "not support");
}
} // namespace internode

} // namespace primus_turbo::deep_ep
