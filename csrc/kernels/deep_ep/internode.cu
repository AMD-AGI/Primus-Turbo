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

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void __launch_bounds__(kNumThreads, 1)
    get_dispatch_layout(const int64_t *topk_idx, int *num_tokens_per_rank,
                        int *num_tokens_per_rdma_rank, int *num_tokens_per_expert,
                        bool *is_token_in_rank, int num_tokens, int num_topk, int num_ranks,
                        int num_experts) {
    auto sm_id     = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    int            expert_begin_idx = sm_id * kNumExpertsPerSM,
        expert_end_idx              = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    if (expert_begin_idx < expert_end_idx) {
// Per-thread count
#pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
#pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        PRIMUS_TURBO_STATIC_CHECK(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        PRIMUS_TURBO_DEVICE_CHECK(num_ranks % NUM_MAX_XGMI_PEERS == 0 and
                                  num_ranks > NUM_MAX_XGMI_PEERS);

    // Count rank statistics
    constexpr int  kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_XGMI_PEERS;
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto           sm_begin       = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    int            rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM,
        rank_end_idx              = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx       = rank_begin_idx / NUM_MAX_XGMI_PEERS,
        rdma_rank_end_idx         = rank_end_idx / NUM_MAX_XGMI_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto       expert_begin        = rank_begin_idx * num_expert_per_rank;
        auto       expert_end          = rank_end_idx * num_expert_per_rank;

// Per-thread count
#pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx           = topk_idx + i * num_topk;
            int  is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_XGMI_PEERS]++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

#pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        PRIMUS_TURBO_STATIC_CHECK(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and
            rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(const int64_t *topk_idx, int *num_tokens_per_rank,
                         int *num_tokens_per_rdma_rank, int *num_tokens_per_expert,
                         bool *is_token_in_rank, int num_tokens, int num_topk, int num_ranks,
                         int num_experts, hipStream_t stream) {
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 32, kNumRanksPerSM = 8;
    int           num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
                  (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    PRIMUS_TURBO_STATIC_CHECK(kNumExpertsPerSM % NUM_MAX_XGMI_PEERS == 0,
                              "Invalid number of experts per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL_NON_COOPERATIVE(
        &cfg, (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>), topk_idx,
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank,
        num_tokens, num_topk, num_ranks, num_experts);
}

int get_source_meta_bytes() {
    return 0;
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
