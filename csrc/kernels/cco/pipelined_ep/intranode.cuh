#pragma once

#include <vector>

#include "configs.cuh"

namespace primus_turbo::cco::pipelined_ep {

// Intranode runtime
namespace intranode {

void barrier(int **barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);

} // namespace intranode

// Layout kernels
namespace layout {

template <typename topk_idx_t>
void get_dispatch_layout(const topk_idx_t *topk_idx, int *num_tokens_per_rank,
                         int *num_tokens_per_rdma_rank, int *num_tokens_per_expert,
                         bool *is_token_in_rank, int num_tokens, int num_topk, int num_ranks,
                         int num_experts, cudaStream_t stream);

} // namespace layout

// Intranode kernels
namespace intranode {

void notify_dispatch(const int *num_tokens_per_rank, int *moe_recv_counter_mapped, int num_ranks,
                     const int *num_tokens_per_expert, int *moe_recv_expert_counter_mapped,
                     int num_experts, int num_tokens, const bool *is_token_in_rank,
                     int *channel_prefix_matrix, int *rank_prefix_matrix_copy, int num_memset_int,
                     int expert_alignment, void **buffer_ptrs, int **barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_sms);

void cached_notify_dispatch(const int *rank_prefix_matrix, int num_memset_int, void **buffer_ptrs,
                            int **barrier_signal_ptrs, int rank, int num_ranks,
                            cudaStream_t stream);

template <typename topk_idx_t>
void dispatch(void *recv_x, float *recv_x_scales, int *recv_src_idx, topk_idx_t *recv_topk_idx,
              float *recv_topk_weights, int *recv_channel_offset, int *send_head, const void *x,
              const float *x_scales, const topk_idx_t *topk_idx, const float *topk_weights,
              const bool *is_token_in_rank, const int *channel_prefix_matrix, int num_tokens,
              int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
              int scale_token_stride, int scale_hidden_stride, void **buffer_ptrs, int rank,
              int num_ranks, cudaStream_t stream, int num_sms, int num_max_send_tokens,
              int num_recv_buffer_tokens);

} // namespace intranode

} // namespace primus_turbo::cco::pipelined_ep
