#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace primus_turbo::cco::ep {

void get_dispatch_layout(int64_t const *topk_idx, int *num_tokens_per_rank,
                         int *num_tokens_per_rdma_rank, int *num_tokens_per_expert,
                         bool *is_token_in_rank, int num_tokens, int num_topk, int num_ranks,
                         int num_experts, cudaStream_t stream);

namespace intranode {

void notify_dispatch(int const *num_tokens_per_rank, int *moe_recv_counter_mapped, int num_ranks,
                     int const *num_tokens_per_expert, int *moe_recv_expert_counter_mapped,
                     int num_experts, int num_tokens, bool const *is_token_in_rank,
                     int *channel_prefix_matrix, int *rank_prefix_matrix_copy, int num_memset_int,
                     int expert_alignment, void **buffer_ptrs, int **barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_channels);

void cached_notify_dispatch(int const *rank_prefix_matrix, int num_memset_int, void **buffer_ptrs,
                            int **barrier_signal_ptrs, int rank, int num_ranks,
                            cudaStream_t stream);

void dispatch(void **recv_x, float **recv_x_scales, int **recv_src_idx, int64_t **recv_topk_idx,
              float **recv_topk_weights, int **recv_channel_offset, int *send_head, void const *x,
              float const *x_scales, int64_t const *topk_idx, float const *topk_weights,
              bool const *is_token_in_rank, int const *channel_prefix_matrix,
              int const *rank_prefix_matrix, int num_tokens, int hidden_int4, int num_topk,
              int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
              int rank, int num_ranks, cudaStream_t stream, int num_sms);
} // namespace intranode
} // namespace primus_turbo::cco::ep