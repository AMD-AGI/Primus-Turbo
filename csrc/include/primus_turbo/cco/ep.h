#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace primus_turbo::cco::ep {
namespace intranode {

void fused_dispatch_permute(void **buffer_ptrs, int64_t *recv_topk_idx, float *recv_topk_weights,
                            int *permute_src_row_id, int *dense_to_expert_map,
                            void const *x, float const *x_scales, int64_t const *topk_idx,
                            float const *topk_weights, bool const *is_token_in_rank,
                            int const *channel_prefix_matrix, int const *num_recv_tokens_per_expert,
                            void *recv_x, int num_tokens, int hidden_int4, int num_topk,
                            int num_experts, int num_scales, int scale_token_stride,
                            int scale_hidden_stride, int rank, int num_ranks, cudaStream_t stream,
                            int num_sms, int num_max_tokens, int num_max_send_tokens);


void fused_unpermute_combine(void **buffer_ptrs, void const *permuted_x,
                             float const *permuted_weights, int const *dense_to_expert_map,
                             int const *expert_offsets, void *combined_x, int num_recv_tokens,
                             int hidden_int4, int num_topk, int num_experts_per_rank, int rank,
                             int num_ranks, cudaStream_t stream, int num_sms);

} // namespace intranode
} // namespace primus_turbo::cco::ep