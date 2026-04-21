#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace primus_turbo::cco::ep {
namespace intranode {

void fused_dispatch_permute(void **buffer_ptrs, int *expert_tail_idx, int64_t *recv_topk_idx,
                            float *recv_topk_weights, int *dispatch_to_expert_map, void const *x,
                            float const *x_scales, int64_t const *topk_idx,
                            float const *topk_weights, bool const *is_token_in_rank,
                            int const *channel_prefix_matrix, int const *num_recv_tokens_per_expert,
                            void *recv_x, int num_tokens, int hidden_int4, int num_topk,
                            int num_experts, int num_scales, int scale_token_stride,
                            int scale_hidden_stride, int rank, int num_ranks, cudaStream_t stream,
                            int num_sms, int num_max_tokens, int num_max_send_tokens);

// Preprocessing kernel for the expert-grouped dispatch path.
//
// Builds per-(dst_rank, channel) group buckets so that ``expert_grouped_dispatch_permute``
// can start shipping group 0 **immediately** upon launch, with no on-kernel
// classification scan. The kernel writes two tensors to global memory:
//
//   * group_offsets              [num_ranks, num_channels, kMaxExpertGroupsPerRank + 1] int32
//   * sorted_token_offsets       [num_ranks, num_channels, kMaxLocalTokensPerSlice]     int16
//
// It should be scheduled alongside ``notify_dispatch`` (on the same stream,
// after the layout tensors are ready) so that the buckets are visible to
// the dispatch kernel's launch with no extra synchronization.
void expert_grouped_build_buckets(int64_t const *topk_idx, bool const *is_token_in_rank,
                                  int num_tokens, int num_channels, int num_topk, int num_experts,
                                  int num_experts_per_group, int num_ranks, int *group_offsets,
                                  int16_t *sorted_token_offsets, cudaStream_t stream);

// Pipelined expert-grouped dispatch + permute.
//
// Extends `fused_dispatch_permute` by sending tokens in expert-group order on
// every (src_rank, dst_rank, channel) stream. Tokens whose primary local
// expert lies in group `g` (experts `[g * num_experts_per_group,
// (g+1) * num_experts_per_group)`) are forwarded in phase `g`. Because each
// phase strictly precedes the next and the receiver atomically bumps
// `expert_tail_idx[e]` with release semantics, a GroupedGEMM consumer polling
// that counter can start computing experts in group 0 as soon as phase 0
// completes — overlapping the remainder of dispatch with the compute of the
// first expert group.
//
// ``group_offsets`` and ``sorted_token_offsets`` must have been produced by
// ``expert_grouped_build_buckets`` for the same ``num_experts_per_group``.
void expert_grouped_dispatch_permute(
    void **buffer_ptrs, int *expert_tail_idx, int64_t *recv_topk_idx, float *recv_topk_weights,
    int *dispatch_to_expert_map, void const *x, float const *x_scales, int64_t const *topk_idx,
    float const *topk_weights, bool const *is_token_in_rank, int const *channel_prefix_matrix,
    int const *num_recv_tokens_per_expert, int const *group_offsets,
    int16_t const *sorted_token_offsets, void *recv_x, int num_tokens, int hidden_int4,
    int num_topk, int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
    int rank, int num_ranks, cudaStream_t stream, int num_sms, int num_max_tokens,
    int num_max_send_tokens, int num_experts_per_group);

// void fused_unpermute_combine(void **buffer_ptrs, void const *permuted_x,
//                              float const *permuted_weights, int const *dense_to_expert_map,
//                              int const *expert_offsets, void *combined_x, int num_recv_tokens,
//                              int hidden_int4, int num_topk, int num_experts_per_rank, int rank,
//                              int num_ranks, cudaStream_t stream, int num_sms);

} // namespace intranode
} // namespace primus_turbo::cco::ep
