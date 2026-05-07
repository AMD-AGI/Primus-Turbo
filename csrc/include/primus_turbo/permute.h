// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/dtype.h"

namespace primus_turbo {

template <typename expert_map_t>
void permute_preprocessing_impl(const expert_map_t *expert_map, int num_topk,
                                int *num_dispatched_tokens_ptr, int num_local_experts,
                                int max_num_dispatched_tokens, int pad_multiple,
                                int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                int64_t num_permuted_tokens, hipStream_t stream);

template <typename DType, typename ProbType, typename ScalarType>
void permute_impl(const DType *tokens, DType *permuted_tokens, const ScalarType *scaling_factor,
                  ScalarType *permuted_scaling_factor, const ProbType *probs,
                  ProbType *permuted_probs, const int *row_id_map,
                  const int *num_dispatched_tokens_ptr, int pad_multiple, int num_local_experts,
                  int hidden_size, int scales_per_token, int local_rank, int num_ranks_per_node,
                  int num_cu, hipStream_t stream);

template <typename DType, typename ProbType>
void unpermute_impl(const DType *permuted_tokens, DType *tokens, const ProbType *permuted_probs,
                    ProbType *probs, const int *row_id_map, const int *num_dispatched_tokens_ptr,
                    int num_local_experts, int hidden_size, int local_rank, int num_ranks_per_node,
                    int num_cu, hipStream_t stream);

} // namespace primus_turbo
