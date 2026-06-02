// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/dtype.h"

namespace primus_turbo {

template <typename expert_map_t>
void permute_preprocessing_impl(const expert_map_t *expert_map, int num_topk,
                                int *num_dispatched_tokens_out, int num_local_experts,
                                int max_num_dispatched_tokens, int pad_multiple,
                                int64_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                int64_t num_permuted_tokens, int probs_topk_stride,
                                hipStream_t stream);

template <typename dtype_t, typename prob_t, typename scalar_t>
void permute_impl(const dtype_t *tokens, dtype_t *permuted_tokens, const scalar_t *scaling_factor,
                  scalar_t *permuted_scaling_factor, const prob_t *probs, prob_t *permuted_probs,
                  const int *row_id_map, const int *num_dispatched_tokens_ptr, int pad_multiple,
                  int num_local_experts, int hidden_size, int scales_per_token,
                  int num_dispatched_max, int probs_stride, hipStream_t stream);

template <typename dtype_t, typename prob_t>
void unpermute_impl(const dtype_t *permuted_tokens, dtype_t *tokens, const prob_t *permuted_probs,
                    prob_t *probs, const int *row_id_map, const int *num_dispatched_tokens_ptr,
                    int num_local_experts, int hidden_size, int num_dispatched_max,
                    int probs_stride, hipStream_t stream);

} // namespace primus_turbo
