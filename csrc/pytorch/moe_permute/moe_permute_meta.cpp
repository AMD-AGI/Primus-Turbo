// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

namespace primus_turbo::pytorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
permute_preprocessing_meta(torch::Tensor expert_map, int64_t num_local_experts,
                           int64_t /*num_topk*/, int64_t     pad_multiple,
                           int64_t /*num_permuted_tokens*/, int64_t /*probs_topk_stride*/) {
    auto int_opts                  = at::TensorOptions().dtype(at::kInt).device(at::kMeta);
    auto long_opts                 = at::TensorOptions().dtype(at::kLong).device(at::kMeta);
    auto max_num_dispatched_tokens = expert_map.sizes()[0];
    auto row_id_map =
        at::empty({max_num_dispatched_tokens + pad_multiple, 2 * num_local_experts + 1}, int_opts);
    auto tokens_per_expert     = at::empty({num_local_experts}, long_opts);
    auto overflow_flag         = at::empty({1}, int_opts);
    auto num_dispatched_tokens = at::empty({1}, int_opts);
    return {row_id_map, tokens_per_expert, overflow_flag, num_dispatched_tokens};
}

void permute_meta(at::Tensor /*tokens*/, at::Tensor /*output_tokens*/,
                  c10::optional<at::Tensor> /*scaling_factor*/,
                  c10::optional<at::Tensor> /*output_scaling_factor*/,
                  c10::optional<at::Tensor> /*probs*/, c10::optional<at::Tensor> /*output_probs*/,
                  at::Tensor /*row_id_map*/, at::Tensor /*num_dispatched_token_tensor*/,
                  int64_t /*pad_multiple*/, int64_t /*num_local_experts*/, int64_t /*hidden_size*/,
                  int64_t /*scales_per_token*/, bool /*use_fp8*/, bool /*with_probs*/,
                  int64_t /*num_permuted_token*/, int64_t /*probs_stride*/) {}

void unpermute_meta(at::Tensor /*permuted_tokens*/, at::Tensor /*output_tokens*/,
                    c10::optional<at::Tensor> /*permuted_probs*/,
                    c10::optional<at::Tensor> /*output_probs*/, at::Tensor /*row_id_map*/,
                    at::Tensor /*num_dispatched_tokens_tensor*/, int64_t /*num_local_experts*/,
                    int64_t /*hidden_size*/, bool /*with_probs*/, int64_t /*probs_stride*/) {}

} // namespace primus_turbo::pytorch
