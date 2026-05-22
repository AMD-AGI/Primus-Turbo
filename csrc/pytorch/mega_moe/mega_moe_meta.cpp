// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

#include "primus_turbo/mega_moe.h"

namespace primus_turbo::pytorch {

int64_t get_token_alignment_for_mega_moe_meta() {
    return primus_turbo::mega_moe::kTokenAlignment;
}

at::Tensor get_symm_buffer_size_for_mega_moe_meta(
    const int64_t /*num_ranks*/, const int64_t /*num_experts*/,
    const int64_t /*num_max_tokens_per_rank*/, const int64_t /*num_topk*/, const int64_t /*hidden*/,
    const int64_t /*intermediate_hidden*/, const bool /*use_fp8_dispatch*/) {
    return at::empty({14}, at::dtype(at::kLong).device(at::kMeta));
}

void fp8_fp4_mega_moe_meta(
    at::Tensor /*y*/, at::Tensor /*l1_weights*/, at::Tensor /*l1_weights_sf*/,
    at::Tensor /*l2_weights*/, at::Tensor /*l2_weights_sf*/,
    c10::optional<at::Tensor> /*cumulative_local_expert_recv_stats*/, at::Tensor /*sym_buffer*/,
    const std::vector<int64_t> & /*sym_buffer_ptrs*/, const int64_t /*rank_idx*/,
    const int64_t /*num_max_tokens_per_rank*/, const int64_t /*num_experts*/,
    const int64_t /*num_topk*/, const int64_t /*num_tokens*/, const int64_t /*hidden*/,
    const int64_t /*intermediate_hidden*/, const std::vector<int64_t> & /*recipe*/,
    const std::string & /*activation*/, const double /*activation_clamp*/,
    const bool /*fast_math*/) {
    // No outputs other than ``y`` which is written in-place; meta impl is a no-op.
}

} // namespace primus_turbo::pytorch
