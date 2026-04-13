// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/cco/ep.h"
#include "../extensions.h"
#include "../type_traits.h"
#include "kernels/cco/ep/exception.cuh"
#include "primus_turbo/arch.h"

namespace primus_turbo::pytorch::cco::intranode {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fused_dispatch_permute_preprocess(const torch::Tensor &topk_idx,
                                  const torch::Tensor &buffer_ptrs_dev,
                                  const torch::Tensor &barrier_signal_ptrs_dev,
                                  torch::Tensor        moe_recv_counter,
                                  torch::Tensor moe_recv_expert_counter, int num_experts,
                                  int expert_alignment, int num_worst_tokens, int rank,
                                  int num_ranks, int num_sms) {
    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    auto num_experts_per_rank = num_experts / num_ranks;
    auto num_channels         = num_sms / 2;

    auto stream     = at::cuda::getCurrentCUDAStream();
    auto int32_opts = torch::TensorOptions().dtype(torch::kInt32).device(topk_idx.device());
    auto bool_opts  = torch::TensorOptions().dtype(torch::kBool).device(topk_idx.device());

    auto num_tokens            = static_cast<int>(topk_idx.size(0)),
         num_topk              = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank   = torch::empty({num_ranks}, int32_opts);
    auto num_tokens_per_expert = torch::empty({num_experts}, int32_opts);
    auto is_token_in_rank      = torch::empty({num_tokens, num_ranks}, bool_opts);
    // if (false)
    //     num_tokens_per_rdma_rank = torch::empty({1}, int32_opts);

    // 1. Compute dispatch layout from topk_idx
    primus_turbo::deep_ep::layout::get_dispatch_layout(
        topk_idx.data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(), nullptr,
        num_tokens_per_expert.data_ptr<int>(), is_token_in_rank.data_ptr<bool>(), num_tokens,
        num_topk, num_ranks, num_experts, stream);

    // 2. Allocate output tensors — expert_prefix is [R_dest, R_src, E_r]
    auto rank_prefix_matrix    = torch::empty({num_ranks, num_ranks}, int32_opts);
    auto channel_prefix_matrix = torch::empty({num_ranks, num_channels}, int32_opts);
    auto expert_prefix = torch::empty({num_ranks, num_ranks, num_experts_per_rank}, int32_opts);
    auto channel_expert_prefix =
        torch::empty({num_ranks, num_channels, num_experts_per_rank}, int32_opts);

    int num_memset_int = num_channels * num_ranks * 4;

    void **buffer_ptrs_gpu = reinterpret_cast<void **>(buffer_ptrs_dev.data_ptr<int64_t>());
    int  **barrier_signal_ptrs_gpu =
        reinterpret_cast<int **>(barrier_signal_ptrs_dev.data_ptr<int64_t>());

    // 3. notify_dispatch_permute computes combined expert_prefix in-kernel
    primus_turbo::deep_ep::intranode::notify_dispatch(
        num_tokens_per_rank.data_ptr<int>(), moe_recv_counter.data_ptr<int>(), num_ranks,
        num_tokens_per_expert.data_ptr<int>(), moe_recv_expert_counter.data_ptr<int>(), num_experts,
        num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
        rank_prefix_matrix.data_ptr<int>(), num_memset_int, expert_alignment, buffer_ptrs_gpu,
        barrier_signal_ptrs_gpu, rank, stream, num_channels);

    return {num_tokens_per_rank,   num_tokens_per_expert,   is_token_in_rank,
            moe_recv_counter,      moe_recv_expert_counter, rank_prefix_matrix,
            channel_prefix_matrix, expert_prefix,           channel_expert_prefix};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_dispatch_permute(const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales,
                       const torch::Tensor &topk_idx, const std::optional<torch::Tensor> &topk_weights,
                       const torch::Tensor &is_token_in_rank, const torch::Tensor &channel_prefix_matrix,
                       const torch::Tensor &num_recv_tokens_per_expert, const torch::Tensor &buffer_ptrs_dev,
                       int num_worst_tokens, int num_permuted_tokens, int num_experts, int rank,
                       int num_ranks, int num_sms, int num_max_send_tokens) {

    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(num_sms % 2 == 0);

    void **buffer_ptrs_gpu = reinterpret_cast<void **>(buffer_ptrs_dev.data_ptr<int64_t>());

    auto num_tokens          = static_cast<int>(x.size(0));
    auto hidden              = static_cast<int>(x.size(1));
    auto num_topk            = static_cast<int>(topk_idx.size(1));
    auto num_experts_per_rank = num_experts / num_ranks;

    float *topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    float *x_scales_ptr = nullptr;
    int    num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or
                       x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales          = static_cast<int>(x_scales->size(1));
        x_scales_ptr        = static_cast<float *>(x_scales->data_ptr());
        scale_token_stride  = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    auto stream     = at::cuda::getCurrentCUDAStream();
    auto int32_opts = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto recv_x     = torch::zeros({num_worst_tokens * num_topk, hidden}, x.options());

    auto recv_topk_idx =
        torch::empty({num_worst_tokens, num_topk},
                     torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    auto recv_topk_weights =
        torch::empty({num_worst_tokens, num_topk},
                     torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    auto permute_src_row_id = torch::empty({num_permuted_tokens}, int32_opts);
    // dense_to_expert_map: [num_worst_tokens, num_experts_per_rank] initialized to -1
    auto dense_to_expert_map =
        torch::full({num_worst_tokens, num_experts_per_rank}, -1, int32_opts);

    primus_turbo::cco::ep::intranode::fused_dispatch_permute(
        buffer_ptrs_gpu, recv_topk_idx.data_ptr<int64_t>(), recv_topk_weights.data_ptr<float>(),
        permute_src_row_id.data_ptr<int>(), dense_to_expert_map.data_ptr<int>(), x.data_ptr(),
        x_scales_ptr, topk_idx.data_ptr<int64_t>(), topk_weights_ptr,
        is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
        num_recv_tokens_per_expert.data_ptr<int>(), recv_x.data_ptr(), num_tokens,
        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)), num_topk, num_experts,
        num_scales, scale_token_stride, scale_hidden_stride, rank, num_ranks, stream, num_sms,
        num_worst_tokens, num_max_send_tokens);

    return std::make_tuple(recv_x, recv_topk_idx, recv_topk_weights, permute_src_row_id,
                           dense_to_expert_map);
}

// torch::Tensor fused_unpermute_combine(const torch::Tensor &permuted_x,
//                                       const torch::Tensor &permuted_weights,
//                                       const torch::Tensor &dense_to_expert_map,
//                                       const torch::Tensor &expert_offsets,
//                                       const torch::Tensor &buffer_ptrs_dev, int num_recv_tokens,
//                                       int num_experts, int rank, int num_ranks, int num_sms) {
//     EP_HOST_ASSERT(permuted_x.dim() == 2 and permuted_x.is_contiguous());
//     EP_HOST_ASSERT((permuted_x.size(1) * permuted_x.element_size()) % sizeof(int4) == 0);

//     void **buffer_ptrs_gpu = reinterpret_cast<void **>(buffer_ptrs_dev.data_ptr<int64_t>());

//     auto hidden              = static_cast<int>(permuted_x.size(1));
//     auto num_experts_per_rank = num_experts / num_ranks;
//     auto num_topk            = static_cast<int>(dense_to_expert_map.size(1));

//     auto stream     = at::cuda::getCurrentCUDAStream();
//     auto combined_x = torch::zeros({num_recv_tokens, hidden}, permuted_x.options());

//     primus_turbo::cco::ep::intranode::fused_unpermute_combine(
//         buffer_ptrs_gpu, permuted_x.data_ptr(), permuted_weights.data_ptr<float>(),
//         dense_to_expert_map.data_ptr<int>(), expert_offsets.data_ptr<int>(), combined_x.data_ptr(),
//         num_recv_tokens, static_cast<int>(hidden * permuted_x.element_size() / sizeof(int4)),
//         num_topk, num_experts_per_rank, rank, num_ranks, stream, num_sms);

//     return combined_x;
// }

} // namespace primus_turbo::pytorch::cco::intranode
