// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/cco/ep.h"
#include "../extensions.h"
#include "../type_traits.h"
#include "kernels/cco/ep/exception.cuh"
#include "primus_turbo/arch.h"

namespace primus_turbo::pytorch {

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor>
get_dispatch_layout(const torch::Tensor &topk_idx, int num_experts, int num_ranks,
                    int num_rdma_ranks) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto stream = at::cuda::getCurrentCUDAStream();

    auto num_tokens = static_cast<int>(topk_idx.size(0)),
         num_topk   = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank =
        torch::empty({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert    = torch::empty(
        {num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty(
        {num_tokens, num_ranks}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // TODO(zhenhuang12): check if internode is available
    if (num_rdma_ranks > 1)
        num_tokens_per_rdma_rank = torch::empty(
            {num_rdma_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    primus_turbo::deep_ep::layout::get_dispatch_layout(
        topk_idx.data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(),
        num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>()
                                             : nullptr,
        num_tokens_per_expert.data_ptr<int>(), is_token_in_rank.data_ptr<bool>(), num_tokens,
        num_topk, num_ranks, num_experts, stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
intranode_dispatch_with_permute(
    const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales,
    const torch::Tensor &row_id_map, const std::optional<torch::Tensor> &topk_idx,
    const std::optional<torch::Tensor> &topk_weights,
    const std::optional<torch::Tensor> &num_tokens_per_rank, const torch::Tensor &is_token_in_rank,
    const std::optional<torch::Tensor> &num_tokens_per_expert, int cached_num_recv_tokens,
    const std::optional<torch::Tensor> &cached_rank_prefix_matrix,
    const std::optional<torch::Tensor> &cached_channel_prefix_matrix,
    const torch::Tensor &buffer_ptrs_dev, const torch::Tensor &barrier_signal_ptrs_dev,
    const torch::Tensor &moe_recv_counter, const torch::Tensor &moe_recv_expert_counter,
    int expert_alignment, int num_worst_tokens, int num_permuted_tokens, int rank, int num_ranks,
    int num_sms, int num_max_send_tokens) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    assert(num_worst_tokens > 0);

    void **buffer_ptrs_gpu = reinterpret_cast<void **>(buffer_ptrs_dev.data_ptr<int64_t>());
    int  **barrier_signal_ptrs_gpu =
        reinterpret_cast<int **>(barrier_signal_ptrs_dev.data_ptr<int64_t>());

    EP_HOST_ASSERT(num_sms % 2 == 0);
    int num_channels = num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and
                       cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and
                       cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and
                       cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and
                       cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and
                       num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts       = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
         num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int      num_topk         = 0;
    int64_t *topk_idx_ptr     = nullptr;
    float   *topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr     = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float *x_scales_ptr = nullptr;
    int    num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or
                       x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales          = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr        = static_cast<float *>(x_scales->data_ptr());
        scale_token_stride  = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto stream = at::cuda::getCurrentCUDAStream();

    // Create handles (only return for non-cached mode)
    int  num_recv_tokens       = -1;
    auto rank_prefix_matrix    = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail.
    // Also covers the flag buffer used by dispatch_with_permute for
    // sender→permuter synchronisation.
    int num_flag_ints  = (num_worst_tokens + num_max_send_tokens - 1) / num_max_send_tokens;
    int num_memset_int = std::max(num_channels * num_ranks * 4, num_flag_ints);
    if (cached_mode) {
        num_recv_tokens       = cached_num_recv_tokens;
        rank_prefix_matrix    = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        primus_turbo::deep_ep::intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu,
            barrier_signal_ptrs_gpu, rank, num_ranks, stream);
    } else {
        rank_prefix_matrix =
            torch::empty({num_ranks, num_ranks},
                         torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        channel_prefix_matrix =
            torch::empty({num_ranks, num_channels},
                         torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        primus_turbo::deep_ep::intranode::notify_dispatch(
            num_tokens_per_rank->data_ptr<int>(), moe_recv_counter.data_ptr<int>(), num_ranks,
            num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter.data_ptr<int>(),
            num_experts, num_tokens, is_token_in_rank.data_ptr<bool>(),
            channel_prefix_matrix.data_ptr<int>(), rank_prefix_matrix.data_ptr<int>(),
            num_memset_int, expert_alignment, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank,
            stream, num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            // TODO(zhenhuang12): remove this branch
            assert(false);
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            // while (true) {
            //     // Read total count
            //     num_recv_tokens = static_cast<int>(*moe_recv_counter);

            //     // Read per-expert count
            //     bool ready = (num_recv_tokens >= 0);
            //     for (int i = 0; i < num_local_experts and ready; ++i)
            //         ready &= moe_recv_expert_counter[i] >= 0;

            //     if (ready)
            //         break;

            //     // Timeout check
            //     if (std::chrono::duration_cast<std::chrono::seconds>(
            //             std::chrono::high_resolution_clock::now() - start_time)
            //             .count() > NUM_CPU_TIMEOUT_SECS)
            //         throw std::runtime_error("DeepEP error: CPU recv timeout");
            // }
            // num_recv_tokens_per_expert_list = std::vector<int>(
            //     moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x       = torch::empty({num_permuted_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty(
        {num_recv_tokens}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx     = std::optional<torch::Tensor>(),
         recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales     = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix =
        torch::empty({num_ranks, num_channels},
                     torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks},
                                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    // Assign pointers
    int64_t *recv_topk_idx_ptr     = nullptr;
    float   *recv_topk_weights_ptr = nullptr;
    float   *recv_x_scales_ptr     = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx         = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights     = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr     = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales     = x_scales->dim() == 1
                                ? torch::empty({num_recv_tokens}, x_scales->options())
                                : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float *>(recv_x_scales->data_ptr());
    }

    primus_turbo::cco::ep::intranode::dispatch_with_permute(
        buffer_ptrs_gpu, x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
        is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
        row_id_map.data_ptr<int>(), recv_x.data_ptr(), num_tokens,
        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)), num_topk, num_experts,
        num_scales, scale_token_stride, scale_hidden_stride, rank, num_ranks, stream, num_sms,
        num_worst_tokens, num_max_send_tokens);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            moe_recv_counter,
            moe_recv_expert_counter,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head};
}

} // namespace primus_turbo::pytorch