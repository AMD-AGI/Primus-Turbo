// // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// //
// // See LICENSE for license information.

// #include "primus_turbo/cco/ep.h"
// #include "../extensions.h"
// #include "../type_traits.h"
// #include "kernels/cco/ep/exception.cuh"
// #include "primus_turbo/arch.h"
// #include "primus_turbo/common.h"
// #include "primus_turbo/deep_ep/api.h"
// #include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.hpp>

// #define NUM_MAX_LOCAL_EXPERTS 1024

// namespace primus_turbo::pytorch {

// size_t get_workspace_size() {}

// std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor>
// get_dispatch_layout(const torch::Tensor &topk_idx, const std::string &group_name, int num_experts,
//                     const torch::Tensor &workspace) {
//     auto symm_mem = c10d::symmetric_memory::rendezvous(workspace, group_name);

//     int rank      = symm_mem->get_rank();
//     int num_ranks = symm_mem->get_world_size();

//     PRIMUS_TURBO_CHECK(topk_idx.dim() == 2);
//     PRIMUS_TURBO_CHECK(topk_idx.is_contiguous());
//     PRIMUS_TURBO_CHECK(num_experts > 0);

//     // Allocate all tensors on comm stream if set
//     // NOTES: do not allocate tensors upfront!
//     auto compute_stream = at::cuda::getCurrentCUDAStream();

//     auto num_tokens = static_cast<int>(topk_idx.size(0)),
//          num_topk   = static_cast<int>(topk_idx.size(1));
//     auto num_tokens_per_rank =
//         torch::empty({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
//     auto num_tokens_per_expert    = torch::empty(
//         {num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto is_token_in_rank = torch::empty(
//         {num_tokens, num_ranks}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

//     if (num_ranks > NUM_MAX_NVL_PEERS)
//         num_tokens_per_rdma_rank =
//             torch::empty({num_ranks / NUM_MAX_NVL_PEERS},
//                          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

//     primus_turbo::deep_ep::layout::get_dispatch_layout(
//         topk_idx.data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(),
//         num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>()
//                                              : nullptr,
//         num_tokens_per_expert.data_ptr<int>(), is_token_in_rank.data_ptr<bool>(), num_tokens,
//         num_topk, num_ranks, num_experts, compute_stream);

//     return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank};
// }

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
//            at::Tensor>
// fused_dispatch(const at::Tensor &x, const std::optional<at::Tensor> &x_scales,
//                const std::optional<at::Tensor> &topk_idx,
//                const std::optional<at::Tensor> &topk_weights, torch::Tensor &workspace,
//                const std::string &group_name, int num_experts, int expert_alignment, int num_sms) {
//     auto symm_mem = c10d::symmetric_memory::rendezvous(workspace, group_name);

//     int rank      = symm_mem->get_rank();
//     int num_ranks = symm_mem->get_world_size();

//     // get_dispatch_layout
//     EP_HOST_ASSERT(topk_idx->dim() == 2);
//     EP_HOST_ASSERT(topk_idx->is_contiguous());
//     EP_HOST_ASSERT(num_experts > 0);

//     EP_HOST_ASSERT(topk_idx.has_value());

//     auto stream = at::cuda::getCurrentCUDAStream();

//     auto num_tokens = static_cast<int>(topk_idx->size(0)),
//          num_topk   = static_cast<int>(topk_idx->size(1));
//     auto num_tokens_per_rank =
//         torch::empty({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
//     auto num_tokens_per_expert    = torch::empty(
//         {num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto is_token_in_rank = torch::empty(
//         {num_tokens, num_ranks}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

//     primus_turbo::deep_ep::layout::get_dispatch_layout(
//         topk_idx->data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(),
//         num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>()
//                                              : nullptr,
//         num_tokens_per_expert.data_ptr<int>(), is_token_in_rank.data_ptr<bool>(), num_tokens,
//         num_topk, num_ranks, num_experts, stream);

//     // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for
//     // receiving.
//     EP_HOST_ASSERT(num_sms % 2 == 0);
//     int num_channels = num_sms / 2;

//     // Type checks
//     EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
//     EP_HOST_ASSERT(num_tokens_per_expert.scalar_type() == torch::kInt32);
//     EP_HOST_ASSERT(num_tokens_per_rank.scalar_type() == torch::kInt32);

//     // Shape and contiguous checks
//     EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
//     EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
//     EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
//     EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);

//     EP_HOST_ASSERT(num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous());
//     EP_HOST_ASSERT(num_tokens_per_expert.size(0) % num_ranks == 0);
//     EP_HOST_ASSERT(num_tokens_per_expert.size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
//     EP_HOST_ASSERT(num_tokens_per_rank.dim() == 1 and num_tokens_per_rank.is_contiguous());
//     EP_HOST_ASSERT(num_tokens_per_rank.size(0) == num_ranks);

//     auto hidden            = static_cast<int>(x.size(1));
//     auto num_local_experts = num_experts / num_ranks;

//     // Top-k checks
//     int64_t *topk_idx_ptr     = nullptr;
//     float   *topk_weights_ptr = nullptr;
//     EP_HOST_ASSERT(num_experts > 0);
//     EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
//     EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
//     EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
//     EP_HOST_ASSERT(num_topk == topk_weights->size(1));
//     EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
//     topk_idx_ptr     = topk_idx->data_ptr<int64_t>();
//     topk_weights_ptr = topk_weights->data_ptr<float>();

//     // FP8 scales checks
//     float *x_scales_ptr = nullptr;
//     int    num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
//     // if (x_scales.has_value()) {
//     //     EP_HOST_ASSERT(x.element_size() == 1);
//     //     EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or
//     //                    x_scales->scalar_type() == torch::kInt);
//     //     EP_HOST_ASSERT(x_scales->dim() == 2);
//     //     EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
//     //     num_scales          = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
//     //     x_scales_ptr        = static_cast<float *>(x_scales->data_ptr());
//     //     scale_token_stride  = static_cast<int>(x_scales->stride(0));
//     //     scale_hidden_stride = static_cast<int>(x_scales->stride(1));
//     // }

//     // Barrier or send sizes
//     // To clean: channel start/end offset, head and tail
//     auto num_memset_int     = num_channels * num_ranks * 4;
//     auto rank_prefix_matrix = torch::empty(
//         {num_ranks, num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto channel_prefix_matrix =
//         torch::empty({num_ranks, num_channels},
//                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

//     // Send sizes
//     // Meta information:
//     //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
//     //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
//     // NOTES: no more token dropping in this version

//     auto moe_recv_counter =
//         torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto moe_recv_expert_counter = torch::empty(
//         {num_local_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

//     primus_turbo::cco::ep::intranode::notify_dispatch(
//         num_tokens_per_rank.data_ptr<int>(), moe_recv_counter.data_ptr<int>(), num_ranks,
//         num_tokens_per_expert.data_ptr<int>(), moe_recv_expert_counter.data_ptr<int>(), num_experts,
//         num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
//         rank_prefix_matrix.data_ptr<int>(), num_memset_int, expert_alignment,
//         symm_mem->get_buffer_ptrs_dev(),
//         reinterpret_cast<int **>(symm_mem->get_signal_pad_ptrs_dev()), rank, stream, num_channels);

//     // Synchronize total received tokens and tokens per expert
//     int num_recv_tokens = num_tokens * num_ranks;
//     // Allocate new tensors
//     auto recv_x       = torch::empty({num_recv_tokens, hidden}, x.options());
//     auto recv_src_idx = torch::empty(
//         {num_recv_tokens}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto recv_topk_idx     = std::optional<torch::Tensor>(),
//          recv_topk_weights = std::optional<torch::Tensor>(),
//          recv_x_scales     = std::optional<torch::Tensor>();
//     auto recv_channel_prefix_matrix =
//         torch::empty({num_ranks, num_channels},
//                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto send_head = torch::empty({num_tokens, num_ranks},
//                                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

//     // Assign pointers
//     int64_t *recv_topk_idx_ptr     = nullptr;
//     float   *recv_topk_weights_ptr = nullptr;
//     float   *recv_x_scales_ptr     = nullptr;
//     //if (topk_idx->has_value()) {
//         recv_topk_idx         = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
//         recv_topk_weights     = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
//         recv_topk_idx_ptr     = recv_topk_idx->data_ptr<int64_t>();
//         recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
//     // }
//     if (x_scales.has_value()) {
//         recv_x_scales     = x_scales->dim() == 1
//                                 ? torch::empty({num_recv_tokens}, x_scales->options())
//                                 : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
//         recv_x_scales_ptr = static_cast<float *>(recv_x_scales->data_ptr());
//     }

//     // Dispatch

//     int num_max_tokens = num_tokens * num_ranks;

//     primus_turbo::deep_ep::intranode::dispatch(
//         recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr,
//         recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(),
//         send_head.data_ptr<int>(), x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
//         is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(), num_tokens,
//         num_max_tokens, static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)), num_topk,
//         num_experts, num_scales, scale_token_stride, scale_hidden_stride,
//         symm_mem->get_buffer_ptrs_dev(), rank, num_ranks, stream, num_sms, 512, 512);
//     // Return values
//     return std::make_tuple(rank_prefix_matrix, channel_prefix_matrix, send_head,
//                            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank,
//                            moe_recv_counter, moe_recv_expert_counter);
// }

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
// fused_dispatch_groupedgemm(const at::Tensor &x, const std::optional<at::Tensor> &x_scales,
//                            const std::optional<at::Tensor> &topk_idx,
//                            const std::optional<at::Tensor> &topk_weights, int64_t num_experts,
//                            const at::Tensor &workspace, const std::string &group_name,
//                            const int64_t num_sms) {
//     auto symm_mem = c10d::symmetric_memory::rendezvous(workspace, group_name);

//     int rank      = symm_mem->get_rank();
//     int num_ranks = symm_mem->get_world_size();

//     // get_dispatch_layout
//     EP_HOST_ASSERT(topk_idx->dim() == 2);
//     EP_HOST_ASSERT(topk_idx->is_contiguous());
//     EP_HOST_ASSERT(num_experts > 0);

//     EP_HOST_ASSERT(topk_idx.has_value());

//     auto stream = at::cuda::getCurrentCUDAStream();

//     auto num_tokens = static_cast<int>(topk_idx->size(0)),
//          num_topk   = static_cast<int>(topk_idx->size(1));
//     auto num_tokens_per_rank =
//         torch::empty({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
//     auto num_tokens_per_expert    = torch::empty(
//         {num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto is_token_in_rank = torch::empty(
//         {num_tokens, num_ranks}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

//     primus_turbo::deep_ep::layout::get_dispatch_layout(
//         topk_idx->data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(),
//         num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>()
//                                              : nullptr,
//         num_tokens_per_expert.data_ptr<int>(), is_token_in_rank.data_ptr<bool>(), num_tokens,
//         num_topk, num_ranks, num_experts, stream);

//     // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for
//     // receiving.
//     EP_HOST_ASSERT(num_sms % 2 == 0);
//     int num_channels = num_sms / 2;

//     // Type checks
//     EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
//     EP_HOST_ASSERT(num_tokens_per_expert.scalar_type() == torch::kInt32);
//     EP_HOST_ASSERT(num_tokens_per_rank.scalar_type() == torch::kInt32);

//     // Shape and contiguous checks
//     EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
//     EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
//     EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
//     EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);

//     EP_HOST_ASSERT(num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous());
//     EP_HOST_ASSERT(num_tokens_per_expert.size(0) % num_ranks == 0);
//     EP_HOST_ASSERT(num_tokens_per_expert.size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
//     EP_HOST_ASSERT(num_tokens_per_rank.dim() == 1 and num_tokens_per_rank.is_contiguous());
//     EP_HOST_ASSERT(num_tokens_per_rank.size(0) == num_ranks);

//     auto hidden            = static_cast<int>(x.size(1));
//     auto num_local_experts = num_experts / num_ranks;

//     // Top-k checks
//     int64_t *topk_idx_ptr     = nullptr;
//     float   *topk_weights_ptr = nullptr;
//     EP_HOST_ASSERT(num_experts > 0);
//     EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
//     // EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
//     // EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
//     // EP_HOST_ASSERT(num_topk == topk_weights->size(1));
//     // EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
//     topk_idx_ptr = topk_idx->data_ptr<int64_t>();
//     // topk_weights_ptr = topk_weights->data_ptr<float>();

//     // FP8 scales checks
//     float *x_scales_ptr = nullptr;
//     int    num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
//     // if (x_scales.has_value()) {
//     //     EP_HOST_ASSERT(x.element_size() == 1);
//     //     EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or
//     //                    x_scales->scalar_type() == torch::kInt);
//     //     EP_HOST_ASSERT(x_scales->dim() == 2);
//     //     EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
//     //     num_scales          = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
//     //     x_scales_ptr        = static_cast<float *>(x_scales->data_ptr());
//     //     scale_token_stride  = static_cast<int>(x_scales->stride(0));
//     //     scale_hidden_stride = static_cast<int>(x_scales->stride(1));
//     // }

//     // Barrier or send sizes
//     // To clean: channel start/end offset, head and tail
//     auto num_memset_int     = num_channels * num_ranks * 4;
//     auto rank_prefix_matrix = torch::empty(
//         {num_ranks, num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto channel_prefix_matrix =
//         torch::empty({num_ranks, num_channels},
//                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

//     // Send sizes
//     // Meta information:
//     //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
//     //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
//     // NOTES: no more token dropping in this version
//     // *moe_recv_counter = -1;
//     volatile int *moe_recv_counter        = nullptr;
//     int          *moe_recv_counter_mapped = nullptr;

//     // Host-side expert-level MoE info
//     volatile int *moe_recv_expert_counter        = nullptr;
//     int          *moe_recv_expert_counter_mapped = nullptr;

//     CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
//     CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_counter_mapped),
//                                         const_cast<int *>(moe_recv_counter), 0));
//     *moe_recv_counter = -1;

//     // MoE expert-level counter
//     CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS,
//                               cudaHostAllocMapped));
//     CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_expert_counter_mapped),
//                                         const_cast<int *>(moe_recv_expert_counter), 0));

//     // EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <=
//     int expert_alignment = 0;
//     primus_turbo::deep_ep::intranode::notify_dispatch(
//         num_tokens_per_rank.data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
//         num_tokens_per_expert.data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
//         num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
//         rank_prefix_matrix.data_ptr<int>(), num_memset_int, expert_alignment,
//         symm_mem->get_buffer_ptrs_dev(),
//         reinterpret_cast<int **>(symm_mem->get_signal_pad_ptrs_dev()), rank, stream, num_channels);

//     // Synchronize total received tokens and tokens per expert
//     int num_recv_tokens = num_tokens * num_ranks;
//     // Allocate new tensors
//     // auto recv_x       = torch::empty({num_recv_tokens, hidden}, x.options());
//     auto recv_src_idx = torch::empty(
//         {num_recv_tokens}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto recv_topk_idx     = std::optional<torch::Tensor>(),
//          recv_topk_weights = std::optional<torch::Tensor>(),
//          recv_x_scales     = std::optional<torch::Tensor>();
//     auto recv_channel_prefix_matrix =
//         torch::empty({num_ranks, num_channels},
//                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
//     auto send_head = torch::empty({num_tokens, num_ranks},
//                                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

//     // Assign pointers
//     int64_t *recv_topk_idx_ptr     = nullptr;
//     float   *recv_topk_weights_ptr = nullptr;
//     float   *recv_x_scales_ptr     = nullptr;
//     // if (topk_idx->has_value()) {
//     //     recv_topk_idx         = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
//     //     recv_topk_weights     = torch::empty({num_recv_tokens, num_topk},
//     //     topk_weights->options()); recv_topk_idx_ptr     = recv_topk_idx->data_ptr<topk_idx_t>();
//     //     recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
//     // }
//     // if (x_scales.has_value()) {
//     //     recv_x_scales     = x_scales->dim() == 1
//     //                             ? torch::empty({num_recv_tokens}, x_scales->options())
//     //                             : torch::empty({num_recv_tokens, num_scales},
//     //                             x_scales->options());
//     //     recv_x_scales_ptr = static_cast<float *>(recv_x_scales->data_ptr());
//     // }

//     // Dispatch

//     int num_max_tokens = num_tokens * num_ranks;

//     auto recv_x = symm_mem->get_buffer(rank, {num_max_tokens, hidden}, x.dtype().toScalarType(),
//                                        8 * 8 * 4 / 2);
//     primus_turbo::deep_ep::intranode::dispatch(
//         recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr,
//         recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(),
//         send_head.data_ptr<int>(), x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
//         is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(), num_tokens,
//         num_max_tokens, static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)), num_topk,
//         num_experts, num_scales, scale_token_stride, scale_hidden_stride,
//         symm_mem->get_buffer_ptrs_dev(), rank, num_ranks, stream, num_sms, 256, 256);

//     // Return values
//     return std::make_tuple(rank_prefix_matrix, channel_prefix_matrix, send_head,
//                            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, recv_x);
// }

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
//            at::Tensor>
// fused_dispatch_groupedgemm_meta(const at::Tensor &x, const std::optional<at::Tensor> &x_scales,
//                                 const std::optional<at::Tensor> &topk_idx,
//                                 const std::optional<at::Tensor> &topk_weights, int64_t num_experts,
//                                 const at::Tensor &workspace, const std::string &group_name,
//                                 const int64_t num_sms) {}
// } // namespace primus_turbo::pytorch
