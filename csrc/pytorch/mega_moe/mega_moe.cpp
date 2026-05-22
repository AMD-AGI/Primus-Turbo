// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "primus_turbo/mega_moe.h"

#include "../extensions.h"

namespace primus_turbo::pytorch {

using primus_turbo::mega_moe::get_mega_moe_config;
using primus_turbo::mega_moe::kScaleGroupK;
using primus_turbo::mega_moe::launch_fp8_fp4_mega_moe;
using primus_turbo::mega_moe::MegaMoEArgs;
using primus_turbo::mega_moe::MegaMoEBufferLayout;
using primus_turbo::mega_moe::MegaMoEConfig;
namespace mm = primus_turbo::mega_moe;

// Mirrors DeepGEMM's ``deep_gemm::mega::get_token_alignment_for_mega_moe``.
int64_t get_token_alignment_for_mega_moe() {
    return primus_turbo::mega_moe::kTokenAlignment;
}

// Returns the symmetric-buffer byte size plus the offsets of each
// logical region, packed as a 1-D int64 tensor laid out as:
//   [ total_bytes,
//     num_max_pool_tokens, num_padded_sf_pool_tokens,
//     workspace_offset,
//     input_x_offset, input_x_sf_offset,
//     input_topk_idx_offset, input_topk_weights_offset,
//     l1_pool_x_offset, l1_pool_x_sf_offset, l1_pool_weights_offset,
//     l2_pool_x_offset, l2_pool_x_sf_offset,
//     combine_buffer_offset ]
//
// The Python side then slices these offsets into actual tensor views
// over the symmetric memory buffer.  Mirrors DeepGEMM's
// ``deep_gemm::mega::get_symm_buffer_size_for_mega_moe``.  DG returns
// ``(num_bytes, slice_input_buffers_callback)``; we return the raw
// offsets so the Python layer (registered through ``torch.library``)
// can slice them without a second C++ round-trip.
at::Tensor get_symm_buffer_size_for_mega_moe(const int64_t num_ranks, const int64_t num_experts,
                                             const int64_t num_max_tokens_per_rank,
                                             const int64_t num_topk, const int64_t hidden,
                                             const int64_t intermediate_hidden,
                                             const bool    use_fp8_dispatch) {
    const auto layout = mm::get_symm_buffer_size_for_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden), use_fp8_dispatch);

    at::Tensor out = at::empty({14}, at::dtype(at::kLong).device(at::kCPU));
    auto       p   = out.data_ptr<int64_t>();
    p[0]           = layout.total_bytes;
    p[1]           = layout.num_max_pool_tokens;
    p[2]           = layout.num_padded_sf_pool_tokens;
    p[3]           = layout.workspace_offset;
    p[4]           = layout.input_x_offset;
    p[5]           = layout.input_x_sf_offset;
    p[6]           = layout.input_topk_idx_offset;
    p[7]           = layout.input_topk_weights_offset;
    p[8]           = layout.l1_pool_x_offset;
    p[9]           = layout.l1_pool_x_sf_offset;
    p[10]          = layout.l1_pool_weights_offset;
    p[11]          = layout.l2_pool_x_offset;
    p[12]          = layout.l2_pool_x_sf_offset;
    p[13]          = layout.combine_buffer_offset;
    return out;
}

// Fused mega-MoE FP8 activations × FP4 weights entry.  Currently a thin
// host-side validation shell that forwards to ``launch_fp8_fp4_mega_moe``
// (which is itself a no-op stub for unsupported AOT shapes).  All input
// tensors are expected to live in symmetric memory or device memory.
// Mirrors DeepGEMM's ``deep_gemm::mega::fp8_fp4_mega_moe``.
void fp8_fp4_mega_moe(at::Tensor y, at::Tensor l1_weights, at::Tensor l1_weights_sf,
                      at::Tensor l2_weights, at::Tensor l2_weights_sf,
                      c10::optional<at::Tensor> cumulative_local_expert_recv_stats,
                      at::Tensor sym_buffer, const std::vector<int64_t> &sym_buffer_ptrs,
                      const int64_t rank_idx, const int64_t num_max_tokens_per_rank,
                      const int64_t num_experts, const int64_t num_topk, const int64_t num_tokens,
                      const int64_t hidden, const int64_t intermediate_hidden,
                      const std::vector<int64_t> &recipe, const std::string &activation,
                      const double activation_clamp, const bool fast_math) {
    TORCH_CHECK(activation == "swiglu", "mega_moe: only swiglu activation is currently supported");
    TORCH_CHECK(recipe.size() == 3, "mega_moe: recipe must be a length-3 tuple (M, N, K)");
    TORCH_CHECK(recipe[0] == 1 && recipe[1] == 1 && recipe[2] == kScaleGroupK,
                "mega_moe: only recipe=(1, 1, ", kScaleGroupK, ") is currently supported, got (",
                recipe[0], ", ", recipe[1], ", ", recipe[2], ")");
    TORCH_CHECK(activation_clamp >= 0, "mega_moe: activation_clamp must be non-negative, got ",
                activation_clamp);
    TORCH_CHECK(y.is_cuda(), "mega_moe: output tensor must live on GPU");
    TORCH_CHECK(sym_buffer.is_cuda(), "mega_moe: symmetric buffer must live on GPU");
    TORCH_CHECK(!sym_buffer_ptrs.empty(), "mega_moe: sym_buffer_ptrs must be non-empty");
    TORCH_CHECK(num_max_tokens_per_rank % primus_turbo::mega_moe::kTokenAlignment == 0,
                "mega_moe: num_max_tokens_per_rank (", num_max_tokens_per_rank,
                ") must be a multiple of kTokenAlignment (",
                primus_turbo::mega_moe::kTokenAlignment, ")");

    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    TORCH_CHECK(num_experts % num_ranks == 0,
                "mega_moe: num_experts must be divisible by num_ranks");
    const int num_experts_per_rank = static_cast<int>(num_experts) / num_ranks;

    // Shape validation: mirror DG's ``check_grouped_ab_fp8_fp4`` semantics.
    TORCH_CHECK(l1_weights.dim() == 3 && l2_weights.dim() == 3,
                "mega_moe: l1/l2 weights must be 3-D [num_experts_per_rank, N, K_packed]");
    TORCH_CHECK(l1_weights.size(0) == num_experts_per_rank &&
                    l2_weights.size(0) == num_experts_per_rank,
                "mega_moe: l1/l2 weights group dim must equal num_experts_per_rank (",
                num_experts_per_rank, ")");
    TORCH_CHECK(l1_weights.size(1) == 2 * intermediate_hidden,
                "mega_moe: l1_weights N must equal 2*intermediate_hidden (",
                2 * intermediate_hidden, "), got ", l1_weights.size(1));
    TORCH_CHECK(l2_weights.size(1) == hidden, "mega_moe: l2_weights N must equal hidden (", hidden,
                "), got ", l2_weights.size(1));
    TORCH_CHECK(l1_weights.is_contiguous() && l2_weights.is_contiguous(),
                "mega_moe: l1/l2 weights must be contiguous");
    TORCH_CHECK(y.dim() == 2 && y.size(1) == hidden, "mega_moe: y must be 2D with hidden dim ",
                hidden, ", got shape ", y.sizes());
    TORCH_CHECK(num_tokens <= num_max_tokens_per_rank, "mega_moe: num_tokens (", num_tokens,
                ") exceeds num_max_tokens_per_rank (", num_max_tokens_per_rank, ")");

    if (cumulative_local_expert_recv_stats.has_value()) {
        const auto &stats = cumulative_local_expert_recv_stats.value();
        TORCH_CHECK(stats.scalar_type() == at::kInt,
                    "mega_moe: cumulative_local_expert_recv_stats must be int32");
        TORCH_CHECK(stats.numel() == num_experts_per_rank,
                    "mega_moe: cumulative_local_expert_recv_stats must have ", num_experts_per_rank,
                    " elements");
        TORCH_CHECK(stats.is_contiguous(),
                    "mega_moe: cumulative_local_expert_recv_stats must be contiguous");
    }

    const auto layout = mm::get_symm_buffer_size_for_mega_moe(
        num_ranks, static_cast<int>(num_experts), static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk), static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        /*use_fp8_dispatch=*/true);
    TORCH_CHECK(sym_buffer.nbytes() >= static_cast<size_t>(layout.total_bytes),
                "mega_moe: symmetric buffer too small for requested config");

    auto config = get_mega_moe_config(
        num_ranks, static_cast<int>(num_experts), num_experts_per_rank,
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_tokens),
        static_cast<int>(num_topk), static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        layout.num_padded_sf_pool_tokens);
    config.num_max_pool_tokens = layout.num_max_pool_tokens;

    MegaMoEArgs args;
    args.y_ptr             = y.data_ptr();
    args.l1_weights_ptr    = l1_weights.data_ptr();
    args.l1_weights_sf_ptr = l1_weights_sf.data_ptr();
    args.l2_weights_ptr    = l2_weights.data_ptr();
    args.l2_weights_sf_ptr = l2_weights_sf.data_ptr();
    args.cumulative_local_expert_recv_stats =
        cumulative_local_expert_recv_stats.has_value()
            ? cumulative_local_expert_recv_stats->data_ptr<int>()
            : nullptr;
    args.sym_buffer_ptrs         = sym_buffer_ptrs.data();
    args.num_ranks               = num_ranks;
    args.rank_idx                = static_cast<int>(rank_idx);
    args.layout                  = layout;
    args.num_tokens              = static_cast<int>(num_tokens);
    args.num_max_tokens_per_rank = static_cast<int>(num_max_tokens_per_rank);
    args.num_experts             = static_cast<int>(num_experts);
    args.num_experts_per_rank    = num_experts_per_rank;
    args.num_topk                = static_cast<int>(num_topk);
    args.hidden                  = static_cast<int>(hidden);
    args.intermediate_hidden     = static_cast<int>(intermediate_hidden);
    args.recipe_m                = static_cast<int>(recipe[0]);
    args.recipe_n                = static_cast<int>(recipe[1]);
    args.recipe_k                = static_cast<int>(recipe[2]);
    args.activation_clamp        = static_cast<float>(activation_clamp);
    args.fast_math               = fast_math;
    args.config                  = config;
    args.stream                  = at::cuda::getCurrentCUDAStream();

    launch_fp8_fp4_mega_moe(args);
}

} // namespace primus_turbo::pytorch
