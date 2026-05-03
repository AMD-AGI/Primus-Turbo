// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// PyTorch wrappers for the MoE permute / unpermute kernels.
// The device code lives under `csrc/kernels/permute/`; this file only marshals
// torch tensors into the plain-pointer launchers exposed by
// `primus_turbo/permute.h`.

#include "primus_turbo/permute.h"
#include "../extensions.h"

#include <c10/util/Optional.h>

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

namespace {

// Pick a default grid size when the caller doesn't supply one. We fall back to
// "one block per CU" so the kernel always has work for every multiprocessor.
inline int default_grid_size(int64_t num_of_blocks) {
    if (num_of_blocks > 0) return static_cast<int>(num_of_blocks);
    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    int num_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, device_id));
    return (std::max)(num_cu, 1);
}

template <typename T> inline T *opt_data_ptr(const c10::optional<torch::Tensor> &opt) {
    if (!opt.has_value() || !opt->defined()) return nullptr;
    return reinterpret_cast<T *>(opt->data_ptr());
}

} // namespace

// -----------------------------------------------------------------------------
// permute_preprocessing
//
// Builds row_id_map / tokens_per_expert / overflow_flag from the routing map.
// Internally launches a cooperative kernel sized to fit the device.
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
permute_preprocessing(torch::Tensor routing_map,
                      torch::Tensor num_dispatched_token_tensor,
                      // Used to size row_id_map without synchronising on the
                      // pinned-memory copy of num_dispatched_tokens.
                      int64_t max_num_dispatched_tokens, int64_t num_of_local_experts,
                      int64_t pad_multiple, int64_t num_permuted_tokens) {
    PRIMUS_TURBO_CHECK(routing_map.is_cuda(), "routing_map must be CUDA");
    PRIMUS_TURBO_CHECK(routing_map.scalar_type() == at::kBool, "routing_map must be bool");
    PRIMUS_TURBO_CHECK(num_dispatched_token_tensor.is_cuda(),
                       "num_dispatched_token_tensor must be CUDA");
    PRIMUS_TURBO_CHECK(num_dispatched_token_tensor.scalar_type() == at::kInt,
                       "num_dispatched_token_tensor must be int32");

    constexpr int block_size = PermutePreprocessConfig::kBlockSize;
    PRIMUS_TURBO_CHECK(num_of_local_experts > 0 && num_of_local_experts <= block_size,
                       "num_of_local_experts must be in (0, block_size]");

    auto device     = routing_map.device();
    auto int_opts   = at::TensorOptions().dtype(at::kInt).device(device);

    auto row_id_map = at::empty(
        {static_cast<int64_t>(max_num_dispatched_tokens + pad_multiple), num_of_local_experts},
        int_opts);
    auto tokens_per_expert = at::empty({num_of_local_experts}, int_opts);
    auto overflow_flag     = at::empty({1}, int_opts);

    int rows_workspace_1 =
        static_cast<int>((max_num_dispatched_tokens + block_size - 1) / block_size);
    int rows_workspace_2 = (rows_workspace_1 + block_size - 1) / block_size;

    auto workspace1 = at::empty({rows_workspace_1, num_of_local_experts}, int_opts);
    auto workspace2 = at::empty({rows_workspace_2, num_of_local_experts}, int_opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    permute_preprocessing_launch(
        reinterpret_cast<bool *>(routing_map.data_ptr()),
        num_dispatched_token_tensor.data_ptr<int>(), static_cast<int>(num_of_local_experts),
        workspace1.data_ptr<int>(), rows_workspace_1, workspace2.data_ptr<int>(), rows_workspace_2,
        static_cast<int>(pad_multiple), tokens_per_expert.data_ptr<int>(),
        row_id_map.data_ptr<int>(), overflow_flag.data_ptr<int>(),
        static_cast<int64_t>(num_permuted_tokens), stream);

    return std::make_tuple(row_id_map, tokens_per_expert, overflow_flag);
}

// -----------------------------------------------------------------------------
// permute_launcher
//
// Permute (gather) `tokens` into expert-grouped order using `row_id_map`.
// Optional buffers (`scaling_factor` / `output_scaling_factor`,
// `probs` / `output_probs`) are honoured when supplied; pass an undefined
// optional to skip them.
// -----------------------------------------------------------------------------

void permute_launcher(torch::Tensor tokens, torch::Tensor output_tokens,
                      c10::optional<torch::Tensor> scaling_factor,
                      c10::optional<torch::Tensor> output_scaling_factor,
                      c10::optional<torch::Tensor> probs,
                      c10::optional<torch::Tensor> output_probs,
                      torch::Tensor row_id_map, torch::Tensor num_dispatched_token_tensor,
                      int64_t pad_multiple, int64_t num_of_local_experts, int64_t hidden_size,
                      int64_t scales_per_token, int64_t local_rank, int64_t num_ranks_per_node,
                      bool use_fp8, bool with_probs, int64_t num_permuted_token,
                      int64_t num_of_blocks_permute) {
    PRIMUS_TURBO_CHECK(num_permuted_token >= 0, "num_permuted_token must be >= 0");
    if (num_permuted_token == 0) {
        return; // nothing to do
    }

    PRIMUS_TURBO_CHECK(tokens.is_cuda() && output_tokens.is_cuda(),
                       "permute_launcher: tokens / output_tokens must be CUDA");
    PRIMUS_TURBO_CHECK(row_id_map.is_cuda() && row_id_map.scalar_type() == at::kInt,
                       "permute_launcher: row_id_map must be int32 CUDA tensor");
    PRIMUS_TURBO_CHECK(num_dispatched_token_tensor.is_cuda() &&
                           num_dispatched_token_tensor.scalar_type() == at::kInt,
                       "permute_launcher: num_dispatched_token_tensor must be int32 CUDA tensor");

    auto stream = at::cuda::getCurrentCUDAStream();
    int  grid   = default_grid_size(num_of_blocks_permute);

    using ScalarType = float;
    using ProbType   = float;

    if (use_fp8) {
        PRIMUS_TURBO_CHECK(hidden_size % 16 == 0,
                           "permute (fp8): hidden_size must be a multiple of 16");
        permute_impl<uint8_t, ProbType, ScalarType>(
            reinterpret_cast<const uint8_t *>(tokens.data_ptr()),
            reinterpret_cast<uint8_t *>(output_tokens.data_ptr()),
            opt_data_ptr<const ScalarType>(scaling_factor),
            opt_data_ptr<ScalarType>(output_scaling_factor),
            with_probs ? opt_data_ptr<const ProbType>(probs) : nullptr,
            with_probs ? opt_data_ptr<ProbType>(output_probs) : nullptr,
            row_id_map.data_ptr<int>(), num_dispatched_token_tensor.data_ptr<int>(),
            static_cast<int>(pad_multiple), static_cast<int>(num_of_local_experts),
            static_cast<int>(hidden_size), static_cast<int>(scales_per_token),
            static_cast<int>(local_rank), static_cast<int>(num_ranks_per_node), grid, stream);
    } else {
        PRIMUS_TURBO_CHECK(hidden_size % 8 == 0,
                           "permute (16-bit): hidden_size must be a multiple of 8");
        permute_impl<uint16_t, ProbType, ScalarType>(
            reinterpret_cast<const uint16_t *>(tokens.data_ptr()),
            reinterpret_cast<uint16_t *>(output_tokens.data_ptr()),
            /*scaling_factor=*/nullptr, /*permuted_scaling_factor=*/nullptr,
            with_probs ? opt_data_ptr<const ProbType>(probs) : nullptr,
            with_probs ? opt_data_ptr<ProbType>(output_probs) : nullptr,
            row_id_map.data_ptr<int>(), num_dispatched_token_tensor.data_ptr<int>(),
            static_cast<int>(pad_multiple), static_cast<int>(num_of_local_experts),
            static_cast<int>(hidden_size), static_cast<int>(scales_per_token),
            static_cast<int>(local_rank), static_cast<int>(num_ranks_per_node), grid, stream);
    }
}

// -----------------------------------------------------------------------------
// unpermute_launcher
//
// Reduce permuted bf16 tokens back to per-source rows.
//   permuted_tokens : [num_permuted_tokens, hidden_size]   (bfloat16, input)
//   output_tokens   : [num_dispatched_tokens, hidden_size] (bfloat16, output)
// -----------------------------------------------------------------------------

void unpermute_launcher(torch::Tensor permuted_tokens, torch::Tensor output_tokens,
                        c10::optional<torch::Tensor> permuted_probs,
                        c10::optional<torch::Tensor> output_probs, torch::Tensor row_id_map,
                        torch::Tensor num_dispatched_tokens_tensor, int64_t num_of_local_experts,
                        int64_t hidden_size, int64_t local_rank, int64_t num_ranks_per_node,
                        bool with_probs, int64_t num_of_blocks_unpermute) {
    PRIMUS_TURBO_CHECK(permuted_tokens.is_cuda() && output_tokens.is_cuda(),
                       "unpermute_launcher: tensors must be CUDA");
    PRIMUS_TURBO_CHECK(permuted_tokens.scalar_type() == at::kBFloat16,
                       "unpermute_launcher: permuted_tokens must be bfloat16");
    PRIMUS_TURBO_CHECK(output_tokens.scalar_type() == at::kBFloat16,
                       "unpermute_launcher: output_tokens must be bfloat16");
    PRIMUS_TURBO_CHECK(hidden_size % 8 == 0,
                       "unpermute_launcher: hidden_size must be a multiple of 8");
    PRIMUS_TURBO_CHECK(row_id_map.is_cuda() && row_id_map.scalar_type() == at::kInt,
                       "unpermute_launcher: row_id_map must be int32 CUDA tensor");
    PRIMUS_TURBO_CHECK(num_dispatched_tokens_tensor.is_cuda() &&
                           num_dispatched_tokens_tensor.scalar_type() == at::kInt,
                       "unpermute_launcher: num_dispatched_tokens_tensor must be int32 CUDA tensor");
    if (with_probs) {
        PRIMUS_TURBO_CHECK(permuted_probs.has_value() && permuted_probs->defined(),
                           "unpermute_launcher: with_probs but permuted_probs is empty");
        PRIMUS_TURBO_CHECK(permuted_probs->scalar_type() == at::kFloat,
                           "unpermute_launcher: permuted_probs must be float32");
        PRIMUS_TURBO_CHECK(output_probs.has_value() && output_probs->defined(),
                           "unpermute_launcher: with_probs but output_probs is empty");
        PRIMUS_TURBO_CHECK(output_probs->scalar_type() == at::kFloat,
                           "unpermute_launcher: output_probs must be float32");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    int  grid   = default_grid_size(num_of_blocks_unpermute);

    using ProbType = float;

    unpermute_impl<bfloat16, ProbType>(
        reinterpret_cast<const bfloat16 *>(permuted_tokens.data_ptr()),
        reinterpret_cast<bfloat16 *>(output_tokens.data_ptr()),
        with_probs ? opt_data_ptr<const ProbType>(permuted_probs) : nullptr,
        with_probs ? opt_data_ptr<ProbType>(output_probs) : nullptr,
        row_id_map.data_ptr<int>(), num_dispatched_tokens_tensor.data_ptr<int>(),
        static_cast<int>(num_of_local_experts), static_cast<int>(hidden_size),
        static_cast<int>(local_rank), static_cast<int>(num_ranks_per_node), grid, stream);
}

} // namespace primus_turbo::pytorch
