// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#include "primus_turbo/moe_permute.h"
#include "../extensions.h"
#include "primus_turbo/arch.h"

#include <c10/util/Optional.h>

#define SWITCH_EXPERT_MAP_TYPE(case_macro)                                                         \
    switch (expert_map.scalar_type()) {                                                            \
    case at::kBool:                                                                                \
        case_macro(bool);                                                                          \
        break;                                                                                     \
    case at::kInt:                                                                                 \
        case_macro(int);                                                                           \
        break;                                                                                     \
    case at::kLong:                                                                                \
        case_macro(int64_t);                                                                       \
        break;                                                                                     \
    default:                                                                                       \
        PRIMUS_TURBO_CHECK(false, "Invalid expert_map scalar type");                               \
        break;                                                                                     \
    }

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

// Helper: extract a typed data_ptr from an optional tensor, returning nullptr
// when the optional is empty or the tensor is undefined.
template <typename T> static inline T *opt_data_ptr(const c10::optional<at::Tensor> &t) {
    if (!t.has_value() || !t->defined()) {
        return nullptr;
    }
    return reinterpret_cast<T *>(t->data_ptr());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
permute_preprocessing(torch::Tensor expert_map, int64_t num_local_experts, int64_t num_topk,
                      int64_t pad_multiple, int64_t num_permuted_tokens,
                      int64_t probs_topk_stride) {
    PRIMUS_TURBO_CHECK(expert_map.is_cuda(), "routing_map must be CUDA");
    PRIMUS_TURBO_CHECK(expert_map.scalar_type() == at::kBool ||
                           expert_map.scalar_type() == at::kInt ||
                           expert_map.scalar_type() == at::kLong,
                       "expert_map must be bool or int or int64");
    PRIMUS_TURBO_CHECK(expert_map.dim() == 2, "expert_map must be 2D");
    PRIMUS_TURBO_CHECK(expert_map.is_contiguous(), "expert_map must be contiguous");

    if (expert_map.scalar_type() == at::kBool) {
        PRIMUS_TURBO_CHECK(expert_map.size(1) == num_local_experts,
                           "bool expert_map second dimension must equal num_local_experts");
        PRIMUS_TURBO_CHECK(probs_topk_stride == 0,
                           "probs_topk_stride > 0 (topk-aligned probs) requires the topk_idx "
                           "expert_map code path; routing_map mode only supports multihot probs");
    } else {
        PRIMUS_TURBO_CHECK(expert_map.size(1) == num_topk,
                           "index expert_map second dimension must equal num_topk");
        PRIMUS_TURBO_CHECK(probs_topk_stride == 0 || probs_topk_stride == num_topk,
                           "probs_topk_stride must be 0 (multihot probs) or equal to num_topk "
                           "(topk-aligned probs)");
    }

    auto max_num_dispatched_tokens = expert_map.size(0);
    auto device                    = expert_map.device();
    auto int_opts                  = at::TensorOptions().dtype(at::kInt).device(device);
    auto long_opts                 = at::TensorOptions().dtype(at::kLong).device(device);

    auto row_id_map = at::empty(
        {static_cast<int64_t>(max_num_dispatched_tokens + pad_multiple), 2 * num_local_experts + 1},
        int_opts);
    auto tokens_per_expert = at::empty({num_local_experts}, long_opts);
    auto overflow_flag     = at::empty({1}, int_opts);
    // ``at::zeros`` keeps the init CUDA-graph-capturable (no host sync).
    auto num_dispatched_tokens = at::zeros({1}, int_opts);

    auto stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_EXPERT_MAP_TYPE(expert_map_type)                                                  \
    permute_preprocessing_impl<expert_map_type>(                                                   \
        reinterpret_cast<expert_map_type *>(expert_map.data_ptr()), num_topk,                      \
        num_dispatched_tokens.data_ptr<int>(), static_cast<int>(num_local_experts),                \
        static_cast<int>(max_num_dispatched_tokens), static_cast<int>(pad_multiple),               \
        tokens_per_expert.data_ptr<int64_t>(), row_id_map.data_ptr<int>(),                         \
        overflow_flag.data_ptr<int>(), static_cast<int64_t>(num_permuted_tokens),                  \
        static_cast<int>(probs_topk_stride), stream);

    SWITCH_EXPERT_MAP_TYPE(DISPATCH_EXPERT_MAP_TYPE);

    return std::make_tuple(row_id_map, tokens_per_expert, overflow_flag, num_dispatched_tokens);
}

// -----------------------------------------------------------------------------
// permute_launcher
//
// Permute (gather) `tokens` into expert-grouped order using `row_id_map`.
// Optional buffers (`scaling_factor` / `output_scaling_factor`,
// `probs` / `output_probs`) are honoured when supplied; pass an undefined
// optional to skip them.
// -----------------------------------------------------------------------------

void permute(torch::Tensor tokens, torch::Tensor output_tokens,
             c10::optional<torch::Tensor> scaling_factor,
             c10::optional<torch::Tensor> output_scaling_factor, c10::optional<torch::Tensor> probs,
             c10::optional<torch::Tensor> output_probs, torch::Tensor row_id_map,
             torch::Tensor num_dispatched_token_tensor, int64_t pad_multiple,
             int64_t num_local_experts, int64_t hidden_size, int64_t scales_per_token, bool use_fp8,
             bool with_probs, int64_t num_permuted_token, int64_t probs_stride) {
    PRIMUS_TURBO_CHECK(num_permuted_token >= 0, "num_permuted_token must be >= 0");
    if (num_permuted_token == 0) {
        return; // nothing to do
    }
    PRIMUS_TURBO_CHECK(probs_stride >= 0, "probs_stride must be >= 0");

    PRIMUS_TURBO_CHECK(tokens.is_cuda() && output_tokens.is_cuda(),
                       "permute: tokens / output_tokens must be CUDA");
    PRIMUS_TURBO_CHECK(tokens.is_contiguous() && output_tokens.is_contiguous(),
                       "permute: tokens / output_tokens must be contiguous");
    PRIMUS_TURBO_CHECK(tokens.dim() == 2 && output_tokens.dim() == 2,
                       "permute: tokens / output_tokens must be 2D");
    PRIMUS_TURBO_CHECK(tokens.size(1) == hidden_size,
                       "permute: tokens.shape[1] must equal hidden_size");
    PRIMUS_TURBO_CHECK(output_tokens.size(0) == num_permuted_token &&
                           output_tokens.size(1) == hidden_size,
                       "permute: output_tokens shape must be [num_permuted_token, hidden_size]");
    PRIMUS_TURBO_CHECK(row_id_map.is_cuda() && row_id_map.scalar_type() == at::kInt &&
                           row_id_map.is_contiguous(),
                       "permute: row_id_map must be contiguous int32 CUDA tensor");
    PRIMUS_TURBO_CHECK(num_dispatched_token_tensor.is_cuda() &&
                           num_dispatched_token_tensor.scalar_type() == at::kInt &&
                           num_dispatched_token_tensor.is_contiguous() &&
                           num_dispatched_token_tensor.numel() == 1,
                       "permute: num_dispatched_token_tensor must be contiguous int32 CUDA tensor "
                       "with exactly one element");

    if (use_fp8) {
        PRIMUS_TURBO_CHECK(tokens.element_size() == 1 && output_tokens.element_size() == 1,
                           "permute (fp8): tokens / output_tokens must have 1-byte elements");
    } else {
        PRIMUS_TURBO_CHECK(tokens.element_size() == 2 && output_tokens.element_size() == 2,
                           "permute (16-bit): tokens / output_tokens must have 2-byte elements");
    }

    if (scaling_factor.has_value()) {
        PRIMUS_TURBO_CHECK(scaling_factor->is_cuda() && scaling_factor->is_contiguous() &&
                               scaling_factor->scalar_type() == at::kFloat,
                           "permute: scaling_factor must be contiguous float32 CUDA tensor");
    }
    if (output_scaling_factor.has_value()) {
        PRIMUS_TURBO_CHECK(output_scaling_factor->is_cuda() &&
                               output_scaling_factor->is_contiguous() &&
                               output_scaling_factor->scalar_type() == at::kFloat,
                           "permute: output_scaling_factor must be contiguous float32 CUDA tensor");
    }
    if (probs.has_value()) {
        PRIMUS_TURBO_CHECK(probs->is_cuda() && probs->is_contiguous() &&
                               probs->scalar_type() == at::kFloat,
                           "permute: probs must be contiguous float32 CUDA tensor");
        if (with_probs) {
            const int64_t effective_stride = probs_stride > 0 ? probs_stride : num_local_experts;
            PRIMUS_TURBO_CHECK(probs->dim() == 2, "permute: probs must be 2D ([T, probs_stride])");
            PRIMUS_TURBO_CHECK(probs->size(1) == effective_stride,
                               "permute: probs.size(1) must equal probs_stride (or "
                               "num_local_experts when probs_stride == 0)");
        }
    }
    if (output_probs.has_value()) {
        PRIMUS_TURBO_CHECK(output_probs->is_cuda() && output_probs->is_contiguous() &&
                               output_probs->scalar_type() == at::kFloat,
                           "permute: output_probs must be contiguous float32 CUDA tensor");
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    const int num_dispatched_max = static_cast<int>(row_id_map.size(0));

    if (use_fp8) {
        PRIMUS_TURBO_CHECK(hidden_size % 16 == 0,
                           "permute (fp8): hidden_size must be a multiple of 16");
        permute_impl<uint8_t, float, float>(
            reinterpret_cast<const uint8_t *>(tokens.data_ptr()),
            reinterpret_cast<uint8_t *>(output_tokens.data_ptr()),
            opt_data_ptr<const float>(scaling_factor), opt_data_ptr<float>(output_scaling_factor),
            with_probs ? opt_data_ptr<const float>(probs) : nullptr,
            with_probs ? opt_data_ptr<float>(output_probs) : nullptr, row_id_map.data_ptr<int>(),
            num_dispatched_token_tensor.data_ptr<int>(), static_cast<int>(pad_multiple),
            static_cast<int>(num_local_experts), static_cast<int>(hidden_size),
            static_cast<int>(scales_per_token), num_dispatched_max, static_cast<int>(probs_stride),
            stream);
    } else {
        PRIMUS_TURBO_CHECK(hidden_size % 8 == 0,
                           "permute (16-bit): hidden_size must be a multiple of 8");
        permute_impl<uint16_t, float, float>(
            reinterpret_cast<const uint16_t *>(tokens.data_ptr()),
            reinterpret_cast<uint16_t *>(output_tokens.data_ptr()),
            /*scaling_factor=*/nullptr, /*permuted_scaling_factor=*/nullptr,
            with_probs ? opt_data_ptr<const float>(probs) : nullptr,
            with_probs ? opt_data_ptr<float>(output_probs) : nullptr, row_id_map.data_ptr<int>(),
            num_dispatched_token_tensor.data_ptr<int>(), static_cast<int>(pad_multiple),
            static_cast<int>(num_local_experts), static_cast<int>(hidden_size),
            static_cast<int>(scales_per_token), num_dispatched_max, static_cast<int>(probs_stride),
            stream);
    }
}

// -----------------------------------------------------------------------------
// unpermute_launcher
//
// Reduce permuted bf16/fp16 tokens back to per-source rows.
//   permuted_tokens : [num_permuted_tokens, hidden_size]   (bf16/fp16, input)
//   output_tokens   : [num_dispatched_tokens, hidden_size] (bf16/fp16, output)
// -----------------------------------------------------------------------------

void unpermute(torch::Tensor permuted_tokens, torch::Tensor output_tokens,
               c10::optional<torch::Tensor> permuted_probs,
               c10::optional<torch::Tensor> output_probs, torch::Tensor row_id_map,
               torch::Tensor num_dispatched_tokens_tensor, int64_t num_local_experts,
               int64_t hidden_size, bool with_probs, int64_t probs_stride) {
    PRIMUS_TURBO_CHECK(probs_stride >= 0, "probs_stride must be >= 0");
    PRIMUS_TURBO_CHECK(permuted_tokens.is_cuda() && output_tokens.is_cuda(),
                       "unpermute: tensors must be CUDA");
    PRIMUS_TURBO_CHECK(permuted_tokens.scalar_type() == at::kBFloat16 ||
                           permuted_tokens.scalar_type() == at::kHalf,
                       "unpermute: permuted_tokens must be bfloat16 or float16");
    PRIMUS_TURBO_CHECK(output_tokens.scalar_type() == permuted_tokens.scalar_type(),
                       "unpermute: output_tokens dtype must match permuted_tokens");
    PRIMUS_TURBO_CHECK(hidden_size % 8 == 0, "unpermute: hidden_size must be a multiple of 8");
    PRIMUS_TURBO_CHECK(row_id_map.is_cuda() && row_id_map.scalar_type() == at::kInt,
                       "unpermute: row_id_map must be int32 CUDA tensor");
    PRIMUS_TURBO_CHECK(
        num_dispatched_tokens_tensor.is_cuda() &&
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
        const int64_t effective_stride = probs_stride > 0 ? probs_stride : num_local_experts;
        PRIMUS_TURBO_CHECK(output_probs->dim() == 2,
                           "unpermute: output_probs must be 2D ([T, probs_stride])");
        PRIMUS_TURBO_CHECK(output_probs->size(1) == effective_stride,
                           "unpermute: output_probs.size(1) must equal probs_stride "
                           "(or num_local_experts when probs_stride == 0)");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    // output_tokens has exactly num_dispatched rows pre-allocated by the
    // caller; that's the upper bound on tokens the kernel may reduce.
    const int num_dispatched_max = static_cast<int>(output_tokens.size(0));

    if (permuted_tokens.scalar_type() == at::kBFloat16) {
        unpermute_impl<bfloat16, float>(
            reinterpret_cast<const bfloat16 *>(permuted_tokens.data_ptr()),
            reinterpret_cast<bfloat16 *>(output_tokens.data_ptr()),
            with_probs ? opt_data_ptr<const float>(permuted_probs) : nullptr,
            with_probs ? opt_data_ptr<float>(output_probs) : nullptr, row_id_map.data_ptr<int>(),
            num_dispatched_tokens_tensor.data_ptr<int>(), static_cast<int>(num_local_experts),
            static_cast<int>(hidden_size), num_dispatched_max, static_cast<int>(probs_stride),
            stream);
    } else {
        unpermute_impl<float16, float>(
            reinterpret_cast<const float16 *>(permuted_tokens.data_ptr()),
            reinterpret_cast<float16 *>(output_tokens.data_ptr()),
            with_probs ? opt_data_ptr<const float>(permuted_probs) : nullptr,
            with_probs ? opt_data_ptr<float>(output_probs) : nullptr, row_id_map.data_ptr<int>(),
            num_dispatched_tokens_tensor.data_ptr<int>(), static_cast<int>(num_local_experts),
            static_cast<int>(hidden_size), num_dispatched_max, static_cast<int>(probs_stride),
            stream);
    }
}

} // namespace primus_turbo::pytorch
#undef SWITCH_EXPERT_MAP_TYPE
