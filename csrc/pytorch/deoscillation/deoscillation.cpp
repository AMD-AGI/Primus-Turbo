/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include "primus_turbo/deoscillation.h"
#include "pytorch/extensions.h"
#include <ATen/MemoryOverlap.h>
#include <limits>

namespace primus_turbo::pytorch {

namespace {

void check_cuda_contiguous(const at::Tensor &tensor, const char *name) {
    PRIMUS_TURBO_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const at::Tensor &reference, const at::Tensor &tensor, const char *name) {
    PRIMUS_TURBO_CHECK(tensor.device() == reference.device(), name,
                       " must be on the same device as the other deosc tensors");
}

void check_same_numel(const at::Tensor &reference, const at::Tensor &tensor, const char *name) {
    PRIMUS_TURBO_CHECK(tensor.numel() == reference.numel(), name,
                       " must have the same number of elements as the other deosc tensors");
}

} // namespace

void weight_deosc_qdq(at::Tensor master, at::Tensor previous, at::Tensor previous_qdq,
                      at::Tensor dist, at::Tensor dist_qdq, int64_t rows, int64_t cols,
                      int64_t start, int64_t scale_rounding_mode, bool seed, bool close,
                      double ratio_threshold, double eps, c10::optional<at::Tensor> reset_count) {
    const std::vector<at::Tensor> tensors = {master, previous, previous_qdq, dist, dist_qdq};
    for (const auto &tensor : tensors) {
        check_cuda_contiguous(tensor, "deosc_qdq tensor");
        check_same_device(master, tensor, "deosc_qdq tensor");
        check_same_numel(master, tensor, "deosc_qdq tensor");
    }
    for (size_t i = 0; i < tensors.size(); ++i)
        for (size_t j = i + 1; j < tensors.size(); ++j)
            at::assert_no_overlap(tensors[i], tensors[j]);
    PRIMUS_TURBO_CHECK(master.scalar_type() == at::kFloat && dist.scalar_type() == at::kFloat &&
                           dist_qdq.scalar_type() == at::kFloat,
                       "master and distances must be float32");
    PRIMUS_TURBO_CHECK(previous.scalar_type() == at::kBFloat16 &&
                           previous_qdq.scalar_type() == at::kBFloat16,
                       "snapshots must be bfloat16");
    PRIMUS_TURBO_CHECK(rows > 0 && cols > 0 && rows <= std::numeric_limits<int64_t>::max() / cols,
                       "rows and cols must be positive, without int64 overflow");
    PRIMUS_TURBO_CHECK(start >= 0 && start <= std::numeric_limits<int64_t>::max() - master.numel(),
                       "invalid shard start or extent");
    PRIMUS_TURBO_CHECK(scale_rounding_mode >= 0 && scale_rounding_mode <= 2,
                       "scale_rounding_mode must be 0, 1, or 2");
    PRIMUS_TURBO_CHECK(!(seed && close), "cannot seed and close in the same call");
    PRIMUS_TURBO_CHECK(ratio_threshold >= 0.0 && eps > 0.0,
                       "ratio_threshold must be non-negative and eps positive");
    if (reset_count.has_value()) {
        check_cuda_contiguous(*reset_count, "reset_count");
        check_same_device(master, *reset_count, "reset_count");
        PRIMUS_TURBO_CHECK(reset_count->scalar_type() == at::kLong && reset_count->numel() == 1,
                           "reset_count must be an int64 scalar");
        for (const auto &tensor : tensors)
            at::assert_no_overlap(tensor, *reset_count);
    }
    at::cuda::CUDAGuard guard(master.device());
    const auto         *properties = at::cuda::getDeviceProperties(master.get_device());
    PRIMUS_TURBO_CHECK(std::string(properties->gcnArchName).find("gfx950") == 0,
                       "direct deosc QDQ currently requires gfx950");
    primus_turbo::weight_deosc_qdq(
        master.data_ptr<float>(), reinterpret_cast<dtype::bfloat16 *>(previous.data_ptr()),
        reinterpret_cast<dtype::bfloat16 *>(previous_qdq.data_ptr()), dist.data_ptr<float>(),
        dist_qdq.data_ptr<float>(),
        reset_count.has_value() ? reset_count->data_ptr<int64_t>() : nullptr, master.numel(), rows,
        cols, start, static_cast<int>(scale_rounding_mode), seed, close,
        static_cast<float>(ratio_threshold), static_cast<float>(eps),
        at::cuda::getCurrentCUDAStream());
}

void weight_deosc_update(const at::Tensor current, const at::Tensor current_qdq,
                         const at::Tensor previous, const at::Tensor previous_qdq, at::Tensor dist,
                         at::Tensor dist_qdq) {
    for (const auto &entry : std::initializer_list<std::pair<const at::Tensor *, const char *>>{
             {&current, "current"},
             {&current_qdq, "current_qdq"},
             {&previous, "previous"},
             {&previous_qdq, "previous_qdq"},
             {&dist, "dist"},
             {&dist_qdq, "dist_qdq"}}) {
        check_cuda_contiguous(*entry.first, entry.second);
        check_same_device(current, *entry.first, entry.second);
        check_same_numel(current, *entry.first, entry.second);
    }
    PRIMUS_TURBO_CHECK(current.scalar_type() == at::kBFloat16, "current must be bfloat16");
    PRIMUS_TURBO_CHECK(current_qdq.scalar_type() == at::kBFloat16, "current_qdq must be bfloat16");
    PRIMUS_TURBO_CHECK(previous.scalar_type() == at::kBFloat16, "previous must be bfloat16");
    PRIMUS_TURBO_CHECK(previous_qdq.scalar_type() == at::kBFloat16,
                       "previous_qdq must be bfloat16");
    PRIMUS_TURBO_CHECK(dist.scalar_type() == at::kFloat, "dist must be float32");
    PRIMUS_TURBO_CHECK(dist_qdq.scalar_type() == at::kFloat, "dist_qdq must be float32");

    at::cuda::CUDAGuard guard(current.device());
    auto                stream = at::cuda::getCurrentCUDAStream();
    primus_turbo::weight_deosc_update(
        reinterpret_cast<const dtype::bfloat16 *>(current.data_ptr()),
        reinterpret_cast<const dtype::bfloat16 *>(current_qdq.data_ptr()),
        reinterpret_cast<const dtype::bfloat16 *>(previous.data_ptr()),
        reinterpret_cast<const dtype::bfloat16 *>(previous_qdq.data_ptr()), dist.data_ptr<float>(),
        dist_qdq.data_ptr<float>(), current.numel(), stream);
}

void weight_deosc_close(at::Tensor master, at::Tensor previous, const at::Tensor current_qdq,
                        at::Tensor dist, at::Tensor dist_qdq, double ratio_threshold, double eps,
                        c10::optional<at::Tensor> reset_count) {
    for (const auto &entry : std::initializer_list<std::pair<const at::Tensor *, const char *>>{
             {&master, "master"},
             {&previous, "previous"},
             {&current_qdq, "current_qdq"},
             {&dist, "dist"},
             {&dist_qdq, "dist_qdq"}}) {
        check_cuda_contiguous(*entry.first, entry.second);
        check_same_device(master, *entry.first, entry.second);
        check_same_numel(master, *entry.first, entry.second);
    }
    PRIMUS_TURBO_CHECK(master.scalar_type() == at::kFloat, "master must be float32");
    PRIMUS_TURBO_CHECK(previous.scalar_type() == at::kBFloat16, "previous must be bfloat16");
    PRIMUS_TURBO_CHECK(current_qdq.scalar_type() == at::kBFloat16, "current_qdq must be bfloat16");
    PRIMUS_TURBO_CHECK(dist.scalar_type() == at::kFloat, "dist must be float32");
    PRIMUS_TURBO_CHECK(dist_qdq.scalar_type() == at::kFloat, "dist_qdq must be float32");
    PRIMUS_TURBO_CHECK(ratio_threshold >= 0.0, "ratio_threshold must be non-negative");
    PRIMUS_TURBO_CHECK(eps > 0.0, "eps must be positive");

    if (reset_count.has_value()) {
        check_cuda_contiguous(*reset_count, "reset_count");
        check_same_device(master, *reset_count, "reset_count");
        PRIMUS_TURBO_CHECK(reset_count->scalar_type() == at::kLong, "reset_count must be int64");
        PRIMUS_TURBO_CHECK(reset_count->numel() == 1,
                           "reset_count must contain exactly one element");
    }

    at::cuda::CUDAGuard guard(master.device());
    auto                stream = at::cuda::getCurrentCUDAStream();
    primus_turbo::weight_deosc_close(
        master.data_ptr<float>(), reinterpret_cast<dtype::bfloat16 *>(previous.data_ptr()),
        reinterpret_cast<const dtype::bfloat16 *>(current_qdq.data_ptr()), dist.data_ptr<float>(),
        dist_qdq.data_ptr<float>(),
        reset_count.has_value() ? reset_count->data_ptr<int64_t>() : nullptr, master.numel(),
        static_cast<float>(ratio_threshold), static_cast<float>(eps), reset_count.has_value(),
        stream);
}

} // namespace primus_turbo::pytorch
