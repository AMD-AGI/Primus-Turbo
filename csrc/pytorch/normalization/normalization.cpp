// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/normalization.h"

#include "../extensions.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

// Effective grid cap for the rmsnorm kernels. Derived from device CU count
// (portable across MI100/MI250/MI300/MI325/MI355 — and would be NVIDIA-portable
// if the warp-size assumption changed) and clamped to the architectural
// occupancy ceiling for the actual block size we'll launch with. Mirrors TE's
// flow of `multiprocessorCount * cudaOccupancyMaxActiveBlocks(...)`, but
// driven from a closed-form formula since our kernel resource use is regular.
//
// Used both as the kernel grid bound AND to size the backward dgamma_part
// scratch — keeping a single source of truth avoids drift between launch
// config and workspace.
static int64_t rmsnorm_target_ctas(int block_threads) {
    int dev = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&dev));
    int num_cus = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&num_cus, hipDeviceAttributeMultiprocessorCount, dev));
    return static_cast<int64_t>(num_cus) * rmsnorm_effective_ctas_per_cu(block_threads);
}

// Predict the block size that the launcher will pick for this shape. Mirrors
// the dispatch logic in rmsnorm_{fwd,bwd}_impl so the host can size
// dgamma_part (and pick target_ctas) against the same block we'll actually
// launch with — keeping occupancy clamp and scratch in agreement.
static int rmsnorm_predicted_block(int64_t inner_len, int dtype_size_bytes) {
    const int     unroll    = static_cast<int>(sizeof(uint4)) / dtype_size_bytes;
    const bool    aligned   = (inner_len % unroll == 0);
    const int64_t warp_span = (int64_t) THREADS_PER_WARP * unroll;
    if (aligned && inner_len <= warp_span) {
        // warp-per-row fast path
        return RMSNORM_WARPS_PER_BLOCK * THREADS_PER_WARP;
    }
    return rmsnorm_pick_blocksize(inner_len, aligned ? unroll : 1);
}

// Upper bound on rows the bwd stage 0 may write into dgamma_part. The
// warp-per-row variant launches `ceil(outer_len / RMSNORM_WARPS_PER_BLOCK)`
// blocks (bounded by `ceil(target_ctas / RMSNORM_WARPS_PER_BLOCK)`) and writes
// RMSNORM_WARPS_PER_BLOCK partial rows per block. Both inputs are rounded up
// to a RMSNORM_WARPS_PER_BLOCK boundary so the bound holds regardless of
// whether num_cus or RMSNORM_CTAS_PER_CU happens to divide evenly.
static int64_t rmsnorm_bwd_parts_upper_bound(int64_t outer_len, int64_t target_ctas) {
    constexpr int64_t W            = RMSNORM_WARPS_PER_BLOCK;
    const int64_t     outer_padded = ((outer_len + W - 1) / W) * W;
    const int64_t     ctas_padded  = ((target_ctas + W - 1) / W) * W;
    return std::min(outer_padded, ctas_padded);
}

std::vector<at::Tensor> rmsnorm_fwd(const at::Tensor &input, const at::Tensor &gamma,
                                    const double eps) {
    TORCH_CHECK(input.is_contiguous(), "rmsnorm_fwd: input must be contiguous.");
    TORCH_CHECK(gamma.is_contiguous(), "rmsnorm_fwd: gamma must be contiguous.");

    const int64_t inner_len = gamma.numel();
    const int64_t outer_len = input.numel() / inner_len;
    TORCH_CHECK(input.numel() % inner_len == 0, "input.numel() must be divisible by gamma.numel()");

    auto output = at::empty_like(input);
    auto rs     = at::empty({outer_len}, input.options().dtype(at::kFloat));

    const int     dtype_bytes = static_cast<int>(input.element_size());
    const int     block       = rmsnorm_predicted_block(inner_len, dtype_bytes);
    const int64_t target_ctas = rmsnorm_target_ctas(block);
    auto          stream      = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kFloat) {
        rmsnorm_fwd_impl<float>(input.data_ptr<float>(), gamma.data_ptr<float>(),
                                output.data_ptr<float>(), rs.data_ptr<float>(), inner_len,
                                outer_len, static_cast<float>(eps), target_ctas, stream);
    } else if (input.scalar_type() == at::kHalf) {
        rmsnorm_fwd_impl<float16>(reinterpret_cast<float16 *>(input.data_ptr()),
                                  reinterpret_cast<float16 *>(gamma.data_ptr()),
                                  reinterpret_cast<float16 *>(output.data_ptr()),
                                  rs.data_ptr<float>(), inner_len, outer_len,
                                  static_cast<float>(eps), target_ctas, stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        rmsnorm_fwd_impl<bfloat16>(reinterpret_cast<bfloat16 *>(input.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(gamma.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(output.data_ptr()),
                                   rs.data_ptr<float>(), inner_len, outer_len,
                                   static_cast<float>(eps), target_ctas, stream);
    } else {
        PRIMUS_TURBO_ERROR("RMSNorm only support : [float32, float16, bfloat16]");
    }
    return {output, rs};
}

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &input, const at::Tensor &gamma,
                                    const at::Tensor &grad_output, const at::Tensor &rs,
                                    const double eps) {
    TORCH_CHECK(input.is_contiguous(), "rmsnorm_bwd: input must be contiguous.");
    TORCH_CHECK(gamma.is_contiguous(), "rmsnorm_bwd: gamma must be contiguous.");
    TORCH_CHECK(grad_output.is_contiguous(), "rmsnorm_bwd: grad_output must be contiguous.");
    TORCH_CHECK(rs.is_contiguous(), "rmsnorm_bwd: rs must be contiguous.");
    TORCH_CHECK(rs.scalar_type() == at::kFloat, "rmsnorm_bwd: rs must be float32.");

    const int64_t inner_len = gamma.numel();
    const int64_t outer_len = input.numel() / inner_len;
    TORCH_CHECK(input.numel() % inner_len == 0, "input.numel() must be divisible by gamma.numel()");

    auto grad_input = at::empty_like(input);
    auto grad_gamma = at::empty_like(gamma);

    auto stream = at::cuda::getCurrentCUDAStream();

    const int     dtype_bytes  = static_cast<int>(input.element_size());
    const int     block        = rmsnorm_predicted_block(inner_len, dtype_bytes);
    const int64_t target_ctas  = rmsnorm_target_ctas(block);
    const int64_t parts_alloc  = rmsnorm_bwd_parts_upper_bound(outer_len, target_ctas);
    auto          options_fp32 = input.options().dtype(at::kFloat);
    auto          dgamma_part  = at::empty({parts_alloc, inner_len}, options_fp32);

    int64_t n_parts = 0;
    if (input.scalar_type() == at::kFloat) {
        n_parts = rmsnorm_bwd_stage0_impl<float>(
            input.data_ptr<float>(), gamma.data_ptr<float>(), grad_output.data_ptr<float>(),
            rs.data_ptr<float>(), grad_input.data_ptr<float>(), dgamma_part.data_ptr<float>(),
            inner_len, outer_len, target_ctas, stream);
        rmsnorm_bwd_finalize_impl<float>(dgamma_part.data_ptr<float>(),
                                         grad_gamma.data_ptr<float>(), inner_len, n_parts, stream);
    } else if (input.scalar_type() == at::kHalf) {
        n_parts = rmsnorm_bwd_stage0_impl<float16>(
            reinterpret_cast<float16 *>(input.data_ptr()),
            reinterpret_cast<float16 *>(gamma.data_ptr()),
            reinterpret_cast<float16 *>(grad_output.data_ptr()), rs.data_ptr<float>(),
            reinterpret_cast<float16 *>(grad_input.data_ptr()), dgamma_part.data_ptr<float>(),
            inner_len, outer_len, target_ctas, stream);
        rmsnorm_bwd_finalize_impl<float16>(dgamma_part.data_ptr<float>(),
                                           reinterpret_cast<float16 *>(grad_gamma.data_ptr()),
                                           inner_len, n_parts, stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        n_parts = rmsnorm_bwd_stage0_impl<bfloat16>(
            reinterpret_cast<bfloat16 *>(input.data_ptr()),
            reinterpret_cast<bfloat16 *>(gamma.data_ptr()),
            reinterpret_cast<bfloat16 *>(grad_output.data_ptr()), rs.data_ptr<float>(),
            reinterpret_cast<bfloat16 *>(grad_input.data_ptr()), dgamma_part.data_ptr<float>(),
            inner_len, outer_len, target_ctas, stream);
        rmsnorm_bwd_finalize_impl<bfloat16>(dgamma_part.data_ptr<float>(),
                                            reinterpret_cast<bfloat16 *>(grad_gamma.data_ptr()),
                                            inner_len, n_parts, stream);
    } else {
        PRIMUS_TURBO_ERROR("RMSNorm only support : [float32, float16, bfloat16]");
    }

    return {grad_input, grad_gamma};
}

} // namespace primus_turbo::pytorch
