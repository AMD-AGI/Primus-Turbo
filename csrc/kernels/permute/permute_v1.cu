// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// HIP port of the permute / unpermute kernels originally written for CUDA.
//
// Notable changes vs the CUDA prototype:
//   * <cuda_runtime.h>          -> <hip/hip_runtime.h>
//   * cub::BlockScan            -> hipcub::BlockScan
//   * cooperative_groups header -> <hip/hip_cooperative_groups.h>
//   * cudaStream_t              -> hipStream_t
//   * cudaLaunchCooperativeKernel / cudaFuncSetAttribute -> hip equivalents
//   * Removed the bogus `template <..., const int WARP_SIZE = 32>` parameter
//     that conflicted with the `WARP_SIZE` macro and was never used inside
//     the kernel.
//   * Added a host-side launcher that uses `hipLaunchCooperativeKernel`,
//     because `grid.sync()` requires cooperative launch on AMD GPUs.
//   * Replaced std::numeric_limits<int>::max() with INT_MAX in device code.
//   * `__nv_bfloat16` -> `primus_turbo::dtype::bfloat16` (= hip_bfloat16).
//   * The torch-aware host wrappers (PermuteArgs / UnpermuteArgs /
//     permute_preprocessing returning torch::Tensor) belong to the PyTorch
//     binding layer and live in `csrc/pytorch/permute/`. This file exposes
//     plain-pointer launchers only, matching how `csrc/kernels/` ships
//     frontend-agnostic code in `libprimus_turbo_kernels.so`.

#include "primus_turbo/permute.h"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <climits>

namespace primus_turbo {

namespace cg = cooperative_groups;

using dtype::bfloat16;

// =============================================================================
// pad_tokens_per_expert
// =============================================================================

// Kernels must have external linkage on AMD because they declare
// `extern __shared__` shared-memory arrays; clang refuses to give those a
// definition when their enclosing function has internal linkage. Keep them in
// the `primus_turbo` namespace but out of any anonymous namespace.

__global__ void pad_tokens_per_expert_kernel(const int32_t *src, int64_t *dst, int num_experts,
                                             int pad_multiple) {
    int i = threadIdx.x;
    if (i < num_experts) {
        int32_t val = src[i];
        if (pad_multiple > 0) {
            val = ((val + pad_multiple - 1) / pad_multiple) * pad_multiple;
        }
        dst[i] = static_cast<int64_t>(val);
    }
}

// =============================================================================
// permute_preprocessing
// =============================================================================

/**
 * @brief Preprocessing kernel for permute: computes row_id_map, tokens_per_expert and
 * overflow_flag from the routing map using a multi-pass cooperative scan.
 *
 * Must be launched via hipLaunchCooperativeKernel because of the `grid.sync()` calls.
 */
template <int block_size>
__global__ void permute_preprocessing_kernel(bool *routing_map, int *num_dispatched_tokens_ptr,
                                             int num_of_local_experts, int *workspace_1,
                                             int rows_workspace_1, int *workspace_2,
                                             int rows_workspace_2, int pad_multiple,
                                             int32_t *tokens_per_expert, int *row_id_map,
                                             int *overflow_flag, int64_t num_permuted_tokens) {
    auto       grid       = cg::this_grid();
    using BlockScan       = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    extern __shared__ int                      shmem_in_permute_preprocessing_kernel[];
    int                                        num_dispatched_tokens = *num_dispatched_tokens_ptr;

    /**
     * Pass 1: compute the cumsum for each block, then store the result in
     * workspace_1; memset workspace_2 and tokens_per_expert to 0.
     */
    for (int i = grid.thread_rank(); i < rows_workspace_2 * num_of_local_experts; i += grid.size())
        workspace_2[i] = 0;
    for (int i = grid.thread_rank(); i < num_of_local_experts; i += grid.size())
        tokens_per_expert[i] = 0;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *overflow_flag = 0;
    }
    if (num_permuted_tokens < 0) {
        num_permuted_tokens = INT_MAX;
    }

    int *tile_pass_1 = reinterpret_cast<int *>(shmem_in_permute_preprocessing_kernel);
    for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_1; tile_idx += gridDim.x) {
        int tile_offset = tile_idx * block_size;
        for (int i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
            tile_pass_1[i] =
                (tile_offset + i / num_of_local_experts < num_dispatched_tokens)
                    ? static_cast<int>(routing_map[tile_offset * num_of_local_experts + i])
                    : 0;
        }
        __syncthreads();

        // Example for each column: 1,0,1,0,1,1,0 => 1,0,2,0,3,4,0
        for (int i = 0; i < num_of_local_experts; i++) {
            // TODO: many bank conflicts here
            int32_t in = tile_pass_1[threadIdx.x * num_of_local_experts + i];
            int32_t out, sum;
            BlockScan(temp_storage).InclusiveSum(in, out, sum);
            tile_pass_1[threadIdx.x * num_of_local_experts + i] = in == 1 ? out : 0;
            if (threadIdx.x == 0) {
                workspace_1[tile_idx * num_of_local_experts + i] = sum;
            }
        }
        __syncthreads();

        for (int64_t i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
            if ((tile_offset + i / num_of_local_experts < num_dispatched_tokens)) {
                row_id_map[tile_offset * num_of_local_experts + i] =
                    static_cast<int>(tile_pass_1[i]);
            }
        }
    }

    grid.sync();

    /**
     * Pass 2: compute the cumsum for each block in workspace_1.
     * Use atomicAdd to compute the prefix sum of all block-sums, store the result
     * in workspace_2, and update tokens_per_expert.
     */
    int *tile_pass_2 = reinterpret_cast<int *>(shmem_in_permute_preprocessing_kernel);
    for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_2; tile_idx += gridDim.x) {
        int tile_offset = tile_idx * block_size;
        for (int i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
            tile_pass_2[i] = (tile_offset + i / num_of_local_experts < rows_workspace_1)
                                 ? workspace_1[tile_offset * num_of_local_experts + i]
                                 : 0;
        }
        __syncthreads();

        for (int i = 0; i < num_of_local_experts; i++) {
            // TODO: many bank conflicts here
            int32_t in = tile_pass_2[threadIdx.x * num_of_local_experts + i];
            int32_t out, sum;
            BlockScan(temp_storage).ExclusiveSum(in, out, sum);
            tile_pass_2[threadIdx.x * num_of_local_experts + i] = out;
            for (int pos = threadIdx.x + tile_idx + 1; pos < rows_workspace_2; pos += block_size) {
                atomicAdd(&workspace_2[pos * num_of_local_experts + i], sum);
            }
            if (threadIdx.x == 0) {
                atomicAdd(&tokens_per_expert[i], sum);
            }
        }
        __syncthreads();

        for (int64_t i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
            if ((tile_offset + i / num_of_local_experts < rows_workspace_1)) {
                workspace_1[tile_offset * num_of_local_experts + i] = tile_pass_2[i];
            }
        }
        __syncthreads();
    }

    grid.sync();

    int *tokens_per_expert_shmem = reinterpret_cast<int *>(shmem_in_permute_preprocessing_kernel);
    int *tokens_per_expert_prefix_sum =
        reinterpret_cast<int *>(tokens_per_expert_shmem + num_of_local_experts);

    /**
     * Pass 3: compute the prefix sum of tokens_per_expert and use
     * tokens_per_expert + workspace_1 + workspace_2 to update row_id_map.
     */
    for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
        tokens_per_expert_shmem[i] = static_cast<int>(tokens_per_expert[i]);
        tokens_per_expert_prefix_sum[i] =
            pad_multiple > 0
                ? (tokens_per_expert_shmem[i] + pad_multiple - 1) / pad_multiple * pad_multiple
                : tokens_per_expert_shmem[i];
    }
    __syncthreads();
    int value = static_cast<int>(threadIdx.x) < num_of_local_experts
                    ? tokens_per_expert_prefix_sum[threadIdx.x]
                    : 0;
    BlockScan(temp_storage).ExclusiveSum(value, value);
    if (static_cast<int>(threadIdx.x) < num_of_local_experts) {
        tokens_per_expert_prefix_sum[threadIdx.x] = value;
    }
    __syncthreads();

    for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_1; tile_idx += gridDim.x) {
        int tile_offset = tile_idx * block_size;
        for (int64_t i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
            if (tile_offset + i / num_of_local_experts < num_dispatched_tokens) {
                int64_t offset    = (tile_offset * num_of_local_experts + i);
                int     expert_id = i % num_of_local_experts;
                auto    old_value = row_id_map[offset];
                if (old_value != 0) {
                    auto new_value =
                        old_value + workspace_1[tile_idx * num_of_local_experts + expert_id] +
                        workspace_2[(tile_idx / block_size) * num_of_local_experts + expert_id] +
                        tokens_per_expert_prefix_sum[expert_id];
                    if (new_value > num_permuted_tokens) {
                        *overflow_flag     = 1;
                        row_id_map[offset] = 0;
                    } else {
                        row_id_map[offset] = new_value;
                    }
                }
            }
        }
    }

    grid.sync();

    /**
     * Pass 4: compute the padding for tokens_per_expert.
     */
    int *num_padded_tokens =
        reinterpret_cast<int *>(tokens_per_expert_shmem + 2 * num_of_local_experts);
    for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
        int padded_value;
        if (pad_multiple <= 0) {
            padded_value = tokens_per_expert_shmem[i];
        } else {
            padded_value =
                (tokens_per_expert_shmem[i] + pad_multiple - 1) / pad_multiple * pad_multiple;
        }
        num_padded_tokens[i] = padded_value - tokens_per_expert_shmem[i];
    }
    __syncthreads();

    for (int i = blockIdx.x; i < pad_multiple; i += gridDim.x) {
        int64_t offset = (i + num_dispatched_tokens) * num_of_local_experts;
        for (int j = 0; j < num_of_local_experts; j++) {
            if (i < num_padded_tokens[j]) {
                auto padded_offset =
                    -(tokens_per_expert_shmem[j] + tokens_per_expert_prefix_sum[j] + i + 1);
                if (abs(padded_offset) > num_permuted_tokens) {
                    *overflow_flag         = 1;
                    row_id_map[offset + j] = 0;
                } else {
                    row_id_map[offset + j] = padded_offset;
                }
            } else {
                row_id_map[offset + j] = 0;
            }
        }
    }

    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
            auto tokens_for_expert_i = tokens_per_expert_shmem[i] + num_padded_tokens[i];
            auto overflow_num =
                tokens_for_expert_i + tokens_per_expert_prefix_sum[i] - num_permuted_tokens;
            if (overflow_num < 0) {
                tokens_per_expert[i] = tokens_for_expert_i;
            } else {
                tokens_per_expert[i] = max(0, (int) (tokens_for_expert_i - overflow_num));
            }
        }
    }
}

namespace {

// Returns the dynamic shared-memory size in bytes required by
// `permute_preprocessing_kernel<block_size>` for a given `num_of_local_experts`.
inline size_t permute_preprocess_dyn_shmem_bytes(int block_size, int num_of_local_experts) {
    const size_t pass12 = static_cast<size_t>(block_size) * num_of_local_experts * sizeof(int);
    // Pass 3 and 4 share the same buffer split into 3 banks of `num_of_local_experts` ints
    // (tokens_per_expert_shmem, tokens_per_expert_prefix_sum, num_padded_tokens).
    const size_t pass34 = static_cast<size_t>(3) * num_of_local_experts * sizeof(int);
    return std::max(pass12, pass34);
}

// Returns the maximum grid size that allows the cooperative kernel to fit on
// the current device.
inline int permute_preprocess_max_cooperative_grid(int block_size, size_t shmem_bytes) {
    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));

    int num_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, device_id));

    int max_blocks_per_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_cu,
        reinterpret_cast<const void *>(
            &permute_preprocessing_kernel<PermutePreprocessConfig::kBlockSize>),
        block_size, shmem_bytes));

    int max_grid = num_cu * max_blocks_per_cu;
    return std::max(max_grid, 1);
}

} // namespace

void pad_tokens_per_expert(const int32_t *src, int64_t *dst, int num_experts, int pad_multiple,
                           hipStream_t stream) {
    PRIMUS_TURBO_CHECK(num_experts > 0, "num_experts must be > 0");
    PRIMUS_TURBO_CHECK(num_experts <= MAX_THREADS_PER_BLOCK,
                       "num_experts must be <= MAX_THREADS_PER_BLOCK");
    pad_tokens_per_expert_kernel<<<1, num_experts, 0, stream>>>(src, dst, num_experts,
                                                                pad_multiple);
}

void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                                  int *workspace_2, int rows_workspace_2, int pad_multiple,
                                  int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                  int64_t num_permuted_tokens, hipStream_t stream) {
    constexpr int block_size = PermutePreprocessConfig::kBlockSize;
    PRIMUS_TURBO_CHECK(num_of_local_experts > 0, "num_of_local_experts must be > 0");
    PRIMUS_TURBO_CHECK(num_of_local_experts <= block_size,
                       "num_of_local_experts must fit in a single block");

    const size_t shmem_bytes = permute_preprocess_dyn_shmem_bytes(block_size, num_of_local_experts);

    // Verify the shmem fits inside the device's max-per-block budget so we
    // surface a clear error instead of bouncing the cooperative launch later.
    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    int max_shmem_per_block = 0;
    PRIMUS_TURBO_CHECK_HIP(hipDeviceGetAttribute(
        &max_shmem_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));
    PRIMUS_TURBO_CHECK(static_cast<int>(shmem_bytes) <= max_shmem_per_block,
                       "permute_preprocessing requires ", static_cast<int>(shmem_bytes),
                       " B of shared memory (block_size=", block_size,
                       ", num_of_local_experts=", num_of_local_experts,
                       ") but the device only has ", max_shmem_per_block,
                       " B per block. Reduce num_of_local_experts or split the kernel.");

    const int max_grid = permute_preprocess_max_cooperative_grid(block_size, shmem_bytes);
    // Avoid the std::max(initializer_list) overload because PyTorch's include
    // chain may bring in a `max(a,b)` macro that swallows the brace-enclosed
    // form. Nested 2-arg std::max is robust against that.
    const int requested_grid =
        (std::max)(rows_workspace_1, (std::max)(rows_workspace_2, (std::max)(pad_multiple, 1)));
    const int grid_size = (std::min)(max_grid, (std::max)(requested_grid, 1));

    void *args[] = {&routing_map,           &num_dispatched_tokens_ptr,
                    &num_of_local_experts,  &workspace_1,
                    &rows_workspace_1,      &workspace_2,
                    &rows_workspace_2,      &pad_multiple,
                    &tokens_per_expert,     &row_id_map,
                    &overflow_flag,         &num_permuted_tokens};

    PRIMUS_TURBO_CHECK_HIP(hipLaunchCooperativeKernel(
        reinterpret_cast<const void *>(&permute_preprocessing_kernel<block_size>),
        dim3(grid_size), dim3(block_size), args, shmem_bytes, stream));
}

// =============================================================================
// permute / unpermute  (data movement)
// =============================================================================

namespace {

// 16-byte vector type used as the float4 surrogate for vectorised loads/stores.
// Defined as plain bytes so it works for any element type (uint8_t, uint16_t,
// bfloat16) without aliasing problems.
struct alignas(16) PackedBytes16 {
    uint32_t x, y, z, w;
};

template <typename DType>
__device__ inline float dtype_to_float(DType x) {
    return static_cast<float>(x);
}

template <>
__device__ inline float dtype_to_float<bfloat16>(bfloat16 x) {
    return static_cast<float>(x);
}

template <typename DType>
__device__ inline DType float_to_dtype(float x) {
    return static_cast<DType>(x);
}

template <>
__device__ inline bfloat16 float_to_dtype<bfloat16>(float x) {
    return bfloat16(x);
}

constexpr PackedBytes16 kZeroPacked16 = {0u, 0u, 0u, 0u};

} // namespace

template <int block_size, typename DType, typename ProbType, typename ScalarType>
__global__ void permute_kernel(const DType *tokens, DType *permuted_tokens,
                               const ScalarType *scaling_factor, ScalarType *permuted_scaling_factor,
                               const ProbType *probs, ProbType *permuted_probs,
                               const int *row_id_map, const int *num_dispatched_tokens_ptr,
                               int pad_multiple, int num_of_local_experts, int hidden_size,
                               int scales_per_token, int local_rank, int num_ranks_per_node) {
    constexpr int kThreadsPerToken = PermuteKernelConfig::kThreadsPerToken;
    int64_t tokens_per_block      = blockDim.x / kThreadsPerToken;
    int64_t extended_lane_id      = threadIdx.x % kThreadsPerToken;
    int64_t extended_warp_id      = threadIdx.x / kThreadsPerToken;
    int     num_dispatched_tokens = *num_dispatched_tokens_ptr + pad_multiple;
    extern __shared__ int shmem_in_permute_kernel[];
    int *expert_routing_map = shmem_in_permute_kernel;

    constexpr int num_eles_per_pack = sizeof(PackedBytes16) / sizeof(DType);

    for (int64_t block_start = blockIdx.x * tokens_per_block; block_start < num_dispatched_tokens;
         block_start += tokens_per_block * gridDim.x) {
        int64_t token_id = block_start + extended_warp_id;

        // Load the tile of row_id_map into shared memory.
        for (int i = threadIdx.x; i < num_of_local_experts * tokens_per_block; i += block_size) {
            expert_routing_map[i] =
                (block_start + i / num_of_local_experts < num_dispatched_tokens)
                    ? row_id_map[block_start * num_of_local_experts + i]
                    : 0;
        }
        __syncthreads();

        if (token_id < num_dispatched_tokens) {
            int64_t              hidden_size_pack = hidden_size / num_eles_per_pack;
            const PackedBytes16 *tokens_pack =
                reinterpret_cast<const PackedBytes16 *>(tokens);
            PackedBytes16 *permuted_tokens_pack = reinterpret_cast<PackedBytes16 *>(permuted_tokens);

            for (int64_t i = 0; i < num_of_local_experts; i++) {
                int64_t dest_token_id =
                    expert_routing_map[extended_warp_id * num_of_local_experts + i];
                if (dest_token_id > 0) {
                    for (int64_t j = extended_lane_id; j < hidden_size_pack;
                         j += kThreadsPerToken) {
                        permuted_tokens_pack[(dest_token_id - 1) * hidden_size_pack + j] =
                            tokens_pack[token_id * hidden_size_pack + j];
                    }
                } else if (dest_token_id < 0) {
                    for (int64_t j = extended_lane_id; j < hidden_size_pack;
                         j += kThreadsPerToken) {
                        permuted_tokens_pack[(-dest_token_id - 1) * hidden_size_pack + j] =
                            kZeroPacked16;
                    }
                }
            }

            if (scaling_factor != nullptr) {
                for (int64_t i = 0; i < num_of_local_experts; i++) {
                    int64_t dest_token_id =
                        expert_routing_map[extended_warp_id * num_of_local_experts + i];
                    if (dest_token_id > 0) {
                        for (int64_t j = extended_lane_id; j < scales_per_token;
                             j += kThreadsPerToken) {
                            permuted_scaling_factor[(dest_token_id - 1) * scales_per_token + j] =
                                scaling_factor[token_id * scales_per_token + j];
                        }
                    } else if (dest_token_id < 0) {
                        for (int64_t j = extended_lane_id; j < scales_per_token;
                             j += kThreadsPerToken) {
                            permuted_scaling_factor[(-dest_token_id - 1) * scales_per_token + j] =
                                ScalarType{0};
                        }
                    }
                }
            }

            if (probs != nullptr) {
                for (int64_t i = 0; i < num_of_local_experts; i++) {
                    int64_t dest_token_id =
                        expert_routing_map[extended_warp_id * num_of_local_experts + i];
                    if (dest_token_id > 0) {
                        permuted_probs[dest_token_id - 1] =
                            probs[token_id * num_of_local_experts * num_ranks_per_node +
                                  local_rank * num_of_local_experts + i];
                    } else if (dest_token_id < 0) {
                        permuted_probs[-dest_token_id - 1] = ProbType{0};
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <int block_size, typename DType, typename ProbType>
__global__ void unpermute_kernel(const DType *permuted_tokens, DType *tokens,
                                 const ProbType *permuted_probs, ProbType *probs,
                                 const int *row_id_map, const int *num_dispatched_tokens_ptr,
                                 int num_of_local_experts, int hidden_size, int local_rank,
                                 int num_ranks_per_node) {
    constexpr int kThreadsPerToken = PermuteKernelConfig::kThreadsPerToken;
    int64_t tokens_per_block      = blockDim.x / kThreadsPerToken;
    int64_t extended_lane_id      = threadIdx.x % kThreadsPerToken;
    int64_t extended_warp_id      = threadIdx.x / kThreadsPerToken;
    extern __shared__ int shmem_in_permute_kernel[];
    int *expert_routing_map  = shmem_in_permute_kernel;
    int  num_dispatched_tokens = *num_dispatched_tokens_ptr;

    constexpr int num_eles_per_pack = sizeof(PackedBytes16) / sizeof(DType);

    for (int64_t block_start = blockIdx.x * tokens_per_block; block_start < num_dispatched_tokens;
         block_start += tokens_per_block * gridDim.x) {
        int64_t token_id = block_start + extended_warp_id;

        for (int i = threadIdx.x; i < num_of_local_experts * tokens_per_block; i += block_size) {
            expert_routing_map[i] =
                (block_start + i / num_of_local_experts < num_dispatched_tokens)
                    ? row_id_map[block_start * num_of_local_experts + i]
                    : 0;
        }
        __syncthreads();

        if (token_id < num_dispatched_tokens) {
            int64_t              hidden_size_pack = hidden_size / num_eles_per_pack;
            const PackedBytes16 *permuted_tokens_pack =
                reinterpret_cast<const PackedBytes16 *>(permuted_tokens);
            PackedBytes16 *tokens_pack = reinterpret_cast<PackedBytes16 *>(tokens);

            PackedBytes16 buffer_pack;
            float         accumulator[num_eles_per_pack];
            DType        *buffer_ptr = reinterpret_cast<DType *>(&buffer_pack);

            for (int64_t j = extended_lane_id; j < hidden_size_pack; j += kThreadsPerToken) {
#pragma unroll
                for (int k = 0; k < num_eles_per_pack; k++)
                    accumulator[k] = 0.0f;
                for (int i = 0; i < num_of_local_experts; i++) {
                    int64_t source_token_id =
                        expert_routing_map[extended_warp_id * num_of_local_experts + i];
                    if (source_token_id > 0) {
                        buffer_pack =
                            permuted_tokens_pack[(source_token_id - 1) * hidden_size_pack + j];
#pragma unroll
                        for (int k = 0; k < num_eles_per_pack; k++) {
                            accumulator[k] += dtype_to_float<DType>(buffer_ptr[k]);
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < num_eles_per_pack; k++) {
                    buffer_ptr[k] = float_to_dtype<DType>(accumulator[k]);
                }
                tokens_pack[token_id * hidden_size_pack + j] = buffer_pack;
            }

            if (permuted_probs != nullptr) {
                for (int64_t j = extended_lane_id;
                     j < num_of_local_experts * num_ranks_per_node; j += kThreadsPerToken) {
                    float value = 0.0f;
                    if (j / num_of_local_experts == local_rank) {
                        int64_t source_token_id =
                            expert_routing_map[extended_warp_id * num_of_local_experts +
                                               j % num_of_local_experts];
                        if (source_token_id > 0) {
                            value = static_cast<float>(permuted_probs[source_token_id - 1]);
                        }
                    }
                    probs[token_id * num_of_local_experts * num_ranks_per_node + j] =
                        static_cast<ProbType>(value);
                }
            }
        }
        __syncthreads();
    }
}

template <typename DType, typename ProbType, typename ScalarType>
void permute_impl(const DType *tokens, DType *permuted_tokens, const ScalarType *scaling_factor,
                  ScalarType *permuted_scaling_factor, const ProbType *probs,
                  ProbType *permuted_probs, const int *row_id_map,
                  const int *num_dispatched_tokens_ptr, int pad_multiple, int num_of_local_experts,
                  int hidden_size, int scales_per_token, int local_rank, int num_ranks_per_node,
                  int grid_size, hipStream_t stream) {
    constexpr int block_size       = PermuteKernelConfig::kBlockSize;
    constexpr int tokens_per_block = PermuteKernelConfig::kTokensPerBlock;
    constexpr int num_eles_per_pack = sizeof(PackedBytes16) / sizeof(DType);

    PRIMUS_TURBO_CHECK(permuted_tokens != nullptr, "permuted_tokens must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(DType))");
    PRIMUS_TURBO_CHECK(grid_size > 0, "grid_size must be > 0");

    const size_t shmem_bytes =
        static_cast<size_t>(num_of_local_experts) * tokens_per_block * sizeof(int);

    permute_kernel<block_size, DType, ProbType, ScalarType>
        <<<grid_size, block_size, shmem_bytes, stream>>>(
            tokens, permuted_tokens, scaling_factor, permuted_scaling_factor, probs, permuted_probs,
            row_id_map, num_dispatched_tokens_ptr, pad_multiple, num_of_local_experts, hidden_size,
            scales_per_token, local_rank, num_ranks_per_node);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <typename DType, typename ProbType>
void unpermute_impl(const DType *permuted_tokens, DType *tokens, const ProbType *permuted_probs,
                    ProbType *probs, const int *row_id_map, const int *num_dispatched_tokens_ptr,
                    int num_of_local_experts, int hidden_size, int local_rank,
                    int num_ranks_per_node, int grid_size, hipStream_t stream) {
    constexpr int block_size       = PermuteKernelConfig::kBlockSize;
    constexpr int tokens_per_block = PermuteKernelConfig::kTokensPerBlock;
    constexpr int num_eles_per_pack = sizeof(PackedBytes16) / sizeof(DType);

    PRIMUS_TURBO_CHECK(tokens != nullptr, "tokens output must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(DType))");
    PRIMUS_TURBO_CHECK(grid_size > 0, "grid_size must be > 0");

    const size_t shmem_bytes =
        static_cast<size_t>(num_of_local_experts) * tokens_per_block * sizeof(int);

    unpermute_kernel<block_size, DType, ProbType><<<grid_size, block_size, shmem_bytes, stream>>>(
        permuted_tokens, tokens, permuted_probs, probs, row_id_map, num_dispatched_tokens_ptr,
        num_of_local_experts, hidden_size, local_rank, num_ranks_per_node);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

// -----------------------------------------------------------------------------
// Explicit instantiations (currently supported (DType, ProbType, ScalarType)).
// Add new ones here as we extend dtype coverage.
// -----------------------------------------------------------------------------

// permute: 8-bit / 16-bit token paths, float scales, float probs.
template void permute_impl<uint8_t, float, float>(const uint8_t *, uint8_t *, const float *,
                                                  float *, const float *, float *, const int *,
                                                  const int *, int, int, int, int, int, int, int,
                                                  hipStream_t);
template void permute_impl<uint16_t, float, float>(const uint16_t *, uint16_t *, const float *,
                                                   float *, const float *, float *, const int *,
                                                   const int *, int, int, int, int, int, int, int,
                                                   hipStream_t);

// unpermute: bfloat16 only (16-bit accum-via-float32).
template void unpermute_impl<bfloat16, float>(const bfloat16 *, bfloat16 *, const float *, float *,
                                              const int *, const int *, int, int, int, int, int,
                                              hipStream_t);

} // namespace primus_turbo
