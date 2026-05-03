// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// permute_v2.cu — `permute_preprocessing` rewrite that removes `grid.sync()`.
//
// The v1 implementation (`permute_v1.cu`) packs four logical passes into a
// single cooperative kernel and uses `cg::this_grid().sync()` between them.
// `grid.sync()` requires `hipLaunchCooperativeKernel`, which in turn caps the
// grid at the number of blocks that fit concurrently on the device. That
// occupancy cap throttles useful parallelism on small/medium shapes and the
// barrier itself adds noticeable latency per pass.
//
// v2 splits the same algorithm into four ordinary kernels. Every cross-block
// synchronisation point in v1 becomes a kernel boundary in v2, supplied by
// the stream's implicit ordering. This lets us:
//   * launch with arbitrary grid sizes (no cooperative-launch cap);
//   * skip the LDS / register cost of `grid.sync()`;
//   * use plain `hipLaunchKernelGGL` (no special cooperative API).
//
// All v2 symbols live in `primus_turbo::v2` so v1 and v2 can be linked into
// the same library side-by-side. Only the preprocessing path is rewritten;
// the data-movement `permute_kernel` / `unpermute_kernel` continue to come
// from `permute_v1.cu` because they don't use `grid.sync()`.

#include "primus_turbo/permute.h"

#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <climits>

namespace primus_turbo {
namespace v2 {

template <int block_size>
__global__ void permute_pass1_kernel(const bool *routing_map, const int *num_dispatched_tokens_ptr,
                                     int num_of_local_experts, int *workspace_1,
                                     int rows_workspace_1, int *workspace_2, int rows_workspace_2,
                                     int32_t *tokens_per_expert, int *row_id_map,
                                     int *overflow_flag) {
    using BlockScan = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    extern __shared__ int                      smem[];
    int                                        num_dispatched_tokens = *num_dispatched_tokens_ptr;

    const int grid_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride    = gridDim.x * blockDim.x;

    // Init: zero workspace_2 / tokens_per_expert / overflow_flag using a
    // grid-stride loop. Splitting these into a separate kernel would also
    // work, but folding them into K1 keeps the launch count to four.
    for (int i = grid_thread_id; i < rows_workspace_2 * num_of_local_experts; i += grid_stride) {
        workspace_2[i] = 0;
    }
    for (int i = grid_thread_id; i < num_of_local_experts; i += grid_stride) {
        tokens_per_expert[i] = 0;
    }
    if (grid_thread_id == 0) {
        *overflow_flag = 0;
    }

    // Pass 1 body.
    int *tile_pass_1 = reinterpret_cast<int *>(smem);
    for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_1; tile_idx += gridDim.x) {
        int tile_offset = tile_idx * block_size;
        for (int i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
            tile_pass_1[i] =
                (tile_offset + i / num_of_local_experts < num_dispatched_tokens)
                    ? static_cast<int>(routing_map[tile_offset * num_of_local_experts + i])
                    : 0;
        }
        __syncthreads();

        // Per-column inclusive scan; example: 1,0,1,0,1,1,0 => 1,0,2,0,3,4,0
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
            if (tile_offset + i / num_of_local_experts < num_dispatched_tokens) {
                row_id_map[tile_offset * num_of_local_experts + i] = tile_pass_1[i];
            }
        }
        __syncthreads(); // Required: next tile reuses tile_pass_1.
    }
}

template <int block_size>
__global__ void permute_pass2_kernel(int num_of_local_experts, int *workspace_1,
                                     int rows_workspace_1, int *workspace_2, int rows_workspace_2,
                                     int32_t *tokens_per_expert) {
    using BlockScan = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    extern __shared__ int                      smem[];
    int                                       *tile_pass_2 = reinterpret_cast<int *>(smem);

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
            if (tile_offset + i / num_of_local_experts < rows_workspace_1) {
                workspace_1[tile_offset * num_of_local_experts + i] = tile_pass_2[i];
            }
        }
        __syncthreads();
    }
}

// K3: Pass 3 body + Pass 4 padding writes.
//
// Each block independently re-reads tokens_per_expert and recomputes the
// per-expert prefix sum + num_padded_tokens into its own shmem. The amount of
// recomputed work is O(num_of_local_experts) per block, which is negligible
// (num_of_local_experts <= block_size, typically <= 16) compared to the
// per-tile loops below.
//
// Pass 3 writes row_id_map[0..N*E) and Pass 4 writes
// row_id_map[N*E..(N+P)*E); the two regions are disjoint so we can fuse them.
template <int block_size>
__global__ void permute_pass3_pad_kernel(const int *workspace_1, const int *workspace_2,
                                         const int32_t *tokens_per_expert, int num_of_local_experts,
                                         int rows_workspace_1, int pad_multiple, int *row_id_map,
                                         int *overflow_flag, int64_t num_permuted_tokens,
                                         const int *num_dispatched_tokens_ptr) {
    using BlockScan = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    extern __shared__ int                      smem[];
    int                                       *tokens_per_expert_shmem = smem;
    int *tokens_per_expert_prefix_sum = tokens_per_expert_shmem + num_of_local_experts;
    int *num_padded_tokens            = tokens_per_expert_prefix_sum + num_of_local_experts;

    int num_dispatched_tokens = *num_dispatched_tokens_ptr;
    if (num_permuted_tokens < 0) {
        num_permuted_tokens = INT_MAX;
    }

    // Per-expert: read raw count, derive padded count and num_padded.
    for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
        int v                      = static_cast<int>(tokens_per_expert[i]);
        tokens_per_expert_shmem[i] = v;
        int padded =
            (pad_multiple > 0) ? ((v + pad_multiple - 1) / pad_multiple) * pad_multiple : v;
        tokens_per_expert_prefix_sum[i] = padded;
        num_padded_tokens[i]            = padded - v;
    }
    __syncthreads();
    int value = (static_cast<int>(threadIdx.x) < num_of_local_experts)
                    ? tokens_per_expert_prefix_sum[threadIdx.x]
                    : 0;
    BlockScan(temp_storage).ExclusiveSum(value, value);
    if (static_cast<int>(threadIdx.x) < num_of_local_experts) {
        tokens_per_expert_prefix_sum[threadIdx.x] = value;
    }
    __syncthreads();

    // Pass 3 body: finalise row_id_map[0..N*E) using per-tile / per-block /
    // per-expert offsets.
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

    // Pass 4 padding body: write padded entries to row_id_map[N*E..(N+P)*E).
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
}

// K4: single-block patch of tokens_per_expert with overflow handling.
// Must run after K3 (which still reads tokens_per_expert). Recomputes the
// per-expert padded prefix sum locally instead of carrying it forward through
// global memory.
template <int block_size>
__global__ void permute_finalize_kernel(int32_t *tokens_per_expert, int num_of_local_experts,
                                        int pad_multiple, int64_t num_permuted_tokens) {
    using BlockScan = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    extern __shared__ int                      smem[];
    int                                       *tokens_per_expert_shmem = smem;
    int *tokens_per_expert_prefix_sum = tokens_per_expert_shmem + num_of_local_experts;
    int *num_padded_tokens            = tokens_per_expert_prefix_sum + num_of_local_experts;

    if (num_permuted_tokens < 0) {
        num_permuted_tokens = INT_MAX;
    }

    for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
        int v                      = static_cast<int>(tokens_per_expert[i]);
        tokens_per_expert_shmem[i] = v;
        int padded =
            (pad_multiple > 0) ? ((v + pad_multiple - 1) / pad_multiple) * pad_multiple : v;
        tokens_per_expert_prefix_sum[i] = padded;
        num_padded_tokens[i]            = padded - v;
    }
    __syncthreads();
    int value = (static_cast<int>(threadIdx.x) < num_of_local_experts)
                    ? tokens_per_expert_prefix_sum[threadIdx.x]
                    : 0;
    BlockScan(temp_storage).ExclusiveSum(value, value);
    if (static_cast<int>(threadIdx.x) < num_of_local_experts) {
        tokens_per_expert_prefix_sum[threadIdx.x] = value;
    }
    __syncthreads();

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

namespace {

// Dyn shmem byte counts. K1/K2 share the (block_size * num_local_experts) tile
// layout; K3/K4 share the 3 * num_local_experts triple buffer.
inline size_t pass12_dyn_shmem_bytes(int block_size, int num_of_local_experts) {
    return static_cast<size_t>(block_size) * num_of_local_experts * sizeof(int);
}

inline size_t pass34_dyn_shmem_bytes(int num_of_local_experts) {
    return static_cast<size_t>(3) * num_of_local_experts * sizeof(int);
}

// Per-kernel max-occupancy grid cap. Unlike v1 we no longer NEED to clamp the
// grid (no cooperative launch), but capping at the device-resident size avoids
// scheduling thousands of redundant blocks for grid-stride loops that finish
// quickly anyway.
inline int max_active_grid_for_kernel(const void *func, int block_size, size_t shmem_bytes) {
    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    int num_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, device_id));
    int max_blocks_per_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, func,
                                                                        block_size, shmem_bytes));
    return (std::max) (num_cu * max_blocks_per_cu, 1);
}

} // namespace

void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                                  int *workspace_2, int rows_workspace_2, int pad_multiple,
                                  int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                  int64_t num_permuted_tokens, hipStream_t stream) {
    constexpr int block_size = PermutePreprocessConfig::kBlockSize;
    PRIMUS_TURBO_CHECK(num_of_local_experts > 0, "num_of_local_experts must be > 0");
    PRIMUS_TURBO_CHECK(num_of_local_experts <= block_size,
                       "num_of_local_experts must fit in a single block");

    const size_t shmem_pass12 = pass12_dyn_shmem_bytes(block_size, num_of_local_experts);
    const size_t shmem_pass34 = pass34_dyn_shmem_bytes(num_of_local_experts);

    // Sanity-check the dominant shmem requirement so we surface a clean error
    // up front instead of a cryptic launch failure.
    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    int max_shmem_per_block = 0;
    PRIMUS_TURBO_CHECK_HIP(hipDeviceGetAttribute(
        &max_shmem_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));
    PRIMUS_TURBO_CHECK(
        static_cast<int>(shmem_pass12) <= max_shmem_per_block, "permute_preprocessing requires ",
        static_cast<int>(shmem_pass12), " B of shared memory (block_size=", block_size,
        ", num_of_local_experts=", num_of_local_experts, ") but the device only has ",
        max_shmem_per_block, " B per block. Reduce num_of_local_experts or split the kernel.");

    // Per-kernel grid sizes: clamp to device-resident occupancy so every block
    // gets at least one wave of work without massively over-subscribing.
    const int max_grid_pass1 = max_active_grid_for_kernel(
        reinterpret_cast<const void *>(&permute_pass1_kernel<block_size>), block_size,
        shmem_pass12);
    const int max_grid_pass2 = max_active_grid_for_kernel(
        reinterpret_cast<const void *>(&permute_pass2_kernel<block_size>), block_size,
        shmem_pass12);
    const int max_grid_pass34 = max_active_grid_for_kernel(
        reinterpret_cast<const void *>(&permute_pass3_pad_kernel<block_size>), block_size,
        shmem_pass34);

    // Avoid the std::max(initializer_list) overload because PyTorch's include
    // chain may bring in a `max(a,b)` macro that swallows the brace-enclosed
    // form. Nested 2-arg std::max is robust against that.
    const int grid_pass1 = (std::min) (max_grid_pass1, (std::max) (rows_workspace_1, 1));
    const int grid_pass2 = (std::min) (max_grid_pass2, (std::max) (rows_workspace_2, 1));
    const int grid_pass34 =
        (std::min) (max_grid_pass34, (std::max) (rows_workspace_1, (std::max) (pad_multiple, 1)));

    // K1: init + Pass 1.
    permute_pass1_kernel<block_size><<<grid_pass1, block_size, shmem_pass12, stream>>>(
        routing_map, num_dispatched_tokens_ptr, num_of_local_experts, workspace_1, rows_workspace_1,
        workspace_2, rows_workspace_2, tokens_per_expert, row_id_map, overflow_flag);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());

    // K2: Pass 2.
    permute_pass2_kernel<block_size><<<grid_pass2, block_size, shmem_pass12, stream>>>(
        num_of_local_experts, workspace_1, rows_workspace_1, workspace_2, rows_workspace_2,
        tokens_per_expert);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());

    // K3: Pass 3 body + Pass 4 padding writes.
    permute_pass3_pad_kernel<block_size><<<grid_pass34, block_size, shmem_pass34, stream>>>(
        workspace_1, workspace_2, tokens_per_expert, num_of_local_experts, rows_workspace_1,
        pad_multiple, row_id_map, overflow_flag, num_permuted_tokens, num_dispatched_tokens_ptr);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());

    // K4: single-block finalise of tokens_per_expert.
    permute_finalize_kernel<block_size><<<1, block_size, shmem_pass34, stream>>>(
        tokens_per_expert, num_of_local_experts, pad_multiple, num_permuted_tokens);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

} // namespace v2
} // namespace primus_turbo
