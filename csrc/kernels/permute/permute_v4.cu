// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// permute_v4.cu — v1 optimised for AMD GPU shared memory / code-gen.
//
// Derived from `permute_v1.cu` (NOT from v2/v3). The 4-pass cooperative
// algorithm is kept verbatim; only the in-kernel data layout and per-expert
// dispatch change.
//
// Three targeted optimisations vs v1:
//
//   1. LDS layout transposed: the Pass 1 / Pass 2 tile buffer is indexed as
//      `tile_buf[expert][token]` with a `+1`-dword pad on the token stride
//      (`PAD_STRIDE = block_size + 1`). On gfx942 this turns v1's
//      scan-time stride-E access (up to 32-way bank conflict for common E)
//      into a stride-1 access, which is the inherent wave64/bank32 2-way
//      conflict minimum (no conflict at all on gfx950's 64-bank LDS).
//
//   2. `num_of_local_experts` is dispatched at launch time to a template
//      parameter `E_STATIC` (∈ {1, 2, 4, 8, 16, 32}). The per-expert scan
//      loops become compile-time bounded so the compiler can const-fold the
//      `e * PAD_STRIDE + tid` address arithmetic and, for small E, unroll
//      the loop body. Runtime E falls back to `E_STATIC == 0`.
//
//   3. `(in == 1 ? out : 0)` in the scan body is rewritten as `in * out`.
//      Because `in ∈ {0, 1}` the two are equivalent, but the latter compiles
//      to a single VALU integer multiply instead of a masked select.
//
// Everything else — `hipLaunchCooperativeKernel`, `grid.sync()`, the 4-pass
// structure, `atomicAdd` on `workspace_2` / `tokens_per_expert` in Pass 2 —
// matches v1 byte-for-byte. All v4 symbols live in `primus_turbo::v4` so this
// file coexists with v1/v2/v3 in the same translation unit set.

#include "primus_turbo/permute.h"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <climits>

namespace primus_turbo {
namespace v4 {

namespace cg = cooperative_groups;

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------
//
// Template parameters:
//   * block_size — must match PermutePreprocessConfig::kBlockSize
//   * E_STATIC   — compile-time `num_of_local_experts` when > 0; 0 means the
//                  launcher didn't specialise and we fall back to runtime E.
//
// Must be launched via `hipLaunchCooperativeKernel` because of the
// `grid.sync()` calls (same as v1).
template <int block_size, int E_STATIC>
__global__ void permute_preprocessing_kernel(bool *routing_map, int *num_dispatched_tokens_ptr,
                                             int num_of_local_experts, int *workspace_1,
                                             int rows_workspace_1, int *workspace_2,
                                             int rows_workspace_2, int pad_multiple,
                                             int32_t *tokens_per_expert, int *row_id_map,
                                             int *overflow_flag, int64_t num_permuted_tokens) {
    auto       grid       = cg::this_grid();
    using BlockScan       = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    extern __shared__ int                      shmem_buf[];

    // When the launcher specialised on E_STATIC > 0, the ternary folds to the
    // compile-time value and `E` becomes a `constexpr`-like constant that the
    // address arithmetic below can const-fold. For E_STATIC == 0 (rare,
    // uncommon-E fallback) it stays runtime.
    const int     E   = (E_STATIC > 0) ? E_STATIC : num_of_local_experts;
    constexpr int PAD = block_size + 1; // +1 dword → write-back LDS read is 4-way
                                        //             instead of 16-way on gfx942.
    int num_dispatched_tokens = *num_dispatched_tokens_ptr;

    // -------------------------------------------------------------------------
    // Pass 1: zero workspace_2 / tokens_per_expert / overflow_flag, then do
    // per-tile InclusiveSum over each expert column of `routing_map`.
    // -------------------------------------------------------------------------
    for (int i = grid.thread_rank(); i < rows_workspace_2 * E; i += grid.size())
        workspace_2[i] = 0;
    for (int i = grid.thread_rank(); i < E; i += grid.size())
        tokens_per_expert[i] = 0;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *overflow_flag = 0;
    }
    if (num_permuted_tokens < 0) {
        num_permuted_tokens = INT_MAX;
    }

    int *tile_pass_1 = shmem_buf;
    for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_1; tile_idx += gridDim.x) {
        int tile_offset = tile_idx * block_size;

        // Load `routing_map` into LDS with TRANSPOSED layout:
        //   src  = routing_map[tile_offset*E + i]  (flat [token][expert])
        //   dest = tile_pass_1[expert*PAD + token] (transposed [expert][token])
        //
        // The per-expert scan below then reads `tile_pass_1[e*PAD + tid]` with
        // stride-1 across lanes → bank conflict is only the inherent
        // wave64/bank32 2-way on gfx942 (1-way on gfx950), regardless of E.
        for (int i = threadIdx.x; i < block_size * E; i += block_size) {
            const int token  = i / E;
            const int expert = i % E;
            const int v      = (tile_offset + token < num_dispatched_tokens)
                                   ? (int) routing_map[(int64_t) tile_offset * E + i]
                                   : 0;
            tile_pass_1[expert * PAD + token] = v;
        }
        __syncthreads();

        // Per-expert InclusiveSum. Example column: 1,0,1,0,1,1,0 → 1,0,2,0,3,4,0.
        // `#pragma unroll` lets the compiler fully unroll small E (≤ 8) where
        // the scan bodies amortise well, but defers on large E (16, 32) to
        // avoid code-size / register-pressure blow-up. Clang's cost model
        // handles the fallback case on its own.
        #pragma unroll
        for (int e = 0; e < E; e++) {
            int32_t in = tile_pass_1[e * PAD + threadIdx.x];
            int32_t out, sum;
            BlockScan(temp_storage).InclusiveSum(in, out, sum);
            // in ∈ {0,1} → `in * out` is identical to `in == 1 ? out : 0`
            // but compiles to a single V_MUL_LO_U32, no branch.
            tile_pass_1[e * PAD + threadIdx.x] = in * out;
            if (threadIdx.x == 0) {
                workspace_1[tile_idx * E + e] = sum;
            }
        }
        __syncthreads();

        // Write tile back to row_id_map. Flat iteration → coalesced global
        // write (64 lanes store 64 consecutive dwords into 2 cache lines).
        // LDS read pattern `(i%E)*PAD + (i/E)` is up to 4-way conflict on
        // gfx942 (derivation in the v4 commentary), but it's a single pass
        // per tile so the cost is dominated by global-store latency.
        for (int64_t i = threadIdx.x; i < (int64_t) block_size * E; i += block_size) {
            if (tile_offset + i / E < num_dispatched_tokens) {
                row_id_map[(int64_t) tile_offset * E + i] =
                    tile_pass_1[(i % E) * PAD + (i / E)];
            }
        }
    }

    grid.sync();

    // -------------------------------------------------------------------------
    // Pass 2: ExclusiveSum over workspace_1 rows + atomicAdd propagation to
    // workspace_2 / tokens_per_expert. (Same algorithm as v1; the atomics are
    // cheap because `rows_workspace_2 ≤ block_size` in practice, so each
    // thread executes at most O(1) atomicAdd per expert.)
    // -------------------------------------------------------------------------
    int *tile_pass_2 = shmem_buf;
    for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_2; tile_idx += gridDim.x) {
        int tile_offset = tile_idx * block_size;

        for (int i = threadIdx.x; i < block_size * E; i += block_size) {
            const int token  = i / E;
            const int expert = i % E;
            const int v      = (tile_offset + token < rows_workspace_1)
                                   ? workspace_1[(int64_t) tile_offset * E + i]
                                   : 0;
            tile_pass_2[expert * PAD + token] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int e = 0; e < E; e++) {
            int32_t in = tile_pass_2[e * PAD + threadIdx.x];
            int32_t out, sum;
            BlockScan(temp_storage).ExclusiveSum(in, out, sum);
            tile_pass_2[e * PAD + threadIdx.x] = out;
            for (int pos = threadIdx.x + tile_idx + 1; pos < rows_workspace_2; pos += block_size) {
                atomicAdd(&workspace_2[pos * E + e], sum);
            }
            if (threadIdx.x == 0) {
                atomicAdd(&tokens_per_expert[e], sum);
            }
        }
        __syncthreads();

        for (int64_t i = threadIdx.x; i < (int64_t) block_size * E; i += block_size) {
            if (tile_offset + i / E < rows_workspace_1) {
                workspace_1[(int64_t) tile_offset * E + i] =
                    tile_pass_2[(i % E) * PAD + (i / E)];
            }
        }
        __syncthreads();
    }

    grid.sync();

    // -------------------------------------------------------------------------
    // Pass 3: compute prefix sum of (padded) tokens_per_expert, fold
    // workspace_1 / workspace_2 / prefix_sum into row_id_map.
    //
    // LDS from here on is re-used as 3 bands of E ints (≪ Pass 1/2 layout),
    // so no transpose / padding concerns — same layout as v1.
    // -------------------------------------------------------------------------
    int *tokens_per_expert_shmem      = shmem_buf;
    int *tokens_per_expert_prefix_sum = shmem_buf + E;

    for (int i = threadIdx.x; i < E; i += block_size) {
        tokens_per_expert_shmem[i] = (int) tokens_per_expert[i];
        tokens_per_expert_prefix_sum[i] =
            pad_multiple > 0
                ? (tokens_per_expert_shmem[i] + pad_multiple - 1) / pad_multiple * pad_multiple
                : tokens_per_expert_shmem[i];
    }
    __syncthreads();
    int value = ((int) threadIdx.x < E) ? tokens_per_expert_prefix_sum[threadIdx.x] : 0;
    BlockScan(temp_storage).ExclusiveSum(value, value);
    if ((int) threadIdx.x < E) {
        tokens_per_expert_prefix_sum[threadIdx.x] = value;
    }
    __syncthreads();

    for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_1; tile_idx += gridDim.x) {
        int tile_offset = tile_idx * block_size;
        for (int64_t i = threadIdx.x; i < (int64_t) block_size * E; i += block_size) {
            if (tile_offset + i / E < num_dispatched_tokens) {
                int64_t offset    = (int64_t) tile_offset * E + i;
                int     expert_id = (int) (i % E);
                int     old_value = row_id_map[offset];
                if (old_value != 0) {
                    int new_value = old_value + workspace_1[tile_idx * E + expert_id] +
                                    workspace_2[(tile_idx / block_size) * E + expert_id] +
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

    // -------------------------------------------------------------------------
    // Pass 4: padding writes into row_id_map + finalise tokens_per_expert.
    // Identical to v1.
    // -------------------------------------------------------------------------
    int *num_padded_tokens = shmem_buf + 2 * E;
    for (int i = threadIdx.x; i < E; i += block_size) {
        int padded_value =
            (pad_multiple <= 0)
                ? tokens_per_expert_shmem[i]
                : ((tokens_per_expert_shmem[i] + pad_multiple - 1) / pad_multiple) * pad_multiple;
        num_padded_tokens[i] = padded_value - tokens_per_expert_shmem[i];
    }
    __syncthreads();

    for (int i = blockIdx.x; i < pad_multiple; i += gridDim.x) {
        int64_t offset = ((int64_t) i + num_dispatched_tokens) * E;
        for (int j = 0; j < E; j++) {
            if (i < num_padded_tokens[j]) {
                int padded_offset = -(tokens_per_expert_shmem[j] +
                                      tokens_per_expert_prefix_sum[j] + i + 1);
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
        for (int i = threadIdx.x; i < E; i += block_size) {
            int tokens_for_expert_i = tokens_per_expert_shmem[i] + num_padded_tokens[i];
            int overflow_num =
                tokens_for_expert_i + tokens_per_expert_prefix_sum[i] - (int) num_permuted_tokens;
            if (overflow_num < 0) {
                tokens_per_expert[i] = tokens_for_expert_i;
            } else {
                tokens_per_expert[i] = max(0, tokens_for_expert_i - overflow_num);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host-side helpers
// -----------------------------------------------------------------------------

namespace {

// Dynamic LDS bytes:
//   Pass 1/2: `E * (block_size + 1)` ints (transposed tile + pad dword).
//   Pass 3/4: `3 * E` ints (shared bands).
inline size_t dyn_shmem_bytes(int block_size, int E) {
    const size_t pass12 = static_cast<size_t>(E) * (block_size + 1) * sizeof(int);
    const size_t pass34 = static_cast<size_t>(3) * E * sizeof(int);
    return std::max(pass12, pass34);
}

template <int block_size, int E_STATIC>
inline int max_cooperative_grid(size_t shmem_bytes) {
    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    int num_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, device_id));
    int max_blocks_per_cu = 0;
    PRIMUS_TURBO_CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_cu,
        reinterpret_cast<const void *>(&permute_preprocessing_kernel<block_size, E_STATIC>),
        block_size, shmem_bytes));
    return std::max(num_cu * max_blocks_per_cu, 1);
}

// Actually launch one specialised instantiation. Hoisted out so the dispatch
// switch in the external launcher stays readable.
template <int block_size, int E_STATIC>
inline void launch_specialised(bool *routing_map, int *num_dispatched_tokens_ptr,
                               int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                               int *workspace_2, int rows_workspace_2, int pad_multiple,
                               int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                               int64_t num_permuted_tokens, hipStream_t stream) {
    const size_t shmem_bytes = dyn_shmem_bytes(block_size, num_of_local_experts);

    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    int max_shmem_per_block = 0;
    PRIMUS_TURBO_CHECK_HIP(hipDeviceGetAttribute(&max_shmem_per_block,
                                                 hipDeviceAttributeMaxSharedMemoryPerBlock,
                                                 device_id));
    PRIMUS_TURBO_CHECK(static_cast<int>(shmem_bytes) <= max_shmem_per_block,
                       "permute_preprocessing v4 requires ", static_cast<int>(shmem_bytes),
                       " B of shared memory (block_size=", block_size,
                       ", num_of_local_experts=", num_of_local_experts,
                       ") but the device only has ", max_shmem_per_block,
                       " B per block. Reduce num_of_local_experts.");

    const int max_grid       = max_cooperative_grid<block_size, E_STATIC>(shmem_bytes);
    const int requested_grid = (std::max) (rows_workspace_1,
                                           (std::max) (rows_workspace_2,
                                                       (std::max) (pad_multiple, 1)));
    const int grid_size      = (std::min) (max_grid, (std::max) (requested_grid, 1));

    void *args[] = {&routing_map,         &num_dispatched_tokens_ptr,
                    &num_of_local_experts, &workspace_1,
                    &rows_workspace_1,    &workspace_2,
                    &rows_workspace_2,    &pad_multiple,
                    &tokens_per_expert,   &row_id_map,
                    &overflow_flag,       &num_permuted_tokens};

    PRIMUS_TURBO_CHECK_HIP(hipLaunchCooperativeKernel(
        reinterpret_cast<const void *>(&permute_preprocessing_kernel<block_size, E_STATIC>),
        dim3(grid_size), dim3(block_size), args, shmem_bytes, stream));
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

    // Dispatch E to a compile-time specialisation when it's one of the MoE
    // values we see in production; otherwise fall through to the runtime-E
    // fallback (E_STATIC == 0). The specialisation enables const-propagation
    // of `e * PAD + tid` arithmetic and unrolling of the per-expert loops.
#define PRIMUS_TURBO_V4_DISPATCH_CASE(E_VAL)                                                       \
    case E_VAL:                                                                                    \
        launch_specialised<block_size, E_VAL>(routing_map, num_dispatched_tokens_ptr,              \
                                              num_of_local_experts, workspace_1,                   \
                                              rows_workspace_1, workspace_2, rows_workspace_2,     \
                                              pad_multiple, tokens_per_expert, row_id_map,         \
                                              overflow_flag, num_permuted_tokens, stream);         \
        break

    switch (num_of_local_experts) {
        PRIMUS_TURBO_V4_DISPATCH_CASE(1);
        PRIMUS_TURBO_V4_DISPATCH_CASE(2);
        PRIMUS_TURBO_V4_DISPATCH_CASE(4);
        PRIMUS_TURBO_V4_DISPATCH_CASE(8);
        PRIMUS_TURBO_V4_DISPATCH_CASE(16);
        PRIMUS_TURBO_V4_DISPATCH_CASE(32);
    default:
        launch_specialised<block_size, 0>(routing_map, num_dispatched_tokens_ptr,
                                          num_of_local_experts, workspace_1, rows_workspace_1,
                                          workspace_2, rows_workspace_2, pad_multiple,
                                          tokens_per_expert, row_id_map, overflow_flag,
                                          num_permuted_tokens, stream);
        break;
    }
#undef PRIMUS_TURBO_V4_DISPATCH_CASE
}

} // namespace v4
} // namespace primus_turbo
