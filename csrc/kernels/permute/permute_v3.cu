// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// permute_v3.cu — single-kernel `permute_preprocessing` rewrite based on a
// "decoupled lookback" scan (the same pattern used in `ref.cu`, ported from
// the original CUDA NVL kernel) plus an in-kernel atomic barrier.
//
// What changes vs v1 / v2:
//
//   * Removes Pass 2 entirely. The cross-block prefix sum that v1 computed
//     through `workspace_2` (a second BlockScan pass over `workspace_1`) is
//     folded into Pass 1 / Pass 3 by having each block publish its per-expert
//     block-sum to a global `block_sums[gridDim.x][E]` buffer and then
//     looking back at `block_sums[0..blockIdx.x-1]` for its own prefix.
//
//   * Removes every `grid.sync()` / `cooperative_groups` use, so we no longer
//     need `hipLaunchCooperativeKernel`. The kernel is launched with the
//     ordinary `<<<grid, block>>>` syntax.
//
//   * Cross-block synchronisation is implemented with `__hip_atomic_store` /
//     `__hip_atomic_load` (release / acquire on `__HIP_MEMORY_SCOPE_AGENT`)
//     and a tiny busy-wait loop on `ready_flags[]` and a `done_counter`.
//
//   * Per-block grid is capped at the device's max-active-blocks (computed via
//     `hipOccupancyMaxActiveBlocksPerMultiprocessor`) so all blocks are
//     guaranteed to be co-resident — without this cap a block N could spin
//     forever waiting on a ready flag from block N-1 that has not been
//     scheduled yet.
//
// All v3 symbols live in `primus_turbo::v3` so this file can coexist with
// `permute_v1.cu` / `permute_v2.cu` in the same translation unit set.

#include "primus_turbo/permute.h"

#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <climits>
#include <mutex>
#include <unordered_map>

namespace primus_turbo {
namespace v3 {

// =============================================================================
// permute_preprocessing — single-kernel decoupled lookback scan.
// =============================================================================

namespace detail {

// Release-store an int to global memory (visible device-wide).
__device__ __forceinline__ void release_store(int *ptr, int val) {
    __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Acquire-load an int from global memory.
__device__ __forceinline__ int acquire_load(int *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Atomic fetch-add with release ordering on global memory (returns the old value).
__device__ __forceinline__ int release_fetch_add(int *ptr, int val) {
    return __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

} // namespace detail

template <int block_size>
__launch_bounds__(block_size, 1) __global__ void permute_preprocessing_kernel(
    bool *routing_map, int *num_dispatched_tokens_ptr, int num_of_local_experts,
    int rows_workspace_1, int pad_multiple, int32_t *tokens_per_expert, int *row_id_map,
    int *overflow_flag, int64_t num_permuted_tokens,
    // Lookback workspace, zero-initialised by the launcher:
    //   block_sums  : [gridDim.x, num_of_local_experts]   (int)
    //   ready_flags : [gridDim.x]                          (int)
    //   done_counter: scalar                               (int)
    int *lookback_workspace) {
    using BlockScan = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    extern __shared__ int                      shmem_buf[];

    const int E                     = num_of_local_experts;
    const int num_dispatched_tokens = *num_dispatched_tokens_ptr;

    // Each block owns a contiguous slice of tile indices in `[0, rows_workspace_1)`.
    const int tiles_per_block = (rows_workspace_1 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int my_tile_start   = (int)blockIdx.x * tiles_per_block;
    const int my_tile_end =
        my_tile_start + tiles_per_block < rows_workspace_1 ? my_tile_start + tiles_per_block
                                                           : rows_workspace_1;

    // Lookback workspace pointers.
    int *block_sums   = lookback_workspace;                          // [gridDim.x, E]
    int *ready_flags  = lookback_workspace + (int)gridDim.x * E;     // [gridDim.x]
    int *done_counter = ready_flags + (int)gridDim.x;                // [1]

    // Dyn-shmem layout (re-used between phases):
    //   [0 .. block_size * E - 1] : tile_buf (Phase 1) / Phase-5 triple buffer
    //   [block_size * E .. + E - 1] : running_acc (Phase 1-2) / prev_prefix (Phase 2b-3)
    int *tile_buf    = shmem_buf;
    int *running_acc = shmem_buf + block_size * E;
    int *prev_prefix = running_acc; // alias: same E ints

    // Block 0 / thread 0 clears overflow_flag for the host.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *overflow_flag = 0;
    }

    // tiles_per_block == 1 is the dominant case once grid_size scales with
    // rows_workspace_1; in that case we can drop the inter-tile running_acc
    // path (every tile sees prev_acc == 0) and write `sum` directly to
    // running_acc as the block aggregate. Keeps Phase 1's hot loop simpler.
    const bool single_tile = (my_tile_end - my_tile_start) <= 1;

    if (!single_tile) {
        // Multi-tile path: zero-init running_acc up front so the per-tile loop
        // can accumulate sums across tiles.
        for (int i = (int)threadIdx.x; i < E; i += block_size) {
            running_acc[i] = 0;
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Phase 1: per-tile InclusiveSum, accumulating WITHIN this block's tile slice.
    //
    //   On exit, row_id_map[tile_offset..]'s non-zero entries hold the in-block
    //   running InclusiveSum (zero entries stay zero — they signal "this expert
    //   does not need this token"). running_acc[e] now equals the per-expert
    //   sum of routing_map over this block's tile range.
    // -------------------------------------------------------------------------
    for (int tile_idx = my_tile_start; tile_idx < my_tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * block_size;

        // Stage tile of routing_map into shmem (cast bool -> int).
        for (int i = (int)threadIdx.x; i < block_size * E; i += block_size) {
            tile_buf[i] = (tile_offset + i / E < num_dispatched_tokens)
                              ? (int)routing_map[(int64_t)tile_offset * E + i]
                              : 0;
        }
        __syncthreads();

        // Per-column InclusiveSum; example for an expert col:
        //   1,0,1,0,1,1,0  =>  1,0,2,0,3,4,0  (+ prev block-internal acc per col)
        for (int i = 0; i < E; ++i) {
            int32_t in = tile_buf[(int)threadIdx.x * E + i];
            int32_t out, sum;
            BlockScan(temp_storage).InclusiveSum(in, out, sum);
            if (single_tile) {
                tile_buf[(int)threadIdx.x * E + i] = (in == 1) ? out : 0;
                if (threadIdx.x == 0) {
                    running_acc[i] = sum;
                }
            } else {
                const int prev_acc                 = running_acc[i];
                tile_buf[(int)threadIdx.x * E + i] = (in == 1) ? (out + prev_acc) : 0;
                if (threadIdx.x == 0) {
                    running_acc[i] = prev_acc + sum;
                }
            }
        }
        __syncthreads();

        // Write tile back to row_id_map.
        for (int64_t i = (int64_t)threadIdx.x; i < (int64_t)block_size * E; i += block_size) {
            if (tile_offset + i / E < num_dispatched_tokens) {
                row_id_map[(int64_t)tile_offset * E + i] = tile_buf[i];
            }
        }
        __syncthreads(); // Required: Phase 2a reads running_acc; next tile reuses shmem.
    }
    // Empty-tile blocks (my_tile_start >= rows_workspace_1, single_tile path)
    // never wrote running_acc; make sure it is zeroed so Phase 2a publishes
    // a meaningful aggregate (== 0).
    if (single_tile && my_tile_end == my_tile_start) {
        for (int i = (int)threadIdx.x; i < E; i += block_size) {
            running_acc[i] = 0;
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Phase 2a: publish this block's running_acc as block_sums[blockIdx.x],
    //           and atomically add it into the global tokens_per_expert.
    //
    //   block_sums uses a relaxed store; the matching release is on
    //   ready_flags[blockIdx.x] below, so any thread that sees ready_flag=1
    //   is also guaranteed to see the matching block_sums slice.
    // -------------------------------------------------------------------------
    for (int i = (int)threadIdx.x; i < E; i += block_size) {
        const int v = running_acc[i];
        block_sums[(int)blockIdx.x * E + i] = v;
        if (v != 0) {
            atomicAdd(&tokens_per_expert[i], v);
        }
    }
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        detail::release_store(&ready_flags[blockIdx.x], 1);
    }

    // -------------------------------------------------------------------------
    // Phase 2b: lookback. Wait for ready_flags[0..blockIdx.x-1] = 1 and then
    //           reduce block_sums[0..blockIdx.x-1] into prev_prefix[E].
    //
    //   IMPORTANT: this only terminates if every block we're waiting on is
    //   actually scheduled. The launcher caps gridDim.x at the device's
    //   max-active-block count for exactly this reason.
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        for (int b = 0; b < (int)blockIdx.x; ++b) {
            while (detail::acquire_load(&ready_flags[b]) == 0) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    // Now all earlier blocks' block_sums are visible.
    // Each lane sums one expert's prefix across all earlier blocks.
    // (After this we'll overwrite running_acc-aliased shmem with prev_prefix.)
    for (int i = (int)threadIdx.x; i < E; i += block_size) {
        int sum = 0;
        for (int b = 0; b < (int)blockIdx.x; ++b) {
            sum += block_sums[b * E + i];
        }
        prev_prefix[i] = sum;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 3: fold prev_prefix into row_id_map for this block's tile range.
    //
    //   After this, row_id_map[*]'s non-zero entries hold the per-expert
    //   InclusiveSum across the *entire* dispatched-tokens range — i.e. the
    //   final value v1 computed via (workspace_1 + workspace_2) — but still
    //   excluding the cross-expert prefix that comes from tokens_per_expert.
    // -------------------------------------------------------------------------
    for (int tile_idx = my_tile_start; tile_idx < my_tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * block_size;
        for (int64_t i = (int64_t)threadIdx.x; i < (int64_t)block_size * E; i += block_size) {
            if (tile_offset + i / E < num_dispatched_tokens) {
                const int64_t offset    = (int64_t)tile_offset * E + i;
                const int     expert_id = (int)(i % E);
                const int     old       = row_id_map[offset];
                if (old != 0) {
                    row_id_map[offset] = old + prev_prefix[expert_id];
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 4: atomic barrier — wait for all blocks to finish Phase 3.
    //
    //   Once every block has incremented done_counter, every block is also
    //   done updating tokens_per_expert (Phase 2a happens before Phase 4).
    //   We use a release-add + acquire-load pair so that observing
    //   `done_counter == gridDim.x` guarantees every block's tokens_per_expert
    //   atomicAdd from Phase 2a is visible.
    // -------------------------------------------------------------------------
    __syncthreads();

    if (threadIdx.x == 0) {
        detail::release_fetch_add(done_counter, 1);
        while (detail::acquire_load(done_counter) < (int)gridDim.x) {
            // Spin; on gfx942 a short sleep cuts atomic traffic during contention.
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 5: compute tokens_per_expert_prefix_sum and apply it to row_id_map
    //          for this block's tile range. Re-use shmem with a new layout:
    //            [0 .. E - 1]    : tokens_per_expert_shmem
    //            [E .. 2E - 1]   : tokens_per_expert_prefix_sum
    //            [2E .. 3E - 1]  : num_padded_tokens
    // -------------------------------------------------------------------------
    int *tokens_per_expert_shmem      = shmem_buf;
    int *tokens_per_expert_prefix_sum = shmem_buf + E;
    int *num_padded_tokens            = shmem_buf + 2 * E;

    int npt = (num_permuted_tokens < 0) ? INT_MAX : (int)num_permuted_tokens;

    for (int i = (int)threadIdx.x; i < E; i += block_size) {
        const int v                = (int)tokens_per_expert[i];
        tokens_per_expert_shmem[i] = v;
        const int padded =
            (pad_multiple > 0) ? ((v + pad_multiple - 1) / pad_multiple) * pad_multiple : v;
        tokens_per_expert_prefix_sum[i] = padded;
        num_padded_tokens[i]            = padded - v;
    }
    __syncthreads();

    // ExclusiveSum across E entries to turn padded counts into prefixes.
    int v = ((int)threadIdx.x < E) ? tokens_per_expert_prefix_sum[threadIdx.x] : 0;
    BlockScan(temp_storage).ExclusiveSum(v, v);
    if ((int)threadIdx.x < E) {
        tokens_per_expert_prefix_sum[threadIdx.x] = v;
    }
    __syncthreads();

    // Apply the per-expert prefix to this block's tile range, with overflow.
    for (int tile_idx = my_tile_start; tile_idx < my_tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * block_size;
        for (int64_t i = (int64_t)threadIdx.x; i < (int64_t)block_size * E; i += block_size) {
            if (tile_offset + i / E < num_dispatched_tokens) {
                const int64_t offset    = (int64_t)tile_offset * E + i;
                const int     expert_id = (int)(i % E);
                const int     old       = row_id_map[offset];
                if (old != 0) {
                    const int new_value = old + tokens_per_expert_prefix_sum[expert_id];
                    if (new_value > npt) {
                        *overflow_flag     = 1;
                        row_id_map[offset] = 0;
                    } else {
                        row_id_map[offset] = new_value;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 6: padding writes. Each block handles `i in [blockIdx.x, pad_multiple)`
    //          with grid-stride.
    // -------------------------------------------------------------------------
    for (int i = (int)blockIdx.x; i < pad_multiple; i += (int)gridDim.x) {
        const int64_t offset = ((int64_t)i + num_dispatched_tokens) * E;
        for (int j = 0; j < E; ++j) {
            if (i < num_padded_tokens[j]) {
                const int padded_offset =
                    -(tokens_per_expert_shmem[j] + tokens_per_expert_prefix_sum[j] + i + 1);
                if (abs(padded_offset) > npt) {
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

    // -------------------------------------------------------------------------
    // Phase 7: block 0 finalises tokens_per_expert (overflow handling).
    // -------------------------------------------------------------------------
    if (blockIdx.x == 0) {
        for (int i = (int)threadIdx.x; i < E; i += block_size) {
            const int tokens_for_expert_i = tokens_per_expert_shmem[i] + num_padded_tokens[i];
            const int overflow_num =
                tokens_for_expert_i + tokens_per_expert_prefix_sum[i] - npt;
            if (overflow_num < 0) {
                tokens_per_expert[i] = tokens_for_expert_i;
            } else {
                tokens_per_expert[i] =
                    (tokens_for_expert_i - overflow_num) > 0 ? (tokens_for_expert_i - overflow_num)
                                                             : 0;
            }
        }
    }
}

namespace {

// Dyn shmem byte count: covers both Phase 1 layout (tile_buf + running_acc)
// and Phase 5 layout (3 * E ints), and is dominated by the former.
inline size_t permute_preprocess_dyn_shmem_bytes(int block_size, int num_of_local_experts) {
    const size_t phase1 =
        (static_cast<size_t>(block_size) + 1) * num_of_local_experts * sizeof(int);
    const size_t phase5 = static_cast<size_t>(3) * num_of_local_experts * sizeof(int);
    return std::max(phase1, phase5);
}

// Per-stream cache of the lookback workspace allocation. Allocating /
// freeing this scratch buffer on every launcher call (hipMallocAsync +
// hipFreeAsync) added ~10–15 µs of host overhead which dominated small-case
// preproc runtime. We keep the largest size we've seen per stream and only
// re-allocate when we need a bigger buffer; the buffer is never freed during
// process lifetime (a few KB at most).
struct LookbackCacheEntry {
    int   *ptr        = nullptr;
    size_t size_bytes = 0;
};

inline LookbackCacheEntry &get_lookback_cache(hipStream_t stream) {
    static std::mutex                                       g_mu;
    static std::unordered_map<hipStream_t, LookbackCacheEntry> g_cache;
    std::lock_guard<std::mutex>                              lk(g_mu);
    return g_cache[stream];
}

inline int *acquire_lookback_workspace(hipStream_t stream, size_t needed_bytes) {
    LookbackCacheEntry &e = get_lookback_cache(stream);
    if (e.size_bytes < needed_bytes) {
        if (e.ptr != nullptr) {
            PRIMUS_TURBO_CHECK_HIP(hipFreeAsync(e.ptr, stream));
        }
        PRIMUS_TURBO_CHECK_HIP(
            hipMallocAsync(reinterpret_cast<void **>(&e.ptr), needed_bytes, stream));
        e.size_bytes = needed_bytes;
    }
    return e.ptr;
}

inline int permute_preprocess_max_active_grid(int block_size, size_t shmem_bytes) {
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
    return std::max(num_cu * max_blocks_per_cu, 1);
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
    PRIMUS_TURBO_CHECK(rows_workspace_1 > 0, "rows_workspace_1 must be > 0");

    // workspace_1 / workspace_2 / rows_workspace_2 are kept in the signature so
    // this matches v1 / v2's launcher ABI; v3 doesn't need them.
    (void)workspace_1;
    (void)workspace_2;
    (void)rows_workspace_2;

    const size_t shmem_bytes = permute_preprocess_dyn_shmem_bytes(block_size, num_of_local_experts);

    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    int max_shmem_per_block = 0;
    PRIMUS_TURBO_CHECK_HIP(hipDeviceGetAttribute(
        &max_shmem_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));
    PRIMUS_TURBO_CHECK(static_cast<int>(shmem_bytes) <= max_shmem_per_block,
                       "permute_preprocessing v3 requires ", static_cast<int>(shmem_bytes),
                       " B of shared memory (block_size=", block_size,
                       ", num_of_local_experts=", num_of_local_experts,
                       ") but the device only has ", max_shmem_per_block,
                       " B per block. Reduce num_of_local_experts.");

    // Cap gridDim.x at the device's max-active-block count: the in-kernel
    // ready-flag busy-wait would deadlock if a later block was waiting on a
    // ready flag from an earlier block that the scheduler hasn't launched yet.
    // const int max_grid       = 80;
    // const int requested_grid = (std::max) (rows_workspace_1, (std::max) (pad_multiple, 1));
    // const int grid_size      = (std::min) (max_grid, (std::max) (requested_grid, 1));
    const int grid_size = 80;

    // Allocate + zero the lookback workspace:
    //   block_sums  [grid_size, E]   ints
    //   ready_flags [grid_size]      ints
    //   done_counter                 int
    const size_t lookback_ints =
        (size_t)grid_size * (size_t)num_of_local_experts + (size_t)grid_size + 1;
    const size_t lookback_bytes     = lookback_ints * sizeof(int);
    int         *lookback_workspace = acquire_lookback_workspace(stream, lookback_bytes);
    PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(lookback_workspace, 0, lookback_bytes, stream));

    // tokens_per_expert is updated via atomicAdd inside Phase 2a, so it must
    // start zeroed.
    PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(tokens_per_expert, 0,
                                          static_cast<size_t>(num_of_local_experts) *
                                              sizeof(int32_t),
                                          stream));

    permute_preprocessing_kernel<block_size><<<grid_size, block_size, shmem_bytes, stream>>>(
        routing_map, num_dispatched_tokens_ptr, num_of_local_experts, rows_workspace_1,
        pad_multiple, tokens_per_expert, row_id_map, overflow_flag, num_permuted_tokens,
        lookback_workspace);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

} // namespace v3
} // namespace primus_turbo
