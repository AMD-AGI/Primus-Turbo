// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "moe_permute.cuh"
#include "primus_turbo/common.h"
#include "primus_turbo/moe_permute.h"

#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace primus_turbo {

using ::primus_turbo::deep_ep::st_na_global;
using ::primus_turbo::dtype::bfloat16;
using ::primus_turbo::dtype::float16;

template <int kNumThreads>
__device__ __forceinline__ void
fill_s_tile_from_routing_map(int *s_tile, const bool *routing_map, int tile_offset, int E,
                             int kNumItemsPerTile, int num_dispatched_tokens) {
    const int     thread_id    = static_cast<int>(threadIdx.x);
    const int64_t routing_base = static_cast<int64_t>(tile_offset) * E;
    for (int i = thread_id; i < kNumItemsPerTile * E; i += kNumThreads) {
        const int gtoken = tile_offset + i / E;
        s_tile[i] =
            (gtoken < num_dispatched_tokens) ? static_cast<int>(routing_map[routing_base + i]) : 0;
    }
}

template <int kNumThreads, typename topk_idx_t>
__device__ __forceinline__ void
fill_s_tile_from_topk_idx(int *s_tile, const topk_idx_t *topk_idx, int num_topk, int tile_offset,
                          int E, int kNumItemsPerTile, int num_dispatched_tokens) {
    const int thread_id = static_cast<int>(threadIdx.x);

    for (int i = thread_id; i < kNumItemsPerTile * E; i += kNumThreads) {
        s_tile[i] = 0;
    }
    __syncthreads();

    for (int i = thread_id; i < kNumItemsPerTile * num_topk; i += kNumThreads) {
        const int t      = i / num_topk;
        const int k      = i % num_topk;
        const int gtoken = tile_offset + t;
        if (gtoken >= num_dispatched_tokens) {
            continue;
        }
        const int e = topk_idx[static_cast<int64_t>(gtoken) * num_topk + k];
        if (e >= 0 && e < E) {
            s_tile[t * E + e] = 1;
        }
    }
}

template <int kNumThreads, typename expert_map_t>
__device__ __forceinline__ void fill_s_tile(int *s_tile, const expert_map_t *expert_map,
                                            int tile_offset, int num_experts, int num_topk,
                                            int kNumItemsPerTile, int num_dispatched_tokens) {
    if constexpr (std::is_same_v<expert_map_t, bool>) {
        fill_s_tile_from_routing_map<kNumThreads>(s_tile, expert_map, tile_offset, num_experts,
                                                  kNumItemsPerTile, num_dispatched_tokens);
    } else {
        fill_s_tile_from_topk_idx<kNumThreads, expert_map_t>(
            s_tile, expert_map, num_topk, tile_offset, num_experts, kNumItemsPerTile,
            num_dispatched_tokens);
    }
}

template <int kNumThreads, typename expert_map_t>
__launch_bounds__(kNumThreads, 1) __global__
    void permute_preprocessing_kernel(const expert_map_t *expert_map,
                                      const int *num_dispatched_tokens_ptr, int num_experts,
                                      int num_topk, int pad_multiple, int32_t *tokens_per_expert,
                                      int *row_id_map, int *overflow_flag,
                                      int64_t num_permuted_tokens, TempStorageLayout layout) {
    uint64_t *tile_state = layout.tile_state;
    using BlockScan      = hipcub::BlockScan<int32_t, kNumThreads>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    extern __shared__ int                      dyn_shmem[];

    constexpr int kNumItemsPerTile = kNumThreads;

    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto block_id  = static_cast<int>(blockIdx.x);
    const auto grid_size = static_cast<int>(gridDim.x);
    const int  E         = num_experts;

    // Per-block scratch: LDS, or a slice of the global vsmem fallback.
    int *temp_storage  = get_temp_storage<int>(dyn_shmem, layout.vsmem);
    int *s_tile        = temp_storage;
    int *s_acc         = s_tile + kNumItemsPerTile * E;
    int *s_excl_prefix = s_acc + E;
    int *s_tpe_prefix  = s_excl_prefix + E;
    int *s_num_padded  = s_tpe_prefix + E;

    const int num_dispatched_tokens = *num_dispatched_tokens_ptr;
    const int num_token_tiles = (num_dispatched_tokens + kNumItemsPerTile - 1) / kNumItemsPerTile;
    const int tiles_per_block = (num_token_tiles + grid_size - 1) / grid_size;
    const int tile_begin      = block_id * tiles_per_block;
    const int tile_end        = min(tile_begin + tiles_per_block, num_token_tiles);
    // single_tile blocks keep s_tile in LDS through Phase 6 and skip the
    // row_id_map spill / read-back.
    const bool single_tile = (tile_end - tile_begin) == 1;

    const int npt = num_permuted_tokens < 0 ? INT_MAX : static_cast<int>(num_permuted_tokens);

    if (block_id == 0 and thread_id == 0)
        *overflow_flag = 0;

    for (int i = thread_id; i < E; i += kNumThreads)
        s_acc[i] = 0;

    __syncthreads();

    // row_id_map row layout: [dst_rows | expert_idx | n_routed], length 2*E + 1.
    const int row_stride = 2 * E + 1;

    // Phase 1-2: per-tile InclusiveSum, accumulate into s_acc; multi-tile
    // blocks spill the partial scan to row_id_map for Phase 6 to read back.
    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * kNumItemsPerTile;

        fill_s_tile<kNumThreads, expert_map_t>(s_tile, expert_map, tile_offset, E, num_topk,
                                               kNumItemsPerTile, num_dispatched_tokens);
        __syncthreads();

        for (int e = 0; e < E; ++e) {
            const int local = s_tile[thread_id * E + e];
            int       excl_block, scan_total;
            BlockScan(scan_temp).ExclusiveSum(local, excl_block, scan_total);
            const int prev            = s_acc[e];
            s_tile[thread_id * E + e] = (local == 1) ? (excl_block + prev + 1) : 0;
            if (thread_id == 0) {
                s_acc[e] += scan_total;
            }
            __syncthreads();
        }

        if (not single_tile) {
            for (int i = thread_id; i < kNumItemsPerTile * E; i += kNumThreads) {
                const int gtoken = tile_offset + i / E;
                const int e      = i % E;
                if (gtoken < num_dispatched_tokens) {
                    row_id_map[static_cast<int64_t>(gtoken) * row_stride + e] = s_tile[i];
                }
            }
            __syncthreads();
        }
    }

    // Phase 3: per-expert decoupled lookback. Thread e publishes PARTIAL,
    // walks predecessors backward until it hits a COMPLETE, then re-publishes
    // the inclusive prefix as COMPLETE.
    if (thread_id < E) {
        const int      e         = thread_id;
        const int32_t  agg       = s_acc[e];
        const uint32_t init_flag = (block_id == 0) ? TileState::kComplete : TileState::kPartial;
        store_tile_state(&tile_state[static_cast<int64_t>(block_id) * E + e], init_flag, agg);

        int32_t accum = 0;
        for (int b = block_id - 1; b >= 0; --b) {
            TileState s;
            do {
                s = load_tile_state(&tile_state[static_cast<int64_t>(b) * E + e]);
            } while (s.flag == TileState::kInvalid);
            accum += s.value;
            if (s.flag == TileState::kComplete) {
                break;
            }
        }
        s_excl_prefix[e] = accum;

        if (block_id != 0) {
            store_tile_state(&tile_state[static_cast<int64_t>(block_id) * E + e],
                             TileState::kComplete, accum + agg);
        }
    }
    __syncthreads();

    // Phase 5: read full-grid inclusive prefix from the LAST block's slot,
    // then ExclusiveSum across experts to get padded base offsets.
    if (thread_id < E) {
        TileState s;
        do {
            s = load_tile_state(&tile_state[static_cast<int64_t>(grid_size - 1) * E + thread_id]);
        } while (s.flag != TileState::kComplete);
        const int v      = s.value;
        s_acc[thread_id] = v;
        const int padded =
            (pad_multiple > 0) ? ((v + pad_multiple - 1) / pad_multiple) * pad_multiple : v;
        s_tpe_prefix[thread_id] = padded;
        s_num_padded[thread_id] = padded - v;
    }
    __syncthreads();
    {
        const int v = (thread_id < E) ? s_tpe_prefix[thread_id] : 0;
        int       excl;
        BlockScan(scan_temp).ExclusiveSum(v, excl);
        if (thread_id < E) {
            s_tpe_prefix[thread_id] = excl;
        }
    }
    __syncthreads();

    // Phase 6: patch + compact each token's row into the dense
    // [dst | idx | n] layout. One thread per token; in-place write is WAR
    // safe because n <= e during the scan.
    auto patch = [&](int local, int expert) -> int {
        if (local == 0) {
            return 0;
        }
        const int new_val = local + s_excl_prefix[expert] + s_tpe_prefix[expert];
        if (new_val > npt) {
            *overflow_flag = 1;
            return 0;
        }
        return new_val;
    };

    auto compact_row = [&](int gtoken, auto &&read_slot) {
        const int64_t row_base = static_cast<int64_t>(gtoken) * row_stride;
        int           n        = 0;
        for (int e = 0; e < E; ++e) {
            const int s = patch(read_slot(e), e);
            if (s != 0) {
                row_id_map[row_base + n]     = s;
                row_id_map[row_base + E + n] = e;
                ++n;
            }
        }
        row_id_map[row_base + 2 * E] = n;
    };

    if (single_tile) {
        const int tile_offset = tile_begin * kNumItemsPerTile;
        for (int t = thread_id; t < kNumItemsPerTile; t += kNumThreads) {
            const int gtoken = tile_offset + t;
            if (gtoken >= num_dispatched_tokens)
                continue;
            compact_row(gtoken, [&](int e) { return s_tile[t * E + e]; });
        }
    } else {
        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int tile_offset = tile_idx * kNumItemsPerTile;
            for (int t = thread_id; t < kNumItemsPerTile; t += kNumThreads) {
                const int gtoken = tile_offset + t;
                if (gtoken >= num_dispatched_tokens)
                    continue;
                const int64_t row_base = static_cast<int64_t>(gtoken) * row_stride;
                compact_row(gtoken, [&](int e) { return row_id_map[row_base + e]; });
            }
        }
    }

    // Phase 7: padding rows live at row_id_map[N + i, :] and use NEGATIVE
    // 1-indexed offsets, signalling the data-movement kernel to write zeros.
    if (block_id == 0) {
        for (int i = thread_id; i < pad_multiple; i += kNumThreads) {
            const int64_t row_base = (static_cast<int64_t>(num_dispatched_tokens) + i) * row_stride;
            int           n        = 0;
            for (int e = 0; e < E; ++e) {
                if (i >= s_num_padded[e])
                    continue;
                int padded_offset = -(s_acc[e] + s_tpe_prefix[e] + i + 1);
                if (-padded_offset > npt) {
                    *overflow_flag = 1;
                    continue;
                }
                row_id_map[row_base + n]     = padded_offset;
                row_id_map[row_base + E + n] = e;
                ++n;
            }
            row_id_map[row_base + 2 * E] = n;
        }
    }

    // Phase 8: the LAST block finalises tokens_per_expert and zeroes the
    // PREVIOUS launch's tile_state buffer (double-buffer maintenance).
    // Stream serialisation makes the clear visible to the next launch.
    if (block_id == grid_size - 1) {
        if (thread_id < E) {
            const int tokens_for_expert = s_acc[thread_id] + s_num_padded[thread_id];
            const int overflow          = tokens_for_expert + s_tpe_prefix[thread_id] - npt;
            tokens_per_expert[thread_id] =
                (overflow < 0) ? tokens_for_expert : max(0, tokens_for_expert - overflow);
        }

        for (int i = thread_id; i < layout.num_memset_int64; i += kNumThreads)
            layout.prev_tile_state[i] = 0;
    }
}

// Process-lifetime cache for device caps and per-shmem-bytes occupancy.
struct DeviceCaps {
    int dev                 = -1;
    int num_cu              = 0;
    int max_shmem_per_block = 0;
};

inline const DeviceCaps &cached_device_caps() {
    static thread_local DeviceCaps caps;
    int                            dev = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&dev));
    if (caps.dev != dev) {
        PRIMUS_TURBO_CHECK_HIP(
            hipDeviceGetAttribute(&caps.num_cu, hipDeviceAttributeMultiprocessorCount, dev));
        PRIMUS_TURBO_CHECK_HIP(hipDeviceGetAttribute(
            &caps.max_shmem_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, dev));
        caps.dev = dev;
    }
    return caps;
}

// Per-stream double-buffered tile_state + optional vsmem region. Realloc on
// shape change; toggle active buffer each launch so Phase 8's clear of the
// other half is visible to the next launch via stream ordering.
static inline TempStorageLayout get_temp_storage_layout(size_t lookback_bytes,
                                                        size_t vsmem_bytes_per_block,
                                                        size_t grid_size, hipStream_t stream) {
    constexpr size_t kMinBytes = 512 * 1024;

    const size_t buf_bytes   = ALIGN(lookback_bytes, kVsmemCacheLineSize);
    const size_t vsmem_bytes = ALIGN(vsmem_bytes_per_block, kVsmemCacheLineSize);
    const size_t total_bytes = std::max(2 * buf_bytes + vsmem_bytes * grid_size, kMinBytes);

    static std::mutex                                     mu;
    static std::unordered_map<hipStream_t, LookbackCache> cache;
    std::lock_guard<std::mutex>                           lk(mu);

    LookbackCache &c = cache[stream];

    if (c.ptr == nullptr || c.total < total_bytes || c.buf_bytes != buf_bytes) {
        if (c.total < total_bytes) {
            if (c.ptr != nullptr) {
                PRIMUS_TURBO_CHECK_HIP(hipFreeAsync(c.ptr, stream));
            }
            PRIMUS_TURBO_CHECK_HIP(hipMallocAsync(&c.ptr, total_bytes, stream));
            c.total = total_bytes;
        }
        c.buf_bytes  = buf_bytes;
        c.active_idx = 0;
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(c.ptr, 0, 2 * buf_bytes, stream));
    }

    char *const base = static_cast<char *>(c.ptr);
    const int   cur  = c.active_idx;
    const int   nxt  = 1 - cur;

    TempStorageLayout layout{};
    layout.tile_state       = reinterpret_cast<uint64_t *>(base + cur * buf_bytes);
    layout.prev_tile_state  = reinterpret_cast<uint64_t *>(base + nxt * buf_bytes);
    layout.num_memset_int64 = buf_bytes / sizeof(uint64_t);
    // gmem_ptr MUST be nullptr when vsmem isn't requested; the kernel's
    // get_temp_storage() keys off it to choose between LDS and global mem.
    layout.vsmem.gmem_ptr        = (vsmem_bytes_per_block > 0) ? (base + 2 * buf_bytes) : nullptr;
    layout.vsmem.bytes_per_block = vsmem_bytes;

    c.active_idx = nxt;

    return layout;
}

// Shared host-side launch path for both routing_map and topk_idx inputs. The
// only thing that varies is the `expert_map_t` type tag and the `num_topk`
// parameter (ignored in routing_map mode); everything else (scratch sizing,
// lookback layout, grid size) is identical.
template <typename expert_map_t>
void permute_preprocessing_impl(const expert_map_t *expert_map, int num_topk,
                                int *num_dispatched_tokens_ptr, int num_local_experts,
                                int max_num_dispatched_tokens, int pad_multiple,
                                int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                int64_t num_permuted_tokens, hipStream_t stream) {
    constexpr int kNumThreads = 512;
    PRIMUS_TURBO_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
    PRIMUS_TURBO_CHECK(num_local_experts <= kNumThreads,
                       "num_local_experts must fit in a single block");
    PRIMUS_TURBO_CHECK(max_num_dispatched_tokens >= 0, "max_num_dispatched_tokens must be >= 0");

    if (max_num_dispatched_tokens == 0) {
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(
            tokens_per_expert, 0, num_local_experts * sizeof(*tokens_per_expert), stream));
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(overflow_flag, 0, sizeof(*overflow_flag), stream));
        return;
    }

    const auto &caps            = cached_device_caps();
    const int   per_tile        = kNumThreads;
    const int   num_token_tiles = (max_num_dispatched_tokens + per_tile - 1) / per_tile;

    const auto required_temp_storage_bytes = (static_cast<size_t>(kNumThreads) * num_local_experts +
                                              4 * static_cast<size_t>(num_local_experts)) *
                                             sizeof(int);
    // Spill per-block scratch to global memory (vsmem) when LDS is too small.
    const bool use_vsmem =
        required_temp_storage_bytes > static_cast<size_t>(caps.max_shmem_per_block);
    const size_t kernel_lds_bytes       = use_vsmem ? 0 : required_temp_storage_bytes;
    const size_t vshmem_bytes_per_block = use_vsmem ? required_temp_storage_bytes : 0;

    const int grid_size = std::min(num_token_tiles, caps.num_cu);

    const auto lookback_workspace_bytes =
        static_cast<size_t>(grid_size) * num_local_experts * sizeof(uint64_t);

    auto tmp_layout = get_temp_storage_layout(lookback_workspace_bytes, vshmem_bytes_per_block,
                                              grid_size, stream);

    permute_preprocessing_kernel<kNumThreads, expert_map_t>
        <<<grid_size, kNumThreads, kernel_lds_bytes, stream>>>(
            expert_map, num_dispatched_tokens_ptr, num_local_experts, num_topk, pad_multiple,
            tokens_per_expert, row_id_map, overflow_flag, num_permuted_tokens, tmp_layout);

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <int kBlockHiddenPacks, typename prob_t, typename scalar_t>
__launch_bounds__(kBlockHiddenPacks, 4) __global__
    void permute_kernel(const int4 *tokens, int4 *permuted_tokens, const scalar_t *scaling_factor,
                        scalar_t *permuted_scaling_factor, const prob_t *probs,
                        prob_t *permuted_probs, const int *row_id_map,
                        const int *num_dispatched_tokens_ptr, int pad_multiple,
                        int num_local_experts, int hidden_int4, int scales_per_token) {
    const int lane_id  = static_cast<int>(threadIdx.x);
    const int token_id = static_cast<int>(blockIdx.x);
    const int chunk_id = static_cast<int>(blockIdx.y);
    const int j        = chunk_id * kBlockHiddenPacks + lane_id;

    const int E                     = num_local_experts;
    const int row_stride            = 2 * E + 1;
    const int actual_dispatched     = *num_dispatched_tokens_ptr;
    const int num_dispatched_tokens = actual_dispatched + pad_multiple;

    if (token_id >= num_dispatched_tokens) {
        return;
    }

    // Padding rows (token_id >= actual_dispatched) have no backing storage in
    // tokens / scaling_factor / probs - loads there can fault when the
    // allocator places those buffers near unmapped VA. Skip the loads.
    const bool is_padding_token = token_id >= actual_dispatched;

    const int *row      = row_id_map + token_id * row_stride;
    const int  n_routed = row[2 * E];

    // Hidden-chunk fan-out: load source pack once, scatter to every
    // routed destination at the same column.
    if (j < hidden_int4) {
        const int4 zero4    = make_int4(0, 0, 0, 0);
        const int4 src_pack = is_padding_token ? zero4 : __ldg(tokens + token_id * hidden_int4 + j);

        for (int idx = 0; idx < n_routed; ++idx) {
            const int dst_row = row[idx];
            if (dst_row > 0) {
                st_na_global(permuted_tokens + (dst_row - 1) * hidden_int4 + j, src_pack);
            } else {
                st_na_global(permuted_tokens + (-dst_row - 1) * hidden_int4 + j, zero4);
            }
        }
    }

    // Per-(token, dst_row) tail data — emit only from chunk_id == 0.
    if (chunk_id != 0) {
        return;
    }

    if (scaling_factor != nullptr) {
        for (int idx = 0; idx < n_routed; ++idx) {
            const int dst_row = row[idx];
            if (dst_row > 0) {
                for (int sj = lane_id; sj < scales_per_token; sj += kBlockHiddenPacks) {
                    const scalar_t v = is_padding_token
                                           ? scalar_t{0}
                                           : scaling_factor[token_id * scales_per_token + sj];
                    permuted_scaling_factor[(dst_row - 1) * scales_per_token + sj] = v;
                }
            } else {
                for (int sj = lane_id; sj < scales_per_token; sj += kBlockHiddenPacks) {
                    permuted_scaling_factor[(-dst_row - 1) * scales_per_token + sj] = scalar_t{0};
                }
            }
        }
    }

    // probs: [num_dispatched_tokens, E]  (E == num_local_experts).
    // Matches the Triton reference and the multihot probs emitted by
    // indices_to_multihot in the dispatcher.
    if (probs != nullptr && lane_id == 0) {
        for (int idx = 0; idx < n_routed; ++idx) {
            const int dst_row    = row[idx];
            const int expert_idx = row[E + idx];
            if (dst_row > 0) {
                permuted_probs[dst_row - 1] =
                    is_padding_token ? prob_t{0} : probs[token_id * E + expert_idx];
            } else {
                permuted_probs[-dst_row - 1] = prob_t{0};
            }
        }
    }
}

// E=1 specialised gather-copy: 1 warp per token, no float reduction, no
// E-walk. Row layout collapses to 3 ints [dst | idx=0 | n ∈ {0,1}].
template <int kNumThreads, typename dtype_t, typename prob_t>
__global__ void unpermute_kernel_e1(const int4 *permuted_tokens, int4 *tokens,
                                    const prob_t *permuted_probs, prob_t *probs,
                                    const int *row_id_map, const int *num_dispatched_tokens_ptr,
                                    int hidden_int4) {
    constexpr int num_warps  = kNumThreads / kWarpSize;
    constexpr int row_stride = 3;

    const auto thread_id             = static_cast<int>(threadIdx.x);
    const auto lane_id               = thread_id % kWarpSize;
    const auto warp_id               = thread_id / kWarpSize;
    const int  num_dispatched_tokens = *num_dispatched_tokens_ptr;

    for (int64_t block_start = blockIdx.x * num_warps; block_start < num_dispatched_tokens;
         block_start += static_cast<int64_t>(num_warps) * gridDim.x) {
        const int64_t token_id = block_start + warp_id;
        if (token_id >= num_dispatched_tokens)
            continue;

        const int n_routed = row_id_map[token_id * row_stride + 2];
        const int s        = (n_routed > 0) ? row_id_map[token_id * row_stride + 0] : 0;
        int4     *dst      = tokens + token_id * hidden_int4;
        if (s > 0) {
            const int4 *src = permuted_tokens + (s - 1) * hidden_int4;
            UNROLLED_WARP_COPY(4, lane_id, hidden_int4, dst, src, __ldg, st_na_global);
        } else {
            const int4 zero4 = make_int4(0, 0, 0, 0);
            for (int64_t j = lane_id; j < hidden_int4; j += kWarpSize) {
                st_na_global(dst + j, zero4);
            }
        }

        // probs collapses to [T, 1] when E == 1; only lane 0 writes the row.
        if (probs != nullptr && permuted_probs != nullptr && lane_id == 0) {
            probs[token_id] = (s > 0) ? permuted_probs[s - 1] : prob_t{0};
        }
    }
}

template <int kNumThreads, typename dtype_t, typename prob_t>
__launch_bounds__(kNumThreads, 4) __global__
    void unpermute_kernel(const int4 *permuted_tokens, int4 *tokens, const prob_t *permuted_probs,
                          prob_t *probs, const int *row_id_map,
                          const int *num_dispatched_tokens_ptr, int num_local_experts,
                          int hidden_int4) {
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(dtype_t);

    const int lane_id  = static_cast<int>(threadIdx.x);
    const int token_id = static_cast<int>(blockIdx.x);
    const int chunk_id = static_cast<int>(blockIdx.y);
    const int j        = chunk_id * kNumThreads + lane_id;

    const int E          = num_local_experts;
    const int row_stride = 2 * E + 1;

    if (token_id >= *num_dispatched_tokens_ptr) {
        return;
    }

    const int *row      = row_id_map + token_id * row_stride;
    const int  n_routed = row[2 * E];

    if (j < hidden_int4) {
        float acc[num_eles_per_pack];
#pragma unroll
        for (int k = 0; k < num_eles_per_pack; ++k)
            acc[k] = 0.0f;

#pragma unroll 2
        for (int idx = 0; idx < n_routed; ++idx) {
            const int s = row[idx];
            if (s <= 0)
                continue;
            int4           pack = __ldg(permuted_tokens + (s - 1) * hidden_int4 + j);
            const dtype_t *p    = reinterpret_cast<const dtype_t *>(&pack);
#pragma unroll
            for (int k = 0; k < num_eles_per_pack; ++k) {
                acc[k] += static_cast<float>(p[k]);
            }
        }

        int4     out_pack;
        dtype_t *outp = reinterpret_cast<dtype_t *>(&out_pack);
#pragma unroll
        for (int k = 0; k < num_eles_per_pack; ++k) {
            outp[k] = static_cast<dtype_t>(acc[k]);
        }
        st_na_global(tokens + token_id * hidden_int4 + j, out_pack);
    }

    // Probs scatter: only chunk_id == 0 emits the per-token probs row.
    // probs shape is [num_dispatched_tokens, E].
    if (probs != nullptr && permuted_probs != nullptr && chunk_id == 0) {
        for (int p_j = lane_id; p_j < E; p_j += kNumThreads) {
            probs[token_id * E + p_j] = prob_t{0};
        }
        for (int idx = lane_id; idx < n_routed; idx += kNumThreads) {
            const int s = row[idx];
            if (s > 0) {
                const int e             = row[E + idx];
                probs[token_id * E + e] = permuted_probs[s - 1];
            }
        }
    }
}

// =============================================================================
// Host-side permute / unpermute launchers.
// =============================================================================

template <typename dtype_t, typename prob_t, typename scalar_t>
void permute_impl(const dtype_t *tokens, dtype_t *permuted_tokens, const scalar_t *scaling_factor,
                  scalar_t *permuted_scaling_factor, const prob_t *probs, prob_t *permuted_probs,
                  const int *row_id_map, const int *num_dispatched_tokens_ptr, int pad_multiple,
                  int num_local_experts, int hidden_size, int scales_per_token,
                  int num_dispatched_max, hipStream_t stream) {
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(dtype_t);

    PRIMUS_TURBO_CHECK(permuted_tokens != nullptr, "permuted_tokens must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(dtype_t))");
    PRIMUS_TURBO_CHECK(num_dispatched_max > 0, "num_dispatched_max must be > 0");

    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *tokens_int4          = reinterpret_cast<const int4 *>(tokens);
    int4       *permuted_tokens_int4 = reinterpret_cast<int4 *>(permuted_tokens);

#define LAUNCH_PERMUTE(num_hidden_per_block)                                                       \
    do {                                                                                           \
        const int num_chunks =                                                                     \
            (hidden_int4 + (num_hidden_per_block) - 1) / (num_hidden_per_block);                   \
        dim3 grid(static_cast<unsigned int>(num_dispatched_max),                                   \
                  static_cast<unsigned int>(num_chunks));                                          \
        permute_kernel<(num_hidden_per_block), prob_t, scalar_t>                                   \
            <<<grid, (num_hidden_per_block), /*shmem=*/0, stream>>>(                               \
                tokens_int4, permuted_tokens_int4, scaling_factor, permuted_scaling_factor, probs, \
                permuted_probs, row_id_map, num_dispatched_tokens_ptr, pad_multiple,               \
                num_local_experts, hidden_int4, scales_per_token);                                 \
    } while (0)

    DISPATCH_PERMUTE_UNPERMUTE(hidden_size, LAUNCH_PERMUTE);

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
#undef LAUNCH_PERMUTE
}

template <typename dtype_t, typename prob_t>
void unpermute_impl(const dtype_t *permuted_tokens, dtype_t *tokens, const prob_t *permuted_probs,
                    prob_t *probs, const int *row_id_map, const int *num_dispatched_tokens_ptr,
                    int num_local_experts, int hidden_size, int num_dispatched_max,
                    hipStream_t stream) {

    constexpr int kE1NumThreads     = 512;
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(dtype_t);

    PRIMUS_TURBO_CHECK(tokens != nullptr, "tokens output must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(dtype_t))");
    PRIMUS_TURBO_CHECK(num_dispatched_max > 0, "num_dispatched_max must be > 0");

    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *permuted_tokens_int4 = reinterpret_cast<const int4 *>(permuted_tokens);
    int4       *tokens_int4          = reinterpret_cast<int4 *>(tokens);

    if (num_local_experts == 1) {
        // E=1 path keeps the persistent-block design (one warp per token).
        constexpr int num_warps_e1  = kE1NumThreads / kWarpSize;
        const int     blocks_needed = (num_dispatched_max + num_warps_e1 - 1) / num_warps_e1;
        const int     e1_grid       = blocks_needed;
        unpermute_kernel_e1<kE1NumThreads, dtype_t, prob_t>
            <<<e1_grid, kE1NumThreads, /*shmem=*/0, stream>>>(
                permuted_tokens_int4, tokens_int4, permuted_probs, probs, row_id_map,
                num_dispatched_tokens_ptr, hidden_int4);
    } else {
#define LAUNCH_UNPERMUTE(num_hidden_per_block)                                                     \
    do {                                                                                           \
        const int num_chunks =                                                                     \
            (hidden_int4 + (num_hidden_per_block) - 1) / (num_hidden_per_block);                   \
        dim3 grid(static_cast<unsigned int>(num_dispatched_max),                                   \
                  static_cast<unsigned int>(num_chunks));                                          \
        unpermute_kernel<(num_hidden_per_block), dtype_t, prob_t>                                  \
            <<<grid, (num_hidden_per_block), /*shmem=*/0, stream>>>(                               \
                permuted_tokens_int4, tokens_int4, permuted_probs, probs, row_id_map,              \
                num_dispatched_tokens_ptr, num_local_experts, hidden_int4);                        \
    } while (0)

        DISPATCH_PERMUTE_UNPERMUTE(hidden_size, LAUNCH_UNPERMUTE);
#undef LAUNCH_UNPERMUTE
    }

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

// =============================================================================
// Explicit template instantiations consumed by csrc/pytorch/moe_permute/moe_permute.cpp.
// =============================================================================

#define INSTANTIATE_PERMUTE_PREPROCESSING_IMPL(expert_map_t)                                       \
    template void permute_preprocessing_impl<expert_map_t>(                                        \
        const expert_map_t *expert_map, int num_topk, int *num_dispatched_tokens_ptr,              \
        int num_local_experts, int max_num_dispatched_tokens, int pad_multiple,                    \
        int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,                           \
        int64_t num_permuted_tokens, hipStream_t stream)

#define INSTANTIATE_UNPERMUTE_IMPL(dtype_t, prob_t)                                                \
    template void unpermute_impl<dtype_t, prob_t>(const dtype_t *, dtype_t *, const prob_t *,      \
                                                  prob_t *, const int *, const int *, int, int,    \
                                                  int, hipStream_t)
#define INSTANTIATE_PERMUTE_IMPL(dtype_t, prob_t, scalar_t)                                        \
    template void permute_impl<dtype_t, prob_t, scalar_t>(                                         \
        const dtype_t *, dtype_t *, const scalar_t *, scalar_t *, const prob_t *, prob_t *,        \
        const int *, const int *, int, int, int, int, int, hipStream_t)

INSTANTIATE_PERMUTE_PREPROCESSING_IMPL(bool);
INSTANTIATE_PERMUTE_PREPROCESSING_IMPL(int);
INSTANTIATE_PERMUTE_PREPROCESSING_IMPL(int64_t);

INSTANTIATE_PERMUTE_IMPL(uint8_t, float, float);
INSTANTIATE_PERMUTE_IMPL(uint16_t, float, float);

INSTANTIATE_UNPERMUTE_IMPL(bfloat16, float);
INSTANTIATE_UNPERMUTE_IMPL(float16, float);

#undef INSTANTIATE_PERMUTE_PREPROCESSING_IMPL
#undef INSTANTIATE_UNPERMUTE_IMPL
#undef INSTANTIATE_PERMUTE_IMPL

} // namespace primus_turbo
