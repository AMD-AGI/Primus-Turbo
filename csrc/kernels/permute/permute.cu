// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// MoE permute / unpermute kernels.
//   * permute_preprocessing_kernel — single-kernel decoupled-lookback scan
//     over the routing map; produces tokens_per_expert and a dense
//     row_id_map in [dst | idx | n] layout.
//   * permute_kernel / unpermute_kernel — vectorised int4 data movement
//     driven by row_id_map. Falls back to a global-memory tile (vsmem)
//     when per-block scratch exceeds the LDS budget.

#include "permute.cuh"
#include "primus_turbo/permute.h"

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

template <int kBlockSize>
__launch_bounds__(kBlockSize, 1) __global__
    void permute_preprocessing_kernel(const bool *routing_map, const int *num_dispatched_tokens_ptr,
                                      int num_experts, int pad_multiple, int32_t *tokens_per_expert,
                                      int *row_id_map, int *overflow_flag,
                                      int64_t num_permuted_tokens, TempStorageLayout layout) {
    uint64_t *tile_state = layout.tile_state;
    using BlockScan      = hipcub::BlockScan<int32_t, kBlockSize>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    extern __shared__ int                      dyn_shmem[];

    constexpr int kNumItemsPerTile = kBlockSize;

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
    const int internal_rows   = (num_dispatched_tokens + kNumItemsPerTile - 1) / kNumItemsPerTile;
    const int tiles_per_block = (internal_rows + grid_size - 1) / grid_size;
    const int tile_begin      = block_id * tiles_per_block;
    const int tile_end        = min(tile_begin + tiles_per_block, internal_rows);
    // single_tile blocks keep s_tile in LDS through Phase 6 and skip the
    // row_id_map spill / read-back.
    const bool single_tile = (tile_end - tile_begin) == 1;

    const int npt = num_permuted_tokens < 0 ? INT_MAX : static_cast<int>(num_permuted_tokens);

    if (block_id == 0 and thread_id == 0)
        *overflow_flag = 0;

    for (int i = thread_id; i < E; i += kBlockSize)
        s_acc[i] = 0;

    __syncthreads();

    // row_id_map row layout: [dst_rows | expert_idx | n_routed], length 2*E + 1.
    const int row_stride = 2 * E + 1;

    // Phase 1-2: per-tile InclusiveSum, accumulate into s_acc; multi-tile
    // blocks spill the partial scan to row_id_map for Phase 6 to read back.
    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        const int     tile_offset  = tile_idx * kNumItemsPerTile;
        const int64_t routing_base = static_cast<int64_t>(tile_offset) * E;

        for (int i = thread_id; i < kNumItemsPerTile * E; i += kBlockSize) {
            const int gtoken = tile_offset + i / E;
            s_tile[i]        = (gtoken < num_dispatched_tokens)
                                   ? static_cast<int>(routing_map[routing_base + i])
                                   : 0;
        }
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
            for (int i = thread_id; i < kNumItemsPerTile * E; i += kBlockSize) {
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
        for (int t = thread_id; t < kNumItemsPerTile; t += kBlockSize) {
            const int gtoken = tile_offset + t;
            if (gtoken >= num_dispatched_tokens)
                continue;
            compact_row(gtoken, [&](int e) { return s_tile[t * E + e]; });
        }
    } else {
        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int tile_offset = tile_idx * kNumItemsPerTile;
            for (int t = thread_id; t < kNumItemsPerTile; t += kBlockSize) {
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
        for (int i = thread_id; i < pad_multiple; i += kBlockSize) {
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

        for (int i = thread_id; i < layout.num_memset_int64; i += kBlockSize)
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
    constexpr size_t kCacheLine = 128;
    constexpr size_t kMinBytes  = 512 * 1024;

    const size_t buf_bytes   = align_up(lookback_bytes, kCacheLine);
    const size_t vsmem_bytes = align_up(vsmem_bytes_per_block, kCacheLine);
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

void permute_preprocessing_impl(bool *routing_map, int *num_dispatched_tokens_ptr,
                                int num_of_local_experts, int max_num_dispatched_tokens,
                                int pad_multiple, int32_t *tokens_per_expert, int *row_id_map,
                                int *overflow_flag, int64_t num_permuted_tokens,
                                hipStream_t stream) {
    constexpr int kBlockSize = PermutePreprocessConfig::kBlockSize;
    PRIMUS_TURBO_CHECK(num_of_local_experts > 0, "num_of_local_experts must be > 0");
    PRIMUS_TURBO_CHECK(num_of_local_experts <= kBlockSize,
                       "num_of_local_experts must fit in a single block");
    PRIMUS_TURBO_CHECK(max_num_dispatched_tokens > 0, "max_num_dispatched_tokens must be > 0");

    const auto &caps          = cached_device_caps();
    const int   per_tile      = kBlockSize;
    const int   internal_rows = (max_num_dispatched_tokens + per_tile - 1) / per_tile;

    const int grid_size = std::min(MAX_NUM_CU, internal_rows);

    const auto required_temp_storage_bytes =
        (static_cast<size_t>(kBlockSize) * num_of_local_experts +
         4 * static_cast<size_t>(num_of_local_experts)) *
        sizeof(int);
    // Spill per-block scratch to global memory (vsmem) when LDS is too small.
    const bool use_vsmem =
        required_temp_storage_bytes > static_cast<size_t>(caps.max_shmem_per_block);
    const size_t kernel_lds_bytes         = use_vsmem ? 0 : required_temp_storage_bytes;
    const size_t vshmem_bytes_per_block   = use_vsmem ? required_temp_storage_bytes : 0;
    const auto   lookback_workspace_bytes = grid_size * num_of_local_experts * sizeof(uint64_t);

    auto tmp_layout = get_temp_storage_layout(lookback_workspace_bytes, vshmem_bytes_per_block,
                                              grid_size, stream);

    permute_preprocessing_kernel<kBlockSize><<<grid_size, kBlockSize, kernel_lds_bytes, stream>>>(
        routing_map, num_dispatched_tokens_ptr, num_of_local_experts, pad_multiple,
        tokens_per_expert, row_id_map, overflow_flag, num_permuted_tokens, tmp_layout);

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}
// Stage row_id_map rows into LDS for fast random access by all warps.
template <int kBlockSize>
__device__ __forceinline__ void load_routing_tile(int *s_row, const int *row_id_map,
                                                  int64_t block_start, int tokens_per_block,
                                                  int row_stride, int num_dispatched_tokens) {
    const auto thread_id = static_cast<int>(threadIdx.x);
    for (int i = thread_id; i < tokens_per_block * row_stride; i += kBlockSize) {
        const int64_t gt = block_start + i / row_stride;
        s_row[i] = (gt < num_dispatched_tokens) ? row_id_map[gt * row_stride + i % row_stride] : 0;
    }
}

// permute_kernel: gather tokens into expert-grouped order via the dense
// [dst | idx | n] row_id_map. dst > 0 ⇒ gather to slot (dst-1);
// dst < 0 ⇒ write zeros to slot (-dst-1) (padded slot).
template <int kBlockSize, typename ProbType, typename ScalarType>
__global__ void
permute_kernel(const int4 *tokens, int4 *permuted_tokens, const ScalarType *scaling_factor,
               ScalarType *permuted_scaling_factor, const ProbType *probs, ProbType *permuted_probs,
               const int *row_id_map, const int *num_dispatched_tokens_ptr, int pad_multiple,
               int num_of_local_experts, int hidden_int4, int scales_per_token, int local_rank,
               int num_ranks_per_node) {
    constexpr int num_warps = kBlockSize / kWarpSize;

    const auto thread_id             = static_cast<int>(threadIdx.x);
    const auto lane_id               = thread_id % kWarpSize;
    const auto warp_id               = thread_id / kWarpSize;
    const int  E                     = num_of_local_experts;
    const int  row_stride            = 2 * E + 1;
    const int  num_dispatched_tokens = *num_dispatched_tokens_ptr + pad_multiple;

    extern __shared__ int shared_buf[];
    int                  *s_row = shared_buf;

    for (int64_t block_start = blockIdx.x * num_warps; block_start < num_dispatched_tokens;
         block_start += static_cast<int64_t>(num_warps) * gridDim.x) {
        const int64_t token_id = block_start + warp_id;

        load_routing_tile<kBlockSize>(s_row, row_id_map, block_start, num_warps, row_stride,
                                      num_dispatched_tokens);
        __syncthreads();

        if (token_id >= num_dispatched_tokens) {
            __syncthreads();
            continue;
        }

        const int  *row        = s_row + warp_id * row_stride;
        const int   n_routed   = row[2 * E];
        const int4 *src_tokens = tokens + token_id * hidden_int4;

        for (int idx = 0; idx < n_routed; ++idx) {
            const int dst_row = row[idx];
            if (dst_row > 0) {
                int4 *dst = permuted_tokens + (dst_row - 1) * hidden_int4;
                UNROLLED_WARP_COPY(4, lane_id, hidden_int4, dst, src_tokens, __ldg, st_na_global);
            } else {
                int4      *dst   = permuted_tokens + (-dst_row - 1) * hidden_int4;
                const int4 zero4 = make_int4(0, 0, 0, 0);
                for (int64_t j = lane_id; j < hidden_int4; j += kWarpSize) {
                    st_na_global(dst + j, zero4);
                }
            }
        }

        if (scaling_factor != nullptr) {
            for (int idx = 0; idx < n_routed; ++idx) {
                const int dst_row = row[idx];
                if (dst_row > 0) {
                    for (int64_t j = lane_id; j < scales_per_token; j += kWarpSize) {
                        permuted_scaling_factor[(dst_row - 1) * scales_per_token + j] =
                            scaling_factor[token_id * scales_per_token + j];
                    }
                } else {
                    for (int64_t j = lane_id; j < scales_per_token; j += kWarpSize) {
                        permuted_scaling_factor[(-dst_row - 1) * scales_per_token + j] =
                            ScalarType{0};
                    }
                }
            }
        }

        if (probs != nullptr) {
            if (lane_id == 0) {
                for (int idx = 0; idx < n_routed; ++idx) {
                    const int dst_row    = row[idx];
                    const int expert_idx = row[E + idx];
                    if (dst_row > 0) {
                        permuted_probs[dst_row - 1] =
                            probs[token_id * E * num_ranks_per_node + local_rank * E + expert_idx];
                    } else {
                        permuted_probs[-dst_row - 1] = ProbType{0};
                    }
                }
            }
        }

        __syncthreads();
    }
}

static inline constexpr int kUnpermuteWarpsPerToken = 2;

template <int kPackCount, typename DType>
__device__ __forceinline__ void
unpermute_reduce_pack(const int4 *permuted_tokens, int4 *tokens, const int *row, int n_routed,
                      int64_t hidden_int4, int64_t token_id, const int64_t (&js)[kPackCount]) {
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(DType);

    float acc[kPackCount][num_eles_per_pack];
#pragma unroll
    for (int t = 0; t < kPackCount; ++t) {
#pragma unroll
        for (int k = 0; k < num_eles_per_pack; ++k) {
            acc[t][k] = 0.0f;
        }
    }

#pragma unroll 2
    for (int idx = 0; idx < n_routed; ++idx) {
        const int s = row[idx];
        if (s <= 0)
            continue;
        const int src = s - 1;

        int4 packs[kPackCount];
#pragma unroll
        for (int t = 0; t < kPackCount; ++t) {
            packs[t] = permuted_tokens[src * hidden_int4 + js[t]];
        }

#pragma unroll
        for (int t = 0; t < kPackCount; ++t) {
            const DType *p = reinterpret_cast<const DType *>(&packs[t]);
#pragma unroll
            for (int k = 0; k < num_eles_per_pack; ++k) {
                acc[t][k] += static_cast<float>(p[k]);
            }
        }
    }

    int4   buffer_pack;
    DType *buffer_ptr = reinterpret_cast<DType *>(&buffer_pack);
#pragma unroll
    for (int t = 0; t < kPackCount; ++t) {
#pragma unroll
        for (int k = 0; k < num_eles_per_pack; ++k) {
            buffer_ptr[k] = static_cast<DType>(acc[t][k]);
        }
        tokens[token_id * hidden_int4 + js[t]] = buffer_pack;
    }
}

// E=1 specialised gather-copy: 1 warp per token, no float reduction, no
// E-walk. Row layout collapses to 3 ints [dst | idx=0 | n ∈ {0,1}].
template <int kBlockSize, typename DType, typename ProbType>
__global__ void unpermute_kernel_e1(const int4 *permuted_tokens, int4 *tokens,
                                    const ProbType *permuted_probs, ProbType *probs,
                                    const int *row_id_map, const int *num_dispatched_tokens_ptr,
                                    int hidden_int4, int local_rank, int num_ranks_per_node) {
    constexpr int num_warps  = kBlockSize / kWarpSize;
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

        // Probs collapses to [T, num_ranks_per_node]; only probs[t, local_rank]
        // is non-zero (= permuted_probs[s-1] when s > 0).
        if (probs != nullptr && permuted_probs != nullptr) {
            ProbType src_prob_v = ProbType{0};
            if (s > 0)
                src_prob_v = permuted_probs[s - 1];

            for (int64_t j = lane_id; j < num_ranks_per_node; j += kWarpSize) {
                ProbType v = (j == local_rank) ? src_prob_v : ProbType{0};
                probs[token_id * num_ranks_per_node + j] = v;
            }
        }
    }
}

// Generic unpermute (E ≥ 2): kUnpermuteWarpsPerToken warps cooperate per
// token, walking K-routed sources and accumulating in fp32. Outer j-loop
// runs depth-2 (j, j2 = j + cooperative_stride) via unpermute_reduce_pack
// to widen the VMEM fan-out.
template <int kBlockSize, typename DType, typename ProbType>
__global__ void unpermute_kernel(const int4 *permuted_tokens, int4 *tokens,
                                 const ProbType *permuted_probs, ProbType *probs,
                                 const int *row_id_map, const int *num_dispatched_tokens_ptr,
                                 int num_of_local_experts, int hidden_int4, int local_rank,
                                 int num_ranks_per_node) {
    constexpr int kWarpsPerToken       = kUnpermuteWarpsPerToken;
    constexpr int num_warps            = kBlockSize / kWarpSize;
    constexpr int num_tokens_per_block = num_warps / kWarpsPerToken;

    const auto thread_id         = static_cast<int>(threadIdx.x);
    const auto lane_id           = thread_id % kWarpSize;
    const auto warp_id           = thread_id / kWarpSize;
    const auto warp_id_in_token  = warp_id % kWarpsPerToken;
    const auto token_id_in_block = warp_id / kWarpsPerToken;

    const int E                     = num_of_local_experts;
    const int row_stride            = 2 * E + 1;
    const int num_dispatched_tokens = *num_dispatched_tokens_ptr;

    extern __shared__ int shared_buf[];
    int                  *s_row = shared_buf;

    for (int64_t block_start = blockIdx.x * num_tokens_per_block;
         block_start < num_dispatched_tokens;
         block_start += static_cast<int64_t>(num_tokens_per_block) * gridDim.x) {
        const int64_t token_id = block_start + token_id_in_block;

        load_routing_tile<kBlockSize>(s_row, row_id_map, block_start, num_tokens_per_block,
                                      row_stride, num_dispatched_tokens);
        __syncthreads();

        if (token_id >= num_dispatched_tokens) {
            __syncthreads();
            continue;
        }

        const int *row      = s_row + token_id_in_block * row_stride;
        const int  n_routed = row[2 * E];

        constexpr int     kJTile             = 2;
        constexpr int64_t cooperative_stride = kWarpsPerToken * kWarpSize;
        constexpr int64_t step               = kJTile * cooperative_stride;

        for (int64_t j = warp_id_in_token * kWarpSize + lane_id; j < hidden_int4; j += step) {
            const int64_t j2 = j + cooperative_stride;
            if (j2 < hidden_int4) {
                const int64_t js[2] = {j, j2};
                unpermute_reduce_pack<2, DType>(permuted_tokens, tokens, row, n_routed, hidden_int4,
                                                token_id, js);
            } else {
                const int64_t js[1] = {j};
                unpermute_reduce_pack<1, DType>(permuted_tokens, tokens, row, n_routed, hidden_int4,
                                                token_id, js);
            }
        }

        // Probs: per-token output [E * num_ranks_per_node]. One warp per
        // token (warp_id_in_token == 0) zero-fills then splats routed slots.
        if (permuted_probs != nullptr && warp_id_in_token == 0) {
            for (int64_t j = lane_id; j < E * num_ranks_per_node; j += kWarpSize) {
                probs[token_id * E * num_ranks_per_node + j] = ProbType{0};
            }
            for (int idx = lane_id; idx < n_routed; idx += kWarpSize) {
                const int s = row[idx];
                if (s > 0) {
                    const int e = row[E + idx];
                    probs[token_id * E * num_ranks_per_node + local_rank * E + e] =
                        permuted_probs[s - 1];
                }
            }
        }

        __syncthreads();
    }
}

// =============================================================================
// Host-side permute / unpermute launchers.
// =============================================================================

template <typename DType, typename ProbType, typename ScalarType>
void permute_impl(const DType *tokens, DType *permuted_tokens, const ScalarType *scaling_factor,
                  ScalarType *permuted_scaling_factor, const ProbType *probs,
                  ProbType *permuted_probs, const int *row_id_map,
                  const int *num_dispatched_tokens_ptr, int pad_multiple, int num_of_local_experts,
                  int hidden_size, int scales_per_token, int local_rank, int num_ranks_per_node,
                  int grid_size, hipStream_t stream) {
    constexpr int kBlockSize        = PermuteKernelConfig::kBlockSize;
    constexpr int num_warps         = kBlockSize / kWarpSize;
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(DType);

    PRIMUS_TURBO_CHECK(permuted_tokens != nullptr, "permuted_tokens must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(DType))");
    PRIMUS_TURBO_CHECK(grid_size > 0, "grid_size must be > 0");

    const size_t shmem_bytes =
        static_cast<size_t>(2 * num_of_local_experts + 1) * num_warps * sizeof(int);

    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *tokens_int4          = reinterpret_cast<const int4 *>(tokens);
    int4       *permuted_tokens_int4 = reinterpret_cast<int4 *>(permuted_tokens);

    permute_kernel<kBlockSize, ProbType, ScalarType>
        <<<grid_size, kBlockSize, shmem_bytes, stream>>>(
            tokens_int4, permuted_tokens_int4, scaling_factor, permuted_scaling_factor, probs,
            permuted_probs, row_id_map, num_dispatched_tokens_ptr, pad_multiple,
            num_of_local_experts, hidden_int4, scales_per_token, local_rank, num_ranks_per_node);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <typename DType, typename ProbType>
void unpermute_impl(const DType *permuted_tokens, DType *tokens, const ProbType *permuted_probs,
                    ProbType *probs, const int *row_id_map, const int *num_dispatched_tokens_ptr,
                    int num_of_local_experts, int hidden_size, int local_rank,
                    int num_ranks_per_node, int grid_size, hipStream_t stream) {
    // E=1 keeps kBlockSize=512 to mirror permute_kernel; the generic K-reduction
    // path uses kBlockSize=1024 to widen per-CU vmem fan-out.
    constexpr int kBlockSize           = PermuteKernelConfig::kBlockSize;
    constexpr int kUnpermuteBlockSize  = 1024;
    constexpr int num_warps            = kUnpermuteBlockSize / kWarpSize;
    constexpr int num_tokens_per_block = num_warps / kUnpermuteWarpsPerToken;
    constexpr int num_eles_per_pack    = sizeof(int4) / sizeof(DType);

    PRIMUS_TURBO_CHECK(tokens != nullptr, "tokens output must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(DType))");
    PRIMUS_TURBO_CHECK(grid_size > 0, "grid_size must be > 0");

    const size_t shmem_bytes =
        static_cast<size_t>(2 * num_of_local_experts + 1) * num_tokens_per_block * sizeof(int);

    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *permuted_tokens_int4 = reinterpret_cast<const int4 *>(permuted_tokens);
    int4       *tokens_int4          = reinterpret_cast<int4 *>(tokens);

    if (num_of_local_experts == 1) {
        unpermute_kernel_e1<kBlockSize, DType, ProbType>
            <<<grid_size, kBlockSize, /*shmem=*/0, stream>>>(
                permuted_tokens_int4, tokens_int4, permuted_probs, probs, row_id_map,
                num_dispatched_tokens_ptr, hidden_int4, local_rank, num_ranks_per_node);
    } else {
        unpermute_kernel<kUnpermuteBlockSize, DType, ProbType>
            <<<grid_size, kUnpermuteBlockSize, shmem_bytes, stream>>>(
                permuted_tokens_int4, tokens_int4, permuted_probs, probs, row_id_map,
                num_dispatched_tokens_ptr, num_of_local_experts, hidden_int4, local_rank,
                num_ranks_per_node);
    }

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

// =============================================================================
// Explicit template instantiations consumed by csrc/pytorch/permute/permute.cpp.
// =============================================================================

template void permute_impl<uint8_t, float, float>(const uint8_t *, uint8_t *, const float *,
                                                  float *, const float *, float *, const int *,
                                                  const int *, int, int, int, int, int, int, int,
                                                  hipStream_t);
template void permute_impl<uint16_t, float, float>(const uint16_t *, uint16_t *, const float *,
                                                   float *, const float *, float *, const int *,
                                                   const int *, int, int, int, int, int, int, int,
                                                   hipStream_t);
template void unpermute_impl<bfloat16, float>(const bfloat16 *, bfloat16 *, const float *, float *,
                                              const int *, const int *, int, int, int, int, int,
                                              hipStream_t);

} // namespace primus_turbo
