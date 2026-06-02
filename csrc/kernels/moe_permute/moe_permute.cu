// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "moe_permute.cuh"
#include "primus_turbo/arch.h"
#include "primus_turbo/common.h"
#include "primus_turbo/moe_permute.h"

#include <hip/hip_runtime.h>
#include <hipcub/block/block_reduce.hpp>
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

// Nontemporal stores: lower to `global_store_dwordx4 ... slc:1` on gfx950, bypassing L2.
typedef int int4_v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void st_nt_int4(int4 *ptr, const int4 &val) {
    int4_v v = {val.x, val.y, val.z, val.w};
    __builtin_nontemporal_store(v, reinterpret_cast<int4_v *>(ptr));
}

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

    for (int i = thread_id; i < kNumItemsPerTile * E; i += kNumThreads)
        s_tile[i] = 0;
    __syncthreads();

    for (int i = thread_id; i < kNumItemsPerTile * num_topk; i += kNumThreads) {
        const int t = i / num_topk, k = i % num_topk;
        const int gtoken = tile_offset + t;
        if (gtoken >= num_dispatched_tokens)
            continue;
        const int e = topk_idx[static_cast<int64_t>(gtoken) * num_topk + k];
        if (e >= 0 && e < E)
            s_tile[t * E + e] = 1;
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
    void permute_preprocessing_kernel(const expert_map_t *expert_map, int max_num_dispatched_tokens,
                                      int *num_dispatched_tokens_out, int num_experts, int num_topk,
                                      int pad_multiple, int64_t *tokens_per_expert, int *row_id_map,
                                      int *overflow_flag, int64_t num_permuted_tokens,
                                      int probs_topk_stride, TempStorageLayout layout) {
    using BlockScan   = hipcub::BlockScan<int32_t, kNumThreads>;
    using BlockReduce = hipcub::BlockReduce<int32_t, kNumThreads>;
    // Scan and reduce are never live simultaneously; alias to save LDS.
    union SharedTemp {
        typename BlockScan::TempStorage   scan;
        typename BlockReduce::TempStorage reduce;
    };
    __shared__ SharedTemp shared_temp;

    __shared__ int s_block_dispatched;
    __shared__ int s_total_dispatched;

    extern __shared__ int dyn_shmem[];

    constexpr int kNumItemsPerTile = kNumThreads;

    const int thread_id         = static_cast<int>(threadIdx.x);
    const int block_id          = static_cast<int>(blockIdx.x);
    const int grid_size         = static_cast<int>(gridDim.x);
    const int E                 = num_experts;
    const int row_stride        = 2 * E + 1;
    const int tile_state_stride = E + 1;

    uint64_t *tile_state = layout.tile_state;

    int *temp_storage  = get_temp_storage<int>(dyn_shmem, layout.vsmem);
    int *s_tile        = temp_storage;
    int *s_acc         = s_tile + kNumItemsPerTile * E;
    int *s_excl_prefix = s_acc + E;
    int *s_tpe_prefix  = s_excl_prefix + E;
    int *s_num_padded  = s_tpe_prefix + E;

    const int N               = max_num_dispatched_tokens;
    const int num_token_tiles = (N + kNumItemsPerTile - 1) / kNumItemsPerTile;
    const int tiles_per_block = (num_token_tiles + grid_size - 1) / grid_size;
    const int tile_begin      = block_id * tiles_per_block;
    const int tile_end        = min(tile_begin + tiles_per_block, num_token_tiles);
    // Single-tile blocks keep s_tile in LDS and skip row_id_map spill/read-back.
    const bool single_tile = (tile_end - tile_begin) == 1;

    const int npt = num_permuted_tokens <= 0 ? INT_MAX : static_cast<int>(num_permuted_tokens);

    if (block_id == 0 && thread_id == 0)
        *overflow_flag = 0;
    if (thread_id == 0) {
        s_block_dispatched = 0;
        s_total_dispatched = 0;
    }
    for (int i = thread_id; i < E; i += kNumThreads)
        s_acc[i] = 0;
    __syncthreads();

    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * kNumItemsPerTile;

        fill_s_tile<kNumThreads, expert_map_t>(s_tile, expert_map, tile_offset, E, num_topk,
                                               kNumItemsPerTile, N);
        __syncthreads();

        // Count real rows before the per-expert scan overwrites s_tile.
        {
            int       local_real = 0;
            const int t = thread_id, gtoken = tile_offset + t;
            if (gtoken < N) {
                for (int e = 0; e < E; ++e) {
                    if (s_tile[t * E + e]) {
                        local_real = 1;
                        break;
                    }
                }
            }
            const int tile_total = BlockReduce(shared_temp.reduce).Sum(local_real);
            if (thread_id == 0)
                s_block_dispatched += tile_total;
            __syncthreads(); // reduce → scan union reuse
        }

        // Rewrite mask to 1-based slot index per expert.
        for (int e = 0; e < E; ++e) {
            const int local = s_tile[thread_id * E + e];
            int       excl_block, scan_total;
            BlockScan(shared_temp.scan).ExclusiveSum(local, excl_block, scan_total);
            const int prev            = s_acc[e];
            s_tile[thread_id * E + e] = (local == 1) ? (excl_block + prev + 1) : 0;
            if (thread_id == 0)
                s_acc[e] += scan_total;
            __syncthreads();
        }

        if (!single_tile) {
            for (int i = thread_id; i < kNumItemsPerTile * E; i += kNumThreads) {
                const int gtoken = tile_offset + i / E;
                const int e      = i % E;
                if (gtoken < N)
                    row_id_map[static_cast<int64_t>(gtoken) * row_stride + e] = s_tile[i];
            }
            __syncthreads();
        }
    }

    // ----- decoupled lookback over (E + 1) cols -----
    if (thread_id < E) {
        s_excl_prefix[thread_id] = decoupled_lookback(tile_state, block_id, tile_state_stride,
                                                      thread_id, s_acc[thread_id]);
    } else if (thread_id == E) {
        const int32_t agg  = s_block_dispatched;
        const int32_t excl = decoupled_lookback(tile_state, block_id, tile_state_stride, E, agg);
        s_total_dispatched = excl + agg; // last block's value == grid-wide total
    }
    __syncthreads();

    if (thread_id < E) {
        TileState s;
        do {
            s = load_tile_state(
                &tile_state[static_cast<int64_t>(grid_size - 1) * tile_state_stride + thread_id]);
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
        BlockScan(shared_temp.scan).ExclusiveSum(v, excl);
        if (thread_id < E)
            s_tpe_prefix[thread_id] = excl;
    }
    __syncthreads();

    // One thread per token; in-place compact write is WAR-safe because n <= e.
    auto patch = [&](int local, int expert) -> int {
        if (local == 0)
            return 0;
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
            if (s == 0)
                continue;
            row_id_map[row_base + n] = s;
            int aux_val              = e;
            // Topk-aligned probs: aux is the k with topk_idx[gtoken,k]==e.
            if constexpr (!std::is_same_v<expert_map_t, bool>) {
                if (probs_topk_stride > 0) {
                    for (int k = 0; k < num_topk; ++k) {
                        const int ek = static_cast<int>(
                            expert_map[static_cast<int64_t>(gtoken) * num_topk + k]);
                        if (ek == e) {
                            aux_val = k;
                            break;
                        }
                    }
                }
            }
            row_id_map[row_base + E + n] = aux_val;
            ++n;
        }
        row_id_map[row_base + 2 * E] = n;
    };

    if (single_tile) {
        const int tile_offset = tile_begin * kNumItemsPerTile;
        for (int t = thread_id; t < kNumItemsPerTile; t += kNumThreads) {
            const int gtoken = tile_offset + t;
            if (gtoken >= N)
                continue;
            compact_row(gtoken, [&](int e) { return s_tile[t * E + e]; });
        }
    } else {
        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int tile_offset = tile_idx * kNumItemsPerTile;
            for (int t = thread_id; t < kNumItemsPerTile; t += kNumThreads) {
                const int gtoken = tile_offset + t;
                if (gtoken >= N)
                    continue;
                const int64_t row_base = static_cast<int64_t>(gtoken) * row_stride;
                compact_row(gtoken, [&](int e) { return row_id_map[row_base + e]; });
            }
        }
    }

    // Padding rows use negative 1-indexed offsets.
    if (block_id == 0) {
        for (int i = thread_id; i < pad_multiple; i += kNumThreads) {
            const int64_t row_base = (static_cast<int64_t>(N) + i) * row_stride;
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

    if (block_id == grid_size - 1) {
        if (thread_id < E) {
            const int tokens_for_expert  = s_acc[thread_id] + s_num_padded[thread_id];
            const int overflow           = tokens_for_expert + s_tpe_prefix[thread_id] - npt;
            tokens_per_expert[thread_id] = static_cast<int64_t>(
                (overflow < 0) ? tokens_for_expert : max(0, tokens_for_expert - overflow));
        }
        if (thread_id == 0)
            *num_dispatched_tokens_out = s_total_dispatched;
        for (int i = thread_id; i < layout.num_memset_int64; i += kNumThreads)
            layout.prev_tile_state[i] = 0;
    }
}

// Per-stream double-buffered tile_state cache + optional vsmem region.
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

    const bool grow = c.total < total_bytes;
    if (grow || c.buf_bytes != buf_bytes) {
        if (grow) {
            if (c.ptr != nullptr)
                PRIMUS_TURBO_CHECK_HIP(hipFreeAsync(c.ptr, stream));
            PRIMUS_TURBO_CHECK_HIP(hipMallocAsync(&c.ptr, total_bytes, stream));
            c.total = total_bytes;
        }
        c.buf_bytes  = buf_bytes;
        c.active_idx = 0;
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(c.ptr, 0, 2 * buf_bytes, stream));
    }

    char *const base = static_cast<char *>(c.ptr);
    const int   cur = c.active_idx, nxt = 1 - cur;

    TempStorageLayout layout{};
    layout.tile_state       = reinterpret_cast<uint64_t *>(base + cur * buf_bytes);
    layout.prev_tile_state  = reinterpret_cast<uint64_t *>(base + nxt * buf_bytes);
    layout.num_memset_int64 = buf_bytes / sizeof(uint64_t);
    // Null gmem_ptr signals the kernel to use LDS instead of vsmem.
    layout.vsmem.gmem_ptr        = (vsmem_bytes_per_block > 0) ? (base + 2 * buf_bytes) : nullptr;
    layout.vsmem.bytes_per_block = vsmem_bytes;

    c.active_idx = nxt;
    return layout;
}

template <typename expert_map_t>
void permute_preprocessing_impl(const expert_map_t *expert_map, int num_topk,
                                int *num_dispatched_tokens_out, int num_local_experts,
                                int max_num_dispatched_tokens, int pad_multiple,
                                int64_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                int64_t num_permuted_tokens, int probs_topk_stride,
                                hipStream_t stream) {
    constexpr int kNumThreads = 512;
    PRIMUS_TURBO_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
    // Strict ``<``: thread E owns the dispatched-count lookback column.
    PRIMUS_TURBO_CHECK(num_local_experts < kNumThreads, "num_local_experts must be < kNumThreads");
    PRIMUS_TURBO_CHECK(max_num_dispatched_tokens >= 0, "max_num_dispatched_tokens must be >= 0");

    if (max_num_dispatched_tokens == 0) {
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(
            tokens_per_expert, 0, num_local_experts * sizeof(*tokens_per_expert), stream));
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(overflow_flag, 0, sizeof(*overflow_flag), stream));
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(num_dispatched_tokens_out, 0,
                                              sizeof(*num_dispatched_tokens_out), stream));
        return;
    }

    int device_id = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    const int        num_cu              = get_multi_processor_count(device_id);
    static const int max_shmem_per_block = get_max_shmem_per_block(device_id);

    const int num_token_tiles = (max_num_dispatched_tokens + kNumThreads - 1) / kNumThreads;

    // (T + 4) * E ints: s_tile (T*E) + s_acc + s_excl_prefix + s_tpe_prefix + s_num_padded.
    const size_t required_temp_storage_bytes =
        static_cast<size_t>(num_local_experts) * (kNumThreads + 4) * sizeof(int);
    // Spill to vsmem when LDS is too small.
    const bool   use_vsmem = required_temp_storage_bytes > static_cast<size_t>(max_shmem_per_block);
    const size_t kernel_lds_bytes       = use_vsmem ? 0 : required_temp_storage_bytes;
    const size_t vshmem_bytes_per_block = use_vsmem ? required_temp_storage_bytes : 0;

    const int grid_size = std::min(num_token_tiles, num_cu);

    // (E + 1) lookback cols per block: [0, E) per-expert + col E for dispatched count.
    const size_t lookback_workspace_bytes =
        static_cast<size_t>(grid_size) * (num_local_experts + 1) * sizeof(uint64_t);

    auto tmp_layout = get_temp_storage_layout(lookback_workspace_bytes, vshmem_bytes_per_block,
                                              grid_size, stream);

    permute_preprocessing_kernel<kNumThreads, expert_map_t>
        <<<grid_size, kNumThreads, kernel_lds_bytes, stream>>>(
            expert_map, max_num_dispatched_tokens, num_dispatched_tokens_out, num_local_experts,
            num_topk, pad_multiple, tokens_per_expert, row_id_map, overflow_flag,
            num_permuted_tokens, probs_topk_stride, tmp_layout);

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <int kBlockHiddenPacks, int kNumChunks, typename prob_t, typename scalar_t>
__launch_bounds__(kBlockHiddenPacks, 4) __global__
    void permute_kernel(const int4 *tokens, int4 *permuted_tokens, const scalar_t *scaling_factor,
                        scalar_t *permuted_scaling_factor, const prob_t *probs,
                        prob_t *permuted_probs, const int *row_id_map,
                        const int *num_dispatched_tokens_ptr, int pad_multiple,
                        int num_local_experts, int hidden_int4, int scales_per_token,
                        int probs_stride) {
    const int lane_id = static_cast<int>(threadIdx.x);
    // kNumChunks == 1: collapsed form (1 block/token, strided inner loop).
    // kNumChunks >= 2: chunked form (1 block/(token,chunk), single iter/thread).
    int token_id, chunk_id;
    if constexpr (kNumChunks == 1) {
        token_id = static_cast<int>(blockIdx.x);
        chunk_id = 0;
    } else {
        token_id = static_cast<int>(blockIdx.x) / kNumChunks;
        chunk_id = static_cast<int>(blockIdx.x) - token_id * kNumChunks;
    }

    const int E                     = num_local_experts;
    const int row_stride            = 2 * E + 1;
    const int actual_dispatched     = *num_dispatched_tokens_ptr;
    const int num_dispatched_tokens = actual_dispatched + pad_multiple;

    if (token_id >= num_dispatched_tokens)
        return;

    const bool is_padding_token = token_id >= actual_dispatched;

    const int *row      = row_id_map + token_id * row_stride;
    const int  n_routed = row[2 * E];

    const int4 zero4        = make_int4(0, 0, 0, 0);
    auto       scatter_pack = [&](int j) {
        const int4 src_pack = is_padding_token ? zero4 : __ldg(tokens + token_id * hidden_int4 + j);
        for (int idx = 0; idx < n_routed; ++idx) {
            const int dst_row = row[idx];
            if (dst_row > 0)
                st_nt_int4(permuted_tokens + (dst_row - 1) * hidden_int4 + j, src_pack);
            else
                st_nt_int4(permuted_tokens + (-dst_row - 1) * hidden_int4 + j, zero4);
        }
    };
    if constexpr (kNumChunks == 1) {
#pragma unroll
        for (int j = lane_id; j < hidden_int4; j += kBlockHiddenPacks)
            scatter_pack(j);
    } else {
        const int j = chunk_id * kBlockHiddenPacks + lane_id;
        if (j < hidden_int4)
            scatter_pack(j);
    }

    // scaling_factor / probs are per-token; only chunk_id==0 emits them.
    if constexpr (kNumChunks > 1) {
        if (chunk_id != 0)
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
                for (int sj = lane_id; sj < scales_per_token; sj += kBlockHiddenPacks)
                    permuted_scaling_factor[(-dst_row - 1) * scales_per_token + sj] = scalar_t{0};
            }
        }
    }

    if (probs != nullptr && lane_id == 0) {
        for (int idx = 0; idx < n_routed; ++idx) {
            const int dst_row   = row[idx];
            const int probs_col = row[E + idx];
            if (dst_row > 0)
                permuted_probs[dst_row - 1] =
                    is_padding_token ? prob_t{0} : probs[token_id * probs_stride + probs_col];
            else
                permuted_probs[-dst_row - 1] = prob_t{0};
        }
    }
}

template <int kNumThreads, typename dtype_t, typename prob_t>
__global__ void unpermute_kernel_e1(const int4 *permuted_tokens, int4 *tokens,
                                    const prob_t *permuted_probs, prob_t *probs,
                                    const int *row_id_map, const int *num_dispatched_tokens_ptr,
                                    int hidden_int4) {
    constexpr int num_warps  = kNumThreads / kWarpSize;
    constexpr int row_stride = 3;

    const int thread_id             = static_cast<int>(threadIdx.x);
    const int lane_id               = thread_id % kWarpSize;
    const int warp_id               = thread_id / kWarpSize;
    const int num_dispatched_tokens = *num_dispatched_tokens_ptr;

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
            UNROLLED_WARP_COPY(4, lane_id, hidden_int4, dst, src, __ldg, st_nt_int4);
        } else {
            const int4 zero4 = make_int4(0, 0, 0, 0);
            for (int64_t j = lane_id; j < hidden_int4; j += kWarpSize)
                st_nt_int4(dst + j, zero4);
        }

        if (probs != nullptr && permuted_probs != nullptr && lane_id == 0)
            probs[token_id] = (s > 0) ? permuted_probs[s - 1] : prob_t{0};
    }
}

template <int kNumThreads, int kNumChunks, typename dtype_t, typename prob_t>
__launch_bounds__(kNumThreads, 4) __global__
    void unpermute_kernel(const int4 *permuted_tokens, int4 *tokens, const prob_t *permuted_probs,
                          prob_t *probs, const int *row_id_map,
                          const int *num_dispatched_tokens_ptr, int num_local_experts,
                          int hidden_int4, int probs_stride) {
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(dtype_t);

    const int lane_id = static_cast<int>(threadIdx.x);
    // kNumChunks∈[1,4]: 1-D dispatch (avoids gfx950 2-D walker overhead); else 2-D fallback.
    int token_id, chunk_id;
    if constexpr (kNumChunks >= 1 && kNumChunks <= 4) {
        token_id = static_cast<int>(blockIdx.x) / kNumChunks;
        chunk_id = static_cast<int>(blockIdx.x) % kNumChunks;
    } else {
        token_id = static_cast<int>(blockIdx.x);
        chunk_id = static_cast<int>(blockIdx.y);
    }
    const int j = chunk_id * kNumThreads + lane_id;

    const int E          = num_local_experts;
    const int row_stride = 2 * E + 1;

    if (token_id >= *num_dispatched_tokens_ptr)
        return;

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
            for (int k = 0; k < num_eles_per_pack; ++k)
                acc[k] += static_cast<float>(p[k]);
        }

        int4     out_pack;
        dtype_t *outp = reinterpret_cast<dtype_t *>(&out_pack);
#pragma unroll
        for (int k = 0; k < num_eles_per_pack; ++k)
            outp[k] = static_cast<dtype_t>(acc[k]);
        st_nt_int4(tokens + token_id * hidden_int4 + j, out_pack);
    }

    if (probs != nullptr && permuted_probs != nullptr && chunk_id == 0) {
        for (int p_j = lane_id; p_j < probs_stride; p_j += kNumThreads)
            probs[token_id * probs_stride + p_j] = prob_t{0};
        for (int idx = lane_id; idx < n_routed; idx += kNumThreads) {
            const int s = row[idx];
            if (s > 0) {
                const int probs_col                        = row[E + idx];
                probs[token_id * probs_stride + probs_col] = permuted_probs[s - 1];
            }
        }
    }
}

template <typename dtype_t, typename prob_t, typename scalar_t>
void permute_impl(const dtype_t *tokens, dtype_t *permuted_tokens, const scalar_t *scaling_factor,
                  scalar_t *permuted_scaling_factor, const prob_t *probs, prob_t *permuted_probs,
                  const int *row_id_map, const int *num_dispatched_tokens_ptr, int pad_multiple,
                  int num_local_experts, int hidden_size, int scales_per_token,
                  int num_dispatched_max, int probs_stride, hipStream_t stream) {
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(dtype_t);

    PRIMUS_TURBO_CHECK(permuted_tokens != nullptr, "permuted_tokens must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(dtype_t))");
    PRIMUS_TURBO_CHECK(num_dispatched_max > 0, "num_dispatched_max must be > 0");

    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *tokens_int4          = reinterpret_cast<const int4 *>(tokens);
    int4       *permuted_tokens_int4 = reinterpret_cast<int4 *>(permuted_tokens);

    const int effective_probs_stride = probs_stride > 0 ? probs_stride : num_local_experts;

#define LAUNCH_PERMUTE_NC(num_hidden_per_block, NC, grid)                                          \
    permute_kernel<(num_hidden_per_block), (NC), prob_t, scalar_t>                                 \
        <<<grid, (num_hidden_per_block), /*shmem=*/0, stream>>>(                                   \
            tokens_int4, permuted_tokens_int4, scaling_factor, permuted_scaling_factor, probs,     \
            permuted_probs, row_id_map, num_dispatched_tokens_ptr, pad_multiple,                   \
            num_local_experts, hidden_int4, scales_per_token, effective_probs_stride)

#define LAUNCH_PERMUTE(num_hidden_per_block)                                                       \
    do {                                                                                           \
        /* hidden_int4 == 2*B uses the 2-chunk form; everything else collapses to 1. */            \
        const bool use_2chunk = (hidden_int4 == 2 * (num_hidden_per_block));                       \
        if (use_2chunk) {                                                                          \
            dim3 grid(static_cast<unsigned int>(num_dispatched_max) * 2u);                         \
            LAUNCH_PERMUTE_NC(num_hidden_per_block, 2, grid);                                      \
        } else {                                                                                   \
            dim3 grid(static_cast<unsigned int>(num_dispatched_max));                              \
            LAUNCH_PERMUTE_NC(num_hidden_per_block, 1, grid);                                      \
        }                                                                                          \
    } while (0)

    DISPATCH_PERMUTE_UNPERMUTE(hidden_size, LAUNCH_PERMUTE);

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
#undef LAUNCH_PERMUTE
}

template <typename dtype_t, typename prob_t>
void unpermute_impl(const dtype_t *permuted_tokens, dtype_t *tokens, const prob_t *permuted_probs,
                    prob_t *probs, const int *row_id_map, const int *num_dispatched_tokens_ptr,
                    int num_local_experts, int hidden_size, int num_dispatched_max,
                    int probs_stride, hipStream_t stream) {

    constexpr int kE1NumThreads     = 512;
    constexpr int num_eles_per_pack = sizeof(int4) / sizeof(dtype_t);

    PRIMUS_TURBO_CHECK(tokens != nullptr, "tokens output must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(dtype_t))");
    PRIMUS_TURBO_CHECK(num_dispatched_max > 0, "num_dispatched_max must be > 0");

    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *permuted_tokens_int4 = reinterpret_cast<const int4 *>(permuted_tokens);
    int4       *tokens_int4          = reinterpret_cast<int4 *>(tokens);

    const int effective_probs_stride = probs_stride > 0 ? probs_stride : num_local_experts;

    // unpermute_kernel_e1 hard-codes probs row width 1 (multihot only); topk path uses num_topk at
    // E=1.
    if (num_local_experts == 1 && effective_probs_stride == 1) {
        constexpr int num_warps_e1  = kE1NumThreads / kWarpSize;
        const int     blocks_needed = (num_dispatched_max + num_warps_e1 - 1) / num_warps_e1;
        const int     e1_grid       = blocks_needed;
        unpermute_kernel_e1<kE1NumThreads, dtype_t, prob_t>
            <<<e1_grid, kE1NumThreads, /*shmem=*/0, stream>>>(
                permuted_tokens_int4, tokens_int4, permuted_probs, probs, row_id_map,
                num_dispatched_tokens_ptr, hidden_int4);
    } else {
#define LAUNCH_UNPERMUTE_NC(num_hidden_per_block, NC, grid)                                        \
    unpermute_kernel<(num_hidden_per_block), (NC), dtype_t, prob_t>                                \
        <<<grid, (num_hidden_per_block), /*shmem=*/0, stream>>>(                                   \
            permuted_tokens_int4, tokens_int4, permuted_probs, probs, row_id_map,                  \
            num_dispatched_tokens_ptr, num_local_experts, hidden_int4, effective_probs_stride)

#define LAUNCH_UNPERMUTE(num_hidden_per_block)                                                     \
    do {                                                                                           \
        const int num_chunks =                                                                     \
            (hidden_int4 + (num_hidden_per_block) - 1) / (num_hidden_per_block);                   \
        if (num_chunks == 1) {                                                                     \
            dim3 grid(static_cast<unsigned int>(num_dispatched_max), 1u);                          \
            LAUNCH_UNPERMUTE_NC(num_hidden_per_block, 1, grid);                                    \
        } else if (num_chunks == 2) {                                                              \
            dim3 grid(static_cast<unsigned int>(num_dispatched_max) * 2u);                         \
            LAUNCH_UNPERMUTE_NC(num_hidden_per_block, 2, grid);                                    \
        } else if (num_chunks == 3) {                                                              \
            dim3 grid(static_cast<unsigned int>(num_dispatched_max) * 3u);                         \
            LAUNCH_UNPERMUTE_NC(num_hidden_per_block, 3, grid);                                    \
        } else if (num_chunks == 4) {                                                              \
            dim3 grid(static_cast<unsigned int>(num_dispatched_max) * 4u);                         \
            LAUNCH_UNPERMUTE_NC(num_hidden_per_block, 4, grid);                                    \
        } else {                                                                                   \
            dim3 grid(static_cast<unsigned int>(num_dispatched_max),                               \
                      static_cast<unsigned int>(num_chunks));                                      \
            LAUNCH_UNPERMUTE_NC(num_hidden_per_block, 0, grid);                                    \
        }                                                                                          \
    } while (0)

        DISPATCH_PERMUTE_UNPERMUTE(hidden_size, LAUNCH_UNPERMUTE);
#undef LAUNCH_UNPERMUTE
#undef LAUNCH_UNPERMUTE_NC
    }

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

// Explicit template instantiations consumed by csrc/pytorch/moe_permute/moe_permute.cpp.
#define INSTANTIATE(fn, ...) template decltype(fn<__VA_ARGS__>) fn<__VA_ARGS__>

INSTANTIATE(permute_preprocessing_impl, bool);
INSTANTIATE(permute_preprocessing_impl, int);
INSTANTIATE(permute_preprocessing_impl, int64_t);

INSTANTIATE(permute_impl, uint8_t, float, float);
INSTANTIATE(permute_impl, uint16_t, float, float);

INSTANTIATE(unpermute_impl, bfloat16, float);
INSTANTIATE(unpermute_impl, float16, float);

#undef INSTANTIATE

} // namespace primus_turbo
