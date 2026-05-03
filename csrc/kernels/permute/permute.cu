#include "primus_turbo/permute.h"

#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace primus_turbo {

template <int kNumThreads, int kTPT>
__launch_bounds__(kNumThreads, 1) __global__
    void permute_preprocessing_kernel(bool *routing_map, int *num_dispatched_tokens_ptr,
                                      int num_of_local_experts, int rows_workspace_1,
                                      int pad_multiple, int32_t *tokens_per_expert, int *row_id_map,
                                      int *overflow_flag, int64_t num_permuted_tokens,

                                      uint64_t *tile_state, int *barrier_counter_p4,
                                      int *barrier_counter_p5) {

    using BlockScan = hipcub::BlockScan<int32_t, kNumThreads>;
    __shared__ typename BlockScan::TempStorage scan_temp;

    extern __shared__ int dyn_shmem[];

    constexpr int kTokensPerTile = kNumThreads * kTPT;
    const int     E              = num_of_local_experts;

    // Dyn-shmem layout
    int *s_tile        = dyn_shmem;
    int *s_acc         = s_tile + kTokensPerTile * E;
    int *s_excl_prefix = s_acc + E;
    int *s_tpe_prefix  = s_excl_prefix + E;
    int *s_num_padded  = s_tpe_prefix + E;

    const int num_dispatched_tokens = *num_dispatched_tokens_ptr;

    const int internal_rows_w1 = (num_dispatched_tokens + kTokensPerTile - 1) / kTokensPerTile;
    (void) rows_workspace_1;

    const int tiles_per_block = (internal_rows_w1 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int my_tile_start   = (int) blockIdx.x * tiles_per_block;
    const int my_tile_end_raw = my_tile_start + tiles_per_block;
    const int my_tile_end =
        (my_tile_end_raw < internal_rows_w1) ? my_tile_end_raw : internal_rows_w1;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *overflow_flag = 0;
    }
    const int npt = (num_permuted_tokens < 0) ? INT_MAX : (int) num_permuted_tokens;

    // Init s_acc to 0 so the per-tile loop can accumulate sums across tiles.
    for (int i = (int) threadIdx.x; i < E; i += kNumThreads) {
        s_acc[i] = 0;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 1-2: per-tile in-block InclusiveSum + accumulate into s_acc.
    // -------------------------------------------------------------------------
    for (int tile_idx = my_tile_start; tile_idx < my_tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * kTokensPerTile;

        for (int i = (int) threadIdx.x; i < kTokensPerTile * E; i += kNumThreads) {
            const int token  = i / E;
            const int gtoken = tile_offset + token;
            s_tile[i]        = (gtoken < num_dispatched_tokens)
                                   ? (int) routing_map[(int64_t) tile_offset * E + i]
                                   : 0;
        }
        __syncthreads();

        for (int e = 0; e < E; ++e) {
            if constexpr (kTPT == 1) {
                // kTPT == 1 keeps the original InclusiveSum path verbatim so
                // the codegen stays bit-identical for tiny problems
                // (e.g. E=1, T=8K) where any extra register / instruction
                // count is amplified into measurable wall-clock cost.
                const int v = s_tile[(int) threadIdx.x * E + e];
                int       out, sum;
                BlockScan(scan_temp).InclusiveSum(v, out, sum);
                const int prev_acc                = s_acc[e];
                s_tile[(int) threadIdx.x * E + e] = (v == 1) ? (out + prev_acc) : 0;
                if (threadIdx.x == 0) {
                    s_acc[e] = prev_acc + sum;
                }
            } else {
                int local[kTPT];
                int total = 0;
#pragma unroll
                for (int k = 0; k < kTPT; ++k) {
                    const int t = (int) threadIdx.x * kTPT + k;
                    local[k]    = s_tile[t * E + e];
                    total += local[k];
                }

                int excl_block, sum;
                BlockScan(scan_temp).ExclusiveSum(total, excl_block, sum);

                const int prev_acc = s_acc[e];
                int       running  = excl_block + prev_acc;
#pragma unroll
                for (int k = 0; k < kTPT; ++k) {
                    const int t       = (int) threadIdx.x * kTPT + k;
                    const int v       = local[k];
                    s_tile[t * E + e] = (v == 1) ? (running + 1) : 0;
                    running += v;
                }

                if (threadIdx.x == 0) {
                    s_acc[e] = prev_acc + sum;
                }
            }
            __syncthreads();
        }

        for (int i = (int) threadIdx.x; i < kTokensPerTile * E; i += kNumThreads) {
            const int token  = i / E;
            const int gtoken = tile_offset + token;
            if (gtoken < num_dispatched_tokens) {
                row_id_map[(int64_t) tile_offset * E + i] = s_tile[i];
            }
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Phase 3: decoupled lookback per expert (one thread per expert column).
    // -------------------------------------------------------------------------
    if ((int) threadIdx.x < E) {
        const int      e         = (int) threadIdx.x;
        const int32_t  agg       = s_acc[e];
        const uint32_t init_flag = (blockIdx.x == 0) ? detail::FLAG_PREFIX : detail::FLAG_AGGREGATE;
        detail::store_state(&tile_state[(int64_t) blockIdx.x * E + e],
                            detail::pack_state(init_flag, agg));

        int32_t accum = 0;
        for (int b = (int) blockIdx.x - 1; b >= 0; --b) {
            uint64_t v;
            uint32_t f;
            do {
                v = detail::load_state(&tile_state[(int64_t) b * E + e]);
                f = detail::unpack_flag(v);
            } while (f == detail::FLAG_INVALID);
            accum += detail::unpack_val(v);
            if (f == detail::FLAG_PREFIX) {
                break;
            }
        }
        s_excl_prefix[e] = accum;

        if (blockIdx.x != 0) {
            detail::store_state(&tile_state[(int64_t) blockIdx.x * E + e],
                                detail::pack_state(detail::FLAG_PREFIX, accum + agg));
        }

        atomicAdd(&tokens_per_expert[e], agg);
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 4: atomic counter barrier — wait until every block has finished
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(barrier_counter_p4, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        while (__hip_atomic_load(barrier_counter_p4, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
               (int) gridDim.x) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 5: snapshot tokens_per_expert into LDS
    // -------------------------------------------------------------------------
    if ((int) threadIdx.x < E) {
        const int v        = (int) tokens_per_expert[threadIdx.x];
        s_acc[threadIdx.x] = v;
        const int padded =
            (pad_multiple > 0) ? ((v + pad_multiple - 1) / pad_multiple) * pad_multiple : v;
        s_tpe_prefix[threadIdx.x] = padded;
        s_num_padded[threadIdx.x] = padded - v;
    }
    __syncthreads();
    {
        const int v = ((int) threadIdx.x < E) ? s_tpe_prefix[threadIdx.x] : 0;
        int       excl;
        BlockScan(scan_temp).ExclusiveSum(v, excl);
        if ((int) threadIdx.x < E) {
            s_tpe_prefix[threadIdx.x] = excl;
        }
    }
    __syncthreads();

    // Phase 5 done — publish so Phase 8 (block 0 only) can finalise
    // tokens_per_expert without racing other blocks' Phase 5 reads.
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(barrier_counter_p5, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }

    // -------------------------------------------------------------------------
    // Phase 6: patch row_id_map for this block's tile range.
    // -------------------------------------------------------------------------
    for (int tile_idx = my_tile_start; tile_idx < my_tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * kTokensPerTile;
        for (int i = (int) threadIdx.x; i < kTokensPerTile * E; i += kNumThreads) {
            const int token  = i / E;
            const int expert = i % E;
            const int gtoken = tile_offset + token;
            if (gtoken < num_dispatched_tokens) {
                const int64_t offset = (int64_t) tile_offset * E + i;
                const int     old    = row_id_map[offset];
                if (old != 0) {
                    const int new_value = old + s_excl_prefix[expert] + s_tpe_prefix[expert];
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
    // Phase 7: padding writes for `pad_multiple > 0`
    // -------------------------------------------------------------------------
    for (int i = (int) blockIdx.x; i < pad_multiple; i += (int) gridDim.x) {
        const int64_t offset = ((int64_t) i + num_dispatched_tokens) * E;
        for (int j = 0; j < E; ++j) {
            int padded_offset;
            if (i < s_num_padded[j]) {
                padded_offset = -(s_acc[j] + s_tpe_prefix[j] + i + 1);
                if (abs(padded_offset) > npt) {
                    *overflow_flag = 1;
                    padded_offset  = 0;
                }
            } else {
                padded_offset = 0;
            }
            row_id_map[offset + j] = padded_offset;
        }
    }

    // -------------------------------------------------------------------------
    // Phase 8: block 0 finalises tokens_per_expert with overflow handling.
    // -------------------------------------------------------------------------
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            while (__hip_atomic_load(barrier_counter_p5, __ATOMIC_ACQUIRE,
                                     __HIP_MEMORY_SCOPE_AGENT) < (int) gridDim.x) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();
        if ((int) threadIdx.x < E) {
            const int tokens_for_expert_i  = s_acc[threadIdx.x] + s_num_padded[threadIdx.x];
            const int overflow_num         = tokens_for_expert_i + s_tpe_prefix[threadIdx.x] - npt;
            tokens_per_expert[threadIdx.x] = (overflow_num < 0)
                                                 ? tokens_for_expert_i
                                                 : max(0, tokens_for_expert_i - overflow_num);
        }
    }
}
} // namespace primus_turbo
