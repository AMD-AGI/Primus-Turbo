// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// permute_v5.cu — single-kernel `permute_preprocessing` rewrite using
// per-expert "decoupled lookback" scan with FLAG_PREFIX short-circuit.
//
// What changes vs v3:
//
//   v3 also publishes per-block per-expert sums to a global `block_sums`
//   buffer and reads them back, but its lookback step (Phase 2b) sums
//   `block_sums[0..blockIdx.x-1]` UNCONDITIONALLY — i.e. an O(B²) total
//   amount of cross-block work and no early termination.
//
//   v5 adopts the textbook decoupled-lookback algorithm
//   (Merrill & Garland 2016; AMD GPUOpen "Boosting GPU Radix Sort";
//    rocPRIM `device_scan`):
//
//     * each (block, expert) tile-state slot is a single 64-bit atomic word
//       packing `{ flag : low 32 bits, value : high 32 bits }`;
//     * a block first publishes its aggregate as FLAG_AGGREGATE, walks
//       predecessors in reverse, and SHORT-CIRCUITS the moment it observes
//       a FLAG_PREFIX entry — by induction that slot already aggregates
//       everything before it;
//     * the block then re-publishes its full inclusive prefix as
//       FLAG_PREFIX so successors can short-circuit on this slot too.
//
//   Total cross-block work is O(B) on average and O(B²) only in the
//   pathological case where every successor races ahead of every
//   predecessor's PREFIX publication.
//
// Everything else mirrors v3:
//   * single non-cooperative kernel (no `grid.sync()`, no
//     `hipLaunchCooperativeKernel`);
//   * `__hip_atomic_*` (RELAXED + AGENT scope) for cross-block traffic;
//   * gridDim.x is capped at the device's max-active-blocks count for
//     co-residency of the lookback chain;
//   * per-stream lookback workspace cache so we don't `hipMallocAsync` on
//     every launcher invocation.
//
// All v5 symbols live in `primus_turbo::v5`.

#include "primus_turbo/permute.h"

#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace primus_turbo {
namespace v5 {

// =============================================================================
// Decoupled-lookback tile state
// =============================================================================

namespace detail {

constexpr uint32_t FLAG_INVALID   = 0u; // slot has not been published yet
constexpr uint32_t FLAG_AGGREGATE = 1u; // only this block's local aggregate
constexpr uint32_t FLAG_PREFIX    = 2u; // full inclusive prefix (block + all preds)

__device__ __forceinline__ uint64_t pack_state(uint32_t flag, int32_t val) {
    return static_cast<uint64_t>(flag) |
           (static_cast<uint64_t>(static_cast<uint32_t>(val)) << 32);
}
__device__ __forceinline__ uint32_t unpack_flag(uint64_t s) {
    return static_cast<uint32_t>(s & 0xFFFFFFFFu);
}
__device__ __forceinline__ int32_t unpack_val(uint64_t s) {
    return static_cast<int32_t>(static_cast<uint32_t>(s >> 32));
}

__device__ __forceinline__ void store_state(uint64_t *ptr, uint64_t v) {
    __hip_atomic_store(ptr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t load_state(uint64_t *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

} // namespace detail

// =============================================================================
// Kernel
// =============================================================================

// `kTPT` (tokens per thread per tile) follows the rocPRIM `BlockScan`
// items-per-thread pattern. Each thread carries `kTPT` consecutive tokens
// for every expert column; the per-expert in-block scan does an
// in-register inclusive sum over the kTPT items, then a single
// `BlockScan::ExclusiveSum` over the per-thread totals, then adds the
// per-thread exclusive prefix back to each item. Net effect:
//   * # tiles per kernel invocation drops by `kTPT`x
//   * # `BlockScan` calls drops by `kTPT`x
//   * # `__syncthreads()` in the per-expert loop drops by `kTPT`x
// LDS staging cost scales linearly with kTPT, so the launcher dispatches
// `kTPT > 1` only when the LDS budget allows.
template <int block_size, int kTPT>
__launch_bounds__(block_size, 1) __global__ void permute_preprocessing_kernel(
    bool *routing_map, int *num_dispatched_tokens_ptr, int num_of_local_experts,
    int rows_workspace_1, int pad_multiple, int32_t *tokens_per_expert, int *row_id_map,
    int *overflow_flag, int64_t num_permuted_tokens,
    // Lookback workspace, zero-initialised by the launcher:
    //   tile_state          : [gridDim.x, num_of_local_experts]   (uint64_t)
    //   barrier_counter_p4  : scalar — gates Phase 5 (tokens_per_expert read)
    //   barrier_counter_p5  : scalar — gates Phase 8 (block 0 finalises tpe)
    uint64_t *tile_state, int *barrier_counter_p4, int *barrier_counter_p5) {

    using BlockScan = hipcub::BlockScan<int32_t, block_size>;
    __shared__ typename BlockScan::TempStorage scan_temp;

    extern __shared__ int dyn_shmem[];

    constexpr int kTokensPerTile = block_size * kTPT;
    const int E = num_of_local_experts;

    // Dyn-shmem layout (lifetimes annotated; s_acc is reused after Phase 3
    // as the snapshot of `tokens_per_expert`):
    //   s_tile        [kTokensPerTile * E] — Phase 1-2 staging / partial scan
    //                                        kTokensPerTile == block_size * kTPT
    //   s_acc         [E]              — Phase 1-3: per-block running aggregate
    //                                    Phase 5-8: snapshot of tokens_per_expert
    //   s_excl_prefix [E]              — Phase 3 -> end: lookback exclusive prefix
    //   s_tpe_prefix  [E]              — Phase 5 -> end: cross-expert exclusive prefix
    //   s_num_padded  [E]              — Phase 5 -> end: per-expert padding count
    int *s_tile        = dyn_shmem;
    int *s_acc         = s_tile + kTokensPerTile * E;
    int *s_excl_prefix = s_acc + E;
    int *s_tpe_prefix  = s_excl_prefix + E;
    int *s_num_padded  = s_tpe_prefix + E;

    const int num_dispatched_tokens = *num_dispatched_tokens_ptr;

    // Internal tile granularity is kTokensPerTile = block_size * kTPT tokens.
    // For kTPT == 1 this matches the caller's `rows_workspace_1` exactly,
    // so we skip the device-side division and use the caller-supplied value
    // (preserves byte-identical kTPT=1 codegen vs the round-1 baseline).
    int internal_rows_w1;
    if constexpr (kTPT == 1) {
        internal_rows_w1 = rows_workspace_1;
    } else {
        internal_rows_w1 =
            (num_dispatched_tokens + kTokensPerTile - 1) / kTokensPerTile;
    }

    // Each block owns a contiguous slice of tile indices in [0, internal_rows_w1).
    // tiles_per_block == 1 is the dominant case once gridDim.x scales with
    // internal_rows_w1; for very large N (internal_rows_w1 > max_active_blocks)
    // we fall back to multiple tiles per block.
    const int tiles_per_block = (internal_rows_w1 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int my_tile_start   = (int)blockIdx.x * tiles_per_block;
    const int my_tile_end_raw = my_tile_start + tiles_per_block;
    const int my_tile_end     = (my_tile_end_raw < internal_rows_w1) ? my_tile_end_raw
                                                                     : internal_rows_w1;
    // When this block owns exactly one tile, the Phase 1-2 LDS staging
    // (`s_tile`) survives untouched through Phase 3-5, so Phase 6 can
    // patch directly out of LDS instead of round-tripping through global
    // memory. Skips both the Phase 1-2 writeback (T*E*4 B writes) AND the
    // Phase 6 read (T*E*4 B reads) on the row_id_map buffer.
    const bool single_tile = (my_tile_end - my_tile_start) == 1;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *overflow_flag = 0;
    }
    const int npt = (num_permuted_tokens < 0) ? INT_MAX : (int)num_permuted_tokens;

    // Init s_acc to 0 so the per-tile loop can accumulate sums across tiles.
    for (int i = (int)threadIdx.x; i < E; i += block_size) {
        s_acc[i] = 0;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 1-2: per-tile in-block InclusiveSum + accumulate into s_acc.
    //
    //   For each tile (kTokensPerTile = block_size * kTPT tokens) assigned
    //   to this block:
    //     (1) Stage routing_map[tile] -> s_tile (kTokensPerTile * E ints).
    //     (2) For each expert column e:
    //           - Each thread loads kTPT items from s_tile (its kTPT
    //             consecutive tokens).
    //           - In-register inclusive scan over the kTPT items: total =
    //             sum(local[0..kTPT-1]); local_excl[k] = sum of
    //             local[0..k-1].
    //           - 1 BlockScan over per-thread totals -> per-thread
    //             exclusive `excl_block`.
    //           - Write back s_tile[t,e] = (v==1 ? excl_block +
    //             local_excl[k] + 1 + s_acc[e] : 0).
    //         This pattern (rocPRIM `BlockScan` items-per-thread) cuts
    //         BlockScan call count and syncthreads count by `kTPT`x.
    //     (3) s_acc[e] += sum.
    //     (4) Write the partial scan back to row_id_map (Phase 6 patches it
    //         with the lookback prefix + cross-expert prefix).
    //
    //   `BlockScan` reuses scan_temp between iterations -> __syncthreads at
    //   the end of each per-expert iteration before re-entering the body.
    // -------------------------------------------------------------------------
    for (int tile_idx = my_tile_start; tile_idx < my_tile_end; ++tile_idx) {
        const int tile_offset = tile_idx * kTokensPerTile;

        for (int i = (int)threadIdx.x; i < kTokensPerTile * E; i += block_size) {
            const int token  = i / E;
            const int gtoken = tile_offset + token;
            s_tile[i] = (gtoken < num_dispatched_tokens)
                            ? (int)routing_map[(int64_t)tile_offset * E + i]
                            : 0;
        }
        __syncthreads();

        for (int e = 0; e < E; ++e) {
            if constexpr (kTPT == 1) {
                // kTPT == 1 keeps the original InclusiveSum path verbatim so
                // the codegen stays bit-identical for tiny problems
                // (e.g. E=1, T=8K) where any extra register / instruction
                // count is amplified into measurable wall-clock cost.
                const int v = s_tile[(int)threadIdx.x * E + e];
                int       out, sum;
                BlockScan(scan_temp).InclusiveSum(v, out, sum);
                const int prev_acc                  = s_acc[e];
                s_tile[(int)threadIdx.x * E + e]    = (v == 1) ? (out + prev_acc) : 0;
                if (threadIdx.x == 0) {
                    s_acc[e] = prev_acc + sum;
                }
            } else {
                int local[kTPT];
                int total = 0;
#pragma unroll
                for (int k = 0; k < kTPT; ++k) {
                    const int t = (int)threadIdx.x * kTPT + k;
                    local[k]    = s_tile[t * E + e];
                    total      += local[k];
                }

                int excl_block, sum;
                BlockScan(scan_temp).ExclusiveSum(total, excl_block, sum);

                const int prev_acc = s_acc[e];
                int       running  = excl_block + prev_acc;
#pragma unroll
                for (int k = 0; k < kTPT; ++k) {
                    const int t        = (int)threadIdx.x * kTPT + k;
                    const int v        = local[k];
                    s_tile[t * E + e]  = (v == 1) ? (running + 1) : 0;
                    running           += v;
                }

                if (threadIdx.x == 0) {
                    s_acc[e] = prev_acc + sum;
                }
            }
            __syncthreads();
        }

        // Phase 1-2 writeback to row_id_map. Skip when single_tile because
        // s_tile will be re-read directly by Phase 6 from LDS (round-4
        // fusion).
        if (!single_tile) {
            for (int i = (int)threadIdx.x; i < kTokensPerTile * E; i += block_size) {
                const int token  = i / E;
                const int gtoken = tile_offset + token;
                if (gtoken < num_dispatched_tokens) {
                    row_id_map[(int64_t)tile_offset * E + i] = s_tile[i];
                }
            }
            __syncthreads();
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3: decoupled lookback per expert (one thread per expert column).
    //
    //   For expert e (handled by thread t == e):
    //     (1) Publish (AGGREGATE | s_acc[e]) — or (PREFIX | s_acc[e]) for
    //         block 0, which has no predecessors.
    //     (2) Walk tile_state[predecessor, e] backwards from blockIdx.x - 1,
    //         busy-waiting on FLAG_INVALID. Stop the moment a FLAG_PREFIX
    //         entry is observed: by induction it already aggregates every
    //         preceding block, so further lookback is redundant. This is
    //         the short-circuit property that makes the scan single-pass.
    //     (3) Re-publish as (PREFIX | accum + s_acc[e]) so successors can
    //         short-circuit on this slot too.
    //     (4) Contribute s_acc[e] to global tokens_per_expert via atomicAdd.
    // -------------------------------------------------------------------------
    if ((int)threadIdx.x < E) {
        const int      e        = (int)threadIdx.x;
        const int32_t  agg      = s_acc[e];
        const uint32_t init_flag =
            (blockIdx.x == 0) ? detail::FLAG_PREFIX : detail::FLAG_AGGREGATE;
        detail::store_state(&tile_state[(int64_t)blockIdx.x * E + e],
                            detail::pack_state(init_flag, agg));

        int32_t accum = 0;
        for (int b = (int)blockIdx.x - 1; b >= 0; --b) {
            uint64_t v;
            uint32_t f;
            do {
                v = detail::load_state(&tile_state[(int64_t)b * E + e]);
                f = detail::unpack_flag(v);
            } while (f == detail::FLAG_INVALID);
            accum += detail::unpack_val(v);
            if (f == detail::FLAG_PREFIX) {
                break;
            }
        }
        s_excl_prefix[e] = accum;

        if (blockIdx.x != 0) {
            detail::store_state(&tile_state[(int64_t)blockIdx.x * E + e],
                                detail::pack_state(detail::FLAG_PREFIX, accum + agg));
        }

        atomicAdd(&tokens_per_expert[e], agg);
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 4: atomic counter barrier — wait until every block has finished
    //          Phase 3 so `tokens_per_expert` reflects the global per-expert
    //          sum. ACQ_REL on fetch-add + ACQUIRE on the spin-load form a
    //          release/acquire pair; once the load returns gridDim.x, every
    //          block's relaxed atomicAdd to tokens_per_expert is visible.
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(barrier_counter_p4, 1, __ATOMIC_ACQ_REL,
                               __HIP_MEMORY_SCOPE_AGENT);
        while (__hip_atomic_load(barrier_counter_p4, __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_AGENT) < (int)gridDim.x) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 5: snapshot tokens_per_expert into LDS and compute its
    //          cross-expert ExclusiveSum (with optional pad_multiple
    //          rounding). The snapshot is essential: Phase 8 (block 0)
    //          rewrites tokens_per_expert in place, and other blocks must
    //          continue to see the pre-rewrite value through s_acc.
    // -------------------------------------------------------------------------
    if ((int)threadIdx.x < E) {
        const int v = (int)tokens_per_expert[threadIdx.x];
        s_acc[threadIdx.x] = v;
        const int padded =
            (pad_multiple > 0) ? ((v + pad_multiple - 1) / pad_multiple) * pad_multiple : v;
        s_tpe_prefix[threadIdx.x] = padded;
        s_num_padded[threadIdx.x] = padded - v;
    }
    __syncthreads();
    {
        const int v = ((int)threadIdx.x < E) ? s_tpe_prefix[threadIdx.x] : 0;
        int       excl;
        BlockScan(scan_temp).ExclusiveSum(v, excl);
        if ((int)threadIdx.x < E) {
            s_tpe_prefix[threadIdx.x] = excl;
        }
    }
    __syncthreads();

    // Phase 5 done — publish so Phase 8 (block 0 only) can finalise
    // tokens_per_expert without racing other blocks' Phase 5 reads.
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(barrier_counter_p5, 1, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
    }

    // -------------------------------------------------------------------------
    // Phase 6: patch row_id_map for this block's tile range.
    //
    //   The block-local 1-indexed scan position from Phase 1-2 lives either:
    //     * in `s_tile` (when single_tile) — round-4 LDS-fused path, no
    //       row_id_map round-trip;
    //     * in row_id_map (when multi-tile) — Phase 1-2 already wrote it
    //       there because s_tile gets clobbered between tile iterations.
    //   Add the per-expert exclusive prefix from the lookback
    //   (s_excl_prefix) and the cross-expert exclusive prefix from Phase 5
    //   (s_tpe_prefix) to produce the final 1-indexed slot in the
    //   expert-grouped permuted output. Apply overflow check.
    // -------------------------------------------------------------------------
    if (single_tile) {
        const int tile_offset = my_tile_start * kTokensPerTile;
        for (int i = (int)threadIdx.x; i < kTokensPerTile * E; i += block_size) {
            const int token  = i / E;
            const int expert = i % E;
            const int gtoken = tile_offset + token;
            if (gtoken < num_dispatched_tokens) {
                const int local_val = s_tile[i];
                int       new_value = 0;
                if (local_val != 0) {
                    new_value = local_val + s_excl_prefix[expert] + s_tpe_prefix[expert];
                    if (new_value > npt) {
                        *overflow_flag = 1;
                        new_value      = 0;
                    }
                }
                row_id_map[(int64_t)tile_offset * E + i] = new_value;
            }
        }
    } else {
        for (int tile_idx = my_tile_start; tile_idx < my_tile_end; ++tile_idx) {
            const int tile_offset = tile_idx * kTokensPerTile;
            for (int i = (int)threadIdx.x; i < kTokensPerTile * E; i += block_size) {
                const int token  = i / E;
                const int expert = i % E;
                const int gtoken = tile_offset + token;
                if (gtoken < num_dispatched_tokens) {
                    const int64_t offset = (int64_t)tile_offset * E + i;
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
    }

    // -------------------------------------------------------------------------
    // Phase 7: padding writes for `pad_multiple > 0`. Each block handles
    //          `i in [blockIdx.x, pad_multiple)` with grid-stride. Padding
    //          rows live at row_id_map[N..N+pad, :] and use NEGATIVE 1-indexed
    //          offsets to signal the data-movement kernel to write zeros.
    // -------------------------------------------------------------------------
    for (int i = (int)blockIdx.x; i < pad_multiple; i += (int)gridDim.x) {
        const int64_t offset = ((int64_t)i + num_dispatched_tokens) * E;
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
    //          Must wait until every block has snapshotted tokens_per_expert
    //          in Phase 5; barrier_counter_p5 enforces that.
    // -------------------------------------------------------------------------
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            while (__hip_atomic_load(barrier_counter_p5, __ATOMIC_ACQUIRE,
                                     __HIP_MEMORY_SCOPE_AGENT) < (int)gridDim.x) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();
        if ((int)threadIdx.x < E) {
            const int tokens_for_expert_i = s_acc[threadIdx.x] + s_num_padded[threadIdx.x];
            const int overflow_num =
                tokens_for_expert_i + s_tpe_prefix[threadIdx.x] - npt;
            tokens_per_expert[threadIdx.x] =
                (overflow_num < 0) ? tokens_for_expert_i
                                   : max(0, tokens_for_expert_i - overflow_num);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 9 (round-5): in-kernel reset of internal scratch (tile_state +
    // barrier counters) so the next launch on this stream can skip the
    // pre-launch hipMemsetAsync. Per-stream serialisation guarantees these
    // writes complete before the next kernel begins reading.
    //
    //   - Each block resets its OWN row in tile_state[blockIdx.x, :]. By
    //     this point Phase 4 has long since drained the lookback chain, so
    //     no other block is reading our row.
    //   - Block 0 alone resets the two barrier counters (after its Phase 8
    //     spin on barrier_counter_p5 has exited).
    //
    // The launcher signals shape-mismatch (E or grid_size changed across
    // launches) by performing a pre-launch hipMemsetAsync; on the matching-
    // shape steady state this Phase 9 reset takes over.
    // -------------------------------------------------------------------------
    if ((int)threadIdx.x < E) {
        detail::store_state(&tile_state[(int64_t)blockIdx.x * E + threadIdx.x],
                            static_cast<uint64_t>(0));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __hip_atomic_store(barrier_counter_p4, 0, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(barrier_counter_p5, 0, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
    }
}

// =============================================================================
// Host-side helpers
// =============================================================================

namespace {

inline size_t dyn_shmem_bytes(int block_size, int kTPT, int E) {
    // s_tile (block_size * kTPT * E) + s_acc (E) + s_excl_prefix (E) +
    // s_tpe_prefix (E) + s_num_padded (E)
    return (static_cast<size_t>(block_size) * kTPT * E + 4 * static_cast<size_t>(E)) *
           sizeof(int);
}

// Cached device queries. The HIP runtime does not strictly cache
// hipDeviceGetAttribute; for small problems (e.g. permute on T=8K, E=1) the
// extra ~1µs of host overhead per launcher call would mask the kernel
// improvements we're tracking. Snapshot once per process per device.
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
        PRIMUS_TURBO_CHECK_HIP(hipDeviceGetAttribute(
            &caps.num_cu, hipDeviceAttributeMultiprocessorCount, dev));
        PRIMUS_TURBO_CHECK_HIP(hipDeviceGetAttribute(
            &caps.max_shmem_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, dev));
        caps.dev = dev;
    }
    return caps;
}

template <int block_size, int kTPT>
inline int max_active_grid(size_t shmem_bytes) {
    // Cache the per-(kTPT, shmem_bytes) occupancy result. The query goes
    // into the HIP runtime's per-kernel metadata and is not free; for small
    // problems it can be 5-10x the kernel runtime.
    struct CacheKey {
        size_t shmem_bytes;
        int    blocks_per_cu;
    };
    static thread_local std::unordered_map<size_t, int> cache;
    auto                                                it = cache.find(shmem_bytes);
    if (it == cache.end()) {
        int max_blocks_per_cu = 0;
        PRIMUS_TURBO_CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_cu,
            reinterpret_cast<const void *>(
                &permute_preprocessing_kernel<block_size, kTPT>),
            block_size, shmem_bytes));
        it = cache.emplace(shmem_bytes, max_blocks_per_cu).first;
    }
    const int max_blocks_per_cu = it->second;
    const int num_cu            = cached_device_caps().num_cu;
    return std::max(num_cu * max_blocks_per_cu, 1);
}

// Per-stream cache of the lookback workspace allocation. Same rationale as
// v3's cache: hipMallocAsync + hipFreeAsync per launcher call adds ~10–15 µs
// of host overhead which dominates small-case preprocess time.
//
// round-5 extension: the cache also tracks the (E, grid_size) of the last
// launch on this stream. The kernel's Phase 9 resets its own slots in
// tile_state + the barrier counters at end-of-launch, so as long as the
// next launch reuses the same shape the launcher can skip the pre-launch
// hipMemsetAsync entirely (saves ~5–10 µs of host overhead per launch).
// On (re)allocation or any (E, grid_size) change we fall back to a one-
// shot hipMemsetAsync that covers the new footprint.
struct LookbackCacheEntry {
    void  *ptr        = nullptr;
    size_t size_bytes = 0;
    int    last_E     = 0;
    int    last_grid  = 0;
};

inline LookbackCacheEntry &get_lookback_cache(hipStream_t stream) {
    static std::mutex                                          g_mu;
    static std::unordered_map<hipStream_t, LookbackCacheEntry> g_cache;
    std::lock_guard<std::mutex>                                lk(g_mu);
    return g_cache[stream];
}

// Returns the cached scratch pointer; sets `*needs_memset` to true when the
// caller must zero `needed_bytes` (first allocation, growth, or shape
// change). On a matching steady-state (E, grid_size) the pre-launch
// memset is skipped because Phase 9 of the previous kernel handled the
// reset.
inline void *acquire_lookback_workspace(hipStream_t stream, size_t needed_bytes, int E,
                                        int grid_size, bool *needs_memset) {
    LookbackCacheEntry &e          = get_lookback_cache(stream);
    bool                must_clear = false;
    if (e.size_bytes < needed_bytes) {
        if (e.ptr != nullptr) {
            PRIMUS_TURBO_CHECK_HIP(hipFreeAsync(e.ptr, stream));
        }
        PRIMUS_TURBO_CHECK_HIP(hipMallocAsync(&e.ptr, needed_bytes, stream));
        e.size_bytes = needed_bytes;
        must_clear   = true;
    } else if (E != e.last_E || grid_size != e.last_grid) {
        must_clear = true;
    }
    e.last_E      = E;
    e.last_grid   = grid_size;
    *needs_memset = must_clear;
    return e.ptr;
}

// `kTPT` selection policy. We pick the largest kTPT for which BOTH:
//   (a) `s_tile` fits in the device's per-block shared-memory budget;
//   (b) the resulting grid still has enough tiles to *amortise* the
//       per-tile overhead introduced by larger kTPT (each step of kTPT
//       adds register pressure + LDS staging + a longer in-register
//       scan, so the break-even shifts toward higher tile counts).
//
// Empirically on gfx942 (304 CUs, block_size=512), the win region is
//   kTPT=4: T ≳ 240K (tiles ≳ 120, ≥ ~num_cu/2.5)
//   kTPT=2: T ≳ 200K (tiles ≳ 200, ≥ ~num_cu/1.5)
//
// We approximate this with `min_tiles(kTPT) = num_cu / 2 * kTPT/2`, i.e.
// the per-tile overhead grows roughly linearly with kTPT, and the device
// must have at least that many tiles before the kTPT step pays off.
//
// Hard cap at kTPT == 4 because the in-register scan length grows linearly
// with kTPT and starts to pressure VGPRs above that.
inline int pick_kTPT(int block_size, int E, int max_T_static, int num_cu,
                     int max_shmem_per_block) {
    auto fits = [&](int kTPT) {
        return static_cast<int>(dyn_shmem_bytes(block_size, kTPT, E)) <= max_shmem_per_block;
    };
    auto enough_tiles = [&](int kTPT) {
        const int per_tile     = block_size * kTPT;
        const int internal_rws = (max_T_static + per_tile - 1) / per_tile;
        // min_tiles scales with kTPT to reflect the per-tile cost growth.
        const int min_tiles = (num_cu * kTPT) / 4;
        return internal_rws >= min_tiles;
    };
    if (fits(4) && enough_tiles(4)) return 4;
    if (fits(2) && enough_tiles(2)) return 2;
    return 1;
}

template <int block_size, int kTPT>
inline void launch_kernel_impl(bool *routing_map, int *num_dispatched_tokens_ptr,
                               int num_of_local_experts, int rows_workspace_1, int pad_multiple,
                               int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                               int64_t num_permuted_tokens, uint64_t *tile_state,
                               int *barrier_counter_p4, int *barrier_counter_p5, int grid_size,
                               size_t shmem_bytes, hipStream_t stream) {
    permute_preprocessing_kernel<block_size, kTPT>
        <<<grid_size, block_size, shmem_bytes, stream>>>(
            routing_map, num_dispatched_tokens_ptr, num_of_local_experts, rows_workspace_1,
            pad_multiple, tokens_per_expert, row_id_map, overflow_flag, num_permuted_tokens,
            tile_state, barrier_counter_p4, barrier_counter_p5);
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

    // workspace_1 / workspace_2 are kept in the signature to match v1/v2/v3/v4
    // launcher ABI; v5's lookback uses its own per-stream scratch instead.
    (void)workspace_1;
    (void)workspace_2;
    (void)rows_workspace_2;

    const DeviceCaps &caps                = cached_device_caps();
    const int         max_shmem_per_block = caps.max_shmem_per_block;
    const int         num_cu              = caps.num_cu;

    // The caller's `rows_workspace_1` is sized at block_size granularity for
    // the maximum-tokens upper bound, so it doubles as our static T estimate
    // for the kTPT vs. parallelism trade-off.
    const int max_T_static = rows_workspace_1 * block_size;
    // Pass num_cu directly; pick_kTPT scales the min-tile threshold with
    // kTPT to reflect the per-tile overhead growth (see pick_kTPT comment).
    const int kTPT = pick_kTPT(block_size, num_of_local_experts, max_T_static, num_cu,
                               max_shmem_per_block);
    const size_t shmem_bytes = dyn_shmem_bytes(block_size, kTPT, num_of_local_experts);
    PRIMUS_TURBO_CHECK(static_cast<int>(shmem_bytes) <= max_shmem_per_block,
                       "permute_preprocessing v5 requires ", static_cast<int>(shmem_bytes),
                       " B of shared memory (block_size=", block_size, ", kTPT=", kTPT,
                       ", num_of_local_experts=", num_of_local_experts,
                       ") but the device only has ", max_shmem_per_block,
                       " B per block. Reduce num_of_local_experts.");

    // Internal tile granularity is `block_size * kTPT` tokens. Use it (not
    // the caller's `rows_workspace_1`) to size the grid so the lookback
    // chain still has 1 block-state slot per actual tile.
    const int internal_rows_w1 =
        (rows_workspace_1 + kTPT - 1) / kTPT; // ceil(rows_w1 / kTPT)

    // Cap gridDim.x at the device's max-active-blocks count. The lookback
    // chain busy-waits on predecessor flags; later blocks would deadlock if
    // earlier ones had not been scheduled yet.
    int       max_grid = 1;
    if (kTPT == 4) {
        max_grid = max_active_grid<block_size, 4>(shmem_bytes);
    } else if (kTPT == 2) {
        max_grid = max_active_grid<block_size, 2>(shmem_bytes);
    } else {
        max_grid = max_active_grid<block_size, 1>(shmem_bytes);
    }
    const int grid_size = (std::min) (max_grid, internal_rows_w1);

    // Allocate the lookback workspace (cached per stream):
    //   tile_state          : grid_size * E uint64
    //   barrier_counter_p4  : 1 int
    //   barrier_counter_p5  : 1 int
    //
    // The kernel's Phase 9 zeros the per-block tile_state row + the two
    // barrier counters at end-of-launch, so on the steady-state (same
    // (E, grid_size) as the previous launch on this stream) we skip the
    // ~5-10 µs host-side hipMemsetAsync entirely. Allocations / growth /
    // shape changes still trigger a one-shot zero.
    const size_t tile_state_bytes =
        static_cast<size_t>(grid_size) * num_of_local_experts * sizeof(uint64_t);
    const size_t scratch_bytes = tile_state_bytes + 2 * sizeof(int);
    bool         needs_memset  = false;
    void        *scratch       = acquire_lookback_workspace(stream, scratch_bytes,
                                                            num_of_local_experts, grid_size,
                                                            &needs_memset);
    if (needs_memset) {
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(scratch, 0, scratch_bytes, stream));
    }

    uint64_t *tile_state         = reinterpret_cast<uint64_t *>(scratch);
    int      *barrier_counter_p4 = reinterpret_cast<int *>(
        reinterpret_cast<char *>(scratch) + tile_state_bytes);
    int      *barrier_counter_p5 = barrier_counter_p4 + 1;

    // tokens_per_expert is updated via atomicAdd in Phase 3 -> must start zeroed.
    PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(tokens_per_expert, 0,
                                          static_cast<size_t>(num_of_local_experts) *
                                              sizeof(int32_t),
                                          stream));

    if (kTPT == 4) {
        launch_kernel_impl<block_size, 4>(
            routing_map, num_dispatched_tokens_ptr, num_of_local_experts, rows_workspace_1,
            pad_multiple, tokens_per_expert, row_id_map, overflow_flag, num_permuted_tokens,
            tile_state, barrier_counter_p4, barrier_counter_p5, grid_size, shmem_bytes, stream);
    } else if (kTPT == 2) {
        launch_kernel_impl<block_size, 2>(
            routing_map, num_dispatched_tokens_ptr, num_of_local_experts, rows_workspace_1,
            pad_multiple, tokens_per_expert, row_id_map, overflow_flag, num_permuted_tokens,
            tile_state, barrier_counter_p4, barrier_counter_p5, grid_size, shmem_bytes, stream);
    } else {
        launch_kernel_impl<block_size, 1>(
            routing_map, num_dispatched_tokens_ptr, num_of_local_experts, rows_workspace_1,
            pad_multiple, tokens_per_expert, row_id_map, overflow_flag, num_permuted_tokens,
            tile_state, barrier_counter_p4, barrier_counter_p5, grid_size, shmem_bytes, stream);
    }
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

} // namespace v5
} // namespace primus_turbo
