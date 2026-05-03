// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// permute_preprocess.cu — single-kernel MoE permute preprocessing built on
// `hipcub::BlockScan` (intra-block reduce-scan) + a "decoupled lookback" scan
// (inter-block reduce-scan).
//
// The decoupled-lookback pattern is the same one used in:
//   * Merrill & Garland, "Single-pass Parallel Prefix Scan with Decoupled
//     Look-back" (NVIDIA TR 2016).
//   * AMD GPUOpen, "Boosting GPU Radix Sort":
//       https://gpuopen.com/learn/boosting_gpu_radix_sort/
//   * rocPRIM `device_scan`:
//       https://github.com/ROCm/rocPRIM/blob/develop_deprecated/rocprim/include/rocprim/device/device_scan.hpp
//   * Local reference: `csrc/kernels/permute/ref.cu`, ported from the original
//     CUDA NVL kernel.
//
// Inputs / outputs (all device pointers):
//
//   routing_map        [N, E] bool  : routing_map[t, e] = 1 iff token `t` is
//                                     routed to local-expert `e`.
//   tokens_per_expert  [E]   int32  : tokens_per_expert[e] = sum_t routing_map[t, e].
//                                     Zeroed by the launcher; written via atomicAdd.
//   row_id_map         [N, E] int32 : 0 if routing_map[t, e] == 0, otherwise
//                                     the 1-indexed destination slot in the
//                                     expert-grouped permuted output:
//                                       row_id_map[t, e] = (sum_{t'<=t} routing_map[t', e])
//                                                        + (sum_{e'<e}   tokens_per_expert[e']).
//
// Algorithm — one block per token tile, one kernel:
//
//   Phase 1  Stage routing_map[tile_offset:tile_offset+kBlockSize, :] into LDS.
//   Phase 2  Per-expert in-block InclusiveSum via `hipcub::BlockScan` →
//            (in-block scan position, block aggregate) per (token, expert).
//   Phase 3  Decoupled lookback per expert (one thread per expert):
//              (a) publish AGGREGATE / value to `tile_state[blockIdx.x, e]`
//                  (or PREFIX for block 0);
//              (b) scan tile_state[predecessor, e] backwards from blockIdx.x-1,
//                  accumulating values, stopping at the first PREFIX entry;
//              (c) publish full PREFIX so successors can short-circuit;
//              (d) atomically add this block's aggregate to
//                  `tokens_per_expert[e]` for the cross-expert prefix.
//   Phase 4  Atomic counter barrier — wait until every block has finished
//            Phase 3 so `tokens_per_expert` reflects the full sum.
//   Phase 5  Per-expert ExclusiveSum (in-block scan over E entries) of
//            `tokens_per_expert` → cross-expert offset into the permuted layout.
//   Phase 6  Write final `row_id_map` for this block's tile:
//              non-zero entries = inblock_scan + block_excl_prefix + tpe_prefix.
//
// Co-residency requirement: the lookback in Phase 3 busy-waits on flags from
// blocks 0..blockIdx.x-1, so every block must be scheduled. The launcher caps
// `gridDim.x` at `num_cu * max_blocks_per_cu` (computed via
// `hipOccupancyMaxActiveBlocksPerMultiprocessor`). For workloads with more
// tiles than that, raise `kBlockSize` or extend Phase 1-3 to chain multiple
// tiles per block (out of scope for this reference implementation).

#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace primus_turbo::permute_preprocess {

// =============================================================================
// Decoupled-lookback tile state
//
// Each tile-state slot packs `{ flag : low 32 bits, value : high 32 bits }`
// into a single 64-bit word. A 64-bit atomic load/store therefore returns a
// consistent (flag, value) pair without any extra fence, because:
//
//   * The slot has a SINGLE producer block (the block that owns blockIdx.x).
//   * Successor blocks read the slot exactly once per lookback iteration.
//   * Atomic-coherence on a single address guarantees monotonic progression
//     INVALID -> AGGREGATE -> PREFIX (or directly INVALID -> PREFIX for
//     block 0). Whichever snapshot a successor observes is correct on its own.
//
// We use ATOMIC_RELAXED with AGENT scope (gfx942/gfx950 visible device-wide)
// because there is no other shared state that needs to be ordered against
// the publish.
// =============================================================================

constexpr uint32_t FLAG_INVALID   = 0u; // slot has not been published yet
constexpr uint32_t FLAG_AGGREGATE = 1u; // only this block's local aggregate
constexpr uint32_t FLAG_PREFIX    = 2u; // full inclusive prefix (block + all predecessors)

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

// =============================================================================
// Kernel
// =============================================================================

template <int kBlockSize>
__launch_bounds__(kBlockSize, 1) __global__ void permute_preprocess_kernel(
    const bool *routing_map,        // [N, E]
    int32_t    *tokens_per_expert,  // [E]   (zero-initialised by launcher)
    int        *row_id_map,         // [N, E]
    uint64_t   *tile_state,         // [gridDim.x, E] (zero-initialised by launcher)
    int        *barrier_counter,    // scalar         (zero-initialised by launcher)
    int         num_dispatched_tokens,
    int         num_local_experts) {

    using BlockScan = hipcub::BlockScan<int32_t, kBlockSize>;
    __shared__ typename BlockScan::TempStorage scan_temp;

    // Dynamic LDS layout (sized by the launcher):
    //   s_tile        [kBlockSize * E] — tile staging (in: routing_map; out: in-block scan)
    //   s_block_agg   [E]              — per-expert tile aggregate
    //   s_excl_prefix [E]              — per-expert exclusive prefix from preceding blocks
    //   s_tpe_prefix  [E]              — per-expert exclusive prefix of tokens_per_expert
    extern __shared__ int dyn_shmem[];
    const int E             = num_local_experts;
    int      *s_tile        = dyn_shmem;
    int      *s_block_agg   = s_tile + kBlockSize * E;
    int      *s_excl_prefix = s_block_agg + E;
    int      *s_tpe_prefix  = s_excl_prefix + E;

    const int tile_offset = static_cast<int>(blockIdx.x) * kBlockSize;

    // -------------------------------------------------------------------------
    // Phase 1 — stage routing_map tile into shared memory (cast bool -> int).
    // -------------------------------------------------------------------------
    for (int i = threadIdx.x; i < kBlockSize * E; i += kBlockSize) {
        const int token  = i / E;
        const int expert = i % E;
        const int gtoken = tile_offset + token;
        s_tile[token * E + expert] =
            (gtoken < num_dispatched_tokens)
                ? static_cast<int>(routing_map[static_cast<int64_t>(gtoken) * E + expert])
                : 0;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 2 — per-expert in-block InclusiveSum.
    //
    //   For each expert column e (sequentially), every thread t loads
    //   s_tile[t * E + e] and runs hipcub::BlockScan::InclusiveSum, which
    //   returns the inclusive scan output `out` AND the block-wide aggregate
    //   `sum`. Example for one column: 1,0,1,0,1,1,0 -> 1,0,2,0,3,4,0 (with
    //   the 0 entries forced back to 0 since they don't claim a slot).
    //
    //   `BlockScan` reuses `scan_temp` between iterations, so we sync at the
    //   end of each iteration before re-entering the scan body.
    // -------------------------------------------------------------------------
    for (int e = 0; e < E; ++e) {
        const int v = s_tile[threadIdx.x * E + e];
        int       out, sum;
        BlockScan(scan_temp).InclusiveSum(v, out, sum);
        s_tile[threadIdx.x * E + e] = (v == 1) ? out : 0;
        if (threadIdx.x == 0) {
            s_block_agg[e] = sum;
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Phase 3 — decoupled lookback (one thread per expert column).
    //
    //   For expert e, thread t == e:
    //     (1) Publish (AGGREGATE | aggregate). Block 0 has no predecessors,
    //         so it publishes (PREFIX | aggregate) directly.
    //     (2) Walk tile_state[predecessor, e] from blockIdx.x-1 down to 0,
    //         busy-waiting on FLAG_INVALID. Stop the moment we observe a
    //         FLAG_PREFIX entry — by induction it already aggregates every
    //         block before it, so further lookback is unnecessary (this is
    //         the "short-circuit" property that makes the scan single-pass).
    //     (3) Re-publish as (PREFIX | aggregate + lookback_sum) so successors
    //         can short-circuit on this slot too.
    //     (4) Contribute this block's aggregate to tokens_per_expert via a
    //         relaxed atomicAdd; the Phase-4 barrier publishes it.
    // -------------------------------------------------------------------------
    if (threadIdx.x < E) {
        const int      e   = threadIdx.x;
        const int32_t  agg = s_block_agg[e];

        const uint32_t initial_flag =
            (blockIdx.x == 0) ? FLAG_PREFIX : FLAG_AGGREGATE;
        store_state(&tile_state[blockIdx.x * E + e], pack_state(initial_flag, agg));

        int32_t accum = 0;
        for (int b = static_cast<int>(blockIdx.x) - 1; b >= 0; --b) {
            uint64_t v;
            uint32_t f;
            do {
                v = load_state(&tile_state[b * E + e]);
                f = unpack_flag(v);
            } while (f == FLAG_INVALID);
            accum += unpack_val(v);
            if (f == FLAG_PREFIX) {
                break;
            }
        }
        s_excl_prefix[e] = accum;

        if (blockIdx.x != 0) {
            store_state(&tile_state[blockIdx.x * E + e],
                        pack_state(FLAG_PREFIX, accum + agg));
        }

        atomicAdd(&tokens_per_expert[e], agg);
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 4 — atomic counter barrier.
    //
    //   Every block fetch-adds 1 with ACQ_REL ordering, then thread 0 spins
    //   on an acquire-load until the counter reaches `gridDim.x`. The acquire
    //   pairs with the release on every other block's fetch-add, which in
    //   program order happens AFTER its Phase-3 atomicAdd. So once the load
    //   returns gridDim.x, every block's contribution to tokens_per_expert
    //   is visible.
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(barrier_counter, 1, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        while (__hip_atomic_load(barrier_counter, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT) < static_cast<int>(gridDim.x)) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 5 — cross-expert ExclusiveSum on tokens_per_expert.
    //
    //   E <= kBlockSize (enforced by the launcher), so a single BlockScan is
    //   enough. Result: s_tpe_prefix[e] = sum_{e' < e} tokens_per_expert[e'].
    // -------------------------------------------------------------------------
    {
        const int v   = (static_cast<int>(threadIdx.x) < E)
                            ? tokens_per_expert[threadIdx.x]
                            : 0;
        int       excl;
        BlockScan(scan_temp).ExclusiveSum(v, excl);
        if (static_cast<int>(threadIdx.x) < E) {
            s_tpe_prefix[threadIdx.x] = excl;
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 6 — materialise this block's slice of row_id_map.
    //
    //   row_id_map[t, e] = 0                                              if routing_map[t, e] == 0
    //                    = inblock_scan + s_excl_prefix[e] + s_tpe_prefix[e] otherwise.
    //
    //   The inblock_scan is the InclusiveSum from Phase 2 (so it starts at
    //   1 for the first routed token in this block), which makes the final
    //   row_id 1-indexed and matches v1/v2/v3/v4's convention.
    // -------------------------------------------------------------------------
    for (int i = threadIdx.x; i < kBlockSize * E; i += kBlockSize) {
        const int token  = i / E;
        const int expert = i % E;
        const int gtoken = tile_offset + token;
        if (gtoken < num_dispatched_tokens) {
            int v = s_tile[token * E + expert];
            if (v != 0) {
                v += s_excl_prefix[expert] + s_tpe_prefix[expert];
            }
            row_id_map[static_cast<int64_t>(gtoken) * E + expert] = v;
        }
    }
}

// =============================================================================
// Host-side launcher
// =============================================================================

constexpr int kDefaultBlockSize = 256;

namespace {

inline size_t dyn_shmem_bytes(int block_size, int num_local_experts) {
    // s_tile (block_size * E) + s_block_agg (E) + s_excl_prefix (E) + s_tpe_prefix (E)
    return (static_cast<size_t>(block_size) * num_local_experts +
            static_cast<size_t>(3) * num_local_experts) *
           sizeof(int);
}

inline int max_active_grid(int block_size, size_t shmem_bytes) {
    int dev = 0;
    (void)hipGetDevice(&dev);
    int num_cu = 0;
    (void)hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, dev);
    int max_blocks_per_cu = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_cu,
        reinterpret_cast<const void *>(&permute_preprocess_kernel<kDefaultBlockSize>),
        block_size, shmem_bytes);
    return std::max(num_cu * max_blocks_per_cu, 1);
}

} // namespace

void permute_preprocess(const bool *routing_map, int32_t *tokens_per_expert, int *row_id_map,
                        int num_dispatched_tokens, int num_local_experts, hipStream_t stream) {
    constexpr int block_size = kDefaultBlockSize;

    if (num_local_experts <= 0) {
        return;
    }
    if (num_local_experts > block_size) {
        std::fprintf(stderr,
                     "permute_preprocess: num_local_experts (%d) must be <= block_size (%d)\n",
                     num_local_experts, block_size);
        std::abort();
    }

    // tokens_per_expert is updated via atomicAdd in Phase 3 → must start zeroed.
    // We zero it unconditionally so callers never have to.
    (void)hipMemsetAsync(tokens_per_expert, 0,
                         static_cast<size_t>(num_local_experts) * sizeof(int32_t), stream);

    if (num_dispatched_tokens <= 0) {
        // Nothing to scan; tokens_per_expert is already zeroed and row_id_map
        // has no rows. Skip the launch.
        return;
    }

    const int    num_tiles = (num_dispatched_tokens + block_size - 1) / block_size;
    const size_t shmem     = dyn_shmem_bytes(block_size, num_local_experts);
    const int    max_grid  = max_active_grid(block_size, shmem);

    if (num_tiles > max_grid) {
        // The decoupled-lookback chain busy-waits on predecessor flags, so all
        // blocks must be co-resident. Surface a clear error rather than risk
        // a livelock.
        std::fprintf(stderr,
                     "permute_preprocess: num_tiles (%d) > max_active_blocks (%d) for the "
                     "decoupled-lookback scan; raise kDefaultBlockSize, reduce N, or extend "
                     "the kernel to chain multiple tiles per block.\n",
                     num_tiles, max_grid);
        std::abort();
    }

    const int grid_size = num_tiles;

    // Allocate + zero the lookback workspace:
    //   tile_state      : grid_size * E uint64 (FLAG_INVALID == 0)
    //   barrier_counter : 1 int                (must be 0)
    const size_t tile_state_bytes = static_cast<size_t>(grid_size) * num_local_experts *
                                    sizeof(uint64_t);
    // Pad to 8-byte alignment so the trailing int sits cleanly after tile_state.
    const size_t scratch_bytes = tile_state_bytes + sizeof(int);

    void *scratch = nullptr;
    (void)hipMallocAsync(&scratch, scratch_bytes, stream);
    (void)hipMemsetAsync(scratch, 0, scratch_bytes, stream);

    uint64_t *tile_state      = reinterpret_cast<uint64_t *>(scratch);
    int      *barrier_counter = reinterpret_cast<int *>(
        reinterpret_cast<char *>(scratch) + tile_state_bytes);

    permute_preprocess_kernel<block_size><<<grid_size, block_size, shmem, stream>>>(
        routing_map, tokens_per_expert, row_id_map, tile_state, barrier_counter,
        num_dispatched_tokens, num_local_experts);

    (void)hipFreeAsync(scratch, stream);
}

} // namespace primus_turbo::permute_preprocess
