// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// MoE permute / unpermute kernels.
//
//   * `permute_preprocessing_kernel` — single-kernel decoupled-lookback scan
//     (Merrill & Garland 2016) over the routing map. Each (block, expert)
//     tile_state slot is a 64-bit atomic word packing
//     `{flag : 2, epoch : 30, value : 32}`. The 30-bit `launch_epoch` tag
//     lets stale slots from prior launches read as kInvalid, removing the
//     per-launch tile_state memset and the post-Phase-3 grid barrier.
//
//   * `permute_kernel` / `unpermute_kernel` — vectorised data movement.
//     Permute uses one wavefront per token + `UNROLLED_WARP_COPY` (mirrors
//     the DeepEP intranode dispatch loop). Unpermute dispatches between
//     `unpermute_kernel_e1` (E == 1, gather-copy style) and the hoisted
//     depth-2 j-pipeline `unpermute_kernel` (E ≥ 2). The 2-warps-per-token
//     cooperative layout in the latter widens the float-accumulator inner
//     loop and matches the j-tile-2 + depth-2 SW-pipeline structure.
//
// When the per-block temp_storage exceeds the device's dynamic LDS budget
// the launcher transparently spills it to a per-block tile in global
// memory (vsmem fallback, modeled after `rocprim::detail::vsmem_helper_impl`).

#include "primus_turbo/permute.h"

// `UNROLLED_WARP_COPY`, `kWarpSize`, `kFullWarpMask`, and the non-temporal
// `ld_nc_global` / `st_na_global` helpers used by the data-movement kernels.
#include "../deep_ep/utils.cuh"

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

// =============================================================================
// Decoupled-lookback tile-state primitives.
// =============================================================================

// `{flag, value}` pair carried in a single 64-bit slot. The 30-bit `epoch`
// tag (bits 2..31) is incremented per launch by the launcher; slots whose
// epoch differs from the current launch read back as kInvalid.
//   bits  0- 1 : flag           (kInvalid=0, kAggregate=1, kPrefix=2)
//   bits  2-31 : epoch tag      (never 0 — 0 is the post-memset sentinel)
//   bits 32-63 : value          (int32_t signed)
struct TileState {
    static constexpr uint32_t kInvalid   = 0u;
    static constexpr uint32_t kAggregate = 1u;
    static constexpr uint32_t kPrefix    = 2u;

    uint32_t flag;
    int32_t  value;
};

__device__ __forceinline__ TileState load_tile_state(uint64_t *p, uint32_t expected_epoch) {
    const uint64_t raw = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const uint32_t low = static_cast<uint32_t>(raw);
    if ((low >> 2) != expected_epoch) {
        return {TileState::kInvalid, 0};
    }
    return {low & 0x3u, static_cast<int32_t>(static_cast<uint32_t>(raw >> 32))};
}

__device__ __forceinline__ void store_tile_state(uint64_t *p, uint32_t flag, int32_t value,
                                                 uint32_t epoch) {
    const uint32_t low = ((epoch & 0x3FFFFFFFu) << 2) | (flag & 0x3u);
    const uint64_t raw =
        static_cast<uint64_t>(low) | (static_cast<uint64_t>(static_cast<uint32_t>(value)) << 32);
    __hip_atomic_store(p, raw, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void barrier_arrive_release(int *counter) {
    __hip_atomic_fetch_add(counter, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void barrier_wait_acquire(int *counter, int target) {
    while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target) {
        __builtin_amdgcn_s_sleep(1);
    }
}

// =============================================================================
// Virtual shared memory (vsmem) — spill per-block temp_storage to global
// memory when LDS is too small. Padding each per-block tile to a cache line
// keeps adjacent blocks' stores on disjoint L2 lines.
// =============================================================================

struct vsmem_t {
    void  *gmem_ptr;
    size_t bytes_per_block;
};

inline constexpr size_t kVsmemCacheLineSize = 128;

template <typename T>
__device__ __forceinline__ T *get_temp_storage(T *static_temp_storage, vsmem_t vsmem) {
    if (vsmem.gmem_ptr == nullptr) {
        return static_temp_storage;
    }
    return reinterpret_cast<T *>(static_cast<char *>(vsmem.gmem_ptr) +
                                 vsmem.bytes_per_block * blockIdx.x);
}

// =============================================================================
// Preprocessing kernel.
//
// `kPermutePreprocItemsPerThread` (tokens per thread per tile) is the
// rocPRIM-style items-per-thread fold: each thread carries this many
// consecutive tokens per expert column, doing an in-register pre-scan + 1
// BlockScan over per-thread totals. Tile count, BlockScan calls, and inner
// __syncthreads() each drop by this factor; LDS staging cost scales linearly.
//
// Hard-coded to 2 (removes the previous runtime `pick_kTPT` heuristic). The
// gfx942 sweep recorded in
// `agent/historical_experience/gfx942/permute_preprocessing/turbo/tips.md`
// (round-3 entry) shows `2` is the universal sweet spot:
//
//   * kTPT=4 only wins for E ∈ {1, 4} at very large T (≥ 78K) and overflows
//     the 64 KiB LDS for E ≥ 8 — narrow applicability, large vsmem footprint
//     (each per-block tile doubles in size).
//   * kTPT=2 fits LDS up to E=8 and degrades gracefully via the vsmem
//     fallback for E ≥ 16, where the per-tile global-memory traffic is
//     independent of kTPT (same total tokens × experts staged either way)
//     so the halved BlockScan + sync overhead is a net win.
//   * kTPT=1 leaves the scan/sync overhead on the table.
//
// Pinning to 2 lets us drop the launcher's runtime dispatch + two template
// instantiations of this kernel.
// =============================================================================

inline constexpr int kPermutePreprocItemsPerThread = 1;

template <int kBlockSize>
__launch_bounds__(kBlockSize, 1) __global__
    void permute_preprocessing_kernel(const bool *routing_map, const int *num_dispatched_tokens_ptr,
                                      int num_experts, int pad_multiple, int32_t *tokens_per_expert,
                                      int *row_id_map, int *overflow_flag,
                                      int64_t num_permuted_tokens, uint64_t *tile_state,
                                      int *barrier_p5, uint32_t launch_epoch, vsmem_t vsmem) {
    using BlockScan = hipcub::BlockScan<int32_t, kBlockSize>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    extern __shared__ int                      dyn_shmem[];

    constexpr int kNumItemsPerThread = 2;
    constexpr int kNumItemsPerTile   = kBlockSize * kNumItemsPerThread;

    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto block_id  = static_cast<int>(blockIdx.x);
    const auto grid_size = static_cast<int>(gridDim.x);
    const int  E         = num_experts;

    // Per-block scratch — either LDS or a slice of the global vsmem buffer.
    int *temp_storage  = get_temp_storage<int>(dyn_shmem, vsmem);
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
    // When this block owns exactly one tile, keep `s_tile` in LDS through
    // Phase 6 and skip both the Phase 1-2 row_id_map writeback and the
    // Phase 6 read-back.
    const bool single_tile = (tile_end - tile_begin) == 1;

    const int npt = num_permuted_tokens < 0 ? INT_MAX : static_cast<int>(num_permuted_tokens);

    if (block_id == 0 and thread_id == 0)
        *overflow_flag = 0;
    
    for (int i = thread_id; i < E; i += kBlockSize)
        s_acc[i] = 0;

    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 1-2: per-tile in-block InclusiveSum + accumulate into s_acc.
    // For multi-tile blocks, each iteration also spills the patched
    // partial-scan to row_id_map before s_tile gets reused; single_tile
    // blocks keep s_tile in LDS through Phase 6 (round-4 LDS-fusion).
    // -------------------------------------------------------------------------
    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        const int     tile_offset = tile_idx * kNumItemsPerTile;
        const int64_t tile_base   = static_cast<int64_t>(tile_offset) * E;

        for (int i = thread_id; i < kNumItemsPerTile * E; i += kBlockSize) {
            const int gtoken = tile_offset + i / E;
            s_tile[i] =
                (gtoken < num_dispatched_tokens) ? static_cast<int>(routing_map[tile_base + i]) : 0;
        }
        __syncthreads();

        for (int e = 0; e < E; ++e) {
            int scan_total;
            int local[kNumItemsPerThread];
            int total = 0;
#pragma unroll
            for (int k = 0; k < kNumItemsPerThread; ++k) {
                local[k] = s_tile[(thread_id * kNumItemsPerThread + k) * E + e];
                total += local[k];
            }
            int excl_block;
            BlockScan(scan_temp).ExclusiveSum(total, excl_block, scan_total);
            const int prev    = s_acc[e];
            int       running = excl_block + prev;
#pragma unroll
            for (int k = 0; k < kNumItemsPerThread; ++k) {
                const int v                                          = local[k];
                s_tile[(thread_id * kNumItemsPerThread + k) * E + e] = (v == 1) ? running + 1 : 0;
                running += v;
            }
            if (thread_id == 0) {
                s_acc[e] += scan_total;
            }
            __syncthreads();
        }

        // Multi-tile only: spill the partial scan because s_tile gets reused
        // next iteration. single_tile keeps it in LDS for Phase 6.
        if (not single_tile) {
            for (int i = thread_id; i < kNumItemsPerTile * E; i += kBlockSize) {
                const int gtoken = tile_offset + i / E;
                if (gtoken < num_dispatched_tokens) {
                    row_id_map[tile_base + i] = s_tile[i];
                }
            }
            __syncthreads();
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3: per-expert decoupled lookback. Thread `e` (e < E) publishes
    // (AGGREGATE | s_acc[e]), walks predecessors backward, short-circuits on
    // the first PREFIX, and re-publishes (PREFIX | inclusive prefix).
    // -------------------------------------------------------------------------
    if (thread_id < E) {
        const int      e         = thread_id;
        const int32_t  agg       = s_acc[e];
        const uint32_t init_flag = (block_id == 0) ? TileState::kPrefix : TileState::kAggregate;
        store_tile_state(&tile_state[static_cast<int64_t>(block_id) * E + e], init_flag, agg,
                         launch_epoch);

        int32_t accum = 0;
        for (int b = block_id - 1; b >= 0; --b) {
            TileState s;
            do {
                s = load_tile_state(&tile_state[static_cast<int64_t>(b) * E + e], launch_epoch);
            } while (s.flag == TileState::kInvalid);
            accum += s.value;
            if (s.flag == TileState::kPrefix) {
                break;
            }
        }
        s_excl_prefix[e] = accum;

        if (block_id != 0) {
            store_tile_state(&tile_state[static_cast<int64_t>(block_id) * E + e],
                             TileState::kPrefix, accum + agg, launch_epoch);
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 5: read the full-grid inclusive prefix straight from the LAST
    // block's kPrefix slot (lookback already guaranteed it equals the full
    // grid sum). Then ExclusiveSum across experts to get per-expert padded
    // base offsets. Snapshot into s_acc so Phase 7 sees pre-overflow values
    // even after Phase 8 finalises tokens_per_expert.
    // -------------------------------------------------------------------------
    if (thread_id < E) {
        TileState s;
        do {
            s = load_tile_state(&tile_state[static_cast<int64_t>(grid_size - 1) * E + thread_id],
                                launch_epoch);
        } while (s.flag != TileState::kPrefix);
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

    if (thread_id == 0) {
        barrier_arrive_release(barrier_p5);
    }

    // -------------------------------------------------------------------------
    // Phase 6: patch row_id_map with the lookback prefix + cross-expert prefix.
    //   single_tile : read partial scan straight from LDS (always write).
    //   multi-tile  : read from row_id_map (Phase 1-2 spilled it), only
    //                 write back when the slot was nonzero.
    // -------------------------------------------------------------------------
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

    if (single_tile) {
        const int     tile_offset = tile_begin * kNumItemsPerTile;
        const int64_t tile_base   = static_cast<int64_t>(tile_offset) * E;
        for (int i = thread_id; i < kNumItemsPerTile * E; i += kBlockSize) {
            const int gtoken = tile_offset + i / E;
            if (gtoken < num_dispatched_tokens) {
                row_id_map[tile_base + i] = patch(s_tile[i], i % E);
            }
        }
    } else {
        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int     tile_offset = tile_idx * kNumItemsPerTile;
            const int64_t tile_base   = static_cast<int64_t>(tile_offset) * E;
            for (int i = thread_id; i < kNumItemsPerTile * E; i += kBlockSize) {
                const int gtoken = tile_offset + i / E;
                if (gtoken >= num_dispatched_tokens) {
                    continue;
                }
                const int old = row_id_map[tile_base + i];
                if (old != 0) {
                    row_id_map[tile_base + i] = patch(old, i % E);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 7: padding rows. Each padding row `i` lives at
    // row_id_map[N + i, :] and uses NEGATIVE 1-indexed offsets to signal the
    // data-movement kernel to write zeros. Block-strided over rows, thread-
    // strided over experts → one coalesced store per wavefront.
    // -------------------------------------------------------------------------
    for (int i = block_id; i < pad_multiple; i += grid_size) {
        const int64_t base = (static_cast<int64_t>(i) + num_dispatched_tokens) * E;
        for (int j = thread_id; j < E; j += kBlockSize) {
            int padded_offset = 0;
            if (i < s_num_padded[j]) {
                padded_offset = -(s_acc[j] + s_tpe_prefix[j] + i + 1);
                if (abs(padded_offset) > npt) {
                    *overflow_flag = 1;
                    padded_offset  = 0;
                }
            }
            row_id_map[base + j] = padded_offset;
        }
    }

    // -------------------------------------------------------------------------
    // Phase 8: block 0 finalises tokens_per_expert with overflow handling
    // (must wait for every block's Phase 5 snapshot via barrier_p5), then
    // resets barrier_p5 for the next launch. tile_state is NOT scrubbed —
    // the launch_epoch tag handles cross-launch staleness.
    // -------------------------------------------------------------------------
    if (block_id == 0) {
        if (thread_id == 0) {
            barrier_wait_acquire(barrier_p5, grid_size);
        }
        __syncthreads();
        if (thread_id < E) {
            const int tokens_for_expert = s_acc[thread_id] + s_num_padded[thread_id];
            const int overflow          = tokens_for_expert + s_tpe_prefix[thread_id] - npt;
            tokens_per_expert[thread_id] =
                (overflow < 0) ? tokens_for_expert : max(0, tokens_for_expert - overflow);
        }
        if (thread_id == 0) {
            __hip_atomic_store(barrier_p5, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// =============================================================================
// Preprocessing host-side helpers.
// =============================================================================

namespace {

inline size_t dyn_shmem_bytes(int block_size, int E) {
    // s_tile (block_size * kPermutePreprocItemsPerThread * E) + 4*E ints
    // (s_acc, s_excl_prefix, s_tpe_prefix, s_num_padded) + 2*E*num_warps ints
    // (fused-scan staging).
    const int num_waves = block_size / kWarpSize;
    return (static_cast<size_t>(block_size) * kPermutePreprocItemsPerThread * E +
            4 * static_cast<size_t>(E) + 2 * static_cast<size_t>(E) * num_waves) *
           sizeof(int);
}

inline size_t pad_to_cache_line(size_t bytes) {
    return (bytes + kVsmemCacheLineSize - 1) / kVsmemCacheLineSize * kVsmemCacheLineSize;
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

template <int kBlockSize> inline int max_active_grid(size_t shmem_bytes) {
    static thread_local std::unordered_map<size_t, int> cache;
    auto                                                it = cache.find(shmem_bytes);
    if (it == cache.end()) {
        int max_blocks_per_cu = 0;
        PRIMUS_TURBO_CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_cu,
            reinterpret_cast<const void *>(&permute_preprocessing_kernel<kBlockSize>),
            kBlockSize, shmem_bytes));
        it = cache.emplace(shmem_bytes, max_blocks_per_cu).first;
    }
    return std::max(cached_device_caps().num_cu * it->second, 1);
}

// Per-stream lookback workspace cache. The kernel relies on the launch_epoch
// tag for cross-launch staleness, so we only memset on (a) fresh allocation /
// size grow, (b) shape change (E or grid_size), or (c) epoch wrap (~10^9
// launches per stream).
struct LookbackCacheEntry {
    void    *ptr        = nullptr;
    size_t   size_bytes = 0;
    int      last_E     = 0;
    int      last_grid  = 0;
    uint32_t next_epoch = 1u;
};

inline LookbackCacheEntry &get_lookback_cache(hipStream_t stream) {
    static std::mutex                                          g_mu;
    static std::unordered_map<hipStream_t, LookbackCacheEntry> g_cache;
    std::lock_guard<std::mutex>                                lk(g_mu);
    return g_cache[stream];
}

inline void *acquire_lookback_workspace(hipStream_t stream, size_t needed_bytes, int E,
                                        int grid_size, bool *needs_memset, uint32_t *epoch_out) {
    LookbackCacheEntry &e          = get_lookback_cache(stream);
    bool                must_clear = false;
    if (e.size_bytes < needed_bytes) {
        if (e.ptr != nullptr) {
            PRIMUS_TURBO_CHECK_HIP(hipFreeAsync(e.ptr, stream));
        }
        PRIMUS_TURBO_CHECK_HIP(hipMallocAsync(&e.ptr, needed_bytes, stream));
        e.size_bytes = needed_bytes;
        must_clear   = true;
        e.next_epoch = 1u;
    } else if (E != e.last_E or grid_size != e.last_grid) {
        // Lingering slots from old grid_size / E may sit at coordinates the
        // new launch never visits in Phase 3; epoch tagging alone wouldn't
        // help, so force a memset.
        must_clear   = true;
        e.next_epoch = 1u;
    }
    const uint32_t epoch_cur = e.next_epoch;
    uint32_t       epoch_nxt = (epoch_cur + 1u) & 0x3FFFFFFFu;
    if (epoch_nxt == 0u) {
        // 0 is reserved as the post-memset sentinel; on wrap, scrub and
        // restart at 1.
        must_clear = true;
        epoch_nxt  = 1u;
    }
    e.next_epoch  = epoch_nxt;
    e.last_E      = E;
    e.last_grid   = grid_size;
    *needs_memset = must_clear;
    *epoch_out    = epoch_cur;
    return e.ptr;
}

} // anonymous namespace

void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int max_num_dispatched_tokens,
                                  int pad_multiple, int32_t *tokens_per_expert, int *row_id_map,
                                  int *overflow_flag, int64_t num_permuted_tokens,
                                  hipStream_t stream) {
    constexpr int kBlockSize = PermutePreprocessConfig::kBlockSize;
    PRIMUS_TURBO_CHECK(num_of_local_experts > 0, "num_of_local_experts must be > 0");
    PRIMUS_TURBO_CHECK(num_of_local_experts <= kBlockSize,
                       "num_of_local_experts must fit in a single block");
    PRIMUS_TURBO_CHECK(max_num_dispatched_tokens > 0, "max_num_dispatched_tokens must be > 0");

    const auto  &caps     = cached_device_caps();
    const int    E        = num_of_local_experts;
    const size_t required = dyn_shmem_bytes(kBlockSize, E);

    // vsmem fallback when the per-block temp_storage exceeds the LDS budget.
    // With the items-per-thread fold pinned at 2, this triggers for E ≳ 16
    // on gfx942's 64 KiB LDS.
    const bool   use_vsmem    = required > static_cast<size_t>(caps.max_shmem_per_block);
    const size_t kernel_shmem = use_vsmem ? 0 : required;
    const size_t vsmem_bpb    = use_vsmem ? pad_to_cache_line(required) : 0;

    const int per_tile      = kBlockSize * kPermutePreprocItemsPerThread;
    const int internal_rows = (max_num_dispatched_tokens + per_tile - 1) / per_tile;

    // Cap gridDim.x at the device's max-active-blocks count: the lookback
    // chain busy-waits on predecessor flags, so all blocks must be co-resident.
    const int max_grid  = max_active_grid<kBlockSize>(kernel_shmem);
    const int grid_size = std::min(max_grid, internal_rows);

    // Lookback workspace layout:
    //   tile_state : grid_size * E uint64 (epoch-tagged)
    //   barrier_p5 : 1 int
    //   vsmem      : grid_size * vsmem_bpb bytes (only when use_vsmem)
    const size_t tile_state_bytes = static_cast<size_t>(grid_size) * E * sizeof(uint64_t);
    const size_t base_bytes       = tile_state_bytes + sizeof(int);
    const size_t vsmem_offset     = use_vsmem ? pad_to_cache_line(base_bytes) : base_bytes;
    const size_t vsmem_total      = static_cast<size_t>(grid_size) * vsmem_bpb;
    const size_t scratch_bytes    = vsmem_offset + vsmem_total;
    bool         needs_memset     = false;
    uint32_t     launch_epoch     = 1u;
    void *scratch = acquire_lookback_workspace(stream, scratch_bytes, E, grid_size, &needs_memset,
                                               &launch_epoch);
    if (needs_memset) {
        // Only the lookback portion needs zeroing; the kernel always writes
        // each vsmem tile before reading from it.
        PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(scratch, 0, base_bytes, stream));
    }

    auto *tile_state = reinterpret_cast<uint64_t *>(scratch);
    auto *barrier_p5 =
        reinterpret_cast<int *>(reinterpret_cast<char *>(scratch) + tile_state_bytes);

    vsmem_t vsmem{};
    if (use_vsmem) {
        vsmem.gmem_ptr        = reinterpret_cast<char *>(scratch) + vsmem_offset;
        vsmem.bytes_per_block = vsmem_bpb;
    }

    permute_preprocessing_kernel<kBlockSize><<<grid_size, kBlockSize, kernel_shmem, stream>>>(
        routing_map, num_dispatched_tokens_ptr, E, pad_multiple, tokens_per_expert, row_id_map,
        overflow_flag, num_permuted_tokens, tile_state, barrier_p5, launch_epoch, vsmem);

    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

// =============================================================================
// Data movement: permute (gather) and unpermute (scatter+reduce).
// =============================================================================

// `permute_kernel` runs one wavefront per token so the inner loop dispatches
// directly through `UNROLLED_WARP_COPY` (mirrors the DeepEP intranode
// dispatch loop). `unpermute_kernel` runs `kUnpermuteWarpsPerToken` warps
// per token because its float-accumulator inner loop benefits from a wider
// per-token stride.
//
// **R18 rolled back**: tried `kUnpermuteWarpsPerToken = 1` to double
// `num_tokens_per_block` (8 → 16) and amortise the routing-tile LDS load
// further. Result: −2.5 % avg on quick (Qwen3-30B-A3B/E=16/h=2048
// −6.6 %). Root cause: with a single warp per token the K-loop body
// only issues 2 vmem requests per K-iter (j + j2) instead of 4 (R17 has
// 2 warps × 2 j positions = 4), so per-warp work density drops below
// the compiler's SW-pipe threshold (`vmcnt(2)/(3)` count fell). At
// kBlockSize = 1024 the optimum is `kUnpermuteWarpsPerToken = 2`.
inline constexpr int kPermuteCopyUnrollFactor = 4;
inline constexpr int kUnpermuteWarpsPerToken  = 2;

// Stage `row_id_map[block_start, :]` through LDS so all warps in the block
// get fast random access.
template <int kBlockSize>
__device__ __forceinline__ void load_routing_tile(int *expert_routing_map, const int *row_id_map,
                                                  int64_t block_start, int tokens_per_block,
                                                  int num_experts, int num_dispatched_tokens) {
    const auto thread_id = static_cast<int>(threadIdx.x);
    for (int i = thread_id; i < num_experts * tokens_per_block; i += kBlockSize) {
        expert_routing_map[i] = (block_start + i / num_experts < num_dispatched_tokens)
                                    ? row_id_map[block_start * num_experts + i]
                                    : 0;
    }
}

template <typename DType> __device__ __forceinline__ float to_float(DType x) {
    return static_cast<float>(x);
}

template <typename DType> __device__ __forceinline__ DType from_float(float x) {
    return static_cast<DType>(x);
}

// =============================================================================
// permute_kernel: gather tokens into expert-grouped order.
//   row_id_map[token, expert] > 0  → write into permuted slot (value-1)
//   row_id_map[token, expert] < 0  → write zeros to slot (-value-1)
//   row_id_map[token, expert] == 0 → skip
// =============================================================================

template <int kUnrollFactor, int kBlockSize, typename ProbType, typename ScalarType>
__global__ void
permute_kernel(const int4 *tokens, int4 *permuted_tokens, const ScalarType *scaling_factor,
               ScalarType *permuted_scaling_factor, const ProbType *probs, ProbType *permuted_probs,
               const int *row_id_map, const int *num_dispatched_tokens_ptr, int pad_multiple,
               int num_of_local_experts, int hidden_int4, int scales_per_token, int local_rank,
               int num_ranks_per_node) {
    // One warp owns one token, so per-block warp count == per-block token count.
    constexpr int num_warps = kBlockSize / kWarpSize;

    const auto thread_id             = static_cast<int>(threadIdx.x);
    const auto lane_id               = thread_id % kWarpSize;
    const auto warp_id               = thread_id / kWarpSize;
    const int  E                     = num_of_local_experts;
    const int  num_dispatched_tokens = *num_dispatched_tokens_ptr + pad_multiple;

    extern __shared__ int shared_buf[];
    int                  *expert_routing_map = shared_buf;

    for (int64_t block_start = blockIdx.x * num_warps; block_start < num_dispatched_tokens;
         block_start += static_cast<int64_t>(num_warps) * gridDim.x) {
        const int64_t token_id = block_start + warp_id;

        load_routing_tile<kBlockSize>(expert_routing_map, row_id_map, block_start, num_warps, E,
                                      num_dispatched_tokens);
        __syncthreads();

        if (token_id >= num_dispatched_tokens) {
            __syncthreads();
            continue;
        }

        // Token data.
        const int4 *src_tokens = tokens + token_id * hidden_int4;
        for (int i = 0; i < E; ++i) {
            const int64_t dest_token_id = expert_routing_map[warp_id * E + i];
            if (dest_token_id > 0) {
                int4 *dst = permuted_tokens + (dest_token_id - 1) * hidden_int4;
                UNROLLED_WARP_COPY(kUnrollFactor, lane_id, hidden_int4, dst, src_tokens, __ldg,
                                   st_na_global);
            } else if (dest_token_id < 0) {
                int4      *dst   = permuted_tokens + (-dest_token_id - 1) * hidden_int4;
                const int4 zero4 = make_int4(0, 0, 0, 0);
                for (int64_t j = lane_id; j < hidden_int4; j += kWarpSize) {
                    st_na_global(dst + j, zero4);
                }
            }
        }

        if (scaling_factor != nullptr) {
            for (int i = 0; i < E; ++i) {
                const int64_t dest_token_id = expert_routing_map[warp_id * E + i];
                if (dest_token_id > 0) {
                    for (int64_t j = lane_id; j < scales_per_token; j += kWarpSize) {
                        permuted_scaling_factor[(dest_token_id - 1) * scales_per_token + j] =
                            scaling_factor[token_id * scales_per_token + j];
                    }
                } else if (dest_token_id < 0) {
                    for (int64_t j = lane_id; j < scales_per_token; j += kWarpSize) {
                        permuted_scaling_factor[(-dest_token_id - 1) * scales_per_token + j] =
                            ScalarType{0};
                    }
                }
            }
        }

        if (probs != nullptr) {
            for (int i = 0; i < E; ++i) {
                const int64_t dest_token_id = expert_routing_map[warp_id * E + i];
                if (dest_token_id > 0) {
                    permuted_probs[dest_token_id - 1] =
                        probs[token_id * E * num_ranks_per_node + local_rank * E + i];
                } else if (dest_token_id < 0) {
                    permuted_probs[-dest_token_id - 1] = ProbType{0};
                }
            }
        }

        __syncthreads();
    }
}

// =============================================================================
// unpermute_kernel: scatter+reduce permuted tokens back to per-source rows.
//
// Round-14 simplification — the `bool kHoist` template parameter is gone:
//
//   * E == 1 dispatches to `unpermute_kernel_e1` (round-13). That kernel
//     drops the float-accumulator, the E-walk, and the multi-warp cooperation
//     entirely — the cases R5's `kHoist=false` lean path was originally
//     introduced to protect (R4 -3.88 % on E=1 from VGPR-pressure) are now
//     handled by a structurally specialised kernel rather than a register-
//     allocation tradeoff inside the generic kernel.
//
//   * Every other `num_of_local_experts >= 2` shape goes through the single
//     hoisted + j-tile-2 + depth-2-software-pipeline kernel below. Pre-R14
//     R5 used `kUnpermuteHoistMinExperts = 5` to keep E=2 / E=4 on the lean
//     path because R4 measured -6.41 % on E=2 and -0.88 % on E=4 from the
//     hoist VGPR pressure. After R13 lifts the E=1 burden out of the generic
//     kernel, that protection is no longer load-bearing — the hoist kernel
//     has slightly more VGPR slack on E=2 / E=4 (positive_srcs at most 2 of
//     8 slots used; the unused tail is dead-coded by the unrolled `n+1 <
//     n_pos_clamped` branches), and the ICache pressure of carrying two
//     kernel binaries is removed. R6 / R7 / R9 optimisations
//     (depth-2 SW pipe + j-tile widening) all live inside this kernel and
//     now apply uniformly across the E ≥ 2 shapes.
// =============================================================================

// Per-token positive source upper bound `K_local <= min(num_topk, E_local)`.
// 8 covers all current MoE configs (Mixtral=2, DeepSeek=8, Qwen3=8, Kimi-K2=8,
// Grok-2=2, etc.). Overflow falls through to a cold runtime-conditional path.
inline constexpr int kUnpermuteSourceMax = 8;

// Round-13 E=1 specialised kernel: 1 warp per token, UNROLLED_WARP_COPY-style
// gather-copy. No float accumulator (single source ⇒ no reduction). No E-walk
// (`row_id_map` is shape `[1, T]` ⇒ a single scalar load per token). No multi-
// warp cooperation. `kPermuteCopyUnrollFactor` (= 4) matches the sibling
// `permute_kernel`, which already saturates ~80 % HBM3 peak on this shape.
template <int kUnrollFactor, int kBlockSize, typename DType, typename ProbType>
__global__ void unpermute_kernel_e1(const int4 *permuted_tokens, int4 *tokens,
                                    const ProbType *permuted_probs, ProbType *probs,
                                    const int *row_id_map, const int *num_dispatched_tokens_ptr,
                                    int hidden_int4, int local_rank, int num_ranks_per_node) {
    constexpr int num_warps = kBlockSize / kWarpSize;

    const auto thread_id             = static_cast<int>(threadIdx.x);
    const auto lane_id               = thread_id % kWarpSize;
    const auto warp_id               = thread_id / kWarpSize;
    const int  num_dispatched_tokens = *num_dispatched_tokens_ptr;

    // Persistent grid: each block strides through the dispatched-tokens axis
    // by `num_warps * gridDim.x`. Mirrors `permute_kernel`'s scheduling.
    for (int64_t block_start = blockIdx.x * num_warps; block_start < num_dispatched_tokens;
         block_start += static_cast<int64_t>(num_warps) * gridDim.x) {
        const int64_t token_id = block_start + warp_id;
        if (token_id >= num_dispatched_tokens)
            continue;

        // E=1 routing slot. `row_id_map` shape collapses to `[1, T]` ⇒ a
        // single int per token. `s > 0` ⇒ gather `permuted_tokens[s-1]`;
        // `s <= 0` ⇒ this output row receives no source contribution and
        // must be zero-initialised (matches the generic kernel's behaviour:
        // accumulator starts at 0 and the `s <= 0` source is skipped).
        const int s   = row_id_map[token_id];
        int4     *dst = tokens + token_id * hidden_int4;
        if (s > 0) {
            const int4 *src = permuted_tokens + (s - 1) * hidden_int4;
            // Plain `__ldg` matches `permute_kernel`. The R2 negative result
            // for `__ldg` on the *generic* unpermute kernel (-1.36 % gmean,
            // tip: "Mixtral E=1 unchanged at +0.07 %") was driven by the
            // high-E shapes where each source row is read once across the
            // launch and the read-only L1 layer adds latency. On E=1 the
            // structural pattern is identical to permute_kernel (single
            // source per token), and `__ldg` was empirically neutral on
            // Mixtral E=1 in R2 — keep it for parity with permute_kernel.
            UNROLLED_WARP_COPY(kUnrollFactor, lane_id, hidden_int4, dst, src, __ldg, st_na_global);
        } else {
            const int4 zero4 = make_int4(0, 0, 0, 0);
            for (int64_t j = lane_id; j < hidden_int4; j += kWarpSize) {
                st_na_global(dst + j, zero4);
            }
        }

        // Probs side-output. Generic kernel writes `probs[t, j]` for j in
        // `[0, E * num_ranks_per_node)`; for E=1 the output collapses to
        // shape `[T, num_ranks_per_node]` and the only non-zero slot per
        // token is `probs[t, local_rank] = permuted_probs[s-1]` (when s > 0).
        if (probs != nullptr && permuted_probs != nullptr) {
            // All lanes broadcast-load the same scalar (1 transaction). The
            // value is replicated across the warp's VGPRs.
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

template <int kBlockSize, typename DType, typename ProbType>
__global__ void unpermute_kernel(const int4 *permuted_tokens, int4 *tokens,
                                 const ProbType *permuted_probs, ProbType *probs,
                                 const int *row_id_map, const int *num_dispatched_tokens_ptr,
                                 int num_of_local_experts, int hidden_int4, int local_rank,
                                 int num_ranks_per_node) {
    // `kUnpermuteWarpsPerToken` warps cooperate on each token's reduction.
    // The cooperative group is `kUnpermuteWarpsPerToken * kWarpSize` threads
    // wide; per-block we hold `num_warps / kUnpermuteWarpsPerToken` tokens.
    constexpr int kWarpsPerToken       = kUnpermuteWarpsPerToken;
    constexpr int num_warps            = kBlockSize / kWarpSize;
    constexpr int num_tokens_per_block = num_warps / kWarpsPerToken;
    constexpr int num_eles_per_pack    = sizeof(int4) / sizeof(DType);

    const auto thread_id         = static_cast<int>(threadIdx.x);
    const auto lane_id           = thread_id % kWarpSize;
    const auto warp_id           = thread_id / kWarpSize;
    const auto warp_id_in_token  = warp_id % kWarpsPerToken;
    const auto token_id_in_block = warp_id / kWarpsPerToken;

    const int E                     = num_of_local_experts;
    const int num_dispatched_tokens = *num_dispatched_tokens_ptr;

    extern __shared__ int shared_buf[];
    int                  *expert_routing_map = shared_buf;

    for (int64_t block_start = blockIdx.x * num_tokens_per_block;
         block_start < num_dispatched_tokens;
         block_start += static_cast<int64_t>(num_tokens_per_block) * gridDim.x) {
        const int64_t token_id = block_start + token_id_in_block;

        load_routing_tile<kBlockSize>(expert_routing_map, row_id_map, block_start,
                                      num_tokens_per_block, E, num_dispatched_tokens);
        __syncthreads();

        if (token_id >= num_dispatched_tokens) {
            __syncthreads();
            continue;
        }

        int4   buffer_pack;
        DType *buffer_ptr = reinterpret_cast<DType *>(&buffer_pack);
        float  accumulator[num_eles_per_pack];

        // Pre-collect positive source offsets. Overflow (rare) falls
        // through to the conditional in-loop fallback below.
        int positive_srcs[kUnpermuteSourceMax];
        int n_pos          = 0;
        int n_pos_overflow = 0;
        for (int i = 0; i < E; ++i) {
            const int s = expert_routing_map[token_id_in_block * E + i];
            if (s > 0) {
                if (n_pos < kUnpermuteSourceMax) {
                    positive_srcs[n_pos] = s - 1;
                } else {
                    n_pos_overflow = 1;
                }
                ++n_pos;
            }
        }

        // Round-16: deep_ep `intranode.cu` combine-receiver-style packed-sparse
        // K-loop. R14 used a compile-time `for (n = 0; n < kUnpermuteSourceMax;
        // ++n) #pragma unroll` body with `if (n < n_pos_clamped)` and
        // `if (n + 1 < n_pos_clamped)` predicates inside the unrolled body —
        // the compiler emitted 78 pairs of `s_and_saveexec_b64 / s_or_b64
        // exec` (160 EXEC scaffolding ops total) and 78 `v_cndmask`s vs only
        // 72 actual `v_pk_add_f32`s, a 2.22 EXEC/ALU overhead ratio. R11's
        // lesson generalised: wave-uniform predicates inside `#pragma
        // unroll` bodies are NOT optimised to scalar branches on AMDGPU —
        // each unrolled instance gets its own EXEC mask scaffolding.
        //
        // The fix: replace the static-unroll-with-branches K-loop with a
        // runtime-bounded native loop that walks exactly K_local sources
        // (the deep_ep pattern: `for (j = 0; j < num_topk_ranks; ++j)` with
        // `#pragma unroll N` as a hint, not a hard unroll). The explicit
        // SW-pipeline buffers (R6/R7) and j-tile-prefetch buffers (R9
        // inner) are dropped — they only made sense when `n` was a
        // compile-time constant. The compiler now schedules across the
        // runtime loop body. j-tile 2× outer structure (R9 surface) is
        // retained: each thread still handles `j` and `j2 = j +
        // cooperative_stride` per iteration.
        constexpr int     kJTile             = 2;
        constexpr int64_t cooperative_stride = kWarpsPerToken * kWarpSize;
        float             accumulator2[num_eles_per_pack];

        const int64_t step = kJTile * cooperative_stride;
        for (int64_t j = warp_id_in_token * kWarpSize + lane_id; j < hidden_int4; j += step) {
            const int64_t j2     = j + cooperative_stride;
            const bool    has_j2 = (j2 < hidden_int4);
#pragma unroll
            for (int k = 0; k < num_eles_per_pack; ++k) {
                accumulator[k]  = 0.0f;
                accumulator2[k] = 0.0f;
            }

            if (!n_pos_overflow) {
                const int K_local = (n_pos < kUnpermuteSourceMax) ? n_pos : kUnpermuteSourceMax;

                // Wave-uniform branch hoisted OUTSIDE the K-loop so the
                // K-loop body is single-block and the compiler does not
                // duplicate the prefetch/consume into the EXEC-mask scheduler.
                if (has_j2) {
#pragma unroll 2
                    for (int n = 0; n < K_local; ++n) {
                        const int    src = positive_srcs[n];
                        const int4   v0  = permuted_tokens[src * hidden_int4 + j];
                        const int4   v1  = permuted_tokens[src * hidden_int4 + j2];
                        const DType *p0  = reinterpret_cast<const DType *>(&v0);
                        const DType *p1  = reinterpret_cast<const DType *>(&v1);
#pragma unroll
                        for (int k = 0; k < num_eles_per_pack; ++k) {
                            accumulator[k] += to_float<DType>(p0[k]);
                            accumulator2[k] += to_float<DType>(p1[k]);
                        }
                    }
                } else {
#pragma unroll 2
                    for (int n = 0; n < K_local; ++n) {
                        const int    src = positive_srcs[n];
                        const int4   v0  = permuted_tokens[src * hidden_int4 + j];
                        const DType *p0  = reinterpret_cast<const DType *>(&v0);
#pragma unroll
                        for (int k = 0; k < num_eles_per_pack; ++k) {
                            accumulator[k] += to_float<DType>(p0[k]);
                        }
                    }
                }
            } else {
                // Cold fallback for K_local > kUnpermuteSourceMax.
                for (int i = 0; i < E; ++i) {
                    const int64_t source_token_id = expert_routing_map[token_id_in_block * E + i];
                    if (source_token_id <= 0)
                        continue;
                    buffer_pack = permuted_tokens[(source_token_id - 1) * hidden_int4 + j];
#pragma unroll
                    for (int k = 0; k < num_eles_per_pack; ++k) {
                        accumulator[k] += to_float<DType>(buffer_ptr[k]);
                    }
                    if (has_j2) {
                        buffer_pack = permuted_tokens[(source_token_id - 1) * hidden_int4 + j2];
#pragma unroll
                        for (int k = 0; k < num_eles_per_pack; ++k) {
                            accumulator2[k] += to_float<DType>(buffer_ptr[k]);
                        }
                    }
                }
            }

#pragma unroll
            for (int k = 0; k < num_eles_per_pack; ++k) {
                buffer_ptr[k] = from_float<DType>(accumulator[k]);
            }
            tokens[token_id * hidden_int4 + j] = buffer_pack;
            if (has_j2) {
#pragma unroll
                for (int k = 0; k < num_eles_per_pack; ++k) {
                    buffer_ptr[k] = from_float<DType>(accumulator2[k]);
                }
                tokens[token_id * hidden_int4 + j2] = buffer_pack;
            }
        }

        if (permuted_probs != nullptr) {
            for (int64_t j = warp_id_in_token * kWarpSize + lane_id; j < E * num_ranks_per_node;
                 j += kWarpsPerToken * kWarpSize) {
                float value = 0.0f;
                if (j / E == local_rank) {
                    const int64_t source_token_id =
                        expert_routing_map[token_id_in_block * E + j % E];
                    if (source_token_id > 0) {
                        value = static_cast<float>(permuted_probs[source_token_id - 1]);
                    }
                }
                probs[token_id * E * num_ranks_per_node + j] = static_cast<ProbType>(value);
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

    // shmem stages `row_id_map[block_start, :]` for the `num_warps` tokens
    // this block owns.
    const size_t shmem_bytes = static_cast<size_t>(num_of_local_experts) * num_warps * sizeof(int);

    // Pre-cast tokens / permuted_tokens to int4 once on the host (mirrors
    // DeepEP intranode dispatch's dtype-agnostic packed view).
    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *tokens_int4          = reinterpret_cast<const int4 *>(tokens);
    int4       *permuted_tokens_int4 = reinterpret_cast<int4 *>(permuted_tokens);

    permute_kernel<kPermuteCopyUnrollFactor, kBlockSize, ProbType, ScalarType>
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
    // Round-17: experiment — bump the *generic* unpermute kernel's block size
    // from 512 → 1024 to test whether trading occupancy (24 → 16 active
    // waves/CU at VGPR=72 from R16) for a larger per-block token batch
    // (num_tokens_per_block 4 → 8) and more concurrent vmem fan-out per CU
    // wins on this kernel. The E=1 specialised kernel (`unpermute_kernel_e1`,
    // R13) keeps its symmetric kBlockSize=512 with `permute_kernel`; only the
    // K-reduction generic path moves to 1024.
    constexpr int kBlockSize           = PermuteKernelConfig::kBlockSize; // 512, used by e1
    constexpr int kUnpermuteBlockSize  = 1024;                            // R17 generic
    constexpr int num_warps            = kUnpermuteBlockSize / kWarpSize;
    constexpr int num_tokens_per_block = num_warps / kUnpermuteWarpsPerToken;
    constexpr int num_eles_per_pack    = sizeof(int4) / sizeof(DType);

    PRIMUS_TURBO_CHECK(tokens != nullptr, "tokens output must be allocated");
    PRIMUS_TURBO_CHECK(hidden_size % num_eles_per_pack == 0,
                       "hidden_size must be a multiple of (16 / sizeof(DType))");
    PRIMUS_TURBO_CHECK(grid_size > 0, "grid_size must be > 0");

    // shmem stages `row_id_map[block_start, :]` for the `num_tokens_per_block`
    // tokens this block reduces. With kBlockSize=1024 / kWarpsPerToken=2,
    // num_tokens_per_block = 8, so peak shmem = 8 * E * 4B = 1.5 KiB at E=48.
    const size_t shmem_bytes =
        static_cast<size_t>(num_of_local_experts) * num_tokens_per_block * sizeof(int);

    // Mirror permute_impl: pre-cast to int4 + compute hidden_int4 once on the
    // host so the kernel takes a dtype-agnostic packed view.
    const int   hidden_int4          = hidden_size / num_eles_per_pack;
    const int4 *permuted_tokens_int4 = reinterpret_cast<const int4 *>(permuted_tokens);
    int4       *tokens_int4          = reinterpret_cast<int4 *>(tokens);

    // Round-13/14 dispatch:
    //   * `num_of_local_experts == 1` ⇒ structurally specialised gather-copy
    //     kernel (round-13). No float reduce, no E-walk, 1 warp / token.
    //     Block size stays at 512 to mirror `permute_kernel`'s shape.
    //   * `num_of_local_experts >= 2` ⇒ generic packed-sparse runtime K-loop
    //     kernel (R16). Block size bumped to 1024 (R17 experiment).
    if (num_of_local_experts == 1) {
        constexpr int kUnrollFactor = kPermuteCopyUnrollFactor;
        unpermute_kernel_e1<kUnrollFactor, kBlockSize, DType, ProbType>
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
