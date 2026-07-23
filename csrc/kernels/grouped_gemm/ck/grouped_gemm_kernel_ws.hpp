// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Vendored work-stealing variant of ck_tile::GroupedGemmKernel. Inherits all
// host-side machinery (BlockSize, MakeKargs, IsSupportedArgument, ...) and
// the device-side per-tile compute (Run) from the upstream kernel; replaces
// the persistent operator() with an atomicAdd-based tile claim.
//
// Source: 3rdparty/composable_kernel/include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp
// (specifically the persistent `operator()` at lines 540-573).
//
// The static-stride persistent loop (block_id += grid_size) gives no
// straggler tolerance: when one CU is preempted by RCCL, the kernel waits
// for it to grind through its full quota of tiles. The work-stealing variant
// has each CU claim the next tile via atomicAdd, so faster CUs absorb work
// that slow CUs would otherwise have done.
//
// Counter layout (caller-allocated int32 buffer of size NUM_XCDS_WS + 2):
//   [0..NUM_XCDS_WS-1] : per-XCD slots
//   [NUM_XCDS_WS]      : global slot
//   [NUM_XCDS_WS + 1]  : done slot (last-out CTA detection for self-reset)
// `local_per_xcd` selects the mode:
//   = 0                                : global-only (phase 1 empty)
//   = ceil(total_tiles / NUM_XCDS_WS)  : per-XCD-only (phase 2 empty)
//   = anything in-between              : hierarchical (some local + global tail)
//
// Self-reset: the kernel guarantees the entire counter buffer is back to 0
// before it exits. The last CTA out (detected via atomicAdd on the done slot)
// writes zeros to all slots. Stream ordering then makes those zeros visible
// to the next kernel launch -- no host-side `counter.zero_()` needed.

#pragma once

#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

#include <hip/hip_runtime.h>

namespace ck_tile {

// MI355X / MI350 chiplet count. Used to size the counter buffer and to map
// pid -> xcd_id (round-robin via pid % NUM_XCDS_WS).
constexpr index_t NUM_XCDS_WS = 8;

// (`block_start`/`block_end` are populated by primus's
// `compute_grouped_gemm_args` kernel, which the runner extends with a
// prefix-sum loop using the runner's tile shape. The WS kernel's
// FindGroupId binary search reads those fields.)

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GroupedGemmKernelWS
    : public GroupedGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_> {
    using Base = GroupedGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;
    using Self = GroupedGemmKernelWS<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    using TilePartitioner                        = typename Base::TilePartitioner;
    using OffsetTile1DPartitioner                = typename Base::OffsetTile1DPartitioner;
    static constexpr index_t NumDTensor_         = Base::NumDTensor_;
    static constexpr index_t kBlockSize          = Base::kBlockSize;
    static constexpr bool    UsePersistentKernel = Base::UsePersistentKernel;

    static_assert(UsePersistentKernel, "GroupedGemmKernelWS requires a persistent GemmPipeline.");

    // Override the host-side occupancy query to match the WS kernel's
    // 4-arg launch signature.
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config &s) -> dim3 {
        using ConstantPointer = const void CK_TILE_CONSTANT_ADDRESS_SPACE *;
        const auto kernel     = kentry<1, Self, ConstantPointer, index_t, int32_t *, index_t>;
        int        occupancy;
        HIP_CHECK_ERROR(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, kBlockSize, 0));
        const int grid_size = get_available_compute_units(s) * occupancy;
        return dim3(grid_size, 1, 1);
    }

    // Hierarchical WS: each CTA first drains its XCD's local counter for up
    // to `local_per_xcd` claims, then falls back to the global counter for
    // the ragged tail. Speculative prefetch hides per-claim atomic latency
    // under Run; FindGroupId is O(log G) and reads precomputed
    // [block_start, block_end) populated by primus's args-setup kernel.
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE *gemm_descs_const,
                                   const index_t group_count, int32_t *tile_counter_ptr,
                                   const index_t local_per_xcd) const {
        // gfx942 has 64 KB LDS per CU and CK's 256x256x64 tile config already
        // uses the full 64 KB. Our WS kernel adds a 4-byte ``__shared__``
        // broadcast slot for the atomicAdd result (see s_block_id below),
        // which pushes total LDS to 65540 bytes and hits the lld error
        // ``local memory (65540) exceeds limit (65536)``. WS was designed
        // and tuned for gfx950 (MI355X) where the CK static kernel leaves
        // enough LDS headroom for the extra slot. On gfx942 we compile an
        // empty stub; the Python dispatcher's ``can_handle`` refuses to
        // route ``schedule="work_steal"`` to CK on gfx942, so the stub is
        // never actually launched -- it exists only to make the gfx942
        // device object link successfully.
#if defined(__gfx942__)
        (void) gemm_descs_const;
        (void) group_count;
        (void) tile_counter_ptr;
        (void) local_per_xcd;
#else
        const auto gemm_desc_ptr = reinterpret_cast<const GemmTransKernelArg<NumDTensor_> *>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        // Total tiles across all groups. ``block_start``/``block_end`` are
        // populated as a prefix sum by ``compute_grouped_gemm_args`` in
        // ck_grouped_gemm.cu, so the last group's ``block_end`` is
        // exactly the cumulative tile count. One field read instead of
        // an O(G) recompute per CTA.
        const index_t total_tiles = gemm_desc_ptr[group_count - 1].block_end;

        // AMD round-robin pid -> xcd_id mapping. When the persistent grid is
        // capped to fewer CUs than there are XCDs (gridDim.x < NUM_XCDS_WS),
        // only XCD ids [0, gridDim.x) ever issue phase-1 claims, so both the
        // per-XCD slot count AND the phase-1 ID span must use
        // min(gridDim.x, NUM_XCDS_WS). Otherwise phase 2 starts past where
        // phase 1 actually ended and the tiles in the gap are silently
        // dropped. The public ``grouped_gemm`` API rejects num_cu != None +
        // schedule="work_steal", so callers should never hit this branch
        // from the high-level op, but the kernel-level binding still exposes
        // num_cu -- belt-and-braces.
        const index_t  active_xcds    = min(static_cast<index_t>(gridDim.x), NUM_XCDS_WS);
        const index_t  xcd_id         = blockIdx.x % active_xcds;
        int32_t *const local_counter  = tile_counter_ptr + xcd_id;
        int32_t *const global_counter = tile_counter_ptr + NUM_XCDS_WS;
        const index_t  phase1_total   = local_per_xcd * active_xcds;

        // Single claim slot -- no speculative prefetch. Originally this kernel
        // double-buffered the next claim while the current Run was in flight,
        // hoping to hide atomic latency under the GEMM. Under heavy contention
        // (small shapes / global mode) that doubles the atomic queue depth at
        // the L2 -- each CTA has *two* atomics in flight at once -- which
        // measurably increases per-atomic stall (TCC_EA0_ATOMIC_LEVEL_sum was
        // 2.7x Triton's for the same atomic count). Holding to one atomic per
        // claim halves the peak queue depth at modest cost to dense shapes
        // (where speculation would have been useful but contention is already
        // amortized).
        __shared__ int32_t s_block_id;

        // -- Phase 1: per-XCD claims ----------------------------------------
        //
        // ``s_block_id`` is written by thread 0 and read by all threads.
        // It needs a barrier on both sides of the write: after (so all
        // threads see the new value -- the mid-loop ``__syncthreads()``)
        // and before (so a straggler warp still reading the previous
        // iteration's value isn't clobbered). On the normal path the
        // trailing ``block_sync_lds()`` after ``Run()`` in iteration N
        // provides the "before" for iteration N+1, but two exit paths
        // skip that trailing barrier and let the next write race the
        // previous read: the ``continue`` on ``block_id >= total_tiles``
        // and the Phase-1 ``break`` that falls into Phase 2's atomicAdd.
        // The top-of-loop ``__syncthreads()`` plugs both. Loop-exit
        // conditions are block-uniform (derived from ``s_block_id``
        // alone), so this barrier cannot diverge.
        while (true) {
            __syncthreads();
            if (threadIdx.x == 0) {
                s_block_id = atomicAdd(local_counter, 1);
            }
            __syncthreads();
            const index_t local_idx = s_block_id;
            if (local_idx >= local_per_xcd) {
                break;
            }
            const index_t block_id = xcd_id * local_per_xcd + local_idx;
            // Bound check: per-XCD-only mode (local_per_xcd = ceil(t/N))
            // can produce block_ids past total_tiles for the last XCD.
            if (block_id >= total_tiles) {
                continue;
            }

            const index_t group_id = this->FindGroupId(gemm_desc_ptr, block_id, group_count);
            const auto   &kargs    = gemm_desc_ptr[group_id];
            const auto    grid_size_2d =
                TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);
            const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
                0, kargs.group_karg.M, kargs.group_karg.N,
                (block_id - kargs.block_start) % grid_size_2d);
            this->Run(kargs.group_karg, block_idx_2d,
                      (block_id - kargs.block_start) / grid_size_2d);
            block_sync_lds();
        }

        // -- Phase 2: global fallback ----------------------------------------
        // Same top-of-loop ``__syncthreads()`` rationale as Phase 1.
        while (true) {
            __syncthreads();
            if (threadIdx.x == 0) {
                s_block_id = atomicAdd(global_counter, 1);
            }
            __syncthreads();
            const index_t g_idx    = s_block_id;
            const index_t block_id = phase1_total + g_idx;
            if (block_id >= total_tiles) {
                break;
            }

            const index_t group_id = this->FindGroupId(gemm_desc_ptr, block_id, group_count);
            const auto   &kargs    = gemm_desc_ptr[group_id];
            const auto    grid_size_2d =
                TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);
            const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
                0, kargs.group_karg.M, kargs.group_karg.N,
                (block_id - kargs.block_start) % grid_size_2d);
            this->Run(kargs.group_karg, block_idx_2d,
                      (block_id - kargs.block_start) / grid_size_2d);
            block_sync_lds();
        }

        // -- Self-reset: last-out CTA zeros the counter buffer --------------
        // Detect which CTA is last to exit by atomically incrementing a
        // dedicated "done" slot. The CTA that brings the count to gridDim.x
        // (i.e. sees gridDim.x - 1 returned) is the last one -- all other CTAs
        // have already finished claiming and running their tiles, so it is
        // safe to zero the work counters.
        //
        // Memory ordering: the atomicAdd on done_counter below is acquire+release
        // in HIP's relaxed-equivalent semantics, providing the necessary
        // happens-before relationship with the prior atomicAdds on the work
        // counters by all other CTAs. Within this launch, no other CTA will
        // read or write the work counters after we observe the last-out
        // condition, so plain stores to the work slots are race-free. Visibility
        // to the next kernel launch on the same stream is guaranteed by HIP
        // stream-ordering semantics (kernel launches on the same stream are
        // serialized; the next launch sees all writes from the prior launch).
        // No explicit __threadfence() needed.
        __syncthreads();
        if (threadIdx.x == 0) {
            int32_t *const done_counter = tile_counter_ptr + NUM_XCDS_WS + 1;
            const int32_t  prev         = atomicAdd(done_counter, 1);
            if (prev == static_cast<int32_t>(gridDim.x - 1)) {
                // I am the last CTA out. Zero every slot (work + done) so the
                // next launch starts clean.
                for (index_t i = 0; i < NUM_XCDS_WS + 2; ++i) {
                    tile_counter_ptr[i] = 0;
                }
            }
        }
#endif // !defined(__gfx942__)
    }
};

} // namespace ck_tile
