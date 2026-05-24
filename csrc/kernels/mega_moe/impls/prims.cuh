// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// HIP-side equivalents of the DeepGEMM ``ptx::`` primitives used by the
// gfx950 mega-MoE kernel (and its scheduler).  Names mirror
// ``deep_gemm/include/deep_gemm/ptx/{utils,ld_st}.cuh`` so a side-by-side
// diff against the SM100 sources stays readable - the kernel calls
// ``prims::sync_aligned`` / ``prims::ld_acq`` / ``prims::red_add_rel`` /
// ... at the same call sites where DG calls ``ptx::sync_aligned`` etc.
//
// Memory-scope choice rationale:
//   - On-device counters/masks (grid-sync count, L1 arrival count, L2
//     arrival mask, expert_send_count, ...) live in workspace pages that
//     are written by SMs on the SAME device.  Plain ``__atomic_*`` lowers
//     to system-scope on ROCm, which inserts a PCIe fence + L2 invalidate
//     on every operation - measurable in tens of ns per op.  Use
//     ``__hip_atomic_*`` with ``__HIP_MEMORY_SCOPE_AGENT`` to confine
//     coherency traffic to the GPU's L1/L2 hierarchy.
//   - Cross-rank IPC counters (the ``_sys`` variants used inside
//     ``nvlink_barrier``) span devices over XGMI and MUST keep
//     ``__HIP_MEMORY_SCOPE_SYSTEM`` so the remote agent observes the
//     update.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo::mega_moe::prims {

// Compile-time wave size.  Sourced from ``__AMDGCN_WAVEFRONT_SIZE`` so
// the value tracks ``--offload-arch`` (wave64 on gfx9xx, wave32 on RDNA)
// instead of being hard-coded.  HIP's builtin ``warpSize`` is an ``int``
// variable, not constexpr, so it cannot appear in template parameters,
// ``static_assert``, or array sizes — use ``kWarpSize`` for those.
inline constexpr int kWarpSize = __AMDGCN_WAVEFRONT_SIZE;
static_assert(kWarpSize == 64, "gfx950 mega-MoE assumes wave64");

// ---------------------------------------------------------------------
//  Lane / warp id helpers (mirror ``ptx::get_lane_idx`` / friends).
// ---------------------------------------------------------------------
__device__ __forceinline__ uint32_t get_lane_idx() {
    // ``__lane_id()`` lowers to a single ``v_mbcnt_{lo,hi}`` pair and
    // reads the true hardware lane id, so it stays correct under 2D/3D
    // block layouts where ``threadIdx.x & (warpSize - 1)`` would not.
    return __lane_id();
}

__device__ __forceinline__ uint32_t get_warp_idx() {
    return __builtin_amdgcn_workitem_id_x() / kWarpSize;
}

// ---------------------------------------------------------------------
//  Global memory loads.  ``ld_acq`` / ``ld_acq_gpu`` are on-device
//  (AGENT scope); ``ld_acq_sys`` is the cross-rank IPC variant.
// ---------------------------------------------------------------------
__device__ __forceinline__ uint32_t ld_acq(const uint32_t *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t ld_acq(const uint64_t *ptr) {
    return __hip_atomic_load(reinterpret_cast<const unsigned long long *>(ptr), __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t ld_acq_gpu(const uint64_t *ptr) {
    return __hip_atomic_load(reinterpret_cast<const unsigned long long *>(ptr), __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint32_t ld_acq_sys(const uint32_t *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ int ld_acq_sys(const int *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t ld_volatile(const uint32_t *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t ld_volatile(const uint64_t *ptr) {
    return __hip_atomic_load(reinterpret_cast<const unsigned long long *>(ptr), __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
}

// ---------------------------------------------------------------------
//  Atomics / reductions.  ``_sys`` variants are cross-rank IPC.
// ---------------------------------------------------------------------
__device__ __forceinline__ uint32_t atomic_add_rel(uint32_t *ptr, uint32_t value) {
    return __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t atomic_add(uint64_t *ptr, uint64_t value) {
    return __hip_atomic_fetch_add(reinterpret_cast<unsigned long long *>(ptr),
                                  static_cast<unsigned long long>(value), __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint32_t atomic_add(uint32_t *ptr, uint32_t value) {
    return __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void red_add(int *ptr, int value) {
    __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void red_add_rel(uint32_t *ptr, uint32_t value) {
    __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void red_add_rel_sys(int *ptr, int value) {
    __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void red_or_rel_gpu(uint64_t *ptr, uint64_t value) {
    __hip_atomic_fetch_or(reinterpret_cast<unsigned long long *>(ptr),
                          static_cast<unsigned long long>(value), __ATOMIC_RELEASE,
                          __HIP_MEMORY_SCOPE_AGENT);
}
// Cross-rank IPC counter update - SYSTEM scope so the remote agent sees it.
// RELEASE ordering: SM0's preceding plain store of the per-rank token count
// MUST be observable on the remote agent before the counter increment is,
// otherwise the dst's consumer can fall through and read 0 for the
// just-bumped slot - wedging the per-token round-robin in the dispatch
// fast path.  Empirically observed at EP8 / np=8 (the hang reproduces
// with RELAXED + plain store and goes away once both writes are
// SYSTEM-scope ordered).
__device__ __forceinline__ void atomic_add_sys(uint64_t *ptr, uint64_t value) {
    __hip_atomic_fetch_add(reinterpret_cast<unsigned long long *>(ptr),
                           static_cast<unsigned long long>(value), __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}

// Block-scope ``atomicAdd`` (workgroup atomic, fits in LDS).
template <typename T> __device__ __forceinline__ T atomic_add_block(T *ptr, T value) {
    return atomicAdd(ptr, value);
}

// ---------------------------------------------------------------------
//  Cross-lane primitives.  DG uses ``__reduce_min_sync`` /
//  ``__reduce_add_sync`` / ``__ballot_sync``; HIP exposes wave64
//  equivalents which we wrap behind the same names.
// ---------------------------------------------------------------------
__device__ __forceinline__ uint32_t reduce_min(uint32_t v) {
#pragma unroll
    for (int o = kWarpSize / 2; o > 0; o >>= 1) {
        const uint32_t other = __shfl_xor(v, o);
        v                    = other < v ? other : v;
    }
    return v;
}

__device__ __forceinline__ uint32_t reduce_add(uint32_t v) {
#pragma unroll
    for (int o = kWarpSize / 2; o > 0; o >>= 1)
        v += __shfl_xor(v, o);
    return v;
}

__device__ __forceinline__ uint64_t ballot(bool pred) {
    return __ballot(pred);
}

__device__ __forceinline__ uint32_t popcnt(uint64_t mask) {
    return __popcll(static_cast<unsigned long long>(mask));
}

// ``__ffs`` returning 1-based index of the lowest set bit, or 0 if mask==0.
__device__ __forceinline__ uint32_t ffs(uint64_t mask) {
    return mask ? (__ffsll(static_cast<long long>(mask))) : 0u;
}

// 0-based position of the n-th set bit in ``mask`` (``n`` is 1-based:
// ``n == 1`` returns the position of the lowest set bit).  Matches the
// semantics of CUDA's ``__fns(mask, 0, n)`` used by DG's round-robin
// rank selection - HIP does not expose ``__fns``, so we emulate by
// peeling lowest set bits.  Bound is ``n <= popcnt(mask) <= 64``.
__device__ __forceinline__ uint32_t nth_set_bit(uint64_t mask, uint32_t n) {
    for (uint32_t i = 1; i < n; ++i)
        mask &= (mask - 1ull); // clear lowest set bit
    return __builtin_ctzll(mask);
}

// elect_one - pick lane 0 as the canonical thread within the warp.
__device__ __forceinline__ bool elect_one() {
    return get_lane_idx() == 0u;
}

// Cross-lane broadcast for inter-iteration phase tracking.  Mirrors
// ``ptx::exchange`` but with a single template type (no struct
// decomposition needed at our call sites).
template <typename T> __device__ __forceinline__ T exchange(T value, int src_lane) {
    return __shfl(value, src_lane);
}

// ---------------------------------------------------------------------
//  Wave / cluster sync helpers.
// ---------------------------------------------------------------------
// Per-warp wave barrier (s_barrier within a single wave64).
__device__ __forceinline__ void sync_warp() {
    __builtin_amdgcn_wave_barrier();
}

// Cluster sync - no AMD equivalent; collapse to ``__syncthreads``.
__device__ __forceinline__ void cluster_sync() {
    __syncthreads();
}

// ---------------------------------------------------------------------
// Software-emulated NVPTX ``bar.sync id, count`` (named barrier).
//
// AMD gfx950 has no equivalent of CUDA's ``bar.sync`` / NVPTX
// ``barrier.sync`` that synchronises an arbitrary *subset* of waves in
// the workgroup.  ``__syncthreads()`` requires *all* threads in the CTA
// to arrive, which deadlocks when one role (dispatch / loader / MMA /
// epilogue) tries to sync only its own waves.
//
// We emulate the named barrier with one LDS counter per ``bar_id`` plus
// a per-thread ``expected[bar_id]`` arrival register, following the
// pattern from ``primus_turbo/csrc/kernels/deep_ep/utils.cuh``
// (``BARRIER_SYNC_INIT`` / ``BARRIER_SYNC``).  The deep_ep macros use a
// scalar ``___bar_sync_wg_expected_reg`` because each thread there
// sticks to a single ``bar_id`` (its ``responsible_rank``); mega-MoE
// dispatch threads call multiple bar_ids, so we widen ``expected`` to a
// per-thread array.  ``num_threads`` is the *total* participant count
// across all calling roles - both producer and consumer roles must pass
// the same value so the per-thread expected counters match.
// ---------------------------------------------------------------------
constexpr uint32_t kNumMaxNamedBars = 8u;

// CONSTRAINT: at most one ``NamedBarrierWg`` instance per kernel.
// ``init()`` allocates the LDS counter pad via a static ``__shared__``
// inside the function; CUDA/HIP static-shared is per-block, not
// per-object, so multiple instances would silently alias the same
// 16-int counter array and cause cross-barrier interference.  If a
// second arena is ever needed, lift the counter into kernel-scope
// dynamic shared memory and have ``init`` accept its base pointer
// (the pre-refactor API).
struct NamedBarrierWg {
    int *state;                      // [kNumMaxNamedBars] in LDS, owned by init()
    int  expected[kNumMaxNamedBars]; // per-thread arrival accumulator

    __device__ __forceinline__ void init() {
        // ``__shared__`` variables "have the lifetime of the block"
        // (CUDA C++ Programming Guide), so ``smem_state`` outlives this
        // function call and ``state`` is not a dangling pointer.
        __shared__ int smem_state[kNumMaxNamedBars];
        state = smem_state;
        if (threadIdx.x < kNumMaxNamedBars)
            state[threadIdx.x] = 0;
#pragma unroll
        for (uint32_t i = 0; i < kNumMaxNamedBars; ++i)
            expected[i] = 0;
        // Bring-up sync so every wave sees a zero-initialised counter
        // pad before the first arrive.
        __syncthreads();
    }

    __device__ __forceinline__ void arrive_and_wait(uint32_t bar_id, uint32_t num_threads) {
        const int num_waves = static_cast<int>(num_threads / kWarpSize);
        expected[bar_id] += num_waves;
        // Make preceding LDS / register writes visible to peer waves
        // before the atomic - mirrors deep_ep's ``__fence`` argument.
        __threadfence_block();
        if (get_lane_idx() == 0u) {
            __hip_atomic_fetch_add(state + bar_id, 1, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_WORKGROUP);
            while (__hip_atomic_load(state + bar_id, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_WORKGROUP) < expected[bar_id])
                __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_wave_barrier();
    }
};

// Replacements for DG's ``ptx::sync_aligned`` / ``ptx::sync_unaligned``
// (which are PTX ``bar.sync.aligned`` / ``bar.sync`` underneath).  On
// AMD both collapse to the named-barrier emulation - there is no
// aligned/unaligned distinction at the s_barrier level.  Param order
// mirrors DG with the per-workgroup ``NamedBarrierWg`` reference
// prepended (the AMD emulation needs LDS state DG gets for free from
// hardware).
__device__ __forceinline__ void sync_aligned(NamedBarrierWg &bar, uint32_t num_threads,
                                             uint32_t bar_id) {
    bar.arrive_and_wait(bar_id, num_threads);
}

__device__ __forceinline__ void sync_unaligned(NamedBarrierWg &bar, uint32_t num_threads,
                                               uint32_t bar_id) {
    bar.arrive_and_wait(bar_id, num_threads);
}

} // namespace primus_turbo::mega_moe::prims
