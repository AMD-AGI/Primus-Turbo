// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// HIP-side equivalents of the DeepGEMM ``ptx::`` primitives used by the
// gfx950 mega-MoE kernel (and its scheduler).  Names mirror
// ``deep_gemm/include/deep_gemm/ptx/{utils,ld_st}.cuh`` so a side-by-side
// diff against the SM100 sources stays readable - the kernel calls
// ``prims::sync_aligned`` / ``prims::ld_acquire_global`` / ``prims::red_add_rel`` /
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

// EXPERIMENT (default 0): MEGA_MOE_RELAXED downgrades ALL acquire/release memory
// ordering to relaxed (and drops the _sys s_waitcnt + signal fences).  Perf-only
// probe of the cost of the ordering/fences -- BREAKS correctness, run with
// --num-correctness-tests 0.
#ifndef MEGA_MOE_RELAXED
#define MEGA_MOE_RELAXED 0
#endif
// Sub-probe: no-op ONLY the agent acquire fence (buffer_inv L1-invalidate), the
// per-block one in the compute arrival wait.  Isolates the L1-invalidate cost
// from the rel atomics / _sys fences.  Correctness-breaking probe.
#ifndef MEGA_MOE_NOACQFENCE
#define MEGA_MOE_NOACQFENCE 0
#endif
// Sub-probe: relax ONLY the agent-scope RELEASE atomics (-> relaxed).
#ifndef MEGA_MOE_RELAXED_REL
#define MEGA_MOE_RELAXED_REL 0
#endif
// Sub-probe: drop ONLY the _sys s_waitcnt(vmcnt/lgkmcnt) + signal fences.
#ifndef MEGA_MOE_RELAXED_SYS
#define MEGA_MOE_RELAXED_SYS 0
#endif
#if MEGA_MOE_RELAXED || MEGA_MOE_RELAXED_REL
#define MEGA_MOE_REL_ORDER __ATOMIC_RELAXED
#else
#define MEGA_MOE_REL_ORDER __ATOMIC_RELEASE
#endif
#if MEGA_MOE_RELAXED
#define MEGA_MOE_ACQ_ORDER __ATOMIC_RELAXED
#else
#define MEGA_MOE_ACQ_ORDER __ATOMIC_ACQUIRE
#endif
// Combined guard for the _sys drain/signal fences.
#if MEGA_MOE_RELAXED || MEGA_MOE_RELAXED_SYS
#define MEGA_MOE_SYS_FENCE 0
#else
#define MEGA_MOE_SYS_FENCE 1
#endif

// Compile-time wave size.  HIP's builtin ``warpSize`` is an ``int``
// variable (not constexpr), so it cannot appear in template parameters,
// ``static_assert``, or array sizes — use ``kWarpSize`` for those.
// Hard-coded to 64: this kernel is gfx950-only (wave64), matching the
// pattern in ``primus_turbo/deep_ep/configs.h``.  ``__AMDGCN_WAVEFRONT_SIZE``
// would have been the arch-tracking alternative, but it is not exposed
// as a constant-expression in the ROCm toolchain we build against.
inline constexpr int kWarpSize = 64;

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

// AGENT-scope acquire fence.  Lowers to an L1/vector-cache INVALIDATE (buffer_inv)
// + drain; required after a relaxed spin so the consumer's cached pool reads see
// the producer's payload.  A bare ``s_waitcnt`` drain does NOT invalidate (it only
// waits) -> stale reads -> gate-3 fails.  Used cheap-fence-ONCE: one fence after a
// relaxed ld_volatile spin, not one per iteration.
__device__ __forceinline__ void acquire_fence_agent() {
#if !MEGA_MOE_RELAXED && !MEGA_MOE_NOACQFENCE
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#endif
}

// AGENT-scope release fence: drains outstanding vector stores (s_waitcnt vmcnt(0))
// so they are visible in L2 before a subsequent (relaxed) signal.  Used ONCE at a
// phase boundary instead of per-op release ordering.
__device__ __forceinline__ void release_fence_agent() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

__device__ __forceinline__ uint32_t ld_volatile(const uint32_t *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t ld_volatile(const uint64_t *ptr) {
    return __hip_atomic_load(reinterpret_cast<const unsigned long long *>(ptr), __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint32_t ld_acquire_global(const uint32_t *ptr) {
    return __hip_atomic_load(ptr, MEGA_MOE_ACQ_ORDER, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t ld_acquire_global(const uint64_t *ptr) {
    return __hip_atomic_load(reinterpret_cast<const unsigned long long *>(ptr), MEGA_MOE_ACQ_ORDER,
                             __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t *ptr) {
#if MEGA_MOE_SYS_FENCE
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
    const uint32_t ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#if MEGA_MOE_SYS_FENCE
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#endif
    return ret;
}
__device__ __forceinline__ int ld_acquire_sys(const int *ptr) {
#if MEGA_MOE_SYS_FENCE
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
    const int ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#if MEGA_MOE_SYS_FENCE
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#endif
    return ret;
}

// ---------------------------------------------------------------------
//  Atomics / reductions.  Plain (non-``_rel``) ops are RELAXED.  AGENT ``_rel``
//  variants use native ``__ATOMIC_RELEASE`` (the on-device write-back the
//  consumer's acquire invalidate pairs with); the SYSTEM ``_rel_sys`` variant
//  uses the cheap fence above.  ``_sys`` = cross-rank IPC (SYSTEM scope).
// ---------------------------------------------------------------------
__device__ __forceinline__ uint32_t atomic_add_rel(uint32_t *ptr, uint32_t value) {
    return __hip_atomic_fetch_add(ptr, value, MEGA_MOE_REL_ORDER, __HIP_MEMORY_SCOPE_AGENT);
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
    __hip_atomic_fetch_add(ptr, value, MEGA_MOE_REL_ORDER, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void red_add_rel_sys(int *ptr, int value) {
#if MEGA_MOE_SYS_FENCE
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
    __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#if MEGA_MOE_SYS_FENCE
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#endif
}
__device__ __forceinline__ void red_or_rel_gpu(uint64_t *ptr, uint64_t value) {
    __hip_atomic_fetch_or(reinterpret_cast<unsigned long long *>(ptr),
                          static_cast<unsigned long long>(value), MEGA_MOE_REL_ORDER,
                          __HIP_MEMORY_SCOPE_AGENT);
}
// Release-ordered 64-bit add, AGENT scope.  Used by the L2 arrival COUNT
// (R182): every MMA wave that finishes its M-row slice of a pool block
// increments, so Linear2 only proceeds once ALL waves (not just the first)
// have written — fixing the read-before-write race that the OR-mask had.
__device__ __forceinline__ void red_add_rel_gpu(uint64_t *ptr, uint64_t value) {
    __hip_atomic_fetch_add(reinterpret_cast<unsigned long long *>(ptr),
                           static_cast<unsigned long long>(value), MEGA_MOE_REL_ORDER,
                           __HIP_MEMORY_SCOPE_AGENT);
}
// Cross-rank IPC counter update.  Mirrors SM100's ``atom.sys.global.add``:
// SYSTEM scope so the remote agent sees it (the AMD-mandatory equivalent of
// SM100 touching peer-mapped memory), RELAXED ordering — the cross-rank
// happens-before for the dispatch pull loop is established by the
// ``kBeforeDispatchPull`` ``nvlink_barrier`` (its ``red.release.sys`` /
// ``ld.acquire.sys`` handshake), which runs after these writes and before
// the reads, exactly as on SM100.
__device__ __forceinline__ void atomic_add_sys(uint64_t *ptr, uint64_t value) {
    __hip_atomic_fetch_add(reinterpret_cast<unsigned long long *>(ptr),
                           static_cast<unsigned long long>(value), __ATOMIC_RELAXED,
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

template <typename T>
__device__ __forceinline__ void atomicAdd_block(T *ptr, std::type_identity_t<T> val) {
    __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// ---------------------------------------------------------------------
//  Cooperative warp copy.  The HIP analogue of the bulk-async TMA
//  transfer the SM100 dispatch uses to pull a token's bytes
//  (remote-global -> local-pool global): every lane streams 16-byte
//  (int4) chunks, issuing ``kUnroll`` non-temporal loads before the
//  matching stores so the loads stay in flight together.  Mirrors the
//  ``UNROLLED_WARP_COPY`` / ``ld_nc_global`` / ``st_na_global`` pattern
//  in ``primus_turbo/csrc/kernels/deep_ep/utils.cuh`` (AMD has no TMA, so
//  there is no shared staging buffer / mbarrier - the copy goes straight
//  global->global).  ``n_int4`` is the element count in 16-byte units;
//  the tail handles a final partial batch.
// ---------------------------------------------------------------------
template <uint32_t kUnroll = 5u>
__device__ __forceinline__ void warp_copy_int4(void *dst, const void *src, uint32_t n_int4,
                                               uint32_t lane_idx) {
    using vec_t                = int __attribute__((ext_vector_type(4)));
    auto              *d       = reinterpret_cast<vec_t *>(dst);
    auto              *s       = reinterpret_cast<const vec_t *>(src);
    constexpr uint32_t kStride = kWarpSize * kUnroll;
    vec_t              vals[kUnroll];

    const uint32_t body = (n_int4 / kStride) * kStride;
    for (uint32_t i = lane_idx; i < body; i += kStride) {
#pragma unroll
        for (uint32_t j = 0; j < kUnroll; ++j)
            vals[j] = __builtin_nontemporal_load(s + i + j * kWarpSize);
#pragma unroll
        for (uint32_t j = 0; j < kUnroll; ++j)
            __builtin_nontemporal_store(vals[j], d + i + j * kWarpSize);
    }

    // Tail: final partial batch (< kStride elements left).
    const uint32_t tail = body + lane_idx;
#pragma unroll
    for (uint32_t j = 0; j < kUnroll; ++j)
        if (tail + j * kWarpSize < n_int4)
            vals[j] = __builtin_nontemporal_load(s + tail + j * kWarpSize);
#pragma unroll
    for (uint32_t j = 0; j < kUnroll; ++j)
        if (tail + j * kWarpSize < n_int4)
            __builtin_nontemporal_store(vals[j], d + tail + j * kWarpSize);
}
} // namespace primus_turbo::mega_moe::prims
