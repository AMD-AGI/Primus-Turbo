/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once
#include "primus_turbo/common.h"
#include "primus_turbo/deep_ep/configs.h"
#include "primus_turbo/device/memory.cuh"

// gfx12 (gfx1250 / gfx1251, CDNA5) removed the unified `s_waitcnt`; the combined
// LDS + VMEM wait is expressed via the split counters.
#if defined(__gfx1250__)
#define PRIMUS_TURBO_WAIT_ALL_STR                                                                  \
    "s_wait_dscnt 0\n\ts_wait_kmcnt 0\n\ts_wait_loadcnt 0\n\ts_wait_storecnt 0"
#else
#define PRIMUS_TURBO_WAIT_ALL_STR "s_waitcnt lgkmcnt(0) vmcnt(0)"
#endif

#define NUM_MAX_BARRIERS 32

// gfx942 / gfx950: no hardware sub-group barrier — emulate one with an LDS
// atomic counter + s_sleep.
#define BARRIER_SYNC_INIT()                                                                        \
    __shared__ int ___bar_sync_wg_state[NUM_MAX_BARRIERS];                                         \
    if (threadIdx.x < NUM_MAX_BARRIERS)                                                            \
        ___bar_sync_wg_state[threadIdx.x] = 0;                                                     \
    int ___bar_sync_wg_expected_reg = 0;                                                           \
    __syncthreads();

#define BARRIER_SYNC(__fence, ___bar_id, ___num_threads_per_group)                                 \
    {                                                                                              \
        auto ___num_wave_per_group = ___num_threads_per_group / kWarpSize;                         \
        ___bar_sync_wg_expected_reg += ___num_wave_per_group;                                      \
        __fence;                                                                                   \
        if (__lane_id() == 0) {                                                                    \
            __hip_atomic_fetch_add(___bar_sync_wg_state + ___bar_id, 1, __ATOMIC_RELAXED,          \
                                   __HIP_MEMORY_SCOPE_WORKGROUP);                                  \
            while (__hip_atomic_load(___bar_sync_wg_state + ___bar_id, __ATOMIC_RELAXED,           \
                                     __HIP_MEMORY_SCOPE_WORKGROUP) < ___bar_sync_wg_expected_reg)  \
                __builtin_amdgcn_s_sleep(1);                                                       \
        }                                                                                          \
        __syncwarp();                                                                              \
    }

#define sync_barrier_init() BARRIER_SYNC_INIT()
#define sync_barrier(bar_id, num_threads) BARRIER_SYNC(__threadfence_block(), bar_id, num_threads)
// The barrier in front of a channel-tail publish; see `publish_barrier_fence`.
#define sync_barrier_before_publish(kCheapFence, bar_id, num_threads)                              \
    BARRIER_SYNC(publish_barrier_fence<kCheapFence>(), bar_id, num_threads)

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                  \
    {                                                                                              \
        constexpr int kLoopStride = kWarpSize * (UNROLL_FACTOR);                                   \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type                         \
             unrolled_values[(UNROLL_FACTOR)];                                                     \
        auto __src = (SRC);                                                                        \
        auto __dst = (DST);                                                                        \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {   \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                      \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * kWarpSize);                     \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                      \
                ST_FUNC(__dst + __i + __j * kWarpSize, unrolled_values[__j]);                      \
        }                                                                                          \
        {                                                                                          \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                               \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                    \
                if (__i + __j * kWarpSize < (N)) {                                                 \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * kWarpSize);                 \
                }                                                                                  \
            }                                                                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                    \
                if (__i + __j * kWarpSize < (N)) {                                                 \
                    ST_FUNC(__dst + __i + __j * kWarpSize, unrolled_values[__j]);                  \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define UNROLLED_WARP_COPY_EMULATED(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)         \
    {                                                                                              \
        constexpr int kLoopStride = kEmulatedWarpSize * (UNROLL_FACTOR);                           \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type                         \
             unrolled_values[(UNROLL_FACTOR)];                                                     \
        auto __src = (SRC);                                                                        \
        auto __dst = (DST);                                                                        \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {   \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                      \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * kEmulatedWarpSize);             \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                      \
                ST_FUNC(__dst + __i + __j * kEmulatedWarpSize, unrolled_values[__j]);              \
        }                                                                                          \
        for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N);                   \
             __i += kEmulatedWarpSize)                                                             \
            ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                                            \
    }
// HELPER FUNCTIONS
// #####################################################################################

template <typename T>
__device__ __forceinline__ T shfl_xor(const T val, int laneMask, int width = kWarpSize,
                                      uint64_t shfl_sync_mask = kFullWarpMask) {
    return __shfl_xor(val, laneMask, width);
}

__device__ __forceinline__ int
shfl_sync(const int val, int srcLane = 0, int width = kWarpSize,
          uint64_t shfl_sync_mask = kFullWarpMask) { // Let compiler deduce type
    return __shfl(val, srcLane, width);
}

__device__ __forceinline__ int __any_sync(uint64_t mask, int predicate) {
    uint64_t predicate_bit_pattern = __ballot(predicate);
    return (predicate_bit_pattern & mask) > 0;
}

__device__ __forceinline__ int __all_sync(uint64_t mask, int predicate) {
    uint64_t predicate_bit_pattern = __ballot(predicate);
    return (~predicate_bit_pattern & mask) == 0;
}

__device__ __forceinline__ void syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
// ######################################################################################################

namespace primus_turbo::deep_ep {

template <int kBytes> struct VecInt {};
template <> struct VecInt<1> {
    using vec_t = int8_t;
};
template <> struct VecInt<2> {
    using vec_t = int16_t;
};
template <> struct VecInt<4> {
    using vec_t = int;
};
template <> struct VecInt<8> {
    using vec_t = int64_t;
};
template <> struct VecInt<16> {
    using native_int4 = int __attribute__((ext_vector_type(4)));
    using vec_t       = native_int4;
};

__device__ __forceinline__ void trap() {
    abort();
}

__device__ __forceinline__ void memory_fence() {

    __threadfence_system();
}

__device__ __forceinline__ void memory_fence_gpu() {
    __threadfence();
}

__device__ __forceinline__ void memory_fence_cta() {
    __threadfence_block();
}

__device__ __forceinline__ void st_relaxed_sys_global(int *ptr, int val) {
    __builtin_nontemporal_store(val, ptr);
}

__device__ __forceinline__ void st_release_sys_global(const int *ptr, int val) {
    __hip_atomic_store(const_cast<int *>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void st_release_cta(const int *ptr, int val) {
    __hip_atomic_store(const_cast<int *>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
}

__device__ __forceinline__ int ld_relaxed_sys_global(const int *ptr) {
    int res = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return res;
}
__device__ __forceinline__ int ld_relaxed_sys_global(const uint64_t *ptr) {
    uint64_t ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {
    int ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
    uint64_t ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
    int ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    return ret;
}

__device__ __forceinline__ int atomic_add_release_global(const int *ptr, int value) {
    int ret;
    ret = __hip_atomic_fetch_add(const_cast<int *>(ptr), value, __ATOMIC_RELEASE,
                                 __HIP_MEMORY_SCOPE_AGENT);
    return ret;
}

__device__ __forceinline__ int ld_acquire_cta(const int *ptr) {
    int ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP);
    return ret;
}

__device__ __forceinline__ int ld_volatile_global(const volatile int *ptr) {
    int ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ float ld_volatile_global(const volatile float *ptr) {
    float ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const volatile int64_t *ptr) {
    int64_t ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const volatile uint64_t *ptr) {
    int64_t ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

template <typename dtype_t> __device__ __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {
    using T  = typename VecInt<sizeof(dtype_t)>::vec_t;
    auto ret = __builtin_nontemporal_load(reinterpret_cast<const T *>(ptr));
    return *reinterpret_cast<dtype_t *>(&ret);
}

////////////////// used in ibgda
__device__ __forceinline__ void st_na_relaxed(const uint8_t *ptr, uint8_t val) {
    uint8_t *non_const_ptr = const_cast<uint8_t *>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t *ptr, uint16_t val) {
    uint16_t *non_const_ptr = const_cast<uint16_t *>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t *ptr, uint32_t val) {
    uint32_t *non_const_ptr = const_cast<uint32_t *>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const int *ptr, int val) {
    int *non_const_ptr = const_cast<int *>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const int4 *ptr, int4 val) {
    int4 *non_const_ptr = const_cast<int4 *>(ptr);
    non_const_ptr->x    = val.x;
    non_const_ptr->y    = val.y;
    non_const_ptr->z    = val.z;
    non_const_ptr->w    = val.w;
}

__device__ __forceinline__ void st_na_release(const int *ptr, int val) {
    int *non_const_ptr = const_cast<int *>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_release(const uint32_t *ptr, uint32_t val) {
    uint32_t *non_const_ptr = const_cast<uint32_t *>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_release(const uint64_t *ptr, uint64_t val) {
    uint64_t *non_const_ptr = const_cast<uint64_t *>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// TODO:: apply "st.global.L1::no_allocate" in ROCM
template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t *ptr, const dtype_t &value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(&value));
}

template <> __device__ __forceinline__ void st_na_global(const int *ptr, const int &value) {
    int *non_const_ptr = const_cast<int *>(ptr);
    *non_const_ptr     = value;
}

template <> __device__ __forceinline__ void st_na_global(const int64_t *ptr, const int64_t &value) {
    int64_t *non_const_ptr = const_cast<int64_t *>(ptr);
    *non_const_ptr         = value;
}

template <> __device__ __forceinline__ void st_na_global(const float *ptr, const float &value) {
    float *non_const_ptr = const_cast<float *>(ptr);
    *non_const_ptr       = value;
}

template <> __device__ __forceinline__ void st_na_global(const int4 *ptr, const int4 &value) {
    int4 *non_const_ptr = const_cast<int4 *>(ptr);
    *non_const_ptr      = value;
}

// Cheap fence for the intranode channel ring buffers.
//
// A channel's payload slots are written by every warp of a rank group, and one warp then
// publishes the new tail with a relaxed system-scope store. The full fence makes that store a
// release and the receiver's poll an acquire, which on gfx942 / gfx950 costs a GPU-wide L2
// write-back per publish and an L1 + L2 invalidate per poll -- both stall concurrent kernels too.
// The cheap fence keeps neither and instead:
//   1. has every sending warp wait for its own stores (`s_waitcnt vmcnt(0)`) before the barrier
//      in front of the publish. The channel buffers are uncached (`hipDeviceMallocUncached`), so a
//      store to them is acknowledged only once it has reached memory, local or peer: by the time
//      the tail is stored, all of the group's payload is visible at system scope. A waitcnt in
//      the publishing warp alone covers only its own stores.
//   2. reads the payload with system-scope loads, which miss every GPU cache and so cannot return
//      a slot's previous contents.
// Other targets (gfx1250) encode cache policy and wait counters differently; they keep the full
// fence.
#if defined(__gfx942__) || defined(__gfx950__)
constexpr bool kHasCheapFence = true;
#else
constexpr bool kHasCheapFence = false;
#endif

// Buffer-load cache policy on gfx942 / gfx950: SC0 (1) | SC1 (16) is system scope, NT (2) marks
// the access as streaming.
constexpr int32_t kSysScopeStreamCPol = 0x13;

// HIP makes every call convergent, which would keep loops around this from unrolling; the
// intrinsic itself is not.
__device__ __attribute__((noconvergent)) device::int32x4_t
llvm_amdgcn_raw_buffer_load_v4i32(device::int32x4_t rsrc, int32_t voffset, int32_t soffset,
                                  int32_t cpol) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// Reads of a channel's `int4` payload slots by whole warps. `base` must be the same in every
// lane, and the element pointers handed to `ld` must lie less than 4 GiB past it.
// `kSysScope = false` is the full fence's plain load. The system-scope variant is a buffer
// descriptor, built with a `readfirstlane`; that is convergent and keeps a loop with a runtime
// trip count from unrolling, so construct it outside the token loops.
template <bool kSysScope> struct ChannelPayload {
    __device__ __forceinline__ explicit ChannelPayload(const int4 *) {}

    __device__ __forceinline__ int4 ld(const int4 *ptr) const { return ld_nc_global(ptr); }
};

template <> struct ChannelPayload<true> {
    device::BufferSRD srd;
    const int4       *base;

    __device__ __forceinline__ explicit ChannelPayload(const int4 *base) : srd(base), base(base) {}
    __device__ __forceinline__ int4 ld(const int4 *ptr) const {
        auto value = llvm_amdgcn_raw_buffer_load_v4i32(
            srd.srd, static_cast<int32_t>((ptr - base) * sizeof(int4)), 0, kSysScopeStreamCPol);
        return *reinterpret_cast<int4 *>(&value);
    }
};

// Scalar payload reads (`src_idx`, `topk_idx`, `topk_weights`, `x_scales`). The system-scope
// variant is a relaxed atomic, which carries SC0 | SC1 and no waitcnt.
template <bool kSysScope, typename dtype_t>
__device__ __forceinline__ dtype_t ld_payload(const dtype_t *ptr) {
    if constexpr (kSysScope)
        return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    else
        return ld_nc_global(ptr);
}

template <bool kCheapFence> __device__ __forceinline__ void publish_barrier_fence() {
    __threadfence_block();
    if constexpr (kCheapFence) {
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        asm volatile(PRIMUS_TURBO_WAIT_ALL_STR ::: "memory");
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
    }
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id,
                                                       int &token_start_idx, int &token_end_idx) {
    int num_tokens_per_sm = DIVUP(num_tokens, num_sms);
    token_start_idx       = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx         = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t &ptr, int src_lane_idx) {
    PRIMUS_TURBO_STATIC_CHECK(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int *>(&ptr);
    int  recv_int_values[sizeof(dtype_t) / sizeof(int)];
#pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        recv_int_values[i] = shfl_sync(send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t *>(recv_int_values);
}

__forceinline__ __device__ int warp_reduce_sum(int value) {
    if constexpr (kWarpSize == 64)
        value += shfl_xor<int>(value, 32);
    value += shfl_xor<int>(value, 16);
    value += shfl_xor<int>(value, 8);
    value += shfl_xor<int>(value, 4);
    value += shfl_xor<int>(value, 2);
    value += shfl_xor<int>(value, 1);
    return value;
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id = threadIdx.x % kWarpSize;
    return lane_id;
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int **barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible
    // to the `sys` scope
    if constexpr (not kSyncOnly) {
        memory_fence();
        __syncthreads();
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
    }
    PRIMUS_TURBO_DEVICE_CHECK(kNumRanks <= blockDim.x);

    // Check timeout
    auto start_time = clock64();
    while (true) {
        auto value =
            thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (__all_sync(kFullWarpMask, value <= 0))
            break;

        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank,
                   thread_id, value);
            trap();
        }
    }
    __syncthreads();
}

template <bool kUseCheapFence, typename dtype_t>
__device__ __forceinline__ void st_release_sys_global(const dtype_t *ptr, dtype_t val) {

    if constexpr (kUseCheapFence) {
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        asm volatile(PRIMUS_TURBO_WAIT_ALL_STR);
        __hip_atomic_store(const_cast<dtype_t *>(ptr), val, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
    } else {
        __hip_atomic_store(const_cast<dtype_t *>(ptr), val, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

template <bool kUseCheapFence, typename dtype_t>
__device__ __forceinline__ dtype_t ld_acquire_sys_global(const dtype_t *ptr) {
    dtype_t ret;
    if constexpr (kUseCheapFence) {
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        asm volatile(PRIMUS_TURBO_WAIT_ALL_STR);
        ret = __hip_atomic_load(const_cast<dtype_t *>(ptr), __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SYSTEM);
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
    } else {
        ret = __hip_atomic_load(const_cast<dtype_t *>(ptr), __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_SYSTEM);
    }
    return ret;
}

} // namespace primus_turbo::deep_ep
