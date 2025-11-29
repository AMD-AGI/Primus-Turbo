/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once
#include "primus_turbo/common.h"
#include "primus_turbo/deep_ep/configs.h"

#define __syncwarp() __builtin_amdgcn_wave_barrier()
#ifndef clock64
#define clock64 wall_clock64
#endif
#ifndef DISABLE_AGGRESSIVE_ATOMIC
#define HIP_ATOMIC_LOAD(ptr, order, scope) __hip_atomic_load((ptr), __ATOMIC_RELAXED, (scope))
#define HIP_ATOMIC_STORE(val, ptr, order, scope)                                                   \
    __hip_atomic_store((ptr), (val), __ATOMIC_RELAXED, (scope))
#define HIP_ATOMIC_ADD(ptr, val, order, scope)                                                     \
    __hip_atomic_fetch_add((ptr), (val), __ATOMIC_RELAXED, (scope))
#else
#define HIP_ATOMIC_LOAD(ptr, order, scope) __hip_atomic_load((ptr), (order), (scope))
#define HIP_ATOMIC_STORE(val, ptr, order, scope) __hip_atomic_store((ptr), (val), (order), (scope))
#define HIP_ATOMIC_ADD(ptr, val, order, scope)                                                     \
    __hip_atomic_fetch_add((ptr), (val), (order), (scope))
#endif

// workgroup-level barrier sync used shared memory
namespace primus_turbo::deep_ep {
struct SharedData {
    uint32_t barrier[MAX_GROUPS];
};

__shared__ SharedData shared_data;

__device__ __forceinline__ void barrier_init(int barrier_id) {
    shared_data.barrier[barrier_id] = 0;
}

template <typename T, int MemoryOrder = __ATOMIC_RELAXED,
          int MemoryScope = __HIP_MEMORY_SCOPE_WORKGROUP>
__device__ __forceinline__ T barrier_arrive(T *bar_ptr, int num_participants) {
    T v = __hip_atomic_fetch_add(bar_ptr, 1U, MemoryOrder, MemoryScope);

    if ((v & MAX_GROUPS_MASK) == num_participants - 1)
        __hip_atomic_fetch_add(bar_ptr, MAX_GROUPS - num_participants, MemoryOrder, MemoryScope);

    return v & ~MAX_GROUPS_MASK;
}

template <typename T, int MemoryOrder = __ATOMIC_RELAXED,
          int MemoryScope = __HIP_MEMORY_SCOPE_WORKGROUP>
__device__ __forceinline__ void barrier_wait(T *bar_ptr, T target) {
    while ((__hip_atomic_load(bar_ptr, MemoryOrder, MemoryScope) & ~MAX_GROUPS_MASK) == target)
        __builtin_amdgcn_s_sleep(1);
}

template <typename T, int MemoryOrder = __ATOMIC_RELAXED,
          int MemoryScope = __HIP_MEMORY_SCOPE_WORKGROUP>
__device__ __forceinline__ void barrier_sync(T *bar_ptr, uint32_t num_participants) {
    // bound check
    if (num_participants >= MAX_GROUPS) {
        __syncthreads();
        return;
    }
    if (num_participants == 1) {
        __syncwarp();
        return;
    }

    auto const lane_id = __lane_id();
    if (lane_id == 0) {
        barrier_wait(bar_ptr, barrier_arrive(bar_ptr, num_participants));
    }
    __syncwarp();
}

template <typename T, int MemoryOrder = __ATOMIC_RELAXED,
          int MemoryScope = __HIP_MEMORY_SCOPE_AGENT>
__device__ __forceinline__ void grid_sync(T *bar_ptr, int num_participants) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        __hip_atomic_fetch_add(bar_ptr, 1, MemoryOrder, MemoryScope);
        while (__hip_atomic_load(bar_ptr, MemoryOrder, MemoryScope) < num_participants)
            __builtin_amdgcn_s_sleep(1);

        asm volatile("s_wakeup");
    }

    __syncthreads();
}

__forceinline__ __device__ int get_lane_id() {
    return __lane_id();
}

template <typename dtype_t> __host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

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
    using vec_t = int4;
};

// Unified reduction function
template <uint32_t kNumLanes, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    PRIMUS_TURBO_STATIC_CHECK(kNumLanes == 64 or kNumLanes == 32 or kNumLanes == 16 or
                                  kNumLanes == 8 or kNumLanes == 4 or kNumLanes == 2 or
                                  kNumLanes == 1,
                              "Invalid number of lanes");
    if constexpr (kNumLanes >= 64)
        value = op(value, __shfl_xor_sync(WARP_MASK, value, 32));
    if constexpr (kNumLanes >= 32)
        value = op(value, __shfl_xor_sync(WARP_MASK, value, 16));
    if constexpr (kNumLanes >= 16)
        value = op(value, __shfl_xor_sync(WARP_MASK, value, 8));
    if constexpr (kNumLanes >= 8)
        value = op(value, __shfl_xor_sync(WARP_MASK, value, 4));
    if constexpr (kNumLanes >= 4)
        value = op(value, __shfl_xor_sync(WARP_MASK, value, 2));
    if constexpr (kNumLanes >= 2)
        value = op(value, __shfl_xor_sync(WARP_MASK, value, 1));
    return value;
}

template <typename T> struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T> struct ReduceMax {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T> struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};

template <uint32_t kNumLanes = WARP_SIZE, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanes, T>(value, ReduceMin<T>{});
}

template <uint32_t kNumLanes = WARP_SIZE, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanes, T>(value, ReduceMax<T>{});
}

__device__ __forceinline__ float log2f_approx(float const &x) {
    return __builtin_amdgcn_logf(x);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(dtype_b_t const &packed, dtype_a_t &x, dtype_a_t &y) {
    PRIMUS_TURBO_STATIC_CHECK(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<dtype_a_t const *>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

// Convenience aliases
template <uint32_t kNumLanes = WARP_SIZE, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanes, T>(value, ReduceSum<T>{});
}

template <typename dtype_t> __device__ __forceinline__ dtype_t ld_nc_global(dtype_t const *ptr) {
    auto ret = ld_nc_global(reinterpret_cast<typename VecInt<sizeof(dtype_t)>::vec_t const *>(ptr));
    return *reinterpret_cast<dtype_t *>(&ret);
}

template <> __device__ __forceinline__ uint8_t ld_nc_global(uint8_t const *ptr) {
    uint16_t ret;
    ret = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ int ld_nc_global(int const *ptr) {
    int ret;
    ret = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ int64_t ld_nc_global(int64_t const *ptr) {
    int64_t ret;
    ret = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ float ld_nc_global(float const *ptr) {
    float ret;
    ret = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ int2 ld_nc_global(int2 const *ptr) {
    int2 ret;
    ret.x = __builtin_nontemporal_load(&ptr->x);
    ret.y = __builtin_nontemporal_load(&ptr->y);
    return ret;
}

template <> __device__ __forceinline__ int4 ld_nc_global(int4 const *ptr) {
    int4 ret;
    ret.x = __builtin_nontemporal_load(&ptr->x);
    ret.y = __builtin_nontemporal_load(&ptr->y);
    ret.z = __builtin_nontemporal_load(&ptr->z);
    ret.w = __builtin_nontemporal_load(&ptr->w);
    return ret;
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif
template <typename dtype_t>
__device__ __forceinline__ void st_na_global(dtype_t const *ptr, dtype_t const &value) {
    st_na_global(reinterpret_cast<typename VecInt<sizeof(dtype_t)>::vec_t const *>(ptr),
                 *reinterpret_cast<typename VecInt<sizeof(dtype_t)>::vec_t const *>(&value));
}

template <> __device__ __forceinline__ void st_na_global(int const *ptr, int const &value) {
    __builtin_nontemporal_store(value, ptr);
}

template <> __device__ __forceinline__ void st_na_global(int64_t const *ptr, int64_t const &value) {
    __builtin_nontemporal_store(value, ptr);
}

template <> __device__ __forceinline__ void st_na_global(float const *ptr, float const &value) {
    __builtin_nontemporal_store(value, ptr);
}

template <> __device__ __forceinline__ void st_na_global(int4 const *ptr, int4 const &value) {
    __builtin_nontemporal_store(value.x, &ptr->x);
    __builtin_nontemporal_store(value.y, &ptr->y);
    __builtin_nontemporal_store(value.z, &ptr->z);
    __builtin_nontemporal_store(value.w, &ptr->w);
}

__device__ __forceinline__ int ld_acquire_sys_global(int const *ptr) {
    int ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(dtype_a_t const &x, dtype_a_t const &y) {
    PRIMUS_TURBO_STATIC_CHECK(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto      unpacked_ptr = reinterpret_cast<dtype_a_t *>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t *>(&value)) >> 23);
    } else {
        return value;
    }
}

// 32-bit system-consistent load with acquire semantics (GH200-safe)
__device__ __forceinline__ uint32_t ld_acquire_sys_global(uint32_t const volatile *p) {
    uint32_t v;
    v = HIP_ATOMIC_LOAD(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    return v;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t const volatile *p) {
    uint64_t v;
    v = HIP_ATOMIC_LOAD(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    return v;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t const *ptr) {
    uint64_t ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

template <typename dtype_t> __host__ __device__ constexpr dtype_t align(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

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

__device__ __forceinline__ int atomic_add_release_global(int const *ptr, int value) {
    int ret;
    ret = HIP_ATOMIC_ADD(const_cast<int *>(ptr), value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    return ret;
}

__device__ __forceinline__ int ld_acquire_global(int const *ptr) {
    int ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    return ret;
}

__device__ __forceinline__ void st_release_sys_global(int const *ptr, int val) {
    HIP_ATOMIC_STORE(val, const_cast<int *>(ptr), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void st_release_cta(int const *ptr, int val) {
    HIP_ATOMIC_STORE(val, const_cast<int *>(ptr), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template <bool kUseUnsafeSync = false>
__device__ inline void workgroup_sync_barrier(int barrier_id, int num_threads) {
    // If __syncthreads is feasible in kernel,
    // using __syncthreads directly will be better than shared memory based
    // barrier.
    if constexpr (kUseUnsafeSync) {
        // maybe stuck in __syncthreads
        num_threads >= WARP_SIZE ? __syncthreads() : __syncwarp();
    } else {
        PRIMUS_TURBO_DEVICE_CHECK(num_threads % WARP_SIZE == 0 and "invalid number of threads");

        auto      *bar_ptr          = &shared_data.barrier[barrier_id];
        auto const num_participants = static_cast<uint32_t>(num_threads / WARP_SIZE);
        barrier_sync(bar_ptr, num_participants);
    }
}

template <bool kUseUnsafeSync = false>
__device__ inline void sync_barrier(int barrier_id, int num_threads) {
    workgroup_sync_barrier<kUseUnsafeSync>(barrier_id, num_threads);
}

template <bool kUseUnsafedSync = false> __device__ inline void sync_barrier_1(int num_threads) {
    workgroup_sync_barrier<kUseUnsafedSync>(1, num_threads);
}

__device__ __forceinline__ void trap() {
    abort();
}

__device__ __forceinline__ int ld_volatile_global(int const *ptr) {
    int ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ float ld_volatile_global(float const *ptr) {
    float ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(int64_t const *ptr) {
    int64_t ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(uint64_t const *ptr) {
    int64_t ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}

__device__ __forceinline__ void memory_fence() {
    __threadfence_system();
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int *addr, int x, int y) {
    // TODO: __hip_atomic_compare_exchange_strong or
    // __hip_atomic_compare_exchange_weak
    __hip_atomic_compare_exchange_strong(addr, &x, y, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_WORKGROUP);
    return x;
}

__forceinline__ __device__ int atomic_exch_cta_release(int *addr, int x) {
    int ret;
    ret = __hip_atomic_exchange(addr, x, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
    return ret;
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int **barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the
    // block must be visible to the `sys` scope
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
        if (__all_sync(WARP_MASK, value <= 0))
            break;

        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = "
                   "%d)\n",
                   rank, thread_id, value);
            trap();
        }
    }
    __syncthreads();
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id,
                                                       int &token_start_idx, int &token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
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
        recv_int_values[i] = __shfl_sync(WARP_MASK, send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t *>(recv_int_values);
}

__device__ __forceinline__ void memory_fence_gpu() {
    __threadfence();
}

__device__ __forceinline__ void memory_fence_cta() {
    __threadfence_block();
}

__device__ __forceinline__ void st_relaxed_sys_global(int const *ptr, int val) {
    HIP_ATOMIC_STORE(val, const_cast<int *>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ int ld_acquire_cta(int const *ptr) {
    int ret;
    ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP);
    return ret;
}

__forceinline__ __device__ void acquire_lock(int *mutex) {
    // To make later memory operations valid, we must use `acquire` for memory
    // semantics
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
        ;
}

__forceinline__ __device__ void release_lock(int *mutex) {
    // To make previous memory operations visible to other threads, we must
    // use `release` for memory semantics
    atomic_exch_cta_release(mutex, 0);
}
} // namespace primus_turbo::deep_ep
