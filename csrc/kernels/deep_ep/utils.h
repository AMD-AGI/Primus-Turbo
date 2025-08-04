/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once
#include "configs.h"
#include "primus_turbo/common.h"

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
        for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += kWarpSize) \
            ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                                            \
    }

__device__ inline void syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
namespace primus_turbo {

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
    __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void st_release_sys_global(const int *ptr, int val) {
    __hip_atomic_store(const_cast<int *>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void st_release_cta(const int *ptr, int val) {
    __hip_atomic_store(const_cast<int *>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
}

__device__ __forceinline__ int ld_relaxed_sys_global(const int *ptr) {
    int ret;
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
// inter
__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
    int ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    return ret;
}

// inter
__device__ __forceinline__ int atomic_add_release_global(const int *ptr, int value) {
    int ret;
    ret = __hip_atomic_fetch_add(const_cast<int *>(ptr), value, __ATOMIC_RELEASE,
                                 __HIP_MEMORY_SCOPE_AGENT);
    return ret;
}
// inter
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

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t> __device__ __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr));
    return *reinterpret_cast<dtype_t *>(&ret);
}

template <> __device__ __forceinline__ uint8_t ld_nc_global(const uint8_t *ptr) {
    uint8_t ret = *ptr;
    ret         = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ int ld_nc_global(const int *ptr) {
    int ret;
    ret = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ int64_t ld_nc_global(const int64_t *ptr) {
    int64_t ret;
    ret = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ float ld_nc_global(const float *ptr) {
    float ret;
    ret = __builtin_nontemporal_load(ptr);
    return ret;
}

template <> __device__ __forceinline__ int2 ld_nc_global(const int2 *ptr) {
    int2 ret;
    int  x, y;
    x   = __builtin_nontemporal_load(&(ptr->x));
    y   = __builtin_nontemporal_load(&(ptr->y));
    ret = {x, y};
    return ret;
}

template <> __device__ __forceinline__ int4 ld_nc_global(const int4 *ptr) {
    int4 ret;
    int  x, y, z, w;
    x   = __builtin_nontemporal_load(&(ptr->x));
    y   = __builtin_nontemporal_load(&(ptr->y));
    z   = __builtin_nontemporal_load(&(ptr->z));
    w   = __builtin_nontemporal_load(&(ptr->w));
    ret = {x, y, z, w};
    return ret;
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

template <typename dtype_t> __host__ __device__ dtype_t cell_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t> __host__ __device__ dtype_t align(dtype_t a, dtype_t b) {
    return cell_div<dtype_t>(a, b) * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id,
                                                       int &token_start_idx, int &token_end_idx) {
    int num_tokens_per_sm = cell_div(num_tokens, num_sms);
    token_start_idx       = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx         = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t &x, const dtype_a_t &y) {
    PRIMUS_TURBO_STATIC_CHECK(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto      unpacked_ptr = reinterpret_cast<dtype_a_t *>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t &packed, dtype_a_t &x, dtype_a_t &y) {
    PRIMUS_TURBO_STATIC_CHECK(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t *>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t &ptr, int src_lane_idx) {
    PRIMUS_TURBO_STATIC_CHECK(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int *>(&ptr);
    int  recv_int_values[sizeof(dtype_t) / sizeof(int)];
#pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        recv_int_values[i] = __shfl_sync(kFullWarpMask, send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t *>(recv_int_values);
}

// Operation functors
template <typename T> struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T> struct ReduceMax {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T> struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};

// Unified reduction function
template <uint32_t kNumLanes, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    PRIMUS_TURBO_STATIC_CHECK(kNumLanes == 64 or kNumLanes == 32 or kNumLanes == 16 or
                                  kNumLanes == 8 or kNumLanes == 4 or kNumLanes == 2 or
                                  kNumLanes == 1,
                              "Invalid number of lanes");

    if constexpr (kNumLanes >= 64)
        value = op(value, __shfl_xor_sync(kFullWarpMask, value, 32));
    if constexpr (kNumLanes >= 32)
        value = op(value, __shfl_xor_sync(kFullWarpMask, value, 16));
    if constexpr (kNumLanes >= 16)
        value = op(value, __shfl_xor_sync(kFullWarpMask, value, 8));
    if constexpr (kNumLanes >= 8)
        value = op(value, __shfl_xor_sync(kFullWarpMask, value, 4));
    if constexpr (kNumLanes >= 4)
        value = op(value, __shfl_xor_sync(kFullWarpMask, value, 2));
    if constexpr (kNumLanes >= 2)
        value = op(value, __shfl_xor_sync(kFullWarpMask, value, 1));
    return value;
}

// Convenience aliases
template <uint32_t kNumLanes = 64, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanes, T>(value, ReduceSum<T>{});
}

template <uint32_t kNumLanes = 64, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanes, T>(value, ReduceMax<T>{});
}

template <uint32_t kNumLanes = 64, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanes, T>(value, ReduceMin<T>{});
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    lane_id = threadIdx.x % kWarpSize;
    return lane_id;
}

template <int kNumRanks> __forceinline__ __device__ void move_fifo_slots(int &head) {
    head = (head + kNumRanks) % NUM_MAX_FIFO_SLOTS;
}

template <int kNumRanks> __device__ __forceinline__ bool not_finished(int *task, int expected) {
    auto result  = false;
    auto lane_id = threadIdx.x % kWarpSize;
    if (lane_id < kNumRanks)
        result = ld_volatile_global(task + lane_id) != expected;
    return __any_sync(kFullWarpMask, result);
}

template <int kNumRanks>
__forceinline__ __device__ void timeout_check(int **task_fifo_ptrs, int head, int rank,
                                              int expected, int tag = 0) {
    auto start_time = wall_clock64();
    while (not_finished<kNumRanks>(task_fifo_ptrs[rank] + head, expected)) {
        long long int elapsed_time = wall_clock64() > start_time ? wall_clock64() - start_time : 0;
        if (elapsed_time > NUM_TIMEOUT_CYCLES and threadIdx.x == 0) {
            printf("DeepEP timeout check failed: %d (rank = %d)\n", tag, rank);
            trap();
        }
    }
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

template <int kNumRanks>
__forceinline__ __device__ void barrier_device(int **task_fifo_ptrs, int head, int rank,
                                               int tag = 0) {
    auto thread_id = static_cast<int>(threadIdx.x);
    PRIMUS_TURBO_DEVICE_CHECK(kNumRanks <= kWarpSize);

    if (thread_id < kNumRanks) {
        atomicAdd_system(task_fifo_ptrs[rank] + head + thread_id, FINISHED_SUM_TAG);
        memory_fence();
        atomicSub_system(task_fifo_ptrs[thread_id] + head + rank, FINISHED_SUM_TAG);
    }
    timeout_check<kNumRanks>(task_fifo_ptrs, head, rank, 0, tag);
}

} // namespace primus_turbo
