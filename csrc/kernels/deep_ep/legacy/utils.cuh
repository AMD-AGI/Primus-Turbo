#pragma once

#include <primus_turbo/deep_ep/common/compiled.cuh>
#include <primus_turbo/deep_ep/common/exception.cuh>

// ROCm: 32 -> WARP_SIZE, the loop stride is one wave of lanes
#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                              \
    {                                                                                                                                         \
        constexpr int kLoopStride = WARP_SIZE * (UNROLL_FACTOR);                                                                              \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                                  \
        auto __src = (SRC);                                                                                                                    \
        auto __dst = (DST);                                                                                                                    \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                              \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * WARP_SIZE);  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * WARP_SIZE, unrolled_values[__j]);   \
        }                                                                                                                                     \
        {                                                                                                                                     \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                          \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                               \
                if (__i + __j * WARP_SIZE < (N)) {                                                                                            \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * WARP_SIZE);                                                            \
                }                                                                                                                             \
            }                                                                                                                                 \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                               \
                if (__i + __j * WARP_SIZE < (N)) {                                                                                            \
                    ST_FUNC(__dst + __i + __j * WARP_SIZE, unrolled_values[__j]);                                                             \
                }                                                                                                                             \
            }                                                                                                                                 \
        }                                                                                                                                     \
    }

// ROCm: deep_ep::legacy -> primus_turbo::deep_ep::legacy
namespace primus_turbo::deep_ep::legacy {

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
    using vec_t = int8_t;
};
template <>
struct VecInt<2> {
    using vec_t = int16_t;
};
template <>
struct VecInt<4> {
    using vec_t = int;
};
template <>
struct VecInt<8> {
    using vec_t = int64_t;
};
template <>
struct VecInt<16> {
    // ROCm: int4 -> ext_vector_type(4), the nontemporal builtins reject HIP vector structs
    using vec_t = int __attribute__((ext_vector_type(4)));
};

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__ explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

    __device__ __host__ auto operator[](const uint32_t& i) { return func(i); }
};

__device__ __forceinline__ void trap() {
    abort();  // ROCm: asm("trap;") -> abort()
}

// ROCm: PTX fences -> __builtin_amdgcn_fence, scope "" is system.
// Ordering lives in the fence; the cache modifier that makes a write visible at
// all lives in the `coherent` family below. The two are independent here.
__device__ __forceinline__ void memory_fence() {
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
}

__device__ __forceinline__ void memory_fence_gpu() {
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
}

__device__ __forceinline__ void memory_fence_cta() {
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
}

// ROCm: new, the `coherent` family carries cache modifiers and no ordering: `sc1` for
// GPU-wide visibility, `sc0 sc1` for peers. The `*_wave_global` flavour keeps the full
// dwordx4 but needs a wave-uniform base, the plain one takes any pointer at 8 bytes.
template <int kMaxBytes, typename dtype_t>
struct CoherentChunk {
    static constexpr int kBytes = (kMaxBytes >= 16 and alignof(dtype_t) >= 16) ? 16
                                  : alignof(dtype_t) >= 8                      ? 8
                                  : alignof(dtype_t) >= 4                      ? 4
                                  : alignof(dtype_t) >= 2                      ? 2
                                                                               : 1;
    static constexpr int kCount = sizeof(dtype_t) / kBytes;
    EP_STATIC_ASSERT(sizeof(dtype_t) % kBytes == 0, "unsupported coherent access layout");
};

// widest lock-free type per chunk size, for the generic (atomic) flavour
template <int kBytes>
struct AtomicChunk {};
template <>
struct AtomicChunk<8> {
    using type = long long;
};
template <>
struct AtomicChunk<4> {
    using type = int;
};
template <>
struct AtomicChunk<2> {
    using type = short;
};
template <>
struct AtomicChunk<1> {
    using type = signed char;
};

template <int kScope, typename dtype_t>
__device__ __forceinline__ dtype_t ld_coherent_impl(const dtype_t* ptr) {
    using info    = CoherentChunk<8, dtype_t>;
    using chunk_t = typename AtomicChunk<info::kBytes>::type;
    dtype_t ret;
    auto    src = reinterpret_cast<const chunk_t*>(ptr);
    auto    dst = reinterpret_cast<chunk_t*>(&ret);
#pragma unroll
    for (int i = 0; i < info::kCount; ++i)
        dst[i] = __hip_atomic_load(src + i, __ATOMIC_RELAXED, kScope);
    return ret;
}

template <int kScope, typename dtype_t>
__device__ __forceinline__ void st_coherent_impl(const dtype_t* ptr, const dtype_t& val) {
    using info    = CoherentChunk<8, dtype_t>;
    using chunk_t = typename AtomicChunk<info::kBytes>::type;
    auto src = reinterpret_cast<const chunk_t*>(&val);
    auto dst = reinterpret_cast<chunk_t*>(const_cast<dtype_t*>(ptr));
#pragma unroll
    for (int i = 0; i < info::kCount; ++i)
        __hip_atomic_store(dst + i, src[i], __ATOMIC_RELAXED, kScope);
}

template <int kCPol, typename dtype_t>
__device__ __forceinline__ dtype_t ld_coherent_wave_impl(const dtype_t* ptr) {
    constexpr int kBytes = CoherentChunk<16, dtype_t>::kBytes;
    using chunk_t        = typename BufferChunk<kBytes>::type;
    dtype_t ret;
    auto    src = reinterpret_cast<const chunk_t*>(ptr);
    auto    dst = reinterpret_cast<chunk_t*>(&ret);
#pragma unroll
    for (int i = 0; i < CoherentChunk<16, dtype_t>::kCount; ++i)
        dst[i] = buffer_ld_chunk<kCPol, kBytes>(src + i);
    return ret;
}

template <int kCPol, typename dtype_t>
__device__ __forceinline__ void st_coherent_wave_impl(const dtype_t* ptr, const dtype_t& val) {
    constexpr int kBytes = CoherentChunk<16, dtype_t>::kBytes;
    using chunk_t        = typename BufferChunk<kBytes>::type;
    auto src = reinterpret_cast<const chunk_t*>(&val);
    auto dst = reinterpret_cast<chunk_t*>(const_cast<dtype_t*>(ptr));
#pragma unroll
    for (int i = 0; i < CoherentChunk<16, dtype_t>::kCount; ++i)
        buffer_st_chunk<kCPol, kBytes>(dst + i, src[i]);
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_coherent_global(const dtype_t* ptr) {
    return ld_coherent_impl<__HIP_MEMORY_SCOPE_AGENT>(ptr);
}

template <typename dtype_t>
__device__ __forceinline__ void st_coherent_global(const dtype_t* ptr, const dtype_t& val) {
    st_coherent_impl<__HIP_MEMORY_SCOPE_AGENT>(ptr, val);
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_coherent_sys_global(const dtype_t* ptr) {
    return ld_coherent_impl<__HIP_MEMORY_SCOPE_SYSTEM>(ptr);
}

template <typename dtype_t>
__device__ __forceinline__ void st_coherent_sys_global(const dtype_t* ptr, const dtype_t& val) {
    st_coherent_impl<__HIP_MEMORY_SCOPE_SYSTEM>(ptr, val);
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_coherent_sys_wave_global(const dtype_t* ptr) {
    return ld_coherent_wave_impl<kCPolSys>(ptr);
}

template <typename dtype_t>
__device__ __forceinline__ void st_coherent_sys_wave_global(const dtype_t* ptr, const dtype_t& val) {
    st_coherent_wave_impl<kCPolSys>(ptr, val);
}

// ROCm: PTX ld/st with ordering -> __hip_atomic_*, scope sys/gpu/cta -> SYSTEM/AGENT/WORKGROUP
__device__ __forceinline__ void st_relaxed_sys_global(const int* ptr, int val) {
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// ROCm: `st.release.sys` -> a system release fence, then the coherent store.
// The fence carries the ordering, `sc0 sc1` carries the visibility.
template <typename dtype_t>
__device__ __forceinline__ void st_release_sys_global(const dtype_t* ptr, dtype_t val) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    st_coherent_sys_global(ptr, val);
}

// ROCm: `st.release.cta` -> workgroup release fence + relaxed store
__device__ __forceinline__ void st_release_cta(const int* ptr, int val) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// ROCm: `ld.acquire.sys` -> the coherent load, then a system acquire fence
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_acquire_sys_global(const dtype_t* ptr) {
    dtype_t ret = ld_coherent_sys_global(ptr);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    return ret;
}

// ROCm: `ld.acquire.gpu` -> mirror of ld_acquire_sys_global at agent scope
__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
    int ret = ld_coherent_global(ptr);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return ret;
}

// ROCm: `atom.release.*` -> the RMW's own release ordering
__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    return __hip_atomic_fetch_add(const_cast<int*>(ptr), value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    return __hip_atomic_fetch_add(const_cast<int*>(ptr), value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// ROCm: `ld.acquire.cta` -> an acquire load, LDS/workgroup needs no cache modifier
__device__ __forceinline__ int ld_acquire_cta(const int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// ROCm: `L1::no_allocate` -> agent-scope relaxed, which bypasses L1 via `sc1`
__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ROCm: ld.volatile.global -> relaxed system-scope load, these read a peer's flags
__device__ __forceinline__ int ld_volatile_global(const volatile int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ float ld_volatile_global(const volatile float* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ int64_t ld_volatile_global(const volatile int64_t* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ int64_t ld_volatile_global(const volatile uint64_t* ptr) {
    return static_cast<int64_t>(__hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
}

// ROCm: `ld.global.nc.*` -> ld_coherent_sys_global, every caller reads a peer's write
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    return ld_coherent_sys_global(ptr);
}

// ROCm: new, wave-uniform-base variant of ld_nc_global that keeps the dwordx4
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_wave_global(const dtype_t* ptr) {
    return ld_coherent_sys_wave_global(ptr);
}

// ROCm: st.relaxed.gpu.global -> relaxed agent-scope store
__device__ __forceinline__ void st_na_relaxed(const uint8_t* ptr, uint8_t val) {
    __hip_atomic_store(const_cast<uint8_t*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t* ptr, uint16_t val) {
    __hip_atomic_store(const_cast<uint16_t*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
    __hip_atomic_store(const_cast<uint32_t*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const int* ptr, int val) {
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void st_na_relaxed(const int4* ptr, int4 val) {
    // ROCm: no 128-bit atomic store on AMDGPU, so a plain vector store
    *const_cast<int4*>(ptr) = val;
}

// ROCm: `st.release.gpu.L1::no_allocate` -> agent release fence + coherent store
__device__ __forceinline__ void st_na_release(const int* ptr, int val) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    st_coherent_global(ptr, val);
}

__device__ __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    st_coherent_global(ptr, val);
}

__device__ __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    st_coherent_global(ptr, val);
}

// ROCm: `st.global.L1::no_allocate` -> st_coherent_sys_global, most callers write a peer's buffer
template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    st_coherent_sys_global(ptr, value);
}

// ROCm: new, wave-uniform-base variant of st_na_global
template <typename dtype_t>
__device__ __forceinline__ void st_na_wave_global(const dtype_t* ptr, const dtype_t& value) {
    st_coherent_sys_wave_global(ptr, value);
}

// ROCm: lg2.approx/ex2.approx -> the v_log_f32/v_exp_f32 builtins
__device__ __forceinline__ float log2f_approx(const float& x) {
    return __builtin_amdgcn_logf(x);
}

__device__ __forceinline__ float exp2f_approx(const float& x) {
    return __builtin_amdgcn_exp2f(x);
}

// ROCm: `elect.sync` is SM90-only, and lane 0 is always active here
__device__ __forceinline__ uint32_t elect_one_sync() {
    return __lane_id() == 0;
}

// ROCm: upstream's TMA block is dropped, gfx942/gfx950 have no cp.async/mbarrier

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
        recv_int_values[i] = __shfl(send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;

__forceinline__ __device__ float fast_pow2(int x) {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
}

__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
        scale = fast_pow2(-exp_scale_inv);
        scale_inv = fast_pow2(exp_scale_inv);
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;
        scale = kFinfoAmaxE4M3 / amax;
    }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
        return value;
    }
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
    if constexpr (not kSyncOnly) {
        // ROCm: `memory_fence()` -> a drain. Everything this block published carries
        // `sc0 sc1` and never entered L2, so the `buffer_wbl2` / `buffer_inv` an
        // ACQ_REL sys fence emits write back and invalidate nothing -- and cost 21%
        // of end-to-end bandwidth, measured. Only the drain is load-bearing.
        s_waitcnt();
        __syncthreads();
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, LEGACY_FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, LEGACY_FINISHED_SUM_TAG);
    }
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

    // Check timeout
    auto start_time = clock64();
    while (true) {
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        // ROCm: __all_sync(0xffffffff, ..) -> __all, the whole wave participates
        if (__all(value <= 0))
            break;

        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            trap();
        }
    }
    __syncthreads();
}

// ROCm: LDS CAS/exch PTX -> __hip_atomic_*, which take the ordering directly
__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
    __hip_atomic_compare_exchange_strong(addr, &x, y, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP);
    return x;
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
    return __hip_atomic_exchange(addr, x, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
}

__forceinline__ __device__ void acquire_lock(int* mutex) {
    // To make later memory operations valid, we must use `acquire` for memory semantics
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
        ;
}

__forceinline__ __device__ void release_lock(int* mutex) {
    // To make previous memory operations visible to other threads, we must use `release` for memory semantics
    atomic_exch_cta_release(mutex, 0);
}

// Operation functors
template <typename T>
struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct ReduceAnd {
    __device__ T operator()(T a, T b) const { return a & b; }
};
template <typename T>
struct ReduceOr {
    __device__ T operator()(T a, T b) const { return a | b; }
};

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    // ROCm: +64, a whole wave is 64 lanes here
    EP_STATIC_ASSERT(kNumLanesPerGroup == 64 or kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                         kNumLanesPerGroup == 4 or kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    // ROCm: __shfl_xor_sync(0xffffffff, ..) -> __shfl_xor, the whole wave participates
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, __shfl_xor(value, 1, WARP_SIZE));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, __shfl_xor(value, 2, WARP_SIZE));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, __shfl_xor(value, 4, WARP_SIZE));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, __shfl_xor(value, 8, WARP_SIZE));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, __shfl_xor(value, 16, WARP_SIZE));
        // ROCm: +1 step, lanes 32..63 exist on wave64
        if constexpr (kNumLanesPerGroup <= 32 and WARP_SIZE == 64)
            value = op(value, __shfl_xor(value, 32, WARP_SIZE));
    } else {
        // ROCm: +1 step, same for the intra-group direction
        if constexpr (kNumLanesPerGroup >= 64)
            value = op(value, __shfl_xor(value, 32, WARP_SIZE));
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, __shfl_xor(value, 16, WARP_SIZE));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, __shfl_xor(value, 8, WARP_SIZE));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, __shfl_xor(value, 4, WARP_SIZE));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, __shfl_xor(value, 2, WARP_SIZE));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, __shfl_xor(value, 1, WARP_SIZE));
    }
    return value;
}

// Convenience aliases
// ROCm: default 32 -> WARP_SIZE, callers that omit it mean the whole wave
template <int kNumLanesPerGroup = WARP_SIZE, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = WARP_SIZE, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = WARP_SIZE, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = WARP_SIZE, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = WARP_SIZE, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}

}  // namespace primus_turbo::deep_ep::legacy
