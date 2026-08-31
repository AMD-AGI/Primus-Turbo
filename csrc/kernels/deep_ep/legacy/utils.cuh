#pragma once

#include <primus_turbo/deep_ep/common/compiled.cuh>
#include <primus_turbo/deep_ep/common/exception.cuh>

// [agent modifed]: 32 -> kWarpSize   the loop stride is one warp of lanes
#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                              \
    {                                                                                                                                         \
        constexpr int kLoopStride = kWarpSize * (UNROLL_FACTOR);                                                                              \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                                  \
        auto __src = (SRC);                                                                                                                    \
        auto __dst = (DST);                                                                                                                    \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                              \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * kWarpSize);  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * kWarpSize, unrolled_values[__j]);   \
        }                                                                                                                                     \
        {                                                                                                                                     \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                          \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                               \
                if (__i + __j * kWarpSize < (N)) {                                                                                            \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * kWarpSize);                                                            \
                }                                                                                                                             \
            }                                                                                                                                 \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                               \
                if (__i + __j * kWarpSize < (N)) {                                                                                            \
                    ST_FUNC(__dst + __i + __j * kWarpSize, unrolled_values[__j]);                                                             \
                }                                                                                                                             \
            }                                                                                                                                 \
        }                                                                                                                                     \
    }

// [agent modifed]: new -- half-wave variant for the internode paths whose lane
// assignment is hard-wired to 32 peers and cannot widen to a whole wave64.
#define UNROLLED_WARP_COPY_EMULATED(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                          \
    {                                                                                                                                              \
        constexpr int kLoopStride = kEmulatedWarpSize * (UNROLL_FACTOR);                                                                           \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                                       \
        auto __src = (SRC);                                                                                                                         \
        auto __dst = (DST);                                                                                                                         \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                                   \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * kEmulatedWarpSize); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * kEmulatedWarpSize, unrolled_values[__j]);  \
        }                                                                                                                                          \
        for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += kEmulatedWarpSize)                                         \
            ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                                                                                            \
    }

// [agent modifed]: deep_ep::legacy -> primus_turbo::deep_ep::legacy
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
    // [agent modifed]: int4 -> ext_vector_type(4)   __builtin_nontemporal_load rejects HIP vector structs
    using vec_t = int __attribute__((ext_vector_type(4)));
};

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__ explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

    __device__ __host__ auto operator[](const uint32_t& i) { return func(i); }
};

__device__ __forceinline__ void trap() {
    abort();  // [agent modifed]: asm("trap;") -> abort()
}

// [agent modifed]: PTX fences -> wait_all_vmem(), i.e. s_waitcnt only.
// __threadfence_system()/__threadfence() are acq_rel fences in all but name: they add
// `buffer_wbl2`/`buffer_inv` to push the payload through this XCD's L2. Every shared
// access on this port carries `sc0 sc1` and never lands in L2, so the L2 maintenance
// has nothing to do and only the ordering half -- draining the wave -- is real.
__device__ __forceinline__ void memory_fence() {
    wait_all_vmem();
}

__device__ __forceinline__ void memory_fence_gpu() {
    wait_all_vmem();
}

__device__ __forceinline__ void memory_fence_cta() {
    wait_all_vmem();
}

// [agent modifed]: new -- the `coherent` access family, upstream has no counterpart.
//
// Upstream reaches the shared comm buffer with `ld.global.nc` / `st.global` and
// leans on NVIDIA's single, GPU-wide, peer-snooped L2: a plain load there is
// already coherent with every writer that matters. CDNA has neither half of that
// property. gfx950 splits L2 per XCD, and a peer writing over xGMI never
// invalidates the reader's copy, so a plain access to memory some *other* agent
// reads or writes can read a line that is arbitrarily stale.
//
// These four spell the one thing the algorithm actually needs there -- visibility,
// at a stated reach, with no ordering attached:
//
//   *_coherent_global      visible across the whole GPU  (`sc1`, bypasses CU L1)
//   *_coherent_sys_global  visible to peer GPUs as well  (`sc0 sc1`, also L2)
//
// They are named against upstream's `ld_nc_global` ("non-coherent"), and slot into
// this file's `<op>_<ordering>_<scope>_global` shape with `coherent` in the
// ordering slot: strictly weaker than `relaxed`, since no atomicity is implied
// either -- only the cache modifiers. That is also why they are not atomics: the
// bits are all the algorithm needs, and asking for ordering on top would emit
// `buffer_inv` / `buffer_wbl2`, whole-cache operations far more expensive than the
// point ordering DeepEP wants -- the `s_waitcnt vmcnt(0)` a wave already executes
// before publishing a flag orders the payload behind it.
//
// Getting these bits right is what lets the comm buffer stay ordinary cached
// memory. An uncached allocation hides a missing modifier, but only for the owner:
// an IPC mapping does not inherit the owner's MTYPE, so the peer doing the write
// still lands in its own L2.
//
// There are two flavours, because only one of the two ways to get the bits onto an
// instruction is universally applicable:
//
//   *_coherent_*_global       any pointer. A relaxed system/agent-scope atomic,
//                             which AMDGPU lowers to exactly the bits and nothing
//                             else. Capped at 8 bytes -- HIP has no wider
//                             lock-free type -- so an `int4` splits into two
//                             dwordx2. Fine for the metadata this moves.
//   *_coherent_*_wave_global  wave-uniform base pointer only. Rides a buffer
//                             descriptor (see common/arch.cuh) and keeps the full
//                             dwordx4, which is what the payload copy needs: the
//                             8-byte cap alone costs ~55% of dispatch bandwidth.
//
// The wave flavour's precondition is load-bearing, not stylistic. The descriptor
// must sit in SGPRs, so its base is `readfirstlane` of the pointer and every other
// lane addresses it through a 32-bit unsigned `voffset`. That only holds when all
// lanes share a base and differ by a non-negative per-lane offset -- true of
// UNROLLED_WARP_COPY over one token, false of a metadata access where each lane
// picks a different rank's buffer. Get it wrong and the delta wraps: out-of-range
// reads return zero, writes vanish, and the descriptor faults.
//
// An element is moved as the widest chunk its alignment allows, so the byte count
// matches a plain copy either way and the compiler still tracks vmcnt.
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

// [agent modifed]: PTX ld/st with explicit ordering -> __hip_atomic_*.
// Scope map: sys -> SYSTEM, gpu -> AGENT, cta -> WORKGROUP. On AMDGPU a relaxed
// system-scope access already carries the `sc0 sc1` bits that make it visible
// across ranks; only acquire/release add the L2 writeback/invalidate on top
// (ARCH_NOTES.md 7), so relaxed is preferred wherever the protocol allows it.
__device__ __forceinline__ void st_relaxed_sys_global(const int* ptr, int val) {
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// [agent modifed]: `release` -> s_waitcnt + relaxed. The release lowering adds a
// `buffer_wbl2` on top of the vmcnt wait, to push the payload out of the sender's L2.
// Every payload access on this port already carries `sc0 sc1` and never lands in L2
// (see kCPolSys in common/arch.cuh), so the write-back has nothing left to flush and
// only the ordering half is real: drain this wave, then publish. Sibling waves are
// drained by barrier_arrive_fence(), so callers must sync_barrier() the group first.
template <typename dtype_t>
__device__ __forceinline__ void st_release_sys_global(const dtype_t* ptr, dtype_t val) {
    wait_all_vmem();
    __hip_atomic_store(const_cast<dtype_t*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// [agent modifed]: `release` -> s_waitcnt + relaxed, same reasoning as the sys-scope
// pair above; at workgroup scope the release adds nothing but the drain anyway.
__device__ __forceinline__ void st_release_cta(const int* ptr, int val) {
    wait_all_vmem();
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// [agent modifed]: `acquire` -> s_waitcnt + relaxed, the mirror of the store above.
// The acquire lowering adds a `buffer_inv` so later reads miss the stale L2 copy; the
// payload reads that follow carry `sc0 sc1` and bypass L2 anyway, so the invalidate is
// dead weight and the vmcnt wait is what actually orders flag-then-payload.
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_acquire_sys_global(const dtype_t* ptr) {
    wait_all_vmem();
    dtype_t ret = __hip_atomic_load(const_cast<dtype_t*>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return ret;
}

// [agent modifed]: `acquire` -> s_waitcnt + relaxed, mirror of ld_acquire_sys_global.
__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
    wait_all_vmem();
    int ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return ret;
}

// [agent modifed]: `release` -> s_waitcnt + relaxed; the RMW itself is unchanged.
__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    wait_all_vmem();
    int ret = __hip_atomic_fetch_add(const_cast<int*>(ptr), value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return ret;
}

// [agent modifed]: `release` -> s_waitcnt + relaxed.
__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    wait_all_vmem();
    int ret = __hip_atomic_fetch_add(const_cast<int*>(ptr), value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return ret;
}

// [agent modifed]: `acquire` -> s_waitcnt + relaxed.
__device__ __forceinline__ int ld_acquire_cta(const int* ptr) {
    wait_all_vmem();
    int ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return ret;
}

// [agent modifed]: `L1::no_allocate` has no AMDGPU spelling; agent-scope relaxed
// is the closest (it bypasses L1 via `sc1`). Used by the IBGDA path only.
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

// [agent modifed]: ld.volatile.global -> relaxed system-scope load. These read
// flags a peer rank published, so they must carry the full scope bits. The
// operand stays `volatile` like upstream's: these all sit in flag spin loops,
// and without it the compiler is free to hoist the load out of the loop.
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

// [agent modifed]: `ld.global.nc.L1::no_allocate.L2::256B` -> ld_coherent_sys_global.
// Every caller reads the comm buffer, which a peer rank wrote; upstream's
// non-coherent hint is safe there only because one snooped L2 serves the whole
// NVIDIA GPU. On CDNA the same load would hit a line the writer never
// invalidated, so the hint inverts into its opposite. One template replaces
// upstream's per-width PTX specialisations.
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    return ld_coherent_sys_global(ptr);
}

// [agent modifed]: new -- wave-uniform-base variant of ld_nc_global, for the
// UNROLLED_WARP_COPY payload path only. Same bits, but keeps the dwordx4.
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_wave_global(const dtype_t* ptr) {
    return ld_coherent_sys_wave_global(ptr);
}

// [agent modifed]: st.relaxed.gpu.global -> relaxed agent-scope store
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
    // [agent modifed]: no 128-bit atomic store on AMDGPU; a plain vector store is
    // what the PTX lowers to anyway, the `relaxed` here orders nothing extra.
    *const_cast<int4*>(ptr) = val;
}

// [agent modifed]: `release` -> s_waitcnt + relaxed, as everywhere else on this port.
__device__ __forceinline__ void st_na_release(const int* ptr, int val) {
    wait_all_vmem();
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

__device__ __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
    wait_all_vmem();
    __hip_atomic_store(const_cast<uint32_t*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

__device__ __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
    wait_all_vmem();
    __hip_atomic_store(const_cast<uint64_t*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// [agent modifed]: `st.global.L1::no_allocate` -> st_coherent_sys_global. Upstream's
// cache hint is a pure performance request -- a plain store is already correct on
// NVIDIA. It is not on CDNA: most callers here are the dispatch/combine senders
// writing straight into a *peer's* comm buffer, where a store left in the local L2
// is never seen. Two callers write a local output tensor instead and pay the
// bypass for nothing; splitting them is a perf lever to pull once the protocol is
// green, not a correctness matter.
template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    st_coherent_sys_global(ptr, value);
}

// [agent modifed]: new -- wave-uniform-base variant of st_na_global, see above.
template <typename dtype_t>
__device__ __forceinline__ void st_na_wave_global(const dtype_t* ptr, const dtype_t& value) {
    st_coherent_sys_wave_global(ptr, value);
}

// [agent modifed]: lg2.approx/ex2.approx -> the v_log_f32/v_exp_f32 builtins,
// which are the same one-instruction approximations. HIP has no __exp2f.
__device__ __forceinline__ float log2f_approx(const float& x) {
    return __builtin_amdgcn_logf(x);
}

__device__ __forceinline__ float exp2f_approx(const float& x) {
    return __builtin_amdgcn_exp2f(x);
}

// [agent modifed]: get_lane_id / elect_one_sync / warp primitives live in
// common/arch.cuh now; `elect.sync` is SM90-only and upstream's fallback below
// is the branch ROCm takes.
__device__ __forceinline__ uint32_t elect_one_sync() {
    return get_lane_id() == 0;
}

// [agent modifed]: upstream's TMA block (mbarrier_*, tma_load_1d, tma_store_1d)
// is guarded by DISABLE_SM90_FEATURES, which common/compiled.cuh defines on
// ROCm. gfx942/gfx950 have no cp.async/mbarrier hardware at all, so the block is
// dropped rather than stubbed -- an empty shell would only mislead. See
// PRIMUS_TURBO_HAS_ASYNC_COPY in arch.cuh for where gfx1250 would hook in.

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
        // [agent modifed]: __shfl_sync(0xffffffff, ..) -> shfl_sync   32-bit mask is rejected by HIP
        recv_int_values[i] = shfl_sync(send_int_values[i], src_lane_idx);
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
        memory_fence();
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
        // [agent modifed]: __all_sync(0xffffffff, ..) -> all_sync(kFullWarpMask, ..)
        if (all_sync(kFullWarpMask, value <= 0))
            break;

        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            trap();
        }
    }
    __syncthreads();
}

// [agent modifed]: shared-memory CAS/exch PTX -> __hip_atomic_* on the LDS pointer
// [agent modifed]: `acquire` -> s_waitcnt + relaxed; the lock guards LDS state, so the
// lgkmcnt half of the drain is what actually orders it.
__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
    __hip_atomic_compare_exchange_strong(addr, &x, y, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    wait_all_vmem();
    return x;
}

// [agent modifed]: `release` -> s_waitcnt + relaxed.
__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
    wait_all_vmem();
    int ret = __hip_atomic_exchange(addr, x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return ret;
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
    // [agent modifed]: +64   a whole wave is 64 lanes here, so the group size can be too
    EP_STATIC_ASSERT(kNumLanesPerGroup == 64 or kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                         kNumLanesPerGroup == 4 or kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    // [agent modifed]: 0xffffffff -> kFullWarpMask   HIP rejects a 32-bit mask
    constexpr lane_mask_t mask = kFullWarpMask;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, shfl_xor_sync(value, 1, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, shfl_xor_sync(value, 2, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, shfl_xor_sync(value, 4, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, shfl_xor_sync(value, 8, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, shfl_xor_sync(value, 16, kWarpSize, mask));
        // [agent modifed]: +1 step   lanes 32..63 exist on wave64
        if constexpr (kNumLanesPerGroup <= 32 and kWarpSize == 64)
            value = op(value, shfl_xor_sync(value, 32, kWarpSize, mask));
    } else {
        // [agent modifed]: +1 step   ditto, for the intra-group direction
        if constexpr (kNumLanesPerGroup >= 64)
            value = op(value, shfl_xor_sync(value, 32, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, shfl_xor_sync(value, 16, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, shfl_xor_sync(value, 8, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, shfl_xor_sync(value, 4, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, shfl_xor_sync(value, 2, kWarpSize, mask));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, shfl_xor_sync(value, 1, kWarpSize, mask));
    }
    return value;
}

// Convenience aliases
// [agent modifed]: default 32 -> kWarpSize   callers that omit it mean "the whole warp"
template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}

}  // namespace primus_turbo::deep_ep::legacy
