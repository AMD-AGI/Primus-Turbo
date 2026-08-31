#pragma once

// ROCm: new file, single home for every arch-dependent choice.
// Rule of thumb for what belongs here: it must be architecture-dependent *and*
// have no native HIP spelling. Cross-lane shuffles/votes/barriers do have one
// (`__shfl`, `__any`, `__syncwarp`, ...), so they are used directly at the call
// sites instead of being wrapped here.

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo::deep_ep {

// ---------------------------------------------------------------------------
// 1. Wave geometry
// ---------------------------------------------------------------------------
#if defined(__GFX12__) || defined(__GFX11__)
#define WARP_SIZE 32
#else
#define WARP_SIZE 64
#endif

// ---------------------------------------------------------------------------
// 2. Timing and spin throttling
// ---------------------------------------------------------------------------
// ROCm: clock64() -> the steady 100 MHz counter, CDNA has no working s_memtime
__device__ __forceinline__ int64_t clock64() {
    return static_cast<int64_t>(wall_clock64());
}

// ROCm: new, an unthrottled flag spin starves the peer's xGMI write and hangs
__device__ __forceinline__ void spin_backoff() {
    __builtin_amdgcn_s_sleep(1);
}

// ---------------------------------------------------------------------------
// 3. Named barrier (CUDA `bar.sync <id>, <count>`)
// ---------------------------------------------------------------------------
// ROCm: `bar.sync <id>` -> an LDS arrival counter, CDNA has no named barriers.
// The counter is monotonic and never reset, so a wave that runs ahead just raises
// its own target; lane 0 (fixed, not first-active) votes once per wave.
#define PRIMUS_TURBO_NUM_MAX_BARRIERS 32

#define PRIMUS_TURBO_BARRIER_SYNC_INIT()                                                           \
    __shared__ int ___bar_sync_count[PRIMUS_TURBO_NUM_MAX_BARRIERS];                               \
    if (threadIdx.x < PRIMUS_TURBO_NUM_MAX_BARRIERS)                                               \
        ___bar_sync_count[threadIdx.x] = 0;                                                        \
    int ___bar_sync_expected = 0;                                                                  \
    __syncthreads();

#define PRIMUS_TURBO_BARRIER_SYNC(__fence, ___bar_id, ___num_threads_per_group)                    \
    {                                                                                              \
        ___bar_sync_expected += (___num_threads_per_group) / WARP_SIZE;                            \
        __fence;                                                                                   \
        if (__lane_id() == 0) {                                                                    \
            __hip_atomic_fetch_add(___bar_sync_count + (___bar_id), 1, __ATOMIC_RELAXED,           \
                                   __HIP_MEMORY_SCOPE_WORKGROUP);                                  \
            while (__hip_atomic_load(___bar_sync_count + (___bar_id), __ATOMIC_RELAXED,            \
                                     __HIP_MEMORY_SCOPE_WORKGROUP) < ___bar_sync_expected)         \
                __builtin_amdgcn_s_sleep(1);                                                       \
        }                                                                                          \
        __syncwarp();                                                                              \
    }

// ROCm: the instruction, named after itself -- waits for this wave's own accesses
// to land and nothing else. That is all `bar.sync`'s implicit fence needs to be
// here: every fence scope wide enough to emit `s_waitcnt vmcnt(0)` also drags in
// `buffer_wbl2` / `buffer_inv`, which have nothing to write back or invalidate
// when the payload carries `sc0 sc1` and never enters L2.
__device__ __forceinline__ void s_waitcnt() {
#if defined(__GFX12__)
#error "deep_ep: gfx12 splits s_waitcnt into per-class counters, needs its own spelling"
#endif
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __builtin_amdgcn_s_waitcnt(0);  // 0 == every counter at 0
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// ROCm: new, __syncthreads() alone orders only LDS on CDNA, not global stores
__device__ __forceinline__ void sync_threads_global() {
    s_waitcnt();
    __syncthreads();
}

#define sync_barrier_init()               PRIMUS_TURBO_BARRIER_SYNC_INIT()
// Each wave drains before it votes, so the group's stores have all landed on release
#define sync_barrier(bar_id, num_threads) PRIMUS_TURBO_BARRIER_SYNC(s_waitcnt(),                    \
                                                                    bar_id, num_threads)

// ---------------------------------------------------------------------------
// 4. Cache-policy bits on a data access
// ---------------------------------------------------------------------------
// ROCm: PTX cache hints -> gfx9 CPol bits, `sc0` (1) past the device, `sc1` (16)
// past the XCD's L2. CDNA's L2 is per-XCD and does not snoop xGMI, so these bits
// are what creates cross-rank visibility, not a tuning knob.
#if defined(__GFX12__) || defined(__GFX11__)
#error "deep_ep: CPol encoding for GFX11/GFX12 is not implemented yet"
#endif
// ROCm: FlyDSL's 19 minus `nt`, the peer reads the buffer back immediately
static constexpr int kCPolSys = 1 | 16;  // sc0|sc1 -- visible to peers

// ROCm: new, a buffer op is how CPol reaches the instruction at full width.
// Precondition: the base must be wave-uniform and the per-lane delta non-negative
// and under 4 GiB. Divergent bases belong on the atomic flavour in legacy/utils.cuh.

// ROCm: buffer resource word 3 must be gfx9 DATA_FORMAT=32, 0 is silently dropped
#if defined(__GFX12__) || defined(__GFX11__)
#error "deep_ep: buffer resource word 3 for GFX11/GFX12 is not implemented yet"
#endif
static constexpr int kBufferRsrcWord3 = 0x00020000;
template <int kBytes> struct BufferChunk;
template <> struct BufferChunk<16> { using type = int __attribute__((ext_vector_type(4))); };
template <> struct BufferChunk<8>  { using type = int __attribute__((ext_vector_type(2))); };
template <> struct BufferChunk<4>  { using type = int; };
template <> struct BufferChunk<2>  { using type = short; };
template <> struct BufferChunk<1>  { using type = signed char; };

__device__ __forceinline__ auto make_wave_rsrc(const void* ptr, uint32_t& voffset) {
    auto     addr = reinterpret_cast<uintptr_t>(ptr);
    uint32_t lo   = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr));
    uint32_t hi   = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32));
    auto     base = (static_cast<uint64_t>(hi) << 32) | lo;
    voffset       = static_cast<uint32_t>(addr - base);
    return __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>(base), static_cast<int16_t>(0), 0xffffffffu, kBufferRsrcWord3);
}

template <int kCPol, int kBytes>
__device__ __forceinline__ typename BufferChunk<kBytes>::type buffer_ld_chunk(const void* ptr) {
    uint32_t voffset;
    auto     rsrc = make_wave_rsrc(ptr, voffset);
    if constexpr (kBytes == 16)
        return __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, 0, kCPol);
    else if constexpr (kBytes == 8)
        return __builtin_amdgcn_raw_buffer_load_b64(rsrc, voffset, 0, kCPol);
    else if constexpr (kBytes == 4)
        return __builtin_amdgcn_raw_buffer_load_b32(rsrc, voffset, 0, kCPol);
    else if constexpr (kBytes == 2)
        return __builtin_amdgcn_raw_buffer_load_b16(rsrc, voffset, 0, kCPol);
    else
        return __builtin_amdgcn_raw_buffer_load_b8(rsrc, voffset, 0, kCPol);
}

template <int kCPol, int kBytes>
__device__ __forceinline__ void buffer_st_chunk(void* ptr,
                                                typename BufferChunk<kBytes>::type val) {
    uint32_t voffset;
    auto     rsrc = make_wave_rsrc(ptr, voffset);
    if constexpr (kBytes == 16)
        __builtin_amdgcn_raw_buffer_store_b128(val, rsrc, voffset, 0, kCPol);
    else if constexpr (kBytes == 8)
        __builtin_amdgcn_raw_buffer_store_b64(val, rsrc, voffset, 0, kCPol);
    else if constexpr (kBytes == 4)
        __builtin_amdgcn_raw_buffer_store_b32(val, rsrc, voffset, 0, kCPol);
    else if constexpr (kBytes == 2)
        __builtin_amdgcn_raw_buffer_store_b16(val, rsrc, voffset, 0, kCPol);
    else
        __builtin_amdgcn_raw_buffer_store_b8(val, rsrc, voffset, 0, kCPol);
}

// ---------------------------------------------------------------------------
// 5. Multi-node paths
// ---------------------------------------------------------------------------
// ROCm: internode is not ported yet, its kernels are upstream NVSHMEM + PTX
#ifndef PRIMUS_TURBO_DEEPEP_HAS_INTERNODE
#define PRIMUS_TURBO_DEEPEP_HAS_INTERNODE 0
#endif

// ROCm: IBGDA is compiled out, rocSHMEM exposes no device-side QP handle
#ifndef PRIMUS_TURBO_DEEPEP_HAS_IBGDA
#define PRIMUS_TURBO_DEEPEP_HAS_IBGDA 0
#endif

} // namespace primus_turbo::deep_ep
