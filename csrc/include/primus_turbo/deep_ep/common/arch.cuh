#pragma once

// ROCm: new file, the home for choices that are arch-dependent *and* have no native HIP spelling

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
template <int kSleepTicks>  // ~64 clocks per tick, and the builtin needs an immediate
__device__ __forceinline__ void s_sleep() {
    __builtin_amdgcn_s_sleep(kSleepTicks);
}

// ---------------------------------------------------------------------------
// 3. Named barrier (CUDA `bar.sync <id>, <count>`)
// ---------------------------------------------------------------------------
// ROCm: `bar.sync <id>` -> an LDS arrival counter, CDNA has no named barriers.
// The counter is never reset, so a wave that runs ahead just raises its own target.
static constexpr int kNumMaxBarriers = 32;

// One static object per kernel, so init and every wait land on the same counters
__device__ __forceinline__ int* barrier_counters() {
    __shared__ int counters[kNumMaxBarriers];
    return counters;
}

// ROCm: new, waits for this wave's own accesses to land and orders nothing else
__device__ __forceinline__ void s_waitcnt() {
#if defined(__GFX12__)
#error "deep_ep: gfx12 splits s_waitcnt into per-class counters, needs its own spelling"
#endif
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __builtin_amdgcn_s_waitcnt(0);  // 0 == every counter at 0
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// Returns the caller's arrival target, per-thread state that sync_barrier takes back
__device__ __forceinline__ int sync_barrier_init() {
    int* counters = barrier_counters();
    if (threadIdx.x < kNumMaxBarriers)
        counters[threadIdx.x] = 0;
    __syncthreads();
    return 0;
}

// Each wave drains before it votes, so the group's stores have all landed on release
__device__ __forceinline__ void sync_barrier(int& expected, int bar_id, int num_threads_per_group) {
    int* counters = barrier_counters();
    expected += num_threads_per_group / WARP_SIZE;
    s_waitcnt();
    if (__lane_id() == 0) {  // fixed lane, not first-active: one vote per wave
        __hip_atomic_fetch_add(counters + bar_id, 1, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_WORKGROUP);
        while (__hip_atomic_load(counters + bar_id, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_WORKGROUP) < expected)
            s_sleep<1>();
    }
    __syncwarp();
}

// ---------------------------------------------------------------------------
// 4. Cache-policy bits on a data access
// ---------------------------------------------------------------------------
// ROCm: PTX cache hints -> gfx9 CPol bits, `sc0` (1) past the device, `sc1` (16) past L2.
// CDNA's L2 is per-XCD and does not snoop xGMI, so these bits are the cross-rank visibility.
#if defined(__GFX12__) || defined(__GFX11__)
#error "deep_ep: CPol encoding for GFX11/GFX12 is not implemented yet"
#endif
// ROCm: no `nt`, the peer reads the buffer back immediately
static constexpr int kSystemScopeCachePolicy = 1 | 16;

// ROCm: new, a buffer op is how CPol reaches the instruction at full width.
// Precondition: wave-uniform base, non-negative per-lane delta, both bounded by `num_records`.

// ROCm: buffer resource word 3, DATA_FORMAT=7 | NUM_FORMAT=4; 0 silently drops every access
#if defined(__GFX12__) || defined(__GFX11__)
#error "deep_ep: buffer resource word 3 for GFX11/GFX12 is not implemented yet"
#endif
static constexpr int kBufferRsrcWord3 = (7 << 12) | (4 << 15);
template <int kBytes> struct BufferChunk;
template <> struct BufferChunk<16> { using type = int __attribute__((ext_vector_type(4))); };
template <> struct BufferChunk<8>  { using type = int __attribute__((ext_vector_type(2))); };
template <> struct BufferChunk<4>  { using type = int; };
template <> struct BufferChunk<2>  { using type = short; };
template <> struct BufferChunk<1>  { using type = signed char; };

// `num_records` counts bytes from the base, stride 0 leaves it unswizzled
__device__ __forceinline__ auto make_wave_rsrc(const void* ptr, uint32_t num_records,
                                               uint32_t& voffset) {
    auto     addr = reinterpret_cast<uintptr_t>(ptr);
    uint32_t lo   = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr));
    uint32_t hi   = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32));
    auto     base = (static_cast<uint64_t>(hi) << 32) | lo;
    voffset       = static_cast<uint32_t>(addr - base);
    return __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>(base), static_cast<int16_t>(0), num_records, kBufferRsrcWord3);
}

template <int kCachePolicy, int kBytes>
__device__ __forceinline__ typename BufferChunk<kBytes>::type buffer_ld_chunk(const void* ptr,
                                                                              uint32_t num_records) {
    uint32_t voffset;
    auto     rsrc = make_wave_rsrc(ptr, num_records, voffset);
    if constexpr (kBytes == 16)
        return __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, 0, kCachePolicy);
    else if constexpr (kBytes == 8)
        return __builtin_amdgcn_raw_buffer_load_b64(rsrc, voffset, 0, kCachePolicy);
    else if constexpr (kBytes == 4)
        return __builtin_amdgcn_raw_buffer_load_b32(rsrc, voffset, 0, kCachePolicy);
    else if constexpr (kBytes == 2)
        return __builtin_amdgcn_raw_buffer_load_b16(rsrc, voffset, 0, kCachePolicy);
    else
        return __builtin_amdgcn_raw_buffer_load_b8(rsrc, voffset, 0, kCachePolicy);
}

template <int kCachePolicy, int kBytes>
__device__ __forceinline__ void buffer_st_chunk(void* ptr,
                                                typename BufferChunk<kBytes>::type val,
                                                uint32_t                           num_records) {
    uint32_t voffset;
    auto     rsrc = make_wave_rsrc(ptr, num_records, voffset);
    if constexpr (kBytes == 16)
        __builtin_amdgcn_raw_buffer_store_b128(val, rsrc, voffset, 0, kCachePolicy);
    else if constexpr (kBytes == 8)
        __builtin_amdgcn_raw_buffer_store_b64(val, rsrc, voffset, 0, kCachePolicy);
    else if constexpr (kBytes == 4)
        __builtin_amdgcn_raw_buffer_store_b32(val, rsrc, voffset, 0, kCachePolicy);
    else if constexpr (kBytes == 2)
        __builtin_amdgcn_raw_buffer_store_b16(val, rsrc, voffset, 0, kCachePolicy);
    else
        __builtin_amdgcn_raw_buffer_store_b8(val, rsrc, voffset, 0, kCachePolicy);
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

// ---------------------------------------------------------------------------
// 6. FP8 capability
// ---------------------------------------------------------------------------
// ROCm: upstream's DISABLE_SM90_FEATURES gates TMA and FP8 together. CDNA has no TMA, so
// the flag stays set and the kernels keep taking upstream's non-SM90 branch -- but FP8 is
// there, so the two have to be split. This one only says "the caller may hand us FP8".
//
// Arch-independent because the ported intranode path never interprets the bytes: dispatch
// and combine move opaque payload plus a separate `float` scale tensor, so gfx942's fnuz
// e4m3 and gfx950's OCP e4m3 both just ride through. The encode/decode that *does* care
// lives in internode_ll.cu (hard-coded `__NV_E4M3`), and that file is compiled out --
// porting it means picking the format per arch, not reusing this flag.
constexpr bool kEnableFP8Features = true;

} // namespace primus_turbo::deep_ep
