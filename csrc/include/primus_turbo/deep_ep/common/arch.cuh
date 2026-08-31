#pragma once

// ROCm: new file, single home for every arch-dependent choice (see ARCH_NOTES.md)

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo::deep_ep {

// ---------------------------------------------------------------------------
// 1. Wave geometry
// ---------------------------------------------------------------------------
// Host pass sees no arch macro, so wave32 targets need -DPRIMUS_TURBO_DEEPEP_WARP_SIZE=32
#ifndef PRIMUS_TURBO_DEEPEP_WARP_SIZE
#if defined(__GFX12__) || defined(__GFX11__)
#define PRIMUS_TURBO_DEEPEP_WARP_SIZE 32
#else
#define PRIMUS_TURBO_DEEPEP_WARP_SIZE 64
#endif
#endif

static constexpr int32_t kWarpSize = PRIMUS_TURBO_DEEPEP_WARP_SIZE;

// HIP's *_sync intrinsics static_assert on a 64-bit mask, on every arch.
using lane_mask_t = uint64_t;

static constexpr lane_mask_t kFullWarpMask = ~lane_mask_t(0) >> (64 - kWarpSize);

// For upstream algorithms hard-wired to 32 lanes per group
static constexpr int32_t     kEmulatedWarpSize = 32;
static constexpr lane_mask_t kFirstHalfMask    = 0x00000000ffffffffull;
static constexpr lane_mask_t kSecondHalfMask   = 0xffffffff00000000ull;

// ---------------------------------------------------------------------------
// 2. Wave-local drain
// ---------------------------------------------------------------------------
// gfx12 splits s_waitcnt into per-class counters
#if defined(__GFX12__)
#define PRIMUS_TURBO_WAIT_ALL_STR                                                                  \
    "s_wait_dscnt 0\n\ts_wait_kmcnt 0\n\ts_wait_loadcnt 0\n\ts_wait_storecnt 0"
#else
#define PRIMUS_TURBO_WAIT_ALL_STR "s_waitcnt lgkmcnt(0) vmcnt(0)"
#endif

// Drains this wave's memory ops; sibling waves need a sync_barrier first
__device__ __forceinline__ void wait_all_vmem() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile(PRIMUS_TURBO_WAIT_ALL_STR);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// ROCm: clock64() -> the steady 100 MHz counter, CDNA has no working s_memtime
__device__ __forceinline__ int64_t clock64() {
    return static_cast<int64_t>(wall_clock64());
}

// ROCm: __nanosleep -> s_sleep, the ns argument is dropped
__device__ __forceinline__ void nanosleep(int ns) {
    __builtin_amdgcn_s_sleep(16);
}

// ROCm: new, an unthrottled flag spin starves the peer's xGMI write and hangs
__device__ __forceinline__ void spin_backoff() {
    __builtin_amdgcn_s_sleep(1);
}

// ---------------------------------------------------------------------------
// 3. Cross-lane primitives
// ---------------------------------------------------------------------------
// ROCm: HIP's *_sync wrappers cost a reconvergence loop, so the mask is ignored

template <typename T>
__device__ __forceinline__ T shfl_sync(const T val, int src_lane, int width = kWarpSize,
                                       lane_mask_t mask = kFullWarpMask) {
    return __shfl(val, src_lane, width);
}

template <typename T>
__device__ __forceinline__ T shfl_xor_sync(const T val, int lane_mask, int width = kWarpSize,
                                           lane_mask_t mask = kFullWarpMask) {
    return __shfl_xor(val, lane_mask, width);
}

// __ballot spans the whole wave, so a half-wave group masks its own half
__device__ __forceinline__ int any_sync(lane_mask_t mask, int predicate) {
    return (__ballot(predicate) & mask) != 0;
}

__device__ __forceinline__ int all_sync(lane_mask_t mask, int predicate) {
    return (~__ballot(predicate) & mask) == 0;
}

__device__ __forceinline__ int get_lane_id() {
    return __lane_id();
}

// Lane id and ballot mask of the calling thread's 32-lane group
__device__ __forceinline__ int get_emulated_lane_id() {
    return get_lane_id() % kEmulatedWarpSize;
}

__device__ __forceinline__ lane_mask_t emulated_warp_mask() {
    if constexpr (kWarpSize == kEmulatedWarpSize)
        return kFullWarpMask;
    return get_lane_id() < kEmulatedWarpSize ? kFirstHalfMask : kSecondHalfMask;
}

// ROCm: wavefront-scope fences -> signal fences, no instruction either way
__device__ __forceinline__ void syncwarp() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// ---------------------------------------------------------------------------
// 4. Named barrier (CUDA `bar.sync <id>, <count>`)
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
        ___bar_sync_expected += (___num_threads_per_group) / kWarpSize;                            \
        __fence;                                                                                   \
        if (get_lane_id() == 0) {                                                                  \
            __hip_atomic_fetch_add(___bar_sync_count + (___bar_id), 1, __ATOMIC_RELAXED,           \
                                   __HIP_MEMORY_SCOPE_WORKGROUP);                                  \
            while (__hip_atomic_load(___bar_sync_count + (___bar_id), __ATOMIC_RELAXED,            \
                                     __HIP_MEMORY_SCOPE_WORKGROUP) < ___bar_sync_expected)         \
                __builtin_amdgcn_s_sleep(1);                                                       \
        }                                                                                          \
        syncwarp();                                                                                \
    }

// ROCm: new, __syncthreads() alone orders only LDS on CDNA, not global stores
__device__ __forceinline__ void sync_threads_global() {
    wait_all_vmem();
    __syncthreads();
}

// ROCm: `bar.sync`'s implicit fence -> wait_all_vmem, each wave drains before it votes
__device__ __forceinline__ void barrier_arrive_fence() {
    wait_all_vmem();
}

#define sync_barrier_init()               PRIMUS_TURBO_BARRIER_SYNC_INIT()
#define sync_barrier(bar_id, num_threads) PRIMUS_TURBO_BARRIER_SYNC(barrier_arrive_fence(),         \
                                                                    bar_id, num_threads)

// ---------------------------------------------------------------------------
// 5. Async global -> LDS copy (CUDA cp.async.bulk + mbarrier)
// ---------------------------------------------------------------------------
// ROCm: gfx942/gfx950 have no cp.async, so the TMA paths fall back to plain copies
#if defined(__gfx1250__)
#define PRIMUS_TURBO_HAS_ASYNC_COPY 1
#else
#define PRIMUS_TURBO_HAS_ASYNC_COPY 0
#endif

// ---------------------------------------------------------------------------
// 5b. Cache-policy bits on a data access
// ---------------------------------------------------------------------------
// ROCm: PTX cache hints -> gfx9 CPol bits, `sc0` (1) past the device, `sc1` (16)
// past the XCD's L2. CDNA's L2 is per-XCD and does not snoop xGMI, so these bits
// are what creates cross-rank visibility, not a tuning knob.
#if defined(__GFX12__) || defined(__GFX11__)
#error "deep_ep: CPol encoding for GFX11/GFX12 is not implemented yet"
#endif
// ROCm: FlyDSL's 18/19 minus `nt`, the peer reads the buffer back immediately
static constexpr int kCPolAgent = 16;      // sc1       -- visible GPU-wide
static constexpr int kCPolSys   = 1 | 16;  // sc1|sc0   -- visible to peers

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
    using chunk_t = typename BufferChunk<kBytes>::type;
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
    using chunk_t = typename BufferChunk<kBytes>::type;
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
// 6. GPU-initiated RDMA (NVSHMEM IBGDA)
// ---------------------------------------------------------------------------
// ROCm: IBGDA is compiled out, rocSHMEM exposes no device-side QP handle
#ifndef PRIMUS_TURBO_DEEPEP_HAS_IBGDA
#define PRIMUS_TURBO_DEEPEP_HAS_IBGDA 0
#endif

} // namespace primus_turbo::deep_ep
