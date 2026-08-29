#pragma once

// [agent modifed]: new file -- the one place every arch-dependent choice lives (R8).
//
// Upstream DeepEP bakes NVIDIA facts straight into the kernels: warp == 32,
// PTX cache hints, `bar.sync <id>`, cp.async/mbarrier pipelines. Each of those
// becomes a single traits value / macro / helper here, so the kernel bodies stay
// arch-neutral and a new target only has to touch this header.
//
// Target tiers (see deepep_refactor/ARCH_NOTES.md):
//   gfx942 / gfx950 (CDNA3/4) -- wave 64, `sc0`/`sc1`, s_barrier, no cp.async
//   gfx11xx / gfx1250 (GFX11/12) -- wave 32, `th:`/`scope:`, split barriers,
//                                   split s_wait counters, async LDS copy

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo::deep_ep {

// ---------------------------------------------------------------------------
// 1. Wave geometry
// ---------------------------------------------------------------------------
// The device pass keys off the per-arch predefined macros. The host pass sees
// none of them, so it falls back to the CDNA value; building for a wave32 arch
// means passing -DPRIMUS_TURBO_DEEPEP_WARP_SIZE=32 so both passes agree.
// __AMDGCN_WAVEFRONT_SIZE__ is deliberately not used: ROCm 7.2.1 removed it.
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

// Upstream algorithms that hard-depend on 32 lanes per group (internode's
// combine) run on half a wave instead of a whole one. Fixed at 32 rather than
// kWarpSize / 2 so that a wave32 arch degenerates to the native wave.
static constexpr int32_t     kEmulatedWarpSize = 32;
static constexpr lane_mask_t kFirstHalfMask    = 0x00000000ffffffffull;
static constexpr lane_mask_t kSecondHalfMask   = 0xffffffff00000000ull;

// ---------------------------------------------------------------------------
// 2. Wave-local drain
// ---------------------------------------------------------------------------
// gfx12 split the unified s_waitcnt into per-class counters, so the combined
// "all my memory traffic has landed" wait needs four instructions there.
#if defined(__GFX12__)
#define PRIMUS_TURBO_WAIT_ALL_STR                                                                  \
    "s_wait_dscnt 0\n\ts_wait_kmcnt 0\n\ts_wait_loadcnt 0\n\ts_wait_storecnt 0"
#else
#define PRIMUS_TURBO_WAIT_ALL_STR "s_waitcnt lgkmcnt(0) vmcnt(0)"
#endif

// Drains this wave's outstanding memory ops. Covers only the calling wave --
// sibling waves need a sync_barrier first (see ARCH_NOTES.md 7).
__device__ __forceinline__ void wait_all_vmem() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile(PRIMUS_TURBO_WAIT_ALL_STR);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// CUDA's __nanosleep has no CDNA counterpart. s_sleep backs off in units of
// roughly 64 clocks and only takes a compile-time constant, so the nanosecond
// argument is dropped: ~1024 clocks is about the 500 ns DeepEP asks for.
__device__ __forceinline__ void nanosleep(int ns) {
    __builtin_amdgcn_s_sleep(16);
}

// ---------------------------------------------------------------------------
// 3. Cross-lane primitives
// ---------------------------------------------------------------------------
// HIP's __shfl_*_sync / __ballot_sync run a reconvergence loop plus a mask
// assert on every call, which is far too expensive for DeepEP's inner loops.
// The mask argument is kept for call-site parity with upstream and ignored;
// AMD wavefronts reconverge on their own.

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

// __ballot cannot reduce over a sub-range of the wave, so a half-wave group has
// to test its own half of the mask -- that is what kFirstHalfMask is for.
__device__ __forceinline__ int any_sync(lane_mask_t mask, int predicate) {
    return (__ballot(predicate) & mask) != 0;
}

__device__ __forceinline__ int all_sync(lane_mask_t mask, int predicate) {
    return (~__ballot(predicate) & mask) == 0;
}

__device__ __forceinline__ int get_lane_id() {
    return __lane_id();
}

// Lane id inside the calling thread's 32-lane group, plus the ballot mask of that
// group -- the pair an emulated 32-lane warp needs.
__device__ __forceinline__ int get_emulated_lane_id() {
    return get_lane_id() % kEmulatedWarpSize;
}

__device__ __forceinline__ lane_mask_t emulated_warp_mask() {
    if constexpr (kWarpSize == kEmulatedWarpSize)
        return kFullWarpMask;
    return get_lane_id() < kEmulatedWarpSize ? kFirstHalfMask : kSecondHalfMask;
}

// Compiler-only barrier on AMD; no hardware instruction is emitted.
__device__ __forceinline__ void syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// ---------------------------------------------------------------------------
// 4. Named barrier (CUDA `bar.sync <id>, <count>`)
// ---------------------------------------------------------------------------
// CDNA has one whole-workgroup s_barrier and no named barriers, so a
// sub-group barrier is emulated with an LDS counter. gfx12 does have named
// barriers (s_barrier_signal m0); specialising this is left for later.
//
// A monotonic arrival counter per id, plus a per-thread register holding the
// count this thread is waiting for. Nothing is ever reset, so a wave that runs
// several rounds ahead simply raises its own target -- unlike sense reversal,
// there is no window in which a late wave can miss the release and hang.
//
// Lane 0 votes for the whole wave, to keep LDS atomic pressure down; the wave
// closes up with syncwarp(). It must be a *fixed* lane rather than the first
// active one: several call sites reach the barrier straight out of a divergent
// spin, and "first active lane" would vote once per exec region, i.e. twice for
// one wave.
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

// Workgroup barrier for a *global* handoff between waves. `__syncthreads()` alone
// only orders LDS on CDNA: every wave shares the CU's L1, so LLVM emits no vmcnt
// wait and a sibling wave's global stores can still be in flight when the barrier
// releases. Use this wherever one wave writes global memory that another wave
// reads on the far side of the barrier.
__device__ __forceinline__ void sync_threads_global() {
    wait_all_vmem();
    __syncthreads();
}

// The arrival fence has to retire this wave's VMEM, not just order its LDS:
// callers use the barrier to hand a payload written by sibling waves over to
// one publishing wave, and a workgroup fence leaves those stores in flight on
// CDNA (all waves share the CU's L1, so LLVM emits no vmcnt wait for it).
__device__ __forceinline__ void barrier_arrive_fence() {
    __threadfence_block();
    wait_all_vmem();
}

#define sync_barrier_init()               PRIMUS_TURBO_BARRIER_SYNC_INIT()
#define sync_barrier(bar_id, num_threads) PRIMUS_TURBO_BARRIER_SYNC(barrier_arrive_fence(),         \
                                                                    bar_id, num_threads)

// ---------------------------------------------------------------------------
// 5. Async global -> LDS copy (CUDA cp.async.bulk + mbarrier)
// ---------------------------------------------------------------------------
// gfx942/gfx950 have no cp.async equivalent: `global_load_lds` shares vmcnt with
// ordinary loads and has no LDS->global direction. DeepEP's TMA paths therefore
// fall back to the plain synchronous copy the upstream sources already carry
// under DISABLE_SM90_FEATURES; what is lost is latency hiding, not correctness.
#if defined(__gfx1250__)
#define PRIMUS_TURBO_HAS_ASYNC_COPY 1
#else
#define PRIMUS_TURBO_HAS_ASYNC_COPY 0
#endif

// ---------------------------------------------------------------------------
// 6. GPU-initiated RDMA (NVSHMEM IBGDA)
// ---------------------------------------------------------------------------
// The low-latency kernels ring InfiniBand doorbells from the device through
// NVSHMEM's internal IBGDA structs (nvshmemi_ibgda_device_state_t and friends).
// rocSHMEM exposes no device-side QP handle at all, so there is nothing to
// emulate -- a software fallback would have to be a whole transport. The
// low-latency path is therefore compiled out; the normal (internode/intranode)
// kernels are unaffected. Flip this to 1 and implement the ibgda_device.cuh
// bodies against rocSHMEM's device API to bring the path back.
#ifndef PRIMUS_TURBO_DEEPEP_HAS_IBGDA
#define PRIMUS_TURBO_DEEPEP_HAS_IBGDA 0
#endif

} // namespace primus_turbo::deep_ep
