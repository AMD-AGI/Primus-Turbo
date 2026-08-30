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

// CDNA3/4 dropped `s_memtime`, but HIP's clock64() still lowers to it, so every
// spin-timeout in DeepEP compares against a garbage counter and trips at random
// (measured: a 200G-cycle budget "expiring" after ~2 ms of real time). Shadow it
// with the steady 100 MHz counter (`s_memrealtime`); unqualified clock64() calls
// in the kernels resolve here by namespace lookup, so no call site changes.
// LEGACY_NUM_TIMEOUT_CYCLES is expressed in these ticks (see compiled.cuh).
__device__ __forceinline__ int64_t clock64() {
    return static_cast<int64_t>(wall_clock64());
}

// CUDA's __nanosleep has no CDNA counterpart. s_sleep backs off in units of
// roughly 64 clocks and only takes a compile-time constant, so the nanosecond
// argument is dropped: ~1024 clocks is about the 500 ns DeepEP asks for.
__device__ __forceinline__ void nanosleep(int ns) {
    __builtin_amdgcn_s_sleep(16);
}

// Backoff for the flag spins upstream leaves with an empty body. A system-scope
// load lowers to `sc0 sc1`, which bypasses the caches, so an empty spin polls the
// fabric back-to-back; with hundreds of waves doing that the peer's XGMI write to
// the same line is starved and the spin never observes it (measured: hangs on the
// first dispatch, 100/100 clean with this backoff). Kept separate from nanosleep()
// so the wait can be retuned per arch without touching upstream call sites.
__device__ __forceinline__ void spin_backoff() {
    __builtin_amdgcn_s_sleep(1);
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
// [agent modifed]: wavefront-scope acquire/release fences -> signal fences. A
// wavefront-scope fence emits no instruction anyway (the wave is lockstep), it only
// pins the scheduler, which is exactly what a signal fence is -- and this port bans
// acquire/release everywhere (see the coherence rule in legacy/utils.cuh).
__device__ __forceinline__ void syncwarp() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
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

// [agent modifed]: `bar.sync`'s implicit fence -> wait_all_vmem().
// A workgroup fence emits no vmcnt wait on CDNA (every wave shares the CU's L1), so a
// sibling wave's payload store could still be in flight when the publishing wave moves
// the channel tail. The vmcnt wait is what each wave contributes to the handoff: it
// drains that wave before it votes, so once the barrier releases the whole group's
// payload has landed and the tail store may go out relaxed (see legacy/utils.cuh).
__device__ __forceinline__ void barrier_arrive_fence() {
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
// 5b. Cache-policy bits on a data access
// ---------------------------------------------------------------------------
// Upstream spells its cache hints in PTX (`ld.global.nc.L1::no_allocate`,
// `st.global.L1::no_allocate`) and treats them as pure performance requests: one
// snooped, GPU-wide L2 makes a plain NVIDIA access coherent with every writer that
// matters. Neither half holds on CDNA -- L2 is per-XCD and does not snoop xGMI --
// so on this arch the same hints are what *creates* visibility, and getting them
// onto the instruction is a correctness requirement, not a tuning knob.
//
// On gfx9 they live in the CPol field: `sc0` (1) takes the access past the device,
// `sc1` (16) past the XCD's L2, `nt` (2) is upstream's streaming hint. gfx11/gfx12
// spell the same intent as `th:`/`scope:` with a different encoding, hence the
// branch. The pairing below is the one FlyDSL's mega-MoE EP kernels already run on
// gfx950 (primus_turbo/flydsl/mega/ep_intranode.py: 18 for same-agent reads,
// 19 to publish to a peer).
#if defined(__GFX12__) || defined(__GFX11__)
#error "deep_ep: CPol encoding for GFX11/GFX12 is not implemented yet"
#endif
// [agent modifed]: FlyDSL's 18/19 minus `nt`. Non-temporal is the wrong hint for a
// buffer the peer reads back immediately, and it costs: with `nt` the small-batch
// cases sat 3-4.5% under baseline, without it every case is at or above baseline.
static constexpr int kCPolAgent = 16;      // sc1       -- visible GPU-wide
static constexpr int kCPolSys   = 1 | 16;  // sc1|sc0   -- visible to peers

// Only a buffer (or flat) op carries CPol, and only a *wave-uniform* descriptor
// keeps the backend from expanding a divergent base into a readfirstlane waterfall.
// So the access is rebuilt as `readfirstlane(ptr)` + a per-lane byte offset.
//
// That makes the descriptor correct ONLY when the caller's base is wave-uniform and
// the per-lane delta is non-negative and under 4 GiB -- the first active lane must
// hold the lowest address. Callers that cannot promise this (a metadata access where
// each lane picks a different rank's buffer) must not come here; legacy/utils.cuh
// keeps a second, pointer-agnostic flavour for them.
//
// The alternatives were measured and rejected: `volatile` gets the bits but pins an
// `s_waitcnt vmcnt(0)` after every access, relaxed system-scope atomics cap at 8
// bytes (halving the copy width), and inline asm gets the bits but leaves the
// backend blind to vmcnt.

// [agent modifed]: buffer resource word 3 (dst_sel / num_format / data_format).
// Must be gfx9's DATA_FORMAT=32; leaving it 0 encodes an invalid format and the
// hardware silently drops every load and store.
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
