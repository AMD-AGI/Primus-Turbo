#pragma once

#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <type_traits>

#ifndef __grid_constant__
#define __grid_constant__
#endif

#include "primus_turbo/device/lds_swizzle.cuh"
#include "primus_turbo/device/memory.cuh"
#include "primus_turbo/device/mfma.cuh"
#include "primus_turbo/device/register.cuh"
#include "primus_turbo/dtype.h"

#include "../layout/mega_moe.cuh"
#include "../layout/sym_buffer.cuh"
#include "../scheduler/mega_moe.cuh"
#include "prims.cuh"

namespace primus_turbo {
namespace mega_moe {

// PROBE (default 0): drop the A-frag ds_read wait at the k-burst head.  turbo
// never waits right after issuing ds_read -- the operand's own consumer-side
// wait (run_op's B[0] wait_lgkmcnt<0>) drains the A frags before the first
// MFMA, so the head wait may be a redundant fully-exposed stall.
#ifndef MEGA_MOE_NO_AHEAD_WAIT
#define MEGA_MOE_NO_AHEAD_WAIT 1
#endif

// PROBE (default 0): cross-operand B[0] prefetch.  turbo hides an operand's
// first ds_read under the previous operand's tail MFMA.  In L1, gate's last
// MFMA (kPinB1) leaves kPinB0 free -> issue up's B[0] there so up's prologue
// wait_lgkmcnt<0> finds it already landed (its latency hidden by gate's tail).
#ifndef MEGA_MOE_XOP_B0_PREFETCH
#define MEGA_MOE_XOP_B0_PREFETCH 0
#endif

// PROBE (default 0): turbo-tile k-loop -- a 2-stage cur/next LDS double buffer
// (kNumStages=2) with shifted-LDG (issue k+2's tile back into the just-read
// `cur` slot) + partial wait_vmcnt to keep k+1 in flight, mirroring
// GEMM_Tile_MXFP8_NT_256x256x128's pipeline.  Keeps FP8xFP4 + Is2B + the
// persistent megakernel; replaces the >=4-stage full-drain loop.  When on,
// jit_launch.cu forces KNUMSTAGES=2.
#ifndef MEGA_MOE_TURBO_PIPE
#define MEGA_MOE_TURBO_PIPE 0
#endif

enum class MegaMoEArch : uint32_t {
    Unknown = 0,
    Gfx942  = 942,
    Gfx950  = 950,
};

// ── Optional in-kernel per-stage profiler (-DMEGA_MOE_PROFILE=1) ──────────────
// Each stage's wall-clock span (s_memrealtime ticks) is accumulated into device
// globals by thread 0 of every block; the host reads them back and reports
// avg = acc/cnt per stage.  Zero overhead when the macro is off.
#if defined(MEGA_MOE_PROFILE) && (MEGA_MOE_PROFILE + 0)
namespace prof {
inline constexpr int kNumStages = 8; // see kStage* below
enum Stage {
    kStageDispatchPre = 0, // routing: count + topk route + cross-rank recv_count + barriers
    kStageDispatchPull,    // cross-rank token pull (warp_copy + SF)
    kStageL1Mma,           // Linear1 grouped-GEMM k-loop (gate||up)
    kStageSwiGLU,          // Linear1 epilogue: SwiGLU + FP8 requant
    kStageL2Mma,           // Linear2 grouped-GEMM k-loop
    kStageL2Epi,           // Linear2 epilogue: BF16 write to combine buffer
    kStageCombine,         // top-k reduce into y
    kStageTotal,           // whole-kernel span
};
__device__ unsigned long long                 g_acc[kNumStages];
__device__ unsigned long long                 g_cnt[kNumStages];
__device__ __forceinline__ unsigned long long clk() {
    return __builtin_amdgcn_s_memrealtime();
}
__device__ __forceinline__ void accumulate(int slot, unsigned long long span, uint32_t thread_idx) {
    if (thread_idx == 0u) {
        atomicAdd(&g_acc[slot], span);
        atomicAdd(&g_cnt[slot], 1ull);
    }
}
} // namespace prof
#define MEGA_PROF_T(var) unsigned long long var = ::primus_turbo::mega_moe::prof::clk()
#define MEGA_PROF_ADD(slot, t0)                                                                    \
    ::primus_turbo::mega_moe::prof::accumulate(::primus_turbo::mega_moe::prof::slot,               \
                                               ::primus_turbo::mega_moe::prof::clk() - (t0),       \
                                               thread_idx)
#else
#define MEGA_PROF_T(var)
#define MEGA_PROF_ADD(slot, t0)
#endif

// Cross-CU / cross-rank rendezvous.  Block-local sync is a plain
// ``__syncthreads()`` (every barrier spans the whole block); ``grid_sync`` adds
// the cross-CU handshake and ``nvlink_barrier`` the cross-rank signal.
namespace comm {

// Grid-wide sync: atomic count + relaxed spin + a single acquire fence.
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0>
__device__ __forceinline__ void grid_sync(const layout::Workspace &workspace,
                                          uint32_t leader_thread_idx, uint32_t sm_idx,
                                          uint32_t thread_idx) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    __syncthreads();
    if (thread_idx == leader_thread_idx) {
        auto          *count_ptr = workspace.get_grid_sync_count_ptr(kGridSyncIndex);
        const uint32_t old_value =
            prims::atomic_add_rel(count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1u)) : 1u);
        uint32_t new_value;
        // Cheap-fence-ONCE spin: relaxed ld_volatile loop (atomic loads bypass
        // L1 so the producer's released count write is observed), then a SINGLE
        // acquire fence (one cache invalidate, not one per iteration).
        do {
            new_value = prims::ld_volatile(count_ptr);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0u);
        prims::acquire_fence_agent();
    }
    __syncthreads();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kGridSyncIndex, uint32_t kTag>
__device__ __forceinline__ void
nvlink_barrier(const layout::Workspace &workspace, const layout::SymBuffer<kNumRanks> &sym_buffer,
               uint32_t leader_thread_idx, uint32_t sm_idx, uint32_t thread_idx,
               bool sync_prologue = true, bool sync_epilogue = true) {
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, leader_thread_idx, sm_idx, thread_idx);

    if constexpr (kNumRanks > 1) {
        if (sm_idx == 0) {
            auto *counter_ptr = workspace.get_nvl_barrier_counter_ptr();

            const auto status       = prims::ld_volatile(counter_ptr) & 3u;
            const auto signal_phase = status & 1u;
            const auto signal_sign  = status >> 1u;
            auto      *signal_ptr   = workspace.get_nvl_barrier_signal_ptr(signal_phase);

            if (thread_idx >= leader_thread_idx && thread_idx < leader_thread_idx + kNumRanks)
                prims::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx - leader_thread_idx),
                                       signal_sign ? -1 : 1);
            // sm_idx==0 is uniform across the block, so this __syncthreads() is
            // reached by all threads of the block (all-or-none).
            __syncthreads();

            if (thread_idx == leader_thread_idx) {
                prims::red_add(reinterpret_cast<int *>(counter_ptr), 1);
                const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
                while (prims::ld_acquire_sys(signal_ptr) != target)
                    __builtin_amdgcn_s_sleep(1);
            }
        }
    } else {
        (void) sym_buffer;
    }

    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, leader_thread_idx, sm_idx, thread_idx);
}

} // namespace comm

template <typename Adtype, typename Bdtype>
__device__ __forceinline__ dtype::float32x4 mfma_scaled(dtype::int32x8 a, dtype::int32x8 b,
                                                        dtype::float32x4 c, uint32_t scale_a,
                                                        uint32_t scale_b) {
    return device::mfma_scale_f32_16x16x128_f8f6f4<Adtype, Bdtype>::run(a, b, c, scale_a, scale_b);
}

// ── Pinned-AGPR (turbo-style) scaffolding ──────────────────────────────────
// Compile-time unroll so run_pinned_acc_agpr / read_agpr receive constexpr
// register indices ("n" immediates).  kSubTilesM/N are constexpr but the
// #pragma-unroll loop variables are not usable as template args, so the pinned
// burst and the epilogue AGPR read-back drive the (sub_m,sub_n) index through
// this instead.
template <int I, int N, typename F> __device__ __forceinline__ void mega_static_for(F &&f) {
    if constexpr (I < N) {
        f.template operator()<I>();
        mega_static_for<I + 1, N>(static_cast<F &&>(f));
    }
}
// Pinned VGPR operand window (top of the 256-VGPR file) for the Linear2 burst:
// A frag = 8 VGPR (FP8 e4m3, cbsz=0), B frag = 4 VGPR (FP4 e2m1, blgp=4), one
// scale VGPR each.  Reserved via reserve_vgpr_range so the compiler will not
// reuse them between the set_vgpr stage and the pinned MFMA.
inline constexpr int kPinA0  = 224; // v[224:231] A frag 0 (8)
inline constexpr int kPinA1  = 232; // v[232:239] A frag 1 (8)
inline constexpr int kPinB0  = 240; // v[240:243] B frag buf0 (4)
inline constexpr int kPinB1  = 244; // v[244:247] B frag buf1 (4) -- double buffer
inline constexpr int kPinSA0 = 248; // v[248] A0 scale
inline constexpr int kPinSA1 = 249; // v[249] A1 scale
inline constexpr int kPinSB0 = 250; // v[250] B scale buf0
inline constexpr int kPinSB1 = 251; // v[251] B scale buf1
// AGPR accumulator bases (per loader wave, 16 subtiles x 4 = 64 AGPR each):
//   up   acc -> a[0:63]   (Linear2 uses only this range)
//   gate acc -> a[64:127] (Linear1 only)
inline constexpr int kAccUp   = 0;
inline constexpr int kAccGate = 64;
// Read ONE subtile's float32x4 accumulator from a compile-time AGPR BASE + a
// RUNTIME subtile index (0..15), via a switch so only ONE v_accvgpr_read runs
// (~4 VGPR live).  Avoids hoisting all 16 subtiles (128 VGPR) at the epilogue
// top, which kept the kernel pinned at 256 VGPR with no room for the pin window.
// Assumes kSubTilesPerWave == 16 (the 4-warp AGPR config).
template <int BASE> __device__ __forceinline__ dtype::float32x4 read_acc_sub16(uint32_t sub) {
    switch (sub) {
    case 0:
        return device::read_agpr<dtype::float32x4, BASE + 0>();
    case 1:
        return device::read_agpr<dtype::float32x4, BASE + 4>();
    case 2:
        return device::read_agpr<dtype::float32x4, BASE + 8>();
    case 3:
        return device::read_agpr<dtype::float32x4, BASE + 12>();
    case 4:
        return device::read_agpr<dtype::float32x4, BASE + 16>();
    case 5:
        return device::read_agpr<dtype::float32x4, BASE + 20>();
    case 6:
        return device::read_agpr<dtype::float32x4, BASE + 24>();
    case 7:
        return device::read_agpr<dtype::float32x4, BASE + 28>();
    case 8:
        return device::read_agpr<dtype::float32x4, BASE + 32>();
    case 9:
        return device::read_agpr<dtype::float32x4, BASE + 36>();
    case 10:
        return device::read_agpr<dtype::float32x4, BASE + 40>();
    case 11:
        return device::read_agpr<dtype::float32x4, BASE + 44>();
    case 12:
        return device::read_agpr<dtype::float32x4, BASE + 48>();
    case 13:
        return device::read_agpr<dtype::float32x4, BASE + 52>();
    case 14:
        return device::read_agpr<dtype::float32x4, BASE + 56>();
    default:
        return device::read_agpr<dtype::float32x4, BASE + 60>();
    }
}

template <uint32_t kNumMaxTokensPerRank, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kNumTopk, uint32_t kNumExpertsPerWave, uint32_t BLOCK_M,
          uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t STORE_BLOCK_M, uint32_t SF_BLOCK_M,
          uint32_t SF_BLOCK_N, uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens,
          uint32_t kNumStages, uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads, uint32_t kNumSMs, uint32_t kNumRanks,
          float kActivationClamp, bool kFastMath, uint32_t L1_SHAPE_N = kIntermediateHidden * 2u,
          uint32_t L1_SHAPE_K = kHidden, uint32_t L2_SHAPE_N = kHidden,
          uint32_t L2_SHAPE_K  = kIntermediateHidden,
          uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
          uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks>
__global__ __launch_bounds__(kNumThreads, 1) void gfx950_fp8_fp4_mega_moe_kernel(
    void *y, int *cumulative_local_expert_recv_stats, const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, void *l1_weights,
    void *l1_weights_sf, void *l2_weights, void *l2_weights_sf) {
#if defined(__gfx950__)
    using namespace prims;

    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = get_warp_idx();
    const uint32_t lane_idx   = get_lane_idx();

    constexpr uint32_t kNumDispatchWarps       = kNumDispatchThreads / kWarpSize;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / kWarpSize;
    // Epilogue role folded into the MFMA warps (combine runs there post-compute),
    // so kNumEpilogueThreads is 0 in the live config; no separate warp count.

    // Token and buffer layouts
    // NOTES: activations are FP8 (1 B/elem); the L2 activation is the SwiGLU
    // intermediate, also FP8
    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(__hip_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32u);
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 32u);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    constexpr uint64_t kInterBufAlign = 256u;
    auto               align256       = [](void *p) -> void                     *{
        const auto v = reinterpret_cast<uintptr_t>(p);
        return reinterpret_cast<void *>((v + kInterBufAlign - 1u) &
                                                            ~uintptr_t(kInterBufAlign - 1u));
    };

    const auto workspace = layout::Workspace(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
                                             kNumMaxTokensPerRank, kNumTopk);

    const auto input_token_buffer    = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank,
                                                      align256(workspace.get_end_ptr()));
    const auto input_sf_buffer       = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank,
                                                      align256(input_token_buffer.get_end_ptr()));
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank, align256(input_sf_buffer.get_end_ptr()));
    const auto input_topk_weights_buffer =
        layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank,
                       align256(input_topk_idx_buffer.get_end_ptr()));

    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens,
                                                       align256(input_topk_weights_buffer.get_end_ptr()));
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
                                                       align256(l1_token_buffer.get_end_ptr()));
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens,
                                                       align256(l1_sf_buffer.get_end_ptr()));

    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
                                                align256(l1_topk_weights_buffer.get_end_ptr()));
    const auto l2_sf_buffer = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
                                             align256(l2_token_buffer.get_end_ptr()));
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, align256(l2_sf_buffer.get_end_ptr()));

    constexpr uint32_t kGranK                 = 32u;
    constexpr uint32_t kNumUTCCPAlignedElems  = 128u;
    auto               transform_sf_token_idx = [](const uint32_t &token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M + (idx & ~127u) + (idx & 31u) * 4u +
               ((idx >> 5) & 3u);
    };

    // MMA tile shape constraints.  Activations are FP8 (e4m3, 1 B/elem),
    // weights are FP4 (e2m1, packed 2/byte); the K-major 16x16x128 MFMA below
    // depends on these.
    constexpr uint32_t LAYOUT_AD_M = 128u;
    static_assert(BLOCK_M % 16u == 0u, "Invalid block M");
    static_assert(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    static_assert(BLOCK_K == 128u, "Invalid block K");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_expert_count = reinterpret_cast<uint32_t *>(smem_buffer);

    constexpr uint32_t kMfmaM       = 16u;
    constexpr uint32_t kMfmaN       = 16u;
    constexpr uint32_t kMfmaK       = 128u;
    constexpr uint32_t kInnerKIters = BLOCK_K / kMfmaK;
    static_assert(BLOCK_K % kMfmaK == 0u, "BLOCK_K must be a multiple of MFMA K");

    static_assert(BLOCK_M % kNumMMANonEpilogueWarps == 0u,
                  "BLOCK_M must be evenly partitionable across loader waves");
    constexpr uint32_t kRowsPerLoaderWave = BLOCK_M / kNumMMANonEpilogueWarps;
    constexpr uint32_t kColsPerLoaderWave = BLOCK_N;
    static_assert(kRowsPerLoaderWave % kMfmaM == 0u,
                  "loader wave row partition must be a multiple of MFMA M");
    static_assert(kColsPerLoaderWave % kMfmaN == 0u,
                  "loader wave col partition must be a multiple of MFMA N");
    constexpr uint32_t kSubTilesM       = kRowsPerLoaderWave / kMfmaM;
    constexpr uint32_t kSubTilesN       = kColsPerLoaderWave / kMfmaN;
    constexpr uint32_t kSubTilesPerWave = kSubTilesM * kSubTilesN;

    constexpr uint32_t kBPackBytesPerRow = BLOCK_K / 2u;
    constexpr uint32_t kATileBytes       = kRowsPerLoaderWave * BLOCK_K;
    constexpr uint32_t kBTileBytes       = kColsPerLoaderWave * kBPackBytesPerRow;
    constexpr uint32_t kLaneLoadBytes    = 16u;
    constexpr uint32_t kBytesPerLoadCall = kWarpSize * kLaneLoadBytes;
    static_assert(kATileBytes % kBytesPerLoadCall == 0u,
                  "A tile must be a multiple of one cooperative load call");
    static_assert(kBTileBytes % kBytesPerLoadCall == 0u,
                  "B tile must be a multiple of one cooperative load call");
    constexpr uint32_t kATileLoadsPerWave = kATileBytes / kBytesPerLoadCall;
    constexpr uint32_t kBTileLoadsPerWave = kBTileBytes / kBytesPerLoadCall;

    // Scale factors are staged into the loader's LDS shadow: SFA per-warp (one
    // UE8M0 dword per loader-wave row) rides each stage's A SF region; SFB shared
    // (one dword per BLOCK_N column) gets a pool after the B pool.
    // NOTES: L1 fuses gate+up into one A x two-B MFMA stream (CK Is2B), so the B
    // and SFB pools hold kMaxBOperands operands; Linear2 uses only operand 0
    constexpr uint32_t kMaxBOperands    = 2u;
    constexpr uint32_t kSFAStageBytes   = kRowsPerLoaderWave * sizeof(uint32_t);
    constexpr uint32_t kSFBStageBytes   = kMaxBOperands * BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t kLoaderASFBytes  = kSFAStageBytes;
    constexpr uint32_t kLoaderSinkBytes = kSubTilesPerWave * sizeof(dtype::float32x4);

    constexpr uint32_t kStagedABytesPerStage = kATileBytes + kLoaderASFBytes;
    constexpr uint32_t kStagedBBytesPerStage = kMaxBOperands * kBTileBytes;
    constexpr uint32_t kLoaderBaseBytes =
        ((sizeof(uint32_t) * kNumExperts + 16u + 1023u) / 1024u) * 1024u;

    static constexpr uint32_t kSmemBytesBudget = 140u * 1024u;
    static_assert(kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderSinkBytes < kSmemBytesBudget,
                  "loader sink + base already exceed kSmemBytes budget");
    // Per-stage LDS cost: per-warp A (+SFA) tiles, shared B tile, and the
    // shared SFB pool slot.
    constexpr uint32_t kPerStageLdsBytes =
        kNumMMANonEpilogueWarps * kStagedABytesPerStage + kStagedBBytesPerStage + kSFBStageBytes;
    constexpr uint32_t kMaxLoaderStagesByLds =
        (kSmemBytesBudget - kLoaderBaseBytes - kNumMMANonEpilogueWarps * kLoaderSinkBytes) /
        kPerStageLdsBytes;
    static_assert(kMaxLoaderStagesByLds >= 1u,
                  "loader staged carve-out too large for one stage even - shrink BLOCK_K?");
    constexpr uint32_t kLoaderStages =
        (kNumStages < kMaxLoaderStagesByLds) ? kNumStages : kMaxLoaderStagesByLds;

    constexpr uint32_t kLoaderAWaveBytes = kLoaderStages * kStagedABytesPerStage + kLoaderSinkBytes;
    constexpr uint32_t kLoaderBPoolBaseBytes =
        kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderAWaveBytes;
    constexpr uint32_t kLoaderBPoolBytes = kLoaderStages * kStagedBBytesPerStage;
    static_assert(kLoaderBPoolBaseBytes + kLoaderBPoolBytes <= kSmemBytesBudget,
                  "loader tile carve-out (per-warp A + shared B) exceeds kSmemBytes budget");

    // Shared SFB pool: kLoaderStages slots of kSFBStageBytes, appended after the
    // shared-B pool.  SFA lives inside each per-warp A stage (kATileBytes..).
    constexpr uint32_t kLoaderSFBPoolBaseBytes = kLoaderBPoolBaseBytes + kLoaderBPoolBytes;
    constexpr uint32_t kLoaderSFBPoolBytes     = kLoaderStages * kSFBStageBytes;
    static_assert(kLoaderSFBPoolBaseBytes + kLoaderSFBPoolBytes <= kSmemBytesBudget,
                  "shared SFB pool exceeds kSmemBytes budget");

    constexpr uint32_t kLoadsPerStage = kATileLoadsPerWave + kMaxBOperands * kBTileLoadsPerWave;

    static_assert(kLoadsPerStage < 63u,
                  "kLoadsPerStage exceeds vmcnt range - reduce per-stage loads");

    if (num_tokens == 0u) {
        (void) l1_weights;
        (void) l1_weights_sf;
        (void) l2_weights;
        (void) l2_weights_sf;
        (void) sym_buffer;
        (void) cumulative_local_expert_recv_stats;
        (void) y;
        return;
    }

    // Grid-sync / NVLink-barrier leader threads (combine is folded onto the MMA
    // warps, so its barriers are led by the first MMA thread)
    constexpr uint32_t kDispLeader = 0u;
    constexpr uint32_t kEpiLeader  = kNumDispatchWarps * kWarpSize;

    if (warp_idx == 0u && elect_one()) {
#pragma unroll
        for (uint32_t i = 0; i < kNumExperts; ++i)
            smem_expert_count[i] = 0u;
    }
    __syncthreads();

    auto scheduler = sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K,
                                             L2_SHAPE_N, L2_SHAPE_K, kNumExpertsPerRank,
                                             kNumExpertsPerWave, kNumSMs, kNumRanks>(workspace);

    MEGA_PROF_T(t_total); // whole-kernel span
    MEGA_PROF_T(t_pre);   // dispatch preprocess span

    // Warp roles: all warps run dispatch (Phase 1), then the same warps run the
    // MMA loader + compute + epilogue + combine (Phase 2).  The SM100 dedicated
    // dispatch / TMA-load / epilogue warp roles are folded together here because
    // there is no TMA and combine reuses the MMA warps (kNumDispatchThreads ==
    // kNumEpilogueThreads == 0).

    // Dispatch: count + route the topk indices, then pull each owned token from
    // its remote rank into the local L1 pool.  All warps participate.
    {
        constexpr uint32_t kDispWarps   = kNumMMANonEpilogueWarps;
        constexpr uint32_t kDispThreads = kNumThreads;

        constexpr uint32_t kNumTokensPerWarp = kWarpSize / kNumTopk;
        constexpr uint32_t kNumGlobalWarps   = kNumSMs * kDispWarps;
        static_assert(kNumTokensPerWarp * kNumTopk <= kWarpSize,
                      "kNumTopk does not divide wave size");

        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto         read_topk_idx     = [&](const auto &process) {

#pragma unroll
            for (uint32_t i = (sm_idx * kDispWarps + warp_idx) * kNumTokensPerWarp; i < num_tokens;
                 i += kNumSMs * kDispWarps * kNumTokensPerWarp) {

                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(__ldg(
                        input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        read_topk_idx([&](const uint32_t &token_topk_idx, const int &expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        __syncthreads();

        for (uint32_t i = thread_idx; i < kNumExperts; i += kDispThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i]      = static_cast<uint32_t>(
                atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        __syncthreads();

        for (uint32_t i = (sm_idx * kDispWarps + warp_idx) * kNumTokensPerWarp; i < num_tokens;
             i += kNumGlobalWarps * kNumTokensPerWarp) {
            const uint32_t slot = i + (lane_idx / kNumTopk);
            if (slot < num_tokens && lane_idx < kNumTokensPerWarp * kNumTopk) {
                const uint32_t topk_idx_in_token = lane_idx % kNumTopk;
                const int e = static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                                     slot * kNumTopk + topk_idx_in_token));
                if (e >= 0) {
                    const uint32_t dst_rank_idx = e / kNumExpertsPerRank;
                    const uint32_t dst_slot_idx = atomic_add_block(smem_expert_count + e, 1u);
                    auto          *dst_ptr      = workspace.get_src_token_topk_idx_ptr(
                        e % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = slot * kNumTopk + topk_idx_in_token;
                }
            }
        }

        comm::grid_sync<kNumSMs, 0>(workspace, kDispLeader, sm_idx, thread_idx);

        if (sm_idx == 0u) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += kDispThreads) {
                const uint32_t dst_rank         = i / kNumExpertsPerRank;
                const uint32_t dst_local_expert = i % kNumExpertsPerRank;
                const uint64_t expert_status    = *workspace.get_expert_send_count_ptr(i);

                // Cross-rank recv_count write.  RELAXED, SYSTEM scope: the
                // happens-before for the pull loop's read comes from the
                // nvlink_barrier below, not this store; SYSTEM scope is required
                // so the write reaches the remote agent.
                __hip_atomic_store(
                    reinterpret_cast<unsigned int *>(sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert),
                        dst_rank)),
                    static_cast<unsigned int>(expert_status & 0xffffffffull), __ATOMIC_RELAXED,
                    __HIP_MEMORY_SCOPE_SYSTEM);
                atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert),
                                   dst_rank),
                    expert_status);
            }
        }
        __syncthreads();

        comm::nvlink_barrier<kNumRanks, kNumSMs, 0, 1>(workspace, sym_buffer, kDispLeader, sm_idx,
                                                       thread_idx, false, true);

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = (kNumRanks + kWarpSize - 1u) / kWarpSize;

        int      current_expert_idx                  = -1;
        uint32_t expert_start_idx                    = 0u;
        uint32_t expert_end_idx                      = 0u;
        uint32_t expert_pool_block_offset            = 0u;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};

        // One token's pull, extracted so the PINGPONG path can interleave it with
        // GEMM blocks (dispatch<->mma role-switch).  Returns false once this warp's
        // expert stream is exhausted.  Loop-carried expert state is captured above.
        auto pull_one_token = [&](uint32_t token_idx) -> bool {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= static_cast<int>(kNumExpertsPerRank))
                    break;
                expert_pool_block_offset +=
                    (expert_end_idx - expert_start_idx + BLOCK_M - 1u) / BLOCK_M;
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= static_cast<int>(kNumExpertsPerRank))
                return false;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    const uint32_t j = i * kWarpSize + lane_idx;
                    if (j < kNumRanks) {
                        const auto raw = __hip_atomic_load(
                            reinterpret_cast<const unsigned long long *>(
                                workspace.get_expert_recv_count_ptr(
                                    j, static_cast<uint32_t>(current_expert_idx))),
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                        stored_rank_count[i] = static_cast<uint32_t>(raw);
                    } else {
                        stored_rank_count[i] = 0u;
                    }
                }
            }

            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t       remaining[kNumRanksPerLane];
#pragma unroll
            for (uint32_t i = 0u; i < kNumRanksPerLane; ++i)
                remaining[i] = stored_rank_count[i];

            uint32_t current_rank_in_expert_idx = 0u;
            uint32_t token_idx_in_rank          = 0u;
            uint32_t slot_idx                   = token_idx_in_expert;
            uint32_t offset                     = 0u;
            while (true) {

                uint32_t num_actives_in_lane = 0u;
                uint32_t min_in_lane         = 0xffffffffu;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    if (remaining[i] > 0u) {
                        ++num_actives_in_lane;
                        if (remaining[i] < min_in_lane)
                            min_in_lane = remaining[i];
                    }
                }
                const uint32_t num_active_ranks = reduce_add(num_actives_in_lane);
                const uint32_t length           = reduce_min(min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {

                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t       num_seen_ranks    = 0u;
#pragma unroll
                    for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                        const uint64_t mask             = ballot(remaining[i] > 0u);
                        const uint32_t num_active_lanes = popcnt(mask);
                        if (slot_idx_in_round >= num_seen_ranks &&
                            slot_idx_in_round < num_seen_ranks + num_active_lanes) {
                            current_rank_in_expert_idx =
                                i * kWarpSize +
                                nth_set_bit(mask, slot_idx_in_round - num_seen_ranks + 1u);
                        }
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                slot_idx -= num_round_tokens;
                offset += length;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    if (remaining[i] >= length)
                        remaining[i] -= length;
                    else
                        remaining[i] = 0u;
                }
            }

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                static_cast<uint32_t>(current_expert_idx), current_rank_in_expert_idx,
                token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            const uint32_t pool_token_idx =
                expert_pool_block_offset * BLOCK_M + token_idx_in_expert;

            auto *src_token_ptr = sym_buffer.map(
                input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint8_t>(),
                current_rank_in_expert_idx);
            auto *dst_token_ptr =
                l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr<uint8_t>();
            // Pull the dispatched token (remote-global -> local-pool global).
            // NOTES: no TMA on AMD -- a cooperative unrolled warp copy streams the
            // kHidden FP8 bytes directly (mirrors UNROLLED_WARP_COPY in deep_ep)
            static_assert(kHidden % 16u == 0u, "token bytes must be int4-aligned");
            // kHidden/16/64 = 7 int4/lane; unroll 8 puts all in flight in ONE
            // batch -> max XGMI latency hiding (dispatch phase has VGPR headroom).
            warp_copy_int4<5u>(dst_token_ptr, src_token_ptr, kHidden / 16u, lane_idx);

            constexpr uint32_t kNumSFUint32  = kHidden / 128u;
            const auto         remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            auto      *local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            const auto sf_pool_token_idx =
                expert_pool_block_offset * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert);
#pragma unroll
            for (uint32_t i = 0u; i < (kNumSFUint32 + 31u) / 32u; ++i) {
                const uint32_t j = i * 32u + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }

            if (elect_one()) {
                const float weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() =
                    weight;

                *workspace.get_token_src_metadata_ptr(pool_token_idx) = {
                    current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Relaxed increment: the per-token XGMI store drain is replaced by
                // ONE phase-boundary release-fence + grid_sync below.
                red_add(reinterpret_cast<int *>(workspace.get_l1_arrival_count_ptr(
                            expert_pool_block_offset + token_idx_in_expert / BLOCK_M)),
                        1);
            }
            return true;
        }; // pull_one_token

        MEGA_PROF_ADD(kStageDispatchPre, t_pre); // end preprocess
        MEGA_PROF_T(t_pull);                     // begin token pull
        for (uint32_t token_idx = sm_idx * kDispWarps + warp_idx;; token_idx += kNumGlobalWarps)
            if (!pull_one_token(token_idx))
                break;
        MEGA_PROF_ADD(kStageDispatchPull, t_pull);

        // NOTES: no cross-role rendezvous before compute -- the same warps run
        // both phases, and the per-block l1_arrival_count spin in the compute
        // loop provides the cross-SM producer->consumer ordering.  recv-stats and
        // the next-launch counter reset are done in the combine tail below.
    }

    // One drain of every wave's relaxed dispatch stores into L2, then a grid-wide
    // barrier so ALL dispatch writes are globally visible before ANY compute read
    // (replaces the per-token release ordering above).
    prims::release_fence_agent();
    comm::grid_sync<kNumSMs, 0>(workspace, 0u, sm_idx, thread_idx);

    // Compute: persistently schedule over blocks.  Per block, wait the producer
    // arrival (signaled by dispatch above, possibly on another SM), cooperatively
    // load A/B into staged LDS, run the overlapped MFMA k-loop, then the epilogue
    // (L1: SwiGLU + FP8 requant into the L2 pool; L2: BF16 write into the remote
    // combine buffer).
    if (warp_idx >= kNumDispatchWarps && warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {

        const uint32_t loader_warp_local = warp_idx - kNumDispatchWarps;

        const uint32_t wave_base_byte = kLoaderBaseBytes + loader_warp_local * kLoaderAWaveBytes;

        const uint32_t a_lds_stage0 =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_buffer + wave_base_byte));
        const uint32_t b_lds_stage0 =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_buffer + kLoaderBPoolBaseBytes));

        // Pinned-AGPR accumulators: up -> a[0:63], gate -> a[64:127], resident in
        // the AGPR file across the whole L1 k-loop (Is2B: one A read drives both),
        // read back per-subtile in the epilogue via read_acc_sub16.  Reserve both
        // AGPR ranges + the operand-staging VGPR window (v[224:251]) so the compiler
        // keeps the pin window free across the k-loop + pinned MFMA.
        static_assert(kSubTilesPerWave == 16u, "4-warp turbo config assumes kSubTilesPerWave==16");
        device::reserve_agpr_range<0, 2 * static_cast<int>(kSubTilesPerWave) * 4 - 1>();
        device::reserve_vgpr_range<kPinA0, kPinSB1>();

        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
            // Terminal sentinel: nothing to do for the non-pipelined path.
            if (phase == sched::BlockPhase::None)
                return;
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            // Cross-SM producer-arrival wait.  All kNumMMANonEpilogueWarps loader
            // warps cooperate on the SAME block and wait on the SAME per-block
            // arrival counter, so having all 512 threads spin on one global
            // address is pure L2 contention.  Elect ONE thread to poll; its single
            // acquire fence invalidates the CU-wide vector L1, and the
            // __syncthreads broadcasts the release ordering to the whole block.
            // Wrapped in a lambda so DEFER_EPI can move it AFTER the pending-epilogue
            // flush (an L2 arrival-wait must see the deferred L1 signals first).
            auto do_arrival_wait = [&]() {
                if (thread_idx == 0u) {
                    if (phase == sched::BlockPhase::Linear1) {
                        const auto    *ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                        const uint32_t expected = scheduler.template get_valid_m<false>();
                        while (ld_volatile(ptr) != expected)
                            __builtin_amdgcn_s_sleep(1);
                    } else {
                        const auto        *ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        constexpr uint32_t kL1GateNBlocksWait = (L1_SHAPE_N / BLOCK_N) / 2u;
                        const uint64_t expected = static_cast<uint64_t>(kNumMMANonEpilogueWarps) *
                                                  static_cast<uint64_t>(kL1GateNBlocksWait);
                        while (ld_volatile(ptr) != expected)
                            __builtin_amdgcn_s_sleep(1);
                    }
                    acquire_fence_agent();
                }
                __syncthreads();
            };
            do_arrival_wait();

            // Build only the ACTIVE A/B data SRD for this block's phase (instead of
            // holding all four L1/L2 SRDs live across the whole kernel) -> ~8 fewer
            // SGPRs in flight, less spill.
            device::BufferSRD srd_a((phase == sched::BlockPhase::Linear1)
                                        ? l1_token_buffer.get_base_ptr<void>()
                                        : l2_token_buffer.get_base_ptr<void>());
            device::BufferSRD srd_b((phase == sched::BlockPhase::Linear1) ? l1_weights
                                                                          : l2_weights);

            // A-row byte stride: FP8 full-width (1 B/elem) for both phases.
            const uint32_t a_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? kHidden : kIntermediateHidden;
            const uint32_t b_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? (L1_SHAPE_K / 2u) : (L2_SHAPE_K / 2u);
            const uint32_t b_expert_stride_bytes = (phase == sched::BlockPhase::Linear1)
                                                       ? (L1_SHAPE_N * L1_SHAPE_K) / 2u
                                                       : (L2_SHAPE_N * L2_SHAPE_K) / 2u;

            const uint32_t a_wave_row0 =
                pool_block_idx * BLOCK_M + loader_warp_local * kRowsPerLoaderWave;
            const uint32_t a_tile_base_bytes = a_wave_row0 * a_row_stride_bytes;

            uint32_t b_tile_base_bytes = local_expert_idx * b_expert_stride_bytes +
                                         (n_block_idx * BLOCK_N) * b_row_stride_bytes;

            constexpr uint32_t kScaleOne = 0x7f7f7f7fu;

            const uint32_t sfa_pool_token_idx_base = pool_block_idx * SF_BLOCK_M;
            const uint32_t sfb_expert_stride_bytes = (phase == sched::BlockPhase::Linear1)
                                                         ? (L1_SHAPE_N * L1_SHAPE_K) / kGranK
                                                         : (L2_SHAPE_N * L2_SHAPE_K) / kGranK;

            uint32_t sfb_n_global_base = n_block_idx * BLOCK_N;

            // SRDs for the SF pools (buffer_load_lds path); built once per block,
            // base ptr inlined so the phase-selected pointer isn't a separate live.
            device::BufferSRD srd_sfa(
                (phase == sched::BlockPhase::Linear1)
                    ? reinterpret_cast<const void *>(l1_sf_buffer.get_base_ptr<uint32_t>())
                    : reinterpret_cast<const void *>(l2_sf_buffer.get_base_ptr<uint32_t>()));
            device::BufferSRD srd_sfb((phase == sched::BlockPhase::Linear1)
                                          ? reinterpret_cast<const void *>(l1_weights_sf)
                                          : reinterpret_cast<const void *>(l2_weights_sf));

            // Stage A and B tiles for one (stage, k_block): direct global->LDS
            // (buffer_load to LDS, vmcnt-only, no ds_write), kept separate from SF
            // so the prefetch can stay in flight across the wait below.
            // turbo-style interleave: `half` selects a slice of the loads (0/1)
            // so the burst can drip them between MFMA operands; 2u = issue all.
            // turbo-style interleave: `half` selects a slice of the loads (0/1) so
            // the burst can drip them between MFMA operands; 2u = issue all.
            const auto issue_ab = [&](uint32_t stage, uint32_t k_block, uint32_t half = 2u) {
                const uint32_t b_k_offset_bytes = k_block * (BLOCK_K / 2u);
                const uint32_t a_lds            = a_lds_stage0 + stage * kStagedABytesPerStage;
                const uint32_t b_lds            = b_lds_stage0 + stage * kStagedBBytesPerStage;
                {
                    const uint32_t a_k_offset_bytes = k_block * BLOCK_K;
#pragma unroll
                    for (uint32_t c = 0u; c < kATileLoadsPerWave; ++c) {
                        if (half != 2u && (c & 1u) != half)
                            continue;
                        const uint32_t m_lane          = lane_idx / 8u;
                        const uint32_t m_in_wave       = c * 8u + m_lane;
                        const uint32_t k_chunk_in_lane = (lane_idx % 8u) ^ (m_lane & 7u);
                        const uint32_t k_byte_in_tile  = k_chunk_in_lane * kLaneLoadBytes;
                        // turbo soffset-split: all warp-uniform address terms ride the
                        // scalar soffset (readfirstlane -> SGPR); only the per-lane
                        // part stays in the VGPR voffset, freeing VGPRs.
                        const int32_t  a_soff = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(
                            a_tile_base_bytes + a_k_offset_bytes + c * 8u * a_row_stride_bytes));
                        const uint32_t ldg_offset = m_lane * a_row_stride_bytes + k_byte_in_tile;
                        const uint32_t lds_offset =
                            device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                0u, m_in_wave, k_byte_in_tile);
                        device::load_gmem_to_smem_srd<16, 0>(srd_a, ldg_offset, a_lds + lds_offset,
                                                             a_soff);
                    }
                }
                // Balanced shared-B load: loader warps round-robin the B chunks
                // (warp w loads chunk c where c % nwarps == w) into the shared-B
                // LDS slot, published to all warps by the __syncthreads() below.
                // NOTES: Is2B stages both gate-B (operand 0) and up-B (operand 1,
                // shifted kIntermediateHidden columns) so one A read drives both
                {
                    const uint32_t num_b_ops =
                        (phase == sched::BlockPhase::Linear1) ? kMaxBOperands : 1u;
                    for (uint32_t b_op = 0u; b_op < num_b_ops; ++b_op) {
                        const uint32_t b_op_base =
                            b_tile_base_bytes + b_op * kIntermediateHidden * b_row_stride_bytes;
                        const uint32_t b_op_lds = b_lds + b_op * kBTileBytes;
#pragma unroll
                        for (uint32_t c = 0u; c < kBTileLoadsPerWave; ++c) {
                            if (c % kNumMMANonEpilogueWarps != loader_warp_local)
                                continue;
                            if (half != 2u && (c & 1u) != half)
                                continue;
                            const uint32_t n_lane          = lane_idx / 4u;
                            const uint32_t n_in_wave       = c * 16u + n_lane;
                            const uint32_t k_chunk_in_lane = lane_idx % 4u;
                            const uint32_t k_byte_in_tile  = k_chunk_in_lane * kLaneLoadBytes;
                            // turbo soffset-split: warp-uniform terms -> scalar soffset.
                            const int32_t b_soff =
                                __builtin_amdgcn_readfirstlane(static_cast<int32_t>(
                                    b_op_base + b_k_offset_bytes + c * 16u * b_row_stride_bytes));
                            const uint32_t ldg_offset =
                                n_lane * b_row_stride_bytes + k_byte_in_tile;
                            const uint32_t lds_offset = device::b_tile_smem_byte_offset_rowmajor<
                                kColsPerLoaderWave, BLOCK_K / 2u>(0u, n_in_wave, k_byte_in_tile);
                            device::load_gmem_to_smem_srd<16, 0>(srd_b, ldg_offset,
                                                                 b_op_lds + lds_offset, b_soff);
                        }
                    }
                }
            }; // issue_ab

            // Stage scale factors for one (stage, k_block) via __ldg -> ds_write.
            // SFA: per-warp, one UE8M0 dword per loader-wave row (Linear1 only;
            // Linear2 uses scale 1.0).  SFB: shared, one dword per BLOCK_N column,
            // round-robined across loader warps.
            const auto issue_sf = [&](uint32_t stage, uint32_t k_block, uint32_t half = 2u) {
                {
                    if (half == 1u)
                        return; // SF is tiny: issue it all on the half-0 slice
                    if (phase == sched::BlockPhase::Linear1) {
                        uint32_t *sfa_lds = reinterpret_cast<uint32_t *>(
                            smem_buffer + wave_base_byte + stage * kStagedABytesPerStage +
                            kATileBytes);
#pragma unroll
                        for (uint32_t r = lane_idx; r < kRowsPerLoaderWave; r += kWarpSize) {
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave + r;
                            const uint32_t sf_token_idx =
                                sfa_pool_token_idx_base + transform_sf_token_idx(m_in_block);
                            // global->LDS direct (vmcnt), same dest sfa_lds[r].
                            device::load_gmem_to_smem_srd<4>(
                                srd_sfa,
                                (k_block * kNumPaddedSFPoolTokens + sf_token_idx) *
                                    static_cast<uint32_t>(sizeof(uint32_t)),
                                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sfa_lds + r)), 0);
                        }
                    }
                    uint32_t *sfb_lds = reinterpret_cast<uint32_t *>(
                        smem_buffer + kLoaderSFBPoolBaseBytes + stage * kSFBStageBytes);
                    const uint32_t num_sfb_ops =
                        (phase == sched::BlockPhase::Linear1) ? kMaxBOperands : 1u;
                    for (uint32_t b_op = 0u; b_op < num_sfb_ops; ++b_op) {
                        const uint32_t sfb_op_n_base =
                            sfb_n_global_base + b_op * kIntermediateHidden;
                        uint32_t *sfb_lds_op = sfb_lds + b_op * BLOCK_N;
                        // K-major weight-SF: [E][k_block][N] -> consecutive n
                        // (lanes) are CONTIGUOUS -> coalesced 4B load (vs the
                        // n-major scatter at stride K/32).  Offline transpose
                        // in the test matches this layout.
                        const uint32_t sfb_N =
                            (phase == sched::BlockPhase::Linear1) ? L1_SHAPE_N : L2_SHAPE_N;
#if MEGA_MOE_TURBO_PIPE
                        // Warp-uniform SFB: each warp loads BLOCK_N/nwarps
                        // contiguous cols (identical LDS result) so every wave
                        // issues the SAME vmem count -> a single compile-time
                        // wait_vmcnt drains one k_block in the 2-stage loop.
                        constexpr uint32_t kSfbColsPerWarp = BLOCK_N / kNumMMANonEpilogueWarps;
                        static_assert(BLOCK_N % kNumMMANonEpilogueWarps == 0u,
                                      "BLOCK_N must split evenly across warps for SFB");
#pragma unroll
                        for (uint32_t cc = lane_idx; cc < kSfbColsPerWarp; cc += kWarpSize) {
                            const uint32_t c        = loader_warp_local * kSfbColsPerWarp + cc;
                            const uint32_t n_global = sfb_op_n_base + c;
                            const uint32_t off      = local_expert_idx * sfb_expert_stride_bytes +
                                                 k_block * (sfb_N * 4u) + n_global * 4u;
                            device::load_gmem_to_smem_srd<4>(
                                srd_sfb, off,
                                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sfb_lds_op + c)),
                                0);
                        }
#else
#pragma unroll
                        for (uint32_t c = loader_warp_local * kWarpSize + lane_idx; c < BLOCK_N;
                             c += kNumMMANonEpilogueWarps * kWarpSize) {
                            const uint32_t n_global = sfb_op_n_base + c;
                            const uint32_t off      = local_expert_idx * sfb_expert_stride_bytes +
                                                 k_block * (sfb_N * 4u) + n_global * 4u;
                            // global->LDS direct (vmcnt), same dest sfb_lds_op[c].
                            device::load_gmem_to_smem_srd<4>(
                                srd_sfb, off,
                                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sfb_lds_op + c)),
                                0);
                        }
#endif
                    }
                }
            };

            // Run the full k-loop for the current N-tile, accumulating into
            // acc[] (the live VGPR MFMA accumulators).
            // Split into prologue (issue the first loads) + main (reset acc, MFMA
            // loop) so the block loop can run a DEFERRED epilogue between them,
            // overlapping the prologue load latency.  Default calls both back-to-back;
            // moving acc-reset after the issue is independent -> neutral.
            auto kloop_prologue = [&]() {
                if (num_k_blocks > 0u) {
                    issue_ab(0u, 0u);
                    issue_sf(0u, 0u);
                }
                if (num_k_blocks > 1u) {
                    issue_ab(1u % kLoaderStages, 1u);
                    issue_sf(1u % kLoaderStages, 1u);
                }
            };

            auto kloop_main = [&]() {
                // Zero the AGPR accumulator range(s): up a[0:63] always; gate
                // a[64:127] additionally for Linear1.  acc[]/acc_gate[] VGPR unused.
                if (phase == sched::BlockPhase::Linear1)
                    device::zero_agpr_range<
                        kAccUp, kAccGate + static_cast<int>(kSubTilesPerWave) * 4 - 1>();
                else
                    device::zero_agpr_range<kAccUp, static_cast<int>(kSubTilesPerWave) * 4 - 1>();

                // One k_block's MFMA burst: read A/SFA/B/SFB from this stage's LDS
                // slot and accumulate kSubTilesM x kSubTilesN MFMAs into acc[].  A
                // lambda so the overlap loop can issue two back-to-back per barrier.
                auto do_burst = [&](uint32_t k_block, uint32_t this_stage, uint32_t load_stage,
                                    uint32_t load_kb, bool do_load) {
                    const uint32_t a_stage_byte =
                        wave_base_byte + this_stage * kStagedABytesPerStage;
                    const uint32_t b_stage_byte =
                        kLoaderBPoolBaseBytes + this_stage * kStagedBBytesPerStage;
                    // SF read shadows for this stage: SFA per-warp (in the A
                    // stage SF region), SFB shared (in the SFB pool).
                    const uint32_t *sfa_lds_u32 = reinterpret_cast<const uint32_t *>(
                        smem_buffer + a_stage_byte + kATileBytes);
                    const uint32_t *sfb_lds_u32 = reinterpret_cast<const uint32_t *>(
                        smem_buffer + kLoaderSFBPoolBaseBytes + this_stage * kSFBStageBytes);

#pragma unroll
                    for (uint32_t k_inner = 0u; k_inner < kInnerKIters; ++k_inner) {
                        const uint32_t m_in_subtile = lane_idx & 15u;
                        const uint32_t kb_in_lane   = (lane_idx >> 4u) & 3u;
                        const uint32_t k_lo         = kb_in_lane * 16u;
                        const uint32_t a_win_hi     = k_lo + 64u;
                        const uint32_t perm_kblk    = kb_in_lane;

                        // ds_read both A frags DIRECTLY into pinned VGPRs (no a_vec
                        // C++ array, no v_mov): A0->v[kPinA0:+7], A1->v[kPinA1:+7];
                        // two ds_read_b128 per frag (k_lo and k_lo+64, swizzled).
                        // SFA scales are computed -> set_vgpr.  Staged ONCE per
                        // k_inner, shared by gate+up and all sub_n.
                        static_assert(kSubTilesM == 2u, "AGPR A-staging assumes kSubTilesM==2");
                        {
                            const uint32_t a_base = static_cast<uint32_t>(
                                reinterpret_cast<uintptr_t>(smem_buffer + a_stage_byte));
                            const uint32_t m0 = 0u * kMfmaM + m_in_subtile;
                            const uint32_t m1 = 1u * kMfmaM + m_in_subtile;
                            device::ds_read_pinned<16, kPinA0 + 0, 0>(
                                a_base +
                                device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(0u, m0,
                                                                                             k_lo));
                            device::ds_read_pinned<16, kPinA0 + 4, 0>(
                                a_base +
                                device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                    0u, m0, a_win_hi));
                            device::ds_read_pinned<16, kPinA1 + 0, 0>(
                                a_base +
                                device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(0u, m1,
                                                                                             k_lo));
                            device::ds_read_pinned<16, kPinA1 + 4, 0>(
                                a_base +
                                device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                    0u, m1, a_win_hi));
                            if (phase == sched::BlockPhase::Linear2) {
                                device::set_vgpr<kPinSA0>(kScaleOne);
                                device::set_vgpr<kPinSA1>(kScaleOne);
                            } else {
                                const uint32_t d0 = sfa_lds_u32[0u * kMfmaM + m_in_subtile];
                                const uint32_t d1 = sfa_lds_u32[1u * kMfmaM + m_in_subtile];
                                device::set_vgpr<kPinSA0>(0x7f7f7f00u |
                                                          ((d0 >> (perm_kblk * 8u)) & 0xffu));
                                device::set_vgpr<kPinSA1>(0x7f7f7f00u |
                                                          ((d1 >> (perm_kblk * 8u)) & 0xffu));
                            }
#if !MEGA_MOE_NO_AHEAD_WAIT
                            device::wait_lgkmcnt<0>(); // A frags landed in pinned VGPRs
#endif
                        }

                        auto b_off = [&](uint32_t sub_n) {
                            const uint32_t n_in_wave = sub_n * kMfmaN + m_in_subtile;
                            return device::b_tile_smem_byte_offset_rowmajor<kColsPerLoaderWave,
                                                                            BLOCK_K / 2u>(
                                0u, n_in_wave, kb_in_lane * 16u);
                        };
                        // Pinned-AGPR operand burst: accumulate one B operand into the
                        // AGPR range based at ACC_BASE.  A (a_vec) + scales (sa_arr) are
                        // read above; B + SFB read per sub_n.  Operands staged into the
                        // reserved pinned VGPR window via set_vgpr (the A LDS swizzle
                        // precludes turbo's compile-time-offset ds_read_pinned), then
                        // run_pinned_acc_agpr issues the MFMA with acc resident in AGPR.
                        using AgprMfma =
                            device::mfma_scale_f32_16x16x128_f8f6f4<__hip_fp8_e4m3,
                                                                    dtype::float4x2_e2m1>;
                        auto run_op_agpr = [&]<int ACC_BASE>(
                                               uint32_t b_lds_off, uint32_t sfb_u32_off,
                                               bool b0_preissued = false, int next_b_lds_off = -1) {
                            const uint32_t b_base =
                                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
                                    smem_buffer + b_stage_byte + b_lds_off));
                            // SFB scale: read ALL raw dwords upfront into registers and
                            // drain them ONCE.  The old inline path (read d -> format ->
                            // set_vgpr each sub_n) made the compiler's lgkmcnt wait for d
                            // over-drain the in-flight B prefetch (the compiler can't see
                            // the asm B ds_reads, so mixing compiler LDS loads with manual
                            // waits corrupts the count).  Reading all up front (clean
                            // slate) + formatting from registers removed ~12% of stalls.
                            constexpr int kN = static_cast<int>(kSubTilesN);
                            uint32_t      draw[kN];
#pragma unroll
                            for (int s = 0; s < kN; ++s)
                                draw[s] =
                                    sfb_lds_u32[sfb_u32_off + static_cast<uint32_t>(s) * kMfmaN +
                                                m_in_subtile];
                            // prologue B[0] issued BEFORE the scale drain so its ds_read
                            // overlaps the raw-scale reads (one wait covers both).  When
                            // XOP prefetch is on, the previous operand already issued our
                            // B[0] into kPinB0, so skip the re-issue here.
#if MEGA_MOE_XOP_B0_PREFETCH
                            if (!b0_preissued)
                                device::ds_read_pinned<16, kPinB0, 0>(b_base + b_off(0u));
#else
                            (void) b0_preissued;
                            (void) next_b_lds_off;
                            device::ds_read_pinned<16, kPinB0, 0>(b_base + b_off(0u));
#endif
                            device::wait_lgkmcnt<0>(); // raw scales + B[0] landed (parallel)
                            auto fmt = [&](uint32_t d) {
                                return 0x7f7f7f00u | ((d >> (kb_in_lane * 8u)) & 0xffu);
                            };
                            mega_static_for<0, kN>([&]<int SN>() {
                                constexpr int buf = SN & 1;
                                if constexpr (SN + 1 < kN) {
                                    constexpr int nbuf = (SN + 1) & 1;
                                    if constexpr (nbuf == 0)
                                        device::ds_read_pinned<16, kPinB0, 0>(
                                            b_base + b_off(static_cast<uint32_t>(SN + 1)));
                                    else
                                        device::ds_read_pinned<16, kPinB1, 0>(
                                            b_base + b_off(static_cast<uint32_t>(SN + 1)));
                                    device::wait_lgkmcnt<1>(); // B[SN] done, B[SN+1] in flight
                                } else {
                                    device::wait_lgkmcnt<0>(); // last: drain
#if MEGA_MOE_XOP_B0_PREFETCH
                                    // kPinB0 is free at the last (odd) sub_n -> pre-issue
                                    // the NEXT operand's B[0] here so its latency hides
                                    // under this operand's final MFMA + the up prologue.
                                    if constexpr (kN % 2 == 0) {
                                        if (next_b_lds_off >= 0) {
                                            const uint32_t nb_base =
                                                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
                                                    smem_buffer + b_stage_byte +
                                                    static_cast<uint32_t>(next_b_lds_off)));
                                            device::ds_read_pinned<16, kPinB0, 0>(nb_base +
                                                                                  b_off(0u));
                                        }
                                    }
#endif
                                }
                                // scale formatted from registers (no LDS read in the loop)
                                if constexpr (buf == 0)
                                    device::set_vgpr<kPinSB0>(fmt(draw[SN]));
                                else
                                    device::set_vgpr<kPinSB1>(fmt(draw[SN]));
                                if constexpr (buf == 0) {
                                    AgprMfma::run_pinned_acc_agpr<kPinA0, kPinB0,
                                                                  ACC_BASE + (0 * kN + SN) * 4,
                                                                  kPinSA0, kPinSB0>();
                                    AgprMfma::run_pinned_acc_agpr<kPinA1, kPinB0,
                                                                  ACC_BASE + (1 * kN + SN) * 4,
                                                                  kPinSA1, kPinSB0>();
                                } else {
                                    AgprMfma::run_pinned_acc_agpr<kPinA0, kPinB1,
                                                                  ACC_BASE + (0 * kN + SN) * 4,
                                                                  kPinSA0, kPinSB1>();
                                    AgprMfma::run_pinned_acc_agpr<kPinA1, kPinB1,
                                                                  ACC_BASE + (1 * kN + SN) * 4,
                                                                  kPinSA1, kPinSB1>();
                                }
                            });
                        };
                        // turbo phase_mfma_lds_ldg: drip the NEXT stage's global
                        // loads between the MFMA operands so the buffer_load_lds
                        // overlaps the matrix pipe (half 0 before the first operand,
                        // half 1 between operands).  Only on k_inner 0 (one issue/burst).
                        const bool drip = do_load && k_inner == 0u;
                        if (phase == sched::BlockPhase::Linear1) {
                            if (drip) {
                                issue_ab(load_stage, load_kb, 0u);
                                issue_sf(load_stage, load_kb, 0u);
                            }
                            // gate: pre-issue up's B[0] into kPinB0 at its last sub_n.
                            run_op_agpr.template operator()<kAccGate>(
                                0u, 0u, false, static_cast<int>(kBTileBytes));
                            if (drip)
                                issue_ab(load_stage, load_kb, 1u);
                            // up: B[0] already landed in kPinB0 by gate's tail prefetch.
                            run_op_agpr.template operator()<kAccUp>(kBTileBytes, BLOCK_N, true,
                                                                    -1); // up
                        } else {
                            if (drip) {
                                issue_ab(load_stage, load_kb, 0u);
                                issue_sf(load_stage, load_kb, 0u);
                            }
                            run_op_agpr.template operator()<kAccUp>(0u, 0u); // L2 -> a[0:63]
                            if (drip)
                                issue_ab(load_stage, load_kb, 1u);
                        }
                    }
                }; // do_burst

#if MEGA_MOE_TURBO_PIPE
                // ── turbo 2-stage shifted-LDG pipeline (GEMM_Tile_MXFP8 style) ──
                // One k_block per iteration into a cur/next LDS double buffer.  After
                // reading stage `cur`, the next-but-one k_block's tile is issued back
                // into `cur` (shifted-LDG) so its load overlaps the following MFMA; a
                // partial wait_vmcnt<kVmem*> drains ONLY the oldest stage, keeping the
                // next k_block in flight.  This relies on in-order buffer_load_lds
                // completion (the old >=4-stage loop avoided partial vmcnt for that
                // reason -- gate-3 cos is the correctness check).
                static_assert(kLoaderStages == 2u,
                              "turbo pipe is a cur/next double buffer -- set kNumStages=2");
                static_assert(kBTileLoadsPerWave % kNumMMANonEpilogueWarps == 0u,
                              "B loads must split evenly across warps for a uniform vmcnt");
                constexpr uint32_t kWarpBLoads = kBTileLoadsPerWave / kNumMMANonEpilogueWarps;
                constexpr uint32_t kSfaLoadsPerWarp =
                    (kRowsPerLoaderWave + kWarpSize - 1u) / kWarpSize;
                constexpr uint32_t kSfbColsPerWarp = BLOCK_N / kNumMMANonEpilogueWarps;
                constexpr uint32_t kSfbLoadsPerWarp =
                    (kSfbColsPerWarp + kWarpSize - 1u) / kWarpSize;
                // per-wave vmem ops issued for ONE k_block's tile (A + B + SF):
                constexpr uint32_t kVmemL2 = kATileLoadsPerWave + kWarpBLoads + kSfbLoadsPerWarp;
                constexpr uint32_t kVmemL1 = kATileLoadsPerWave + 2u * kWarpBLoads +
                                             kSfaLoadsPerWarp + 2u * kSfbLoadsPerWarp;

                // stage-0/1 prologue loads (k0->stage0, k1->stage1) by kloop_prologue.
                for (uint32_t k_block = 0u; k_block < num_k_blocks; ++k_block) {
                    const uint32_t cur = k_block & 1u;

                    // RAW: drain stage `cur` (this k_block).  Steady state leaves the
                    // next k_block's loads in flight; the last two tiles drain fully.
                    if (k_block + 2u < num_k_blocks) {
                        if (phase == sched::BlockPhase::Linear1)
                            device::wait_vmcnt<kVmemL1>();
                        else
                            device::wait_vmcnt<kVmemL2>();
                    } else {
                        device::wait_vmcnt<0>();
                    }
                    device::wait_lgkmcnt<0>();
                    __syncthreads(); // publish shared-B for stage cur

                    // MFMA burst over stage cur (no drip; loads issued explicitly below).
                    do_burst(k_block, cur, 0u, 0u, false);

                    // WAR: stage cur fully consumed -> safe to overwrite with k_block+2.
                    device::wait_lgkmcnt<0>();
                    __syncthreads();
                    if (k_block + 2u < num_k_blocks) {
                        issue_ab(cur, k_block + 2u);
                        issue_sf(cur, k_block + 2u);
                    }
                }
#else
                // k_block overlap: load two B-stages (k, k+1), drain with a
                // single full wait_vmcnt<0>, publish with ONE __syncthreads(), then
                // issue BOTH MFMA bursts back-to-back so the matrix pipe stays fed.
                // NOTES: the two bursts read stages s0=k%S, s1=(k+1)%S while the
                // loader fills (k+2)%S, (k+3)%S -- 4 distinct in-flight stages, so
                // requires kLoaderStages>=4.  A partial wait_vmcnt<K> is unsafe on
                // CDNA4 (no vmem completion-order guarantee) and the 2-wave/SIMD
                // occupancy already hides the full-drain latency.
                static_assert(kLoaderStages >= 4u,
                              "k_block-overlap needs kLoaderStages>=4 (reads s0,s1 "
                              "while loading s0+2,s1+2 -> 4 distinct in-flight stages)");

                // stage-0/1 prologue loads issued by kloop_prologue() before this.

                for (uint32_t k_block = 0u; k_block < num_k_blocks; k_block += 2u) {
                    const uint32_t s0 = k_block % kLoaderStages;
                    const uint32_t s1 = (k_block + 1u) % kLoaderStages;

                    // Both stage-k and stage-(k+1) loads have landed.
                    device::wait_vmcnt<0>();
                    device::wait_lgkmcnt<0>();

                    // ONE rendezvous publishes BOTH stages' shared-B to all warps.
                    __syncthreads();

                    // Two MFMA bursts back-to-back, NO barrier between.  Each burst
                    // DRIPS its stage's global loads between the MFMA operands (turbo
                    // phase_mfma_lds_ldg interleave) -- spreading beats front-loading
                    // (front-load measured -3%); the residual A/B LDG drain is
                    // lookahead-depth-limited (4-stage LDS cap), not schedule-limited.
                    do_burst(k_block, s0, (k_block + 2u) % kLoaderStages, k_block + 2u,
                             k_block + 2u < num_k_blocks);
                    if (k_block + 1u < num_k_blocks) // odd-tail: skip 2nd burst
                        do_burst(k_block + 1u, s1, (k_block + 3u) % kLoaderStages, k_block + 3u,
                                 k_block + 3u < num_k_blocks);
                }
#endif
            };

            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            static_assert(kNumL1BlockNs >= 2u && kNumL1BlockNs % 2u == 0u,
                          "L1 N must split evenly into gate||up halves");
            static_assert(kIntermediateHidden % BLOCK_N == 0u,
                          "BLOCK_N must divide kIntermediateHidden for SwiGLU pairing");
            constexpr uint32_t kL1GateNBlocks = kNumL1BlockNs / 2u;

            // (Block orchestration -- skip-check, kloop, epilogue -- is placed AFTER
            // run_epilogue's definition below so the DEFER path can call it.)

            // Epilogue extracted into a lambda so the block loop can DEFER it (run
            // block N's epilogue during block N+1's prologue loads).  Params carry
            // the (possibly deferred) block's state since the scheduler advances.
            auto run_epilogue = [&](sched::BlockPhase phase, uint32_t pool_block_idx,
                                    uint32_t n_block_idx, uint32_t valid_m) {
                const uint32_t n_lane     = lane_idx & 15u;
                const uint32_t m_out_base = ((lane_idx >> 4u) & 3u) * 4u;

                if (phase == sched::BlockPhase::Linear1) {

                    auto *l2_pool_base = l2_token_buffer.get_base_ptr<uint8_t>();
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub              = sub_m * kSubTilesN + sub_n;
                            const uint32_t n_in_wave        = sub_n * kMfmaN + n_lane;
                            const uint32_t intermediate_col = n_block_idx * BLOCK_N + n_in_wave;
                            if (intermediate_col >= kIntermediateHidden)
                                continue;
                            // Stream this subtile's gate/up acc from AGPR (4 VGPR each,
                            // live only across the i-loop) instead of a 128-VGPR hoist.
                            const dtype::float32x4 g_sub = read_acc_sub16<kAccGate>(sub);
                            const dtype::float32x4 u_sub = read_acc_sub16<kAccUp>(sub);
#pragma unroll
                            for (uint32_t i = 0u; i < 4u; ++i) {
                                const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                            sub_m * kMfmaM + m_out_base + i;
                                if (m_in_block >= valid_m)
                                    continue;
                                const uint32_t pool_token_idx =
                                    pool_block_idx * BLOCK_M + m_in_block;
                                auto *dst_row =
                                    l2_pool_base +
                                    pool_token_idx * fp8_intermediate_token_layout.num_bytes;

                                const float gate_raw = g_sub[i];
                                const float up_raw   = u_sub[i];
                                // fmed3f = median(x,lo,hi) = clamp in ONE op vs
                                // fmaxf(fminf()) = two.  Numerically identical.
                                const float gate = __builtin_amdgcn_fmed3f(
                                    gate_raw, -kActivationClamp, kActivationClamp);
                                const float up = __builtin_amdgcn_fmed3f(up_raw, -kActivationClamp,
                                                                         kActivationClamp);

                                const float silu_gate = gate / (1.0f + __expf(-gate));
                                const float swiglu    = silu_gate * up;
                                // swiglu is already clamped to +/-kActivationClamp
                                // (<< fp8 e4m3 range +/-448), so skip the cast's
                                // internal saturate/NAN-INF branch: convert directly
                                // via the HW pack instruction and take byte 0.
                                const uint32_t quant_pk =
                                    __builtin_amdgcn_cvt_pk_fp8_f32(swiglu, swiglu, 0, false);
                                const uint8_t quant_byte  = static_cast<uint8_t>(quant_pk & 0xffu);
                                dst_row[intermediate_col] = quant_byte;
                            }
                        }
                    }

                    // Signal L2 arrival (release-ordered): publishes the L2-pool
                    // token writes above before the consumer's gating count read.
                    if (elect_one()) {
                        auto *l2arr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        // EXPERIMENTAL (default-off, NOT clean): relaxed increments
                        // for gate 0..22 + one release on gate-23.  +21% but cos_sim
                        // 0.99997 (small error).  The per-gate-block release
                        // GRANULARITY is itself the correctness requirement on gfx950
                        // -- batching to one +24 release FAILED gate-3 worse (0.9976),
                        // and __threadfence writeback did not help.  No clean batching
                        // exists; left as a perf-vs-tiny-error opt-in only.
                        if (n_block_idx == kL1GateNBlocks - 1u)
                            red_add_rel_gpu(l2arr, 1ull);
                        else
                            (void) atomic_add(l2arr, 1ull); // relaxed
                    }
                } else {
                    // L2 epilogue (scalar): write each output element (BF16) into the
                    // remote combine buffer.  The source-token metadata (rank/token/
                    // topk) depends ONLY on m_in_block, so the m-row is the OUTER loop
                    // and the metadata + remote-row base are computed ONCE per row and
                    // reused across all sub_n columns.
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t i = 0u; i < 4u; ++i) {
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                        sub_m * kMfmaM + m_out_base + i;
                            if (m_in_block >= valid_m)
                                continue;
                            const auto meta = *workspace.get_token_src_metadata_ptr(
                                pool_block_idx * BLOCK_M + m_in_block);
                            auto *dst_local = combine_token_buffer.get_rank_buffer(meta.topk_idx)
                                                  .get_data_buffer(meta.token_idx)
                                                  .get_base_ptr<uint8_t>();
                            auto *dst_remote = sym_buffer.map(dst_local, meta.rank_idx);
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const uint32_t sub       = sub_m * kSubTilesN + sub_n;
                                const uint32_t n_in_wave = sub_n * kMfmaN + n_lane;
                                const uint32_t n_global  = n_block_idx * BLOCK_N + n_in_wave;
                                if (n_global >= kHidden)
                                    continue;
                                const __hip_bfloat16 bf =
                                    __float2bfloat16(read_acc_sub16<kAccUp>(sub)[i]);
                                *reinterpret_cast<__hip_bfloat16 *>(
                                    dst_remote + n_global * sizeof(__hip_bfloat16)) = bf;
                            }
                        }
                    }
                }
            };

            // ---- Block orchestration ----
            if (phase == sched::BlockPhase::Linear1 && n_block_idx >= kL1GateNBlocks)
                return;
            kloop_prologue();
            MEGA_PROF_T(t_mma);
            kloop_main();
            if (phase == sched::BlockPhase::Linear1) {
                MEGA_PROF_ADD(kStageL1Mma, t_mma);
            } else {
                MEGA_PROF_ADD(kStageL2Mma, t_mma);
            }
            MEGA_PROF_T(t_epi);
            run_epilogue(phase, pool_block_idx, n_block_idx,
                         scheduler.template get_valid_m<false>());
            if (phase == sched::BlockPhase::Linear1) {
                MEGA_PROF_ADD(kStageSwiGLU, t_epi);
            } else {
                MEGA_PROF_ADD(kStageL2Epi, t_epi);
            }
        });

        // Cumulative recv-stats: expert_recv_count_sum is globally visible (the
        // dispatch nvlink_barrier used sync_epilogue=true); each non-zero SM folds
        // its disjoint expert subset into the host-visible stat.
        if (sm_idx != 0u && cumulative_local_expert_recv_stats != nullptr) {
            for (uint32_t i = sm_idx - 1u; i < kNumExpertsPerRank; i += kNumSMs - 1u) {
                if (warp_idx == 0u && elect_one()) {
                    const uint32_t num_recv_tokens =
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
                    red_add(cumulative_local_expert_recv_stats + i,
                            static_cast<int>(num_recv_tokens));
                }
            }
        }

        // Combine: top-k reduce the per-rank combine buffers into the BF16 output
        // y.  The nvlink_barrier (grid_sync + cross-rank signal) makes every L2
        // write_combine globally visible before the reduction reads it.
        comm::nvlink_barrier<kNumRanks, kNumSMs, 1, 2>(workspace, sym_buffer, kEpiLeader, sm_idx,
                                                       thread_idx, true, true);

        const uint32_t     combine_warp_local = warp_idx - kNumDispatchWarps;
        constexpr uint32_t kNumElemsPerUint4  = sizeof(uint4) / sizeof(__hip_bfloat162);
        constexpr uint32_t kNumUint4PerToken  = (kHidden * sizeof(__hip_bfloat16)) / sizeof(uint4);
        static_assert(
            (kHidden * sizeof(__hip_bfloat16)) % sizeof(uint4) == 0u,
            "hidden * sizeof(bf16) must be a multiple of uint4 for combine vectorization");

        MEGA_PROF_T(t_combine);
        for (uint32_t token_idx = sm_idx * kNumMMANonEpilogueWarps + combine_warp_local;
             token_idx < num_tokens; token_idx += kNumSMs * kNumMMANonEpilogueWarps) {

            const int slot =
                lane_idx < kNumTopk
                    ? static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                             token_idx * kNumTopk + lane_idx))
                    : -1;
            const uint64_t mask = ballot(slot >= 0);

            for (uint32_t i = lane_idx; i < kNumUint4PerToken; i += kWarpSize) {
                const uint32_t off                        = i * sizeof(uint4);
                float2         reduced[kNumElemsPerUint4] = {};

                uint64_t remaining = mask;
                while (remaining) {
                    const uint32_t b = ffs(remaining) - 1u;
                    remaining ^= 1ull << b;
                    auto *src_ptr = combine_token_buffer.get_rank_buffer(b)
                                        .get_data_buffer(token_idx)
                                        .get_base_ptr<uint8_t>();
                    const uint4 partial = *reinterpret_cast<const uint4 *>(src_ptr + off);
                    const auto *bf16    = reinterpret_cast<const __hip_bfloat162 *>(&partial);
#pragma unroll
                    for (uint32_t l = 0u; l < kNumElemsPerUint4; ++l) {
                        const float2 fp32 = __bfloat1622float2(bf16[l]);
                        reduced[l].x += fp32.x;
                        reduced[l].y += fp32.y;
                    }
                }

                uint4 out;
                auto *bf16_out = reinterpret_cast<__hip_bfloat162 *>(&out);
#pragma unroll
                for (uint32_t l = 0u; l < kNumElemsPerUint4; ++l)
                    bf16_out[l] = __float22bfloat162_rn(reduced[l]);
                auto *dst =
                    reinterpret_cast<uint8_t *>(y) + token_idx * kHidden * sizeof(__hip_bfloat16);
                *reinterpret_cast<uint4 *>(dst + off) = out;
            }
        }
        MEGA_PROF_ADD(kStageCombine, t_combine);
        MEGA_PROF_ADD(kStageTotal, t_total);

        // Reset workspace counters for the next launch.  All SMs are past the
        // grid_sync above, so every l1_arrival_count / l2_arrival_mask has been
        // consumed and expert_recv_count_sum read; each non-zero SM owns a
        // disjoint expert subset, so no extra rendezvous is needed.
        if (sm_idx == 0u) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads)
                *workspace.get_expert_send_count_ptr(i) = 0u;
        } else {
            for (uint32_t i = sm_idx - 1u; i < kNumExpertsPerRank; i += kNumSMs - 1u) {
                const uint32_t num_recv_tokens =
                    static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
                const uint32_t num_recv_m_blocks       = (num_recv_tokens + BLOCK_M - 1u) / BLOCK_M;
                const uint32_t reset_pool_block_offset = scheduler.get_pool_block_offset(i);

                if (warp_idx == 0u && elect_one())
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0u;

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0u;

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumThreads) {
                    *workspace.get_l1_arrival_count_ptr(reset_pool_block_offset + j) = 0u;
                    *workspace.get_l2_arrival_mask_ptr(reset_pool_block_offset + j)  = 0ull;
                }
            }
        }
        return;
    }

#else
    PRIMUS_TURBO_DEVICE_CHECK(false);
#endif
}

template <MegaMoEArch kArch, uint32_t kNumMaxTokensPerRank, uint32_t kHidden,
          uint32_t kIntermediateHidden, uint32_t kNumExperts, uint32_t kNumTopk,
          uint32_t kNumExpertsPerWave, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t STORE_BLOCK_M, uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
          uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens, uint32_t kNumStages,
          uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads, uint32_t kNumSMs, uint32_t kNumRanks,
          float kActivationClamp, bool kFastMath>
hipError_t launch_fp8_fp4_mega_moe_impl(void *y, int *cumulative_local_expert_recv_stats,
                                        const uint32_t                      num_tokens,
                                        const layout::SymBuffer<kNumRanks> &sym_buffer,
                                        const void *l1_weights, const void *l1_weights_sf,
                                        const void *l2_weights, const void *l2_weights_sf,
                                        hipStream_t stream) {
    static_assert(kArch == MegaMoEArch::Gfx950, "Only gfx950 (MI355X) is supported for now");

    constexpr uint32_t kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    constexpr uint32_t kSmemBytes = 140u * 1024u;

    const dim3 grid(kNumSMs);
    const dim3 block(kNumThreads);

    auto kernel = gfx950_fp8_fp4_mega_moe_kernel<
        kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk,
        kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M, SF_BLOCK_M, SF_BLOCK_N,
        kNumMaxPoolTokens, kNumPaddedSFPoolTokens, kNumStages, kNumDispatchThreads,
        kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp,
        kFastMath>;

    if constexpr (kSmemBytes > 64u * 1024u) {
        const auto attr_err = hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(kSmemBytes));
        if (attr_err != hipSuccess)
            return attr_err;
    }

    hipLaunchKernelGGL(kernel, grid, block, kSmemBytes, stream, y,
                       cumulative_local_expert_recv_stats, num_tokens, sym_buffer,
                       const_cast<void *>(l1_weights), const_cast<void *>(l1_weights_sf),
                       const_cast<void *>(l2_weights), const_cast<void *>(l2_weights_sf));
    return hipGetLastError();
}

} // namespace mega_moe
} // namespace primus_turbo
