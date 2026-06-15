// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// JIT entry point for the DG-aligned mega-MoE kernel-level layout
// helpers.  Compiled at test-time via ``torch.utils.cpp_extension.load``.
//
// The previous version of this TU launched a chain of five device
// kernels (Round 0) plus a fused Round 3 / per-wave Round 4 path; that
// device kernel has been removed in favor of a single DG-aligned
// launcher whose body has not yet been implemented (the alignment pass
// only brings the C++ API surface across).  To keep the JIT path
// useful, this TU now exposes the host-side layout helpers from the
// new kernel-level headers so the Python parity test can verify the
// new ``layout::Workspace`` / ``layout::Buffer`` / pool-token formulas
// match the Python reference byte-for-byte.

#include <algorithm>
#include <array>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include "primus_turbo/mega_moe.h"

#include "kernels/mega_moe/impls/gfx950_fp8_fp4_mega_moe.cuh"
#include "kernels/mega_moe/layout/mega_moe.cuh"
#include "kernels/mega_moe/layout/sym_buffer.cuh"
#include "kernels/mega_moe/scheduler/mega_moe.cuh"

namespace mm = primus_turbo::mega_moe;
namespace ly = primus_turbo::mega_moe::layout;

// ---------------------------------------------------------------------
//  Workspace probe.  Re-runs ``layout::Workspace::get_num_bytes()`` so
//  the Python parity test can confirm the C++ constant matches the
//  expected value computed from ``num_ranks`` / ``num_experts`` /
//  ``num_max_tokens_per_rank`` / ``num_topk`` alone.
// ---------------------------------------------------------------------
extern "C" void mega_moe_jit_workspace_probe(int num_ranks, int num_experts,
                                             int num_max_tokens_per_rank, int num_topk,
                                             int64_t  *out_workspace_bytes,
                                             uint32_t *out_num_max_pool_tokens,
                                             uint32_t *out_num_max_pool_blocks) {
    const ly::Workspace workspace(
        /*base=*/nullptr, static_cast<uint32_t>(num_ranks), static_cast<uint32_t>(num_experts),
        static_cast<uint32_t>(num_max_tokens_per_rank), static_cast<uint32_t>(num_topk));
    *out_workspace_bytes     = static_cast<int64_t>(workspace.get_num_bytes());
    *out_num_max_pool_tokens = workspace.num_max_pool_tokens;
    *out_num_max_pool_blocks = workspace.num_max_pool_blocks;
}

// ---------------------------------------------------------------------
//  Pool-token probe.  Exposes the templated helpers so Python can
//  cross-check ``get_num_max_pool_tokens`` and
//  ``get_num_padded_sf_pool_tokens`` against its own implementation.
// ---------------------------------------------------------------------
extern "C" int mega_moe_jit_num_max_pool_tokens(int num_ranks, int num_max_tokens_per_rank,
                                                int num_topk, int num_experts_per_rank) {
    return ly::get_num_max_pool_tokens<int>(num_ranks, num_max_tokens_per_rank, num_topk,
                                            num_experts_per_rank);
}

extern "C" int mega_moe_jit_num_padded_sf_pool_tokens(int num_max_pool_tokens, int block_m) {
    return ly::get_num_padded_sf_pool_tokens<int>(num_max_pool_tokens, block_m);
}

extern "C" int mega_moe_jit_get_token_alignment_for_mega_moe() {
    return static_cast<int>(mm::kTokenAlignment);
}

// ---------------------------------------------------------------------
//  Buffer-layout probe.  Replicates the offset math in
//  ``csrc/kernels/mega_moe/mega_moe.cu::get_symm_buffer_size_for_mega_moe``
//  using the kernel-level layout helpers, so the Python reference (in
//  ``test_mega_moe_jit_perf.py::python_symm_buffer_layout``) can be
//  validated as exactly matching the C++ side.
//
//  The function intentionally re-implements the same offsets the host
//  helper would compute, rather than calling into the linked
//  ``get_symm_buffer_size_for_mega_moe`` symbol — the JIT TU stays
//  self-contained that way (no need to link against the precompiled
//  ``libprimus_turbo_kernels.so``).
// ---------------------------------------------------------------------
namespace {

inline int max_num_padded_sf_pool_tokens(int num_max_pool_tokens) {
    int result = 0;
    for (int i = 0; i < mm::kNumCandidateBlockMs; ++i) {
        result = std::max(result, ly::get_num_padded_sf_pool_tokens<int>(num_max_pool_tokens,
                                                                         mm::kCandidateBlockM[i]));
    }
    return result;
}

template <typename T> constexpr T jit_align(T v, T a) {
    return ((v + a - 1) / a) * a;
}

} // anonymous namespace

extern "C" void mega_moe_jit_compute_layout(int num_ranks, int num_experts,
                                            int num_max_tokens_per_rank, int num_topk, int hidden,
                                            int      intermediate_hidden,
                                            int64_t *out_offsets, // size 11
                                            int64_t *out_total_bytes, int *out_num_max_pool_tokens,
                                            int *out_num_padded_sf_pool_tokens) {
    const int num_experts_per_rank = num_experts / num_ranks;
    const int num_max_pool_tokens  = ly::get_num_max_pool_tokens<int>(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int num_padded_sf_pool_tokens = max_num_padded_sf_pool_tokens(num_max_pool_tokens);

    const ly::Workspace workspace(
        /*base=*/nullptr, static_cast<uint32_t>(num_ranks), static_cast<uint32_t>(num_experts),
        static_cast<uint32_t>(num_max_tokens_per_rank), static_cast<uint32_t>(num_topk));
    const int64_t workspace_bytes = static_cast<int64_t>(workspace.get_num_bytes());

    const int64_t bf16_token_bytes = static_cast<int64_t>(hidden) * 2; // BF16
#if MEGA_MOE_FP4_ACT
    // MEGA_MOE_FP4_ACT: BOTH activations are MXFP4 e2m1 nibble-packed ->
    // 0.5 B/elem.  The dispatched INPUT (Linear1's A, input_x and l1_pool_x)
    // and the L2 SwiGLU intermediate (l2_pool_x) are each halved to match the
    // kernel's halved fp8_token_layout / fp8_intermediate_token_layout so the
    // symm buffer offsets line up.
    const int64_t fp8_token_bytes = hidden / 2;              // FP4 e2m1 input
    const int64_t fp8_inter_bytes = intermediate_hidden / 2; // FP4 e2m1
#else
    const int64_t fp8_token_bytes = hidden;              // FP8
    const int64_t fp8_inter_bytes = intermediate_hidden; // FP8
#endif
    const int64_t fp8_sf_bytes = hidden / mm::kScaleGroupK;
    const int64_t fp8_inter_sf_bytes = intermediate_hidden / mm::kScaleGroupK;
    const int64_t topk_idx_bytes     = static_cast<int64_t>(num_topk) * sizeof(int64_t);
    const int64_t topk_weights_bytes = static_cast<int64_t>(num_topk) * sizeof(float);

    auto bump = [](int64_t &cursor, int64_t bytes) -> int64_t {
        const int64_t aligned = jit_align<int64_t>(cursor, 256);
        cursor                = aligned + bytes;
        return aligned;
    };

    int64_t cursor = 0;
    out_offsets[0] = bump(cursor, workspace_bytes);                           // workspace
    out_offsets[1] = bump(cursor, num_max_tokens_per_rank * fp8_token_bytes); // input_x
    out_offsets[2] = bump(cursor, num_max_tokens_per_rank * fp8_sf_bytes);    // input_x_sf
    out_offsets[3] = bump(cursor, num_max_tokens_per_rank * topk_idx_bytes);  // input_topk_idx
    out_offsets[4] =
        bump(cursor, num_max_tokens_per_rank * topk_weights_bytes);          // input_topk_weights
    out_offsets[5] = bump(cursor, num_max_pool_tokens * fp8_token_bytes);    // l1_pool_x
    out_offsets[6] = bump(cursor, num_padded_sf_pool_tokens * fp8_sf_bytes); // l1_pool_x_sf
    out_offsets[7] =
        bump(cursor, num_max_pool_tokens * static_cast<int64_t>(sizeof(float))); // l1_pool_weights
    out_offsets[8]  = bump(cursor, num_max_pool_tokens * fp8_inter_bytes);       // l2_pool_x
    out_offsets[9]  = bump(cursor, num_padded_sf_pool_tokens * fp8_inter_sf_bytes); // l2_pool_x_sf
    out_offsets[10] = bump(cursor,
                           static_cast<int64_t>(num_topk) * num_max_tokens_per_rank *
                               bf16_token_bytes); // combine_buffer

    *out_total_bytes               = cursor;
    *out_num_max_pool_tokens       = num_max_pool_tokens;
    *out_num_padded_sf_pool_tokens = num_padded_sf_pool_tokens;
}

// ---------------------------------------------------------------------
//  Compile-time "smoke" shape that the JIT TU pre-instantiates.  Mirrors
//  the SymBuffer layout used by ``test_mega_moe_jit_perf.py``.  Each
//  constant below can be overridden from the Python loader by passing
//  ``-DMEGA_MOE_JIT_K<NAME>=<value>`` in ``extra_cuda_cflags`` so a
//  single TU can be re-JIT'd for arbitrary DG-style shapes (e.g.
//  ``hidden=7168, intermediate=3072, experts=384, topk=6``).  The
//  ``MEGA_MOE_JIT_KNUMRANKS`` override is driven by the Python
//  ``--num-processes`` argument; the device kernel currently
//  ``static_assert``-s ``kNumRanks == 1u`` (impls/gfx950 head comment),
//  so picking ``num_processes > 1`` will fail the JIT compile with a
//  clear diagnostic until TODO.md §A.14 (multi-rank dispatch
//  round-robin) lands.
// ---------------------------------------------------------------------
#ifndef MEGA_MOE_JIT_KNUMMAXTOKENSPERRANK
#define MEGA_MOE_JIT_KNUMMAXTOKENSPERRANK 384u // aligned to kLCMCandidateBlockM
#endif
#ifndef MEGA_MOE_JIT_KHIDDEN
#define MEGA_MOE_JIT_KHIDDEN 256u
#endif
#ifndef MEGA_MOE_JIT_KINTERMEDIATEHIDDEN
#define MEGA_MOE_JIT_KINTERMEDIATEHIDDEN 128u
#endif
#ifndef MEGA_MOE_JIT_KNUMEXPERTS
#define MEGA_MOE_JIT_KNUMEXPERTS 8u
#endif
#ifndef MEGA_MOE_JIT_KNUMTOPK
#define MEGA_MOE_JIT_KNUMTOPK 2u
#endif
#ifndef MEGA_MOE_JIT_KNUMRANKS
#define MEGA_MOE_JIT_KNUMRANKS 1u
#endif
#ifndef MEGA_MOE_JIT_KNUMEXPERTSPERWAVE
#define MEGA_MOE_JIT_KNUMEXPERTSPERWAVE 8u
#endif
#ifndef MEGA_MOE_JIT_KBLOCK_M
#define MEGA_MOE_JIT_KBLOCK_M 128u
#endif
#ifndef MEGA_MOE_JIT_KBLOCK_N
#define MEGA_MOE_JIT_KBLOCK_N 128u
#endif
#ifndef MEGA_MOE_JIT_KBLOCK_K
#define MEGA_MOE_JIT_KBLOCK_K 128u
#endif
#ifndef MEGA_MOE_JIT_KSTORE_BLOCK_M
#define MEGA_MOE_JIT_KSTORE_BLOCK_M 32u
#endif
#ifndef MEGA_MOE_JIT_KSF_BLOCK_M
#define MEGA_MOE_JIT_KSF_BLOCK_M 128u
#endif
#ifndef MEGA_MOE_JIT_KSF_BLOCK_N
#define MEGA_MOE_JIT_KSF_BLOCK_N 128u
#endif
// 2 blocks/CU occupancy build: 2-stage loader (halves LDS) + 512 logical SMs
// (grid=512 on 256 CUs -> 2 resident blocks/CU).  Defined before the #ifndef
// defaults so these win.
#if MEGA_MOE_2BLK
#define MEGA_MOE_JIT_KNUMSTAGES 2u
#define MEGA_MOE_JIT_KNUMSMS 512u
#endif
#ifndef MEGA_MOE_JIT_KNUMSTAGES
#define MEGA_MOE_JIT_KNUMSTAGES 4u
#endif
#ifndef MEGA_MOE_JIT_KNUMDISPATCHTHREADS
// 4-warp DISPATCH-FOLD layout: the dedicated dispatch warp role is removed.
// The 4 MFMA warps now run the route + cross-rank token pull themselves
// (PHASE 1 in gfx950_fp8_fp4_mega_moe.cuh) before the compute, so there are
// no dispatch-only threads.  Block = 0 + 256 + 0 = 256 (4 warps).  Fewer
// warps/CU at occupancy 1 give each wave ~1.5x the VGPR budget of the 6-warp
// layout (relieves the spill that tanked the fold to 34 TF); the cost is the
// loss of the dispatch<->compute overlap.
#define MEGA_MOE_JIT_KNUMDISPATCHTHREADS 0u
#endif
#ifndef MEGA_MOE_JIT_KNUMNONEPILOGUETHREADS
// 8 MFMA warps (512 thr, 1 block/CU = 2 waves/SIMD) — latency-hiding experiment:
// PC sampling showed the 4-warp/1-wave-per-SIMD build is global-load-latency
// bound (51% of samples park on s_waitcnt vmcnt(0)); a 2nd wave/SIMD covers the
// vmcnt stall.  8 warps keeps the same per-stage LDS / 4 stages and HALVES the
// accumulator footprint (kSubTilesPerWave 16->8, ~64 acc-VGPR).  Was 256u (4
// warps, 245 TF) — revert if this regresses.
#if MEGA_MOE_AGPR_ACC
// Pinned-AGPR redesign runs 4 warps (turbo-style single-wave software pipeline:
// acc resident in AGPR a[0:127], A/B/scale in pinned VGPRs, k-loop hand-scheduled).
#define MEGA_MOE_JIT_KNUMNONEPILOGUETHREADS 256u
#else
#define MEGA_MOE_JIT_KNUMNONEPILOGUETHREADS 512u
#endif
#endif
#ifndef MEGA_MOE_JIT_KNUMEPILOGUETHREADS
// epilogue role removed (combine folded into the 4 MFMA warps post-compute).
#define MEGA_MOE_JIT_KNUMEPILOGUETHREADS 0u
#endif
// MI355X exposes 256 CUs (8 XCDs x 32 CUs).  Keep as a multiple of 8 to
// preserve per-XCD locality of the scheduler state machine.
#ifndef MEGA_MOE_JIT_KNUMSMS
#define MEGA_MOE_JIT_KNUMSMS 256u
#endif

namespace smoke {
constexpr uint32_t kNumMaxTokensPerRank   = MEGA_MOE_JIT_KNUMMAXTOKENSPERRANK;
constexpr uint32_t kHidden                = MEGA_MOE_JIT_KHIDDEN;
constexpr uint32_t kIntermediateHidden    = MEGA_MOE_JIT_KINTERMEDIATEHIDDEN;
constexpr uint32_t kNumExperts            = MEGA_MOE_JIT_KNUMEXPERTS;
constexpr uint32_t kNumTopk               = MEGA_MOE_JIT_KNUMTOPK;
constexpr uint32_t kNumRanks              = MEGA_MOE_JIT_KNUMRANKS;
constexpr uint32_t kBLOCK_M               = MEGA_MOE_JIT_KBLOCK_M;
constexpr uint32_t kBLOCK_N               = MEGA_MOE_JIT_KBLOCK_N;
constexpr uint32_t kBLOCK_K               = MEGA_MOE_JIT_KBLOCK_K;
constexpr uint32_t kSTORE_BLOCK_M         = MEGA_MOE_JIT_KSTORE_BLOCK_M;
constexpr uint32_t kSF_BLOCK_M            = MEGA_MOE_JIT_KSF_BLOCK_M;
constexpr uint32_t kSF_BLOCK_N            = MEGA_MOE_JIT_KSF_BLOCK_N;
constexpr uint32_t kNumStages             = MEGA_MOE_JIT_KNUMSTAGES;
constexpr uint32_t kNumDispatchThreads    = MEGA_MOE_JIT_KNUMDISPATCHTHREADS;
constexpr uint32_t kNumNonEpilogueThreads = MEGA_MOE_JIT_KNUMNONEPILOGUETHREADS;
constexpr uint32_t kNumEpilogueThreads    = MEGA_MOE_JIT_KNUMEPILOGUETHREADS;
constexpr uint32_t kNumSMs                = MEGA_MOE_JIT_KNUMSMS;
static_assert(kNumExperts % kNumRanks == 0,
              "MEGA_MOE_JIT_KNUMEXPERTS must be divisible by MEGA_MOE_JIT_KNUMRANKS");
constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
// Clamp the requested wave size to ``kNumExpertsPerRank``.  The scheduler
// enforces ``kNumExpertsPerRank % kNumExpertsPerWave == 0``; when the
// caller hands us a config with fewer experts per rank than the default
// wave width (e.g. EP8 + 8 experts -> 1 expert/rank), shrink the wave
// to 1 so the kernel still instantiates.
constexpr uint32_t kNumExpertsPerWave = MEGA_MOE_JIT_KNUMEXPERTSPERWAVE < kNumExpertsPerRank
                                            ? MEGA_MOE_JIT_KNUMEXPERTSPERWAVE
                                            : kNumExpertsPerRank;

// Derived pool sizing - constexpr-computed from the shape constants
// above so callers only need to override the primary knobs.
constexpr uint32_t kNumMaxPoolTokens = ly::get_num_max_pool_tokens<uint32_t>(
    kNumRanks, kNumMaxTokensPerRank, kNumTopk, kNumExpertsPerRank);

constexpr uint32_t compute_num_padded_sf_pool_tokens(uint32_t pool_tokens) {
    uint32_t result = 0;
    for (int i = 0; i < mm::kNumCandidateBlockMs; ++i) {
        const uint32_t v = ly::get_num_padded_sf_pool_tokens<uint32_t>(
            pool_tokens, static_cast<uint32_t>(mm::kCandidateBlockM[i]));
        if (v > result)
            result = v;
    }
    return result;
}
constexpr uint32_t kNumPaddedSFPoolTokens = compute_num_padded_sf_pool_tokens(kNumMaxPoolTokens);
} // namespace smoke

// ---------------------------------------------------------------------
//  DG-aligned runtime entry point.  Accepts raw device pointers (typed
//  via ``int64_t`` so it can be called directly with ``Tensor.data_ptr()``
//  from Python) and dispatches the gfx950 mega-MoE kernel.
//
//  Only the compile-time ``smoke`` shape above is currently supported;
//  the runtime shape arguments are validated against the template
//  instantiation and a non-zero error code is returned on mismatch so
//  the Python test can print a helpful diagnostic.
//
//  Return codes:
//    0   -> launch + sync succeeded.
//    -1  -> shape mismatch vs. the JIT-instantiated template.
//    >0  -> ``hipError_t`` reported by ``hipLaunchKernelGGL`` /
//           ``hipDeviceSynchronize`` (cast to int).
// ---------------------------------------------------------------------
extern "C" int mega_moe_jit_run_mega_moe(const int64_t *sym_buffer_bases, int num_sym_buffer_bases,
                                         int rank_idx, int num_tokens, int num_max_tokens_per_rank,
                                         int hidden, int intermediate_hidden, int num_experts,
                                         int num_topk, int num_ranks, int64_t y_ptr,
                                         int64_t l1_weights_ptr, int64_t l1_weights_sf_ptr,
                                         int64_t l2_weights_ptr, int64_t l2_weights_sf_ptr,
                                         int64_t recv_stats_ptr, float activation_clamp,
                                         int fast_math) {
    if (num_max_tokens_per_rank != static_cast<int>(smoke::kNumMaxTokensPerRank) ||
        hidden != static_cast<int>(smoke::kHidden) ||
        intermediate_hidden != static_cast<int>(smoke::kIntermediateHidden) ||
        num_experts != static_cast<int>(smoke::kNumExperts) ||
        num_topk != static_cast<int>(smoke::kNumTopk) ||
        num_ranks != static_cast<int>(smoke::kNumRanks)) {
        return -1;
    }
    // Caller must pass exactly ``smoke::kNumRanks`` peer pointers (one
    // per rank, IPC-rendezvous'd into this process's address space).
    if (num_sym_buffer_bases != static_cast<int>(smoke::kNumRanks))
        return -1;
    (void) fast_math; // captured at compile time via the smoke template
    (void) activation_clamp;

    std::vector<int64_t> sym_ptrs(sym_buffer_bases, sym_buffer_bases + num_sym_buffer_bases);
    ly::SymBuffer<smoke::kNumRanks> sym_buffer(sym_ptrs, static_cast<uint32_t>(rank_idx));

    const hipError_t launch_err = mm::launch_fp8_fp4_mega_moe_impl<
        mm::MegaMoEArch::Gfx950, smoke::kNumMaxTokensPerRank, smoke::kHidden,
        smoke::kIntermediateHidden, smoke::kNumExperts, smoke::kNumTopk, smoke::kNumExpertsPerWave,
        smoke::kBLOCK_M, smoke::kBLOCK_N, smoke::kBLOCK_K, smoke::kSTORE_BLOCK_M,
        smoke::kSF_BLOCK_M, smoke::kSF_BLOCK_N, smoke::kNumMaxPoolTokens,
        smoke::kNumPaddedSFPoolTokens, smoke::kNumStages, smoke::kNumDispatchThreads,
        smoke::kNumNonEpilogueThreads, smoke::kNumEpilogueThreads, smoke::kNumSMs, smoke::kNumRanks,
        /*kActivationClamp=*/10.0f, /*kFastMath=*/true>(
        reinterpret_cast<void *>(y_ptr), reinterpret_cast<int *>(recv_stats_ptr),
        static_cast<uint32_t>(num_tokens), sym_buffer, reinterpret_cast<void *>(l1_weights_ptr),
        reinterpret_cast<void *>(l1_weights_sf_ptr), reinterpret_cast<void *>(l2_weights_ptr),
        reinterpret_cast<void *>(l2_weights_sf_ptr),
        /*stream=*/0);
    const hipError_t sync_err = hipDeviceSynchronize();

    if (launch_err != hipSuccess)
        return static_cast<int>(launch_err);
    if (sync_err != hipSuccess)
        return static_cast<int>(sync_err);
    return 0;
}

// ---------------------------------------------------------------------
//  Launch probe.  Actually dispatches the gfx950 mega-MoE kernel with
//  ``num_tokens == 0`` so the Python smoke test can confirm that:
//    1. The launcher template parses cleanly under hipcc.
//    2. ``hipLaunchKernelGGL`` accepts the kernel symbol (i.e., the
//       ``__global__`` body emitted for ``--offload-arch=gfx950`` has
//       valid ISA).
//    3. The kernel actually runs end-to-end on the device and the
//       early-exit path for ``num_tokens == 0`` terminates cleanly.
//
//  Return codes:
//    0  -> launch succeeded + ``hipDeviceSynchronize`` returned success.
//    >0 -> ``hipError_t`` value reported by HIP (cast to int).  Non-zero
//          values propagate the failing stage so the Python side can
//          print a meaningful diagnostic.
// ---------------------------------------------------------------------
extern "C" int mega_moe_jit_run_stub() {
    // Reuse the ``smoke`` namespace constants so we share a single
    // template instantiation with ``mega_moe_jit_run_mega_moe``.

    // Symmetric buffer: one contiguous device allocation that hosts the
    // workspace + all per-rank pools.  For the smoke test we
    // overallocate a fixed 8 MiB — the kernel's early-exit path returns
    // before touching any byte outside the workspace header.
    constexpr size_t kSymBytes = static_cast<size_t>(8) * 1024u * 1024u;

    void *d_sym = nullptr;
    if (hipError_t err = hipMalloc(&d_sym, kSymBytes); err != hipSuccess)
        return static_cast<int>(err);
    if (hipError_t err = hipMemset(d_sym, 0, kSymBytes); err != hipSuccess) {
        (void) hipFree(d_sym);
        return static_cast<int>(err);
    }

    void            *d_y           = nullptr;
    void            *d_l1_w        = nullptr;
    void            *d_l1_w_sf     = nullptr;
    void            *d_l2_w        = nullptr;
    void            *d_l2_w_sf     = nullptr;
    constexpr size_t kAuxBytes     = 1024u * 1024u;
    auto             alloc_or_fail = [&](void **p) -> int {
        if (hipError_t err = hipMalloc(p, kAuxBytes); err != hipSuccess)
            return static_cast<int>(err);
        return static_cast<int>(hipMemset(*p, 0, kAuxBytes));
    };
    auto cleanup_all = [&]() {
        if (d_sym)
            (void) hipFree(d_sym);
        if (d_y)
            (void) hipFree(d_y);
        if (d_l1_w)
            (void) hipFree(d_l1_w);
        if (d_l1_w_sf)
            (void) hipFree(d_l1_w_sf);
        if (d_l2_w)
            (void) hipFree(d_l2_w);
        if (d_l2_w_sf)
            (void) hipFree(d_l2_w_sf);
    };
    if (int rc = alloc_or_fail(&d_y); rc != hipSuccess) {
        cleanup_all();
        return rc;
    }
    if (int rc = alloc_or_fail(&d_l1_w); rc != hipSuccess) {
        cleanup_all();
        return rc;
    }
    if (int rc = alloc_or_fail(&d_l1_w_sf); rc != hipSuccess) {
        cleanup_all();
        return rc;
    }
    if (int rc = alloc_or_fail(&d_l2_w); rc != hipSuccess) {
        cleanup_all();
        return rc;
    }
    if (int rc = alloc_or_fail(&d_l2_w_sf); rc != hipSuccess) {
        cleanup_all();
        return rc;
    }

    std::vector<int64_t>            sym_ptrs(smoke::kNumRanks, reinterpret_cast<int64_t>(d_sym));
    ly::SymBuffer<smoke::kNumRanks> sym_buffer(sym_ptrs, /*rank_idx=*/0u);

    const hipError_t launch_err = mm::launch_fp8_fp4_mega_moe_impl<
        mm::MegaMoEArch::Gfx950, smoke::kNumMaxTokensPerRank, smoke::kHidden,
        smoke::kIntermediateHidden, smoke::kNumExperts, smoke::kNumTopk, smoke::kNumExpertsPerWave,
        smoke::kBLOCK_M, smoke::kBLOCK_N, smoke::kBLOCK_K, smoke::kSTORE_BLOCK_M,
        smoke::kSF_BLOCK_M, smoke::kSF_BLOCK_N, smoke::kNumMaxPoolTokens,
        smoke::kNumPaddedSFPoolTokens, smoke::kNumStages, smoke::kNumDispatchThreads,
        smoke::kNumNonEpilogueThreads, smoke::kNumEpilogueThreads, smoke::kNumSMs, smoke::kNumRanks,
        /*kActivationClamp=*/10.0f,
        /*kFastMath=*/true>(d_y, /*cumulative_local_expert_recv_stats=*/nullptr,
                            /*num_tokens=*/0u, sym_buffer, d_l1_w, d_l1_w_sf, d_l2_w, d_l2_w_sf,
                            /*stream=*/0);

    const hipError_t sync_err = hipDeviceSynchronize();

    (void) hipFree(d_sym);
    (void) hipFree(d_y);
    (void) hipFree(d_l1_w);
    (void) hipFree(d_l1_w_sf);
    (void) hipFree(d_l2_w);
    (void) hipFree(d_l2_w_sf);

    if (launch_err != hipSuccess)
        return static_cast<int>(launch_err);
    if (sync_err != hipSuccess)
        return static_cast<int>(sync_err);
    return 0;
}
