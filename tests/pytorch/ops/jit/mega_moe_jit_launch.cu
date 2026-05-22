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

    const int64_t fp8_token_bytes    = hidden;                           // FP8
    const int64_t bf16_token_bytes   = static_cast<int64_t>(hidden) * 2; // BF16
    const int64_t fp8_inter_bytes    = intermediate_hidden;              // FP8
    const int64_t fp8_sf_bytes       = hidden / mm::kScaleGroupK;
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
//  the SymBuffer layout used by ``test_mega_moe_jit_perf.py`` and is
//  small enough to compile + launch on a single MI355X with
//  ``num_processes == 1``.  Larger shapes (matching DG defaults such as
//  ``hidden=7168, intermediate=3072, experts=384, topk=6``) will be
//  added as additional template instantiations once the runtime path
//  (partial barriers + MFMA tile body) lands.
// ---------------------------------------------------------------------
namespace smoke {
constexpr uint32_t kNumMaxTokensPerRank = 384u; // aligned to kLCMCandidateBlockM
constexpr uint32_t kHidden              = 256u;
constexpr uint32_t kIntermediateHidden  = 128u;
constexpr uint32_t kNumExperts          = 8u;
constexpr uint32_t kNumTopk             = 2u;
constexpr uint32_t kNumExpertsPerWave   = 8u;
constexpr uint32_t kBLOCK_M             = 128u;
constexpr uint32_t kBLOCK_N             = 128u;
constexpr uint32_t kBLOCK_K             = 128u;
constexpr uint32_t kSTORE_BLOCK_M       = 32u;
constexpr uint32_t kSF_BLOCK_M          = 128u;
constexpr uint32_t kSF_BLOCK_N          = 128u;
// align_up(1*384*min(2,8) + 8*(192-1), 384) = 2304
constexpr uint32_t kNumMaxPoolTokens = 2304u;
// max over candidate block_ms of (2304//bm)*align_up(bm,128) = 36864 (bm=8)
constexpr uint32_t kNumPaddedSFPoolTokens = 36864u;
constexpr uint32_t kNumStages             = 4u;
constexpr uint32_t kNumDispatchThreads    = 128u;
constexpr uint32_t kNumNonEpilogueThreads = 128u;
constexpr uint32_t kNumEpilogueThreads    = 256u;
constexpr uint32_t kNumSMs                = 64u;
constexpr uint32_t kNumRanks              = 1u;
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
extern "C" int mega_moe_jit_run_mega_moe(int64_t sym_buffer_base, int rank_idx, int num_tokens,
                                         int num_max_tokens_per_rank, int hidden,
                                         int intermediate_hidden, int num_experts, int num_topk,
                                         int num_ranks, int64_t y_ptr, int64_t l1_weights_ptr,
                                         int64_t l1_weights_sf_ptr, int64_t l2_weights_ptr,
                                         int64_t l2_weights_sf_ptr, int64_t recv_stats_ptr,
                                         float activation_clamp, int fast_math) {
    if (num_max_tokens_per_rank != static_cast<int>(smoke::kNumMaxTokensPerRank) ||
        hidden != static_cast<int>(smoke::kHidden) ||
        intermediate_hidden != static_cast<int>(smoke::kIntermediateHidden) ||
        num_experts != static_cast<int>(smoke::kNumExperts) ||
        num_topk != static_cast<int>(smoke::kNumTopk) ||
        num_ranks != static_cast<int>(smoke::kNumRanks)) {
        return -1;
    }
    (void) fast_math; // captured at compile time via the smoke template
    (void) activation_clamp;

    std::array<int64_t, 1>          sym_ptrs{sym_buffer_base};
    ly::SymBuffer<smoke::kNumRanks> sym_buffer(sym_ptrs, static_cast<uint32_t>(rank_idx));

    const hipError_t launch_err = mm::impls::launch_fp8_fp4_mega_moe_impl<
        mm::impls::MegaMoEArch::Gfx950, smoke::kNumMaxTokensPerRank, smoke::kHidden,
        smoke::kIntermediateHidden, smoke::kNumExperts, smoke::kNumTopk, smoke::kNumExpertsPerWave,
        smoke::kBLOCK_M, smoke::kBLOCK_N, smoke::kBLOCK_K, smoke::kSTORE_BLOCK_M,
        smoke::kSF_BLOCK_M, smoke::kSF_BLOCK_N, smoke::kNumMaxPoolTokens,
        smoke::kNumPaddedSFPoolTokens, smoke::kNumStages, smoke::kNumDispatchThreads,
        smoke::kNumNonEpilogueThreads, smoke::kNumEpilogueThreads, smoke::kNumSMs, smoke::kNumRanks,
        /*kActivationClamp=*/0.0f, /*kFastMath=*/true>(
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

    std::array<int64_t, 1>          sym_ptrs{reinterpret_cast<int64_t>(d_sym)};
    ly::SymBuffer<smoke::kNumRanks> sym_buffer(sym_ptrs, /*rank_idx=*/0u);

    const hipError_t launch_err = mm::impls::launch_fp8_fp4_mega_moe_impl<
        mm::impls::MegaMoEArch::Gfx950, smoke::kNumMaxTokensPerRank, smoke::kHidden,
        smoke::kIntermediateHidden, smoke::kNumExperts, smoke::kNumTopk, smoke::kNumExpertsPerWave,
        smoke::kBLOCK_M, smoke::kBLOCK_N, smoke::kBLOCK_K, smoke::kSTORE_BLOCK_M,
        smoke::kSF_BLOCK_M, smoke::kSF_BLOCK_N, smoke::kNumMaxPoolTokens,
        smoke::kNumPaddedSFPoolTokens, smoke::kNumStages, smoke::kNumDispatchThreads,
        smoke::kNumNonEpilogueThreads, smoke::kNumEpilogueThreads, smoke::kNumSMs, smoke::kNumRanks,
        /*kActivationClamp=*/0.0f,
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
