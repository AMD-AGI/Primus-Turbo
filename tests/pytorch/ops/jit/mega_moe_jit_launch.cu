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
//  Stub kernel launcher.  The DG-aligned device kernel body has not
//  yet been implemented; the Python test calls this only to confirm
//  the host launcher template instantiates cleanly under hipcc.
//  Returns a sentinel error code so callers can ``skip`` gracefully.
// ---------------------------------------------------------------------
extern "C" int mega_moe_jit_run_stub() {
    // Touch the impls/ template by referencing its symbol type — this
    // ensures the header parses under hipcc but does not emit a device
    // kernel until the body is supplied.
    using LauncherT = decltype(&mm::impls::launch_fp8_fp4_mega_moe_impl<
                               mm::impls::MegaMoEArch::Gfx950,
                               /*kNumMaxTokensPerRank=*/64u,
                               /*kHidden=*/256u, /*kIntermediateHidden=*/128u,
                               /*kNumExperts=*/8u, /*kNumTopk=*/2u,
                               /*kNumExpertsPerWave=*/8u,
                               /*BLOCK_M=*/128u, /*BLOCK_N=*/128u, /*BLOCK_K=*/128u,
                               /*STORE_BLOCK_M=*/32u,
                               /*SF_BLOCK_M=*/128u, /*SF_BLOCK_N=*/128u,
                               /*kNumMaxPoolTokens=*/384u,
                               /*kNumPaddedSFPoolTokens=*/384u,
                               /*kNumStages=*/4u,
                               /*kNumDispatchThreads=*/128u, /*kNumNonEpilogueThreads=*/128u,
                               /*kNumEpilogueThreads=*/256u,
                               /*kNumSMs=*/64u, /*kNumRanks=*/1u,
                               /*kActivationClamp=*/0.0f, /*kFastMath=*/true>);
    (void) sizeof(LauncherT);
    // Sentinel: 1 == kernel not implemented yet (matches Python check).
    return 1;
}
