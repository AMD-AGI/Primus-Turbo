// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Mega MoE host launcher — fused EP dispatch + L1 GEMM + activation +
// L2 GEMM + EP combine.  Mirrors DeepGEMM's host wrapper
// ``deep_gemm::mega::fp8_fp4_mega_moe`` (csrc/apis/mega.hpp) and uses
// the kernel-level C++ layout helpers in ``layout/`` (a 1:1 port of
// ``deep_gemm/include/deep_gemm/layout/mega_moe.cuh``) to compute
// buffer offsets that both host and device agree on.
//
// The ``__global__`` device kernel itself (per-CTA dispatch / MFMA /
// combine warps) lives in ``impls/gfx950_fp8_fp4_mega_moe.cuh`` and is
// implemented in a follow-up patch; this host file only wires the
// public ``launch_fp8_fp4_mega_moe`` entry point and the binding-layer
// helpers ``get_symm_buffer_size_for_mega_moe`` / ``get_mega_moe_config``.

#include <algorithm>

#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"
#include "primus_turbo/mega_moe.h"

#include "layout/mega_moe.cuh"

namespace primus_turbo {
namespace mega_moe {

namespace {

// Maximum SF padded pool token count across all candidate ``block_m``
// values — so the symmetric buffer is large enough for whichever tile
// the heuristic eventually picks.  Mirrors the
// ``for (int block_m: layout::kCandidateBlockM) { ... std::max ... }``
// loop in DG's ``get_symm_buffer_size_for_mega_moe``.
inline int max_num_padded_sf_pool_tokens(int num_max_pool_tokens) {
    int result = 0;
    for (int i = 0; i < kNumCandidateBlockMs; ++i) {
        result = std::max(result, layout::get_num_padded_sf_pool_tokens<int>(num_max_pool_tokens,
                                                                             kCandidateBlockM[i]));
    }
    return result;
}

} // anonymous namespace

MegaMoEBufferLayout get_symm_buffer_size_for_mega_moe(int num_ranks, int num_experts,
                                                      int num_max_tokens_per_rank, int num_topk,
                                                      int hidden, int intermediate_hidden,
                                                      bool /*use_fp8_dispatch*/) {
    PRIMUS_TURBO_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    PRIMUS_TURBO_CHECK(hidden % 128 == 0, "hidden must be divisible by 128");
    PRIMUS_TURBO_CHECK(intermediate_hidden % 128 == 0,
                       "intermediate_hidden must be divisible by 128");

    const int num_experts_per_rank = num_experts / num_ranks;

    // Use the DG-aligned layout helpers so the pool capacity matches
    // ``deep_gemm::layout::Workspace`` byte-for-byte.
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens<int>(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int num_padded_sf_pool_tokens = max_num_padded_sf_pool_tokens(num_max_pool_tokens);

    // Workspace region: barrier pad, per-expert send/recv counters,
    // arrival masks, dispatch-pulling source indices, combine source
    // metadata.  ``layout::Workspace::get_num_bytes()`` mirrors DG.
    const layout::Workspace workspace(
        /*base=*/nullptr, static_cast<uint32_t>(num_ranks), static_cast<uint32_t>(num_experts),
        static_cast<uint32_t>(num_max_tokens_per_rank), static_cast<uint32_t>(num_topk));
    const int64_t workspace_bytes = static_cast<int64_t>(workspace.get_num_bytes());

    // Element sizes (in bytes) for each region.
    const int64_t fp8_token_bytes    = static_cast<int64_t>(hidden);              // FP8
    const int64_t bf16_token_bytes   = static_cast<int64_t>(hidden) * 2;          // BF16
    const int64_t fp8_inter_bytes    = static_cast<int64_t>(intermediate_hidden); // FP8
    const int64_t fp8_sf_bytes       = static_cast<int64_t>(hidden) / kScaleGroupK;
    const int64_t fp8_inter_sf_bytes = static_cast<int64_t>(intermediate_hidden) / kScaleGroupK;
    const int64_t topk_idx_bytes     = static_cast<int64_t>(num_topk) * sizeof(int64_t);
    const int64_t topk_weights_bytes = static_cast<int64_t>(num_topk) * sizeof(float);

    // Tightly pack regions; alignment requirements (e.g. 1024 B for
    // shared/atomic regions) are applied via the 256-byte bump cursor.
    auto bump = [](int64_t &cursor, int64_t bytes) -> int64_t {
        const int64_t aligned = ALIGN<int64_t>(cursor, 256);
        cursor                = aligned + bytes;
        return aligned;
    };

    MegaMoEBufferLayout out;
    int64_t             cursor = 0;

    // Workspace (signal pad, per-expert counters, etc.).
    out.workspace_offset = bump(cursor, workspace_bytes);

    // Input region (single rank's untransformed tokens).
    out.input_x_offset            = bump(cursor, num_max_tokens_per_rank * fp8_token_bytes);
    out.input_x_sf_offset         = bump(cursor, num_max_tokens_per_rank * fp8_sf_bytes);
    out.input_topk_idx_offset     = bump(cursor, num_max_tokens_per_rank * topk_idx_bytes);
    out.input_topk_weights_offset = bump(cursor, num_max_tokens_per_rank * topk_weights_bytes);

    // L1 pool (FP8 tokens + SF + topk weight column).
    out.l1_pool_x_offset       = bump(cursor, num_max_pool_tokens * fp8_token_bytes);
    out.l1_pool_x_sf_offset    = bump(cursor, num_padded_sf_pool_tokens * fp8_sf_bytes);
    out.l1_pool_weights_offset = bump(cursor, num_max_pool_tokens * sizeof(float));

    // L2 pool (post-SwiGLU FP8 intermediate tokens + SF).
    out.l2_pool_x_offset    = bump(cursor, num_max_pool_tokens * fp8_inter_bytes);
    out.l2_pool_x_sf_offset = bump(cursor, num_padded_sf_pool_tokens * fp8_inter_sf_bytes);

    // Combine buffer (BF16 partial sums for cross-rank reduction).
    out.combine_buffer_offset = bump(cursor, num_topk * num_max_tokens_per_rank * bf16_token_bytes);

    out.total_bytes               = cursor;
    out.num_max_pool_tokens       = num_max_pool_tokens;
    out.num_padded_sf_pool_tokens = num_padded_sf_pool_tokens;
    return out;
}

MegaMoEConfig get_mega_moe_config(int num_ranks, int /*num_experts*/, int num_experts_per_rank,
                                  int /*num_max_tokens_per_rank*/, int num_tokens, int num_topk,
                                  int hidden, int intermediate_hidden,
                                  int num_padded_sf_pool_tokens) {
    // Block tiling defaults that match common AMD MFMA tile sizes; the
    // real heuristic will inspect arch (gfx942 vs gfx950) and token /
    // expert imbalance to pick block_m similar to DeepGEMM.
    MegaMoEConfig config;
    config.block_m       = 128;
    config.block_n       = 128;
    config.block_k       = 128;
    config.load_block_m  = config.block_m / 2;
    config.load_block_n  = config.block_n;
    config.store_block_m = 32;

    config.sf_block_m = config.block_m;
    config.sf_block_n = config.block_n;

    config.num_max_pool_tokens       = 0; // filled in by the caller (from layout)
    config.num_padded_sf_pool_tokens = num_padded_sf_pool_tokens;

    // Trivial wave heuristic: process every local expert in one wave.
    config.num_experts_per_wave = num_experts_per_rank;

    // Pipeline placeholder; the real heuristic will solve for the max
    // number of stages that fit in LDS / scratch.
    config.num_stages = 4;
    config.smem_size  = 0; // computed once we know per-stage tile sizes

    // 4-warp DISPATCH-FOLD layout: dispatch + combine both folded into the
    // 4 MFMA warps (dispatch and epilogue roles removed).  nwarps must divide
    // BLOCK_M.  Block = 0 + 256 + 0 = 256 (4 warps).
    config.num_dispatch_threads     = 0;
    config.num_non_epilogue_threads = 256;
    config.num_epilogue_threads     = 0;

    (void) num_ranks;
    (void) num_tokens;
    (void) num_topk;
    (void) hidden;
    (void) intermediate_hidden;
    return config;
}

void launch_fp8_fp4_mega_moe(const MegaMoEArgs &args) {
    PRIMUS_TURBO_CHECK(args.y_ptr != nullptr, "Mega MoE: output tensor is null");
    PRIMUS_TURBO_CHECK(args.sym_buffer_ptrs != nullptr,
                       "Mega MoE: symmetric buffer pointers are null");
    PRIMUS_TURBO_CHECK(args.num_ranks > 0, "Mega MoE: num_ranks must be > 0");

    // The DG-aligned device kernel (impls/gfx950_fp8_fp4_mega_moe.cuh)
    // is not yet wired into the AOT dispatch table.  The kernel-level
    // C++ API is fully aligned with DeepGEMM; specializations for
    // concrete shape tuples land alongside the gfx950 device kernel
    // implementation.
    PRIMUS_TURBO_ERROR("Mega MoE launcher: device kernel is not yet implemented for shape ",
                       "(max_tokens=", args.num_max_tokens_per_rank, ", hidden=", args.hidden,
                       ", intermediate=", args.intermediate_hidden, ", experts=", args.num_experts,
                       ", topk=", args.num_topk, ", ranks=", args.num_ranks,
                       ").  The kernel-level C++ API (layout::Workspace, layout::SymBuffer, "
                       "sched::MegaMoEScheduler, launch_fp8_fp4_mega_moe_impl) is "
                       "aligned with DeepGEMM; the gfx950 device kernel body will be "
                       "added in a follow-up patch.");
}

} // namespace mega_moe
} // namespace primus_turbo
