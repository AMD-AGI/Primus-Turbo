// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Mega MoE kernel — fused EP dispatch + L1 GEMM + activation + L2
// GEMM + EP combine.  This file currently provides a stub launcher;
// the actual warp-specialized device kernel will be implemented in
// a follow-up patch and slotted in via ``launch_fp8_mega_moe``.

#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"
#include "primus_turbo/mega_moe.h"

namespace primus_turbo {
namespace mega_moe {

namespace {

// Smallest token-pool capacity needed to absorb all top-k fan-out
// from every rank without spilling.  Mirrors the layout helper in
// DeepGEMM's ``layout/mega_moe.cuh`` but keeps the formula on the
// host side until the device-side scheduler is wired up.
inline int compute_num_max_pool_tokens(int num_ranks, int num_max_tokens_per_rank, int num_topk,
                                       int num_experts_per_rank) {
    // Worst case: every token in the global batch is routed top-k
    // times to this rank's experts.
    const int global_max_tokens = num_ranks * num_max_tokens_per_rank;
    const int worst_recv        = global_max_tokens * num_topk;
    // Cap at the local expert pool capacity (top-k * tokens / experts).
    PRIMUS_TURBO_CHECK(num_experts_per_rank > 0, "num_experts_per_rank must be > 0");
    return ALIGN<int>(worst_recv, kTokenAlignment);
}

inline int align_padded_sf_pool_tokens(int num_max_pool_tokens) {
    // SF tile size requires 4-element alignment along the token axis
    // (UE8M0 is packed 4-wide).
    constexpr int kSfTokenAlignment = 4;
    return ALIGN<int>(num_max_pool_tokens, kSfTokenAlignment);
}

} // anonymous namespace

MegaMoEBufferLayout get_symm_buffer_layout(int num_ranks, int num_experts,
                                           int num_max_tokens_per_rank, int num_topk, int hidden,
                                           int intermediate_hidden, bool /*use_fp8_dispatch*/) {
    PRIMUS_TURBO_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    PRIMUS_TURBO_CHECK(hidden % 128 == 0, "hidden must be divisible by 128");
    PRIMUS_TURBO_CHECK(intermediate_hidden % 128 == 0,
                       "intermediate_hidden must be divisible by 128");

    const int num_experts_per_rank = num_experts / num_ranks;
    const int num_max_pool_tokens  = compute_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank,
                                                                 num_topk, num_experts_per_rank);
    const int num_padded_sf_pool_tokens = align_padded_sf_pool_tokens(num_max_pool_tokens);

    // Element sizes (in bytes) for each region.
    const int64_t fp8_token_bytes    = static_cast<int64_t>(hidden);              // FP8
    const int64_t bf16_token_bytes   = static_cast<int64_t>(hidden) * 2;          // BF16
    const int64_t fp8_inter_bytes    = static_cast<int64_t>(intermediate_hidden); // FP8
    const int64_t fp8_sf_bytes       = static_cast<int64_t>(hidden) / kScaleGroupK;
    const int64_t fp8_inter_sf_bytes = static_cast<int64_t>(intermediate_hidden) / kScaleGroupK;
    const int64_t topk_idx_bytes     = static_cast<int64_t>(num_topk) * sizeof(int64_t);
    const int64_t topk_weights_bytes = static_cast<int64_t>(num_topk) * sizeof(float);

    // Tightly pack regions; alignment requirements (e.g. 1024 B for
    // shared/atomic regions) will be applied during a future refactor.
    auto bump = [](int64_t &cursor, int64_t bytes) -> int64_t {
        const int64_t aligned = ALIGN<int64_t>(cursor, 256);
        cursor                = aligned + bytes;
        return aligned;
    };

    MegaMoEBufferLayout out;
    int64_t             cursor = 0;

    // Workspace (signal pad, per-expert counters, etc.).
    constexpr int64_t kWorkspaceBytes = 1 << 20; // 1 MiB placeholder
    out.workspace_offset              = bump(cursor, kWorkspaceBytes);

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

    // Pipeline placeholder; real heuristic will solve for max stages
    // that fit in LDS / scratch.
    config.num_stages = 4;
    config.smem_size  = 0; // computed once we know per-stage tile sizes

    config.num_dispatch_threads     = 128;
    config.num_non_epilogue_threads = 128;
    config.num_epilogue_threads     = 256;

    (void) num_ranks;
    (void) num_tokens;
    (void) num_topk;
    (void) hidden;
    (void) intermediate_hidden;
    return config;
}

void launch_fp8_mega_moe(const MegaMoEArgs &args) {
    // TODO: launch the warp-specialized mega MoE kernel here.  The
    // skeleton exists so that the PyTorch binding and Python frontend
    // can be exercised end-to-end (return NotImplementedError) without
    // having to stub them out individually.
    PRIMUS_TURBO_CHECK(args.y_ptr != nullptr, "Mega MoE: output tensor is null");
    PRIMUS_TURBO_CHECK(args.sym_buffer_ptrs != nullptr,
                       "Mega MoE: symmetric buffer pointers are null");
    PRIMUS_TURBO_CHECK(args.num_ranks > 0, "Mega MoE: num_ranks must be > 0");

    // Intentionally a no-op so callers can wire up the full pipeline
    // (allocations, layout, distributed setup) before the kernel
    // lands.  Real launch will look like:
    //
    //     hipLaunchKernelGGL(mega_moe_kernel,
    //                        dim3(num_persistent_ctas), dim3(num_threads),
    //                        args.config.smem_size, args.stream,
    //                        device_args);
    (void) args;
}

} // namespace mega_moe
} // namespace primus_turbo
