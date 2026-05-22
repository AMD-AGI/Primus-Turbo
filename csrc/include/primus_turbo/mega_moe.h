// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

#include "primus_turbo/common.h"

namespace primus_turbo {

//==================================================================
//  Mega MoE Kernel — fused EP dispatch + L1 GEMM + activation +
//  L2 GEMM + EP combine for FP8 / FP4 MoE FFN.
//
//  The mega kernel runs in a persistent, warp-specialized fashion:
//      dispatch warps : pull remote tokens into a shared pool buffer
//      compute warps  : issue MFMA / WMMA for L1 and L2 GEMMs
//      epilogue warps : apply activation, quantize and scatter back
//
//  This header only describes the public surface (params + config +
//  launch declarations) used by the PyTorch bindings.  The actual GPU
//  kernel is intentionally left as a stub and will be implemented in
//  a follow-up patch.
//==================================================================

namespace mega_moe {

// Candidate values of ``block_m`` the heuristic can pick from.  Mirrors
// DeepGEMM's ``deep_gemm::layout::kCandidateBlockM`` so the token-pool
// layout stays compatible across both implementations.
constexpr int kNumCandidateBlockMs                   = 7;
constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
constexpr int kMaxCandidateBlockM                    = 192;
constexpr int kMinCandidateBlockM                    = 8;

// Token granularity constraint enforced by the symmetric memory layout.
// All ``num_max_tokens_per_rank`` values must be aligned to this value.
// Mirrors DG's ``kLCMCandidateBlockM`` (LCM of ``kCandidateBlockM``) so
// every candidate ``block_m`` cleanly divides the per-rank token count.
constexpr int kTokenAlignment = 384;

// Per-expert SF granularity along the K axis (UE8M0 packing factor).
constexpr int kScaleGroupK = 32;

// SF tile is laid out in groups of 128 along the MN axis.
constexpr int kScaleBlockMN = 128;

//------------------------------------------------------------------
//  Heuristic config selected by the host before launching the kernel.
//------------------------------------------------------------------
struct MegaMoEConfig {
    // Block tiling
    int block_m       = 0;
    int block_n       = 0;
    int block_k       = 0;
    int load_block_m  = 0;
    int load_block_n  = 0;
    int store_block_m = 0;

    // SF block sizes (aligned to UTCCP / MFMA granularity).
    int sf_block_m = 0;
    int sf_block_n = 0;

    // Pool capacity (number of tokens the symmetric buffer can hold
    // after fan-out across topk) and the SF padded variant.
    int num_max_pool_tokens       = 0;
    int num_padded_sf_pool_tokens = 0;

    // LDS swizzle mode for activations / weights (analogous to DG's
    // TMA swizzle mode; 0 means no swizzle and the kernel falls back
    // to the default layout).
    int swizzle_acts_mode    = 0;
    int swizzle_weights_mode = 0;

    // Number of experts processed per persistent wave.
    int num_experts_per_wave = 0;

    // Pipeline depth and total shared memory bytes per CTA.
    int num_stages = 0;
    int smem_size  = 0;

    // Thread / warp partition across the persistent CTA.
    int num_dispatch_threads     = 0;
    int num_non_epilogue_threads = 0;
    int num_epilogue_threads     = 0;
};

//------------------------------------------------------------------
//  Symmetric buffer layout.
//
//  All ranks allocate identically-sized buffers and exchange device
//  pointers once during rendezvous.  The single contiguous allocation
//  is then sub-sliced into logical regions:
//
//      [ workspace | input_x | input_x_sf | input_topk_idx |
//        input_topk_weights | l1_pool_x | l1_pool_x_sf |
//        l1_pool_weights | l2_pool_x | l2_pool_x_sf |
//        combine_buffer ]
//
//  The exact offsets are computed by ``get_symm_buffer_size_for_mega_moe``
//  so that both host and device agree on the addressing scheme.
//------------------------------------------------------------------
struct MegaMoEBufferLayout {
    int64_t workspace_offset          = 0;
    int64_t input_x_offset            = 0;
    int64_t input_x_sf_offset         = 0;
    int64_t input_topk_idx_offset     = 0;
    int64_t input_topk_weights_offset = 0;
    int64_t l1_pool_x_offset          = 0;
    int64_t l1_pool_x_sf_offset       = 0;
    int64_t l1_pool_weights_offset    = 0;
    int64_t l2_pool_x_offset          = 0;
    int64_t l2_pool_x_sf_offset       = 0;
    int64_t combine_buffer_offset     = 0;
    int64_t total_bytes               = 0;

    // Cached so the Python layer does not need to recompute them.
    int num_max_pool_tokens       = 0;
    int num_padded_sf_pool_tokens = 0;
};

// Returns the layout (offsets + total bytes) of the symmetric buffer
// for a given MoE configuration.  Mirrors DG's
// ``deep_gemm::mega::get_symm_buffer_size_for_mega_moe`` (DG returns
// ``(num_bytes, slice_callback)``; the slicing is done in our Python
// frontend instead).
MegaMoEBufferLayout get_symm_buffer_size_for_mega_moe(int num_ranks, int num_experts,
                                                      int num_max_tokens_per_rank, int num_topk,
                                                      int hidden, int intermediate_hidden,
                                                      bool use_fp8_dispatch);

// Returns the selected configuration for a given runtime shape.
MegaMoEConfig get_mega_moe_config(int num_ranks, int num_experts, int num_experts_per_rank,
                                  int num_max_tokens_per_rank, int num_tokens, int num_topk,
                                  int hidden, int intermediate_hidden,
                                  int num_padded_sf_pool_tokens);

//------------------------------------------------------------------
//  Runtime arguments forwarded to the mega kernel launcher.
//------------------------------------------------------------------
struct MegaMoEArgs {
    // Output (BF16 tokens).
    void *y_ptr = nullptr;

    // Weights (FP4 packed, MN-major SF in UE8M0).
    const void *l1_weights_ptr    = nullptr;
    const void *l1_weights_sf_ptr = nullptr;
    const void *l2_weights_ptr    = nullptr;
    const void *l2_weights_sf_ptr = nullptr;

    // Optional per-local-expert cumulative receive counter.
    int *cumulative_local_expert_recv_stats = nullptr;

    // Symmetric memory: per-rank base pointers + this rank index.
    const int64_t *sym_buffer_ptrs = nullptr;
    int            num_ranks       = 0;
    int            rank_idx        = 0;

    // Pre-computed buffer layout.
    MegaMoEBufferLayout layout;

    // Runtime shape.
    int num_tokens              = 0;
    int num_max_tokens_per_rank = 0;
    int num_experts             = 0;
    int num_experts_per_rank    = 0;
    int num_topk                = 0;
    int hidden                  = 0;
    int intermediate_hidden     = 0;

    // SF recipe along (M, N, K).  Mirrors DG's ``recipe`` tuple; only
    // ``(1, 1, 32)`` is supported at the moment.
    int recipe_m = 1;
    int recipe_n = 1;
    int recipe_k = kScaleGroupK;

    // Activation options.  Only ``swiglu`` is supported by the
    // skeleton API; the clamp value (``inf`` when unused) and the
    // ``fast_math`` flag map to the device-side activation impl.
    float activation_clamp = 0.0f;
    bool  fast_math        = true;

    // Heuristic config.
    MegaMoEConfig config;

    // Stream to launch on.
    hipStream_t stream = nullptr;
};

// Launches the fused mega MoE kernel (FP8 activations × FP4 weights).
// Mirrors DG's ``sm100_fp8_fp4_mega_moe`` host launcher.  This is
// currently a stub: the real implementation will land alongside the
// device-side kernel.
void launch_fp8_fp4_mega_moe(const MegaMoEArgs &args);

} // namespace mega_moe
} // namespace primus_turbo
