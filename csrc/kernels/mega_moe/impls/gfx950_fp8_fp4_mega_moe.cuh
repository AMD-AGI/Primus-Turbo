// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// gfx950 (MI355X) fused FP8 × FP4 mega-MoE kernel — API surface only.
//
// Mirrors the host-launchable template signature of DeepGEMM's
// ``deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh``.
// The actual ``__global__`` device kernel body is not part of this
// alignment pass — only the C++ launcher API is exposed here so the
// JIT / AOT host wrapper, the binding layer and Python kernels can be
// implemented and tested against a stable surface.
//
// The launcher is declared here as a template; specializations land
// in a follow-up patch that wires it into the persistent warp-
// specialized device kernel (dispatch warps → L1 MFMA → SwiGLU UE8M0
// quant → L2 MFMA → combine).

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

#include "../layout/mega_moe.cuh"
#include "../layout/sym_buffer.cuh"

namespace primus_turbo {
namespace mega_moe {
namespace impls {

// ---------------------------------------------------------------------
//  Compile-time descriptor of a single mega-MoE specialization.
//  Lifted 1:1 from DG's ``sm100_fp8_fp4_mega_moe_impl`` template
//  parameter list, with the addition of an explicit ``arch`` tag so
//  multiple AMD specializations (gfx942 / gfx950 / ...) can coexist.
// ---------------------------------------------------------------------

enum class MegaMoEArch : uint32_t {
    Unknown = 0,
    Gfx942  = 942,
    Gfx950  = 950,
};

// ---------------------------------------------------------------------
//  Host-side launcher template.
//
//  Signature mirrors DG so that the calling host wrapper (mega_moe.cu)
//  can dispatch to the AMD implementation by simply changing the
//  namespace / arch tag.  The body is intentionally a stub at this
//  alignment stage; it is filled in by the gfx950 device kernel patch.
// ---------------------------------------------------------------------

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
                                        hipStream_t stream) {}

} // namespace impls
} // namespace mega_moe
} // namespace primus_turbo
