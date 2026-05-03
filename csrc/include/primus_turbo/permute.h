// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/dtype.h"

namespace primus_turbo {

// =============================================================================
// Preprocessing
// =============================================================================

// Pad each per-expert token count up to a multiple of `pad_multiple` and copy
// from int32 (device) to int64 (device).
//
// Inputs / outputs are device pointers of length `num_experts`.
void pad_tokens_per_expert(const int32_t *src, int64_t *dst, int num_experts, int pad_multiple,
                           hipStream_t stream);

// Permute preprocessing kernel configuration. The block size is fixed because
// the kernel template is instantiated with this constant.
struct PermutePreprocessConfig {
    static constexpr int kBlockSize = 512;
};

// Build `row_id_map`, `tokens_per_expert` and `overflow_flag` from the routing
// map using a 4-pass cooperative scan. The kernel uses `grid.sync()` so it
// must be launched via `hipLaunchCooperativeKernel`; this launcher computes a
// safe grid size from device occupancy and dispatches it for the caller.
//
// Shapes (all device pointers):
//   routing_map               : [num_dispatched_tokens, num_of_local_experts] (bool)
//   num_dispatched_tokens_ptr : scalar int
//   workspace_1               : [rows_workspace_1, num_of_local_experts] (int)
//   workspace_2               : [rows_workspace_2, num_of_local_experts] (int)
//   tokens_per_expert         : [num_of_local_experts] (int32)
//   row_id_map                : [num_dispatched_tokens + pad_multiple, num_of_local_experts] (int)
//   overflow_flag             : scalar int
//
// `num_permuted_tokens < 0` is treated as "no cap".
void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                                  int *workspace_2, int rows_workspace_2, int pad_multiple,
                                  int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                  int64_t num_permuted_tokens, hipStream_t stream);

// =============================================================================
// Permute / Unpermute (data movement)
// =============================================================================

// Block configuration shared by permute / unpermute. The kernels are
// vectorised over `THREADS_PER_TOKEN` lanes and pack `block_size /
// THREADS_PER_TOKEN` tokens per CTA.
struct PermuteKernelConfig {
    static constexpr int kBlockSize       = 512;
    static constexpr int kThreadsPerToken = 128;
    static constexpr int kTokensPerBlock  = kBlockSize / kThreadsPerToken;
};

// Permute tokens into expert-grouped order using `row_id_map` (produced by
// `permute_preprocessing_launch`).
//
//   tokens                   : [num_dispatched_tokens, hidden_size]  (DType)
//   permuted_tokens          : [num_permuted_tokens,    hidden_size] (DType, output)
//   scaling_factor           : [num_dispatched_tokens, scales_per_token] or nullptr
//   permuted_scaling_factor  : [num_permuted_tokens,    scales_per_token] or nullptr
//   probs                    : [num_dispatched_tokens, num_of_local_experts * num_ranks_per_node]
//                              or nullptr
//   permuted_probs           : [num_permuted_tokens]   or nullptr
//   row_id_map               : [num_dispatched_tokens + pad_multiple, num_of_local_experts] (int)
//
// `hidden_size` must be a multiple of `sizeof(float4)/sizeof(DType)` so the
// kernel can use float4 vectorised loads/stores.
template <typename DType, typename ProbType, typename ScalarType>
void permute_impl(const DType *tokens, DType *permuted_tokens, const ScalarType *scaling_factor,
                  ScalarType *permuted_scaling_factor, const ProbType *probs,
                  ProbType *permuted_probs, const int *row_id_map,
                  const int *num_dispatched_tokens_ptr, int pad_multiple, int num_of_local_experts,
                  int hidden_size, int scales_per_token, int local_rank, int num_ranks_per_node,
                  int grid_size, hipStream_t stream);

// Reduce permuted tokens back to per-source rows by accumulating values from
// all experts a given source token was routed to. `DType` is currently
// restricted to `bfloat16` (16-bit accum-via-float).
template <typename DType, typename ProbType>
void unpermute_impl(const DType *permuted_tokens, DType *tokens, const ProbType *permuted_probs,
                    ProbType *probs, const int *row_id_map, const int *num_dispatched_tokens_ptr,
                    int num_of_local_experts, int hidden_size, int local_rank,
                    int num_ranks_per_node, int grid_size, hipStream_t stream);

} // namespace primus_turbo
