// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "jax/deep_ep/deep_ep.h"
#include "jax/extensions.h"
#include "primus_turbo/common.h"
#include "primus_turbo/deep_ep/config.hpp"

namespace primus_turbo::jax::deep_ep {

constexpr int64_t kInProcLaunchMode     = 0;
constexpr int64_t kPerProcessLaunchMode = 1;

int64_t get_hidden_bytes(ffi::AnyBuffer x) {
    PRIMUS_TURBO_CHECK(x.dimensions().size() == 2);
    return x.dimensions()[1] * std::max(ffi::ByteWidth(x.element_type()), static_cast<size_t>(2));
}

ffi::Error MoEDispatchFFI(hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> x_scales,
                          ffi::Buffer<ffi::S32> topk_idx, ffi::Buffer<ffi::F32> topk_weights,
                          /* attributes */
                          int64_t num_experts, int64_t expert_alignment, int64_t num_worst_tokens,
                          int64_t ep_size, int64_t launch_mode,
                          /*dispatch config*/
                          int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
                          int64_t num_max_nvl_chunked_recv_tokens,
                          int64_t num_max_rdma_chunked_send_tokens,
                          int64_t num_max_rdma_chunked_recv_tokens,
                          /* dispatched outputs */
                          ffi::Result<ffi::AnyBuffer>         recv_x,
                          ffi::Result<ffi::Buffer<ffi::F32>>  recv_x_scales,
                          ffi::Result<ffi::Buffer<ffi::S32>>  recv_topk_idx,
                          ffi::Result<ffi::Buffer<ffi::F32>>  recv_topk_weights,
                          ffi::Result<ffi::Buffer<ffi::PRED>> is_token_in_rank,
                          ffi::Result<ffi::Buffer<ffi::S32>>  num_tokens_per_rank,
                          ffi::Result<ffi::Buffer<ffi::S32>>  num_tokens_per_expert,
                          /* dispatch handle for cached mode*/
                          ffi::Result<ffi::Buffer<ffi::S32>> rank_prefix_matrix,
                          ffi::Result<ffi::Buffer<ffi::S32>> channel_prefix_matrix,
                          ffi::Result<ffi::Buffer<ffi::S32>> recv_channel_prefix_matrix,
                          ffi::Result<ffi::Buffer<ffi::S32>> recv_src_idx,
                          ffi::Result<ffi::Buffer<ffi::S32>> send_head) {
    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    int num_ranks = -1;
    int rank      = -1;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&rank));
    PRIMUS_TURBO_CHECK_HIP(hipGetDeviceCount(&num_ranks));
    if (launch_mode != kInProcLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_dispatch_inproc received a non-inproc mode");
    }
    if (num_ranks != ep_size) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "moe_dispatch_inproc world-size mismatch between lowering attrs and visible devices");
    }

    auto    hidden_bytes = get_hidden_bytes(x);
    Buffer *buffer       = get_buffer(rank, num_ranks, hidden_bytes, cfg);

    buffer->DispatchLayout(stream, topk_idx, static_cast<int>(num_experts), num_tokens_per_rank,
                           std::nullopt, num_tokens_per_expert, is_token_in_rank);

    bool is_fp8 = x_scales.element_count() > 0;
    buffer->IntranodeDispatch(
        stream, x, is_fp8 ? std::make_optional(x_scales) : std::nullopt, topk_idx, topk_weights,
        *num_tokens_per_rank, *is_token_in_rank, *num_tokens_per_expert, 0, std::nullopt,
        std::nullopt, expert_alignment, num_worst_tokens, cfg, recv_x, recv_x_scales, recv_topk_idx,
        recv_topk_weights, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix,
        recv_src_idx, send_head);

    return ffi::Error::Success();
}

ffi::Error MoECachedDispatchFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> x_scales,
    ffi::Buffer<ffi::PRED> is_token_in_rank, ffi::Buffer<ffi::S32> cached_rank_prefix_matrix,
    ffi::Buffer<ffi::S32> cached_channel_prefix_matrix, int64_t num_recv_tokens,
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t ep_size, int64_t launch_mode,
    int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
    int64_t num_max_rdma_chunked_recv_tokens,
    /* dispatch handle for cached mode*/
    ffi::Result<ffi::AnyBuffer> recv_x, ffi::Result<ffi::Buffer<ffi::F32>> recv_x_scales,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_src_idx, ffi::Result<ffi::Buffer<ffi::S32>> send_head) {
    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    int num_ranks = -1;
    int rank      = -1;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&rank));
    PRIMUS_TURBO_CHECK_HIP(hipGetDeviceCount(&num_ranks));
    if (launch_mode != kInProcLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_cached_dispatch_inproc received a non-inproc mode");
    }
    if (num_ranks != ep_size) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "moe_cached_dispatch_inproc world-size mismatch between lowering attrs and visible "
            "devices");
    }

    auto    hidden_bytes = get_hidden_bytes(x);
    Buffer *buffer       = get_buffer(rank, num_ranks, hidden_bytes, cfg);
    bool    is_fp8       = x_scales.element_count() > 0;
    buffer->IntranodeDispatch(stream, x, is_fp8 ? std::make_optional(x_scales) : std::nullopt,
                              std::nullopt, std::nullopt, std::nullopt, is_token_in_rank,
                              std::nullopt, num_recv_tokens, cached_rank_prefix_matrix,
                              cached_channel_prefix_matrix, expert_alignment, num_worst_tokens, cfg,
                              recv_x, recv_x_scales, std::nullopt, std::nullopt, std::nullopt,
                              std::nullopt, recv_channel_prefix_matrix, recv_src_idx, send_head);

    return ffi::Error::Success();
}

ffi::Error
MoECombineFFI(hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> topk_weights,
              ffi::AnyBuffer bias_0, ffi::AnyBuffer bias_1, ffi::Buffer<ffi::S32> src_idx,
              ffi::Buffer<ffi::S32> rank_prefix_matrix, ffi::Buffer<ffi::S32> channel_prefix_matrix,
              ffi::Buffer<ffi::S32> send_head, int64_t ep_size, int64_t launch_mode,
              int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
              int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
              int64_t num_max_rdma_chunked_recv_tokens, ffi::Result<ffi::AnyBuffer> recv_x,
              ffi::Result<ffi::Buffer<ffi::F32>> recv_topk_weights) {
    int num_ranks = -1;
    int rank      = -1;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&rank));
    PRIMUS_TURBO_CHECK_HIP(hipGetDeviceCount(&num_ranks));
    if (launch_mode != kInProcLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_combine_inproc received a non-inproc mode");
    }
    if (num_ranks != ep_size) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_combine_inproc world-size mismatch between lowering attrs and "
                          "visible devices");
    }

    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    auto    hidden_bytes = get_hidden_bytes(x);
    Buffer *buffer       = get_buffer(rank, num_ranks, hidden_bytes, cfg);

    bool has_weights = topk_weights.element_count() > 0;
    bool has_bias0   = bias_0.element_count() > 0;
    bool has_bias1   = bias_1.element_count() > 0;

    buffer->IntranodeCombine(stream, x,
                             has_weights ? std::make_optional(topk_weights) : std::nullopt,
                             has_bias0 ? std::make_optional(bias_0) : std::nullopt,
                             has_bias1 ? std::make_optional(bias_1) : std::nullopt, src_idx,
                             rank_prefix_matrix, channel_prefix_matrix, send_head, cfg, recv_x,
                             has_weights ? std::make_optional(recv_topk_weights) : std::nullopt);

    return ffi::Error::Success();
}

ffi::Error MoEDispatchPerProcessFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> x_scales,
    ffi::Buffer<ffi::S32> topk_idx, ffi::Buffer<ffi::F32> topk_weights, int64_t num_experts,
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t ep_size, int64_t launch_mode,
    int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
    int64_t num_max_rdma_chunked_recv_tokens, ffi::Result<ffi::AnyBuffer> recv_x,
    ffi::Result<ffi::Buffer<ffi::F32>>  recv_x_scales,
    ffi::Result<ffi::Buffer<ffi::S32>>  recv_topk_idx,
    ffi::Result<ffi::Buffer<ffi::F32>>  recv_topk_weights,
    ffi::Result<ffi::Buffer<ffi::PRED>> is_token_in_rank,
    ffi::Result<ffi::Buffer<ffi::S32>>  num_tokens_per_rank,
    ffi::Result<ffi::Buffer<ffi::S32>>  num_tokens_per_expert,
    ffi::Result<ffi::Buffer<ffi::S32>>  rank_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>>  channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>>  recv_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_src_idx, ffi::Result<ffi::Buffer<ffi::S32>> send_head) {
    if (launch_mode != kPerProcessLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_dispatch_per_process received a non-per_process launch mode");
    }
    Buffer *buffer = get_per_process_buffer();
    if (buffer == nullptr || !buffer->is_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "per_process buffer not initialized; call ensure_deepep_runtime first");
    }
    if (buffer->num_ranks() != ep_size) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_dispatch_per_process ep_size mismatch with buffer num_ranks");
    }

    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    buffer->DispatchLayout(stream, topk_idx, static_cast<int>(num_experts), num_tokens_per_rank,
                           std::nullopt, num_tokens_per_expert, is_token_in_rank);

    bool is_fp8 = x_scales.element_count() > 0;
    buffer->IntranodeDispatch(
        stream, x, is_fp8 ? std::make_optional(x_scales) : std::nullopt, topk_idx, topk_weights,
        *num_tokens_per_rank, *is_token_in_rank, *num_tokens_per_expert, 0, std::nullopt,
        std::nullopt, expert_alignment, num_worst_tokens, cfg, recv_x, recv_x_scales, recv_topk_idx,
        recv_topk_weights, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix,
        recv_src_idx, send_head);

    return ffi::Error::Success();
}

ffi::Error MoECachedDispatchPerProcessFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> x_scales,
    ffi::Buffer<ffi::PRED> is_token_in_rank, ffi::Buffer<ffi::S32> cached_rank_prefix_matrix,
    ffi::Buffer<ffi::S32> cached_channel_prefix_matrix, int64_t num_recv_tokens,
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t ep_size, int64_t launch_mode,
    int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
    int64_t num_max_rdma_chunked_recv_tokens, ffi::Result<ffi::AnyBuffer> recv_x,
    ffi::Result<ffi::Buffer<ffi::F32>> recv_x_scales,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_src_idx, ffi::Result<ffi::Buffer<ffi::S32>> send_head) {
    if (launch_mode != kPerProcessLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_cached_dispatch_per_process received a non-per_process launch mode");
    }
    Buffer *buffer = get_per_process_buffer();
    if (buffer == nullptr || !buffer->is_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "per_process buffer not initialized; call ensure_deepep_runtime first");
    }
    if (buffer->num_ranks() != ep_size) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_cached_dispatch_per_process ep_size mismatch with buffer num_ranks");
    }

    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    bool is_fp8 = x_scales.element_count() > 0;
    buffer->IntranodeDispatch(stream, x, is_fp8 ? std::make_optional(x_scales) : std::nullopt,
                              std::nullopt, std::nullopt, std::nullopt, is_token_in_rank,
                              std::nullopt, num_recv_tokens, cached_rank_prefix_matrix,
                              cached_channel_prefix_matrix, expert_alignment, num_worst_tokens, cfg,
                              recv_x, recv_x_scales, std::nullopt, std::nullopt, std::nullopt,
                              std::nullopt, recv_channel_prefix_matrix, recv_src_idx, send_head);

    return ffi::Error::Success();
}

ffi::Error MoECombinePerProcessFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> topk_weights, ffi::AnyBuffer bias_0,
    ffi::AnyBuffer bias_1, ffi::Buffer<ffi::S32> src_idx, ffi::Buffer<ffi::S32> rank_prefix_matrix,
    ffi::Buffer<ffi::S32> channel_prefix_matrix, ffi::Buffer<ffi::S32> send_head, int64_t ep_size,
    int64_t launch_mode, int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
    int64_t num_max_rdma_chunked_recv_tokens, ffi::Result<ffi::AnyBuffer> recv_x,
    ffi::Result<ffi::Buffer<ffi::F32>> recv_topk_weights) {
    if (launch_mode != kPerProcessLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_combine_per_process received a non-per_process launch mode");
    }
    Buffer *buffer = get_per_process_buffer();
    if (buffer == nullptr || !buffer->is_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "per_process buffer not initialized; call ensure_deepep_runtime first");
    }
    if (buffer->num_ranks() != ep_size) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "moe_combine_per_process ep_size mismatch with buffer num_ranks");
    }

    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    bool has_weights = topk_weights.element_count() > 0;
    bool has_bias0   = bias_0.element_count() > 0;
    bool has_bias1   = bias_1.element_count() > 0;

    buffer->IntranodeCombine(stream, x,
                             has_weights ? std::make_optional(topk_weights) : std::nullopt,
                             has_bias0 ? std::make_optional(bias_0) : std::nullopt,
                             has_bias1 ? std::make_optional(bias_1) : std::nullopt, src_idx,
                             rank_prefix_matrix, channel_prefix_matrix, send_head, cfg, recv_x,
                             has_weights ? std::make_optional(recv_topk_weights) : std::nullopt);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoECachedDispatchHandler, MoECachedDispatchFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()           // stream
        .Arg<ffi::AnyBuffer>()                             // x
        .Arg<ffi::Buffer<ffi::F32>>()                      // x_scales
        .Arg<ffi::Buffer<ffi::PRED>>()                     // is_token_in_rank
        .Arg<ffi::Buffer<ffi::S32>>()                      // cached_rank_prefix_matrix
        .Arg<ffi::Buffer<ffi::S32>>()                      // cached_channel_prefix_matrix
        .Attr<int64_t>("num_recv_tokens")                  // num_recv_tokens
        .Attr<int64_t>("expert_alignment")                 // expert_alignment
        .Attr<int64_t>("num_worst_tokens")                 // num_worst_tokens
        .Attr<int64_t>("ep_size")                          // ep_size
        .Attr<int64_t>("launch_mode")                      // launch_mode
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_x_scales
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_channel_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_src_idx
        .Ret<ffi::Buffer<ffi::S32>>()                      // send_head
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoEDispatchHandler, MoEDispatchFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()           // stream
        .Arg<ffi::AnyBuffer>()                             // x
        .Arg<ffi::Buffer<ffi::F32>>()                      // x_scales
        .Arg<ffi::Buffer<ffi::S32>>()                      // topk_idx
        .Arg<ffi::Buffer<ffi::F32>>()                      // topk_weights
        .Attr<int64_t>("num_experts")                      // num_experts
        .Attr<int64_t>("expert_alignment")                 // expert_alignment
        .Attr<int64_t>("num_worst_tokens")                 // num_worst_tokens
        .Attr<int64_t>("ep_size")                          // ep_size
        .Attr<int64_t>("launch_mode")                      // launch_mode
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_x_scales
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_topk_idx
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_topk_weights
        .Ret<ffi::Buffer<ffi::PRED>>()                     // is_token_in_rank
        .Ret<ffi::Buffer<ffi::S32>>()                      // num_tokens_per_rank
        .Ret<ffi::Buffer<ffi::S32>>()                      // num_tokens_per_expert
        .Ret<ffi::Buffer<ffi::S32>>()                      // rank_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // channel_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_channel_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_src_idx
        .Ret<ffi::Buffer<ffi::S32>>()                      // send_head
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoECombineHandler, MoECombineFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()           // stream
        .Arg<ffi::AnyBuffer>()                             // x
        .Arg<ffi::Buffer<ffi::F32>>()                      // topk_weights
        .Arg<ffi::AnyBuffer>()                             // bias_0
        .Arg<ffi::AnyBuffer>()                             // bias_1
        .Arg<ffi::Buffer<ffi::S32>>()                      // src_idx
        .Arg<ffi::Buffer<ffi::S32>>()                      // rank_prefix_matrix
        .Arg<ffi::Buffer<ffi::S32>>()                      // channel_prefix_matrix
        .Arg<ffi::Buffer<ffi::S32>>()                      // send_head
        .Attr<int64_t>("ep_size")                          // ep_size
        .Attr<int64_t>("launch_mode")                      // launch_mode
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_topk_weights
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoECachedDispatchPerProcessHandler, MoECachedDispatchPerProcessFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()           // stream
        .Arg<ffi::AnyBuffer>()                             // x
        .Arg<ffi::Buffer<ffi::F32>>()                      // x_scales
        .Arg<ffi::Buffer<ffi::PRED>>()                     // is_token_in_rank
        .Arg<ffi::Buffer<ffi::S32>>()                      // cached_rank_prefix_matrix
        .Arg<ffi::Buffer<ffi::S32>>()                      // cached_channel_prefix_matrix
        .Attr<int64_t>("num_recv_tokens")                  // num_recv_tokens
        .Attr<int64_t>("expert_alignment")                 // expert_alignment
        .Attr<int64_t>("num_worst_tokens")                 // num_worst_tokens
        .Attr<int64_t>("ep_size")                          // ep_size
        .Attr<int64_t>("launch_mode")                      // launch_mode
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_x_scales
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_channel_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_src_idx
        .Ret<ffi::Buffer<ffi::S32>>()                      // send_head
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoEDispatchPerProcessHandler, MoEDispatchPerProcessFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()           // stream
        .Arg<ffi::AnyBuffer>()                             // x
        .Arg<ffi::Buffer<ffi::F32>>()                      // x_scales
        .Arg<ffi::Buffer<ffi::S32>>()                      // topk_idx
        .Arg<ffi::Buffer<ffi::F32>>()                      // topk_weights
        .Attr<int64_t>("num_experts")                      // num_experts
        .Attr<int64_t>("expert_alignment")                 // expert_alignment
        .Attr<int64_t>("num_worst_tokens")                 // num_worst_tokens
        .Attr<int64_t>("ep_size")                          // ep_size
        .Attr<int64_t>("launch_mode")                      // launch_mode
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_x_scales
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_topk_idx
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_topk_weights
        .Ret<ffi::Buffer<ffi::PRED>>()                     // is_token_in_rank
        .Ret<ffi::Buffer<ffi::S32>>()                      // num_tokens_per_rank
        .Ret<ffi::Buffer<ffi::S32>>()                      // num_tokens_per_expert
        .Ret<ffi::Buffer<ffi::S32>>()                      // rank_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // channel_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_channel_prefix_matrix
        .Ret<ffi::Buffer<ffi::S32>>()                      // recv_src_idx
        .Ret<ffi::Buffer<ffi::S32>>()                      // send_head
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoECombinePerProcessHandler, MoECombinePerProcessFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()           // stream
        .Arg<ffi::AnyBuffer>()                             // x
        .Arg<ffi::Buffer<ffi::F32>>()                      // topk_weights
        .Arg<ffi::AnyBuffer>()                             // bias_0
        .Arg<ffi::AnyBuffer>()                             // bias_1
        .Arg<ffi::Buffer<ffi::S32>>()                      // src_idx
        .Arg<ffi::Buffer<ffi::S32>>()                      // rank_prefix_matrix
        .Arg<ffi::Buffer<ffi::S32>>()                      // channel_prefix_matrix
        .Arg<ffi::Buffer<ffi::S32>>()                      // send_head
        .Attr<int64_t>("ep_size")                          // ep_size
        .Attr<int64_t>("launch_mode")                      // launch_mode
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_topk_weights
);

// ===========================================================================
//  Internode per-process handlers
// ===========================================================================

ffi::Error MoEInternodeDispatchPerProcessFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> x_scales,
    ffi::Buffer<ffi::S32> topk_idx, ffi::Buffer<ffi::F32> topk_weights, int64_t num_experts,
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t ep_size, int64_t launch_mode,
    int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
    int64_t num_max_rdma_chunked_recv_tokens, ffi::Result<ffi::AnyBuffer> recv_x,
    ffi::Result<ffi::Buffer<ffi::F32>>  recv_x_scales,
    ffi::Result<ffi::Buffer<ffi::S32>>  recv_topk_idx,
    ffi::Result<ffi::Buffer<ffi::F32>>  recv_topk_weights,
    ffi::Result<ffi::Buffer<ffi::PRED>> is_token_in_rank,
    ffi::Result<ffi::Buffer<ffi::S32>>  num_tokens_per_rank,
    ffi::Result<ffi::Buffer<ffi::S32>>  num_tokens_per_rdma_rank,
    ffi::Result<ffi::Buffer<ffi::S32>>  num_tokens_per_expert,
    ffi::Result<ffi::Buffer<ffi::S32>>  rdma_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>>  recv_rdma_rank_prefix_sum,
    ffi::Result<ffi::Buffer<ffi::S32>>  gbl_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>>  recv_gbl_rank_prefix_sum,
    ffi::Result<ffi::Buffer<ffi::U8>>   recv_src_meta,
    ffi::Result<ffi::Buffer<ffi::S32>>  recv_rdma_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>>  recv_gbl_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>>  send_rdma_head,
    ffi::Result<ffi::Buffer<ffi::S32>>  send_nvl_head) {

    if (launch_mode != kPerProcessLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode dispatch received a non-per_process launch mode");
    }
    Buffer *buffer = get_per_process_buffer();
    if (buffer == nullptr || !buffer->is_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "per_process buffer not initialized");
    }
    if (buffer->num_ranks() != ep_size) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode dispatch ep_size mismatch with buffer num_ranks");
    }
    if (!buffer->is_internode_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode dispatch requires an initialized RDMA buffer");
    }

    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    buffer->DispatchLayout(stream, topk_idx, static_cast<int>(num_experts), num_tokens_per_rank,
                           num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank);

    bool is_fp8 = x_scales.element_count() > 0;
    buffer->InternodeDispatch(
        stream, x, is_fp8 ? std::make_optional(x_scales) : std::nullopt, topk_idx, topk_weights,
        *num_tokens_per_rank, *num_tokens_per_rdma_rank, *is_token_in_rank, *num_tokens_per_expert,
        0, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt, expert_alignment,
        num_worst_tokens, cfg, recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
        is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert, rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
        recv_src_meta, recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix,
        send_rdma_head, send_nvl_head);

    return ffi::Error::Success();
}

ffi::Error MoEInternodeCachedDispatchPerProcessFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> x_scales,
    ffi::Buffer<ffi::PRED> is_token_in_rank,
    ffi::Buffer<ffi::S32>  cached_rdma_channel_prefix_matrix,
    ffi::Buffer<ffi::S32>  cached_recv_rdma_rank_prefix_sum,
    ffi::Buffer<ffi::S32>  cached_gbl_channel_prefix_matrix,
    ffi::Buffer<ffi::S32> cached_recv_gbl_rank_prefix_sum, int64_t num_recv_tokens,
    int64_t num_rdma_recv_tokens, int64_t expert_alignment, int64_t num_worst_tokens,
    int64_t ep_size, int64_t launch_mode, int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
    int64_t num_max_rdma_chunked_recv_tokens, ffi::Result<ffi::AnyBuffer> recv_x,
    ffi::Result<ffi::Buffer<ffi::F32>> recv_x_scales,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_rdma_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_gbl_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>> send_rdma_head,
    ffi::Result<ffi::Buffer<ffi::S32>> send_nvl_head) {

    if (launch_mode != kPerProcessLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode cached dispatch received a non-per_process launch mode");
    }
    Buffer *buffer = get_per_process_buffer();
    if (buffer == nullptr || !buffer->is_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "per_process buffer not initialized");
    }
    if (buffer->num_ranks() != ep_size) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode cached dispatch ep_size mismatch with buffer num_ranks");
    }
    if (!buffer->is_internode_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode cached dispatch requires an initialized RDMA buffer");
    }

    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    bool is_fp8 = x_scales.element_count() > 0;
    buffer->InternodeDispatch(
        stream, x, is_fp8 ? std::make_optional(x_scales) : std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, is_token_in_rank, std::nullopt, num_recv_tokens,
        num_rdma_recv_tokens, cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum,
        cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum, expert_alignment,
        num_worst_tokens, cfg, recv_x, recv_x_scales, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix,
        send_rdma_head, send_nvl_head);

    return ffi::Error::Success();
}

ffi::Error MoEInternodeCombinePerProcessFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> topk_weights,
    ffi::Buffer<ffi::U8> src_meta, ffi::Buffer<ffi::PRED> is_combined_token_in_rank,
    ffi::Buffer<ffi::S32> rdma_channel_prefix_matrix, ffi::Buffer<ffi::S32> rdma_rank_prefix_sum,
    ffi::Buffer<ffi::S32> gbl_channel_prefix_matrix, ffi::Buffer<ffi::S32> gbl_rank_prefix_sum,
    ffi::Buffer<ffi::S32> combined_rdma_head, ffi::Buffer<ffi::S32> combined_nvl_head,
    int64_t ep_size, int64_t launch_mode, int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
    int64_t num_max_rdma_chunked_recv_tokens, ffi::Result<ffi::AnyBuffer> combined_x,
    ffi::Result<ffi::Buffer<ffi::F32>> combined_topk_weights) {

    if (launch_mode != kPerProcessLaunchMode) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode combine received a non-per_process launch mode");
    }
    Buffer *buffer = get_per_process_buffer();
    if (buffer == nullptr || !buffer->is_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "per_process buffer not initialized");
    }
    if (buffer->num_ranks() != ep_size) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode combine ep_size mismatch with buffer num_ranks");
    }
    if (!buffer->is_internode_available()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "internode combine requires an initialized RDMA buffer");
    }

    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    bool has_weights             = topk_weights.element_count() > 0;
    bool has_gbl_rank_prefix_sum = gbl_rank_prefix_sum.element_count() > 0;

    buffer->InternodeCombine(
        stream, x, has_weights ? std::make_optional(topk_weights) : std::nullopt, src_meta,
        is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        has_gbl_rank_prefix_sum ? std::make_optional(gbl_rank_prefix_sum) : std::nullopt,
        combined_rdma_head, combined_nvl_head, cfg, combined_x,
        has_weights ? std::make_optional(combined_topk_weights) : std::nullopt);

    return ffi::Error::Success();
}

// ===========================================================================
//  Internode handler symbol definitions
// ===========================================================================

XLA_FFI_DEFINE_HANDLER_SYMBOL(MoEInternodeDispatchPerProcessHandler,
                              MoEInternodeDispatchPerProcessFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // x
                                  .Arg<ffi::Buffer<ffi::F32>>()            // x_scales
                                  .Arg<ffi::Buffer<ffi::S32>>()            // topk_idx
                                  .Arg<ffi::Buffer<ffi::F32>>()            // topk_weights
                                  .Attr<int64_t>("num_experts")
                                  .Attr<int64_t>("expert_alignment")
                                  .Attr<int64_t>("num_worst_tokens")
                                  .Attr<int64_t>("ep_size")
                                  .Attr<int64_t>("launch_mode")
                                  .Attr<int64_t>("num_sms")
                                  .Attr<int64_t>("num_max_nvl_chunked_send_tokens")
                                  .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")
                                  .Attr<int64_t>("num_max_rdma_chunked_send_tokens")
                                  .Attr<int64_t>("num_max_rdma_chunked_recv_tokens")
                                  .Ret<ffi::AnyBuffer>()         // recv_x
                                  .Ret<ffi::Buffer<ffi::F32>>()  // recv_x_scales
                                  .Ret<ffi::Buffer<ffi::S32>>()  // recv_topk_idx
                                  .Ret<ffi::Buffer<ffi::F32>>()  // recv_topk_weights
                                  .Ret<ffi::Buffer<ffi::PRED>>() // is_token_in_rank
                                  .Ret<ffi::Buffer<ffi::S32>>()  // num_tokens_per_rank
                                  .Ret<ffi::Buffer<ffi::S32>>()  // num_tokens_per_rdma_rank
                                  .Ret<ffi::Buffer<ffi::S32>>()  // num_tokens_per_expert
                                  .Ret<ffi::Buffer<ffi::S32>>()  // rdma_channel_prefix_matrix
                                  .Ret<ffi::Buffer<ffi::S32>>()  // recv_rdma_rank_prefix_sum
                                  .Ret<ffi::Buffer<ffi::S32>>()  // gbl_channel_prefix_matrix
                                  .Ret<ffi::Buffer<ffi::S32>>()  // recv_gbl_rank_prefix_sum
                                  .Ret<ffi::Buffer<ffi::U8>>()   // recv_src_meta
                                  .Ret<ffi::Buffer<ffi::S32>>()  // recv_rdma_channel_prefix_matrix
                                  .Ret<ffi::Buffer<ffi::S32>>()  // recv_gbl_channel_prefix_matrix
                                  .Ret<ffi::Buffer<ffi::S32>>()  // send_rdma_head
                                  .Ret<ffi::Buffer<ffi::S32>>()  // send_nvl_head
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(MoEInternodeCachedDispatchPerProcessHandler,
                              MoEInternodeCachedDispatchPerProcessFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // x
                                  .Arg<ffi::Buffer<ffi::F32>>()            // x_scales
                                  .Arg<ffi::Buffer<ffi::PRED>>()           // is_token_in_rank
                                  .Arg<ffi::Buffer<ffi::S32>>() // cached_rdma_channel_prefix_matrix
                                  .Arg<ffi::Buffer<ffi::S32>>() // cached_recv_rdma_rank_prefix_sum
                                  .Arg<ffi::Buffer<ffi::S32>>() // cached_gbl_channel_prefix_matrix
                                  .Arg<ffi::Buffer<ffi::S32>>() // cached_recv_gbl_rank_prefix_sum
                                  .Attr<int64_t>("num_recv_tokens")
                                  .Attr<int64_t>("num_rdma_recv_tokens")
                                  .Attr<int64_t>("expert_alignment")
                                  .Attr<int64_t>("num_worst_tokens")
                                  .Attr<int64_t>("ep_size")
                                  .Attr<int64_t>("launch_mode")
                                  .Attr<int64_t>("num_sms")
                                  .Attr<int64_t>("num_max_nvl_chunked_send_tokens")
                                  .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")
                                  .Attr<int64_t>("num_max_rdma_chunked_send_tokens")
                                  .Attr<int64_t>("num_max_rdma_chunked_recv_tokens")
                                  .Ret<ffi::AnyBuffer>()        // recv_x
                                  .Ret<ffi::Buffer<ffi::F32>>() // recv_x_scales
                                  .Ret<ffi::Buffer<ffi::S32>>() // recv_rdma_channel_prefix_matrix
                                  .Ret<ffi::Buffer<ffi::S32>>() // recv_gbl_channel_prefix_matrix
                                  .Ret<ffi::Buffer<ffi::S32>>() // send_rdma_head
                                  .Ret<ffi::Buffer<ffi::S32>>() // send_nvl_head
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(MoEInternodeCombinePerProcessHandler,
                              MoEInternodeCombinePerProcessFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // x
                                  .Arg<ffi::Buffer<ffi::F32>>()            // topk_weights
                                  .Arg<ffi::Buffer<ffi::U8>>()             // src_meta
                                  .Arg<ffi::Buffer<ffi::PRED>>() // is_combined_token_in_rank
                                  .Arg<ffi::Buffer<ffi::S32>>()  // rdma_channel_prefix_matrix
                                  .Arg<ffi::Buffer<ffi::S32>>()  // rdma_rank_prefix_sum
                                  .Arg<ffi::Buffer<ffi::S32>>()  // gbl_channel_prefix_matrix
                                  .Arg<ffi::Buffer<ffi::S32>>()  // gbl_rank_prefix_sum
                                  .Arg<ffi::Buffer<ffi::S32>>()  // combined_rdma_head
                                  .Arg<ffi::Buffer<ffi::S32>>()  // combined_nvl_head
                                  .Attr<int64_t>("ep_size")
                                  .Attr<int64_t>("launch_mode")
                                  .Attr<int64_t>("num_sms")
                                  .Attr<int64_t>("num_max_nvl_chunked_send_tokens")
                                  .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")
                                  .Attr<int64_t>("num_max_rdma_chunked_send_tokens")
                                  .Attr<int64_t>("num_max_rdma_chunked_recv_tokens")
                                  .Ret<ffi::AnyBuffer>()        // combined_x
                                  .Ret<ffi::Buffer<ffi::F32>>() // combined_topk_weights
);

} // namespace primus_turbo::jax::deep_ep
