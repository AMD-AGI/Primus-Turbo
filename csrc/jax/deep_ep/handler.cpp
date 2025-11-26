// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "jax/deep_ep/deep_ep.h"
#include "jax/extensions.h"
#include "primus_turbo/common.h"
#include "primus_turbo/deep_ep/config.hpp"

namespace primus_turbo::jax::deep_ep {

int64_t get_hidden_bytes(ffi::AnyBuffer x) {
    PRIMUS_TURBO_CHECK(x.dimensions().size() == 2);
    return x.dimensions()[1] * std::max(ffi::ByteWidth(x.element_type()), static_cast<size_t>(2));
}

ffi::Error MoEDispatchFFI(hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> x_scales,
                          ffi::Buffer<ffi::S64> topk_idx, ffi::Buffer<ffi::F32> topk_weights,
                          /* attributes */
                          int64_t num_experts, int64_t expert_alignment, int64_t num_worst_tokens,
                          /*dispatch config*/
                          int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
                          int64_t num_max_nvl_chunked_recv_tokens,
                          int64_t num_max_rdma_chunked_send_tokens,
                          int64_t num_max_rdma_chunked_recv_tokens,
                          /* dispatched outputs */
                          ffi::Result<ffi::AnyBuffer>         recv_x,
                          ffi::Result<ffi::Buffer<ffi::F32>>  recv_x_scales,
                          ffi::Result<ffi::Buffer<ffi::S64>>  recv_topk_idx,
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

    int num_ranks = NUM_MAX_NVL_PEERS;
    int rank      = -1;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&rank));

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
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t num_sms,
    int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens,
    int64_t num_max_rdma_chunked_send_tokens, int64_t num_max_rdma_chunked_recv_tokens,
    /* dispatch handle for cached mode*/
    ffi::Result<ffi::AnyBuffer> recv_x, ffi::Result<ffi::Buffer<ffi::F32>> recv_x_scales,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_src_idx, ffi::Result<ffi::Buffer<ffi::S32>> send_head) {
    auto cfg = primus_turbo::deep_ep::Config(
        num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens);

    int num_ranks = NUM_MAX_NVL_PEERS;
    int rank      = -1;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&rank));

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

ffi::Error MoECombineFFI(
    hipStream_t stream, ffi::AnyBuffer x, ffi::Buffer<ffi::F32> topk_weights, ffi::AnyBuffer bias_0,
    ffi::AnyBuffer bias_1, ffi::Buffer<ffi::S32> src_idx, ffi::Buffer<ffi::S32> rank_prefix_matrix,
    ffi::Buffer<ffi::S32> channel_prefix_matrix, ffi::Buffer<ffi::S32> send_head, int64_t num_sms,
    int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens,
    int64_t num_max_rdma_chunked_send_tokens, int64_t num_max_rdma_chunked_recv_tokens,
    ffi::Result<ffi::AnyBuffer> recv_x, ffi::Result<ffi::Buffer<ffi::F32>> recv_topk_weights) {
    int num_ranks = NUM_MAX_NVL_PEERS;
    int rank      = -1;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&rank));

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
        .Arg<ffi::Buffer<ffi::S64>>()                      // topk_idx
        .Arg<ffi::Buffer<ffi::F32>>()                      // topk_weights
        .Attr<int64_t>("num_experts")                      // num_experts
        .Attr<int64_t>("expert_alignment")                 // expert_alignment
        .Attr<int64_t>("num_worst_tokens")                 // num_worst_tokens
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_x_scales
        .Ret<ffi::Buffer<ffi::S64>>()                      // recv_topk_idx
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
        .Attr<int64_t>("num_sms")                          // num_sms
        .Attr<int64_t>("num_max_nvl_chunked_send_tokens")  // num_max_nvl_chunked_send_tokens
        .Attr<int64_t>("num_max_nvl_chunked_recv_tokens")  // num_max_nvl_chunked_recv_tokens
        .Attr<int64_t>("num_max_rdma_chunked_send_tokens") // num_max_rdma_chunked_send_tokens
        .Attr<int64_t>("num_max_rdma_chunked_recv_tokens") // num_max_rdma_chunked_recv_tokens
        .Ret<ffi::AnyBuffer>()                             // recv_x
        .Ret<ffi::Buffer<ffi::F32>>()                      // recv_topk_weights
);

} // namespace primus_turbo::jax::deep_ep
