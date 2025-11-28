/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once

#include <torch/types.h>
#include <tuple>
#include <vector>

#include "primus_turbo/deep_ep/configs.h"

#include "config.hpp"
#include "event.hpp"

#include <torch/custom_class.h>
namespace primus_turbo::pytorch::deep_ep {

struct Buffer : torch::CustomClassHolder {

private:
    // Low-latency mode buffer
    int  low_latency_buffer_idx = 0;
    bool low_latency_mode       = false;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void   *buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void  **buffer_ptrs_gpu                = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes;
    void   *rdma_buffer_ptr = nullptr;

    // Device info and communication
    int               device_id;
    int               num_device_sms;
    int               rank, rdma_rank, nvl_rank;
    int               num_ranks, num_rdma_ranks, num_nvl_ranks;
    hipIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

    // Stream for communication
    at::hip::HIPStreamMasqueradingAsCUDA comm_stream;

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int  *barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int **barrier_signal_ptrs_gpu                = nullptr;

    // Workspace
    void *workspace = nullptr;

    // Host-side MoE info
    volatile int *moe_recv_counter        = nullptr;
    int          *moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int *moe_recv_expert_counter        = nullptr;
    int          *moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int *moe_recv_rdma_counter        = nullptr;
    int          *moe_recv_rdma_counter_mapped = nullptr;

    bool use_default_stream_as_comm_stream = false;

public:
    Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes,
           bool low_latency_mode, bool explicitly_destroy, bool use_default_stream_as_comm_stream);

    std::tuple<std::tuple<std::string, int64_t>, std::tuple<std::string, int64_t>,
               std::tuple<std::string, int64_t>, std::tuple<std::string, int64_t>,
               std::tuple<std::string, bool>, std::tuple<std::string, bool>,
               std::tuple<std::string, bool>>
    get_state() const;

    ~Buffer() noexcept(false) override;

    bool is_available() const;

    bool is_internode_available() const;

    int64_t get_num_rdma_ranks() const;

    int64_t get_rdma_rank() const;

    int64_t get_root_rdma_rank(bool global) const;

    int64_t get_local_device_id() const;

    torch::Tensor get_local_ipc_handle() const;

    torch::Tensor get_local_nvshmem_unique_id() const;

    torch::Tensor get_local_buffer_tensor(const torch::ScalarType &dtype, int64_t offset,
                                          bool use_rdma_buffer) const;

    torch::Stream get_comm_stream() const;

    void sync(const torch::Tensor                &all_gathered_handles,
              const std::optional<torch::Tensor> &root_unique_id_opt);

    void destroy();

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
               std::optional<c10::intrusive_ptr<EventHandle>>>
    get_dispatch_layout(const torch::Tensor &topk_idx, int64_t num_experts,
                        std::optional<c10::intrusive_ptr<EventHandle>> previous_event, bool async,
                        bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::vector<int64_t>, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor,
               std::optional<c10::intrusive_ptr<EventHandle>>>
    intranode_dispatch(const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales,
                       const std::optional<torch::Tensor> &topk_idx,
                       const std::optional<torch::Tensor> &topk_weights,
                       const std::optional<torch::Tensor> &num_tokens_per_rank,
                       const torch::Tensor                &is_token_in_rank,
                       const std::optional<torch::Tensor> &num_tokens_per_expert,
                       int64_t                             cached_num_recv_tokens,
                       const std::optional<torch::Tensor> &cached_rank_prefix_matrix,
                       const std::optional<torch::Tensor> &cached_channel_prefix_matrix,
                       int64_t expert_alignment, int64_t num_worst_tokens,
                       const c10::intrusive_ptr<Config>               config,
                       std::optional<c10::intrusive_ptr<EventHandle>> previous_event, bool async,
                       bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<c10::intrusive_ptr<EventHandle>>>
    intranode_combine(const torch::Tensor &x, const std::optional<torch::Tensor> &topk_weights,
                      const std::optional<torch::Tensor> &bias_0,
                      const std::optional<torch::Tensor> &bias_1, const torch::Tensor &src_idx,
                      const torch::Tensor &rank_prefix_matrix,
                      const torch::Tensor &channel_prefix_matrix, const torch::Tensor &send_head,
                      const c10::intrusive_ptr<Config>               config,
                      std::optional<c10::intrusive_ptr<EventHandle>> previous_event, bool async,
                      bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::vector<int64_t>, torch::Tensor, torch::Tensor,
               std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>,
               torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<c10::intrusive_ptr<EventHandle>>>
    internode_dispatch(const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales,
                       const std::optional<torch::Tensor> &topk_idx,
                       const std::optional<torch::Tensor> &topk_weights,
                       const std::optional<torch::Tensor> &num_tokens_per_rank,
                       const std::optional<torch::Tensor> &num_tokens_per_rdma_rank,
                       const torch::Tensor                &is_token_in_rank,
                       const std::optional<torch::Tensor> &num_tokens_per_expert,
                       int64_t cached_num_recv_tokens, int64_t cached_num_rdma_recv_tokens,
                       const std::optional<torch::Tensor> &cached_rdma_channel_prefix_matrix,
                       const std::optional<torch::Tensor> &cached_recv_rdma_rank_prefix_sum,
                       const std::optional<torch::Tensor> &cached_gbl_channel_prefix_matrix,
                       const std::optional<torch::Tensor> &cached_recv_gbl_rank_prefix_sum,
                       int64_t expert_alignment, int64_t num_worst_tokens,
                       const c10::intrusive_ptr<Config>               config,
                       std::optional<c10::intrusive_ptr<EventHandle>> previous_event, bool async,
                       bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<c10::intrusive_ptr<EventHandle>>>
    internode_combine(const torch::Tensor &x, const std::optional<torch::Tensor> &topk_weights,
                      const std::optional<torch::Tensor> &bias_0,
                      const std::optional<torch::Tensor> &bias_1, const torch::Tensor &src_meta,
                      const torch::Tensor                           &is_combined_token_in_rank,
                      const torch::Tensor                           &rdma_channel_prefix_matrix,
                      const torch::Tensor                           &rdma_rank_prefix_sum,
                      const torch::Tensor                           &gbl_channel_prefix_matrix,
                      const std::optional<torch::Tensor>            &gbl_rank_prefix_sum,
                      const torch::Tensor                           &combined_rdma_head,
                      const torch::Tensor                           &combined_nvl_head,
                      const c10::intrusive_ptr<Config>               config,
                      std::optional<c10::intrusive_ptr<EventHandle>> previous_event, bool async,
                      bool allocate_on_comm_stream);

    void clean_low_latency_buffer(int64_t num_max_dispatch_tokens_per_rank, int64_t hidden,
                                  int64_t num_experts);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
               torch::Tensor, std::optional<c10::intrusive_ptr<EventHandle>>,
               std::optional<std::function<void()>>>
    low_latency_dispatch(const torch::Tensor &x, const torch::Tensor &topk_idx,
                         const std::optional<torch::Tensor> &cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor> &dispatch_wait_recv_cost_stats,
                         int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
                         bool use_fp8, bool round_scale, bool use_ue8m0, bool async,
                         bool return_recv_hook);

    std::tuple<torch::Tensor, std::optional<c10::intrusive_ptr<EventHandle>>,
               std::optional<std::function<void()>>>
    low_latency_combine(const torch::Tensor &x, const torch::Tensor &topk_idx,
                        const torch::Tensor &topk_weights, const torch::Tensor &src_info,
                        const torch::Tensor                &layout_range,
                        const std::optional<torch::Tensor> &combine_wait_recv_cost_stats,
                        int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
                        bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                        const std::optional<torch::Tensor> &out = std::nullopt);

    torch::Tensor get_next_low_latency_combine_buffer(int64_t num_max_dispatch_tokens_per_rank,
                                                      int64_t hidden, int64_t num_experts) const;
};

c10::intrusive_ptr<Buffer> make_buffer(std::string group_name, int64_t num_nvl_bytes,
                                       int64_t num_rdma_bytes, bool low_latency_mode,
                                       bool explicitly_destroy,
                                       bool use_default_stream_as_comm_stream, bool available);

} // namespace primus_turbo::pytorch::deep_ep
