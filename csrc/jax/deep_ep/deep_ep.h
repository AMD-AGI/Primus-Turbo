/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once

#include "jax/ffi.h"
#include "primus_turbo/deep_ep/config.hpp"
#include "primus_turbo/deep_ep/configs.h"
#include <hip/hip_runtime.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>

namespace primus_turbo::jax::deep_ep {
class Buffer {

public:
    explicit Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes,
                    bool explicitly_destroy);

    ~Buffer() noexcept;

    void Destroy();

    // IPC handle access for cross-process synchronization.
    pybind11::bytearray get_local_ipc_handle() const;

    // Synchronize buffer pointers using IPC handles gathered from all ranks.
    // Each element is the raw bytes of a hipIpcMemHandle_t (size HIP_IPC_HANDLE_SIZE).
    void
    SyncFromIPCHandles(const std::vector<std::optional<pybind11::bytearray>> &all_handles,
                       const std::optional<pybind11::bytes> &root_unique_id_opt = std::nullopt);

    int64_t num_nvl_bytes() const { return num_nvl_bytes_; }
    int64_t num_rdma_bytes() const { return num_rdma_bytes_; }

    bool is_available() const { return is_available_; }

    bool is_internode_available() const;

    int rank() const { return rank_; }

    int num_ranks() const { return num_ranks_; }

    int num_rdma_ranks() const { return num_rdma_ranks_; }

    int rdma_rank() const { return rdma_rank_; }

    int root_rdma_rank(bool global) const { return global ? nvl_rank_ : 0; }

    int local_device_id() const { return device_id_; }

    void DispatchLayout(hipStream_t stream, ffi::Buffer<ffi::S32> topk_idx, int num_experts,
                        ffi::Result<ffi::Buffer<ffi::S32>>                num_tokens_per_rank,
                        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>> num_tokens_per_rdma_rank,
                        ffi::Result<ffi::Buffer<ffi::S32>>                num_tokens_per_expert,
                        ffi::Result<ffi::Buffer<ffi::PRED>>               is_token_in_rank);

    void IntranodeDispatch(hipStream_t stream, ffi::AnyBuffer x,
                           std::optional<ffi::Buffer<ffi::F32>> x_scales,
                           std::optional<ffi::Buffer<ffi::S32>> topk_idx,
                           std::optional<ffi::Buffer<ffi::F32>> topk_weights,
                           std::optional<ffi::Buffer<ffi::S32>> num_tokens_per_rank,
                           ffi::Buffer<ffi::PRED>               is_token_in_rank,
                           std::optional<ffi::Buffer<ffi::S32>> num_tokens_per_expert,
                           int                                  cached_num_recv_tokens,
                           std::optional<ffi::Buffer<ffi::S32>> cached_rank_prefix_matrix,
                           std::optional<ffi::Buffer<ffi::S32>> cached_channel_prefix_matrix,
                           int expert_alignment, int num_worst_tokens,
                           primus_turbo::deep_ep::Config config, ffi::Result<ffi::AnyBuffer> recv_x,
                           std::optional<ffi::Result<ffi::Buffer<ffi::F32>>> recv_x_scales,
                           std::optional<ffi::Result<ffi::Buffer<ffi::S32>>> recv_topk_idx,
                           std::optional<ffi::Result<ffi::Buffer<ffi::F32>>> recv_topk_weights,
                           std::optional<ffi::Result<ffi::Buffer<ffi::S32>>> rank_prefix_matrix,
                           std::optional<ffi::Result<ffi::Buffer<ffi::S32>>> channel_prefix_matrix,
                           ffi::Result<ffi::Buffer<ffi::S32>> recv_channel_prefix_matrix,
                           ffi::Result<ffi::Buffer<ffi::S32>> recv_src_idx,
                           ffi::Result<ffi::Buffer<ffi::S32>> send_head);

    void Sync();

    void IntranodeCombine(hipStream_t stream, ffi::AnyBuffer x,
                          std::optional<ffi::Buffer<ffi::F32>> topk_weights,
                          std::optional<ffi::AnyBuffer>        bias_0,
                          std::optional<ffi::AnyBuffer> bias_1, ffi::Buffer<ffi::S32> src_idx,
                          ffi::Buffer<ffi::S32> rank_prefix_matrix,
                          ffi::Buffer<ffi::S32> channel_prefix_matrix,
                          ffi::Buffer<ffi::S32> send_head, primus_turbo::deep_ep::Config config,
                          ffi::Result<ffi::AnyBuffer>                       recv_x,
                          std::optional<ffi::Result<ffi::Buffer<ffi::F32>>> recv_topk_weights);

    void InternodeDispatch(
        hipStream_t stream, ffi::AnyBuffer x, std::optional<ffi::Buffer<ffi::F32>> x_scales,
        std::optional<ffi::Buffer<ffi::S32>> topk_idx,
        std::optional<ffi::Buffer<ffi::F32>> topk_weights,
        std::optional<ffi::Buffer<ffi::S32>> num_tokens_per_rank,
        std::optional<ffi::Buffer<ffi::S32>> num_tokens_per_rdma_rank,
        ffi::Buffer<ffi::PRED>               is_token_in_rank,
        std::optional<ffi::Buffer<ffi::S32>> num_tokens_per_expert, int cached_num_recv_tokens,
        int                                  cached_num_rdma_recv_tokens,
        std::optional<ffi::Buffer<ffi::S32>> cached_rdma_channel_prefix_matrix,
        std::optional<ffi::Buffer<ffi::S32>> cached_recv_rdma_rank_prefix_sum,
        std::optional<ffi::Buffer<ffi::S32>> cached_gbl_channel_prefix_matrix,
        std::optional<ffi::Buffer<ffi::S32>> cached_recv_gbl_rank_prefix_sum, int expert_alignment,
        int num_worst_tokens, primus_turbo::deep_ep::Config config,
        ffi::Result<ffi::AnyBuffer>                        recv_x,
        std::optional<ffi::Result<ffi::Buffer<ffi::F32>>>  recv_x_scales,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  recv_topk_idx,
        std::optional<ffi::Result<ffi::Buffer<ffi::F32>>>  recv_topk_weights,
        std::optional<ffi::Result<ffi::Buffer<ffi::PRED>>> is_token_in_rank_out,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  num_tokens_per_rank_out,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  num_tokens_per_expert_out,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  rdma_channel_prefix_matrix,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  recv_rdma_rank_prefix_sum,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  gbl_channel_prefix_matrix,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  recv_gbl_rank_prefix_sum,
        std::optional<ffi::Result<ffi::Buffer<ffi::U8>>>   recv_src_meta,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  recv_rdma_channel_prefix_matrix,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  recv_gbl_channel_prefix_matrix,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  send_rdma_head,
        std::optional<ffi::Result<ffi::Buffer<ffi::S32>>>  send_nvl_head);

    void InternodeCombine(
        hipStream_t stream, ffi::AnyBuffer x, std::optional<ffi::Buffer<ffi::F32>> topk_weights,
        ffi::Buffer<ffi::U8> src_meta, ffi::Buffer<ffi::PRED> is_combined_token_in_rank,
        ffi::Buffer<ffi::S32> rdma_channel_prefix_matrix,
        ffi::Buffer<ffi::S32> rdma_rank_prefix_sum, ffi::Buffer<ffi::S32> gbl_channel_prefix_matrix,
        std::optional<ffi::Buffer<ffi::S32>> gbl_rank_prefix_sum,
        ffi::Buffer<ffi::S32> combined_rdma_head, ffi::Buffer<ffi::S32> combined_nvl_head,
        primus_turbo::deep_ep::Config config, ffi::Result<ffi::AnyBuffer> combined_x,
        std::optional<ffi::Result<ffi::Buffer<ffi::F32>>> combined_topk_weights);

private:
    // NVLink Buffer
    int64_t           num_nvl_bytes_;
    hipIpcMemHandle_t ipc_handles_[NUM_MAX_NVL_PEERS] = {};
    bool              ipc_synced_                     = false;
    void             *buffer_ptrs_[NUM_MAX_NVL_PEERS] = {nullptr};
    void            **buffer_ptrs_gpu_                = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes_;
    void   *rdma_buffer_ptr_      = nullptr;
    bool    rocshmem_initialized_ = false;

    // Device info and communication
    int device_id_;
    int num_device_sms_;
    int rank_, rdma_rank_, nvl_rank_;
    int num_ranks_, num_rdma_ranks_, num_nvl_ranks_;

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool is_available_ = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy_;
    // After `destroy()` be called, this flag will be true
    bool destroyed_ = false;

    // Barrier signals
    int  *barrier_signal_ptrs_[NUM_MAX_NVL_PEERS] = {nullptr};
    int **barrier_signal_ptrs_gpu_                = nullptr;

    // Workspace reserved for future low-latency kernels.
    void *workspace_ = nullptr;

    // Host-side MoE info
    volatile int *moe_recv_counter_        = nullptr;
    int          *moe_recv_counter_mapped_ = nullptr;

    // Host-side expert-level MoE info
    volatile int *moe_recv_expert_counter_        = nullptr;
    int          *moe_recv_expert_counter_mapped_ = nullptr;

    // Host-side RDMA-level MoE info
    volatile int *moe_recv_rdma_counter_        = nullptr;
    int          *moe_recv_rdma_counter_mapped_ = nullptr;
};

// Inproc mode: per-device buffer pool with in-process barrier sync.
Buffer *get_buffer(int rank, int num_ranks, int64_t hidden_bytes,
                   const primus_turbo::deep_ep::Config &config);

// Per-process mode: single buffer per process, IPC-synced from Python.
Buffer             *get_per_process_buffer();
pybind11::bytearray create_per_process_buffer(int rank, int num_ranks, int64_t num_nvl_bytes,
                                              int64_t num_rdma_bytes);
void                sync_per_process_buffer(
                   const std::vector<std::optional<pybind11::bytearray>> &all_handles,
                   const std::optional<pybind11::bytes>                  &root_unique_id_opt = std::nullopt);
void    destroy_per_process_buffer();
bool    is_per_process_buffer_ready();
int64_t per_process_buffer_nvl_bytes();

} // namespace primus_turbo::jax::deep_ep
