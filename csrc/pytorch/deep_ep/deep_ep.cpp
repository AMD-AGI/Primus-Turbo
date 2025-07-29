/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPDataType.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <pybind11/functional.h>
#include <torch/python.h>

#include "../kernels/deep_ep/api.h"
#include "../kernels/deep_ep/configs.h"
#include "deep_ep.hpp"

int get_env_with_default_value(const std::string &env_path, const std::string &default_value) {
    const char *value     = std::getenv(env_path.c_str());
    std::string value_str = (value != nullptr) ? std::string(value) : default_value;
    return std::stoi(value_str);
}

namespace primus_turbo::pytorch::deep_ep {

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes,
               bool low_latency_mode)
    : low_latency_mode(low_latency_mode), num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes), rank(rank), num_ranks(num_ranks),
      comm_stream(at::hip::getStreamFromPoolMasqueradingAsCUDA(true)) {
    // Task fifo memory
    int64_t fifo_bytes       = sizeof(int) * NUM_MAX_FIFO_SLOTS;
    int64_t buffer_ptr_bytes = sizeof(void *) * NUM_MAX_XGMI_PEERS;
    int64_t task_ptr_bytes   = sizeof(int *) * NUM_MAX_XGMI_PEERS;

    if (low_latency_mode) {
        if (get_env_with_default_value("LOW_LATENCY_OPTIMIZE", "0") == 1) {
            low_latency_optimize = true;
        }
    }

    // Common checks
    PRIMUS_TURBO_CHECK(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    PRIMUS_TURBO_CHECK(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    PRIMUS_TURBO_CHECK(0 <= rank and rank < num_ranks and
                       (num_ranks <= NUM_MAX_XGMI_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    PRIMUS_TURBO_CHECK(num_ranks < NUM_MAX_XGMI_PEERS or num_ranks % NUM_MAX_XGMI_PEERS == 0);
    if (num_rdma_bytes > 0)
        PRIMUS_TURBO_CHECK(num_ranks > NUM_MAX_XGMI_PEERS or low_latency_mode);

    // Get ranks
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_XGMI_PEERS, nvl_rank = rank % NUM_MAX_XGMI_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_XGMI_PEERS),
    num_nvl_ranks  = std::min(num_ranks, NUM_MAX_XGMI_PEERS);

    // Get device info
    hipDeviceProp_t device_prop = {};
    PRIMUS_TURBO_CHECK_HIP(hipGetDeviceProperties(&device_prop, device_id));

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handle
        PRIMUS_TURBO_CHECK_HIP(hipExtMallocWithFlags(
            &buffer_ptrs[nvl_rank], num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes,
            hipDeviceMallocUncached));
        PRIMUS_TURBO_CHECK_HIP(hipIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void **>(
            reinterpret_cast<uint8_t *>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + fifo_bytes);

        // Set task fifo
        PRIMUS_TURBO_CHECK(NUM_MAX_FIFO_SLOTS % num_nvl_ranks == 0);
        task_fifo_ptrs[nvl_rank] = reinterpret_cast<int *>(
            reinterpret_cast<uint8_t *>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        task_fifo_ptrs_gpu =
            reinterpret_cast<int **>(reinterpret_cast<uint8_t *>(buffer_ptrs[nvl_rank]) +
                                     num_nvl_bytes + fifo_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        PRIMUS_TURBO_CHECK_HIP(
            hipMemsetAsync(task_fifo_ptrs[nvl_rank], 0, fifo_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    PRIMUS_TURBO_CHECK_HIP(
        hipExtMallocWithFlags(&workspace, NUM_WORKSPACE_BYTES, hipDeviceMallocUncached));

    PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    PRIMUS_TURBO_CHECK_HIP(hipHostMalloc(&moe_recv_counter, sizeof(int64_t), hipHostMallocMapped));
    PRIMUS_TURBO_CHECK_HIP(
        hipHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_counter_mapped),
                                const_cast<int *>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    PRIMUS_TURBO_CHECK_HIP(hipHostMalloc(&moe_recv_expert_counter,
                                         sizeof(int) * NUM_MAX_LOCAL_EXPERTS, hipHostMallocMapped));
    PRIMUS_TURBO_CHECK_HIP(
        hipHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_expert_counter_mapped),
                                const_cast<int *>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        PRIMUS_TURBO_CHECK_HIP(
            hipHostMalloc(&moe_recv_rdma_counter, sizeof(int), hipHostMallocMapped));
        PRIMUS_TURBO_CHECK_HIP(
            hipHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_rdma_counter_mapped),
                                    const_cast<int *>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    // Synchronize
    PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());

    if (num_nvl_bytes > 0) {
        // Barrier
        primus_turbo::deep_ep::intranode::barrier(task_fifo_ptrs_gpu, head, nvl_rank, num_nvl_ranks,
                                                  comm_stream);
        move_fifo_slots();
        PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    PRIMUS_TURBO_CHECK_HIP(hipIpcCloseMemHandle(buffer_ptrs[i]));
        }

        // Free local buffer and error flag
        PRIMUS_TURBO_CHECK_HIP(hipFree(buffer_ptrs[nvl_rank]));
    }

    // Free NVSHMEM
    if (num_rdma_bytes > 0) {
        PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());
        primus_turbo::deep_ep::internode::barrier();
        primus_turbo::deep_ep::internode::free(rdma_buffer_ptr);
        primus_turbo::deep_ep::internode::finalize();
    }

    // Free cuBLAS handle, workspace and MoE counter
    PRIMUS_TURBO_CHECK_HIP(hipFree(workspace));
    PRIMUS_TURBO_CHECK_HIP(hipHostFree(const_cast<int *>(moe_recv_counter)));

    // Free chunked mode staffs
    PRIMUS_TURBO_CHECK_HIP(hipHostFree(const_cast<int *>(moe_recv_expert_counter)));
}

void Buffer::move_fifo_slots(int num_slots) {
    head = (head + num_ranks * num_slots) % NUM_MAX_FIFO_SLOTS;
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_low_latency_optimize() const {
    return low_latency_optimize;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_XGMI_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    return {ipc_handles[nvl_rank].reserved, HIP_IPC_HANDLE_SIZE};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
    PRIMUS_TURBO_CHECK(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
    auto unique_id = primus_turbo::deep_ep::internode::get_unique_id();
    return {reinterpret_cast<const char *>(unique_id.data()), unique_id.size()};
}

pybind11::bytearray Buffer::get_local_pxn_ipc_handle() const {
    return {pxn_ipc_handles[nvl_rank].reserved, HIP_IPC_HANDLE_SIZE};
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object &dtype, int64_t offset,
                                              bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype  = torch::python::detail::py_object_to_dtype(dtype);
    auto              element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto              base_ptr =
        reinterpret_cast<uint8_t *>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) +
        offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes,
                            torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

void Buffer::sync(const std::vector<int>                                &device_ids,
                  const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
                  const std::optional<pybind11::bytearray>              &root_unique_id_opt) {
    PRIMUS_TURBO_CHECK(not is_available());

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
        PRIMUS_TURBO_CHECK(num_ranks == device_ids.size());
        PRIMUS_TURBO_CHECK(device_ids.size() == all_gathered_handles.size());
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            PRIMUS_TURBO_CHECK(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            PRIMUS_TURBO_CHECK(handle_str.size() == HIP_IPC_HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), HIP_IPC_HANDLE_SIZE);
                PRIMUS_TURBO_CHECK_HIP(hipIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i],
                                                           hipIpcMemLazyEnablePeerAccess));
                task_fifo_ptrs[i] = reinterpret_cast<int *>(
                    reinterpret_cast<uint8_t *>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                PRIMUS_TURBO_CHECK(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(),
                                               HIP_IPC_HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and task pointers to GPU
        PRIMUS_TURBO_CHECK_HIP(hipMemcpy(buffer_ptrs_gpu, buffer_ptrs,
                                         sizeof(void *) * NUM_MAX_XGMI_PEERS,
                                         hipMemcpyHostToDevice));
        PRIMUS_TURBO_CHECK_HIP(hipMemcpy(task_fifo_ptrs_gpu, task_fifo_ptrs,
                                         sizeof(int *) * NUM_MAX_XGMI_PEERS,
                                         hipMemcpyHostToDevice));
        PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());
    }

    // Sync NVSHMEM handles and allocate memory
    if (num_rdma_bytes > 0) {
        // Initialize NVSHMEM
        PRIMUS_TURBO_CHECK(root_unique_id_opt.has_value());
        std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
        auto                 root_unique_id_str = root_unique_id_opt->cast<std::string>();
        std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
        auto nvshmem_rank      = low_latency_mode ? rank : rdma_rank;
        auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
        PRIMUS_TURBO_CHECK(nvshmem_rank ==
                           primus_turbo::deep_ep::internode::init(
                               root_unique_id, nvshmem_rank, num_nvshmem_ranks, low_latency_mode));
        primus_turbo::deep_ep::internode::barrier();

        // Allocate
        rdma_buffer_ptr =
            primus_turbo::deep_ep::internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

        // Clean buffer (mainly for low-latency mode)
        PRIMUS_TURBO_CHECK_HIP(hipMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
        if (not low_latency_optimize) {
            // Barrier
            primus_turbo::deep_ep::internode::barrier();
            PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());
        } else {
            PRIMUS_TURBO_CHECK_HIP(hipExtMallocWithFlags(&nvl_buffer_ptrs[nvl_rank], num_rdma_bytes,
                                                         hipDeviceMallocUncached));
            PRIMUS_TURBO_CHECK_HIP(
                hipIpcGetMemHandle(&pxn_ipc_handles[nvl_rank], nvl_buffer_ptrs[nvl_rank]));
        }
    }
    if (not low_latency_optimize) {
        // Ready to use
        available = true;
    }
}

void Buffer::sync_pxn_handles(
    const std::vector<int>                                &device_ids,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles) {
    // Sync NVSHMEM handles and allocate memory
    if (num_rdma_bytes > 0) {
        PRIMUS_TURBO_CHECK(not is_available());
        PRIMUS_TURBO_CHECK(num_ranks == device_ids.size());
        PRIMUS_TURBO_CHECK(device_ids.size() == all_gathered_handles.size());
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            PRIMUS_TURBO_CHECK(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            PRIMUS_TURBO_CHECK(handle_str.size() == HIP_IPC_HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(pxn_ipc_handles[i].reserved, handle_str.c_str(), HIP_IPC_HANDLE_SIZE);
                PRIMUS_TURBO_CHECK_HIP(hipIpcOpenMemHandle(&nvl_buffer_ptrs[i], pxn_ipc_handles[i],
                                                           hipIpcMemLazyEnablePeerAccess));
            } else {
                PRIMUS_TURBO_CHECK(std::memcmp(pxn_ipc_handles[i].reserved, handle_str.c_str(),
                                               HIP_IPC_HANDLE_SIZE) == 0);
            }
        }
        int64_t buffer_ptr_bytes = sizeof(void *) * NUM_MAX_XGMI_PEERS;
        PRIMUS_TURBO_CHECK_HIP(
            hipExtMallocWithFlags(reinterpret_cast<void **>(&nvl_buffer_ptrs_gpu), buffer_ptr_bytes,
                                  hipDeviceMallocUncached));
        PRIMUS_TURBO_CHECK_HIP(hipMemcpy(nvl_buffer_ptrs_gpu, nvl_buffer_ptrs,
                                         sizeof(void *) * NUM_MAX_XGMI_PEERS,
                                         hipMemcpyHostToDevice));
        // Barrier
        primus_turbo::deep_ep::internode::barrier();
        PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());
        // Ready to use
        available = true;
    }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
           std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor &topk_idx, int num_experts,
                            std::optional<EventHandle> &previous_event, bool async,
                            bool allocate_on_comm_stream) {
    PRIMUS_TURBO_CHECK(topk_idx.dim() == 2);
    PRIMUS_TURBO_CHECK(topk_idx.is_contiguous());
    PRIMUS_TURBO_CHECK(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (allocate_on_comm_stream) {
        PRIMUS_TURBO_CHECK(previous_event.has_value() and async);
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)),
         num_topk   = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank =
        torch::empty({num_ranks}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert =
        torch::empty({num_experts}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty(
        {num_tokens, num_ranks}, at::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty(
            {num_rdma_ranks}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    primus_turbo::deep_ep::internode::get_dispatch_layout(
        topk_idx.data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(),
        num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>()
                                             : nullptr,
        num_tokens_per_expert.data_ptr<int>(), is_token_in_rank.data_ptr<bool>(), num_tokens,
        num_topk, num_ranks, num_experts, comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto &t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto &to : {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::intranode_dispatch(
    const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales,
    const std::optional<torch::Tensor> &topk_idx, const std::optional<torch::Tensor> &topk_weights,
    const std::optional<torch::Tensor> &num_tokens_per_rank, const torch::Tensor &is_token_in_rank,
    const std::optional<torch::Tensor> &num_tokens_per_expert, int cached_num_recv_tokens,
    const std::optional<torch::Tensor> &cached_rank_prefix_matrix,
    const std::optional<torch::Tensor> &cached_channel_prefix_matrix, int expert_alignment,
    const Config &config, std::optional<EventHandle> &previous_event, bool async,
    bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for
    // receiving.
    PRIMUS_TURBO_CHECK(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        PRIMUS_TURBO_CHECK(cached_rank_prefix_matrix.has_value());
        PRIMUS_TURBO_CHECK(cached_channel_prefix_matrix.has_value());
    } else {
        PRIMUS_TURBO_CHECK(num_tokens_per_rank.has_value());
        PRIMUS_TURBO_CHECK(num_tokens_per_expert.has_value());
    }

    // Type checks
    PRIMUS_TURBO_CHECK(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        PRIMUS_TURBO_CHECK(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        PRIMUS_TURBO_CHECK(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->scalar_type() == torch::kInt32);
        PRIMUS_TURBO_CHECK(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    PRIMUS_TURBO_CHECK(x.dim() == 2 and x.is_contiguous());
    PRIMUS_TURBO_CHECK((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    PRIMUS_TURBO_CHECK(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    PRIMUS_TURBO_CHECK(is_token_in_rank.size(0) == x.size(0) and
                       is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        PRIMUS_TURBO_CHECK(cached_rank_prefix_matrix->dim() == 2 and
                           cached_rank_prefix_matrix->is_contiguous());
        PRIMUS_TURBO_CHECK(cached_rank_prefix_matrix->size(0) == num_ranks and
                           cached_rank_prefix_matrix->size(1) == num_ranks);
        PRIMUS_TURBO_CHECK(cached_channel_prefix_matrix->dim() == 2 and
                           cached_channel_prefix_matrix->is_contiguous());
        PRIMUS_TURBO_CHECK(cached_channel_prefix_matrix->size(0) == num_ranks and
                           cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->dim() == 1 and
                           num_tokens_per_expert->is_contiguous());
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->size(0) % num_ranks == 0);
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        PRIMUS_TURBO_CHECK(num_tokens_per_rank->dim() == 1 and
                           num_tokens_per_rank->is_contiguous());
        PRIMUS_TURBO_CHECK(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts       = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
         num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int      num_topk         = 0;
    int64_t *topk_idx_ptr     = nullptr;
    float   *topk_weights_ptr = nullptr;
    PRIMUS_TURBO_CHECK(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        PRIMUS_TURBO_CHECK(num_experts > 0);
        PRIMUS_TURBO_CHECK(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        PRIMUS_TURBO_CHECK(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        PRIMUS_TURBO_CHECK(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        PRIMUS_TURBO_CHECK(num_topk == topk_weights->size(1));
        PRIMUS_TURBO_CHECK(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr     = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float *x_scales_ptr = nullptr;
    int    num_scales   = 0;
    if (x_scales.has_value()) {
        PRIMUS_TURBO_CHECK(x.element_size() == 1);
        PRIMUS_TURBO_CHECK(x_scales->scalar_type() == torch::kFloat32);
        PRIMUS_TURBO_CHECK(x_scales->dim() > 0 and x_scales->dim() < 3 and
                           x_scales->is_contiguous());
        PRIMUS_TURBO_CHECK(x_scales->size(0) == num_tokens);
        num_scales   = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = x_scales->data_ptr<float>();
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (allocate_on_comm_stream) {
        PRIMUS_TURBO_CHECK(previous_event.has_value() and async);
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int              num_recv_tokens       = -1;
    auto             rank_prefix_matrix    = torch::Tensor();
    auto             channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens       = cached_num_recv_tokens;
        rank_prefix_matrix    = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        primus_turbo::deep_ep::intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, task_fifo_ptrs_gpu,
            head, rank, num_ranks, comm_stream);
        move_fifo_slots(2);
    } else {
        rank_prefix_matrix = torch::empty(
            {num_ranks, num_ranks}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        channel_prefix_matrix =
            torch::empty({num_ranks, num_channels},
                         at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        PRIMUS_TURBO_CHECK(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <=
                           num_nvl_bytes);
        primus_turbo::deep_ep::intranode::notify_dispatch(
            num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
            num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
            num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, expert_alignment, buffer_ptrs_gpu,
            task_fifo_ptrs_gpu, head, rank, comm_stream, num_channels);
        move_fifo_slots(3);

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start_time)
                    .count() > NUM_CPU_TIMEOUT_SECS)
                throw std::runtime_error("DeepEP error: CPU recv timeout");
        }
        num_recv_tokens_per_expert_list =
            std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    auto recv_x                     = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx               = torch::empty({num_recv_tokens},
                                                   at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx              = std::optional<torch::Tensor>(),
         recv_topk_weights          = std::optional<torch::Tensor>(),
         recv_x_scales              = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty(
        {num_ranks, num_channels}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks},
                                  at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    // Assign pointers
    int64_t *recv_topk_idx_ptr     = nullptr;
    float   *recv_topk_weights_ptr = nullptr;
    float   *recv_x_scales_ptr     = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx         = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights     = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr     = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales     = x_scales->dim() == 1
                                ? torch::empty({num_recv_tokens}, x_scales->options())
                                : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
    }

    // Dispatch
    PRIMUS_TURBO_CHECK(
        static_cast<int64_t>(num_ranks * num_ranks * sizeof(int) +        // Size prefix matrix
                             num_channels * num_ranks * sizeof(int) +     // Channel start offset
                             num_channels * num_ranks * sizeof(int) +     // Channel end offset
                             num_channels * num_ranks * sizeof(int) * 2 + // Queue head and tail
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 hidden * recv_x.element_size() + // Data buffer
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 sizeof(int) + // Source index buffer
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 num_topk * sizeof(int64_t) + // Top-k index buffer
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 num_topk * sizeof(float) + // Top-k weight buffer
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 sizeof(float) * num_scales) // FP8 scale buffer
        <= num_nvl_bytes);
    primus_turbo::deep_ep::intranode::dispatch(
        recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr,
        recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(),
        send_head.data_ptr<int>(), x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
        is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(), num_tokens,
        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)), num_topk, num_experts,
        num_scales, buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
        config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto &t : {x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, recv_x,
                        recv_src_idx, recv_channel_prefix_matrix, send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto &to :
             {x_scales, topk_idx, topk_weights, num_tokens_per_rank, num_tokens_per_expert,
              cached_channel_prefix_matrix, cached_rank_prefix_matrix, recv_topk_idx,
              recv_topk_weights, recv_x_scales}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::intranode_combine(const torch::Tensor &x, const std::optional<torch::Tensor> &topk_weights,
                          const torch::Tensor &src_idx, const torch::Tensor &rank_prefix_matrix,
                          const torch::Tensor &channel_prefix_matrix,
                          const torch::Tensor &send_head, const Config &config,
                          std::optional<EventHandle> &previous_event, bool async,
                          bool allocate_on_comm_stream) {
    PRIMUS_TURBO_CHECK(x.dim() == 2 and x.is_contiguous());
    PRIMUS_TURBO_CHECK(src_idx.dim() == 1 and src_idx.is_contiguous() and
                       src_idx.scalar_type() == torch::kInt32);
    PRIMUS_TURBO_CHECK(send_head.dim() == 2 and send_head.is_contiguous() and
                       send_head.scalar_type() == torch::kInt32);
    PRIMUS_TURBO_CHECK(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                       rank_prefix_matrix.scalar_type() == torch::kInt32);
    PRIMUS_TURBO_CHECK(channel_prefix_matrix.dim() == 2 and
                       channel_prefix_matrix.is_contiguous() and
                       channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for
    // receiving.
    PRIMUS_TURBO_CHECK(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    PRIMUS_TURBO_CHECK(src_idx.size(0) == num_tokens);
    PRIMUS_TURBO_CHECK(send_head.size(1) == num_ranks);
    PRIMUS_TURBO_CHECK(rank_prefix_matrix.size(0) == num_ranks and
                       rank_prefix_matrix.size(1) == num_ranks);
    PRIMUS_TURBO_CHECK(channel_prefix_matrix.size(0) == num_ranks and
                       channel_prefix_matrix.size(1) == num_channels);
    PRIMUS_TURBO_CHECK((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (allocate_on_comm_stream) {
        PRIMUS_TURBO_CHECK(previous_event.has_value() and async);
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int    num_topk              = 0;
    auto   recv_topk_weights     = std::optional<torch::Tensor>();
    float *topk_weights_ptr      = nullptr;
    float *recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        PRIMUS_TURBO_CHECK(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        PRIMUS_TURBO_CHECK(topk_weights->size(0) == num_tokens);
        PRIMUS_TURBO_CHECK(topk_weights->scalar_type() == torch::kFloat32);
        num_topk              = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr      = topk_weights->data_ptr<float>();
        recv_topk_weights     = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    PRIMUS_TURBO_CHECK(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    primus_turbo::deep_ep::intranode::cached_notify_combine(
        buffer_ptrs_gpu, send_head.data_ptr<int>(), num_channels, num_recv_tokens,
        num_channels * num_ranks * 2, task_fifo_ptrs_gpu, head, rank, num_ranks, comm_stream);

    // NOTES: this function uses two FIFO slots (barrier before and after)
    move_fifo_slots(2);

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    PRIMUS_TURBO_CHECK(
        static_cast<int64_t>(num_channels * num_ranks * sizeof(int) * 2 + // Queue head and tail
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 hidden * x.element_size() + // Data buffer
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 sizeof(int) + // Source index buffer
                             num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                                 num_topk * sizeof(float)) // Top-k weight buffer
        <= num_nvl_bytes);
    primus_turbo::deep_ep::intranode::combine(
        at::cuda::ScalarTypeToCudaDataType(x.scalar_type()), recv_x.data_ptr(),
        recv_topk_weights_ptr, x.data_ptr(), topk_weights_ptr, src_idx.data_ptr<int>(),
        rank_prefix_matrix.data_ptr<int>(), channel_prefix_matrix.data_ptr<int>(),
        send_head.data_ptr<int>(), num_tokens, num_recv_tokens, hidden, num_topk, buffer_ptrs_gpu,
        rank, num_ranks, comm_stream, config.num_sms, config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto &t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto &to : {topk_weights, recv_topk_weights}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(compute_stream);

    return {recv_x, recv_topk_weights, event};
}

// Internode functionality
std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor,
           std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
           std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales,
                           const std::optional<torch::Tensor> &topk_idx,
                           const std::optional<torch::Tensor> &topk_weights,
                           const std::optional<torch::Tensor> &num_tokens_per_rank,
                           const std::optional<torch::Tensor> &num_tokens_per_rdma_rank,
                           const torch::Tensor                &is_token_in_rank,
                           const std::optional<torch::Tensor> &num_tokens_per_expert,
                           int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor> &cached_rdma_channel_prefix_matrix,
                           const std::optional<torch::Tensor> &cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor> &cached_gbl_channel_prefix_matrix,
                           const std::optional<torch::Tensor> &cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment, const Config &config,
                           std::optional<EventHandle> &previous_event, bool async,
                           bool allocate_on_comm_stream) {
    const int num_channels = config.num_sms / 2;
    PRIMUS_TURBO_CHECK(config.num_sms % 2 == 0);
    PRIMUS_TURBO_CHECK(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        PRIMUS_TURBO_CHECK(cached_rdma_channel_prefix_matrix.has_value());
        PRIMUS_TURBO_CHECK(cached_recv_rdma_rank_prefix_sum.has_value());
        PRIMUS_TURBO_CHECK(cached_gbl_channel_prefix_matrix.has_value());
        PRIMUS_TURBO_CHECK(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        PRIMUS_TURBO_CHECK(num_tokens_per_rank.has_value());
        PRIMUS_TURBO_CHECK(num_tokens_per_rdma_rank.has_value());
        PRIMUS_TURBO_CHECK(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        PRIMUS_TURBO_CHECK(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        PRIMUS_TURBO_CHECK(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        PRIMUS_TURBO_CHECK(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        PRIMUS_TURBO_CHECK(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        PRIMUS_TURBO_CHECK(num_tokens_per_rank->scalar_type() == torch::kInt32);
        PRIMUS_TURBO_CHECK(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    PRIMUS_TURBO_CHECK(x.dim() == 2 and x.is_contiguous());
    PRIMUS_TURBO_CHECK((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        PRIMUS_TURBO_CHECK(cached_rdma_channel_prefix_matrix->dim() == 2 and
                           cached_rdma_channel_prefix_matrix->is_contiguous());
        PRIMUS_TURBO_CHECK(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                           cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        PRIMUS_TURBO_CHECK(cached_recv_rdma_rank_prefix_sum->dim() == 1 and
                           cached_recv_rdma_rank_prefix_sum->is_contiguous());
        PRIMUS_TURBO_CHECK(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        PRIMUS_TURBO_CHECK(cached_gbl_channel_prefix_matrix->dim() == 2 and
                           cached_gbl_channel_prefix_matrix->is_contiguous());
        PRIMUS_TURBO_CHECK(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                           cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        PRIMUS_TURBO_CHECK(cached_recv_gbl_rank_prefix_sum->dim() == 1 and
                           cached_recv_gbl_rank_prefix_sum->is_contiguous());
        PRIMUS_TURBO_CHECK(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        PRIMUS_TURBO_CHECK(num_tokens_per_rank->dim() == 1 and
                           num_tokens_per_rank->is_contiguous());
        PRIMUS_TURBO_CHECK(num_tokens_per_rdma_rank->dim() == 1 and
                           num_tokens_per_rdma_rank->is_contiguous());
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->dim() == 1 and
                           num_tokens_per_expert->is_contiguous());
        PRIMUS_TURBO_CHECK(num_tokens_per_rank->size(0) == num_ranks);
        PRIMUS_TURBO_CHECK(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->size(0) % num_ranks == 0);
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4       = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts       = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
         num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int      num_topk         = 0;
    int64_t *topk_idx_ptr     = nullptr;
    float   *topk_weights_ptr = nullptr;
    PRIMUS_TURBO_CHECK(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        PRIMUS_TURBO_CHECK(num_experts > 0);
        PRIMUS_TURBO_CHECK(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        PRIMUS_TURBO_CHECK(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        PRIMUS_TURBO_CHECK(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        PRIMUS_TURBO_CHECK(num_topk == topk_weights->size(1));
        PRIMUS_TURBO_CHECK(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr     = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float *x_scales_ptr = nullptr;
    int    num_scales   = 0;
    if (x_scales.has_value()) {
        PRIMUS_TURBO_CHECK(x.element_size() == 1);
        PRIMUS_TURBO_CHECK(x_scales->scalar_type() == torch::kFloat32);
        PRIMUS_TURBO_CHECK(x_scales->dim() > 0 and x_scales->dim() < 3 and
                           x_scales->is_contiguous());
        PRIMUS_TURBO_CHECK(x_scales->size(0) == num_tokens);
        num_scales   = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = x_scales->data_ptr<float>();
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (allocate_on_comm_stream) {
        PRIMUS_TURBO_CHECK(previous_event.has_value() and async);
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int              num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto             rdma_channel_prefix_matrix = torch::Tensor();
    auto             recv_rdma_rank_prefix_sum  = torch::Tensor();
    auto             gbl_channel_prefix_matrix  = torch::Tensor();
    auto             recv_gbl_rank_prefix_sum   = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens            = cached_num_recv_tokens;
        num_rdma_recv_tokens       = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum  = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix  = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum   = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        primus_turbo::deep_ep::internode::cached_notify(
            hidden_int4, num_scales, num_topk, num_topk, num_ranks, num_channels, 0, nullptr,
            nullptr, nullptr, nullptr, rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
            buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens, task_fifo_ptrs_gpu, head, rank,
            comm_stream, config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
            num_nvl_bytes, true, low_latency_mode);
        move_fifo_slots(2);
    } else {
        rdma_channel_prefix_matrix =
            torch::empty({num_rdma_ranks, num_channels},
                         at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        recv_rdma_rank_prefix_sum = torch::empty(
            {num_rdma_ranks}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        gbl_channel_prefix_matrix =
            torch::empty({num_ranks, num_channels},
                         at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_rank_prefix_sum = torch::empty(
            {num_ranks}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        primus_turbo::deep_ep::internode::notify_dispatch(
            num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
            num_tokens_per_rdma_rank->data_ptr<int>(), moe_recv_rdma_counter_mapped,
            num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
            is_token_in_rank.data_ptr<bool>(), num_tokens, num_channels, hidden_int4, num_scales,
            num_topk, expert_alignment, rdma_channel_prefix_matrix.data_ptr<int>(),
            recv_rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(),
            recv_gbl_rank_prefix_sum.data_ptr<int>(), rdma_buffer_ptr,
            config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
            config.num_max_nvl_chunked_recv_tokens, task_fifo_ptrs_gpu, head, rank, comm_stream,
            config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks), num_nvl_bytes,
            low_latency_mode);
        move_fifo_slots(3);

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens      = static_cast<int>(*moe_recv_counter);
            num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start_time)
                    .count() > NUM_CPU_TIMEOUT_SECS) {
                printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank,
                       num_recv_tokens, num_rdma_recv_tokens);
                for (int i = 0; i < num_local_experts; ++i)
                    printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
            }
        }
        num_recv_tokens_per_expert_list =
            std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    auto recv_x                          = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx                   = std::optional<torch::Tensor>(),
         recv_topk_weights               = std::optional<torch::Tensor>(),
         recv_x_scales                   = std::optional<torch::Tensor>();
    auto recv_src_meta                   = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix  = std::optional<torch::Tensor>();
    auto send_rdma_head                  = std::optional<torch::Tensor>();
    auto send_nvl_head                   = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty(
            {num_recv_tokens, primus_turbo::deep_ep::internode::get_source_meta_bytes()},
            at::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
        recv_rdma_channel_prefix_matrix =
            torch::empty({num_rdma_ranks, num_channels},
                         at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_channel_prefix_matrix =
            torch::empty({num_ranks, num_channels},
                         at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        send_rdma_head =
            torch::empty({num_tokens, num_rdma_ranks},
                         at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_XGMI_PEERS},
                                     at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }

    // Assign pointers
    int64_t *recv_topk_idx_ptr     = nullptr;
    float   *recv_topk_weights_ptr = nullptr;
    float   *recv_x_scales_ptr     = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx         = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights     = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr     = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales     = x_scales->dim() == 1
                                ? torch::empty({num_recv_tokens}, x_scales->options())
                                : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    primus_turbo::deep_ep::internode::dispatch(
        recv_x.data_ptr(), recv_x_scales_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
        cached_mode ? nullptr : recv_src_meta->data_ptr(), x.data_ptr(), x_scales_ptr, topk_idx_ptr,
        topk_weights_ptr, cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
        cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
        rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
        gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
        num_tokens, hidden_int4, num_scales, num_topk, num_experts,
        is_token_in_rank.data_ptr<bool>(), rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens, rank,
        num_ranks, cached_mode, comm_stream, num_channels, low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto &t :
             {x, is_token_in_rank, recv_x, rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
              gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto &to :
             {x_scales, topk_idx, topk_weights, num_tokens_per_rank, num_tokens_per_rdma_rank,
              num_tokens_per_expert, cached_rdma_channel_prefix_matrix,
              cached_recv_rdma_rank_prefix_sum, cached_gbl_channel_prefix_matrix,
              cached_recv_gbl_rank_prefix_sum, recv_topk_idx, recv_topk_weights, recv_x_scales,
              recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, send_rdma_head,
              send_nvl_head, recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_combine(
    const torch::Tensor &x, const std::optional<torch::Tensor> &topk_weights,
    const torch::Tensor &src_meta, const torch::Tensor &is_combined_token_in_rank,
    const torch::Tensor &rdma_channel_prefix_matrix, const torch::Tensor &rdma_rank_prefix_sum,
    const torch::Tensor &gbl_channel_prefix_matrix, const torch::Tensor &combined_rdma_head,
    const torch::Tensor &combined_nvl_head, const Config &config,
    std::optional<EventHandle> &previous_event, bool async, bool allocate_on_comm_stream) {
    const int num_channels = config.num_sms / 2;
    PRIMUS_TURBO_CHECK(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    PRIMUS_TURBO_CHECK(x.dim() == 2 and x.is_contiguous());
    PRIMUS_TURBO_CHECK(src_meta.dim() == 2 and src_meta.is_contiguous() and
                       src_meta.scalar_type() == torch::kByte);
    PRIMUS_TURBO_CHECK(is_combined_token_in_rank.dim() == 2 and
                       is_combined_token_in_rank.is_contiguous() and
                       is_combined_token_in_rank.scalar_type() == torch::kBool);
    PRIMUS_TURBO_CHECK(rdma_channel_prefix_matrix.dim() == 2 and
                       rdma_channel_prefix_matrix.is_contiguous() and
                       rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    PRIMUS_TURBO_CHECK(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                       rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    PRIMUS_TURBO_CHECK(gbl_channel_prefix_matrix.dim() == 2 and
                       gbl_channel_prefix_matrix.is_contiguous() and
                       gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    PRIMUS_TURBO_CHECK(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                       combined_rdma_head.scalar_type() == torch::kInt32);
    PRIMUS_TURBO_CHECK(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and
                       combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4         = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    PRIMUS_TURBO_CHECK((hidden * x.element_size()) % sizeof(int4) == 0);
    PRIMUS_TURBO_CHECK(src_meta.size(1) ==
                       primus_turbo::deep_ep::internode::get_source_meta_bytes());
    PRIMUS_TURBO_CHECK(is_combined_token_in_rank.size(1) == num_ranks);
    PRIMUS_TURBO_CHECK(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and
                       rdma_channel_prefix_matrix.size(1) == num_channels);
    PRIMUS_TURBO_CHECK(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    PRIMUS_TURBO_CHECK(gbl_channel_prefix_matrix.size(0) == num_ranks and
                       gbl_channel_prefix_matrix.size(1) == num_channels);
    PRIMUS_TURBO_CHECK(combined_rdma_head.dim() == 2 and
                       combined_rdma_head.size(0) == num_combined_tokens and
                       combined_rdma_head.size(1) == num_rdma_ranks);
    PRIMUS_TURBO_CHECK(combined_nvl_head.dim() == 2 and
                       combined_nvl_head.size(1) == NUM_MAX_XGMI_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (allocate_on_comm_stream) {
        PRIMUS_TURBO_CHECK(previous_event.has_value() and async);
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int    num_topk                  = 0;
    auto   combined_topk_weights     = std::optional<torch::Tensor>();
    float *topk_weights_ptr          = nullptr;
    float *combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        PRIMUS_TURBO_CHECK(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        PRIMUS_TURBO_CHECK(topk_weights->size(0) == num_tokens);
        PRIMUS_TURBO_CHECK(topk_weights->scalar_type() == torch::kFloat32);
        num_topk         = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights =
            torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    PRIMUS_TURBO_CHECK(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    PRIMUS_TURBO_CHECK(config.num_max_nvl_chunked_send_tokens <=
                       config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    primus_turbo::deep_ep::internode::cached_notify(
        hidden_int4, 0, 0, num_topk, num_ranks, num_channels, num_combined_tokens,
        combined_rdma_head.data_ptr<int>(), rdma_channel_prefix_matrix.data_ptr<int>(),
        rdma_rank_prefix_sum.data_ptr<int>(), combined_nvl_head.data_ptr<int>(), rdma_buffer_ptr,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
        config.num_max_nvl_chunked_recv_tokens, task_fifo_ptrs_gpu, head, rank, comm_stream,
        config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks), num_nvl_bytes,
        false, low_latency_mode);
    move_fifo_slots(2);

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    primus_turbo::deep_ep::internode::combine(
        at::cuda::ScalarTypeToCudaDataType(x.scalar_type()), combined_x.data_ptr(),
        combined_topk_weights_ptr, is_combined_token_in_rank.data_ptr<bool>(), x.data_ptr(),
        topk_weights_ptr, combined_rdma_head.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
        src_meta.data_ptr(), rdma_channel_prefix_matrix.data_ptr<int>(),
        rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(), num_tokens,
        num_combined_tokens, hidden, num_topk, rdma_buffer_ptr,
        config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
        buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens, rank, num_ranks, comm_stream, num_channels,
        low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto &t : {x, src_meta, is_combined_token_in_rank, rdma_channel_prefix_matrix,
                        rdma_rank_prefix_sum, gbl_channel_prefix_matrix, combined_x,
                        combined_rdma_head, combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto &to : {topk_weights, combined_topk_weights}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::hip::setCurrentHIPStreamMasqueradingAsCUDA(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden,
                                      int num_experts) {
    PRIMUS_TURBO_CHECK(false, "Low-latency mode is disabled");
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor,
           std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor &x, const torch::Tensor &topk_idx,
                             int num_max_dispatch_tokens_per_rank, int num_experts, bool use_fp8,
                             bool async, bool return_recv_hook) {
    PRIMUS_TURBO_CHECK(false, "Low-latency mode is disabled");
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_combine(const torch::Tensor &x, const torch::Tensor &topk_idx,
                            const torch::Tensor &topk_weights, const torch::Tensor &src_info,
                            const torch::Tensor &layout_range, int num_max_dispatch_tokens_per_rank,
                            int num_experts, bool zero_copy, bool async, bool return_recv_hook,
                            const std::optional<torch::Tensor> &out) {

    PRIMUS_TURBO_CHECK(false, "Low-latency mode is disabled");
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank,
                                                          int hidden, int num_experts) {
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks,
                            num_experts);
    auto             buffer = layout.buffers[low_latency_buffer_idx];
    auto             dtype  = torch::kBFloat16;
    auto             num_msg_elems =
        static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    PRIMUS_TURBO_CHECK(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(
        buffer.combine_rdma_send_buffer_data_start,
        {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
        {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
}

std::string Buffer::get_local_ipc_handle_string() const {
    return std::string(reinterpret_cast<const char *>(ipc_handles[nvl_rank].reserved),
                       HIP_IPC_HANDLE_SIZE);
}

std::string Buffer::get_local_nvshmem_unique_id_string() const {
    PRIMUS_TURBO_CHECK(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
    auto unique_id = primus_turbo::deep_ep::internode::get_unique_id();
    return std::string(reinterpret_cast<const char *>(unique_id.data()), unique_id.size());
}

void Buffer::sync_string(const std::vector<int>         &device_ids,
                         const std::vector<std::string> &all_gathered_handles,
                         const std::string              &root_unique_id_opt) {
    std::vector<std::optional<pybind11::bytearray>> py_all_gathered_handles;
    for (auto &handle : all_gathered_handles) {
        std::optional<pybind11::bytearray> py_handle_opt = std::nullopt;
        if (!handle.empty()) {
            py_handle_opt.emplace(handle.c_str(), handle.size());
        }
        py_all_gathered_handles.push_back(py_handle_opt);
    }
    std::optional<pybind11::bytearray> py_root_unique_id_opt = std::nullopt;
    if (!root_unique_id_opt.empty()) {
        py_root_unique_id_opt.emplace(root_unique_id_opt.c_str(), root_unique_id_opt.size());
    }
    sync(device_ids, py_all_gathered_handles, py_root_unique_id_opt);
}

} // namespace primus_turbo::pytorch::deep_ep
