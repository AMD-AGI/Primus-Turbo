/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#include "primus_turbo/common.h"
#include "primus_turbo/deep_ep/api.h"
#include <atomic>
#include <barrier>
#include <chrono>
#include <hip/hip_runtime.h>
#include <memory>
#include <pybind11/functional.h>

#include "jax/deep_ep/deep_ep.h"
#include "jax/ffi.h"

namespace primus_turbo::jax::deep_ep {

std::barrier g_barrier_signal(NUM_MAX_NVL_PEERS);

static std::vector<std::unique_ptr<Buffer>> g_buffer_pool(NUM_MAX_NVL_PEERS);

Buffer *get_buffer(int rank, int num_ranks, int64_t hidden_bytes,
                   const primus_turbo::deep_ep::Config &config) {

    PRIMUS_TURBO_CHECK(num_ranks == NUM_MAX_NVL_PEERS and
                       "DeepEP only support intranode communication on Jax!");

    int device_id = -1;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));

    // auto num_nvl_bytes = config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks);
    int64_t num_nvl_bytes = 1024 * 1024 * 1024;
    // auto num_rdma_bytes = config.get_rdma_buffer_size_hint(hidden_bytes, num_ranks);
    int64_t num_rdma_bytes = 0;
    if (g_buffer_pool[device_id] == nullptr or g_buffer_pool[device_id]->rank() != rank or
        g_buffer_pool[device_id]->num_ranks() != num_ranks or
        g_buffer_pool[device_id]->num_nvl_bytes() < num_nvl_bytes or
        g_buffer_pool[device_id]->num_rdma_bytes() < num_rdma_bytes) {
        g_buffer_pool[device_id] =
            std::make_unique<Buffer>(rank, num_ranks, num_nvl_bytes, num_rdma_bytes, true);
        g_barrier_signal.arrive_and_wait();
        g_buffer_pool[device_id]->Sync();
    }
    return g_buffer_pool[device_id].get();
}

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes,
               bool explicitly_destroy)
    : num_nvl_bytes_(num_nvl_bytes), num_rdma_bytes_(num_rdma_bytes), rank_(rank),
      num_ranks_(num_ranks), explicitly_destroy_(explicitly_destroy) {
    // Metadata memory
    int64_t barrier_signal_bytes     = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes         = NUM_MAX_NVL_PEERS * sizeof(void *);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int *);

    // Common checks
    PRIMUS_TURBO_CHECK(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    PRIMUS_TURBO_CHECK(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (num_rdma_bytes <= std::numeric_limits<int>::max()));
    PRIMUS_TURBO_CHECK(0 <= rank and rank < num_ranks and
                       (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS));
    PRIMUS_TURBO_CHECK(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        PRIMUS_TURBO_CHECK(num_ranks > NUM_MAX_NVL_PEERS);

    // Get ranks
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id_));
    rdma_rank_ = rank / NUM_MAX_NVL_PEERS, nvl_rank_ = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks_ = std::max(1, num_ranks / NUM_MAX_NVL_PEERS),
    num_nvl_ranks_  = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    // Get device info
    hipDeviceProp_t device_prop = {};
    PRIMUS_TURBO_CHECK_HIP(hipGetDeviceProperties(&device_prop, device_id_));
    num_device_sms_ = device_prop.multiProcessorCount;

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        PRIMUS_TURBO_CHECK_HIP(hipExtMallocWithFlags(
            &buffer_ptrs_[nvl_rank_],
            num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes,
            hipDeviceMallocUncached));
        PRIMUS_TURBO_CHECK_HIP(
            hipIpcGetMemHandle(&ipc_handles_[nvl_rank_], buffer_ptrs_[nvl_rank_]));
        buffer_ptrs_gpu_ = reinterpret_cast<void **>(
            static_cast<uint8_t *>(buffer_ptrs_[nvl_rank_]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs_[nvl_rank_] = reinterpret_cast<int *>(
            static_cast<uint8_t *>(buffer_ptrs_[nvl_rank_]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu_ =
            reinterpret_cast<int **>(static_cast<uint8_t *>(buffer_ptrs_[nvl_rank_]) +
                                     num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        PRIMUS_TURBO_CHECK_HIP(hipMemset(barrier_signal_ptrs_[nvl_rank_], 0, barrier_signal_bytes));
    }

    // Create 32 MiB workspace
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(&workspace_, NUM_WORKSPACE_BYTES));
    PRIMUS_TURBO_CHECK_HIP(hipMemset(workspace_, 0, NUM_WORKSPACE_BYTES));

    // MoE counter
    PRIMUS_TURBO_CHECK_HIP(hipHostMalloc(&moe_recv_counter_, sizeof(int64_t), hipHostAllocMapped));
    PRIMUS_TURBO_CHECK_HIP(
        hipHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_counter_mapped_),
                                const_cast<int *>(moe_recv_counter_), 0));
    *moe_recv_counter_ = -1;

    // MoE expert-level counter
    PRIMUS_TURBO_CHECK_HIP(hipHostMalloc(&moe_recv_expert_counter_,
                                         sizeof(int) * NUM_MAX_LOCAL_EXPERTS, hipHostAllocMapped));
    PRIMUS_TURBO_CHECK_HIP(
        hipHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_expert_counter_mapped_),
                                const_cast<int *>(moe_recv_expert_counter_), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter_[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks_ > 0) {
        PRIMUS_TURBO_CHECK_HIP(
            hipHostMalloc(&moe_recv_rdma_counter_, sizeof(int), hipHostAllocMapped));
        PRIMUS_TURBO_CHECK_HIP(
            hipHostGetDevicePointer(reinterpret_cast<void **>(&moe_recv_rdma_counter_mapped_),
                                    const_cast<int *>(moe_recv_rdma_counter_), 0));
        *moe_recv_rdma_counter_ = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy_) {
        Destroy();
    } else if (not destroyed_) {
        printf("WARNING: Destroy() was not called before DeepEP buffer destruction, which can leak "
               "resources.\n");
        fflush(stdout);
    }
}

void Buffer::Destroy() {
    PRIMUS_TURBO_CHECK(not destroyed_);

    // Synchronize
    PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());

    if (num_nvl_bytes_ > 0 and is_available()) {
        // Barrier
        primus_turbo::deep_ep::intranode::barrier(barrier_signal_ptrs_gpu_, nvl_rank_,
                                                  num_nvl_ranks_, 0);
        PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());

        // Free local buffer and error flag
        PRIMUS_TURBO_CHECK_HIP(hipFree(buffer_ptrs_[nvl_rank_]));
    }
}

bool Buffer::is_internode_available() const {
    PRIMUS_TURBO_CHECK(false and "not implemented");
    return false; // TODO: implement this
}

void Buffer::DispatchLayout(
    hipStream_t stream, ffi::Buffer<ffi::S64> topk_idx, int num_experts,
    ffi::Result<ffi::Buffer<ffi::S32>>                num_tokens_per_rank,
    std::optional<ffi::Result<ffi::Buffer<ffi::S32>>> num_tokens_per_rdma_rank,
    ffi::Result<ffi::Buffer<ffi::S32>>                num_tokens_per_expert,
    ffi::Result<ffi::Buffer<ffi::PRED>>               is_token_in_rank) {
    PRIMUS_TURBO_CHECK(topk_idx.dimensions().size() == 2);
    PRIMUS_TURBO_CHECK(num_experts > 0);

    auto num_tokens = static_cast<int>(topk_idx.dimensions()[0]),
         num_topk   = static_cast<int>(topk_idx.dimensions()[1]);

    int *num_tokens_per_rdma_rank_ptr = nullptr;
    if (num_tokens_per_rdma_rank.has_value())
        num_tokens_per_rdma_rank_ptr = num_tokens_per_rdma_rank.value()->typed_data();

    primus_turbo::deep_ep::layout::get_dispatch_layout(
        topk_idx.typed_data(), num_tokens_per_rank->typed_data(), num_tokens_per_rdma_rank_ptr,
        num_tokens_per_expert->typed_data(), is_token_in_rank->typed_data(), num_tokens, num_topk,
        num_ranks_, num_experts, stream);
}

void Buffer::IntranodeDispatch(
    hipStream_t stream, ffi::AnyBuffer x, std::optional<ffi::Buffer<ffi::F32>> x_scales,
    std::optional<ffi::Buffer<ffi::S64>> topk_idx,
    std::optional<ffi::Buffer<ffi::F32>> topk_weights,
    std::optional<ffi::Buffer<ffi::S32>> num_tokens_per_rank,
    ffi::Buffer<ffi::PRED>               is_token_in_rank,
    std::optional<ffi::Buffer<ffi::S32>> num_tokens_per_expert, int cached_num_recv_tokens,
    std::optional<ffi::Buffer<ffi::S32>> cached_rank_prefix_matrix,
    std::optional<ffi::Buffer<ffi::S32>> cached_channel_prefix_matrix, int expert_alignment,
    int num_worst_tokens, primus_turbo::deep_ep::Config config, ffi::Result<ffi::AnyBuffer> recv_x,
    std::optional<ffi::Result<ffi::Buffer<ffi::F32>>> recv_x_scales,
    std::optional<ffi::Result<ffi::Buffer<ffi::S64>>> recv_topk_idx,
    std::optional<ffi::Result<ffi::Buffer<ffi::F32>>> recv_topk_weights,
    std::optional<ffi::Result<ffi::Buffer<ffi::S32>>> rank_prefix_matrix,
    std::optional<ffi::Result<ffi::Buffer<ffi::S32>>> channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>>                recv_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32>> recv_src_idx, ffi::Result<ffi::Buffer<ffi::S32>> send_head) {

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

    // Shape and contiguous checks
    PRIMUS_TURBO_CHECK(x.dimensions().size() == 2);
    PRIMUS_TURBO_CHECK((x.dimensions()[1] * x.element_count()) % sizeof(int4) == 0);
    PRIMUS_TURBO_CHECK(is_token_in_rank.dimensions().size() == 2);
    PRIMUS_TURBO_CHECK(is_token_in_rank.dimensions()[0] == x.dimensions()[0] and
                       is_token_in_rank.dimensions()[1] == num_ranks_);

    if (cached_mode) {
        PRIMUS_TURBO_CHECK(cached_rank_prefix_matrix->dimensions().size() == 2);
        PRIMUS_TURBO_CHECK(cached_rank_prefix_matrix->dimensions()[0] == num_ranks_ and
                           cached_rank_prefix_matrix->dimensions()[1] == num_ranks_);
        PRIMUS_TURBO_CHECK(cached_channel_prefix_matrix->dimensions().size() == 2);
        PRIMUS_TURBO_CHECK(cached_channel_prefix_matrix->dimensions()[0] == num_ranks_ and
                           cached_channel_prefix_matrix->dimensions()[1] == num_channels);
    } else {
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->dimensions().size() == 1);
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->dimensions()[0] % num_ranks_ == 0);
        PRIMUS_TURBO_CHECK(num_tokens_per_expert->dimensions()[0] / num_ranks_ <=
                           NUM_MAX_LOCAL_EXPERTS);
        PRIMUS_TURBO_CHECK(num_tokens_per_rank->dimensions().size() == 1 and
                           num_tokens_per_rank->dimensions()[0] == num_ranks_);
    }

    auto num_tokens = static_cast<int>(x.dimensions()[0]),
         hidden     = static_cast<int>(x.dimensions()[1]);

    PRIMUS_TURBO_CHECK(num_worst_tokens > num_tokens);
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->dimensions()[0]),
         num_local_experts = num_experts / num_ranks_;

    // Top-k checks
    int      num_topk         = 0;
    int64_t *topk_idx_ptr     = nullptr;
    float   *topk_weights_ptr = nullptr;
    PRIMUS_TURBO_CHECK(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->dimensions()[1]);
        PRIMUS_TURBO_CHECK(num_experts > 0);
        PRIMUS_TURBO_CHECK(topk_idx->dimensions().size() == 2);
        PRIMUS_TURBO_CHECK(topk_weights->dimensions().size() == 2);
        PRIMUS_TURBO_CHECK(num_tokens == topk_idx->dimensions()[0] and
                           num_tokens == topk_weights->dimensions()[0]);
        PRIMUS_TURBO_CHECK(num_topk == topk_weights->dimensions()[1]);
        topk_idx_ptr     = topk_idx->typed_data();
        topk_weights_ptr = topk_weights->typed_data();
    }

    // FP8 scales checks
    float *x_scales_ptr = nullptr;
    int    num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        PRIMUS_TURBO_CHECK(x_scales->element_count() == num_tokens);
        PRIMUS_TURBO_CHECK(x_scales->dimensions().size() == 2);
        PRIMUS_TURBO_CHECK(x_scales->dimensions()[0] == num_tokens);
        num_scales =
            x_scales->dimensions()[1] == 1 ? 1 : static_cast<int>(x_scales->dimensions()[1]);
        x_scales_ptr        = x_scales->typed_data();
        scale_token_stride  = static_cast<int>(x_scales->dimensions()[1]);
        scale_hidden_stride = 1;
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks_ * 4;
    if (cached_mode) {
        num_recv_tokens       = cached_num_recv_tokens;
        rank_prefix_matrix    = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();
        // Copy rank prefix matrix and clean flags
        primus_turbo::deep_ep::intranode::cached_notify_dispatch(
            rank_prefix_matrix.value()->typed_data(), num_memset_int, buffer_ptrs_gpu_,
            barrier_signal_ptrs_gpu_, rank_, num_ranks_, stream);
    } else {

        /// Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks,
        //  num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter_ = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter_[i] = -1;
        PRIMUS_TURBO_CHECK(static_cast<int64_t>(num_ranks_ * (num_ranks_ + num_local_experts) *
                                                sizeof(int)) <= num_nvl_bytes_);
        PRIMUS_TURBO_CHECK(channel_prefix_matrix.has_value());
        PRIMUS_TURBO_CHECK(rank_prefix_matrix.has_value());

        primus_turbo::deep_ep::intranode::notify_dispatch(
            num_tokens_per_rank->typed_data(), moe_recv_counter_mapped_, num_ranks_,
            num_tokens_per_expert->typed_data(), moe_recv_expert_counter_mapped_, num_experts,
            num_tokens, is_token_in_rank.typed_data(), channel_prefix_matrix.value()->typed_data(),
            rank_prefix_matrix.value()->typed_data(), num_memset_int, expert_alignment,
            buffer_ptrs_gpu_, barrier_signal_ptrs_gpu_, rank_, stream, num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            PRIMUS_TURBO_CHECK(topk_idx.has_value());
            PRIMUS_TURBO_CHECK(topk_weights.has_value());
        } else {
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter_);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter_[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - start_time)
                        .count() > NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
        }
    }

    // Assign pointers
    int64_t *recv_topk_idx_ptr     = nullptr;
    float   *recv_topk_weights_ptr = nullptr;
    float   *recv_x_scales_ptr     = nullptr;

    if (topk_idx.has_value()) {
        recv_topk_idx_ptr     = recv_topk_idx.value()->typed_data();
        recv_topk_weights_ptr = recv_topk_weights.value()->typed_data();
    }
    if (x_scales.has_value()) {
        recv_x_scales_ptr = recv_x_scales.value()->typed_data();
    }

    // Dispatch
    PRIMUS_TURBO_CHECK(
        static_cast<int64_t>(num_ranks_ * num_ranks_ * sizeof(int) +       // Size prefix  matrix
                             num_channels * num_ranks_ * sizeof(int) +     // Channel start offset
                             num_channels * num_ranks_ * sizeof(int) +     // Channel end offset
                             num_channels * num_ranks_ * sizeof(int) * 2 + // Queue head and tail
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 hidden * ffi::ByteWidth(recv_x->element_type()) + // Data buffer
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 sizeof(int) + // Source index buffer
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 num_topk * sizeof(int64_t) + // Top-k index buffer
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 num_topk * sizeof(float) + // Top-k weight buffer
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 sizeof(float) * num_scales // FP8 scale buffer
                             ) <= num_nvl_bytes_);

    primus_turbo::deep_ep::intranode::dispatch(
        recv_x->untyped_data(), recv_x_scales_ptr, recv_src_idx->typed_data(), recv_topk_idx_ptr,
        recv_topk_weights_ptr, recv_channel_prefix_matrix->typed_data(), send_head->typed_data(),
        x.untyped_data(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
        is_token_in_rank.typed_data(), channel_prefix_matrix.value()->typed_data(), num_tokens,
        num_worst_tokens,
        static_cast<int>(hidden * ffi::ByteWidth(recv_x->element_type()) / sizeof(int4)), num_topk,
        num_experts, num_scales, scale_token_stride, scale_hidden_stride, buffer_ptrs_gpu_, rank_,
        num_ranks_, stream, config.num_sms, config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens);
}

void Buffer::Sync() {
    PRIMUS_TURBO_CHECK(not is_available());

    if (num_nvl_bytes_ > 0) {
        for (int i = 0; i < num_nvl_ranks_; ++i) {
            if (i != rank_) {
                barrier_signal_ptrs_[i] = reinterpret_cast<int *>(
                    static_cast<uint8_t *>(g_buffer_pool[i]->buffer_ptrs_[i]) + num_nvl_bytes_);

                buffer_ptrs_[i]     = g_buffer_pool[i]->buffer_ptrs_[i];
                int can_access_peer = 0;
                PRIMUS_TURBO_CHECK_HIP(hipDeviceCanAccessPeer(&can_access_peer, device_id_, i));
                PRIMUS_TURBO_CHECK(can_access_peer != -1);

                hipDeviceEnablePeerAccess(i, 0);
            }
        }
    }

    // Copy all buffer and barrier signal pointers to GPU
    PRIMUS_TURBO_CHECK_HIP(hipMemcpy(buffer_ptrs_gpu_, buffer_ptrs_,
                                     sizeof(void *) * NUM_MAX_NVL_PEERS, hipMemcpyHostToDevice));
    PRIMUS_TURBO_CHECK_HIP(hipMemcpy(barrier_signal_ptrs_gpu_, barrier_signal_ptrs_,
                                     sizeof(int *) * NUM_MAX_NVL_PEERS, hipMemcpyHostToDevice));
    PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());

    // Ready to use
    is_available_ = true;
}

void Buffer::IntranodeCombine(hipStream_t stream, ffi::AnyBuffer x,
                              std::optional<ffi::Buffer<ffi::F32>> topk_weights,
                              std::optional<ffi::AnyBuffer>        bias_0,
                              std::optional<ffi::AnyBuffer> bias_1, ffi::Buffer<ffi::S32> src_idx,
                              ffi::Buffer<ffi::S32> rank_prefix_matrix,
                              ffi::Buffer<ffi::S32> channel_prefix_matrix,
                              ffi::Buffer<ffi::S32> send_head, primus_turbo::deep_ep::Config config,
                              ffi::Result<ffi::AnyBuffer>                       recv_x,
                              std::optional<ffi::Result<ffi::Buffer<ffi::F32>>> recv_topk_weights) {

    PRIMUS_TURBO_CHECK(x.dimensions().size() == 2);
    PRIMUS_TURBO_CHECK(src_idx.dimensions().size() == 1);
    PRIMUS_TURBO_CHECK(send_head.dimensions().size() == 2);
    PRIMUS_TURBO_CHECK(rank_prefix_matrix.dimensions().size() == 2);
    PRIMUS_TURBO_CHECK(channel_prefix_matrix.dimensions().size() == 2);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for
    // receiving.
    PRIMUS_TURBO_CHECK(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens      = static_cast<int>(x.dimensions()[0]),
         hidden          = static_cast<int>(x.dimensions()[1]);
    auto num_recv_tokens = static_cast<int>(send_head.dimensions()[0]);

    PRIMUS_TURBO_CHECK(src_idx.dimensions()[0] == num_tokens);
    PRIMUS_TURBO_CHECK(send_head.dimensions()[1] == num_ranks_);
    PRIMUS_TURBO_CHECK(rank_prefix_matrix.dimensions()[0] == num_ranks_ and
                       rank_prefix_matrix.dimensions()[1] == num_ranks_);
    PRIMUS_TURBO_CHECK(channel_prefix_matrix.dimensions()[0] == num_ranks_ and
                       channel_prefix_matrix.dimensions()[1] == num_channels);
    PRIMUS_TURBO_CHECK((hidden * ffi::ByteWidth(x.element_type())) % sizeof(int4) == 0);

    int    num_topk              = 0;
    float *topk_weights_ptr      = nullptr;
    float *recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        PRIMUS_TURBO_CHECK(topk_weights->dimensions().size() == 2);
        PRIMUS_TURBO_CHECK(topk_weights->dimensions()[0] == num_tokens);
        num_topk              = static_cast<int>(topk_weights->dimensions()[1]);
        topk_weights_ptr      = topk_weights.value().typed_data();
        recv_topk_weights_ptr = recv_topk_weights.value()->typed_data();
    }

    // Launch barrier and reset queue head and tail
    PRIMUS_TURBO_CHECK(static_cast<int64_t>(num_channels * num_ranks_ * sizeof(int) * 2) <=
                       num_nvl_bytes_);
    primus_turbo::deep_ep::intranode::cached_notify_combine(
        buffer_ptrs_gpu_, send_head.typed_data(), num_channels, num_recv_tokens,
        num_channels * num_ranks_ * 2, barrier_signal_ptrs_gpu_, rank_, num_ranks_, stream);

    // Assign bias pointers
    auto  bias_opts    = std::vector<std::optional<ffi::AnyBuffer>>({bias_0, bias_1});
    void *bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            PRIMUS_TURBO_CHECK(bias.dimensions().size() == 2);
            PRIMUS_TURBO_CHECK(bias.element_type() == x.element_type());
            PRIMUS_TURBO_CHECK(bias.dimensions()[0] == num_recv_tokens and
                               bias.dimensions()[1] == hidden);
            bias_ptrs[i] = bias.untyped_data();
        }

    // Combine data
    PRIMUS_TURBO_CHECK(
        static_cast<int64_t>(num_channels * num_ranks_ * sizeof(int) * 2 + // Queue head and tail
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 hidden * ffi::ByteWidth(x.element_type()) + // Data buffer
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 sizeof(int) + // Source index buffer
                             num_channels * num_ranks_ * config.num_max_nvl_chunked_recv_tokens *
                                 num_topk * sizeof(float) // Top-k weight buffer
                             ) <= num_nvl_bytes_);
    primus_turbo::deep_ep::intranode::combine(
        primus_turbo::jax::FFIDataTypeToHIPDataType(x.element_type()), recv_x->untyped_data(),
        recv_topk_weights_ptr, x.untyped_data(), topk_weights_ptr, bias_ptrs[0], bias_ptrs[1],
        src_idx.typed_data(), rank_prefix_matrix.typed_data(), channel_prefix_matrix.typed_data(),
        send_head.typed_data(), num_tokens, num_recv_tokens, hidden, num_topk, buffer_ptrs_gpu_,
        rank_, num_ranks_, stream, config.num_sms, config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens);
}

} // namespace primus_turbo::jax::deep_ep
