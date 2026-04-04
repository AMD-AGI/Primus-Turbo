#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#define PRINT_DEBUG(...)                                                                           \
    if (get_lane_id() == 0)                                                                        \
        printf(__VA_ARGS__);

namespace primus_turbo::cco::ep {
namespace intranode {

template <int kNumRanks>
__global__ void
notify_dispatch(const int *num_tokens_per_rank, int *moe_recv_counter,
                const int *num_tokens_per_expert, int *moe_recv_expert_counter, int num_experts,
                int num_tokens, int num_channels, const bool *is_token_in_rank,
                int *channel_prefix_matrix, int *rank_prefix_matrix_copy, int num_memset_int,
                int expert_alignment, void **buffer_ptrs, int **barrier_signal_ptrs, int rank) {
    auto sm_id     = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % WARP_SIZE, warp_id = thread_id / WARP_SIZE,
         num_warps = num_threads / WARP_SIZE;

    if (sm_id == 0) {
        // Barrier first
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {
            per_rank_buffer   = static_cast<int *>(buffer_ptrs[thread_id]);
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert
        //  j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
            per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];
#pragma unroll
            for (int i = 0; i < num_experts_per_rank; ++i)
                per_expert_buffer[rank * num_experts_per_rank + i] =
                    num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }

        // Wait for all ranks to be finished
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        auto local_per_rank_buffer = static_cast<int *>(buffer_ptrs[rank]);
        if (thread_id < kNumRanks) {
#pragma unroll
            for (int i = 1; i < kNumRanks; ++i)
                local_per_rank_buffer[i * kNumRanks + thread_id] +=
                    local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
            if (thread_id == rank)
                *moe_recv_counter = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            moe_recv_expert_counter[thread_id] = sum;
        }
        __syncthreads();

// Copy rank size prefix matrix to another tensor
#pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

// Extra memset for later communication queue
#pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_buffer[i] = 0;

        // Barrier
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        int dst_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx,
                                   token_end_idx);

            // Iterate over tokens
            int count = 0;
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += WARP_SIZE)
                count += is_token_in_rank[i * kNumRanks + dst_rank];
            count = warp_reduce_sum(count);
            if (lane_id == 0)
                channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
        __syncthreads();

        // Pre-compute prefix sum for all channels
        if (thread_id == 0) {
#pragma unroll
            for (int i = 1; i < num_channels; ++i)
                channel_prefix_matrix[dst_rank * num_channels + i] +=
                    channel_prefix_matrix[dst_rank * num_channels + i - 1];
        }
    }
}

void notify_dispatch(const int *num_tokens_per_rank, int *moe_recv_counter, int num_ranks,
                     const int *num_tokens_per_expert, int *moe_recv_expert_counter,
                     int num_experts, int num_tokens, const bool *is_token_in_rank,
                     int *channel_prefix_matrix, int *rank_prefix_matrix_copy, int num_memset_int,
                     int expert_alignment, void **buffer_ptrs, int **barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                         \
    LAUNCH_KERNEL(&cfg, notify_dispatch<ranks>, num_tokens_per_rank, moe_recv_counter,             \
                  num_tokens_per_expert, moe_recv_expert_counter, num_experts, num_tokens,         \
                  num_channels, is_token_in_rank, channel_prefix_matrix, rank_prefix_matrix_copy,  \
                  num_memset_int, expert_alignment, buffer_ptrs, barrier_signal_ptrs, rank);       \
    break

    constexpr int kNumThreads = 128;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

    SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch_with_permute(void **buffer_ptrs, int4 const *x, float const *x_scales,
                          int64_t const *topk_idx, float const *topk_weights,
                          bool const *is_token_in_rank, int const *channel_prefix_matrix,
                          int const *row_id_map, int4 *recv_x, int num_tokens, int hidden_int4,
                          int num_topk, int num_experts, int num_scales, int scale_token_stride,
                          int scale_hidden_stride, int rank, int num_max_tokens,
                          int num_max_send_tokens) {
    auto const num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    auto const thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    const auto num_threads_per_rank = kNumThreads / kNumRanks;
    auto const num_channels         = num_sms / 2;
    const auto responsible_rank     = (static_cast<int>(thread_id)) / num_threads_per_rank;
    auto const responsible_channel  = sm_id / 2;

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= WARP_SIZE);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));

    auto ptr = buffer_ptrs[is_sender ? responsible_rank : rank];

    auto rank_prefix_matrix = Buffer<int>(ptr, kNumRanks * kNumRanks);

    // Channel buffer metadata — placed right after rank_prefix_matrix so that
    // notify_dispatch's memset (which zeros from this offset) covers them.
    int  target_rank         = is_sender ? rank : responsible_rank;
    auto num_channels_total  = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset   = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx     = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

    auto num_tokens_total  = num_tokens * kNumRanks;
    auto channel_x_buffers = Buffer<int4>(ptr, num_tokens_total * hidden_int4);

    if (thread_id < MAX_NUM_BARRIERS)
        amd::barrier_init(thread_id);
    __syncthreads();

    // auto channel_src_idx_buffers = Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens,
    //                                            channel_rank_offset * num_recv_buffer_tokens);
    // auto channel_topk_idx_buffers =
    //     Buffer<int64_t>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
    //                     channel_rank_offset * num_recv_buffer_tokens * num_topk);
    // auto channel_topk_weights_buffers =
    //     Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
    //                   channel_rank_offset * num_recv_buffer_tokens * num_topk);
    // auto channel_x_scales_buffers =
    //     Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_scales,
    //                   channel_rank_offset * num_recv_buffer_tokens * num_scales);

    if (is_sender) {
        constexpr int num_send_warps          = kNumThreads / WARP_SIZE;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto    send_thread_id          = thread_id;
        const auto    send_warp_id_in_rank    = send_thread_id % num_threads_per_rank / WARP_SIZE;
        EP_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx,
                               token_end_idx);

        // Calculate offset first
        int rank_offset =
            rank > 0 ? rank_prefix_matrix[(rank - 1) * kNumRanks + responsible_rank] : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        // `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        total_offset =
            responsible_channel > 0
                ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1]
                : 0;
        if (lane_id == 0)
            st_relaxed_sys_global(channel_start_offset.buffer(), -total_offset - 1);
        num_tokens_to_recv =
            channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
        if (lane_id == 0)
            st_relaxed_sys_global(channel_end_offset.buffer(), -num_tokens_to_recv - 1);
        num_tokens_to_recv -= total_offset;

        total_offset += rank_offset;

        // if (num_tokens_to_recv <= 0)
        //     return;

        // Iterate over all tokens and send tokens to symmetric buffer.
        // Track batch boundaries for flag signalling.
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may be different
                // from the warp assigned to send the following data
                // if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and lane_id == 0)
                //     send_head[token_idx * kNumRanks + responsible_rank] =
                //         is_token_in_rank[token_idx * kNumRanks + responsible_rank]
                //             ? cached_channel_tail_idx
                //             : -1;

                // Skip if not selected
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                auto dst_slot_idx = total_offset + cached_channel_tail_idx++;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers =
                        channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers,
                                       shifted_x, __ldg, st_na_global);

                    // // Copy source index
                    // if (lane_id == 0)
                    //     channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    // if (lane_id < num_topk) {
                    //     // Top-k index
                    //     int recv_expert_begin = responsible_rank * num_experts_per_rank,
                    //         recv_expert_end   = (responsible_rank + 1) * num_experts_per_rank;
                    //     auto idx_value        = __ldg(topk_idx + token_idx * num_topk + lane_id);
                    //     idx_value = (idx_value >= recv_expert_begin and idx_value <
                    //     recv_expert_end)
                    //                     ? idx_value - recv_expert_begin
                    //                     : -1;
                    //     channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                    //     // Top-k weights
                    //     auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                    //     weight_value      = (idx_value >= 0) ? weight_value : 0.0f;
                    //     channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                    //         weight_value;
                    // }

                    // // Copy `x_scales`
                    // #pragma unroll
                    //                     for (int i = lane_id; i < num_scales; i += 32) {
                    //                         auto offset = token_idx * scale_token_stride + i *
                    //                         scale_hidden_stride;
                    //                         channel_x_scales_buffers[dst_slot_idx * num_scales +
                    //                         i] =
                    //                             __ldg(x_scales + offset);
                    //                     }

                    // // Copy `sf_scale_for_nvfp4`
                    // #pragma unroll
                    //                     for (int i = lane_id; i < num_sf_scales_for_nvfp4; i +=
                    //                     32) {
                    //                         auto offset = token_idx *
                    //                         sf_scale_for_nvfp4_token_stride +
                    //                                       i * sf_scale_for_nvfp4_hidden_stride;
                    //                         channel_x_sf_scale_for_nvfp4_buffers[dst_slot_idx *
                    //                                                                  num_sf_scales_for_nvfp4
                    //                                                                  +
                    //                                                              i] =
                    //                             __ldg(sf_scale_for_nvfp4 + offset);
                    //                     }
                    //                 }
                }

                chunk_token_idx++;
                token_idx++;
            }

            // sync_barrier(responsible_rank, num_threads_per_rank);
            __syncthreads();
            __threadfence_system();

            if (send_warp_id_in_rank == 0 and lane_id == 0)
                st_release_sys_global<true>(channel_tail_idx.buffer(), cached_channel_tail_idx);
            __syncthreads();
        }
    } else {
        
        constexpr int num_recv_warps          = kNumThreads / WARP_SIZE;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto    recv_thread_id          = thread_id;
        const auto    recv_thread_id_in_rank  = recv_thread_id % num_threads_per_rank;
        const auto    recv_warp_id_in_rank    = recv_thread_id_in_rank / WARP_SIZE;
        EP_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        int rank_offset = responsible_rank > 0
                              ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank]
                              : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        if (lane_id == 0) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(WARP_MASK, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(WARP_MASK, num_tokens_to_recv, 0);

        __syncthreads();

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        int  cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        auto start_time = clock64();
        while (num_tokens_to_recv > 0) {
            // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by
            // different warps are the same
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global<true>(channel_tail_idx.buffer());

                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = "
                           "%d, tokens remained: %d\n",
                           rank, responsible_channel, num_tokens_to_recv);
                    trap();
                }
            }
            // sync_barrier(responsible_rank, num_threads_per_rank);
            __syncthreads();
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens;
                 chunk_idx += num_recv_warps_per_rank) {
                int  src_slot_idx = total_offset + chunk_idx;
                auto shifted_buffer_x_int4 =
                    channel_x_buffers.buffer() + src_slot_idx * hidden_int4;

                for (int e = 0; e < num_experts_per_rank; ++e) {
                    int target = row_id_map[src_slot_idx * num_experts_per_rank + e];
                    if (target > 0) {
                        auto shifted_recv_x_int4 =
                            recv_x + static_cast<int64_t>(target - 1) * hidden_int4;

                        UNROLLED_WARP_COPY(2, lane_id, hidden_int4, shifted_recv_x_int4,
                                           shifted_buffer_x_int4, ld_nc_global, st_na_global);
                    }
                }
            }
            // sync_barrier(responsible_rank, num_threads_per_rank);

            num_tokens_to_recv -= num_recv_tokens;
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
        }
    }
}

void dispatch_with_permute(void **workspace_ptrs, void const *x, float const *x_scales,
                           int64_t const *topk_idx, float const *topk_weights,
                           bool const *is_token_in_rank, int const *channel_prefix_matrix,
                           int const *row_id_map, void *recv_x, int num_tokens, int hidden_int4,
                           int num_topk, int num_experts, int num_scales, int scale_token_stride,
                           int scale_hidden_stride, int rank, int num_ranks, cudaStream_t stream,
                           int num_sms, int num_max_tokens, int num_max_send_tokens) {
    constexpr int kNumThreads = 1024;

#define DISPATCH_LAUNCH_CASE(ranks)                                                                \
    {                                                                                              \
        auto kernel = dispatch_with_permute<ranks, kNumThreads>;                                   \
        LAUNCH_KERNEL(&cfg, kernel, workspace_ptrs, reinterpret_cast<int4 const *>(x), x_scales,   \
                      topk_idx, topk_weights, is_token_in_rank, channel_prefix_matrix, row_id_map, \
                      reinterpret_cast<int4 *>(recv_x), num_tokens, hidden_int4, num_topk,         \
                      num_experts, num_scales, scale_token_stride, scale_hidden_stride, rank,      \
                      num_max_tokens, num_max_send_tokens);                                        \
    }                                                                                              \
    break

    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

} // namespace intranode

} // namespace primus_turbo::cco::ep
