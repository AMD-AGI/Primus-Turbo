#include "buffer.cuh"
#include "exception.cuh"
#include "intranode.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace primus_turbo::cco::pipelined_ep::intranode {

/*
input:
    num_tokens_per_rank: (num_ranks x num_stages)

output:
    moe_recv_counter_mapped: (num_stages,)
    moe_recv_expert_counter_mapped: (num_stages, num_experts)
    channel_prefix_matrix: (num_ranks x num_channels x num_stages)
    rank_prefix_matrix: (num_ranks x num_ranks x num_stages)
*/

template <int kNumRanks, int kNumStages>
__global__ void
notify_dispatch(int const *num_tokens_per_rank, int *moe_recv_counter_mapped,
                int const *num_tokens_per_expert, int *moe_recv_expert_counter_mapped,
                int num_experts, int num_tokens, int num_channels, bool const *is_token_in_rank,
                int *channel_prefix_matrix, int *rank_prefix_matrix_copy, int num_memset_int,
                int expert_alignment, void **buffer_ptrs, int **barrier_signal_ptrs, int rank) {
    auto sm_id     = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % WARP_SIZE, warp_id = thread_id / WARP_SIZE,
         num_warps = num_threads / WARP_SIZE;

    if (sm_id == 0) {
        int responsible_stage = thread_id < kNumRanks * kNumStages ? thread_id / kNumRanks : -1;
        int responsible_rank  = thread_id < kNumRanks * kNumStages ? thread_id % kNumRanks : -1;

        // Barrier first
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_stage_buffer, *per_expert_stage_buffer;
        if (thread_id < kNumRanks * kNumStages) {
            per_rank_stage_buffer = static_cast<int *>(buffer_ptrs[thread_id]);
            per_expert_stage_buffer     = per_rank_stage_buffer + kNumRanks * kNumRanks * kNumStages;
        }

        // After this loop:
        // per_rank_stage_buffer: (kNumRanks x kNumRanks x kNumStages)
        //  - `per_rank_buffer[rank][i, j, k]` means the number of tokens from rank i
        //  to rank j in stage k
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i
        //  to local expert j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks * kNumStages) {
#pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                per_rank_stage_buffer[rank * kNumRanks * kNumStages + i * kNumStages +
                                      responsible_stage] =
                    num_tokens_per_rank[i * kNumStages + responsible_stage];
            #pragma unroll
                        for (int i = 0; i < num_experts_per_rank; ++i)
                            per_expert_stage_buffer[rank * num_experts_per_rank + i] =
                                num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }

        // Wait for all ranks to be finished
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        auto local_per_rank_stage_buffer = static_cast<int *>(buffer_ptrs[rank]);
        if (thread_id < kNumRanks * kNumStages) {
#pragma unroll
            for (int i = 1; i < kNumRanks; ++i)
                local_per_rank_stage_buffer[i * kNumRanks * kNumStages + thread_id] +=
                    local_per_rank_stage_buffer[(i - 1) * kNumRanks * kNumStages + thread_id];
            if (responsible_rank == rank) {
                moe_recv_counter_mapped[responsible_stage] =
                    local_per_rank_stage_buffer[(kNumRanks - 1) * kNumRanks * kNumStages +
                                                rank * kNumStages + responsible_stage];
            }
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_stage_buffer =
            local_per_rank_stage_buffer + kNumRanks * kNumRanks * kNumStages;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }
        __syncthreads();

// Copy rank size prefix matrix to another tensor
#pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks * kNumStages; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_stage_buffer[i];

// Extra memset for later communication queue
#pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_stage_buffer[i] = 0;

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

void notify_dispatch(int const *num_tokens_per_rank, int *moe_recv_counter_mapped, int num_ranks,
                     int const *num_tokens_per_expert, int *moe_recv_expert_counter_mapped,
                     int num_experts, int num_tokens, bool const *is_token_in_rank,
                     int *channel_prefix_matrix, int *rank_prefix_matrix_copy, int num_memset_int,
                     int expert_alignment, void **buffer_ptrs, int **barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                         \
    LAUNCH_KERNEL(&cfg, notify_dispatch<ranks>, num_tokens_per_rank, moe_recv_counter_mapped,      \
                  num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts, num_tokens,  \
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

template <int kNumRanks>
__global__ void cached_notify_dispatch(int const *rank_prefix_matrix, int num_memset_int,
                                       void **buffer_ptrs, int **barrier_signal_ptrs, int rank) {
    // A simplified version for cached handles
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    // Copy and clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = static_cast<int *>(buffer_ptrs[rank]);
#pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
        ptr[i] = rank_prefix_matrix[i];
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
        ptr[kNumRanks * kNumRanks + i] = 0;

    // Barrier after cleaning
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void cached_notify_dispatch(int const *rank_prefix_matrix, int num_memset_int, void **buffer_ptrs,
                            int **barrier_signal_ptrs, int rank, int num_ranks,
                            cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                  \
    LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, rank_prefix_matrix, num_memset_int,         \
                  buffer_ptrs, barrier_signal_ptrs, rank);                                         \
    break

    SETUP_LAUNCH_CONFIG(1, 128, stream);
    SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <typename topk_idx_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch(int4 **recv_x, float **recv_x_scales, int **recv_src_idx, topk_idx_t **recv_topk_idx,
             float **recv_topk_weights, int **recv_channel_offset, int **send_head, int4 const *x,
             float const *x_scales, topk_idx_t const *topk_idx, float const *topk_weights,
             bool const *is_token_in_rank, int const *channel_prefix_matrix, int num_tokens,
             int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
             int scale_token_stride, int scale_hidden_stride, void **buffer_ptrs, int rank,
             int num_max_send_tokens, int num_recv_buffer_tokens, int num_stages) {
    auto const num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    auto const thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    bool const is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    // Several warps are response for a single rank
    auto const num_threads_per_rank = kNumThreads / kNumRanks;
    auto const num_channels         = num_sms / 2;
    auto const responsible_rank     = (static_cast<int>(thread_id)) / num_threads_per_rank;
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    auto const responsible_channel = sm_id / 2;

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= WARP_SIZE);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    // Calculate pointers by the specific layout
    // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    auto ptr = reinterpret_cast<void *>(
        static_cast<int8_t *>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
        kNumRanks * kNumRanks * sizeof(int));
    int  target_rank         = is_sender ? rank : responsible_rank;
    auto num_channels_total  = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;
    auto channel_stage_rank_offset =
        responsible_channel * kNumRanks * num_stages + target_rank * num_stages;
    auto num_channels_stage_total = channel_rank_offset * num_stages;
    // Channel buffer metadata
    // Senders are responsible for tails, and receivers are responsible for heads
    // Stored on the receiver side
    // The retired signals are actually boolean flags, but to align with 16 bytes,
    // we make it `int64_t` `start_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset   = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx     = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx     = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

    // we make it `int64_t` `start_offset`: kNumChannels * kNumRanks * kNumStages * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * kNumStages * sizeof(int)
    // `head_idx`: kNumChannels * kNumRanks * kNumStages * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * kNumStages * sizeof(int)
    auto channel_stage_start_offset =
        Buffer<int>(ptr, num_channels_stage_total, channel_stage_rank_offset);
    auto channel_stage_end_offset =
        Buffer<int>(ptr, num_channels_stage_total, channel_stage_rank_offset);
    auto channel_stage_head_idx =
        Buffer<int>(ptr, num_channels_stage_total, channel_stage_rank_offset);
    auto channel_stage_tail_idx =
        Buffer<int>(ptr, num_channels_stage_total, channel_stage_rank_offset);

    // Channel data buffers, stored on the receiver side
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens *
    // hidden_int4 * sizeof(int4) `src_idx_buffers`: kNumChannels * kNumRanks *
    // num_recv_buffer_tokens * sizeof(int) `topk_idx_buffers`: kNumChannels *
    // kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(int64_t)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens *
    // num_topk * sizeof(float) `x_scales_buffers`: kNumChannels * kNumRanks *
    // num_recv_buffer_tokens * num_scales * sizeof(float)
    auto channel_x_buffers =
        Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4,
                     channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers = Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens,
                                               channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_idx_buffers =
        Buffer<int64_t>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
                        channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_topk_weights_buffers =
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
                      channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_x_scales_buffers =
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_scales,
                      channel_rank_offset * num_recv_buffer_tokens * num_scales);

    if (is_sender) {
        // Workers for sending
        constexpr int num_send_warps          = kNumThreads / WARP_SIZE;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        auto const    send_thread_id          = thread_id;
        auto const    send_warp_id_in_rank    = send_thread_id % num_threads_per_rank / WARP_SIZE;
        EP_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (lane_id < num_stages and send_warp_id_in_rank == 0) {
            int value = responsible_channel > 0
                            ? channel_prefix_matrix[responsible_rank * num_channels +
                                                    responsible_channel - 1]
                            : 0;
            st_relaxed_sys_global(channel_stage_start_offset.buffer() + lane_id, -value - 1);
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
            st_relaxed_sys_global(channel_stage_end_offset.buffer() + lane_id, -value - 1);
        }
        __syncwarp();

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx,
                               token_end_idx);

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released
            // (rare cases) NOTES: the head index received by different warps may not
            // be the same
            auto start_time = clock64();
            while (lane_id == 0) {
                // NOTES: we only consider the worst case, because counting the real
                // numbers are time-consuming
                int num_used_slots =
                    cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                    break;

                // Rare cases to loop again
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch senders, rank %d, "
                           "responsible_channel = %d\n",
                           rank, responsible_channel);
                    trap();
                }
            }
            __syncwarp();

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may
                // be different from the warp assigned to send the following data
                if (lane_id == 0 and token_idx % num_send_warps_per_rank == send_warp_id_in_rank)
                    send_head[0][token_idx * kNumRanks + responsible_rank] =
                        is_token_in_rank[token_idx * kNumRanks + responsible_rank]
                            ? cached_channel_tail_idx
                            : -1;

                // Skip if not selected
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers =
                        channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers,
                                       shifted_x, __ldg, st_na_global);

                    // Copy source index
                    if (lane_id == 0)
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < num_topk) {
                        // Top-k index
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end   = (responsible_rank + 1) * num_experts_per_rank;
                        auto idx_value        = __ldg(topk_idx + token_idx * num_topk + lane_id);
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end)
                                        ? idx_value - recv_expert_begin
                                        : -1;
                        channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                        weight_value      = (idx_value >= 0) ? weight_value : 0.0f;
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                            weight_value;
                    }

// Copy `x_scales`
#pragma unroll
                    for (int i = lane_id; i < num_scales; i += WARP_SIZE) {
                        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                        channel_x_scales_buffers[dst_slot_idx * num_scales + i] =
                            __ldg(x_scales + offset);
                    }
                }

                // Move token index
                chunk_token_idx++, token_idx++;
            }

            // Move tail index
            // NOTES: here all warps should share the same new tail
            sync_barrier<true>(responsible_rank, num_threads_per_rank);
            if (send_warp_id_in_rank == 0 and lane_id == 0)
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps          = kNumThreads / WARP_SIZE;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        auto const    recv_thread_id          = thread_id;
        auto const    recv_thread_id_in_rank  = recv_thread_id % num_threads_per_rank;
        auto const    recv_warp_id_in_rank    = recv_thread_id_in_rank / WARP_SIZE;
        EP_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int *>(buffer_ptrs[rank]);
        int  rank_offset        = responsible_rank > 0
                                      ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank]
                                      : 0;

        for (int stage = 0; stage < num_stages; ++stage) {
            // Receive channel offset
            int total_offset, num_tokens_to_recv;
            while (lane_id == 0 and (total_offset = ld_volatile_global(
                                         channel_stage_start_offset.buffer() + stage)) == 0)
                ;
            while (lane_id == 0 and (num_tokens_to_recv = ld_volatile_global(
                                         channel_stage_end_offset.buffer() + stage)) == 0)
                ;
            if (lane_id == 0) {
                total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
                if (recv_warp_id_in_rank == 0)
                    recv_channel_offset[stage][responsible_rank * num_channels +
                                               responsible_channel] = total_offset;
                num_tokens_to_recv -= total_offset;
            }
            total_offset = __shfl_sync(WARP_MASK, total_offset, 0);
            total_offset += rank_offset;
            num_tokens_to_recv = __shfl_sync(WARP_MASK, num_tokens_to_recv, 0);

            // Shared tail indices for different warps
            __shared__ int volatile shared_channel_tail_idx[kNumRanks];

            auto start_time              = clock64();
            int  cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
            while (num_tokens_to_recv > 0) {
                // NOTES: unlike the sender, the receiver must ensure that the tail
                // indices hold by different warps are the same
                while (recv_thread_id_in_rank == 0) {
                    cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());

                    // Ready to copy
                    if (cached_channel_head_idx != cached_channel_tail_idx) {
                        shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                        break;
                    }

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for dispatch receivers, rank %d, "
                               "responsible_channel = %d, tokens remained: %d\n",
                               rank, responsible_channel, num_tokens_to_recv);
                        trap();
                    }
                }

                // Synchronize queue tail
                sync_barrier<true>(responsible_rank, num_threads_per_rank);
                cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

                // Copy data
                int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
                for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens;
                     chunk_idx += num_recv_warps_per_rank) {
                    int token_idx_in_buffer =
                        (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                    auto shifted_buffer_x_int4 =
                        channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
                    auto shifted_recv_x_int4 =
                        recv_x[stage] +
                        static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;

                    UNROLLED_WARP_COPY(2, lane_id, hidden_int4, shifted_recv_x_int4,
                                       shifted_buffer_x_int4, ld_nc_global, st_na_global);
                }

// Copy `src_idx`
// Support NC
#pragma unroll 4
                for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank;
                     chunk_idx < cached_channel_tail_idx;
                     chunk_idx += WARP_SIZE * num_recv_warps_per_rank)
                    recv_src_idx[stage][total_offset + chunk_idx - cached_channel_head_idx] =
                        ld_nc_global(channel_src_idx_buffers.buffer() +
                                     chunk_idx % num_recv_buffer_tokens);

// Copy `topk_idx` and `topk_weights`
#pragma unroll 4
                for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk;
                     idx += WARP_SIZE * num_recv_warps_per_rank) {
                    int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
                    int token_idx_in_buffer =
                        (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                    auto recv_idx =
                        static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
                    auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;

                    // Actual read
                    recv_topk_idx[stage][recv_idx] =
                        ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                    recv_topk_weights[stage][recv_idx] =
                        ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
                }

// Copy `x_scales`
#pragma unroll 4
                for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales;
                     i += WARP_SIZE * num_recv_warps_per_rank) {
                    int chunk_idx = i / num_scales, scales_idx = i % num_scales;
                    int token_idx_in_buffer =
                        (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                    recv_x_scales[stage]
                                 [static_cast<int64_t>(total_offset + chunk_idx) * num_scales +
                                  scales_idx] =
                                     ld_nc_global(channel_x_scales_buffers.buffer() +
                                                  token_idx_in_buffer * num_scales + scales_idx);
                }

                // Move queue
                cached_channel_head_idx += num_recv_tokens;
                total_offset += num_recv_tokens;
                sync_barrier<true>(responsible_rank, num_threads_per_rank);
                if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and lane_id == 0)
                    st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

                // Exit
                num_tokens_to_recv -= num_recv_tokens;
            }
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        auto       rank_prefix_matrix = static_cast<int *>(buffer_ptrs[rank]);
        auto const num_recv_tokens    = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        auto const clean_start        = num_recv_tokens * num_topk + sm_id * kNumThreads;
        auto const clean_end          = num_worst_tokens * num_topk;
        auto const clean_stride       = num_sms * kNumThreads;
        // #pragma unroll
        //         for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
        //             recv_topk_idx[0][i] = -1;
    }
}

template <typename topk_idx_t>
void dispatch(void **recv_x, float **recv_x_scales, int **recv_src_idx, topk_idx_t **recv_topk_idx,
              float **recv_topk_weights, int **recv_channel_offset, int **send_head, void const *x,
              float const *x_scales, topk_idx_t const *topk_idx, float const *topk_weights,
              bool const *is_token_in_rank, int const *channel_prefix_matrix, int num_tokens,
              int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
              int scale_token_stride, int scale_hidden_stride, void **buffer_ptrs, int rank,
              int num_ranks, cudaStream_t stream, int num_sms, int num_max_send_tokens,
              int num_recv_buffer_tokens, int num_stages) {
    constexpr int kNumThreads = 1024;

    constexpr int kNumTMABytesPerWarp = 8192;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / WARP_SIZE);
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride <
                   std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                                                \
    {                                                                                              \
        auto kernel = dispatch<topk_idx_t, ranks, kNumThreads, kNumTMABytesPerWarp>;               \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                                         \
        LAUNCH_KERNEL(&cfg, kernel, reinterpret_cast<int4 **>(recv_x), recv_x_scales,              \
                      recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_offset,         \
                      send_head, reinterpret_cast<int4 const *>(x), x_scales, topk_idx,            \
                      topk_weights, is_token_in_rank, channel_prefix_matrix, num_tokens,           \
                      num_worst_tokens, hidden_int4, num_topk, num_experts, num_scales,            \
                      scale_token_stride, scale_hidden_stride, buffer_ptrs, rank,                  \
                      num_max_send_tokens, num_recv_buffer_tokens, num_stages);                    \
    }                                                                                              \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}
} // namespace primus_turbo::cco::pipelined_ep::intranode
