#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace primus_turbo::cco::ep {
namespace intranode {

template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    fused_dispatch_permute(void **buffer_ptrs, int64_t *recv_topk_idx, float *recv_topk_weights,
                           int *dispatch_to_expert_map, int4 const *x, float const *x_scales,
                           int64_t const *topk_idx, float const *topk_weights,
                           bool const *is_token_in_rank, int const *channel_prefix_matrix,
                           int const *num_recv_tokens_per_expert, int4 *recv_x, int num_tokens,
                           int hidden_int4, int num_topk, int num_experts, int num_scales,
                           int scale_token_stride, int scale_hidden_stride, int rank,
                           int num_max_tokens, int num_max_send_tokens) {
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

    auto ptr = buffer_ptrs[is_sender ? responsible_rank : rank];

    auto rank_prefix_matrix = Buffer<int>(ptr, kNumRanks * kNumRanks);

    int  target_rank         = is_sender ? rank : responsible_rank;
    auto num_channels_total  = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset   = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx     = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto expert_slot_idx      = Buffer<int>(ptr, num_experts_per_rank);

    auto num_tokens_total                = num_tokens * kNumRanks;
    auto dispatched_x_buffers            = Buffer<int4>(ptr, num_tokens_total * hidden_int4);
    auto dispatched_topk_idx_buffers     = Buffer<int64_t>(ptr, num_tokens_total * num_topk);
    auto dispatched_topk_weights_buffers = Buffer<float>(ptr, num_tokens_total * num_topk);

    if (thread_id < MAX_NUM_BARRIERS)
        amd::barrier_init(thread_id);
    __syncthreads();

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

        int rank_offset =
            rank > 0 ? rank_prefix_matrix[(rank - 1) * kNumRanks + responsible_rank] : 0;

        int total_offset, num_tokens_to_recv;
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

        if (num_tokens_to_recv <= 0)
            return;

        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                auto dst_slot_idx = total_offset + cached_channel_tail_idx++;
                bool is_my_slot =
                    (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank);

                if (is_my_slot) {
                    auto shifted_dispatched_x_buffers =
                        dispatched_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_dispatched_x_buffers,
                                       shifted_x, __ldg, st_na_global);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < num_topk) {
                        // Top-k index
                        int  recv_expert_begin = responsible_rank * num_experts_per_rank,
                             recv_expert_end   = (responsible_rank + 1) * num_experts_per_rank;
                        auto idx_value         = __ldg(topk_idx + token_idx * num_topk + lane_id);
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end)
                                        ? idx_value - recv_expert_begin
                                        : -1;
                        dispatched_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                        weight_value      = (idx_value >= 0) ? weight_value : 0.0f;
                        dispatched_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                            weight_value;
                    }
                }

                chunk_token_idx++;
                token_idx++;
            }

            sync_barrier(responsible_rank, num_threads_per_rank);

            if (send_warp_id_in_rank == 0 and lane_id == 0)
                st_release_sys_global<true>(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps          = kNumThreads / WARP_SIZE;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto    recv_thread_id          = thread_id;
        const auto    recv_thread_id_in_rank  = recv_thread_id % num_threads_per_rank;
        const auto    recv_warp_id_in_rank    = recv_thread_id_in_rank / WARP_SIZE;
        EP_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int *>(buffer_ptrs[rank]);
        int  rank_offset = responsible_rank > 0
                               ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank]
                               : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        while (lane_id == 0 and
               (total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
            ;
        while (lane_id == 0 and
               (num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
            ;
        if (lane_id == 0) {
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            // if (recv_warp_id_in_rank == 0)
            //     recv_channel_offset[responsible_rank * num_channels + responsible_channel] =
            //         total_offset;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(WARP_MASK, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(WARP_MASK, num_tokens_to_recv, 0);

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];
        __shared__ volatile int shared_expert_prefix_sum[NUM_MAX_LOCAL_EXPERTS];

        if (thread_id < num_experts_per_rank)
            shared_expert_prefix_sum[thread_id] = num_recv_tokens_per_expert[thread_id];
        __syncthreads();
        if (thread_id == 0) {
            int prev = 0;
            for (int i = 0; i < num_experts_per_rank; i++) {
                int curr                    = shared_expert_prefix_sum[i];
                shared_expert_prefix_sum[i] = prev;
                prev += curr;
            }
        }
        __syncthreads();

        auto start_time              = clock64();
        int  cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
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

            // Synchronize queue tail
            sync_barrier(responsible_rank, num_threads_per_rank);
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens;
                 chunk_idx += num_recv_warps_per_rank) {
                int  src_slot_idx = total_offset + chunk_idx;
                auto shifted_buffer_x_int4 =
                    dispatched_x_buffers.buffer() + src_slot_idx * hidden_int4;

#pragma unroll
                for (int i = 0; i < num_topk; i++) {
                    auto mapped_idx = static_cast<int64_t>(src_slot_idx) * num_topk + i;
                    auto recv_topk_idx_i64 =
                        ld_nc_global(dispatched_topk_idx_buffers.buffer() + mapped_idx);
                    int dst_slot_idx = -1;
                    if (recv_topk_idx_i64 >= 0) {

                        int slot = -1;
                        if (lane_id == 0) {
                            slot =
                                atomicAdd_system(expert_slot_idx.buffer() + recv_topk_idx_i64, 1);
                        }
                        slot = __shfl_sync(WARP_MASK, slot, 0);
                        EP_DEVICE_ASSERT(slot >= 0);
                        dst_slot_idx = shared_expert_prefix_sum[recv_topk_idx_i64] + slot;
                        EP_DEVICE_ASSERT(dst_slot_idx < num_max_tokens * num_topk);
                        auto shifted_recv_x_int4 =
                            recv_x + static_cast<int64_t>(dst_slot_idx) * hidden_int4;
                        UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4,
                                           shifted_buffer_x_int4, ld_nc_global, st_na_global);
                    }
                    dispatch_to_expert_map[mapped_idx] = dst_slot_idx;
                }
            }

// Copy `topk_idx` and `topk_weights`
#pragma unroll
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk;
                 idx += WARP_SIZE * num_recv_warps_per_rank) {
                int  chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
                int  src_slot_idx = total_offset + chunk_idx;
                auto mapped_idx   = static_cast<int64_t>(src_slot_idx) * num_topk + token_topk_idx;
                recv_topk_weights[mapped_idx] =
                    ld_nc_global(dispatched_topk_weights_buffers.buffer() + mapped_idx);
                recv_topk_idx[mapped_idx] =
                    ld_nc_global(dispatched_topk_idx_buffers.buffer() + mapped_idx);
            }

            num_tokens_to_recv -= num_recv_tokens;
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
        }
    }
}

void fused_dispatch_permute(void **buffer_ptrs, int64_t *recv_topk_idx, float *recv_topk_weights,
                            int *dispatch_to_expert_map, void const *x, float const *x_scales,
                            int64_t const *topk_idx, float const *topk_weights,
                            bool const *is_token_in_rank, int const *channel_prefix_matrix,
                            int const *num_recv_tokens_per_expert, void *recv_x, int num_tokens,
                            int hidden_int4, int num_topk, int num_experts, int num_scales,
                            int scale_token_stride, int scale_hidden_stride, int rank,
                            int num_ranks, cudaStream_t stream, int num_sms, int num_max_tokens,
                            int num_max_send_tokens) {
    constexpr int kNumThreads = 1024;

#define DISPATCH_LAUNCH_CASE(ranks)                                                                \
    {                                                                                              \
        auto kernel = fused_dispatch_permute<ranks, kNumThreads>;                                  \
        LAUNCH_KERNEL(&cfg, kernel, buffer_ptrs, recv_topk_idx, recv_topk_weights,                 \
                      dispatch_to_expert_map, reinterpret_cast<int4 const *>(x), x_scales,         \
                      topk_idx, topk_weights, is_token_in_rank, channel_prefix_matrix,             \
                      num_recv_tokens_per_expert, reinterpret_cast<int4 *>(recv_x), num_tokens,    \
                      hidden_int4, num_topk, num_experts, num_scales, scale_token_stride,          \
                      scale_hidden_stride, rank, num_max_tokens, num_max_send_tokens);             \
    }                                                                                              \
    break

    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

} // namespace intranode

} // namespace primus_turbo::cco::ep
