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
    dispatch(void **workspace_ptrs, int4 const *x, float const *x_scales, int64_t const *topk_idx,
             float const *topk_weights, bool const *is_token_in_rank,
             int const *channel_prefix_matrix, int num_tokens, int hidden_int4, int num_topk,
             int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
             int rank, int num_max_tokens) {
    auto const num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    auto const thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();

    // Several warps are response for a single rank
    auto const num_threads_per_rank = kNumThreads / kNumRanks;
    auto const num_channels         = num_sms;
    auto const responsible_rank     = (static_cast<int>(thread_id)) / num_threads_per_rank;
    auto const responsible_channel  = sm_id;

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= WARP_SIZE);

    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));

    auto       ptr                      = workspace_ptrs[responsible_rank];
    auto       rank_prefix_matrix       = Buffer<int>(ptr, kNumRanks * kNumRanks);
    auto const num_recv_x_total         = num_max_tokens * hidden_int4;
    auto       recv_x_buffer            = Buffer<int4>(ptr, num_recv_x_total);
    auto       recv_topk_idx_buffer     = Buffer<int64_t>(ptr, num_max_tokens * num_topk);
    auto       recv_topk_weights_buffer = Buffer<float>(ptr, num_max_tokens * num_topk);

    // Workers for sending
    constexpr int num_send_warps          = kNumThreads / WARP_SIZE;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    auto const    send_thread_id          = thread_id;
    auto const    send_warp_id_in_rank    = send_thread_id % num_threads_per_rank / WARP_SIZE;
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
    num_tokens_to_recv =
        channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
    num_tokens_to_recv -= total_offset;

    total_offset += rank_offset;

    if (num_tokens_to_recv <= 0)
        return;

    // Iterate over all tokens and send tokens to symmetric buffer
    int chunk_idx = -1;

    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
        // Skip if not selected
        // TODO(zhenhuang12): use __ldg
        if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank])
            continue;
        ++chunk_idx;

        // Distribute selected tokens by selected-token index for better balance.
        if (chunk_idx % num_send_warps_per_rank != send_warp_id_in_rank)
            continue;

        // Copy data
        auto dst_slot_idx   = total_offset + chunk_idx;
        auto shifted_recv_x = recv_x_buffer.buffer() + dst_slot_idx * hidden_int4;
        auto shifted_x      = x + token_idx * hidden_int4;
        UNROLLED_WARP_COPY(2, lane_id, hidden_int4, shifted_recv_x, shifted_x, __ldg,
                           st_na_global);

        // Copy `topk_idx` and `topk_weights`
        // if (lane_id < num_topk) {
        //     // Top-k index
        //     int recv_expert_begin = responsible_rank * num_experts_per_rank,
        //         recv_expert_end   = (responsible_rank + 1) * num_experts_per_rank;
        //     auto idx_value        = __ldg(topk_idx + token_idx * num_topk + lane_id);
        //     idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end)
        //                     ? idx_value - recv_expert_begin
        //                     : -1;
        //     recv_topk_idx_buffer[dst_slot_idx * num_topk + lane_id] = idx_value;

        //     // Top-k weights
        //     auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
        //     weight_value      = (idx_value >= 0) ? weight_value : 0.0f;
        //     recv_topk_weights_buffer[dst_slot_idx * num_topk + lane_id] = weight_value;
        // }
        // __syncwarp();

        // Copy `x_scales`
    }
    EP_DEVICE_ASSERT(chunk_idx + 1 == num_tokens_to_recv);
}

void dispatch(void **workspace_ptrs, void const *x, float const *x_scales, int64_t const *topk_idx,
              float const *topk_weights, bool const *is_token_in_rank,
              int const *channel_prefix_matrix, int num_tokens, int hidden_int4, int num_topk,
              int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
              int rank, int num_ranks, cudaStream_t stream, int num_sms, int num_max_tokens) {
    constexpr int kNumThreads = 1024;

#define DISPATCH_LAUNCH_CASE(ranks)                                                                \
    {                                                                                              \
        auto kernel = dispatch<ranks, kNumThreads>;                                                \
        LAUNCH_KERNEL(&cfg, kernel, workspace_ptrs, reinterpret_cast<int4 const *>(x), x_scales,   \
                      topk_idx, topk_weights, is_token_in_rank, channel_prefix_matrix, num_tokens, \
                      hidden_int4, num_topk, num_experts, num_scales, scale_token_stride,          \
                      scale_hidden_stride, rank, num_max_tokens);                                  \
    }                                                                                              \
    break

    // Sender-only kernel: one block per channel.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms / 2, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

} // namespace intranode

} // namespace primus_turbo::cco::ep
