#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#define PRINT_DEBUG(...)                                                                           \
    if (get_lane_id() == 0)                                                                        \
        printf(__VA_ARGS__);

namespace primus_turbo::cco::ep {
template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(int64_t const *topk_idx, int *num_tokens_per_rank,
                                    int *num_tokens_per_rdma_rank, int *num_tokens_per_expert,
                                    bool *is_token_in_rank, int num_tokens, int num_topk,
                                    int num_ranks, int num_experts) {
    auto sm_id     = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    int            expert_begin_idx = sm_id * kNumExpertsPerSM,
        expert_end_idx              = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    if (expert_begin_idx < expert_end_idx) {
// Per-thread count
#pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
#pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

    // Count rank statistics
    constexpr int  kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto           sm_begin       = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    int            rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM,
        rank_end_idx              = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx       = rank_begin_idx / NUM_MAX_NVL_PEERS,
        rdma_rank_end_idx         = rank_end_idx / NUM_MAX_NVL_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        auto const num_expert_per_rank = num_experts / num_ranks;
        auto       expert_begin        = rank_begin_idx * num_expert_per_rank;
        auto       expert_end          = rank_end_idx * num_expert_per_rank;

// Per-thread count
#pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx           = topk_idx + i * num_topk;
            int  is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

#pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and
            rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(int64_t const *topk_idx, int *num_tokens_per_rank,
                         int *num_tokens_per_rdma_rank, int *num_tokens_per_expert,
                         bool *is_token_in_rank, int num_tokens, int num_topk, int num_ranks,
                         int num_experts, cudaStream_t stream) {
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
    int           num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
                  (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
                  topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
                  is_token_in_rank, num_tokens, num_topk, num_ranks, num_experts);
}
namespace intranode {

template <int kNumRanks>
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
        // Barrier first
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {
            per_rank_buffer   = static_cast<int *>(buffer_ptrs[thread_id]);
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i
        //  to rank j
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i
        //  to local expert j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
#pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                per_rank_buffer[rank * kNumRanks + i] = num_tokens_per_rank[i];
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
                *moe_recv_counter_mapped =
                    local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
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

template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch(int4 **recv_x, float **recv_x_scales, int **recv_src_idx, int64_t **recv_topk_idx,
             float **recv_topk_weights, int **recv_channel_offset, int *send_head, int4 const *x,
             float const *x_scales, int64_t const *topk_idx, float const *topk_weights,
             bool const *is_token_in_rank, int const *channel_prefix_matrix,
             int const *rank_prefix_matrix, int num_tokens, int hidden_int4, int num_topk,
             int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
             int rank) {
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

    auto ptr = reinterpret_cast<void *>(recv_x[is_sender ? responsible_rank : rank]);
    auto recv_x_buffer = Buffer<int4>(ptr, 128);

    if (is_sender) {
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
        int rank_offset = responsible_rank > 0
                              ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank]
                              : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        // `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (lane_id == 0 and send_warp_id_in_rank == 0) {
            total_offset =
                -(responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels +
                                                                  responsible_channel - 1]
                                          : 0) -
                1;
            num_tokens_to_recv =
                -(channel_prefix_matrix[responsible_rank * num_channels + responsible_channel]) - 1;
        }
        __syncwarp();

        total_offset = __shfl_sync(WARP_MASK, total_offset, 0);
        total_offset += rank_offset;

        PRINT_DEBUG("total_offset: %d, rank_offset: %d\n", total_offset, rank_offset);
        return;

        // Iterate over all tokens and send tokens to symmetric buffer
        int chunk_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {

            // Skip if not selected
            if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                token_idx++;
                continue;
            }

            // Copy data
            auto dst_slot_idx           = total_offset + chunk_idx;
            auto shifted_recv_x_buffers = recv_x_buffer.buffer() + dst_slot_idx * hidden_int4;
            auto shifted_x              = x + token_idx * hidden_int4;
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_buffers, shifted_x, __ldg,
                               st_na_global);

            // Copy source index
            //             if (lane_id == 0)
            //                 channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

            //             // Copy `topk_idx` and `topk_weights` with transformed index
            //             if (lane_id < num_topk) {
            //                 // Top-k index
            //                 int recv_expert_begin = responsible_rank * num_experts_per_rank,
            //                     recv_expert_end   = (responsible_rank + 1) *
            //                     num_experts_per_rank;
            //                 auto idx_value        = __ldg(topk_idx + token_idx * num_topk +
            //                 lane_id); idx_value = (idx_value >= recv_expert_begin and idx_value <
            //                 recv_expert_end)
            //                                 ? idx_value - recv_expert_begin
            //                                 : -1;
            //                 channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] =
            //                 idx_value;

            //                 // Top-k weights
            //                 auto weight_value = __ldg(topk_weights + token_idx * num_topk +
            //                 lane_id); weight_value      = (idx_value >= 0) ? weight_value : 0.0f;
            //                 channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
            //                 weight_value;
            //             }

            // // Copy `x_scales`
            // #pragma unroll
            //             for (int i = lane_id; i < num_scales; i += WARP_SIZE) {
            //                 auto offset = token_idx * scale_token_stride + i *
            //                 scale_hidden_stride; channel_x_scales_buffers[dst_slot_idx *
            //                 num_scales + i] = __ldg(x_scales + offset);
            //             }

            chunk_idx++;
        }
    } else {
        return;
    }
}

void dispatch(void **recv_x, float **recv_x_scales, int **recv_src_idx, int64_t **recv_topk_idx,
              float **recv_topk_weights, int **recv_channel_offset, int *send_head, void const *x,
              float const *x_scales, int64_t const *topk_idx, float const *topk_weights,
              bool const *is_token_in_rank, int const *channel_prefix_matrix,
              int const *rank_prefix_matrix, int num_tokens, int hidden_int4, int num_topk,
              int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
              int rank, int num_ranks, cudaStream_t stream, int num_sms) {
    constexpr int kNumThreads = 1024;

    // Make sure never OOB
    // EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride <
    //                std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                                                \
    {                                                                                              \
        auto kernel = dispatch<ranks, kNumThreads>;                                                \
        LAUNCH_KERNEL(&cfg, kernel, reinterpret_cast<int4 **>(recv_x), recv_x_scales,              \
                      recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_offset,         \
                      send_head, reinterpret_cast<int4 const *>(x), x_scales, topk_idx,            \
                      topk_weights, is_token_in_rank, channel_prefix_matrix, rank_prefix_matrix,   \
                      num_tokens, hidden_int4, num_topk, num_experts, num_scales,                  \
                      scale_token_stride, scale_hidden_stride, rank);                              \
    }                                                                                              \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

} // namespace intranode

} // namespace primus_turbo::cco::ep