#include <primus_turbo/deep_ep/common/arch.cuh>

#include "api.cuh"

// ROCm: hoisted out of the guard -- Config::get_nvl_buffer_size_hint needs this on the
// pure-intranode path too, so it must not be a stub. Upstream returns `sizeof(SourceMeta)`;
// the EP_STATIC_ASSERT next to that struct below pins the two together.
namespace primus_turbo::deep_ep::legacy::internode {

int get_source_meta_bytes() {
    return 2 * sizeof(int);
}

}  // namespace primus_turbo::deep_ep::legacy::internode

#if PRIMUS_TURBO_DEEPEP_HAS_INTERNODE

#include <functional>
#include <optional>

#include <rocshmem/rocshmem.hpp>

#include "buffer.cuh"
#include "compiled.cuh"
#include "launch.cuh"
#include "utils.cuh"

// ROCm: rocSHMEM's headers trip these two on the device pass, and both are outside our code
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-volatile"

namespace primus_turbo::deep_ep {

namespace rocshmem {

extern ::rocshmem::rocshmem_team_t cpu_rdma_team;

}  // namespace rocshmem

namespace legacy {

namespace internode {

using ::rocshmem::rocshmem_ctx_t;
using ::rocshmem::rocshmem_team_t;

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const { return (is_token_in_nvl_rank_bits >> nvl_rank) & 1; }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

// ROCm: `get_source_meta_bytes` is defined above the guard, this keeps it honest
EP_STATIC_ASSERT(sizeof(SourceMeta) == 2 * sizeof(int), "Invalid size of `SourceMeta`");

__host__ __device__ __forceinline__ int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align_up(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) +
                                         num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float),
                                     sizeof(int4)));
}

// ROCm: `int` -> `int64_t`. The payload region already scales past 2 GB at 32 channels,
// and this offset points just past it, so a 32-bit pair silently wraps.
__host__ __device__ __forceinline__ std::pair<int64_t, int64_t> get_rdma_clean_meta(int hidden_int4,
                                                                            int num_scales,
                                                                            int num_topk_idx,
                                                                            int num_topk_weights,
                                                                            int num_rdma_ranks,
                                                                            int num_rdma_recv_buffer_tokens,
                                                                            int num_channels) {
    // Return `int32_t` offset and count to clean
    return {(static_cast<int64_t>(get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights)) *
             num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_channels) /
                sizeof(int),
            static_cast<int64_t>(LEGACY_NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels};
}

// ROCm: `int` -> `int64_t`, same overflow as `get_rdma_clean_meta`
__host__ __device__ __forceinline__ std::pair<int64_t, int64_t> get_nvl_clean_meta(int hidden_int4,
                                                                           int num_scales,
                                                                           int num_topk_idx,
                                                                           int num_topk_weights,
                                                                           int num_rdma_ranks,
                                                                           int num_nvl_ranks,
                                                                           int num_nvl_recv_buffer_tokens,
                                                                           int num_channels,
                                                                           bool is_dispatch) {
    // Return `int32_t` offset and to clean
    EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

    return {
        (static_cast<int64_t>(num_nvl_recv_buffer_tokens) *
         get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_nvl_ranks * num_channels) /
            sizeof(int),
        static_cast<int64_t>(num_nvl_ranks) * (2 * num_rdma_ranks + 2) * num_channels,
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

// ROCm: nvshmem_sync -> rocshmem barrier, rocSHMEM has no sync-without-quiet primitive
template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const rocshmem_team_t& rdma_team) {
    kLowLatencyMode ? void(::rocshmem::rocshmem_ctx_barrier(::rocshmem::ROCSHMEM_CTX_DEFAULT, rdma_team))
                    : ::rocshmem::rocshmem_barrier_all();
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
                                int* moe_recv_counter_mapped,
                                int num_ranks,
                                const int* num_tokens_per_rdma_rank,
                                int* moe_recv_rdma_counter_mapped,
                                const int* num_tokens_per_expert,
                                int* moe_recv_expert_counter_mapped,
                                int num_experts,
                                const bool* is_token_in_rank,
                                int num_tokens,
                                int num_worst_tokens,
                                int num_channels,
                                int expert_alignment,
                                const int64_t rdma_clean_offset,
                                const int64_t rdma_num_int_clean,
                                const int64_t nvl_clean_offset,
                                const int64_t nvl_num_int_clean,
                                int* rdma_channel_prefix_matrix,
                                int* recv_rdma_rank_prefix_sum,
                                int* gbl_channel_prefix_matrix,
                                int* recv_gbl_rank_prefix_sum,
                                void* rdma_buffer_ptr,
                                void** buffer_ptrs,
                                int** barrier_signal_ptrs,
                                int rank,
                                const rocshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    // ROCm: 32 -> WARP_SIZE, a wave is 64 lanes
    auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
    auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / WARP_SIZE;

    auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;
    auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / LEGACY_NUM_MAX_NVL_PEERS;

    if (sm_id == 0) {
        // Communication with others
        // Global barrier: the first warp does intra-node sync, the second warp does internode sync
        EP_DEVICE_ASSERT(num_warps > 1);
        EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);

        // ROCm: the per-QP `nvshmemi_ibgda_quiet` drain is dropped, rocSHMEM exposes no device-side
        // QP handle. The barrier below is a real rocSHMEM barrier, which already quiets the context.

        if (thread_id == WARP_SIZE)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Send numbers of tokens per rank/expert to RDMA ranks
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);

        // Clean up for later data dispatch
        // ROCm: plain stores -> st_coherent_sys_global, the readers of these slots use `sc0 sc1`
        EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
        #pragma unroll
        for (int64_t i = thread_id; i < rdma_num_int_clean; i += num_threads)
            st_coherent_sys_global(rdma_buffer_ptr_int + rdma_clean_offset + i, 0);

        // Copy to send buffer
        // ROCm: plain stores -> st_coherent_sys_global, the NIC reads these slots straight from memory
        #pragma unroll
        for (int i = thread_id; i < num_ranks; i += num_threads)
            st_coherent_sys_global(rdma_recv_num_tokens_mixed.send_buffer(i / LEGACY_NUM_MAX_NVL_PEERS) + i % LEGACY_NUM_MAX_NVL_PEERS,
                                   num_tokens_per_rank[i]);
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads)
            st_coherent_sys_global(rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts) + LEGACY_NUM_MAX_NVL_PEERS +
                                       i % num_rdma_experts,
                                   num_tokens_per_expert[i]);
        if (thread_id < kNumRDMARanks)
            st_coherent_sys_global(rdma_recv_num_tokens_mixed.send_buffer(thread_id) + LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts,
                                   num_tokens_per_rdma_rank[thread_id]);
        // ROCm: __syncthreads() orders only LDS on CDNA, so drain the global stores first
        s_waitcnt();
        __syncthreads();

        // Issue send
        // TODO: more light fence or barrier or signaling
        // TODO: overlap EP barrier and NVL cleaning
        // ROCm: per-warp IBGDA put -> one thread per peer, this is 40 ints and rocSHMEM
        // serves the self-PE case too, so the separate local-copy branch goes away
        if (thread_id < kNumRDMARanks) {
            ::rocshmem::rocshmem_int_put_nbi(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank),
                                             rdma_recv_num_tokens_mixed.send_buffer(thread_id),
                                             LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts + 1,
                                             translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank));
        }
        __syncthreads();

        // Barrier
        // ROCm: the explicit quiet is dropped, the barrier below quiets the context first
        if (thread_id == 0)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        __syncthreads();

        // NVL buffers
        auto nvl_send_buffer = thread_id < LEGACY_NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
        auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
        auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);
        auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS);
        auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, LEGACY_NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, LEGACY_NUM_MAX_NVL_PEERS);

        // Clean up for later data dispatch
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                             nvl_send_num_tokens_per_expert.total_bytes <=
                         nvl_clean_offset * sizeof(int));
        // ROCm: plain stores -> st_coherent_sys_global, NVL peers read these slots with `sc0 sc1`
        #pragma unroll
        for (int64_t i = thread_id; i < nvl_num_int_clean; i += num_threads)
            st_coherent_sys_global(nvl_buffer_ptr_int + nvl_clean_offset + i, 0);

        // Reduce number of tokens per expert into the NVL send buffer
        // TODO: may use NVSHMEM reduction
        EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
        // ROCm: plain loads -> ld_coherent_sys_global, the NIC wrote these slots past our L2
        if (thread_id < num_rdma_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                sum += ld_coherent_sys_global(rdma_recv_num_tokens_mixed.recv_buffer(i) + LEGACY_NUM_MAX_NVL_PEERS + thread_id);
            nvl_reduced_num_tokens_per_expert[thread_id] = sum;  // ours alone, a plain store is enough
        }
        // ROCm: __syncthreads() orders only LDS on CDNA, so drain the global stores first
        s_waitcnt();
        __syncthreads();

        // Reduce RDMA received tokens
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i) {
                // ROCm: plain load -> ld_coherent_sys_global, the NIC wrote this slot past our L2
                sum += ld_coherent_sys_global(rdma_recv_num_tokens_mixed.recv_buffer(i) + LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts);
                recv_rdma_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1)
                    ;
                *moe_recv_rdma_counter_mapped = sum;
            }
        }

        // Send numbers of tokens per rank/expert to NVL ranks
        EP_DEVICE_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= num_threads);
        // ROCm: plain accesses -> the coherent pair, the source is NIC-written and the
        // destination is an NVL peer's buffer; both need `sc0 sc1`
        if (thread_id < LEGACY_NUM_MAX_NVL_PEERS) {
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                st_coherent_sys_global(nvl_send_num_tokens_per_rank.buffer(nvl_rank) + i,
                                       ld_coherent_sys_global(rdma_recv_num_tokens_mixed.recv_buffer(i) + thread_id));
            #pragma unroll
            for (int i = 0; i < num_nvl_experts; ++i)
                st_coherent_sys_global(nvl_send_num_tokens_per_expert.buffer(nvl_rank) + i,
                                       nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i]);
        }
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++i) {
                int src_rdma_rank = i / LEGACY_NUM_MAX_NVL_PEERS, src_nvl_rank = i % LEGACY_NUM_MAX_NVL_PEERS;
                // ROCm: plain load -> ld_coherent_sys_global, an NVL peer wrote this slot
                sum += ld_coherent_sys_global(nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank) + src_rdma_rank);
                recv_gbl_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_counter_mapped) != -1)
                    ;
                *moe_recv_counter_mapped = sum;
            }
        }
        if (thread_id < num_nvl_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                // ROCm: plain load -> ld_coherent_sys_global, an NVL peer wrote this slot
                sum += ld_coherent_sys_global(nvl_recv_num_tokens_per_expert.buffer(i) + thread_id);
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1)
                    ;
                moe_recv_expert_counter_mapped[thread_id] = sum;
            }
        }

        // Finally barrier
        // ROCm: 32 -> WARP_SIZE, this picks the second wave
        if (thread_id == WARP_SIZE)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else {
        // Calculate meta data
        int dst_rdma_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            int total_count = 0, per_nvl_rank_count[LEGACY_NUM_MAX_NVL_PEERS] = {0};
            // ROCm: 32 -> WARP_SIZE, one wave walks the channel's tokens
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += WARP_SIZE) {
                EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
                auto is_token_in_rank_uint64 =
                    *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS);
                auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
                #pragma unroll
                for (int j = 0; j < LEGACY_NUM_MAX_NVL_PEERS; ++j)
                    per_nvl_rank_count[j] += is_token_in_rank_values[j];
                total_count += (is_token_in_rank_uint64 != 0);
            }

            // Warp reduce
            total_count = warp_reduce_sum(total_count);
            #pragma unroll
            for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);

            // Write into channel matrix
            if (elect_one_sync()) {
                #pragma unroll
                for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                    gbl_channel_prefix_matrix[(dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] = per_nvl_rank_count[i];
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;
            }
        }

        // Calculate prefix sum
        __syncthreads();
        if (thread_id == 0) {
            auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }

        EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        if (thread_id < LEGACY_NUM_MAX_NVL_PEERS) {
            auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + thread_id) * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }
    }
}

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                    \
    {                                                                                                                                  \
        auto notify_dispatch_func = low_latency_mode ? notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>; \
        LAUNCH_KERNEL(&cfg,                                                                                                            \
                      notify_dispatch_func,                                                                                            \
                      num_tokens_per_rank,                                                                                             \
                      moe_recv_counter_mapped,                                                                                         \
                      num_ranks,                                                                                                       \
                      num_tokens_per_rdma_rank,                                                                                        \
                      moe_recv_rdma_counter_mapped,                                                                                    \
                      num_tokens_per_expert,                                                                                           \
                      moe_recv_expert_counter_mapped,                                                                                  \
                      num_experts,                                                                                                     \
                      is_token_in_rank,                                                                                                \
                      num_tokens,                                                                                                      \
                      num_worst_tokens,                                                                                                \
                      num_channels,                                                                                                    \
                      expert_alignment,                                                                                                \
                      rdma_clean_meta.first,                                                                                           \
                      rdma_clean_meta.second,                                                                                          \
                      nvl_clean_meta.first,                                                                                            \
                      nvl_clean_meta.second,                                                                                           \
                      rdma_channel_prefix_matrix,                                                                                      \
                      recv_rdma_rank_prefix_sum,                                                                                       \
                      gbl_channel_prefix_matrix,                                                                                       \
                      recv_gbl_rank_prefix_sum,                                                                                        \
                      rdma_buffer_ptr,                                                                                                 \
                      buffer_ptrs,                                                                                                     \
                      barrier_signal_ptrs,                                                                                             \
                      rank,                                                                                                            \
                      rocshmem::cpu_rdma_team);                                                                                        \
    }                                                                                                                                  \
    break

    constexpr int kNumThreads = 512;
    const auto num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta =
        get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,
                                             num_scales,
                                             num_topk,
                                             num_topk,
                                             num_rdma_ranks,
                                             LEGACY_NUM_MAX_NVL_PEERS,
                                             num_max_nvl_chunked_recv_tokens,
                                             num_channels,
                                             true);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    // ROCm: the `< INT_MAX` pair is gone -- both buffers are laid out per channel, so at
    // `num_sms == 64` they are past 2 GB by construction. Every offset is int64_t now.

    // Launch kernel
    SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
    SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

// ROCm: `kNumTMABytesPerWarp` is gone with the TMA path, CDNA copies through the wave
template <bool kLowLatencyMode,
          int kNumRDMARanks,
          bool kCachedMode,
          int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS) * WARP_SIZE), 1)
    dispatch(int4* recv_x,
             float* recv_x_scales,
             topk_idx_t* recv_topk_idx,
             float* recv_topk_weights,
             SourceMeta* recv_src_meta,
             const int4* x,
             const float* x_scales,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             int* send_rdma_head,
             int* send_nvl_head,
             int* recv_rdma_channel_prefix_matrix,
             int* recv_gbl_channel_prefix_matrix,
             const int* rdma_channel_prefix_matrix,
             const int* recv_rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             const int* recv_gbl_rank_prefix_sum,
             const bool* is_token_in_rank,
             int num_tokens,
             int num_worst_tokens,
             int hidden_int4,
             int num_scales,
             int num_topk,
             int num_experts,
             int scale_token_stride,
             int scale_hidden_stride,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks) {
    enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };

    // ROCm: new, rocSHMEM device calls need a workgroup context, and creating one is collective
    // A non-zero return means the pool ran dry -- `ctx` is then garbage and every put on it
    // faults, so say so here instead of dying on a stray address. Raise ROCSHMEM_MAX_NUM_CONTEXTS.
    __shared__ rocshmem_ctx_t ctx;
    EP_DEVICE_ASSERT(::rocshmem::rocshmem_wg_ctx_create(0, &ctx) == 0);

    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    // ROCm: 32 -> WARP_SIZE, a wave is 64 lanes
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / WARP_SIZE;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
    const bool is_forwarder = sm_id % 2 == 0;
    const auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;

    // ROCm: the QP-count assert is dropped, rocSHMEM owns its own QP pool

    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (is_forwarder) {
            if (warp_id < LEGACY_NUM_MAX_NVL_PEERS) {
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % LEGACY_NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kForwarderCoordinator, warp_id - LEGACY_NUM_MAX_NVL_PEERS};
            }
        } else if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASenderCoordinator, -1};
        } else {
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % LEGACY_NUM_MAX_NVL_PEERS};
        }
    }();
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second;  // Not applicable for RDMA senders
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS);

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);

    // RDMA symmetric layout
    EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto scale_bytes = num_scales * sizeof(float);
    auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);
    auto rdma_channel_data = SymBuffer<uint8_t>(
        rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, LEGACY_NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    // NVL buffer layouts
    // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", `ws_rr_buffer_ptr` means "Write for Senders, Read for
    // Receivers"
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        rs_wr_buffer_ptr = buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = buffer_ptrs[target_rank], rs_wr_rank = nvl_rank,
        ws_rr_rank = target_rank;
    if (warp_role == WarpRole::kNVLReceivers)
        rs_wr_buffer_ptr = buffer_ptrs[target_rank], ws_rr_buffer_ptr = buffer_ptrs[nvl_rank], rs_wr_rank = target_rank,
        ws_rr_rank = nvl_rank;

    // Allocate buffers
    auto nvl_channel_x = AsymBuffer<uint8_t>(ws_rr_buffer_ptr,
                                             num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                             LEGACY_NUM_MAX_NVL_PEERS,
                                             channel_id,
                                             num_channels,
                                             rs_wr_rank)
                             .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_start =
        AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<int>(rs_wr_buffer_ptr, 1, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<int>(ws_rr_buffer_ptr, 1, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

    // RDMA sender warp synchronization
    // NOTES: `rdma_send_channel_tail` means the latest released tail
    // NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
    // ROCm: 32 -> 64 slots. The tuner always lands on the largest RDMA chunk, so a 32-slot window
    // makes a sender drain the whole window before the coordinator may issue; 64 lets them overlap.
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint64_t rdma_send_channel_window[kNumRDMARanks];
    // ROCm: `barrier.sync <id>` -> sync_barrier, an LDS arrival counter (arch.cuh)
    int bar_expected = sync_barrier_init();
    auto sync_rdma_sender_smem = [&]() { sync_barrier(bar_expected, 0, (kNumDispatchRDMASenderWarps + 1) * WARP_SIZE); };

    // ROCm: the TMA staging buffer is gone, gfx950 has no cp.async.bulk -- the copies below
    // go straight from global to global through the wave's registers

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[LEGACY_NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[LEGACY_NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = [&]() { sync_barrier(bar_expected, 1, (LEGACY_NUM_MAX_NVL_PEERS + 1) * WARP_SIZE); };

    if (warp_role == WarpRole::kRDMASender) {
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Send number of tokens in this channel by `-value - 1`
        EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] =
                    -(channel_id == 0
                          ? 0
                          : gbl_channel_prefix_matrix[(dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) -
                    1;
            } else if (lane_id < LEGACY_NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] =
                    -gbl_channel_prefix_matrix[(dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + lane_id - LEGACY_NUM_MAX_NVL_PEERS) * num_channels +
                                               channel_id] -
                    1;
            } else if (lane_id == LEGACY_NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
            } else if (lane_id == LEGACY_NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
            }
            __syncwarp();

            // Issue RDMA for non-local ranks
            // ROCm: IBGDA warp put -> rocSHMEM wave put, `dst_rdma_rank` is wave-uniform here
            if (dst_rdma_rank != rdma_rank) {
                ::rocshmem::rocshmem_ctx_int_put_nbi_wave(ctx,
                                                          rdma_channel_meta.recv_buffer(rdma_rank),
                                                          rdma_channel_meta.send_buffer(dst_rdma_rank),
                                                          LEGACY_NUM_MAX_NVL_PEERS * 2 + 2,
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank));
            }
        }
        // ROCm: new, the peer must see the meta before the barrier below lets it read
        ::rocshmem::rocshmem_ctx_quiet(ctx);
        sync_rdma_sender_smem();

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * LEGACY_NUM_MAX_NVL_PEERS));
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            // Skip the token which does not belong to this warp
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            // Wait the remote buffer to be released
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));

                // Timeout check
                if (clock64() - start_time >= LEGACY_NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id,
                           rdma_rank,
                           nvl_rank,
                           lane_id,
                           cached_rdma_channel_head,
                           rdma_tail_idx);
                    trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks and not kCachedMode)
                send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

            // Broadcast tails
            SourceMeta src_meta;
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
                // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                if ((slot_idx = __shfl(rdma_tail_idx, i)) >= 0) {
                    slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                    topk_ranks[num_topk_ranks] = i;
                    auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                    if (lane_id == num_topk_ranks)
                        src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
                    dst_send_buffers[num_topk_ranks++] =
                        reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;
                }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

            // Copy `x` into symmetric send buffer
            // ROCm: st_na_global -> a plain store. Unlike intranode this buffer is our own, not a
            // peer's, so `sc0 sc1` would only cost write-through bandwidth. The release store on
            // `rdma_send_channel_tail` below is what hands the slot to the coordinator warp.
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    reinterpret_cast<int4*>(dst_send_buffers[j])[key] = value;
            };
            // ROCm: ld_nc_global -> __ldg, `x` is our own tensor
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, __ldg, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

            // Copy `x_scales` into symmetric send buffer
            #pragma unroll
            // ROCm: 32 -> WARP_SIZE, one wave walks the scales
            for (int i = lane_id; i < num_scales; i += WARP_SIZE) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                // ROCm: ld_nc_global -> __ldg (our tensor), st_na_global -> a plain store
                auto value = __ldg(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    reinterpret_cast<float*>(dst_send_buffers[j])[i] = value;
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                // ROCm: st_na_global -> a plain store, same buffer as `x` above
                *reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]) = src_meta;
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy `topk_idx` and `topk_weights` into symmetric send buffer
            #pragma unroll
            // ROCm: 32 -> WARP_SIZE
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += WARP_SIZE) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                // ROCm: ld_nc_global -> __ldg (our tensors), st_na_global -> plain stores
                auto idx_value = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + copy_idx));
                auto weight_value = __ldg(topk_weights + token_idx * num_topk + copy_idx);
                reinterpret_cast<int*>(dst_send_buffers[rank_idx])[copy_idx] = idx_value;
                reinterpret_cast<float*>(dst_send_buffers[rank_idx])[num_topk + copy_idx] = weight_value;
            }
            __syncwarp();

            // Release the transaction in the window
            // ROCm: upstream lets every lane grab its own rank's lock at once, which needs CUDA's
            // independent thread scheduling -- on CDNA a lane holding a lock parks at reconvergence
            // until its siblings are done, so two waves holding each other's lock deadlock. Keep the
            // per-rank locks but walk them in ascending order with lane 0 as the only holder: one
            // lock at a time means no hold-and-wait, and the whole wave still moves together.
            bool has_token = (is_token_in_rank_uint64 != 0);
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i) {
                if (not __shfl(has_token, i))
                    continue;

                // Acquire lock first
                int latest_tail = 0, offset = 0;
                while (true) {
                    if (lane_id == 0)
                        acquire_lock(rdma_send_channel_lock + i);
                    __syncwarp();
                    if (lane_id == i) {
                        latest_tail = rdma_send_channel_tail[i];
                        offset = rdma_tail_idx - latest_tail;
                    }
                    // The tail we wait on only moves under this lock, so let go before retrying
                    if (__shfl(offset, i) < 64)
                        break;
                    if (lane_id == 0)
                        release_lock(rdma_send_channel_lock + i);
                    __syncwarp();
                }

                // Release the transaction slot
                // Add the bit and move the ones if possible
                if (lane_id == i) {
                    auto window = rdma_send_channel_window[i] | (1ull << offset);
                    if (offset == 0) {
                        auto num_empty_slots = (~window) == 0 ? 64 : __ffsll(~window) - 1;
                        st_release_cta(rdma_send_channel_tail + i, latest_tail + num_empty_slots);
                        window >>= num_empty_slots;
                    }
                    rdma_send_channel_window[i] = window;
                }

                // Release lock
                __syncwarp();
                if (lane_id == 0)
                    release_lock(rdma_send_channel_lock + i);
            }
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // NOTES: in case of splitting, the issued put at the end of the buffer
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

        // Clean shared memory
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;

        // Synchronize shared memory
        sync_rdma_sender_smem();

        // Get number of tokens to send for each RDMA rank
        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
        }

        // Iterate all RDMA ranks
        int last_issued_tail = 0;
        auto start_time = clock64();
        // ROCm: __any_sync(0xffffffff, ..) -> __any, the whole wave participates
        while (__any(num_tokens_to_send > 0)) {
            // Timeout check
            if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, nvl %d, dst IB: %d, tail: %d, remaining: %d\n",
                       channel_id,
                       rdma_rank,
                       nvl_rank,
                       lane_id,
                       last_issued_tail,
                       num_tokens_to_send);
                trap();
            }

            // TODO: try thread-level `put_nbi`?
            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                // To mitigate incast congestion, shuffle the starting index of target rank for different ranks and channels
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                synced_num_tokens_to_send = __shfl(num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                // Read the latest progress
                // NOTES: `rdma_send_channel_tail` does not need to be protected by lock
                auto processed_tail = __shfl(ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                auto synced_last_issued_tail = __shfl(last_issued_tail, dst_rdma_rank);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;

                // Issue RDMA send
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
                    // ROCm: IBGDA warp put -> rocSHMEM wave put
                    ::rocshmem::rocshmem_ctx_schar_put_nbi_wave(
                        ctx,
                        reinterpret_cast<signed char*>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token),
                        reinterpret_cast<const signed char*>(rdma_channel_data.send_buffer(dst_rdma_rank) +
                                                             dst_slot_idx * num_bytes_per_token),
                        num_bytes_per_msg,
                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank));
                    // ROCm: new, `nvshmemi_ibgda_rma` keeps the payload and the tail AMO on one QP,
                    // a rocSHMEM context may stripe them over several. Without this drain the
                    // receiver can see the tail move before the data lands.
                    ::rocshmem::rocshmem_ctx_quiet(ctx);
                } else {
                    // Lighter fence for local RDMA rank
                    memory_fence();
                }
                __syncwarp();

                // Update tails
                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    // ROCm: IBGDA AMO -> rocSHMEM AMO, which serves the local PE too
                    ::rocshmem::rocshmem_ctx_ulong_atomic_add(ctx,
                                                              rdma_channel_tail.buffer(rdma_rank),
                                                              num_tokens_to_issue,
                                                              translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank));
                }
                __syncwarp();
            }
        }
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // RDMA consumers and NVL producers
        const auto dst_nvl_rank = target_rank;

        // Wait counters to arrive
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + LEGACY_NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + LEGACY_NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + LEGACY_NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    // Notify NVL ranks
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    // Save RDMA channel received token count
                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                    printf(
                        "DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d, "
                        "meta: %d, %d, %d, %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        lane_id,
                        dst_nvl_rank,
                        meta_0,
                        meta_1,
                        meta_2,
                        meta_3);
                    trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head
        send_nvl_head += src_rdma_channel_prefix * LEGACY_NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        // NOTES: always start from the local rank
        int src_rdma_rank = sm_id % kNumRDMARanks;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
        // ROCm: __any_sync(0xffffffff, ..) -> __any, the whole wave participates
        while (__any(num_tokens_to_recv_from_rdma > 0)) {
            // Check destination queue emptiness, or wait a buffer to be released
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;
                // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                cached_nvl_channel_head = __shfl(ld_volatile_global(nvl_channel_head.buffer()), 0);

                // Timeout check
                if (elect_one_sync() and clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                    printf(
                        "DeepEP dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        dst_nvl_rank,
                        ld_volatile_global(nvl_channel_head.buffer()),
                        cached_nvl_channel_tail);
                    trap();
                }
            }

            // Find next source RDMA rank (round-robin)
            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                if (__shfl(num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    if (__shfl(cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf(
                        "DeepEP dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, "
                        "head: %d, tail: %d, expected: %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        dst_nvl_rank,
                        lane_id,
                        cached_rdma_channel_head,
                        cached_rdma_channel_tail,
                        num_tokens_to_recv_from_rdma);
                    trap();
                }
            }
            auto src_rdma_head = __shfl(cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl(cached_rdma_channel_tail, src_rdma_rank);

            // Iterate over every token from the RDMA buffer
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                // ROCm: ld_nc_global -> ld_coherent_sys_global, the NIC wrote this buffer
                auto src_meta = ld_coherent_sys_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    if (not kCachedMode)
                        send_nvl_head[i * LEGACY_NUM_MAX_NVL_PEERS] = cached_head;
                }
                if (not is_in_dst_nvl_rank)
                    continue;

                // Get an empty slot
                int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                // Copy data
                // ROCm: the TMA round trip through shared memory -> one wave copy. The token blob
                // is `int4`-aligned by construction, so both ends stay on dwordx4.
                UNROLLED_WARP_COPY(5,
                                   lane_id,
                                   num_bytes_per_token / static_cast<int>(sizeof(int4)),
                                   reinterpret_cast<int4*>(dst_shifted),
                                   reinterpret_cast<const int4*>(shifted),
                                   ld_coherent_sys_x4,
                                   st_coherent_sys_x4);
                __syncwarp();

                // In case of insufficient NVL buffers, early stopping
                if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            // Move tail index
            __syncwarp();
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        // Retired
        __syncwarp();
        if (elect_one_sync())
            forward_channel_retired[dst_nvl_rank] = true;
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        // Extra warps for forwarder coordinator should exit directly
        // ROCm: `return` -> a guarded block, `rocshmem_wg_ctx_destroy` below is block-collective
        if (target_rank == 0) {
            // Forward warp coordinator
            EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

            // Clean shared memory
            EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            // ROCm: 32 -> WARP_SIZE
            #pragma unroll
            for (int i = lane_id; i < kNumRDMARanks * LEGACY_NUM_MAX_NVL_PEERS; i += WARP_SIZE)
                forward_channel_head[i % LEGACY_NUM_MAX_NVL_PEERS][i / LEGACY_NUM_MAX_NVL_PEERS] = 0;
            if (lane_id < LEGACY_NUM_MAX_NVL_PEERS)
                forward_channel_retired[lane_id] = false;
            sync_forwarder_smem();

            int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
            while (true) {
                // Find minimum head
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                    if (not forward_channel_retired[i])
                        min_head = min(min_head, forward_channel_head[i][target_rdma]);
                // ROCm: __all_sync(0xffffffff, ..) -> __all, the whole wave participates
                if (__all(min_head == std::numeric_limits<int>::max()))
                    break;

                // Update remote head
                if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and
                    lane_id < kNumRDMARanks) {
                    // ROCm: IBGDA AMO -> rocSHMEM AMO, which serves the local PE too
                    ::rocshmem::rocshmem_ctx_ulong_atomic_add(ctx,
                                                              rdma_channel_head.buffer(rdma_rank),
                                                              min_head - last_head,
                                                              translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank));
                    last_head = min_head;
                }

                // Nanosleep and let other warps work
                // ROCm: __nanosleep -> s_sleep, ~64 clocks a tick and the count must be immediate
                s_sleep<LEGACY_NUM_WAIT_TICKS>();
            }
        }
    } else {
        // NVL consumers
        // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
        int src_nvl_rank = target_rank, total_offset = 0;
        const int local_expert_begin = rank * (num_experts / num_ranks);
        const int local_expert_end = local_expert_begin + (num_experts / num_ranks);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * LEGACY_NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * LEGACY_NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                total_offset += start_offset;
                break;
            }

            // Timeout check
            if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                printf(
                    "DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d, start: %d, end: %d\n",
                    channel_id,
                    rdma_rank,
                    nvl_rank,
                    lane_id,
                    src_nvl_rank,
                    start_offset,
                    end_offset);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage
        if (lane_id < kNumRDMARanks and not kCachedMode)
            recv_gbl_channel_prefix_matrix[(lane_id * LEGACY_NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // Check channel status by lane 0
            start_time = clock64();
            while (true) {
                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                cached_channel_tail_idx = __shfl(ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);

                // Timeout check
                if (elect_one_sync() and clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id,
                           rdma_rank,
                           nvl_rank,
                           src_nvl_rank,
                           cached_channel_head_idx,
                           cached_channel_tail_idx);
                    trap();
                }
            }

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                // ROCm: ld_nc_global -> ld_coherent_sys_global, a peer NVL rank wrote this
                auto meta = ld_coherent_sys_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                int64_t recv_token_idx = __shfl(total_offset, meta.src_rdma_rank);
                (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;

                // Copy data
                // ROCm: the TMA round trip is gone; `hidden` and the scales are separate copies
                // again, so the `scale_bytes % 16` special case it existed for goes with it
                auto st_own = [](auto* dst, const auto& value) { *dst = value; };
                // ROCm: unroll 5 -> 2 on wave64, five int4 per lane overruns the load stride
#if defined(PRIMUS_TURBO_GFX942) || defined(PRIMUS_TURBO_GFX950)
                UNROLLED_WARP_COPY(2, lane_id, hidden_int4, recv_x + recv_token_idx * hidden_int4,
                                   reinterpret_cast<const int4*>(shifted), ld_coherent_sys_x4, st_own);
#else
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, recv_x + recv_token_idx * hidden_int4,
                                   reinterpret_cast<const int4*>(shifted), ld_coherent_sys_x4, st_own);
#endif
                shifted += hidden_bytes;

                // Copy scales
                // ROCm: ld_nc_global -> ld_coherent_sys_global (peer), st_na_global -> our tensor
                UNROLLED_WARP_COPY(1, lane_id, num_scales, recv_x_scales + recv_token_idx * num_scales,
                                   reinterpret_cast<float*>(shifted), ld_coherent_sys_global, st_own);
                shifted += scale_bytes;

                // Copy source meta
                // ROCm: st_na_global -> a plain store, `recv_src_meta` is our own tensor
                if (not kCachedMode and elect_one_sync())
                    recv_src_meta[recv_token_idx] = meta;
                shifted += sizeof(SourceMeta);

                // Copy `topk_idx` and `topk_weights`
                if (lane_id < num_topk) {
                    // Read
                    // ROCm: ld_nc_global -> ld_coherent_sys_global, a peer NVL rank wrote this
                    auto idx_value = static_cast<topk_idx_t>(ld_coherent_sys_global(reinterpret_cast<int*>(shifted) + lane_id));
                    auto weight_value = ld_coherent_sys_global(reinterpret_cast<float*>(shifted + sizeof(int) * num_topk) + lane_id);
                    auto recv_idx = recv_token_idx * num_topk + lane_id;

                    // Transform and write
                    // ROCm: st_na_global -> plain stores, both are our own tensors
                    idx_value = (idx_value >= local_expert_begin and idx_value < local_expert_end) ? idx_value - local_expert_begin : -1;
                    weight_value = idx_value >= 0 ? weight_value : 0.0f;
                    recv_topk_idx[recv_idx] = idx_value;
                    recv_topk_weights[recv_idx] = weight_value;
                }
                __syncwarp();
            }

            // Move queue
            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }
    }

    // ROCm: new, block-collective teardown, so it sits after every role branch has rejoined
    ::rocshmem::rocshmem_wg_ctx_destroy(&ctx);

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        if (is_forwarder)
            return;
        // get the actual number of num_recv_tokens on the current rank
        int num_recv_tokens = recv_gbl_rank_prefix_sum[num_ranks - 1];
        // some ForwarderCoordinator threads exit early, so we only use non-forwarder in clean-up
        // channel_id * num_threads is the offset of the current non-forwarder sms
        const auto clean_start = num_recv_tokens * num_topk + channel_id * num_threads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_channels * num_threads;
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;
    }
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              int scale_token_stride,
              int scale_hidden_stride,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode) {
    // ROCm: the TMA scratch is gone, so is the dynamic shared memory it needed
    constexpr int kNumDispatchRDMASenderWarps = 7;

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                   \
    {                                                                                                                          \
        auto dispatch_func = low_latency_mode                                                                                  \
            ? (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, kNumDispatchRDMASenderWarps>                          \
                                  : dispatch<true, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>)                        \
            : (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, kNumDispatchRDMASenderWarps>                         \
                                  : dispatch<false, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>);                      \
        LAUNCH_KERNEL(&cfg,                                                                                                    \
                      dispatch_func,                                                                                           \
                      reinterpret_cast<int4*>(recv_x),                                                                         \
                      recv_x_scales,                                                                                           \
                      recv_topk_idx,                                                                                           \
                      recv_topk_weights,                                                                                       \
                      reinterpret_cast<SourceMeta*>(recv_src_meta),                                                            \
                      reinterpret_cast<const int4*>(x),                                                                        \
                      x_scales,                                                                                                \
                      topk_idx,                                                                                                \
                      topk_weights,                                                                                            \
                      send_rdma_head,                                                                                          \
                      send_nvl_head,                                                                                           \
                      recv_rdma_channel_prefix_matrix,                                                                         \
                      recv_gbl_channel_prefix_matrix,                                                                          \
                      rdma_channel_prefix_matrix,                                                                              \
                      recv_rdma_rank_prefix_sum,                                                                               \
                      gbl_channel_prefix_matrix,                                                                               \
                      recv_gbl_rank_prefix_sum,                                                                                \
                      is_token_in_rank,                                                                                        \
                      num_tokens,                                                                                              \
                      num_worst_tokens,                                                                                        \
                      hidden_int4,                                                                                             \
                      num_scales,                                                                                              \
                      num_topk,                                                                                                \
                      num_experts,                                                                                             \
                      scale_token_stride,                                                                                      \
                      scale_hidden_stride,                                                                                     \
                      rdma_buffer_ptr,                                                                                         \
                      num_max_rdma_chunked_send_tokens,                                                                        \
                      num_max_rdma_chunked_recv_tokens,                                                                        \
                      buffer_ptrs,                                                                                             \
                      num_max_nvl_chunked_send_tokens,                                                                         \
                      num_max_nvl_chunked_recv_tokens,                                                                         \
                      rank,                                                                                                    \
                      num_ranks);                                                                                              \
    }                                                                                                                          \
    break

    EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    // ROCm: 32 -> WARP_SIZE, one wave per role; (7 + 1 + 8) * 64 == 1024, the block-size ceiling
    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS) * WARP_SIZE, stream);
    SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

// ROCm: `kNumTMABytesPerWarp` is gone with the TMA path, the scan runs straight on global
template <bool kLowLatencyMode>
__global__ void cached_notify(const int64_t rdma_clean_offset,
                              const int64_t rdma_num_int_clean,
                              const int64_t nvl_clean_offset,
                              const int64_t nvl_num_int_clean,
                              int* combined_rdma_head,
                              int num_combined_tokens,
                              int num_channels,
                              const int* rdma_channel_prefix_matrix,
                              const int* rdma_rank_prefix_sum,
                              int* combined_nvl_head,
                              void* rdma_buffer_ptr,
                              void** buffer_ptrs,
                              int** barrier_signal_ptrs,
                              int rank,
                              int num_ranks,
                              bool is_cached_dispatch,
                              const rocshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    // ROCm: 32 -> WARP_SIZE, a warp here is a wave
    auto num_warps = num_threads / WARP_SIZE;
    auto warp_id = thread_id / WARP_SIZE;
    auto lane_id = get_lane_id();

    auto nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;
    auto num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;
    auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS;

    // Using two SMs, which clean the RDMA/NVL buffer respectively
    if (sm_id == 0) {
        // ROCm: the per-QP `nvshmemi_ibgda_quiet` drain is dropped, rocSHMEM exposes no
        // device-side QP handle. The barrier below is a real rocSHMEM barrier, which quiets.
        __syncthreads();

        // Barrier for RDMA
        // ROCm: 32 -> WARP_SIZE, upstream means "the second wave's first lane"
        if (thread_id == WARP_SIZE)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);

        // Barrier for NVL
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Clean RDMA buffer
        // ROCm: plain stores -> st_coherent_sys_global, the readers of these slots use `sc0 sc1`
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        #pragma unroll
        for (int64_t i = thread_id; i < rdma_num_int_clean; i += num_threads)
            st_coherent_sys_global(rdma_buffer_ptr_int + rdma_clean_offset + i, 0);

        // Clean NVL buffer
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        // ROCm: plain stores -> st_coherent_sys_global, NVL peers read these slots with `sc0 sc1`
        #pragma unroll
        for (int64_t i = thread_id; i < nvl_num_int_clean; i += num_threads)
            st_coherent_sys_global(nvl_buffer_ptr_int + nvl_clean_offset + i, 0);
        // ROCm: __syncthreads() orders only LDS on CDNA, so drain the cleans first
        s_waitcnt();
        __syncthreads();

        // Barrier again
        // ROCm: 32 -> WARP_SIZE, same as above
        if (thread_id == WARP_SIZE)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else if (sm_id == 1) {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

        // Iterate in reverse order
        // ROCm: upstream gives every channel its own warp, which needs `32 * num_channels`
        // threads. A wave is twice as wide here, so that formula blows past the 1024-thread
        // block limit at `num_sms == 64`. Stride over the channels instead: the block stays
        // legal for any channel count and each wave still owns a whole channel at a time.
        if (lane_id < num_rdma_ranks) {
            for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
                int token_start_idx, token_end_idx;
                get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

                // NOTES: `1 << 25` is a heuristic large number
                int last_head = 1 << 25;
                for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                    auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
                    if (current_head < 0) {
                        combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
                    } else {
                        last_head = current_head;
                    }
                }
            }
        }
    } else {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
        EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

        // ROCm: the TMA staging of `combined_nvl_head` is dropped -- the scan is one int per
        // lane per token either way, so batching it through shared memory buys nothing here
        // ROCm: channel-strided for the same 1024-thread reason as the branch above
        if (lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
            for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
                for (int dst_rdma_rank = sm_id - 2; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 2) {
                    // Iterate in reverse order
                    int token_start_idx =
                        channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
                    int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
                    int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                    token_start_idx += shift, token_end_idx += shift;

                    // NOTES: `1 << 25` is a heuristic large number
                    int last_head = 1 << 25;
                    for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                        auto current_head = __ldg(combined_nvl_head + token_idx * LEGACY_NUM_MAX_NVL_PEERS + lane_id);
                        if (current_head < 0) {
                            combined_nvl_head[token_idx * LEGACY_NUM_MAX_NVL_PEERS + lane_id] = -last_head - 1;
                        } else {
                            last_head = current_head;
                        }
                    }
                }
            }
        }
    }
}

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode) {
    // ROCm: 32 -> WARP_SIZE, but clamped -- a wave is twice as wide, so the upstream formula
    // asks for 2048 threads at `num_sms == 64`. The kernel strides over channels instead.
    const int num_threads = std::min(1024, std::max(128, WARP_SIZE * num_channels));
    const auto num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(
        hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,
                                             num_scales,
                                             num_topk_idx,
                                             num_topk_weights,
                                             num_rdma_ranks,
                                             LEGACY_NUM_MAX_NVL_PEERS,
                                             num_max_nvl_chunked_recv_tokens,
                                             num_channels,
                                             is_cached_dispatch);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    // ROCm: the `< INT_MAX` pair is gone -- both buffers are laid out per channel, so at
    // `num_sms == 64` they are past 2 GB by construction. Every offset is int64_t now.
    EP_HOST_ASSERT(num_channels * 2 > 3);

    // Launch kernel
    auto cached_notify_func = low_latency_mode ? cached_notify<true> : cached_notify<false>;
    SETUP_LAUNCH_CONFIG(num_channels * 2, num_threads, stream);
    LAUNCH_KERNEL(&cfg,
                  cached_notify_func,
                  rdma_clean_meta.first,
                  rdma_clean_meta.second,
                  nvl_clean_meta.first,
                  nvl_clean_meta.second,
                  combined_rdma_head,
                  num_combined_tokens,
                  num_channels,
                  rdma_channel_prefix_matrix,
                  rdma_rank_prefix_sum,
                  combined_nvl_head,
                  rdma_buffer_ptr,
                  buffer_ptrs,
                  barrier_signal_ptrs,
                  rank,
                  num_ranks,
                  is_cached_dispatch,
                  rocshmem::cpu_rdma_team);
}

// ROCm: `kUseTMA`/`kNumStages`/`kNumTMALoadBytes` and the shared-memory staging they drove are
// gone -- CDNA has no `cp.async.bulk`, the wave loads straight from the peer's slot.
template <int kNumRanks,
          bool kMaybeWithBias,
          typename dtype_t,
          int kMaxNumRanks,
          typename GetAddrFn,
          typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank,
                             int head_idx,
                             int lane_id,
                             int hidden_int4,
                             int num_topk,
                             int4* combined_row,
                             float* combined_topk_weights,
                             const int4* bias_0_int4,
                             const int4* bias_1_int4,
                             int num_max_recv_tokens,
                             const GetAddrFn& get_addr_fn,
                             const ReceiveTWFn& recv_tw_fn) {
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // Broadcast current heads
    // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++i)
        // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
        if (__shfl(is_token_in_rank, i)) {
            slot_indices[num_topk_ranks] = __shfl(head_idx, i) % num_max_recv_tokens;
            topk_ranks[num_topk_ranks++] = i;
        }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);

    // Reduce data
    {
        // ROCm: stride 32 -> WARP_SIZE, one full wave per role
        #pragma unroll
        for (int i = lane_id; i < hidden_int4; i += WARP_SIZE) {
            // Read bias
            // TODO: make it as a finer-grained template
            int4 bias_0_value_int4, bias_1_value_int4;
            if constexpr (kMaybeWithBias) {
                // ROCm: ld_nc_global -> __ldg, the bias tensors are our own
                bias_0_value_int4 = bias_0_int4 != nullptr ? __ldg(bias_0_int4 + i) : make_int4(0, 0, 0, 0);
                bias_1_value_int4 = bias_1_int4 != nullptr ? __ldg(bias_1_int4 + i) : make_int4(0, 0, 0, 0);
            }

            // Read buffers
            // TODO: maybe too many registers here
            int4 recv_value_int4[kMaxNumRanks];
            // ROCm: ld_nc_global -> ld_coherent_sys_x4, a peer wrote these slots
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j)
                recv_value_int4[j] = ld_coherent_sys_x4(get_addr_fn(topk_ranks[j], slot_indices[j], i));

            // Clean
            // Reduce bias
            float values[kDtypePerInt4] = {0};
            if constexpr (kMaybeWithBias) {
                auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                #pragma unroll
                for (int j = 0; j < kDtypePerInt4; ++j)
                    values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
            }

            // Reduce all-to-all results
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            // Cast back to `dtype_t` and write
            int4 out_int4;
            auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            // ROCm: st_na_global -> st_coherent_sys_x4, `combined_row` may be a peer's slot
            st_coherent_sys_x4(combined_row + i, out_int4);
        }
    }

    // Reduce `topk_weights`
    if (lane_id < num_topk) {
        float value = 0;
        #pragma unroll
        for (int i = 0; i < num_topk_ranks; ++i)
            value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
        // ROCm: st_na_global -> st_coherent_sys_global, same reason as `combined_row`
        st_coherent_sys_global(combined_topk_weights + lane_id, value);
    }

    // Return the minimum top-k rank
    return topk_ranks[0];
}

// ROCm: new, a wave64 block holds at most 1024 / 64 == 16 waves. Upstream's block is
// `kNumForwarders + 1` warps wide; here the NVL-sender SM may need more, and 16+ RDMA peers
// need more than fits. Clamp so every specialization compiles -- the host rejects the rest.
constexpr int get_num_combine_block_warps(int num_forwarders, int num_rdma_receivers) {
    const int num_recv_sm_warps = LEGACY_NUM_MAX_NVL_PEERS + num_rdma_receivers;
    const int num_warps = (num_recv_sm_warps > num_forwarders ? num_recv_sm_warps : num_forwarders) + 1;
    return num_warps < 1024 / WARP_SIZE ? num_warps : 1024 / WARP_SIZE;
}

// ROCm: the TMA scratch template arguments are gone. `kNumRDMAReceivers` no longer follows
// `kNumForwarders - 8`: on wave64 a block holds half as many waves, so `kNumForwarders` can be
// as low as 8 (at `kNumRDMARanks == 8`) and the receiver count needs a floor of 1. That also
// unties the block width from `kNumForwarders + 1` -- see `get_num_combine_block_warps`.
template <bool kLowLatencyMode,
          int kNumRDMARanks,
          typename dtype_t,
          int kNumCombineForwarderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
          int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
          int kNumForwarders = kNumRDMARanks* kNumWarpsPerForwarder,
          int kNumRDMAReceivers = (kNumForwarders > LEGACY_NUM_MAX_NVL_PEERS) ? kNumForwarders - LEGACY_NUM_MAX_NVL_PEERS : 1,
          int kNumCombineBlockWarps = get_num_combine_block_warps(kNumForwarders, kNumRDMAReceivers)>
__global__ void __launch_bounds__(kNumCombineBlockWarps * WARP_SIZE, 1) combine(int4* combined_x,
                                                                        float* combined_topk_weights,
                                                                        const bool* is_combined_token_in_rank,
                                                                        const int4* x,
                                                                        const float* topk_weights,
                                                                        const int4* bias_0,
                                                                        const int4* bias_1,
                                                                        const int* combined_rdma_head,
                                                                        const int* combined_nvl_head,
                                                                        const SourceMeta* src_meta,
                                                                        const int* rdma_channel_prefix_matrix,
                                                                        const int* rdma_rank_prefix_sum,
                                                                        const int* gbl_channel_prefix_matrix,
                                                                        int num_tokens,
                                                                        int num_combined_tokens,
                                                                        int hidden,
                                                                        int num_topk,
                                                                        void* rdma_buffer_ptr,
                                                                        int num_max_rdma_chunked_send_tokens,
                                                                        int num_max_rdma_chunked_recv_tokens,
                                                                        void** buffer_ptrs,
                                                                        int num_max_nvl_chunked_send_tokens,
                                                                        int num_max_nvl_chunked_recv_tokens,
                                                                        int rank,
                                                                        int num_ranks) {
    // ROCm: new, `kIdle` -- the block is as wide as the busier of the two SMs, so on the other
    // one the tail waves have no work. Upstream had none because both SMs were exactly as wide.
    enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator, kIdle };

    const auto sm_id = static_cast<int>(blockIdx.x);
    // ROCm: 32 -> WARP_SIZE, one wave per role
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / WARP_SIZE;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    // ROCm: new, rocSHMEM device calls need a workgroup context, and creating one is collective
    // A non-zero return means the pool ran dry -- `ctx` is then garbage and every put on it
    // faults, so say so here instead of dying on a stray address. Raise ROCSHMEM_MAX_NUM_CONTEXTS.
    __shared__ rocshmem_ctx_t ctx;
    EP_DEVICE_ASSERT(::rocshmem::rocshmem_wg_ctx_create(0, &ctx) == 0);

    // ROCm: `barrier.sync <id>` -> sync_barrier, an LDS arrival counter (arch.cuh). Must run
    // before the role split, `sync_barrier_init` is a whole-block sync.
    int bar_expected = sync_barrier_init(), bar_expected_large = 0;

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const auto hidden_bytes = hidden_int4 * sizeof(int4);
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

    // NOTES: we decouple a channel into 2 SMs
    const auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        // ROCm: 32 -> WARP_SIZE; the `kNumForwarders` bounds become explicit role counts, they
        // no longer coincide now that the two SMs may differ in width
        auto warp_id = thread_id / WARP_SIZE;
        if (not is_forwarder_sm) {
            if (warp_id < LEGACY_NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = warp_id;
                shuffled_warp_id = (shuffled_warp_id + channel_id) % LEGACY_NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id};
            } else if (warp_id < LEGACY_NUM_MAX_NVL_PEERS + kNumRDMAReceivers) {
                return {WarpRole::kRDMAReceiver, warp_id - LEGACY_NUM_MAX_NVL_PEERS};
            } else if (warp_id == LEGACY_NUM_MAX_NVL_PEERS + kNumRDMAReceivers) {
                return {WarpRole::kCoordinator, 0};
            } else {
                return {WarpRole::kIdle, 0};
            }
        } else {
            if (warp_id < kNumForwarders) {
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else if (warp_id == kNumForwarders) {
                return {WarpRole::kCoordinator, 0};
            } else {
                return {WarpRole::kIdle, 0};
            }
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    EP_DEVICE_ASSERT(num_warps == kNumCombineBlockWarps);
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

    if (warp_role == WarpRole::kNVLSender) {
        // NVL producers
        const auto dst_nvl_rank = warp_id;

        // NVL layouts
        // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr,
                                                 num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                                 LEGACY_NUM_MAX_NVL_PEERS,
                                                 channel_id,
                                                 num_channels,
                                                 nvl_rank)
                                 .advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank)
                                    .advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                                    .advance_also(local_buffer_ptr);

        // ROCm: the TMA staging buffer is dropped, the wave copies straight into the peer

        // Get tasks for each RDMA lane
        int token_start_idx = 0, token_end_idx = 0;
        if (lane_id < kNumRDMARanks) {
            int prefix_idx = (lane_id * LEGACY_NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
        }
        __syncwarp();

        // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Iterate over all tokens and send by chunks
        int current_rdma_idx = channel_id % kNumRDMARanks;
        while (true) {
            // Exit if possible
            // ROCm: __all_sync(0xffffffff, ..) -> __all, the whole wave participates
            if (__all(token_start_idx >= token_end_idx))
                break;

            // Decide the next RDMA buffer to send
            bool is_lane_ready = false;
            auto start_time = clock64();
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and
                    num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
                // ROCm: __any_sync(0xffffffff, ..) -> __any, the whole wave participates
                if (__any(is_lane_ready))
                    break;

                // Retry
                if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
                    cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf(
                        "DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, RDMA lane: %d, head: %d, tail: "
                        "%d, start: %d, end: %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        dst_nvl_rank,
                        lane_id,
                        ld_volatile_global(nvl_channel_head.buffer() + lane_id),
                        cached_channel_tail_idx,
                        token_start_idx,
                        token_end_idx);
                    trap();
                }
            }

            // Sync token start index and count
            for (int i = 0; i < kNumRDMARanks; ++i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;
                // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                if (__shfl((token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                // Sync token start index
                auto token_idx = static_cast<int64_t>(__shfl(token_start_idx, current_rdma_idx));
                int num_tokens_in_chunk =
                    __shfl(min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);

                // Send by chunk
                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    // Get an empty slot
                    int dst_slot_idx = 0;
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    // ROCm: __shfl_sync(0xffffffff, ..) -> __shfl, the whole wave participates
                    dst_slot_idx = __shfl(dst_slot_idx, current_rdma_idx);

                    // Copy data
                    // ROCm: the TMA round trip through shared memory -> one wave copy straight
                    // into the peer's slot. `x` is our own tensor, the slot is the peer's.
                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;
                    auto shifted_x = x + token_idx * hidden_int4;
                    // ROCm: unroll 5 -> 2 on wave64, five int4 per lane overruns the load stride
#if defined(PRIMUS_TURBO_GFX942) || defined(PRIMUS_TURBO_GFX950)
                    UNROLLED_WARP_COPY(2, lane_id, hidden_int4, reinterpret_cast<int4*>(shifted_x_buffers),
                                       shifted_x, __ldg, st_coherent_sys_x4);
#else
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, reinterpret_cast<int4*>(shifted_x_buffers),
                                       shifted_x, __ldg, st_coherent_sys_x4);
#endif

                    // Copy source meta
                    // ROCm: ld_nc_global -> a plain load (our own; `SourceMeta` has no `__ldg`),
                    // st_na_global -> st_coherent_sys_global, the slot belongs to a peer
                    if (lane_id == num_topk)
                        st_coherent_sys_global(reinterpret_cast<SourceMeta*>(shifted_x_buffers + hidden_bytes),
                                               src_meta[token_idx]);

                    // Copy `topk_weights`
                    if (lane_id < num_topk)
                        st_coherent_sys_global(
                            reinterpret_cast<float*>(shifted_x_buffers + hidden_bytes + sizeof(SourceMeta)) + lane_id,
                            __ldg(topk_weights + token_idx * num_topk + lane_id));
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
            }

            // Move queue tail
            __syncwarp();
            if (lane_id < kNumRDMARanks and is_lane_ready)
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
        }
    } else {
        // Combiners and coordinators
        // RDMA symmetric layout
        auto rdma_channel_data = SymBuffer<int8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[LEGACY_NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x =
            AsymBuffer<uint8_t>(
                local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels)
                .advance_also<LEGACY_NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head =
            AsymBuffer<int, LEGACY_NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                .advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels)
                                    .advance_also<LEGACY_NUM_MAX_NVL_PEERS>(nvl_buffers);

        // Combiner warp synchronization
        __shared__ volatile int forwarder_nvl_head[kNumForwarders][LEGACY_NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
        // ROCm: `barrier.sync <id>, <count>` -> sync_barrier (arch.cuh); 32 -> WARP_SIZE.
        // The two ids never meet, the forwarders and the receivers live on different SMs.
        auto sync_forwarder_smem = [&]() { sync_barrier(bar_expected, 0, (kNumForwarders + 1) * WARP_SIZE); };
        auto sync_rdma_receiver_smem = [&]() { sync_barrier(bar_expected, 1, (kNumRDMAReceivers + 1) * WARP_SIZE); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // Receive from NVL ranks and forward to RDMA ranks
            // NOTES: this part is using "large warps" for each RDMA ranks
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
            auto send_buffer =
                dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            // ROCm: `bar.sync <id>, <count>` -> sync_barrier; 32 -> WARP_SIZE. Its own counter,
            // a forwarder wave arrives here many times while barrier 0 is hit once.
            auto sync_large_warp = [&]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();
                } else {
                    sync_barrier(bar_expected_large, dst_rdma_rank + 2, kNumWarpsPerForwarder * WARP_SIZE);
                }
            };
            // ROCm: 16 -> kNumMaxBarriers, the LDS counter array is that wide
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= kNumMaxBarriers, "Barriers are not enough");

            // ROCm: the TMA staging buffer and its mbarriers are gone, `combine_token` reads
            // the peer's slots directly

            // Advance to the corresponding NVL buffer
            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);
            nvl_channel_head.advance(dst_rdma_rank);
            nvl_channel_tail.advance(dst_rdma_rank);

            // Clean shared memory and sync
            EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            lane_id < LEGACY_NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            sync_forwarder_smem();

            // Get count and cached head
            int cached_nvl_channel_tail_idx = 0;
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            num_tokens_to_combine -= num_tokens_prefix;
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            combined_nvl_head += num_tokens_prefix * LEGACY_NUM_MAX_NVL_PEERS;

            // Iterate over all tokens and combine by chunks
            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                // Check destination queue emptiness, or wait a buffer to be released
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
                    // Here, `token_start_idx` is the actual tail
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    // Timeout check
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf(
                            "DeepEP combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, head: %ld, tail: "
                            "%d, chunked: %d\n",
                            channel_id,
                            rdma_rank,
                            nvl_rank,
                            dst_rdma_rank,
                            ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)),
                            token_start_idx,
                            num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                // Combine and write to the RDMA buffer
                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    // Read expected head
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;
                    if (lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
                        // ROCm: ld_nc_global -> __ldg, `combined_nvl_head` is our own tensor
                        expected_head = __ldg(combined_nvl_head + token_idx * LEGACY_NUM_MAX_NVL_PEERS + lane_id);
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    // Wait lanes to be ready
                    start_time = clock64();
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        // Timeout check
                        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
                            printf(
                                "DeepEP combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, dst RDMA: %d, "
                                "tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                                channel_id,
                                rdma_rank,
                                nvl_rank,
                                lane_id,
                                dst_rdma_rank,
                                cached_nvl_channel_tail_idx,
                                token_idx,
                                num_tokens_to_combine,
                                sub_warp_id,
                                kNumWarpsPerForwarder,
                                expected_head);
                            trap();
                        }
                    }

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) +
                            hidden_int4_idx;
                    };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        // ROCm: ld_nc_global -> ld_coherent_sys_global, a peer NVL rank wrote this
                        return ld_coherent_sys_global(
                            reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token + hidden_bytes +
                                                     sizeof(SourceMeta)) +
                            topk_idx);
                    };
                    combine_token<LEGACY_NUM_MAX_NVL_PEERS, false, dtype_t, LEGACY_NUM_MAX_NVL_PEERS>(
                        expected_head >= 0,
                        expected_head,
                        lane_id,
                        hidden_int4,
                        num_topk,
                        static_cast<int4*>(shifted),
                        reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                        nullptr,
                        nullptr,
                        num_max_nvl_chunked_recv_tokens_per_rdma,
                        get_addr_fn,
                        recv_tw_fn);

                    // Update head
                    if (lane_id < LEGACY_NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();

                // Issue RDMA send
                if (sub_warp_id == kNumWarpsPerForwarder - 1) {
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                        // ROCm: IBGDA warp put -> rocSHMEM wave put
                        ::rocshmem::rocshmem_ctx_schar_put_nbi_wave(
                            ctx,
                            reinterpret_cast<signed char*>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token),
                            reinterpret_cast<const signed char*>(rdma_channel_data.send_buffer(dst_rdma_rank) +
                                                                 rdma_slot_idx * num_bytes_per_token),
                            num_bytes_per_msg,
                            translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank));
                        // ROCm: new, same QP-striping hazard as the dispatch sender above
                        ::rocshmem::rocshmem_ctx_quiet(ctx);
                    } else {
                        memory_fence();
                    }

                    // Write new RDMA tail
                    __syncwarp();
                    if (elect_one_sync()) {
                        // ROCm: IBGDA AMO -> rocSHMEM AMO, which serves the local PE too
                        ::rocshmem::rocshmem_ctx_ulong_atomic_add(ctx,
                                                                  rdma_channel_tail.buffer(rdma_rank),
                                                                  num_chunked_tokens,
                                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank));
                    }
                }
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;
        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // Receive from RDMA ranks and write to the output tensor
            // Clean shared memory and sync
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                // Read expected head
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                int expected_head = -1;
                if (lane_id < kNumRDMARanks) {
                    // ROCm: ld_nc_global -> __ldg, `combined_rdma_head` is our own tensor
                    expected_head = __ldg(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                                        : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                // Wait lanes to be ready
                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    // Timeout check
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf(
                            "DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, waiting: %ld, "
                            "expect: %d\n",
                            channel_id,
                            rdma_rank,
                            nvl_rank,
                            lane_id,
                            cached_channel_tail_idx,
                            token_idx,
                            expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Combine current token
                auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                    return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) +
                        hidden_int4_idx;
                };
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
                    // ROCm: ld_nc_global -> ld_coherent_sys_global, the NIC wrote this buffer
                    return ld_coherent_sys_global(
                        reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token +
                                                       hidden_bytes + sizeof(SourceMeta)) +
                        topk_idx);
                };
                combine_token<kNumRDMARanks, true, dtype_t, kNumTopkRDMARanks>(
                    expected_head >= 0,
                    expected_head,
                    lane_id,
                    hidden_int4,
                    num_topk,
                    combined_x + token_idx * hidden_int4,
                    combined_topk_weights + token_idx * num_topk,
                    bias_0 == nullptr ? nullptr : bias_0 + token_idx * hidden_int4,
                    bias_1 == nullptr ? nullptr : bias_1 + token_idx * hidden_int4,
                    num_max_rdma_chunked_recv_tokens,
                    get_addr_fn,
                    recv_tw_fn);
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;
        } else if (warp_role == WarpRole::kCoordinator) {
            // Coordinator
            // Sync shared memory status
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

            int last_rdma_head = 0;
            int last_nvl_head[kNumRDMARanks] = {0};
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
            int dst_nvl_rank = lane_id < LEGACY_NUM_MAX_NVL_PEERS ? lane_id : 0;
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");
            while (true) {
                // Retired
                // ROCm: __all_sync(0xffffffff, ..) -> __all, the whole wave participates
                if (not is_forwarder_sm and __all(lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;
                if (is_forwarder_sm and __all(lane_id >= kNumForwarders or forwarder_retired[lane_id]))
                    break;

                // Find minimum head for RDMA ranks
                if (not is_forwarder_sm) {
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers; ++i)
                        if (not rdma_receiver_retired[i])
                            min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                        lane_id < kNumRDMARanks) {
                        // ROCm: IBGDA AMO -> rocSHMEM AMO, which serves the local PE too
                        ::rocshmem::rocshmem_ctx_ulong_atomic_add(ctx,
                                                                  rdma_channel_head.buffer(rdma_rank),
                                                                  min_head - last_rdma_head,
                                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank));
                        last_rdma_head = min_head;
                    }
                } else {
                    // Find minimum head for NVL ranks
                    #pragma unroll
                    for (int i = 0; i < kNumRDMARanks; ++i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++j)
                            if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < LEGACY_NUM_MAX_NVL_PEERS)
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
                    }
                }

                // Nanosleep and let other warps work
                // ROCm: __nanosleep -> s_sleep, the builtin needs an immediate (compiled.cuh)
                s_sleep<LEGACY_NUM_WAIT_TICKS>();
            }
        }
        // ROCm: `kIdle` waves fall through here, they join no barrier
    }

    // ROCm: new, block-collective teardown, so it sits after every role branch has rejoined
    ::rocshmem::rocshmem_wg_ctx_destroy(&ctx);
}

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_channels,
             bool low_latency_mode) {
    // ROCm: 24 -> 14. A wave64 block holds 16 waves, not 32, and the widest SM here is
    // `kNumForwarders + 1` waves; 14 is the largest value that keeps every supported
    // `num_rdma_ranks` under that ceiling (see `get_num_combine_block_warps`).
    constexpr int kNumCombineForwarderWarps = 14;
    // ROCm: the TMA scratch is gone, so is the dynamic shared memory it needed

#define COMBINE_LAUNCH_CASE(num_rdma_ranks)                                           \
    {                                                                                 \
        auto combine_func =                                                           \
            low_latency_mode ? combine<true, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps> \
                             : combine<false, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps>; \
        LAUNCH_KERNEL(&cfg,                                                           \
                      combine_func,                                                   \
                      reinterpret_cast<int4*>(combined_x),                            \
                      combined_topk_weights,                                          \
                      is_combined_token_in_rank,                                      \
                      reinterpret_cast<const int4*>(x),                               \
                      topk_weights,                                                   \
                      reinterpret_cast<const int4*>(bias_0),                          \
                      reinterpret_cast<const int4*>(bias_1),                          \
                      combined_rdma_head,                                             \
                      combined_nvl_head,                                              \
                      reinterpret_cast<const SourceMeta*>(src_meta),                  \
                      rdma_channel_prefix_matrix,                                     \
                      rdma_rank_prefix_sum,                                           \
                      gbl_channel_prefix_matrix,                                      \
                      num_tokens,                                                     \
                      num_combined_tokens,                                            \
                      hidden,                                                         \
                      num_topk,                                                       \
                      rdma_buffer_ptr,                                                \
                      num_max_rdma_chunked_send_tokens,                               \
                      num_max_rdma_chunked_recv_tokens,                               \
                      buffer_ptrs,                                                    \
                      num_max_nvl_chunked_send_tokens,                                \
                      num_max_nvl_chunked_recv_tokens,                                \
                      rank,                                                           \
                      num_ranks);                                                     \
    }                                                                                 \
    break

    int num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;
    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
    // ROCm: must mirror the kernel's own defaults, the launch geometry is derived from them
    int num_rdma_receivers = num_forwarder_warps > LEGACY_NUM_MAX_NVL_PEERS ? num_forwarder_warps - LEGACY_NUM_MAX_NVL_PEERS : 1;
    int num_block_warps = std::max(LEGACY_NUM_MAX_NVL_PEERS + num_rdma_receivers, num_forwarder_warps) + 1;
    EP_HOST_ASSERT(num_rdma_ranks <= kNumCombineForwarderWarps);
    // ROCm: `num_forwarder_warps > 8` -> the block-width ceiling. On wave64 a block holds
    // 1024 / 64 == 16 waves, which caps combine at 12 RDMA peers (12 nodes).
    EP_HOST_ASSERT(num_forwarder_warps % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_block_warps <= 1024 / WARP_SIZE);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks >
                   std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks - num_warps_per_forwarder >= num_max_nvl_chunked_send_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);
    // ROCm: CUDA_R_16BF -> HIP_R_16BF, hipify does not translate it in every file
    EP_HOST_ASSERT(type == HIP_R_16BF);

    // ROCm: 32 -> WARP_SIZE, and the block is as wide as the busier of the two SMs
    SETUP_LAUNCH_CONFIG(num_channels * 2, num_block_warps * WARP_SIZE, stream);
    SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

}  // namespace internode

}  // namespace legacy

}  // namespace deep_ep

#else  // PRIMUS_TURBO_DEEPEP_HAS_INTERNODE

#include <primus_turbo/deep_ep/common/exception.cuh>

#include "api.cuh"

namespace primus_turbo::deep_ep::legacy {

namespace internode {

[[noreturn]] static void no_internode() {
    EP_HOST_ASSERT(false and "deep_ep internode is not ported to ROCm yet");
    __builtin_unreachable();
}

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode) {
    no_internode();
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              int scale_token_stride,
              int scale_hidden_stride,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode) {
    no_internode();
}

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode) {
    no_internode();
}

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_channels,
             bool low_latency_mode) {
    no_internode();
}

}  // namespace internode

}  // namespace primus_turbo::deep_ep::legacy

#endif  // PRIMUS_TURBO_DEEPEP_HAS_INTERNODE
