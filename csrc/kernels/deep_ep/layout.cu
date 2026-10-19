/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Ported from DeepEP (https://github.com/deepseek-ai/DeepEP), csrc/kernels/layout.cu.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include "launch.cuh"
#include "primus_turbo/deep_ep/configs.h"

namespace primus_turbo::deep_ep {

namespace layout {

template <typename topk_idx_t, int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(const topk_idx_t *topk_idx, int *num_tokens_per_rank,
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
#pragma unroll 2
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll 2
            for (int j = 0, expert_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        PRIMUS_TURBO_STATIC_CHECK(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
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
        PRIMUS_TURBO_DEVICE_CHECK(num_ranks % NUM_MAX_NVL_PEERS == 0 and
                                  num_ranks > NUM_MAX_NVL_PEERS);

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
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto       expert_begin        = rank_begin_idx * num_expert_per_rank;
        auto       expert_end          = rank_end_idx * num_expert_per_rank;

// Per-thread count
#pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll 2
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx           = topk_idx + i * num_topk;
            int  is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll 2
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll 2
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

#pragma unroll 2
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        PRIMUS_TURBO_STATIC_CHECK(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
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

// Intranode, one thread per token. The kernel above gives each block a few experts (or
// ranks) and makes it scan every token, so only ~num_experts / 4 blocks run and the
// single rank block walks all tokens alone. Here each token row is read once; counts
// are gathered with integer atomics in shared memory (order-independent, so still
// deterministic) and added once per block to the zeroed global counters.
template <typename topk_idx_t, int kNumThreads, int kMaxTopk>
__global__ void __launch_bounds__(kNumThreads)
    get_dispatch_layout_per_token(const topk_idx_t *topk_idx, int *num_tokens_per_rank,
                                  int *num_tokens_per_expert, bool *is_token_in_rank,
                                  int num_tokens, int num_topk, int num_ranks, int num_experts) {
    extern __shared__ int smem_counts[];
    const int             num_counts    = num_experts + num_ranks;
    int                  *expert_counts = smem_counts;
    int                  *rank_counts   = smem_counts + num_experts;
    for (int i = threadIdx.x; i < num_counts; i += kNumThreads)
        smem_counts[i] = 0;
    __syncthreads();

    const int token = static_cast<int>(blockIdx.x) * kNumThreads + static_cast<int>(threadIdx.x);
    if (token < num_tokens) {
        // Same validity ranges as the per-expert / per-rank blocks of the kernel above.
        const int   num_expert_per_rank = num_experts / num_ranks;
        const int   rank_expert_end     = num_ranks * num_expert_per_rank;
        const auto *row                 = topk_idx + static_cast<int64_t>(token) * num_topk;
        int         rank_of[kMaxTopk];
#pragma unroll
        for (int j = 0; j < kMaxTopk; ++j) {
            rank_of[j] = -1;
            if (j < num_topk) {
                const int expert_idx = static_cast<int>(row[j]);
                if (0 <= expert_idx and expert_idx < num_experts)
                    atomicAdd(expert_counts + expert_idx, 1);
                if (0 <= expert_idx and expert_idx < rank_expert_end)
                    rank_of[j] = expert_idx / num_expert_per_rank;
            }
        }

#pragma unroll
        for (int j = 0; j < kMaxTopk; ++j) {
            const int rank_idx = rank_of[j];
            if (rank_idx < 0)
                continue;
            bool first_rank = true;
#pragma unroll
            for (int i = 0; i < j; ++i)
                first_rank &= rank_of[i] != rank_idx;
            if (first_rank)
                atomicAdd(rank_counts + rank_idx, 1);
        }

        bool *out_row = is_token_in_rank + static_cast<int64_t>(token) * num_ranks;
        if (num_ranks % 8 == 0) {
            // 8 ranks per 64-bit store; rows stay 8-byte aligned since num_ranks % 8 == 0.
            for (int base = 0; base < num_ranks; base += 8) {
                uint64_t word = 0;
#pragma unroll
                for (int j = 0; j < kMaxTopk; ++j) {
                    const int r = rank_of[j] - base;
                    if (0 <= r and r < 8)
                        word |= 1ull << (8 * r);
                }
                *reinterpret_cast<uint64_t *>(out_row + base) = word;
            }
        } else {
            for (int r = 0; r < num_ranks; ++r) {
                bool in_rank = false;
#pragma unroll
                for (int j = 0; j < kMaxTopk; ++j)
                    in_rank |= rank_of[j] == r;
                out_row[r] = in_rank;
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_counts; i += kNumThreads) {
        const int count = smem_counts[i];
        if (count == 0)
            continue;
        if (i < num_experts)
            atomicAdd(num_tokens_per_expert + i, count);
        else
            atomicAdd(num_tokens_per_rank + (i - num_experts), count);
    }
}

template <typename topk_idx_t>
void get_dispatch_layout(const topk_idx_t *topk_idx, int *num_tokens_per_rank,
                         int *num_tokens_per_rdma_rank, int *num_tokens_per_expert,
                         bool *is_token_in_rank, int num_tokens, int num_topk, int num_ranks,
                         int num_experts, hipStream_t stream) {
    constexpr int kPerTokenThreads = 256, kPerTokenMaxTopk = 16;
    const size_t  per_token_smem = static_cast<size_t>(num_experts + num_ranks) * sizeof(int);
    if (num_tokens_per_rdma_rank == nullptr and num_topk <= kPerTokenMaxTopk and
        per_token_smem <= 32 * 1024) {
        PRIMUS_TURBO_CHECK_HIP(
            hipMemsetAsync(num_tokens_per_expert, 0, num_experts * sizeof(int), stream));
        PRIMUS_TURBO_CHECK_HIP(
            hipMemsetAsync(num_tokens_per_rank, 0, num_ranks * sizeof(int), stream));
        if (num_tokens == 0)
            return;
        get_dispatch_layout_per_token<topk_idx_t, kPerTokenThreads, kPerTokenMaxTopk>
            <<<(num_tokens + kPerTokenThreads - 1) / kPerTokenThreads, kPerTokenThreads,
               per_token_smem, stream>>>(topk_idx, num_tokens_per_rank, num_tokens_per_expert,
                                         is_token_in_rank, num_tokens, num_topk, num_ranks,
                                         num_experts);
        return;
    }

    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
    int           num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
                  (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    PRIMUS_TURBO_STATIC_CHECK(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0,
                              "Invalid number of ranks per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL_NON_COOPERATIVE(
        &cfg, (get_dispatch_layout<topk_idx_t, kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
        topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
        is_token_in_rank, num_tokens, num_topk, num_ranks, num_experts);
}

#define INSTANTIATE_GET_DISPATCH_LAYOUT(topk_idx_t)                                                \
    template void get_dispatch_layout<topk_idx_t>(const topk_idx_t *, int *, int *, int *, bool *, \
                                                  int, int, int, int, hipStream_t);
INSTANTIATE_GET_DISPATCH_LAYOUT(int32_t)
INSTANTIATE_GET_DISPATCH_LAYOUT(int64_t)
#undef INSTANTIATE_GET_DISPATCH_LAYOUT

} // namespace layout

} // namespace primus_turbo::deep_ep
