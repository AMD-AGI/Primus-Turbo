#include "intranode.cuh"
#include "primus_turbo/common.h"
#include "utils.cuh"

namespace primus_turbo::cco::pipelined_ep::intranode {

template <int kNumThreads, int kNumBlocks, int kNumRankPerNode, int kNumNodes,
          int NUM_OF_EXPERTS_PER_RANK>
__global__ __launch_bounds__(kNumThreads, 1) void scan_with_expert_stages(
    const bool *input_routing_map, tmp_state_t *tmp, int32_t *sparse_to_dense_map,
    bool *rdma_to_attn_map, bool *attn_to_rdma_map, int32_t *num_of_tokens_for_experts,
    bool *local_expert_routing_map, const int node_rank, const int local_rank,
    const int num_of_tokens_per_rank) {
    // Calculate the warps per block.
    constexpr int kNumWarpsPerBlock = kNumThreads / WARP_SIZE;

    // Calculate total threads count.
    constexpr int kNumTotalThreads = kNumThreads * kNumBlocks;

    // Calculate the number of tokens belong to each CUDA block, warp and thread.
    // We assign 1 token(row in routing map) to 1 thread.
    const int num_of_total_attn_tokens = num_of_tokens_per_rank * kNumRankPerNode * kNumNodes;
    // static_assert(NUM_OF_TOTAL_ATTN_TOKENS % kNumTotalThreads == 0, "NUM_OF_TOTAL_ATTN_TOKENS
    // must be multiple of kNumTotalThreads");
    const int num_of_tokens_per_thread = ((num_of_total_attn_tokens - 1) / kNumTotalThreads) + 1;
    const int num_of_tokens_per_warp   = num_of_tokens_per_thread * WARP_SIZE;
    const int num_of_tokens_per_block  = num_of_tokens_per_warp * kNumWarpsPerBlock;
    // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
    // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16
    // bools), so need to be paded to 16B per node. That means the size of rdma_to_attn_map should
    // be rdma_to_attn_map_size_per_node * kNumNodes.
    const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

    // For each token(row in routing map), calculate how many bytes need to be loaded from the
    // routing map and how to load them.
    static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
    constexpr int NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN = NUM_OF_EXPERTS_PER_RANK * kNumRankPerNode;
    using copy_t                                      = Copy_t<NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN>;
    static_assert(NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN % sizeof(copy_t) == 0,
                  "NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN and copy_t mismatch");
    constexpr int ROUTING_MAP_LOAD_ITER = NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN / sizeof(copy_t);

    // For each token, calculate how many bytes need to be store to sparse_to_dense_map.
    constexpr int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * kNumRankPerNode;
    using write_t = Copy_t<NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN>;
    static_assert(NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN % sizeof(write_t) == 0,
                  "NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN and write_t mismatch");
    constexpr int S2D_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN / sizeof(write_t);

    // How to convert per-expert routing info to per-rank routing info. We support any number of
    // expert per rank.
    using expert_to_rank_t = Reduce_t<NUM_OF_EXPERTS_PER_RANK>;
    static_assert(NUM_OF_EXPERTS_PER_RANK % sizeof(expert_to_rank_t) == 0,
                  "NUM_OF_EXPERTS_PER_RANK and expert_to_rank_t mismatch");
    constexpr int EXPERTS_TO_RANK_REDUCE_ITER = NUM_OF_EXPERTS_PER_RANK / sizeof(expert_to_rank_t);

    // How to convert per-rank routing info to per-node routing info. We support any number of ranks
    // per node(nvl domain).
    // using rank_to_node_t = Reduce_t<kNumRankPerNode>;
    // static_assert(kNumRankPerNode % sizeof(rank_to_node_t) == 0, "kNumRankPerNode and
    // rank_to_node_t mismatch"); constexpr int RANKS_TO_NODE_REDUCE_ITER = kNumRankPerNode /
    // sizeof(rank_to_node_t);

    // How do a warp save per-rank routing info back to shared memory. What's the max number of
    // elements does each thread save back.
    constexpr int kNumRanksPerThreads = ((kNumRankPerNode - 1) / WARP_SIZE) + 1;

    // Sum of per-rank routing info of all warps within the block.
    __shared__ int32_t warp_token_routing_map_sum[kNumWarpsPerBlock][kNumRankPerNode];
    // Sum of previous blocks' per-rank routing info.
    __shared__ int32_t previous_block_sum[kNumRankPerNode];

    // We assign contiguous tokens called chunk to each CUDA block, each CUDA block get the same
    // size of chunk.
    int block_starting_token = blockIdx.x * num_of_tokens_per_block;
    // warp id and lane id.
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    // We assign contiguous tokens called sub-chunk to each warp within a CUDA block, each warp
    // within a CUDA block get the same size of sub-chunk.
    int warp_starting_token = block_starting_token + warp_id * num_of_tokens_per_warp;
    // Within a sub-chunk, we assign tokens to thread in a interleave pattern. So each thread
    // process a token each time and each warp sum a tile of 32 tokens each time.
    int thread_starting_token = warp_starting_token + lane_id;

    // Step 0: Each warp sum the sub-chunk assigned to them and store the sum back to shared memory.
    // All warps within all CTA attend this step.
    // Also, some tokens need per-node info which store to rdma_to_attn_map, also processed here.

    // Sum of per-rank token routing map within a thread.
    int32_t token_routing_map_sum[kNumRankPerNode];
#pragma unroll
    for (int i = 0; i < kNumRankPerNode; i++) {
        token_routing_map_sum[i] = 0;
    }

    // #pragma unroll
    for (int i = 0; i < num_of_tokens_per_thread; i++) {
        // The global token id conditions for current token.
        int current_token_id = thread_starting_token + i * WARP_SIZE;
        // If the current token is out-of-bound, then just end summing tokens assigned to this
        // thread.
        if (current_token_id >= num_of_total_attn_tokens) {
            break;
        }
        int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * kNumRankPerNode);
        int current_token_local_rank =
            (current_token_id % (num_of_tokens_per_rank * kNumRankPerNode)) /
            num_of_tokens_per_rank;
        int current_token_local_id = current_token_id % num_of_tokens_per_rank;
        // If the token belongs to the inter-node group.
        // We need to calculate the per-node routing info and save back to rdma_to_attn_map.
        bool per_node_routing_info = (current_token_local_rank == local_rank);
        int  current_token_rdma_to_attn_map_id =
            current_token_node_rank * rdma_to_attn_map_size_per_node + current_token_local_id;
        // Global routing map load base addr for current token.
        const copy_t *routing_map_load_base_addr = reinterpret_cast<const copy_t *>(
            input_routing_map +
            current_token_id * (NUM_OF_EXPERTS_PER_RANK * kNumRankPerNode * kNumNodes) +
            node_rank * (NUM_OF_EXPERTS_PER_RANK * kNumRankPerNode));

        // Load the routing map for current token.
        bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * kNumRankPerNode];
#pragma unroll
        for (int j = 0; j < ROUTING_MAP_LOAD_ITER; j++) {
            *(reinterpret_cast<copy_t *>(token_routing_map) + j) = routing_map_load_base_addr[j];
        }

        // Convert the routing map to per rank routing info and accumulate to accumulator.
        // Also convert the per rank routing info to per node routing info.
        bool token_needed_by_this_node = false;
#pragma unroll
        for (int j = 0; j < kNumRankPerNode; j++) {
            bool token_needed_by_this_rank = false;
#pragma unroll
            for (int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++) {
                int              current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
                expert_to_rank_t reduction_data =
                    *(reinterpret_cast<expert_to_rank_t *>(token_routing_map) +
                      current_expert_to_rank_t_id);
                if (reduction_data != (expert_to_rank_t) 0) {
                    token_needed_by_this_rank = true;
                    break;
                }
            }
            if (token_needed_by_this_rank) {
                token_routing_map_sum[j] += 1;
                token_needed_by_this_node = true;
            }
        }

        // Save the per node routing info back to rdma_to_attn_map if needed.
        if (per_node_routing_info) {
            rdma_to_attn_map[current_token_rdma_to_attn_map_id] = token_needed_by_this_node;
        }
    }

// Each warp sum the per-rank routing info from all its threads.
#pragma unroll
    for (int i = 0; i < kNumRankPerNode; i++) {
        int     dst_tid  = i % WARP_SIZE;
        int     dst_id   = i / WARP_SIZE;
        int32_t temp_sum = __reduce_add_sync(~0, token_routing_map_sum[i]);
        if (lane_id == dst_tid) {
            token_routing_map_sum[dst_id] = temp_sum;
        }
    }

// Each warp store the sum of per-rank routing info back to shared memory.
#pragma unroll
    for (int i = 0; i < kNumRanksPerThreads; i++) {
        int element_id = i * WARP_SIZE + lane_id;
        if (element_id < kNumRankPerNode) {
            warp_token_routing_map_sum[warp_id][element_id] = token_routing_map_sum[i];
        }
    }

    // Sync within a CUDA block to make sure all warps have produced the per-rank sum data to the
    // shared memory before any thread can consume them to produce CUDA block level's sum data.
    __syncthreads();

    // Step 1: Communication between CUDA blocks. Each CUDA block's threads need to produce and
    // store the current block's per-rank sum data to global memory, and load and accumulate
    // previous blocks' per-rank sum data and save the result to shared memory.

    // Each thread within a CUDA block calculate the CUDA block level sum for a single rank at a
    // time.
    for (int i = threadIdx.x; i < kNumRankPerNode; i += kNumThreads) {
        int32_t rank_acc = 0;
// Calculate the sum of current rank within this CUDA block.
#pragma unroll
        for (int j = 0; j < kNumWarpsPerBlock; j++) {
            rank_acc += warp_token_routing_map_sum[j][i];
        }

        // Store the sum of current rank within this CUDA block to global memory for later scan
        // opeartions. Strong(atomic) store is needed to be visible to strong(atomic) load from
        // other blocks.
        tmp_state_t *tmp_dst = &tmp[blockIdx.x * kNumRankPerNode + i];
        tmp_state_t  tmp_data{PRIV_SUM, rank_acc};
        uint64_t     data = *reinterpret_cast<uint64_t *>(&tmp_data);
        asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                     :
                     : "l"(__cvta_generic_to_global(tmp_dst)), "l"(data)
                     : "memory");
    }

    // Each thread within a CUDA block load previous blocks' block level sum for a single rank at a
    // time.
    for (int i = threadIdx.x; i < kNumRankPerNode; i += kNumThreads) {
        int32_t previous_block_sum_for_current_rank = 0;
        for (int j = 0; j < blockIdx.x; j++) {
            tmp_state_t  tmp_data{EMPTY, 0};
            tmp_state_t *tmp_src = &tmp[j * kNumRankPerNode + i];
            do {
                // Load previous blocks' per-rank sum from global memory.
                // Strong(atomic) load is needed to view strong(atomic) store from other blocks.
                uint64_t data = 0;
                asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                             : "=l"(data)
                             : "l"(__cvta_generic_to_global(tmp_src))
                             : "memory");
                tmp_data = *reinterpret_cast<tmp_state_t *>(&data);
            } while (tmp_data.state != PRIV_SUM);
            previous_block_sum_for_current_rank += tmp_data.value;
        }
        previous_block_sum[i] = previous_block_sum_for_current_rank;
    }

    // Sync within a CUDA block to make sure all previous blocks' per-rank sum have been produced to
    // the shared memory before any thread can consume them in scan operation.
    __syncthreads();

    // Step 2: Each warp scan the sub-chunk assigned to them(the same sub-chunk as step 0) and
    // produce sparse_to_dense_map, local_expert_routing_map and num_of_tokens_for_experts.
    int32_t previous_token_sum[kNumRankPerNode];

// Each warp load the previous blocks' per-rank sum from shared memory.
#pragma unroll
    for (int i = 0; i < kNumRanksPerThreads; i++) {
        int element_id = i * WARP_SIZE + lane_id;
        if (element_id < kNumRankPerNode) {
            previous_token_sum[i] = previous_block_sum[element_id];
        }
    }

// Each warp accumulate the previous warps' per-rank sum from shared memory.
#pragma unroll
    for (int i = 0; i < kNumRanksPerThreads; i++) {
        int element_id = i * WARP_SIZE + lane_id;
        if (element_id < kNumRankPerNode) {
            for (int j = 0; j < warp_id; j++) {
                previous_token_sum[i] += warp_token_routing_map_sum[j][element_id];
            }
        }
    }

// Each warp broadcast the accumulated previous per-rank routing info to all its threads.
// Exact reverse of warp reduce operation.
#pragma unroll
    for (int i = kNumRankPerNode - 1; i >= 0; i--) {
        int src_tid           = i % WARP_SIZE;
        int src_id            = i / WARP_SIZE;
        previous_token_sum[i] = __shfl_sync(~0, previous_token_sum[src_id], src_tid);
    }

    // Each warp scan all the tiles within its sub-chunk.
    // #pragma unroll
    for (int i = 0; i < num_of_tokens_per_thread; i++) {
        // The global token id conditions for current token.
        int current_token_id = thread_starting_token + i * WARP_SIZE;
        // If the current token is out-of-bound, then mark it as out-of-bound.
        int token_out_of_bound = 0;
        if (current_token_id >= num_of_total_attn_tokens) {
            token_out_of_bound = 1;
        }
        // If the whole tiles are out-of-bound, the warp just finish and exit the scan loop
        // together.
        if (__all_sync(~0, token_out_of_bound) != 0) {
            break;
        }
        int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * kNumRankPerNode);
        int current_token_local_rank =
            (current_token_id % (num_of_tokens_per_rank * kNumRankPerNode)) /
            num_of_tokens_per_rank;
        int current_token_local_id = current_token_id % num_of_tokens_per_rank;

        // Global routing map load base addr for current token.
        const copy_t *routing_map_load_base_addr = reinterpret_cast<const copy_t *>(
            input_routing_map +
            current_token_id * (NUM_OF_EXPERTS_PER_RANK * kNumRankPerNode * kNumNodes) +
            node_rank * (NUM_OF_EXPERTS_PER_RANK * kNumRankPerNode));

        // Load the routing map for current token. Only load when the token is not out-of-bound.
        bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * kNumRankPerNode];
        if (token_out_of_bound == 0) {
#pragma unroll
            for (int j = 0; j < ROUTING_MAP_LOAD_ITER; j++) {
                *(reinterpret_cast<copy_t *>(token_routing_map) + j) =
                    routing_map_load_base_addr[j];
            }
        }

        // Convert the routing map to per rank routing info for current token,
        // then produce the per-rank final exclusive scan within the warp for this tile.
        int32_t final_ex_scan[kNumRankPerNode];
#pragma unroll
        for (int j = 0; j < kNumRankPerNode; j++) {
            int32_t temp_scan                 = 0;
            bool    token_needed_by_this_rank = false;
            // If the token is not out-of-bound, check whether this rank need this token.
            if (token_out_of_bound == 0) {
#pragma unroll
                for (int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++) {
                    int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
                    expert_to_rank_t reduction_data =
                        *(reinterpret_cast<expert_to_rank_t *>(token_routing_map) +
                          current_expert_to_rank_t_id);
                    if (reduction_data != (expert_to_rank_t) 0) {
                        token_needed_by_this_rank = true;
                        break;
                    }
                }
                if (token_needed_by_this_rank) {
                    temp_scan = 1;
                } else {
                    temp_scan = 0;
                }
            }

// Each warp perform a inclusive scan from all threads(lanes).
#pragma unroll
            for (int k = 1; k < WARP_SIZE; k *= 2) {
                int32_t temp = __shfl_up_sync(~0, temp_scan, k);
                if (lane_id >= k) {
                    temp_scan += temp;
                }
            }

            // The inclusive scan from last lane is the sum of this rank of this tile. Need to
            // accumulate that for later tiles.
            int32_t temp_sum = __shfl_sync(~0, temp_scan, WARP_SIZE - 1);

            // Make scan exclusive.
            int32_t exclusive_scan = __shfl_up_sync(~0, temp_scan, 1);
            temp_scan              = (lane_id >= 1) ? exclusive_scan : 0;

            // Calculate the final exclusive scan for current token. -1 represent that the current
            // rank does not need the current token.
            final_ex_scan[j] = token_needed_by_this_rank ? previous_token_sum[j] + temp_scan : -1;

            // Accumulate the sum to accumulator.
            previous_token_sum[j] += temp_sum;

            // Each thread save local routing map for this token of the local rank to
            // local_expert_routing_map if this token is needed by the local rank.
            if (j == local_rank && token_needed_by_this_rank) {
                expert_to_rank_t *local_expert_routing_map_store_base_addr =
                    reinterpret_cast<expert_to_rank_t *>(
                        local_expert_routing_map + (final_ex_scan[j] * NUM_OF_EXPERTS_PER_RANK));
#pragma unroll
                for (int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++) {
                    int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
                    local_expert_routing_map_store_base_addr[k] =
                        *(reinterpret_cast<expert_to_rank_t *>(token_routing_map) +
                          current_expert_to_rank_t_id);
                }
            }

            // The thread that processing the global last token save the final sum for current rank
            // to num_of_tokens_for_experts.
            if (current_token_id == num_of_total_attn_tokens - 1 && j == local_rank) {
                *num_of_tokens_for_experts = previous_token_sum[j];
            }
        }

        // Save final exclusive scan of this token back to sparse_to_dense_map if current token is
        // not out-of-bound and is needed.
        if (token_out_of_bound == 0 && current_token_local_rank == local_rank) {
            // sparse_to_dense_map store base addr for current token.
            write_t *sparse_to_dense_map_store_base_addr = reinterpret_cast<write_t *>(
                sparse_to_dense_map +
                (current_token_node_rank * num_of_tokens_per_rank + current_token_local_id) *
                    kNumRankPerNode);
#pragma unroll
            for (int j = 0; j < S2D_MAP_STORE_ITER; j++) {
                sparse_to_dense_map_store_base_addr[j] =
                    *(reinterpret_cast<write_t *>(final_ex_scan) + j);
            }
        }
    }
}

template <typename topk_idx_t, int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch(int4 *recv_x, float *recv_x_scales, int *recv_src_idx, topk_idx_t *recv_topk_idx,
             float *recv_topk_weights, int *recv_channel_offset, int *send_head, const int4 *x,
             const float *x_scales, const topk_idx_t *topk_idx, const float *topk_weights,
             const bool *is_token_in_rank, const int *channel_prefix_matrix, int num_tokens,
             int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
             int scale_token_stride, int scale_hidden_stride, void **buffer_ptrs, int rank,
             int num_max_send_tokens, int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    PRIMUS_TURBO_DEVICE_CHECK(num_sms % 2 == 0 and "num_sms must be even");

    if (is_sender) {
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx,
                               token_end_idx);

        // Iterate over all tokens and directly send to symmetric buffer
        for (auto token_idx = token_start_idx; token_idx < token_end_idx; token_idx++) {
        }
    } else {
        // Receive
    }
}

template <typename topk_idx_t>
void dispatch(void *recv_x, float *recv_x_scales, int *recv_src_idx, topk_idx_t *recv_topk_idx,
              float *recv_topk_weights, int *recv_channel_offset, int *send_head, const void *x,
              const float *x_scales, const topk_idx_t *topk_idx, const float *topk_weights,
              const bool *is_token_in_rank, const int *channel_prefix_matrix, int num_tokens,
              int hidden_int4, int num_topk, int num_experts, int num_scales,
              int scale_token_stride, int scale_hidden_stride, void **buffer_ptrs, int rank,
              int num_ranks, cudaStream_t stream, int num_sms) {}
} // namespace primus_turbo::cco::pipelined_ep::intranode
