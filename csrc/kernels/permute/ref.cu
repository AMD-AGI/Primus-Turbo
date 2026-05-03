template<int NUM_THREADS_PER_BLOCK,
         int NUM_OF_BLOCKS,
         int LOCAL_EXPERTS_PADDING_SIZE, 
         int NUM_OF_TOKENS_PER_CHUNK, 
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_EXPERTS_PER_RANK>
__launch_bounds__(NUM_THREADS_PER_BLOCK, 1)
__global__ void scan(const bool* input_routing_map, 
                     tmp_state_t* tmp, 
                     tmp_state_t* local_experts_tmp, 
                     int32_t* sparse_to_dense_map, 
                     bool* rdma_to_attn_map,
                     bool* attn_to_rdma_map,
                     int32_t* num_of_tokens_for_experts,
                     bool* local_expert_routing_map,
                     int32_t* dense_chunk_layout,
                     int32_t* dense_to_expert_map,
                     int32_t* num_of_local_experts_tokens,
                     int* token_drop_triggered,
                     const int node_rank,
                     const int local_rank,
                     const int local_experts_tokens_limit, // This size MUST be multiple of LOCAL_EXPERTS_PADDING_SIZE!
                     const int num_of_tokens_per_rank)
{
  // Calculate the warps per block.
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;

  // Calculate total threads count.
  constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;
  
  // Calculate the number of tokens belong to each CUDA block, warp and thread.
  // We assign 1 token(row in routing map) to 1 thread.
  const int num_of_total_attn_tokens = num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
  //static_assert(NUM_OF_TOTAL_ATTN_TOKENS % NUM_OF_TOTAL_THREADS == 0, "NUM_OF_TOTAL_ATTN_TOKENS must be multiple of NUM_OF_TOTAL_THREADS");
  const int num_of_tokens_per_thread = ((num_of_total_attn_tokens - 1) / NUM_OF_TOTAL_THREADS) + 1;
  const int num_of_tokens_per_warp = num_of_tokens_per_thread * WARP_SIZE;
  const int num_of_tokens_per_block = num_of_tokens_per_warp * NUM_OF_WARPS_PER_BLOCK;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // How many total chunks for all ranks.
  const int num_of_total_attn_chunks = num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
#endif
  // For each token(row in routing map), calculate how many bytes need to be loaded from the routing map and how to load them.
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  constexpr int NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN = NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE;
  using copy_t = Copy_t<NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN>;
  static_assert(NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN % sizeof(copy_t) == 0, "NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN and copy_t mismatch");
  constexpr int ROUTING_MAP_LOAD_ITER = NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN / sizeof(copy_t);

  // For each token, calculate how many bytes need to be store to sparse_to_dense_map.
  constexpr int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * NUM_OF_RANKS_PER_NODE;
  using write_t = Copy_t<NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN>;
  static_assert(NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN % sizeof(write_t) == 0, "NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN and write_t mismatch");
  constexpr int S2D_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN / sizeof(write_t);
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, calculate how many bytes need to be store to dense_to_expert_map per token.
  constexpr int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS = sizeof(int32_t) * NUM_OF_EXPERTS_PER_RANK;
  using local_experts_write_t = Copy_t<NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS>;
  static_assert(NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS % sizeof(local_experts_write_t) == 0, "NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS and local_experts_write_t mismatch");
  constexpr int D2E_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS / sizeof(local_experts_write_t);
#endif

  // How to convert per-expert routing info to per-rank routing info. We support any number of expert per rank.
  using expert_to_rank_t = Reduce_t<NUM_OF_EXPERTS_PER_RANK>;
  static_assert(NUM_OF_EXPERTS_PER_RANK % sizeof(expert_to_rank_t) == 0, "NUM_OF_EXPERTS_PER_RANK and expert_to_rank_t mismatch");
  constexpr int EXPERTS_TO_RANK_REDUCE_ITER = NUM_OF_EXPERTS_PER_RANK / sizeof(expert_to_rank_t);

  // How to convert per-rank routing info to per-node routing info. We support any number of ranks per node(nvl domain).
  //using rank_to_node_t = Reduce_t<NUM_OF_RANKS_PER_NODE>;
  //static_assert(NUM_OF_RANKS_PER_NODE % sizeof(rank_to_node_t) == 0, "NUM_OF_RANKS_PER_NODE and rank_to_node_t mismatch");
  //constexpr int RANKS_TO_NODE_REDUCE_ITER = NUM_OF_RANKS_PER_NODE / sizeof(rank_to_node_t);

  // How do a warp save per-rank routing info back to shared memory. What's the max number of elements does each thread save back.
  constexpr int NUM_OF_RANKS_PER_THREAD = ((NUM_OF_RANKS_PER_NODE - 1) / WARP_SIZE) + 1;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // How do a warp save local experts' routing info back to shared memory. What's the max number of elements does each thread save back.
  constexpr int NUM_OF_LOCAL_EXPERTS_PER_THREAD = ((NUM_OF_EXPERTS_PER_RANK - 1) / WARP_SIZE) + 1;
#endif

  // Sum of per-rank routing info of all warps within the block.
  __shared__ int32_t warp_token_routing_map_sum[NUM_OF_WARPS_PER_BLOCK][NUM_OF_RANKS_PER_NODE];
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Sum of local experts' routing info of all warps within the block.
  __shared__ int32_t warp_token_local_experts_routing_map_sum[NUM_OF_WARPS_PER_BLOCK][NUM_OF_EXPERTS_PER_RANK];
#endif
  // Sum of previous blocks' per-rank routing info.
  __shared__ int32_t previous_block_sum[NUM_OF_RANKS_PER_NODE];
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Sum of all blocks' local experts' routing info.
  __shared__ int32_t all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];
  // Sum of previous blocks' local experts' routing info accumulated with previous local experts' routing info.
  __shared__ int32_t previous_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];
#endif

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Init shared memory which are used as accumulator.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
    all_block_local_experts_sum[i] = 0;
    previous_block_local_experts_sum[i] = 0;
  }
#endif

  // We assign contiguous tokens called chunk to each CUDA block, each CUDA block get the same size of chunk.
  int block_starting_token = blockIdx.x * num_of_tokens_per_block;
  // warp id and lane id.
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  // We assign contiguous tokens called sub-chunk to each warp within a CUDA block, each warp within a CUDA block get the same size of sub-chunk.
  int warp_starting_token = block_starting_token + warp_id * num_of_tokens_per_warp;
  // Within a sub-chunk, we assign tokens to thread in a interleave pattern. So each thread process a token each time and each warp sum a tile of 32 tokens each time.
  int thread_starting_token = warp_starting_token + lane_id;
  
  // Step 0: Each warp sum the sub-chunk assigned to them and store the sum back to shared memory.
  // All warps within all CTA attend this step.
  // Also, some tokens need per-node info which store to rdma_to_attn_map, also processed here.

  // Sum of per-rank token routing map within a thread.
  int32_t token_routing_map_sum[NUM_OF_RANKS_PER_NODE];
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
    token_routing_map_sum[i] = 0;
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Sum of local experts' token routing map within a thread.
  int32_t token_local_experts_routing_map_sum[NUM_OF_EXPERTS_PER_RANK];
  #pragma unroll
  for(int i = 0; i < NUM_OF_EXPERTS_PER_RANK; i++){
    token_local_experts_routing_map_sum[i] = 0;
  }
#endif

  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then just end summing tokens assigned to this thread. 
    if(current_token_id >= num_of_total_attn_tokens){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;
    // If the token belongs to the inter-node group.
    // We need to calculate the per-node routing info and save back to rdma_to_attn_map.
    bool per_node_routing_info = (current_token_local_rank == local_rank);
    int current_token_rdma_to_attn_map_id = current_token_node_rank * rdma_to_attn_map_size_per_node + current_token_local_id;
    // Global routing map load base addr for current token.
    const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                               current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                               node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

    // Load the routing map for current token.
    bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
    #pragma unroll
    for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
      *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
    }

    // Convert the routing map to per rank routing info and accumulate to accumulator.
    // Also convert the per rank routing info to per node routing info.
    // When permute fusion is enabled, also accumulate local experts to accumulator.
    bool token_needed_by_this_node = false;
    #pragma unroll
    for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
      bool token_needed_by_this_rank = false;
      #pragma unroll
      for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
        int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
        expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
        if(reduction_data != (expert_to_rank_t)0){
          token_needed_by_this_rank = true;
          break;
        }
      }
      if(token_needed_by_this_rank){
        token_routing_map_sum[j] += 1;
        token_needed_by_this_node = true;
      }
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
      if(j == local_rank){
        int current_local_expert_id = j * NUM_OF_EXPERTS_PER_RANK;
        #pragma unroll
        for(int k = 0; k < NUM_OF_EXPERTS_PER_RANK; k++){
          token_local_experts_routing_map_sum[k] += (int32_t)(token_routing_map[current_local_expert_id + k]);
        }
      }
#endif
    }

    // Save the per node routing info back to rdma_to_attn_map if needed.
    if(per_node_routing_info){
      rdma_to_attn_map[current_token_rdma_to_attn_map_id] = token_needed_by_this_node;
    }
  }

  // Each warp sum the per-rank routing info from all its threads.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
    int dst_tid = i % WARP_SIZE;
    int dst_id = i / WARP_SIZE;
    int32_t temp_sum = __reduce_add_sync(~0, token_routing_map_sum[i]);
    if(lane_id == dst_tid){
      token_routing_map_sum[dst_id] = temp_sum;
    }
  }

  // Each warp store the sum of per-rank routing info back to shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      warp_token_routing_map_sum[warp_id][element_id] = token_routing_map_sum[i];
    }
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, each warp sum the local experts' routing info from all its threads.
  #pragma unroll
  for(int i = 0; i < NUM_OF_EXPERTS_PER_RANK; i++){
    int dst_tid = i % WARP_SIZE;
    int dst_id = i / WARP_SIZE;
    int32_t temp_sum = __reduce_add_sync(~0, token_local_experts_routing_map_sum[i]);
    if(lane_id == dst_tid){
      token_local_experts_routing_map_sum[dst_id] = temp_sum;
    }
  }

  // When permute fusion is enabled, each warp store the sum of local experts' routing info back to shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_LOCAL_EXPERTS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_EXPERTS_PER_RANK){
      warp_token_local_experts_routing_map_sum[warp_id][element_id] = token_local_experts_routing_map_sum[i];
    }
  }
#endif

  // Sync within a CUDA block to make sure all warps have produced the per-rank sum data to the shared memory before any thread can consume them to produce CUDA block level's sum data.
  // When permute fusion is enabled, also make sure all warps have produced the local experts' sum data to the shared memory.
  __syncthreads();

  // Step 1: Communication between CUDA blocks. Each CUDA block's threads need to produce and store the current block's per-rank sum data to global memory,
  // and load and accumulate previous blocks' per-rank sum data and save the result to shared memory.
  // When permute fusion is enabled, Each CUDA block's threads also need to produce and store the current block's local experts' sum data to global memory,
  // and load and accumulate all & previous blocks' local experts' sum data and save the result to shared memory. This is due to the layout requirement of local experts' output buffer.

  // Each thread within a CUDA block calculate the CUDA block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < NUM_OF_RANKS_PER_NODE; i += NUM_THREADS_PER_BLOCK){
    int32_t rank_acc = 0;
    // Calculate the sum of current rank within this CUDA block.
    #pragma unroll
    for(int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++){
      rank_acc += warp_token_routing_map_sum[j][i];
    }

    // Store the sum of current rank within this CUDA block to global memory for later scan opeartions.
    // Strong(atomic) store is needed to be visible to strong(atomic) load from other blocks.
    tmp_state_t* tmp_dst = &tmp[blockIdx.x * NUM_OF_RANKS_PER_NODE + i];
    tmp_state_t tmp_data{PRIV_SUM, rank_acc};
    uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                  :
                  : "l"(__cvta_generic_to_global(tmp_dst)), "l"(data)
                  : "memory");
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, each thread within a CUDA block calculate the CUDA block level sum for a local expert at a time.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
    int32_t local_expert_acc = 0;
    // Calculate the sum of current local expert within this CUDA block.
    #pragma unroll
    for(int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++){
      local_expert_acc += warp_token_local_experts_routing_map_sum[j][i];
    }

    // Store the sum of current local expert within this CUDA block to global memory for later scan opeartions.
    // Strong(atomic) store is needed to be visible to strong(atomic) load from other blocks.
    tmp_state_t* tmp_dst = &local_experts_tmp[blockIdx.x * NUM_OF_EXPERTS_PER_RANK + i];
    tmp_state_t tmp_data{PRIV_SUM, local_expert_acc};
    uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                  :
                  : "l"(__cvta_generic_to_global(tmp_dst)), "l"(data)
                  : "memory");
  }
#endif

  // Each thread within a CUDA block load previous blocks' block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < NUM_OF_RANKS_PER_NODE; i += NUM_THREADS_PER_BLOCK){
    int32_t previous_block_sum_for_current_rank = 0;
    for(int j = 0; j < blockIdx.x; j++){
      tmp_state_t tmp_data{EMPTY, 0};
      tmp_state_t* tmp_src = &tmp[j * NUM_OF_RANKS_PER_NODE + i];
      do{
          // Load previous blocks' per-rank sum from global memory.
          // Strong(atomic) load is needed to view strong(atomic) store from other blocks.
          uint64_t data = 0;
          asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                        : "=l"(data)
                        : "l"(__cvta_generic_to_global(tmp_src))
                        : "memory");
          tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
      }while(tmp_data.state != PRIV_SUM);
      previous_block_sum_for_current_rank += tmp_data.value;
    }
    previous_block_sum[i] = previous_block_sum_for_current_rank;
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, all threads within a CUDA block load all blocks' block level sum and accumulate to shared memory.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK * NUM_OF_BLOCKS; i += NUM_THREADS_PER_BLOCK){
    // Which block and which local expert is this sum element belongs to.
    int block_index = i / NUM_OF_EXPERTS_PER_RANK;
    int local_expert_index = i % NUM_OF_EXPERTS_PER_RANK;
    // Poll the sum element from global memory.
    tmp_state_t tmp_data{EMPTY, 0};
    tmp_state_t* tmp_src = &local_experts_tmp[i];
    do{
        // Load a block-level local expert's sum from global memory.
        // Strong(atomic) load is needed to view strong(atomic) store from other blocks.
        uint64_t data = 0;
        asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                      : "=l"(data)
                      : "l"(__cvta_generic_to_global(tmp_src))
                      : "memory");
        tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
    }while(tmp_data.state != PRIV_SUM);

    // Atomically add the block-level local expert's sum element to shared memory to produce all blocks' sum and previous blocks' sum.
    atomicAdd_block(&all_block_local_experts_sum[local_expert_index], tmp_data.value);
    if(block_index < (int)blockIdx.x){
      atomicAdd_block(&previous_block_local_experts_sum[local_expert_index], tmp_data.value);
    }
  }
#endif

  // Sync within a CUDA block to make sure all previous blocks' per-rank sum have been produced to the shared memory before any thread can consume them in scan operation.
  // When permute fusion is enabled, also make sure all and previous blocks' local experts' sum have been produced to the shared memory.
  __syncthreads();

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE 
  // Load sum of all blocks' local experts' routing info to produce the accumulation of previous local experts' routing info. 
  int32_t thread_local_all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK - 1];
  // Only threads which will participate in accumulation will need to load the data from the shared memory.
  if(threadIdx.x < NUM_OF_EXPERTS_PER_RANK){
    #pragma unroll
    for(int i = 0; i < NUM_OF_EXPERTS_PER_RANK; i++){
      thread_local_all_block_local_experts_sum[i] = all_block_local_experts_sum[i];
    }
  }
  
  // When permute fusion is enabled, all threads within a CUDA block produced sum of previous blocks' local experts' routing info accumulated with previous local experts' routing info.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
    int32_t current_expert_previous_block_sum = previous_block_local_experts_sum[i];
    int32_t previous_experts_acc = 0;
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
    int32_t previous_experts_acc_plus_current_expert_valid_tokens;
    int32_t current_expert_valid_tokens;
    #pragma unroll
    for(int j = 0; j < NUM_OF_EXPERTS_PER_RANK; j++){
      if(j < i){
        // local experts sum can be >= zero, so need to handle the corner case.
        int num_of_padding_tile = (thread_local_all_block_local_experts_sum[j] % LOCAL_EXPERTS_PADDING_SIZE == 0) ? (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE)
                                                                                                                  : (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE + 1);
        int32_t local_expert_sum_with_padding = num_of_padding_tile * LOCAL_EXPERTS_PADDING_SIZE;
        previous_experts_acc += local_expert_sum_with_padding;
      }else if(j == i){
        current_expert_valid_tokens = thread_local_all_block_local_experts_sum[j];
        previous_experts_acc_plus_current_expert_valid_tokens = previous_experts_acc + thread_local_all_block_local_experts_sum[j];
      }
    }
#else
    #pragma unroll
    for(int j = 0; j < NUM_OF_EXPERTS_PER_RANK - 1; j++){
      if(j < i){
        // local experts sum can be >= zero, so need to handle the corner case.
        int num_of_padding_tile = (thread_local_all_block_local_experts_sum[j] % LOCAL_EXPERTS_PADDING_SIZE == 0) ? (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE)
                                                                                                                  : (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE + 1);
        int32_t local_expert_sum_with_padding = num_of_padding_tile * LOCAL_EXPERTS_PADDING_SIZE;
        previous_experts_acc += local_expert_sum_with_padding;
      }
    }
#endif
    previous_block_local_experts_sum[i] = current_expert_previous_block_sum + previous_experts_acc;
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
    // First block will need to save all local experts' sum back to output buffer subject to token drop conditions.
    // First block also need to determine whether token drop is triggered.
    if(blockIdx.x == 0){
      int32_t num_of_current_experts_tokens;
      if(local_experts_tokens_limit > previous_experts_acc){
        // If previous local experts have not already fully occupied the local expert buffer, at least some valid tokens from current experts can be stored to the buffer.
        // This code path ONLY work when local_experts_tokens_limit is guarantee to be multiple of LOCAL_EXPERTS_PADDING_SIZE!
        if(local_experts_tokens_limit >= previous_experts_acc_plus_current_expert_valid_tokens){
          // If the local expert buffer's capacity can hold all the valid tokens from current expert, all valid tokens from current expert can be stored to the buffer.
          num_of_current_experts_tokens = current_expert_valid_tokens;
        }else{
          // If the local expert buffer's capacity cannot hold all the valid tokens from current expert, only partial of valid tokens can be strored to the buffer.
          num_of_current_experts_tokens = local_experts_tokens_limit - previous_experts_acc;
        }
      }else{
        // If all tokens from previous local experts(including both valid tokens and padding tokens) already exceed local expert buffer capacity, no more space for current local expert.
        num_of_current_experts_tokens = 0;
      }
      num_of_local_experts_tokens[i] = num_of_current_experts_tokens;

      // The thread which process the last local expert should determine whether token drop is triggered.
      if(i == NUM_OF_EXPERTS_PER_RANK - 1){
        // If all tokens from all local experts(including both valid tokens and padding tokens) do not exceed local expert buffer capacity, token drop is not triggered. Otherwise triggered.
        // We can use the following condition to determine whether token drop is triggered ONLY when local_experts_tokens_limit is guarantee to be multiple of LOCAL_EXPERTS_PADDING_SIZE!
        if(previous_experts_acc_plus_current_expert_valid_tokens <= local_experts_tokens_limit){
          *token_drop_triggered = 0;
        }else{
          *token_drop_triggered = 1;
        }
      }
    }
#endif
  }

  // Sync within a CUDA block to make sure all the final accumulated previous blocks' local experts' routing info have been produced to the shared memory 
  // before any thread can consume them in scan operation.
  __syncthreads();

#ifndef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
  // First block will need to save all local experts' sum back to output buffer.
  if(blockIdx.x == 0){
    for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
      num_of_local_experts_tokens[i] = all_block_local_experts_sum[i];
    }
  }
#endif
#endif

  // Step 2: Each warp scan the sub-chunk assigned to them(the same sub-chunk as step 0) and produce sparse_to_dense_map, local_expert_routing_map and num_of_tokens_for_experts.
  int32_t previous_token_sum[NUM_OF_RANKS_PER_NODE];

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE 
  // When permute fusion is enabled, each warp will also need to scan and produce dense_chunk_layout and dense_to_expert_map.
  int32_t previous_token_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];
#endif

  // Each warp load the previous blocks' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      previous_token_sum[i] = previous_block_sum[element_id];
    }
  }

  // Each warp accumulate the previous warps' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      for(int j = 0; j < warp_id; j++){
        previous_token_sum[i] += warp_token_routing_map_sum[j][element_id];
      }
    }
  }

  // Each warp broadcast the accumulated previous per-rank routing info to all its threads.
  // Exact reverse of warp reduce operation.
  #pragma unroll
  for(int i = NUM_OF_RANKS_PER_NODE - 1; i >= 0 ; i--){
    int src_tid = i % WARP_SIZE;
    int src_id = i / WARP_SIZE;
    previous_token_sum[i] = __shfl_sync(~0, previous_token_sum[src_id], src_tid);
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, each warp load the previous blocks' local experts' sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_LOCAL_EXPERTS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_EXPERTS_PER_RANK){
      previous_token_local_experts_sum[i] = previous_block_local_experts_sum[element_id];
    }
  }

  // When permute fusion is enabled, each warp accumulate the previous warps' local experts' sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_LOCAL_EXPERTS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_EXPERTS_PER_RANK){
      for(int j = 0; j < warp_id; j++){
        previous_token_local_experts_sum[i] += warp_token_local_experts_routing_map_sum[j][element_id];
      }
    }
  }

  // Each warp broadcast the accumulated previous local experts' routing info to all its threads.
  // Exact reverse of warp reduce operation.
  #pragma unroll
  for(int i = NUM_OF_EXPERTS_PER_RANK - 1; i >= 0 ; i--){
    int src_tid = i % WARP_SIZE;
    int src_id = i / WARP_SIZE;
    previous_token_local_experts_sum[i] = __shfl_sync(~0, previous_token_local_experts_sum[src_id], src_tid);
  }
#endif

  // Each warp scan all the tiles within its sub-chunk.
  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then mark it as out-of-bound. 
    int token_out_of_bound = 0;
    if(current_token_id >= num_of_total_attn_tokens){
      token_out_of_bound = 1;
    }
    // If the whole tiles are out-of-bound, the warp just finish and exit the scan loop together.
    if(__all_sync(~0, token_out_of_bound) != 0){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    // When permute fusion is enabled, calculate chunk-related info.
    bool first_token_of_a_chunk = (current_token_local_id % NUM_OF_TOKENS_PER_CHUNK) == 0;
    int current_token_global_chunk_id = (current_token_node_rank * NUM_OF_RANKS_PER_NODE + current_token_local_rank) * num_of_chunks_per_rank +
                                        (current_token_local_id / NUM_OF_TOKENS_PER_CHUNK);
    // If this token belongs to a valid attn token chunk, and it is the first token of this chunk, then we need to save this token's per-rank ex-scan of local rank to dense_chunk_layout map.
    bool token_needed_by_dense_chunk_layout = first_token_of_a_chunk && current_token_global_chunk_id > 0 && current_token_global_chunk_id < num_of_total_attn_chunks;
#endif

    // Global routing map load base addr for current token.
    const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                               current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                               node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

    // Load the routing map for current token. Only load when the token is not out-of-bound.
    bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
    if(token_out_of_bound == 0){
      #pragma unroll
      for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
        *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
      }
    }
    
    // Convert the routing map to per rank routing info for current token, 
    // then produce the per-rank final exclusive scan within the warp for this tile.
    int32_t final_ex_scan[NUM_OF_RANKS_PER_NODE];
    #pragma unroll
    for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
      int32_t temp_scan = 0;
      bool token_needed_by_this_rank = false;
      // Old warp-level scan implementation, using warp shuffle, suitable for general data type, but not fast enough for bool type.
      // If the token is not out-of-bound, check whether this rank need this token.
      /*if(token_out_of_bound == 0){
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          temp_scan = 1;
        }else{
          temp_scan = 0;
        }
      }
      
      // Each warp perform a inclusive scan from all threads(lanes).
      #pragma unroll
      for(int k = 1; k < WARP_SIZE; k *= 2){
        int32_t temp = __shfl_up_sync(~0, temp_scan, k);
        if(lane_id >= k){
          temp_scan += temp;
        }
      }

      // The inclusive scan from last lane is the sum of this rank of this tile. Need to accumulate that for later tiles.
      int32_t temp_sum = __shfl_sync(~0, temp_scan, WARP_SIZE - 1);

      // Make scan exclusive.
      int32_t exclusive_scan = __shfl_up_sync(~0, temp_scan, 1);
      temp_scan = (lane_id >= 1) ? exclusive_scan : 0;*/

      // New warp-level scan implementation for bool value, using warp vote instead of warp shuffle. Warp vote is way faster than warp shuffle.
      // If the token is not out-of-bound, check whether this rank need this token.
      if(token_out_of_bound == 0){
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
      }

      // Each warp vote to create a bit mask indicating which token is needed by this rank within this tile.
      unsigned vote_result = __ballot_sync(~0, token_needed_by_this_rank);
      // The sum of this rank of this tile. Need to accumulate that for later tiles.
      int32_t temp_sum = __popc(vote_result);
      // Each warp perform a exclusive scan from all threads(lanes).
      temp_scan = __popc(vote_result << (WARP_SIZE - lane_id));

      // Calculate the final exclusive scan for current token. -1 represent that the current rank does not need the current token. 
      final_ex_scan[j] = token_needed_by_this_rank ? previous_token_sum[j] + temp_scan : -1;

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
      // When permute fusion is enabled, we need to do extra work to local experts of local rank.
      if(j == local_rank){
        int32_t final_local_experts_ex_scan[NUM_OF_EXPERTS_PER_RANK];
        // First calculate ex-scan for this tile for all local experts of the local rank.
        #pragma unroll
        for(int k = 0; k < NUM_OF_EXPERTS_PER_RANK; k++){
          bool token_needed_by_this_local_expert = false;
          if(token_out_of_bound == 0){
            token_needed_by_this_local_expert = token_routing_map[j * NUM_OF_EXPERTS_PER_RANK + k];
          }
          unsigned local_expert_vote_result = __ballot_sync(~0, token_needed_by_this_local_expert);
          int32_t local_expert_temp_sum = __popc(local_expert_vote_result);
          int32_t local_expert_temp_scan = __popc(local_expert_vote_result << (WARP_SIZE - lane_id));
          final_local_experts_ex_scan[k] = token_needed_by_this_local_expert ? previous_token_local_experts_sum[k] + local_expert_temp_scan : -1;
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
          if(final_local_experts_ex_scan[k] >= local_experts_tokens_limit){
            final_local_experts_ex_scan[k] = -1;
          }
#endif
          previous_token_local_experts_sum[k] += local_expert_temp_sum;
        }
        // Then save the ex-scan back to dense_to_expert map if the current token is needed by local rank.
        if(token_needed_by_this_rank){
          local_experts_write_t* dense_to_expert_map_store_base_addr = reinterpret_cast<local_experts_write_t*>(dense_to_expert_map + final_ex_scan[j] * NUM_OF_EXPERTS_PER_RANK);
          #pragma unroll
          for(int k = 0; k < D2E_MAP_STORE_ITER; k++){
            dense_to_expert_map_store_base_addr[k] = *(reinterpret_cast<local_experts_write_t*>(final_local_experts_ex_scan) + k);
          }
        }
        // If condition meet, we also need to save current token's local rank's ex-scan to dense_chunk_layout map.
        if(token_needed_by_dense_chunk_layout){
          dense_chunk_layout[current_token_global_chunk_id - 1] = previous_token_sum[j] + temp_scan;
        }
      }
#else
      // Each thread save local routing map for this token of the local rank to local_expert_routing_map if this token is needed by the local rank.
      if(j == local_rank && token_needed_by_this_rank){
        expert_to_rank_t* local_expert_routing_map_store_base_addr = reinterpret_cast<expert_to_rank_t*>(local_expert_routing_map + (final_ex_scan[j] * NUM_OF_EXPERTS_PER_RANK));
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          local_expert_routing_map_store_base_addr[k] = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
        }
      }
#endif
      // Accumulate the sum to accumulator.
      previous_token_sum[j] += temp_sum;
      // The thread that processing the global last token save the final sum for current rank to num_of_tokens_for_experts.
      if(current_token_id == num_of_total_attn_tokens - 1 && j == local_rank){
        *num_of_tokens_for_experts = previous_token_sum[j];
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
        // When permute fusion is enabled, also need to save the final sum for current rank to the last element of dense_chunk_layout.
        dense_chunk_layout[num_of_total_attn_chunks - 1] = previous_token_sum[j];
#endif
      }
    }

    // Save final exclusive scan of this token back to sparse_to_dense_map if current token is not out-of-bound and is needed. 
    if(token_out_of_bound == 0 && current_token_local_rank == local_rank){
      // sparse_to_dense_map store base addr for current token.
      write_t* sparse_to_dense_map_store_base_addr = reinterpret_cast<write_t*>(sparse_to_dense_map + 
                                                                                (current_token_node_rank * num_of_tokens_per_rank + current_token_local_id) * NUM_OF_RANKS_PER_NODE);
      #pragma unroll
      for(int j = 0; j < S2D_MAP_STORE_ITER; j++){
        sparse_to_dense_map_store_base_addr[j] = *(reinterpret_cast<write_t*>(final_ex_scan) + j);
      }
    }
  }

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // Step 3: When NUM_OF_NODES > 1, we need to produce attn_to_rdma_map.
  // Since each token(row) is fully independent, each token(row) is assigned to each threads in a interleave pattern.
  if constexpr(NUM_OF_NODES != 1){
    const int num_of_total_token_rows = (NUM_OF_NODES - 1) * num_of_tokens_per_rank;
    //static_assert(NUM_OF_TOTAL_TOKEN_ROWS % NUM_OF_TOTAL_THREADS == 0, "NUM_OF_TOTAL_TOKEN_ROWS must be multiple of NUM_OF_TOTAL_THREADS.");
    const int num_of_token_rows_per_thread = ((num_of_total_token_rows - 1) / NUM_OF_TOTAL_THREADS) + 1;

    int tid = threadIdx.x + blockIdx.x * NUM_THREADS_PER_BLOCK;

    //#pragma unroll
    for(int i = 0; i < num_of_token_rows_per_thread; i++){
      int current_token_id = i * NUM_OF_TOTAL_THREADS + tid;
      // If the current token is out-of-bound, then just end processing token rows assigned to this thread. 
      if(current_token_id >= num_of_total_token_rows){
        break;
      }
      int current_token_attn_to_rdma_map_node_id = current_token_id % (NUM_OF_NODES - 1);
      int current_token_node_id = current_token_attn_to_rdma_map_node_id < node_rank ? current_token_attn_to_rdma_map_node_id : current_token_attn_to_rdma_map_node_id + 1;
      int current_token_local_id = current_token_id / (NUM_OF_NODES - 1);

      const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                                ((node_rank * NUM_OF_RANKS_PER_NODE + local_rank) * num_of_tokens_per_rank + current_token_local_id) *
                                                                                (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                                (current_token_node_id * NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

      bool* attn_to_rdma_map_base_addr = attn_to_rdma_map + (current_token_local_id * (NUM_OF_NODES - 1) + current_token_attn_to_rdma_map_node_id);

      // Load the routing map for current token row.
      bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
      #pragma unroll
      for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
        *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
      }

      // Convert the routing map to per rank routing info and then to per node routing info.
      bool token_needed_by_this_node = false;
      #pragma unroll
      for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
        bool token_needed_by_this_rank = false;
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          token_needed_by_this_node = true;
          break;
        }
      }

      *attn_to_rdma_map_base_addr = token_needed_by_this_node;
    }
  }
#endif
}