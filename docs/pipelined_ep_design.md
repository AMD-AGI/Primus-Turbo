# Pipelined Expert-Parallel Dispatch Design

基于 DeepEP intranode kernel (`csrc/kernels/deep_ep/intranode.cu`) 的 pipeline overlap 方案设计。

## 1. 问题背景

### 1.1 DeepEP 现有架构

DeepEP intranode dispatch 的数据流：

```
notify_dispatch:
  SM 0:  barrier → 交换 per-rank/per-expert 计数 → prefix sum → barrier
  SM 1~R: 计算 channel_prefix_matrix[dst_rank][channel]

dispatch:
  偶数 SM (sender channel):
    per-rank warp group → 遍历 channel 内 token (token id 顺序)
    → 检查 is_token_in_rank 去重
    → 写入 remote rank 的 per-channel ring buffer (head/tail 流控)
    → 记录 send_head[token, rank] 给 combine 用

  奇数 SM (receiver channel):
    per-rank warp group → poll channel_tail_idx
    → 从 ring buffer 拷贝到 flat recv_x[rank_offset + channel_offset]
    → 同时拷贝 recv_topk_idx, recv_src_idx
```

### 1.2 当前的问题

传统 MoE 流程存在多个串行瓶颈：

```
dispatch_all → D2H sync(拿expert count) → allocate → permute_all → GroupedGEMM → unpermute → combine
```

1. **D2H sync**: `notify_dispatch` 的 expert count 在 GPU 上，GroupedGEMM 需要 CPU 端知道 shape 才能分配 output
2. **Permute scatter**: dispatch 写到 flat `recv_x`，需要独立 scatter kernel 重排到 expert-sorted 布局
3. **无 overlap**: GEMM 必须等整个 dispatch + permute 完成后才能启动
4. **Per-group 方案的问题**: 如果按 expert group 分组做多次 dispatch：
   - 每组需要独立 D2H sync
   - 每组独立分配 worst-case buffer（内存浪费 G 倍）
   - 多组 GroupedGEMM 输出需要 concat kernel

## 2. 设计目标

实现 dispatch → permute → GroupedGEMM 的三级 pipeline overlap：

```
dispatch (通信)   ─────────────────────────►
permute  (本地)     ──────────────────────►
GEMM     (计算)       ───tile0──►──tile1──►──tile2──►
```

核心原则：
- **零 D2H sync**: GEMM 直接从 GPU 内存读 problem sizes
- **零独立 permute kernel**: permute 融合在 dispatch receiver 中
- **零 concat**: 输出直接写到最终位置
- **BLOCK_M 对齐**: 每个 GEMM tile 对应一块连续的 expert input

## 3. 核心设计：Expert-Sorted Dispatch

### 3.1 核心思路

**Sender 按 destination expert 排序发送，Receiver 直接按 BLOCK_M 块写入 expert_input。**

```
原始 DeepEP:
  sender: token 0,1,2,3... (token id 顺序)
  recv_x: [R0_ch0: t3,t7,t12 | R0_ch1: t55,t61 | R1_ch0: t2,t8 | ...]
           → scatter permute (随机写, 无 BLOCK_M 对齐)

Expert-Sorted:
  sender: 按 expert 排序 [exp0: t7,t2,t15... | exp1: t3,t61,t88... | ...]
  expert_input: [exp0: |BLOCK_M tile0|BLOCK_M tile1| exp1: |tile0|tile1|..]
                        ↑ flag=1      ↑ flag=1            ↑ flag=1
                        GEMM 立即可读
```

### 3.2 三级解耦 Pipeline

三个阶段用不同的最优粒度运行，通过 atomic counter 解耦：

```
Dispatch (token-order)        Permute (fused)          GEMM (expert-tile)

Sender channel 0:             Receiver 写入:            Expert 0:
  [tok0,tok1,...tok5] ──ring──► expert_input[e][pos]    [████████|████████|██]
  [tok6,tok7,...tok11]──buf──►  atomicAdd(write_pos[e])   tile 0,1 ready → fire
  ...
                              ──────────────────────
每次推进:                      每次推进:                 每次推进:
  num_max_send_tokens           ~变长 batch              BLOCK_M per expert
  (默认 6, ring buf 流控)       (tail - head 决定)        (128/256)
```

三级之间的接口：

```
                atomic counter 1              atomic counter 2
                ┌───────────────┐             ┌────────────────────┐
  dispatch ────►│ (fused in     │────────────►│ tile_written_count │───► GEMM
  receiver      │  receiver)    │  scatter    │ [E_per_R, max_tile]│    tile
                │               │             │                    │
                └───────────────┘             └────────────────────┘
```

## 4. 预处理：Expert-Sorted Send Schedule

### 4.1 数据结构

```cpp
struct ExpertSortedSchedule {
    // sender 侧: 排序后的发送序列
    int* sorted_send_ids;          // [total_unique_tokens_to_send]
    int* send_expert_boundaries;   // [R × E_per_R + 1]

    // receiver 侧: 每个 src_rank 贡献多少 token 给每个 expert
    int* recv_expert_src_count;    // [E_per_R, R]
    int* recv_expert_src_offset;   // [E_per_R, R] (二维前缀和)

    // tile 信息
    int* tile_ready_flags;         // [E_per_R, max_tiles_per_expert]
    int* tile_written_count;       // [E_per_R, max_tiles_per_expert]
    int  max_tiles_per_expert;
};
```

### 4.2 Kernel 1: 统计 + 分配 Primary Expert

每个 token 到某个 rank 只发送一次（去重）。选该 token 在目标 rank 上的最小 local expert id 作为排序依据。

```cpp
__global__ void count_and_assign_primary(
    const int64_t* topk_idx,        // [N, K]
    const bool*    is_token_in_rank, // [N, R]
    int*           pair_count,       // [R, E_per_R] output
    int*           token_primary,    // [N, R] output
    int N, int K, int num_ranks, int E_per_R
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    for (int r = 0; r < num_ranks; r++) {
        if (!is_token_in_rank[t * num_ranks + r]) {
            token_primary[t * num_ranks + r] = -1;
            continue;
        }
        int min_le = E_per_R;
        for (int k = 0; k < K; k++) {
            int e = topk_idx[t * K + k];
            if (e < 0) continue;
            if (e / E_per_R == r)
                min_le = min(min_le, e % E_per_R);
        }
        token_primary[t * num_ranks + r] = min_le;
        atomicAdd(&pair_count[r * E_per_R + min_le], 1);
    }
}
```

### 4.3 Kernel 2: 前缀和

对 `pair_count[R, E_per_R]` 做 per-dst_rank 的 exclusive prefix sum → `pair_offset[R, E_per_R]`。

### 4.4 Kernel 3: 构建排序序列

```cpp
__global__ void build_sorted_ids(
    const int*  token_primary,      // [N, R]
    const int*  pair_offset,        // [R, E_per_R]
    int*        pair_next,          // [R, E_per_R] atomic counter, init=0
    int*        sorted_send_ids,    // [total_pairs] output
    int N, int num_ranks, int E_per_R
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    for (int r = 0; r < num_ranks; r++) {
        int le = token_primary[t * num_ranks + r];
        if (le < 0) continue;

        int pos = atomicAdd(&pair_next[r * E_per_R + le], 1);
        int base = pair_offset[r * E_per_R + le];
        sorted_send_ids[base + pos] = t;
    }
}
```

优化：在同一 expert 组内按 token_id 升序排列，恢复空间局部性，减少 sender 的 cache miss。

### 4.5 预处理代价

```
Kernel 1 (count_and_assign):  O(N × K),  N=4096, K=8 → 32K ops
Kernel 2 (prefix sum):        O(R × E_per_R) = O(256)
Kernel 3 (build_sorted_ids):  O(N × R) = O(32K)

总计: ~3 个轻量 kernel, <0.1ms (相比 dispatch ~1ms 可忽略)
```

## 5. Modified Dispatch Kernel

### 5.1 Sender 修改

按 `sorted_send_ids` 遍历（而非 token id 顺序）：

```cpp
// channel 分配: 按 sorted_send_ids range 分
int total_pairs = pair_offset[responsible_rank * E_per_R + E_per_R];
get_channel_task_range(total_pairs, num_channels, responsible_channel,
                       pair_start, pair_end);

for (int64_t pair_idx = pair_start; pair_idx < pair_end;) {
    // ring buffer 流控 (同原版)
    while (...) { /* wait for buffer space */ }

    int chunk = 0;
    while (chunk < num_max_send_tokens && pair_idx < pair_end) {
        int token_idx = sorted_send_ids[pair_base + pair_idx];

        // 写入 ring buffer — 非顺序读 x[token_idx], 但每行 14KB 连续
        auto src = x + token_idx * hidden_int4;
        auto dst = channel_x_buffers.buffer() + (tail % cap) * hidden_int4;
        UNROLLED_WARP_COPY(2, lane_id, hidden_int4, dst, src, __ldg, st_na_global);

        // 发送 src_idx (combine 用) + topk metadata (同原版)
        if (lane_id == 0)
            channel_src_idx_buffers[tail % cap] = token_idx;
        // topk_idx/weights 同原版 ...

        chunk++; pair_idx++; tail++;
    }
    st_relaxed_sys_global(channel_tail_idx.buffer(), tail);
}
```

非顺序读代价：每行 14KB 连续（H=7168, bf16），L2 cache line 128B，每行 ~112 次顺序 cache line 读。行间跳转 TLB/prefetch 开销相比行内读取量可忽略。

### 5.2 Receiver 修改 — 直接写入 expert_input + 设置 tile flag

```cpp
// receiver 知道 tokens 按 (expert, src_rank) 排序到达
// 对于从 responsible_rank (= src_rank) 收到的 tokens:

int expert_e = 0;
int pos_in_expert = 0;
int expert_write_base = expert_offsets[0] + recv_expert_src_offset[0][responsible_rank];
int expert_write_limit = recv_expert_src_count[0][responsible_rank];

while (num_tokens_to_recv > 0) {
    // poll tail (同原版)
    ...

    int num_recv = cached_tail - cached_head;
    for (int i = warp_id; i < num_recv; i += num_warps) {
        int slot = (cached_head + i) % cap;
        int write_row = expert_write_base + pos_in_expert + i;

        // 直接写到 expert_input
        auto dst = expert_input + (int64_t)write_row * hidden_int4;
        auto src = channel_x_buffers.buffer() + slot * hidden_int4;
        UNROLLED_WARP_COPY(2, lane_id, hidden_int4, dst, src, ld_nc_global, st_na_global);

        // 检查 tile 完成状态
        int global_row_in_expert = write_row - expert_offsets[expert_e];
        int tile_id = global_row_in_expert / BLOCK_M;
        int tile_size = min(BLOCK_M, expert_count[expert_e] - tile_id * BLOCK_M);

        if (lane_id == 0) {
            int old = atomicAdd(&tile_written_count[expert_e * max_tiles + tile_id], 1);
            if (old + 1 == tile_size) {
                __threadfence();
                st_release_sys_global(&tile_ready[expert_e * max_tiles + tile_id], 1);
            }
        }
    }

    // 推进 expert 边界
    pos_in_expert += num_recv;
    while (pos_in_expert >= expert_write_limit && expert_e < E_per_R - 1) {
        pos_in_expert -= expert_write_limit;
        expert_e++;
        expert_write_base = expert_offsets[expert_e]
                          + recv_expert_src_offset[expert_e][responsible_rank];
        expert_write_limit = recv_expert_src_count[expert_e][responsible_rank];
    }

    // 推进 head (同原版)
    cached_head += num_recv;
    st_relaxed_sys_global(channel_head_idx.buffer(), cached_head);
    num_tokens_to_recv -= num_recv;
}
```

### 5.3 Secondary Expert Copy (~5% tokens)

Token T 在 rank R 上选了 expert e1 和 e2，但只在 e1 组中被发送。需要额外拷贝到 e2 的位置。

概率分析（256 experts, 8 ranks, topk=8）：

```
E[local experts per token per rank] = topk × (E_per_R / E_total) = 8 × (32/256) = 1.0
P(2+ local experts) ≈ 5%
```

在 receiver 写入 primary 位置时，同时检查 secondary expert：

```cpp
for (int k = 0; k < num_topk; k++) {
    int le = channel_topk_idx_buffers[slot * num_topk + k];
    if (le < 0 || le == primary_expert) continue;
    // secondary expert: 拷贝到 le 的位置
    int sec_pos = atomicAdd(&expert_secondary_pos[le], 1);
    int sec_row = expert_offsets[le] + primary_section_size[le] + sec_pos;
    auto sec_dst = expert_input + (int64_t)sec_row * hidden_int4;
    UNROLLED_WARP_COPY(2, lane_id, hidden_int4, sec_dst, src, ld_nc_global, st_na_global);
}
```

## 6. Persistent GroupedGEMM with Tile Flags

### 6.1 Buffer 布局

```
expert_input: [total_permuted_tokens, H_in]

┌────────────────────────────────────────────────────┐
│ expert 0                                           │
│ ┌─────────┬─────────┬─────────┬─────────┐          │
│ │ R0 (40) │ R1 (35) │ R2 (42) │ ...     │ = 300   │
│ └─────────┴─────────┴─────────┴─────────┘          │
│   tile 0    tile 1    tile 2    (BLOCK_M=128)      │
│   ████████  ████████  ██▒▒▒▒▒▒ (tile 2 partial)   │
│                                                     │
│ expert 1                                           │
│ ┌─────────┬─────────┬─────────┬─────────┐          │
│ │ R0 (55) │ R1 (48) │ R2 (30) │ ...     │ = 280   │
│ └─────────┴─────────┴─────────┴─────────┘          │
│   tile 0    tile 1    tile 2                        │
│   ████████  ████████  ██                            │
└────────────────────────────────────────────────────┘

tile_ready:         [e0_t0]=1  [e0_t1]=1  [e0_t2]=0  ← 还在等数据
tile_written_count: [e0_t0]=128 [e0_t1]=128 [e0_t2]=44
```

### 6.2 Persistent GEMM Kernel

```cpp
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void persistent_grouped_gemm(
    const bf16* expert_input,     // [total_permuted, H_in]
    const bf16* expert_weights,   // [E_per_R, H_in, H_out]
    bf16*       output,           // [total_permuted, H_out]
    const int*  expert_count,     // [E_per_R] from notify_dispatch (GPU)
    const int*  expert_offsets,   // [E_per_R] prefix sum
    const int*  tile_ready,       // [E_per_R, max_tiles]
    int H_in, int H_out, int E_per_R, int max_tiles
) {
    // Block-to-(expert, tile_n) mapping
    // 从 GPU 内存读 expert_count, 无 D2H sync
    int expert_id, tile_n_id;
    map_block_to_work(blockIdx.x, expert_count, expert_offsets,
                      H_out, BLOCK_N, &expert_id, &tile_n_id);
    if (expert_id < 0) return;

    int M_e = expert_count[expert_id];
    int num_m_tiles = (M_e + BLOCK_M - 1) / BLOCK_M;
    int e_offset = expert_offsets[expert_id];
    const bf16* B = expert_weights + expert_id * H_in * H_out;

    // 沿 M 维度流式处理
    for (int tm = 0; tm < num_m_tiles; tm++) {
        // 等待 tile 数据就绪
        while (ld_volatile_global(&tile_ready[expert_id * max_tiles + tm]) == 0) {
            /* spin — dispatch 还在写 */
        }
        __threadfence();  // acquire

        int m_start = tm * BLOCK_M;
        int actual_m = min(BLOCK_M, M_e - m_start);

        const bf16* A = expert_input + (int64_t)(e_offset + m_start) * H_in;
        bf16* C = output + (int64_t)(e_offset + m_start) * H_out
                + tile_n_id * BLOCK_N;

        // 标准 GEMM tile — 接入 CK BlockwiseGemm / CUTLASS
        gemm_tile<BLOCK_M, BLOCK_N, BLOCK_K>(A, B, C, actual_m, H_in);
    }
}
```

### 6.3 Block-to-Work Mapping

```cpp
__device__ void map_block_to_work(
    int block_idx,
    const int* expert_count,
    const int* expert_offsets,
    int H_out, int BLOCK_N,
    int* out_expert, int* out_tile_n
) {
    int n_tiles = (H_out + BLOCK_N - 1) / BLOCK_N;
    int tiles_so_far = 0;

    for (int e = 0; e < E_per_R; e++) {
        int m_tiles = (expert_count[e] + BLOCK_M - 1) / BLOCK_M;
        int total_tiles_e = m_tiles * n_tiles;
        if (block_idx < tiles_so_far + total_tiles_e) {
            int local_idx = block_idx - tiles_so_far;
            *out_expert = e;
            *out_tile_n = local_idx % n_tiles;
            return;
        }
        tiles_so_far += total_tiles_e;
    }
    *out_expert = -1;  // 超出范围
}
```

### 6.4 CK 集成

```cpp
// 用 CK 的 block-level abstraction 作为 GEMM tile 实现:
using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;

// CK BlockGemm with dynamic M
using BlockGemm = ck::BlockwiseGemmXdlops<
    bf16, bf16, float, bf16,
    ALayout, BLayout,
    BLOCK_M, BLOCK_N, BLOCK_K,
    MPerXDL, NPerXDL, MXdlPerWave, NXdlPerWave,
    NumPrefetch, LoopScheduler, PipelineVersion
>;
```

## 7. Alternative: Indirection Layout (跳过物化 permute)

如果不想改 sender 排序，可以保持原始 DeepEP dispatch，GEMM 通过 indirection 直接读 flat `recv_x`：

```cpp
// GEMM 的 A 矩阵通过 row_map 间接访问 recv_x
// row_map[i] = recv_x 中第 i 个 permuted token 的行号
// dispatch receiver 只写 row_map (4B/token), 不拷贝 data

struct GatheredLayout {
    const bf16* recv_x;
    const int*  row_map;
    int         hidden;
    int         expert_offset;

    __device__ const bf16* get_row(int local_pos) {
        return recv_x + (int64_t)row_map[expert_offset + local_pos] * hidden;
    }
};
```

对比：

```
                      Expert-Sorted Dispatch    Indirection Layout
──────────────────────────────────────────────────────────────────
额外内存              tile flags (~4KB)          row_map (~128KB)
Permute 带宽          0 (fused in receiver)     0 (只写 row_map)
Sender 改动          需要排序遍历               无改动
GEMM A-load          100% coalesced             行间 gather, 行内 coalesced
                                                ~5-15% 额外 latency
实现复杂度           中等                       低
```

## 8. Dispatch 完成信号

Persistent GEMM 的最后一个 partial tile 需要知道"不会有更多 token 到达"：

```cpp
// Dispatch 结束后 (所有 sender/receiver 退出循环)
if (sm_id == 0 && thread_id == 0)
    st_release_sys_global(&dispatch_done, 1);
```

Persistent GEMM 对 partial tile 的处理：
```cpp
// 最后一个 tile: 等 dispatch_done + tile_written_count 达到预期
if (tm == num_m_tiles - 1) {
    int expected = M_e - tm * BLOCK_M;
    while (ld_volatile_global(&tile_written_count[...]) < expected
           && ld_volatile_global(&dispatch_done) == 0) { /* spin */ }
}
```

## 9. SM 分配

MI300X (CDNA3, 304 CU):

```
Dispatch:   20 CU (10 sender + 10 receiver), Config 可调
GEMM:       284 CU

GEMM tile:  BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 (CK default for bf16)

每个 expert ~300 tokens (avg), 32 experts/rank:
  → 128 tile: 2-3 tiles/expert × 32 experts = 64-96 tiles
  → 284 CU 处理 64-96 tiles: 充分并行
```

## 10. Tile Size 建议

```
BLOCK_M=128 (推荐):
  - tile 数更多 → 更好利用 284 CU
  - pipeline 粒度更细 → 更快启动第一个 tile
  - partial tile 平均浪费 64 tokens (vs BLOCK_M=256 浪费 128)

BLOCK_N=128:
  - H_out=7168 → 56 个 N tiles
  - 足够并行度

BLOCK_K=64:
  - 标准 CK/CUTLASS bf16 配置
  - 匹配 CDNA3 MFMA 指令
```

## 11. 总结对比

```
                  原始 DeepEP              Expert-Sorted Pipeline
──────────────────────────────────────────────────────────────────
预处理             无                       3 个轻量 kernel (~0.1ms)
Sender 访问       顺序读 x[0..N]           非顺序读 x[sorted[i]]
                                           (行内仍 coalesced)
通信量            不变                     不变 (per-rank 去重)
Receiver 写       flat recv_x (顺序)       expert_input (per-expert 顺序)
Permute           独立 scatter kernel      无需 — fused in receiver
D2H sync          需要 (拿 expert count)   无需 — GEMM 读 GPU 内存
BLOCK_M 对齐      无                       天然对齐 (同 expert 连续写)
GEMM 启动         等全部完成               tile_ready poll, 流式启动
输出 Concat       可能需要                 无需 — 直接写最终位置
Secondary expert  不需要                   ~5% token 额外 copy
```

## 12. 实现路径

### Phase 1: GPU-driven GroupedGEMM (无 pipeline, 但无 D2H sync)
- 保持原始 dispatch
- GEMM 从 GPU 读 expert_count
- 验证正确性

### Phase 2: Fused Dispatch-Permute (消除 permute pass)
- Receiver 做 fused scatter 到 expert_input
- 用 atomic expert_write_pos
- tile_written_count 追踪完成状态

### Phase 3: Expert-Sorted Dispatch (BLOCK_M 对齐)
- 添加预处理 kernel
- Sender 按 sorted_send_ids 遍历
- Receiver 顺序写 + tile flags

### Phase 4: Persistent GEMM (完整 pipeline)
- Dispatch 和 GEMM 并发 launch
- GEMM poll tile_ready
- 验证 overlap 效率

### Phase 5: 扩展到完整 MoE FFN
- W1 persistent GEMM → activation → W2 persistent GEMM
- dispatch → W1_tile → act → W2_tile 的 fine-grained pipeline
- Combine 的反向 pipeline
