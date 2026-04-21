# 直接计算 dispatched_token_idx → permuted_token_idx 映射：设计方案

## 1. 核心问题分析

当前瓶颈在于 `row_id_map` 的计算依赖 `global_routing_map`（需 allgather）。`row_id_map[d, e]` 表示：在 receiver 的 dense buffer 中位置 `d` 的 token，对应 local expert `e` 在 permuted output 中的目标位置。

计算该映射需要两个信息：

1. **每个 dispatched token 路由到哪些 local expert** — 只有 sender 知道（来自 topk_idx）
2. **跨所有 source rank 的 per-expert 全局 prefix sum** — 需要跨 rank 交换

方案的核心思想：**notify_dispatch 交换 per-expert counts 计算全局 prefix，sender 在 dispatch 时利用本地 topk_idx + 全局 prefix 直接算出 permuted_idx 并写入对称内存**，receiver 直接读取——等价于 sender 分布式填充 row_id_map。

---

## 2. 符号定义

| 符号 | 含义 | 典型值 |
|------|------|--------|
| N | 每个 rank 的 token 数 | 4096 |
| K | topk（每 token 选择的 expert 数） | 8 |
| E | 总 expert 数 | 256 |
| R | rank 数（GPU 数，单节点） | 8 |
| E_r = E/R | 每个 rank 的 local expert 数 | 32 |
| C | channel 数（= num_sms / 2） | 64 |
| H | hidden size (bytes per token) | 7168 (bf16, dim=3584) |
| P | expert padding alignment | 128 |
| D | 每 rank 接收的 dispatched token 数 | ≈ N·K |

---

## 3. 整体流程

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: get_dispatch_layout (不变，纯本地计算)          │
│  topk_idx [N,K] → is_token_in_rank, counts              │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2: notify_dispatch_v2 (增强)                      │
│  SM 0:  交换 counts → rank_prefix_matrix                │
│         + expert_rank_prefix [R, E_r]  ← 新增           │
│         + expert_offset [E_r]          ← 新增           │
│  SM 1..R: scan is_token_in_rank → channel_prefix_matrix │
│         + scan topk_idx → channel_expert_prefix [R,C,E_r] ← 新增 │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3: fused_dispatch_permute_v2 (增强)               │
│  Sender: 利用 expert_rank_prefix + channel_expert_prefix │
│          + running_count 实时计算 permuted_idx            │
│          写入对称内存 metadata region (row_id_map 格式)    │
│  Receiver: 读取 metadata → 直接 permute (逻辑不变)       │
│  Side-effect: 输出完整 row_id_map 用于 combine           │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 新增数据结构

```cpp
// ===== notify_dispatch_v2 新增输出 =====

// Per-source-rank per-expert 全局 exclusive prefix sum
// expert_rank_prefix[s][e] = receiver 收到的、来自 rank 0..s-1 的 expert e 的 token 总数
// 用途：确定每个 source rank 的 token 在 permuted output 中每个 expert section 的起始偏移
torch::Tensor expert_rank_prefix;   // [R, E_r] int32

// 每个 expert section 在 permuted output 中的起始偏移 (padded cumulative prefix)
// expert_offset[e] = sum_{j<e} ceil(total_expert_count[j] / P) * P
// expert_offset[E_r] = num_permuted_tokens
torch::Tensor expert_offset;        // [E_r + 1] int32

// Per-dest-rank per-channel per-expert exclusive prefix sum (本 rank 发送侧)
// channel_expert_prefix[r][c][e] = 本 rank 在 channel 0..c-1 中发送到 rank r 的 expert e 的 token 数
// 用途：sender 在 dispatch 时确定 within-rank expert offset
torch::Tensor channel_expert_prefix; // [R, C, E_r] int32
```

### Permuted index 计算公式

Sender 发送 token `t` 到 dest_rank `r`、channel `c`，该 token 路由到 local expert `e`：

```
permuted_idx = expert_offset[e]
             + expert_rank_prefix[my_rank][e]
             + channel_expert_prefix[r][c][e]
             + within_channel_running_count[e]
```

其中 `within_channel_running_count[e]` 由 sender 在 dispatch 循环中维护（寄存器/shared memory）。

---

## 5. Phase 1：get_dispatch_layout（不变）

从 topk_idx 统计每个 rank / expert 的 token 数，同时产出 per-token per-rank 的 bool 映射。

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| HBM Read | topk_idx [N, K] int64 | N·K·8 | 256 KB |
| HBM Write | is_token_in_rank [N, R] bool | N·R | 32 KB |
| HBM Write | num_tokens_per_rank [R] int32 | R·4 | 32 B |
| HBM Write | num_tokens_per_expert [E] int32 | E·4 | 1 KB |

不做任何改动。

---

## 6. Phase 2：notify_dispatch_v2（增强）

### 6.1 SM 0 变更：增加 expert_rank_prefix + expert_offset

现有 SM 0 已经交换了 `per_expert_buffer[R, E_r]`（每个 source rank 发送到本 rank 各 local expert 的 token 数）。在现有 barrier 后的计算阶段追加：

```cpp
// ---- 现有逻辑: 交换 counts + barrier + 读回 ----
// local_per_expert_buffer[s * E_r + e] = rank s 发给本 rank 的 expert e 的 token 数

// ---- 新增 1: expert_rank_prefix (exclusive prefix across source ranks) ----
if (thread_id < num_experts_per_rank) {
    int prefix = 0;
    for (int s = 0; s < kNumRanks; ++s) {
        expert_rank_prefix[s * num_experts_per_rank + thread_id] = prefix;
        prefix += local_per_expert_buffer[s * num_experts_per_rank + thread_id];
    }
    // prefix == total_per_expert[thread_id] (未 pad)
    smem_unpadded_expert_total[thread_id] = prefix;
}
__syncthreads();

// ---- 新增 2: expert_offset (padded cumulative prefix) ----
// E_r 很小 (≤128)，单线程 sequential scan 即可
if (thread_id == 0) {
    int offset = 0;
    for (int e = 0; e < num_experts_per_rank; ++e) {
        expert_offset_out[e] = offset;
        int padded = (smem_unpadded_expert_total[e] + expert_alignment - 1)
                   / expert_alignment * expert_alignment;
        offset += padded;
    }
    expert_offset_out[num_experts_per_rank] = offset; // = num_permuted_tokens
}

// ---- 现有: 写回 moe_recv_expert_counter (padded) ----
// 保持不变
```

**额外 NVLink**: 0（counts 已在现有流程中交换，新增计算纯本地）

**额外 HBM Write**: `R·E_r·4 + (E_r+1)·4` ≈ 1.1 KB

### 6.2 SM 1..R 变更：增加 channel_expert_prefix

每个 SM 已对一个 dest_rank 扫描 `is_token_in_rank` 得到 `channel_prefix_matrix`。在同一循环中增加对 `topk_idx` 的扫描：

```cpp
int dst_rank = sm_id - 1;
for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, channel_id,
                           token_start_idx, token_end_idx);

    int rank_count = 0;
    int expert_count[NUM_EXPERTS_PER_RANK];  // 寄存器, E_r=32 → 128B
    for (int e = 0; e < NUM_EXPERTS_PER_RANK; ++e) expert_count[e] = 0;

    for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += WARP_SIZE) {
        bool in_rank = is_token_in_rank[i * kNumRanks + dst_rank];
        rank_count += in_rank;

        // 新增: 如果 token 发往 dst_rank, 统计命中的 local expert
        if (in_rank) {
            auto shifted_topk = topk_idx + i * num_topk;
            for (int k = 0; k < num_topk; ++k) {
                int e = static_cast<int>(shifted_topk[k]);
                if (e / num_experts_per_rank == dst_rank)
                    expert_count[e % num_experts_per_rank]++;
            }
        }
    }

    // Warp reduce: rank count (现有)
    rank_count = warp_reduce_sum(rank_count);
    if (lane_id == 0)
        channel_prefix_matrix[dst_rank * num_channels + channel_id] = rank_count;

    // Warp reduce: expert counts (新增)
    for (int e = 0; e < num_experts_per_rank; ++e) {
        int sum_e = warp_reduce_sum(expert_count[e]);
        if (lane_id == 0)
            channel_expert_count_buf[dst_rank * num_channels * E_r
                                   + channel_id * E_r + e] = sum_e;
    }
}
__syncthreads();

// Sequential prefix sum (thread 0 of each block)
if (thread_id == 0) {
    // channel_prefix_matrix (现有)
    for (int i = 1; i < num_channels; ++i)
        channel_prefix_matrix[dst_rank * num_channels + i] +=
            channel_prefix_matrix[dst_rank * num_channels + i - 1];

    // channel_expert_prefix (新增: exclusive prefix across channels)
    for (int e = 0; e < num_experts_per_rank; ++e) {
        int prefix = 0;
        for (int c = 0; c < num_channels; ++c) {
            int idx = dst_rank * num_channels * E_r + c * E_r + e;
            int count = channel_expert_count_buf[idx];
            channel_expert_prefix[idx] = prefix;
            prefix += count;
        }
    }
}
```

**额外 HBM Read**: `topk_idx [N, K]` int64 — 256 KB。在 get_dispatch_layout 中刚读取过，L2 cache hit 率极高，几乎不产生额外 HBM 流量。

**额外 HBM Write**: `channel_expert_prefix [R, C, E_r]` int32 — 64 KB

**寄存器压力**: 每线程 E_r 个 int32 counter = 128B (E_r=32)，可控。

### 6.3 notify_dispatch_v2 函数签名

```cpp
void notify_dispatch_v2(
    const int *num_tokens_per_rank, int *moe_recv_counter,
    const int *num_tokens_per_expert, int *moe_recv_expert_counter,
    int num_experts, int num_tokens, int num_channels,
    const bool *is_token_in_rank,
    const int64_t *topk_idx, int num_topk,            // 新增: topk_idx 输入
    int *channel_prefix_matrix, int *rank_prefix_matrix_copy,
    int *expert_rank_prefix,                            // 新增输出: [R, E_r]
    int *expert_offset,                                 // 新增输出: [E_r + 1]
    int *channel_expert_prefix,                         // 新增输出: [R, C, E_r]
    int num_memset_int, int expert_alignment,
    void **buffer_ptrs, int **barrier_signal_ptrs,
    int rank, cudaStream_t stream);
```

---

## 7. Phase 3：fused_dispatch_permute_v2（增强）

### 7.1 对称内存布局增强

在现有 Buffer 布局之后，增加 row_id_map metadata region：

```
symmetric_buffer_per_rank (每个 rank 的对称内存区域):
  [0]                          rank_prefix_matrix [R*R] int32
  [R*R*4]                      channel_comm_queues (start/end/tail per channel*rank)
  [...]                        token_data [D_max * hidden_int4] int4
  [token_data_end]             row_id_map_region [D_max * E_r] int32   ← 新增
```

`row_id_map_region` 大小：`D_max * E_r * 4`。典型值 D_max=32768, E_r=32 → 4 MB。

### 7.2 Sender 侧变更

Sender 在写入 token 数据时，同时计算并写入 row_id_map。核心逻辑：

```cpp
// ---- Sender 初始化: per-expert running counter ----
// shared memory, 每个 responsible_rank 独立维护
__shared__ int expert_running_count[NUM_EXPERTS_PER_RANK];

// 初始化为 channel_expert_prefix[responsible_rank][responsible_channel][e]
for (int e = send_warp_id_in_rank * WARP_SIZE + lane_id;
     e < num_experts_per_rank; e += num_send_warps_per_rank * WARP_SIZE) {
    expert_running_count[e] =
        channel_expert_prefix[responsible_rank * num_channels * E_r
                            + responsible_channel * E_r + e];
}
__syncthreads();

// ---- 在现有 token dispatch 循环中新增 ----
while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
    if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
        token_idx++; continue;
    }

    auto dst_slot_idx = total_offset + cached_channel_tail_idx++;

    if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
        // 1. 写入 token 数据 (现有逻辑, 不变)
        auto shifted_channel_x_buffers =
            channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
        auto shifted_x = x + token_idx * hidden_int4;
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
                           shifted_channel_x_buffers, shifted_x,
                           __ldg, st_na_global);

        // 2. 计算并写入 row_id_map entries (新增)
        //    row_id_map_region 位于对称内存中
        auto shifted_row_id_map =
            row_id_map_region + dst_slot_idx * num_experts_per_rank;

        // 先清零（大部分 expert 不被命中）
        if (lane_id < num_experts_per_rank)
            st_na_global(shifted_row_id_map + lane_id, 0);

        // 只有 lane 0 读 topk_idx 并计算 permuted_idx
        if (lane_id == 0) {
            auto shifted_topk = topk_idx + token_idx * num_topk;
            for (int k = 0; k < num_topk; ++k) {
                int e = static_cast<int>(shifted_topk[k]);
                if (e / num_experts_per_rank == responsible_rank) {
                    int le = e % num_experts_per_rank;
                    int pidx = expert_offset[le]
                             + expert_rank_prefix[rank * num_experts_per_rank + le]
                             + expert_running_count[le];
                    expert_running_count[le]++;
                    // 1-indexed, 与现有 row_id_map convention 一致
                    st_na_global(shifted_row_id_map + le, pidx + 1);
                }
            }
        }
    }

    chunk_token_idx++;
    token_idx++;
}

// tail index 的 st_release 保证 row_id_map 写入对 receiver 可见
st_release_sys_global<true>(channel_tail_idx.buffer(), cached_channel_tail_idx);
```

### 7.3 Receiver 侧变更

**几乎不变**。Receiver 原来读 `row_id_map[src_slot_idx * E_r + e]`，现在从对称内存的 `row_id_map_region` 读取，逻辑完全一致：

```cpp
// 现有代码 (receiver 循环内):
for (int e = 0; e < num_experts_per_rank; ++e) {
    // row_id_map 指针指向对称内存中的 row_id_map_region
    int target = row_id_map_region[src_slot_idx * num_experts_per_rank + e];
    if (target > 0) {
        auto shifted_recv_x_int4 =
            recv_x + static_cast<int64_t>(target - 1) * hidden_int4;
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
                           shifted_recv_x_int4,
                           shifted_buffer_x_int4,
                           ld_nc_global, st_na_global);
    }
}
```

唯一变更：`row_id_map` 指针指向对称内存中的 metadata region，而非独立 tensor。

### 7.4 row_id_map 持久化（用于 combine/unpermute）

Dispatch 完成后，row_id_map 已完整存在于每个 rank 的对称内存中。Combine 使用方式：

- **方案 a**: combine 直接从对称内存读取（如果对称内存在 MoE 计算期间不被覆盖）
- **方案 b**: dispatch 结束后 DtoD copy 到独立 tensor（异步，与 expert compute overlap）

### 7.5 fused_dispatch_permute_v2 函数签名

```cpp
void fused_dispatch_permute_v2(
    void **workspace_ptrs, void const *x, float const *x_scales,
    int64_t const *topk_idx, float const *topk_weights,
    bool const *is_token_in_rank, int const *channel_prefix_matrix,
    int const *expert_rank_prefix,                      // 新增输入
    int const *expert_offset,                           // 新增输入
    int const *channel_expert_prefix,                   // 新增输入
    int *row_id_map,                                    // 输出: sender 填充
    void *recv_x,
    int num_tokens, int hidden_int4, int num_topk, int num_experts,
    int num_scales, int scale_token_stride, int scale_hidden_stride,
    int rank, int num_ranks, cudaStream_t stream,
    int num_sms, int num_max_tokens, int num_max_send_tokens);
```

---

## 8. Python 侧更新

### 8.1 DispatchHandle

```python
class DispatchHandle(NamedTuple):
    local_expert_routing_map: torch.Tensor
    num_of_tokens_per_rank: int = 0
    num_permuted_tokens: int = 0
    row_id_map: Optional[torch.Tensor] = None
    expert_rank_prefix: Optional[torch.Tensor] = None      # 新增
    expert_offset: Optional[torch.Tensor] = None            # 新增
    channel_expert_prefix: Optional[torch.Tensor] = None    # 新增
```

### 8.2 调用流程变更

```python
def _fused_dispatch_permute_v2(x, group, topk_idx, topk_weights,
                                num_experts, expert_alignment, ...):
    # Phase 1: get_dispatch_layout (不变)
    is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert = \
        cpp_extensions.get_dispatch_layout(topk_idx, num_ranks, num_experts)

    # Phase 2: notify_dispatch_v2 (增强)
    (moe_recv_counter, moe_recv_expert_counter,
     rank_prefix_matrix, channel_prefix_matrix,
     expert_rank_prefix, expert_offset,             # 新增
     channel_expert_prefix) = \                     # 新增
        cpp_extensions.intranode_notify_dispatch_v2(
            num_tokens_per_rank, num_tokens_per_expert,
            is_token_in_rank, topk_idx,             # 新增: topk_idx 输入
            buffer_ptrs_dev, barrier_signal_ptrs_dev,
            expert_alignment, rank, num_ranks, num_sms)

    # Phase 3: fused_dispatch_permute_v2 (增强)
    permuted_x, row_id_map = cpp_extensions.intranode_dispatch_permute_v2(
        x, x_scales, topk_weights,
        topk_idx,                                   # sender 需要读取
        is_token_in_rank, channel_prefix_matrix,
        expert_rank_prefix, expert_offset,          # 新增
        channel_expert_prefix,                      # 新增
        buffer_ptrs_dev, num_worst_tokens, num_permuted_tokens,
        rank, num_ranks, num_sms, num_max_send_tokens)

    # 无 allgather, 无 global_routing_map, 无 scan kernel
    return permuted_x, row_id_map, handle
```

---

## 9. 性能与开销总结

| 指标 | 现有方案 (allgather routing_map) | 新方案 (v2) |
|------|------|------|
| NVLink metadata 通信 | 7 MB (allgather) | 1.3 KB (对称内存) |
| HBM Read (metadata) | 8 MB (scan 读 global_routing_map) | 256 KB (topk_idx, L2 cached) |
| HBM Write (metadata) | 12 MB | ~100 KB |
| 额外显存 | 8 MB (global_routing_map) | 0 |
| Dispatch 通信增幅 | 0 | +1.8% (row_id_map entries in symm mem) |
| Kernel 数 | 2 (allgather + scan) | 2 (get_layout + notify_v2) |
| Dispatch kernel 改动 | 无 | Sender 新增 ~15 行 |
| Receiver 逻辑变化 | 无 | **无**（读 row_id_map 逻辑不变） |

---

## 10. 实施路径建议

| 步骤 | 内容 | 预计工作量 |
|------|------|---------|
| 1 | `notify_dispatch_v2` SM 0：增加 expert_rank_prefix + expert_offset 计算 | 0.5 天 |
| 2 | `notify_dispatch_v2` SM 1..R：增加 channel_expert_prefix 扫描 | 1 天 |
| 3 | 对称内存布局调整，增加 row_id_map metadata region | 0.5 天 |
| 4 | `fused_dispatch_permute_v2` Sender 侧：新增 row_id_map 计算+写入 | 1 天 |
| 5 | Python binding + DispatchHandle 更新 | 0.5 天 |
| 6 | 正确性测试：row_id_map 与 intranode_prepare_scan 输出对齐 | 1 天 |
| 7 | 端到端性能测试 | 0.5 天 |

---

## 11. 设计核心优势与代价

### 核心优势

1. **Receiver 侧零改动** — `row_id_map` 语义和格式完全保持，sender 分布式填充它
2. **无 allgather** — metadata 通信从 7 MB 降到 1.3 KB
3. **无额外 kernel** — 所有新增计算融合在现有 kernel 中
4. **低耦合** — `notify_dispatch_v2` 的新输出是独立 tensor，不影响现有 dispatch 路径；可以通过 flag 切换 v1/v2

### 代价

1. **Dispatch 通信微增 ~1.8%**：sender 附带 `E_r × 4` bytes row_id_map entries per token。相对 token 正文数据量 H bytes/token 可忽略。
2. **Dispatch kernel 复杂度增加**：sender 需在寄存器中计算 expert 目标偏移（topk_idx + expert_rank_prefix + channel_expert_prefix + running_count），不增加 memory-bound 压力。
3. **对称内存占用增加 4 MB**：row_id_map_region 占 D_max × E_r × 4 bytes。相对已有 >1 GB 的 workspace 可忽略。
4. **前置依赖 topk_idx**：要求 gating 层产出 topk_idx。绝大多数 MoE 实现原生产出 topk_idx，此项不构成实际限制。
