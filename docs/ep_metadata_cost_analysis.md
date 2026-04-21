# EP Metadata 准备阶段：访存量与通信量对比分析

## 1. 符号定义

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
| B | scan kernel 的 block 数 | 108 (SM 数) |
| D | 每 rank 接收的 dispatched token 数 | ≈ N·K |

> **D 的推导**：每个 token 选 K 个 expert，均匀分布在 R 个 rank 上。Token 被 dispatch 到 rank r 的概率 = 1-(1-1/R)^K。对所有 N·R 个全局 token 求和：D = N·R·\[1-(1-1/R)^K\]。当 K=8, R=8 时，D ≈ 5.25N ≈ N·K。

---

## 2. 方案 A：HybridEP（allgather routing_map + scan kernel）

### 2.1 完整流程

```
routing_map [N, E] (gating 层直接产出)
       ──allgather──> global_routing_map [NR, E]
       ──scan kernel──> sparse_to_dense_map, dense_to_expert_map, dense_chunk_layout, ...
```

> HybridEP 的 gating 层直接产出 routing_map（bool 矩阵），无需从 topk_idx 转换。

### 2.2 Allgather routing_map

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| NVLink 发送 | 本 rank 的 routing_map | N·E | 1 MB |
| NVLink 接收 | 其他 R-1 个 rank 的 routing_map | N·(R-1)·E | 7 MB |
| **NVLink 总通信量/rank** | | **N·E·(R-1)** | **7 MB** |
| HBM Write | global_routing_map | N·R·E | 8 MB |

> NCCL ring allgather 需要 R-1 步，每步传输 N·E bytes。Custom allgather（如 DeepEP 的 NVLink multicast 方案）可降低步数，但总数据搬运量不变。

### 2.3 scan kernel (metadata_preprocess)

scan kernel 是一个 decoupled look-back scan，输入 global_routing_map，输出全部 dispatch/permute 所需元数据。

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| HBM Read | global_routing_map (每 token 读 E bytes) | N·R·E | 8 MB |
| HBM Read | scan workspace (look-back polling) | B·(R+E_r)·8 | ~35 KB |
| HBM Write | sparse_to_dense_map [N, R] int32 | N·R·4 | 128 KB |
| HBM Write | dense_to_expert_map [D, E_r] int32 | D·E_r·4 | ~4 MB |
| HBM Write | dense_chunk_layout [⌈N/chunk⌉·R] int32 | ⌈N/chunk⌉·R·4 | ~2 KB |
| HBM Write | scan workspace | B·(R+E_r)·8 | ~35 KB |
| HBM Write | num_of_local_experts_tokens [E_r] int32 | E_r·4 | 128 B |

### 2.4 方案 A 汇总

| 指标 | 公式 | 典型值 (N=4096, K=8, E=256, R=8) |
|------|------|------|
| **NVLink 跨 rank 通信量** | **N·E·(R-1)** | **7 MB** |
| **HBM 总读取** | N·R·E (scan 读 global_routing_map) | **8 MB** |
| **HBM 总写入** | N·R·E (allgather 输出) + D·E_r·4 + N·R·4 | **≈ 12.1 MB** |
| **通信机制** | NCCL / custom allgather | 高延迟 (10–50 μs) |
| **Kernel 启动数** | 1 allgather call + 1 scan kernel | 2 |
| **额外显存** | global_routing_map [NR, E] | **8 MB** |

---

## 3. 方案 B：get_dispatch_layout + notify_dispatch（无 allgather）

### 3.1 完整流程

```
topk_idx [N,K] ──get_dispatch_layout──> is_token_in_rank [N,R], counts
               ──notify_dispatch──> rank_prefix, expert_rank_prefix, channel_prefix, expert_channel_prefix
               ──dispatch kernel──> 边通信边 permute（sender 附带 expert routing metadata）
```

> 核心思路：
> 1. `get_dispatch_layout` 从 topk_idx 解码出本 rank 的统计信息（纯本地计算，无通信）
> 2. `notify_dispatch` 通过 NVLink 对称内存只交换 counts（字节级），计算所有 prefix
> 3. dispatch 时 sender 已知 topk_idx + 全局 expert prefix，附带 expert 路由 metadata 发送

### 3.2 Kernel 1：get_dispatch_layout

从 topk_idx 统计每个 rank / expert 的 token 数，同时产出 per-token per-rank 的 bool 映射。

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| HBM Read | topk_idx [N, K] int64 | N·K·8 | 256 KB |
| HBM Write | is_token_in_rank [N, R] bool | N·R | 32 KB |
| HBM Write | num_tokens_per_rank [R] int32 | R·4 | 32 B |
| HBM Write | num_tokens_per_expert [E] int32 | E·4 | 1 KB |
| **HBM 小计** | | **读 N·K·8，写 N·R + E·4** | **读 256 KB，写 33 KB** |

### 3.3 Kernel 2：notify_dispatch（增强版）

#### 3.3.1 SM 0：跨 rank 元数据交换

通过 NVLink 对称内存直写 + barrier 交换 per-rank 和 per-expert counts。

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| NVLink Write | rank counts 写入每个 peer 的对称内存 | R 个 int32 × R peers = R²·4 | 256 B |
| NVLink Write | expert counts 写入每个 peer 的对称内存 | E_r 个 int32 × R peers = R·E_r·4 | 1 KB |
| **NVLink 总写入/rank** | | **R·(R + E_r)·4** | **1.28 KB** |
| NVLink Read | barrier 后读回所有 rank 写入的 counts | R·(R + E_r)·4 | 1.28 KB |
| NVLink Barrier | 3 次 barrier（start / after-write / end） | 3 × ~0.3 μs | ~1 μs |

Barrier 后 SM0 在寄存器/shared memory 中完成：
- `rank_prefix_matrix[R, R]`：per-rank cumsum → 得到每个 source rank 在 receiver 的 dense buffer 中的起始偏移
- `expert_rank_prefix[R, E_r]`：per-source-rank per-expert cumsum → 得到每个 source rank 的每个 expert 在 permuted output 中的起始偏移

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| HBM Write | rank_prefix_matrix [R, R] int32 | R²·4 | 256 B |
| HBM Write | expert_rank_prefix [R, E_r] int32（新增） | R·E_r·4 | 1 KB |
| HBM Write | moe_recv_counter + moe_recv_expert_counter | (1 + E_r)·4 | 132 B |

#### 3.3.2 SM 1..R：channel prefix scan

每个 SM 负责一个 dest_rank，扫描 is_token_in_rank（和可选的 topk_idx）计算 per-channel 的 prefix。

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| HBM Read | is_token_in_rank [N, R] bool | N·R (total across R SMs, L2 cached) | 32 KB |
| HBM Read | topk_idx [N, K] int64（新增，用于 per-channel expert counts） | N·K·8 (L2 hot from kernel 1) | 256 KB (L2) |
| HBM Write | channel_prefix_matrix [R, C] int32 | R·C·4 | 2 KB |
| HBM Write | expert_channel_prefix [R, C, E_r] int32（新增） | R·C·E_r·4 | 64 KB |

> topk_idx 在 get_dispatch_layout 中刚被读取，极大概率驻留在 L2 cache 中（256 KB << 典型 L2 容量 256 MB），不产生额外 HBM 读取。

### 3.4 Dispatch 阶段附带 Expert 路由信息

Sender 在 dispatch 每个 token 时已知完整信息：
- 该 token 的 topk_idx（哪些 expert，本地寄存器中已有）
- expert_rank_prefix（SM0 计算，标量偏移）
- expert_channel_prefix（SM 1..R 计算，标量偏移）

因此 sender 可在 dispatch 时直接附带 expert routing metadata（bool mask 或预计算的 expert offset），receiver 据此完成 permute。

| 类型 | 数据 | 公式 | 典型值 |
|------|------|------|--------|
| 额外 NVLink per token | expert bool mask | E_r·1 bytes | 32 B |
| **总额外 dispatch 通信** | 所有 dispatched tokens | **D_send · E_r** | **≈ 1 MB** |
| dispatch 正文通信 | token hidden data | D_send · H | ≈ 229 MB |
| **dispatch 通信增幅** | | **E_r / H** | **+0.4%** |

> 其中 D_send ≈ N·K 为本 rank 发送的 dispatch token 总数（去重后 ≈ N·(R-1)·(1-(1-1/R)^K)/R，量级相近）。

### 3.5 方案 B 汇总

| 指标 | 公式 | 典型值 |
|------|------|------|
| **NVLink 跨 rank 通信量（metadata 阶段）** | **R·(R+E_r)·4 + 3×barrier** | **1.3 KB + ~1 μs** |
| **NVLink 额外 dispatch 开销** | D_send·E_r | ~1 MB（占 dispatch 0.4%） |
| **HBM 总读取** | N·K·8 + N·R (L2 cache hot) | **256 KB** (+32 KB L2) |
| **HBM 总写入** | N·R + E·4 + R²·4 + R·E_r·4 + R·C·4 + R·C·E_r·4 | **≈ 100 KB** |
| **通信机制** | NVLink 对称内存直写 + barrier | **超低延迟 (~1 μs)** |
| **Kernel 启动数** | 2（get_dispatch_layout + notify_dispatch） | 2 |
| **额外显存** | 无 | **0** |

---

## 4. 两方案对比总表

### 4.1 NVLink 跨 rank 通信量

| | 方案 A (HybridEP) | 方案 B (notify_dispatch) | B/A 比值 |
|---|---|---|---|
| **公式** | N·E·(R-1) | R·(R+E_r)·4 | |
| **典型值** | **7,340,032 B (7 MB)** | **1,280 B (1.3 KB)** | **1 / 5,734** |
| **通信机制** | NCCL allgather | 对称内存 + barrier | |
| **通信延迟** | 10–50 μs | ~1 μs | **1/10 ~ 1/50** |

> 通信量从 O(N·E·R) 降至 O(R²+R·E_r)。N 和 E 从公式中完全消失——方案 B 的 metadata 通信量与 token 数和 expert 数无关，仅取决于 rank 拓扑。

### 4.2 HBM 访存量

| | 方案 A (HybridEP) | 方案 B (notify_dispatch) | B/A 比值 |
|---|---|---|---|
| **HBM Read 公式** | N·R·E | N·K·8 | |
| **HBM Read 典型值** | **8 MB** | **256 KB** | **1/32** |
| **降低因子** | | R·E / (K·8) = 8·256/64 = **32×** | |
| **HBM Write 公式** | N·R·E + D·E_r·4 + N·R·4 | N·R + R·C·E_r·4 + O(small) | |
| **HBM Write 典型值** | **≈ 12.1 MB** | **≈ 100 KB** | **1/121** |

> 方案 A 的 HBM 访存被两个大 buffer 支配：allgather 写 global_routing_map (N·R·E) + scan 读 global_routing_map (N·R·E)。方案 B 完全消除了这两个操作。

### 4.3 显存占用

| | 方案 A | 方案 B | B/A |
|---|---|---|---|
| 临时 buffer | global_routing_map [NR, E] | 无 | — |
| **典型值** | **8 MB** | **0** | **0** |
| scan workspace | B·(R+E_r)·8 ≈ 35 KB | 无 | — |

### 4.4 Kernel 启动与延迟

| | 方案 A | 方案 B |
|---|---|---|
| 启动序列 | 1 allgather call + 1 scan kernel | 1 get_layout kernel + 1 notify kernel |
| 含 NCCL 调用 | **是**（allgather） | **否** |
| 需要 host 同步 | allgather 内部需同步 | **不需要** |
| 估计 metadata 总延迟 | **20–60 μs** | **3–5 μs** |

### 4.5 Dispatch 阶段开销

| | 方案 A | 方案 B |
|---|---|---|
| Dispatch 正文通信 | D·H ≈ 229 MB | 同左 |
| 额外 metadata 通信 | 0 | D·E_r ≈ 1 MB |
| **dispatch 通信增幅** | 0 | **+0.4%** |

---

## 5. 数值示例

### 参数：N=4096, K=8, E=256, R=8, E_r=32, C=64, H=7168

| 指标 | 方案 A (HybridEP) | 方案 B (notify_dispatch) | B/A 比值 |
|------|--------|--------|----------|
| NVLink metadata 通信 | 7,340,032 B (**7 MB**) | 1,280 B (**1.3 KB**) | **1/5,734** |
| HBM 总读取 | 8,388,608 B (**8 MB**) | 262,144 B (**256 KB**) | **1/32** |
| HBM 总写入 | 12,713,984 B (**12.1 MB**) | 98,688 B (**96 KB**) | **1/129** |
| 额外显存 | 8,388,608 B (**8 MB**) | 0 | **0** |
| Dispatch 额外通信 | 0 | 1,048,576 B (**1 MB**) | — |
| Dispatch 正文通信 | 234,881,024 B (**224 MB**) | 同左 | — |
| Dispatch 通信增幅 | — | **+0.4%** | — |

---

## 6. 随参数变化的 scaling 分析

### 6.1 NVLink metadata 通信量随 N 变化（E=256, R=8）

```
方案 A: N·E·(R-1) = 1792·N bytes    → 与 N 线性增长
方案 B: R·(R+E_r)·4 = 1280 bytes    → 与 N 无关（常数）
```

| N | 方案 A | 方案 B | 比值 |
|---|--------|--------|------|
| 1024 | 1.75 MB | 1.3 KB | 1/1,434 |
| 4096 | 7 MB | 1.3 KB | 1/5,734 |
| 16384 | 28 MB | 1.3 KB | 1/22,938 |
| 65536 | 112 MB | 1.3 KB | 1/91,750 |

> **N 越大，方案 B 优势越明显**。方案 A 的通信量 O(N) 增长，方案 B 恒定。

### 6.2 NVLink metadata 通信量随 E 变化（N=4096, R=8）

```
方案 A: N·E·(R-1) = 28672·E bytes   → 与 E 线性增长
方案 B: R·(R+E/R)·4 = 256+4E bytes  → 与 E 弱线性增长（仅通过 E_r=E/R）
```

| E | E_r | 方案 A | 方案 B | 比值 |
|---|-----|--------|--------|------|
| 64 | 8 | 1.75 MB | 512 B | 1/3,584 |
| 256 | 32 | 7 MB | 1.3 KB | 1/5,734 |
| 1024 | 128 | 28 MB | 4.3 KB | 1/6,827 |

### 6.3 HBM 读取量随 R 变化（N=4096, E=256）

```
方案 A: N·R·E bytes   → O(R) 增长（scan 读整个 global_routing_map）
方案 B: N·K·8 bytes   → 与 R 无关（仅读本地 topk_idx）
```

| R | 方案 A HBM Read | 方案 B HBM Read | 比值 |
|---|--------|--------|------|
| 4 | 4 MB | 256 KB | 1/16 |
| 8 | 8 MB | 256 KB | 1/32 |
| 16 | 16 MB | 256 KB | 1/64 |

---

## 7. 结论

### 方案 B (get_dispatch_layout + notify_dispatch) 的核心优势

1. **消除 allgather 瓶颈**：metadata 阶段 NVLink 通信量从 O(N·E·R) 降至 O(R²+R·E_r)，**与 token 数 N 完全无关**。典型场景下通信量减少 **5000×** 以上，延迟从数十 μs 降至 ~1 μs。

2. **HBM 访存大幅降低**：无需写入和读取 global_routing_map。总 HBM 读取降低 **32×**（从 8 MB 到 256 KB），总 HBM 写入降低 **129×**（从 12 MB 到 96 KB）。

3. **零额外显存**：无需分配 N·R·E 大小的 global_routing_map buffer。

4. **无 NCCL 依赖**：对称内存 + barrier 完成全部跨 rank 协调，无 NCCL launch overhead 和 host 同步。

### 方案 B 的代价

1. **Dispatch 通信微增 ~0.4%**：sender 附带 E_r bytes expert routing metadata per token。相对正文数据量 H bytes/token 可忽略。

2. **Dispatch kernel 复杂度增加**：sender 需在寄存器中计算 expert 目标偏移（topk_idx + expert_rank_prefix + expert_channel_prefix），receiver 需解析 routing metadata 完成 permute。不增加 memory-bound 压力。

3. **前置依赖 topk_idx**：方案 B 要求 gating 层产出 topk_idx，而 HybridEP 可直接接收 routing_map。绝大多数 MoE 实现原生产出 topk_idx，此项不构成实际限制。
