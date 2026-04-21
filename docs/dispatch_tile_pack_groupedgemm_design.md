# Intranode EP 三流水设计草图：Dispatch + Tile-Pack + GroupedGEMM

本文记录针对 `csrc/kernels/cco/ep/intranode.cu` 的设计建议，目标是在单机多卡 EP 场景下实现高性能的三级流水：

```text
dispatch(g+2) -> permute/pack(g+1) -> groupedgemm(g)
```

本文明确忽略 `pipeline_ep.py` 的原型实现，只讨论当前 C++/CUDA 路径上更适合落地的方案。

## 1. 问题定义

当前 `intranode.cu` 的接收侧核心逻辑是：

1. sender 将 token 发送到本 rank 的 staging buffer
2. receiver 读取 `dispatched_topk_idx_buffers`
3. 对每个命中的 local expert 做：
   - `atomicAdd_system(expert_slot_idx[expert])`
   - 计算该 token 在 permuted expert-major buffer 中的目标 row
   - 写 `dispatch_to_expert_map`
   - 后续再将 `recv_x` 搬运成 groupedgemm 可消费的输入

现有做法的问题：

- `atomicAdd_system(expert_slot_idx[expert])` 会在热门 expert 上形成严重热点
- 运行时由“到达顺序”决定 expert row，导致 permute 是 row-owned scatter
- `channel` 是通信划分维度，不是计算划分维度
- 即使按 channel 分工去写 expert output，最终也无法保证 groupedgemm 所需的 `M` 维 block tile 连续
- `dispatch -> permute -> groupedgemm` 的边界错位，导致 overlap 很难做深

结论：

**按 channel ownership 做 scatter，不适合作为 block-tiled groupedgemm 的前置排布。**

## 2. 关键约束

目标后端是 groupedgemm，因此输入 `A` 需要满足：

- expert-major 排布
- 每个 expert 的 `M` 维尽量连续
- 最好按 groupedgemm 偏好的 `BLOCK_M` 对齐
- 一旦某个 `M` tile ready，就可以直接启动计算

这意味着：

- `dispatch` 的自然布局和 `groupedgemm` 的自然布局不是同一个布局
- 必须将“通信布局”和“计算布局”解耦

## 3. 设计原则

推荐采用双层布局：

1. 通信布局：`recv_x`
   - 只负责稳定接收 token
   - 布局按 source rank / channel 连续即可
   - 优先保证传输和 receiver copy 简单高效

2. 计算布局：`packed_x`
   - 专门为 groupedgemm 准备
   - 按 expert-major、按 `BLOCK_M` 对齐
   - 由单独的 tile-pack 阶段从 `recv_x` gather 生成

对应的数据流：

```text
sender -> recv_x(staging, stable rows)
       -> metadata(count/scan/route_rows)
       -> packed_x(expert-major, BM-aligned)
       -> groupedgemm
```

其中 `BM` 表示 groupedgemm 后端偏好的 `M` tile，比如 `64` 或 `128`。

## 4. 推荐方案

### 4.1 当前最推荐的落地方案

如果需要继续复用现有 CK/hipBLASLt groupedgemm，推荐方案是：

```text
dispatch -> group count/scan -> route_rows -> tile-pack -> groupedgemm
```

也就是：

- `dispatch` 只把 token 接收到 `recv_x`
- 不再在 receiver 热路径里通过 atomic 抢 expert slot
- 先构造当前 group 的路由元数据
- 再以 tile-owned gather 的方式生成 `packed_x`
- groupedgemm 直接吃 `packed_x`

### 4.2 理论上限更高的方案

如果后续愿意改 groupedgemm 内核，最高上限的形态是：

```text
dispatch -> route_rows/tile_row_ids -> indexed gather GEMM
```

也就是取消显式 `packed_x` 物化，让 GEMM CTA 直接按 tile gather `recv_x`。

这条路线性能上限最高，但对 GEMM 后端侵入更大。本文先聚焦更现实的 `tile-pack` 路线。

## 5. 为什么 atomic permute 慢

当前思路是 producer-owned scatter：

```text
row 到达
-> 查 expert
-> atomicAdd(expert_slot_idx[e])
-> 抢一个目标 row
-> scatter 到目标位置
```

它的问题：

- 热门 expert 上所有 warp 都在争抢同一个计数器
- 目标写地址随 token 到达动态决定，不利于形成稳定 tile
- 即便某些 row 已经 ready，也不能自然聚合成完整 `BLOCK_M`
- 运行时在 data plane 上处理布局问题，代价太高

而推荐方案是 consumer-owned tile pack：

```text
先决定 tile 需要哪些 row
-> CTA 拿到一个 tile 的 row list
-> 从 recv_x gather
-> 连续写 packed_x[tile]
```

读可以离散，写必须连续。对于 groupedgemm 来说，后者才是关键。

## 6. 三流水的正确切分单位

流水线单位不应该是 `channel`，也不应该是单个 token chunk，而应该是：

- `expert group`
- 或者进一步细化到 `tile group`

原因：

- groupedgemm 的消费粒度是 expert-major 的 tile
- dispatch 的分工维度和 compute 的分工维度不同
- 只有把 pipeline 切到接近 compute 粒度，overlap 才有意义

本文建议先采用 `expert group` 作为流水线单位。

设：

- `R = num_ranks`
- `E_local = num_experts / R`
- `G = num_groups_per_rank`
- `E_g = E_local / G`

第 `g` 个 group 负责：

```text
local experts in [g * E_g, (g + 1) * E_g)
```

流水线时序：

```text
comm_stream:    dispatch(group g+2)
pack_stream:    metadata + tile-pack(group g+1)
gemm_stream:    groupedgemm(group g)
```

## 7. 核心数据结构

### 7.1 长驻 staging buffer

建议 `recv_x` 在一轮 EP 计算期间保持长驻，不做 group ring 复用：

```cpp
recv_x            [max_recv_rows, H]
recv_topk_idx     [max_recv_rows, K]      // local expert id, invalid = -1
recv_topk_weight  [max_recv_rows, K]
recv_send_group   [max_recv_rows]         // token 首次发送到本 rank 时归属的最小 group
```

说明：

- 每个 token 在某个目标 rank 上只接收一次
- 如果同一个 token 命中了该 rank 上多个 local expert，后续 group 通过 metadata 重用同一 row

### 7.2 每个 group 的三槽 ring metadata

```cpp
struct GroupSlot {
    int* cta_counts;        // [num_cta, E_g]
    int* cta_bases;         // [num_cta, E_g]

    int* expert_counts_raw; // [E_g]
    int* expert_offsets;    // [E_g + 1], BM-aligned prefix

    int* route_rows;        // [group_rows_aligned]
    float* route_weights;   // [group_rows_aligned], 可选
    int16_t* route_k;       // [group_rows_aligned], 可选

    void* packed_x;         // [group_rows_aligned, H]
    void* packed_x_scales;  // 若输入量化需要
};
```

其中：

- `route_rows[pos]` 表示 `packed_x[pos]` 应从 `recv_x[route_rows[pos]]` gather
- 尾部 padding 位置用 `-1` 填充，pack 时写零

### 7.3 Dispatch 完成边界

为每个 group 准备一个前缀边界：

```cpp
recv_group_end[g]
```

含义：

- 当 `dispatch(group g)` 完成后
- `recv_x[0 : recv_group_end[g])` 必然已稳定可读

这样 `pack(group g)` 只需要扫描这个 prefix 对应的 metadata。

## 8. 发送规则

对每个 token 和每个目标 rank：

- 只发送一次
- 若该 token 在此 rank 上命中多个 local experts，则取最小 `local_group_id` 作为 `send_group`

即：

```text
send_group(token, dst_rank) = min(local_group_id of all matched local experts on dst_rank)
```

这样可以保证：

- token 不会被重复传输
- 较后的 group 可以复用较早 group 已经收到的 `recv_x` row

## 9. Group 内的 metadata 构建

推荐分三步完成。

### 9.1 第一步：count

扫描：

```text
rows in [0, recv_group_end[g))
experts in current group
```

统计每个 CTA 负责的 row 范围内，每个 expert 的命中数量：

```cpp
cta_counts[cta_id][e_in_group]
```

这一步只扫 metadata：

- `recv_topk_idx`
- `recv_topk_weight`

不搬 hidden 数据。

### 9.2 第二步：scan

对 `cta_counts` 做 scan，得到：

- `expert_counts_raw[e]`
- `expert_offsets[e]`
- `cta_bases[cta_id][e]`

其中：

```text
aligned_count[e] = round_up(expert_counts_raw[e], BM)
expert_offsets[e+1] = expert_offsets[e] + aligned_count[e]
```

### 9.3 第三步：fill route_rows

再次扫描相同 prefix rows。
每个 CTA 对自己负责的行做局部 prefix，把 `recv_x` row id 写入：

```text
route_rows[expert_offsets[e] + cta_bases[cta][e] + local_rank] = recv_row
```

尾部 padding 填 `-1`。

如果后续 combine 需要 route weight，可以同时写：

```text
route_weights[pos] = recv_topk_weight[row, k]
route_k[pos] = k
```

这一步没有全局 atomic，因为目标地址已经被 scan 唯一确定。

## 10. Tile-Pack 内核

这是整个设计里最关键的一步。

### 10.1 ownership

不要让 channel/warp 按到达顺序去 scatter 到 expert output。
而要让一个 CTA 拥有一个 tile：

```text
(expert e, tile t)
```

### 10.2 输入和输出

输入：

```text
route_rows[tile_base : tile_base + BM]
recv_x
```

输出：

```text
packed_x[tile_base : tile_base + BM]
```

### 10.3 逻辑草图

```cpp
__global__ void pack_group_tiles(
    const half* recv_x,      // [max_recv_rows, H]
    const int* route_rows,   // [rows_aligned]
    half* packed_x,          // [rows_aligned, H]
    int rows_aligned,
    int H
) {
    // 每个 CTA 负责一个 tile
    // tile_base = blockIdx.x * BM
    // 对于 tile 内每一行：
    //   row = route_rows[tile_base + m]
    //   if row >= 0: packed_x[...] = recv_x[row, :]
    //   else:        packed_x[...] = 0
}
```

特性：

- `recv_x` 读取是 gather
- `packed_x` 写入是连续
- 输出天然满足 groupedgemm 对 `M` 维 block tile 的要求

## 11. GroupedGEMM 的消费方式

groupedgemm 只需要看到：

- `packed_x`
- 当前 group 覆盖的权重切片
- `expert_counts_raw`
- `expert_offsets`

布局示意：

```text
packed_x:
  [expert0 rows | pad]
  [expert1 rows | pad]
  [expert2 rows | pad]
  ...
```

这样 groupedgemm 可以直接按 expert 的连续区间启动。

如果后端支持真实 group 长度：

- `group_lens` 传 `expert_counts_raw`
- `group_offs` 传 `expert_offsets`

如果后端必须按 tile 对齐：

- `group_lens` 传 aligned count
- pad 行必须写零
- 后处理时忽略 pad 输出

## 12. 三流水 host 侧时序草图

推荐使用三条 stream：

- `comm_stream`
- `pack_stream`
- `gemm_stream`

伪代码：

```cpp
for (int g = 0; g < G + 2; ++g) {
    if (g < G) {
        launch_dispatch_group(comm_stream, g);
        record(dispatch_done[g]);
    }

    if (g - 1 >= 0 && g - 1 < G) {
        int s = (g - 1) % 3;
        wait(pack_stream, dispatch_done[g - 1]);

        launch_count_group_routes(pack_stream, group=g - 1, slot=s);
        launch_scan_group_counts(pack_stream, group=g - 1, slot=s);
        launch_fill_group_route_rows(pack_stream, group=g - 1, slot=s);
        launch_pack_group_tiles(pack_stream, group=g - 1, slot=s);

        record(pack_done[g - 1]);
    }

    if (g - 2 >= 0 && g - 2 < G) {
        int s = (g - 2) % 3;
        wait(gemm_stream, pack_done[g - 2]);

        launch_grouped_gemm(gemm_stream, slot=s, group=g - 2);
    }
}
```

注意：

- `recv_x` 应长驻，不做三槽复用
- 三槽 ring 复用的是当前 group 的 metadata 和 `packed_x`

## 13. 为什么这个方案适合 block-tiled groupedgemm

这一点是本设计相对“channel-owned scatter”最重要的区别。

错误方案：

```text
channel/warp 到达一行
-> 直接把该行 scatter 到 expert output 某个位置
```

问题是：

- 物理写入顺序被 channel 到达顺序决定
- 无法保证 groupedgemm 所需的 `BLOCK_M` 连续 row

推荐方案：

```text
先生成 route_rows
-> 再由 tile-owned CTA gather recv_x
-> 连续写 packed_x[tile]
```

这样：

- 通信布局和计算布局完全解耦
- groupedgemm 拿到的输入天然是连续 tile
- `channel` 只影响 dispatch，不影响最终 A 的计算布局

## 14. 预期收益

相对当前 `atomicAdd_system(expert_slot_idx)` 路线，预期收益包括：

- 去掉热门 expert 上的 system-scope atomic 热点
- 避免 row-owned scatter 导致的随机目标写
- 让 groupedgemm 真正吃到 expert-major、BM-aligned 的输入
- 使三流水切分粒度与 compute 边界一致
- 后续若继续演进到 indexed gather GEMM，可直接复用 route metadata

## 15. 主要代价和风险

代价：

- 需要额外的 metadata kernel：count / scan / fill route rows / pack
- `pack` 仍然是一遍额外的数据搬运
- 某些 group 会重复扫描已到达 prefix rows 的 metadata

风险：

- 如果 group 太小，metadata/kernel launch 开销占比会偏高
- 如果 group 太大，overlap 变差，第一个 groupedgemm 启动变晚
- 若 dispatch 和 groupedgemm 不做 CU/SM 资源隔离，overlap 可能被硬件资源争用吞掉

## 16. 参数建议

初始调参建议：

- `BM` 直接采用 groupedgemm 后端真实偏好的 `M` tile
- `G` 先从每组 `4` 个 local experts 开始尝试
- `pack` kernel 的 hidden tile 可先试 `128` 或 `256`
- `expert_alignment` 与 `BM` 保持一致

如果 dispatch/pack 是 persistent 或长时间驻留 kernel，建议：

- 给 `comm_stream` 和 `pack_stream` 留少量 CU/SM
- 给 `gemm_stream` 留大头资源

## 17. 实施顺序建议

建议按以下顺序推进：

### Phase 1

保持现有 dispatch，只保留 `recv_x` staging，不再在 receiver 里做 atomic permute。

### Phase 2

新增 group 级 metadata 构建：

- `count_group_routes`
- `scan_group_counts`
- `fill_group_route_rows`

### Phase 3

新增 `pack_group_tiles`，生成 groupedgemm 可直接消费的 `packed_x`。

### Phase 4

将 groupedgemm 接到 `packed_x` 上，验证 block-tile 连续输入是否成立。

### Phase 5

再引入三流水调度与 CU mask 分区，观察 overlap 收益。

### Phase 6

若后续需要进一步追求上限，再考虑将 `packed_x` 物化删除，演进到 indexed gather GEMM。

## 18. 最终结论

对于 `csrc/kernels/cco/ep/intranode.cu` 当前这条路径：

- **不推荐**继续以 `atomic counter` 方式在 receiver 热路径中 materialize expert-major rows
- **不推荐**按 `channel` ownership 直接 scatter 到 groupedgemm 输入
- **推荐**采用：

```text
dispatch -> metadata(count/scan) -> tile-owned pack -> groupedgemm
```

更准确地说：

**通信阶段负责把 token 稳定接收到 `recv_x`，计算前置阶段负责把 `recv_x` 重组为 expert-major 的 block-tiled `packed_x`，groupedgemm 只消费这个计算友好的连续布局。**
