# [猜想] 基于 Split-M 的 EP Overlap 映射设计

## 1. 状态说明

本文内容为**猜想**，用于整理当前对 `tests/pytorch/cco/test_ep.py` 这条 `per-expert atomic-based overlap` 路径的设计思考。

本文**不是**已经验证的实现方案，主要目标是回答下面这个问题：

```text
当 local expert 的平均 token 数小于 groupedgemm 的 BLOCK_M 时，
如何把 token -> dispatched token -> permuted token 的映射扩展为
支持 split-M overlap 的元数据设计？
```

## 2. 问题背景

当前实现里，`dispatch_permute` 会在 receiver 侧用 `atomicAdd(expert_slot_idx[expert])`
给每个 local expert 分配连续的 row，最终得到 expert-major 的 `permuted_x`。

对应的 ready 信号是：

- `expert_tail_idx[expert]`
- 或 Python 侧传入的 `group_tail_idx`

它表达的语义本质上是：

```text
expert e 当前已经 materialize 了多少行
```

这套语义在 `BLOCK_M` 较大时存在一个天然问题：

- 若 `BLOCK_M = 256`
- 单机 `ep8`
- 平均每个 expert 仅约 `128` token

那么大多数 expert 都不足以形成一个完整的计算 tile。此时即使 `dispatch` 和 `groupedgemm`
逻辑上是并发的，计算侧仍然会等待更多 expert 行到齐，最终表现接近同步执行，overlap 很弱。

## 3. 核心判断

### 3.1 问题不在于 token 到 permuted 的映射不存在

现有实现实际上已经具备：

```text
dispatched token × topk slot -> permuted token row
```

也就是当前的：

- `recv_topk_idx`
- `dispatch_to_expert_map`
- `expert_tail_idx`

已经足以表达：

1. 一个 dispatched token 命中了哪个 local expert
2. 它最终落在该 expert-major 排布中的哪一行

### 3.2 真正缺的是 compute 侧的 split 视图

当前 `groupedgemm` 的 group 粒度还是：

```text
one group == one expert
```

所以 compute 只能等待“某个 expert 的前缀已经足够长”。

如果希望在小 expert 场景下做出更深的 overlap，应该把：

```text
expert
```

继续切成：

```text
expert -> split_0, split_1, ..., split_{S-1}
```

也就是把整个 expert 的 M 维看成一个完整 GEMM 问题，再切成若干 `num_splits_m` 子块。

## 4. 设计原则

本文的核心猜想是：

**不要为了 split-M 再发明一套新的物理 row 布局，而应当保留当前的 expert-major `permuted_x` 物理布局，仅把 split 作为 `permuted_x` 上的逻辑区间视图。**

也就是说：

- 物理布局仍然是 `expert-major`
- `dispatch_to_expert_map` 仍然映射到最终 `permuted_row`
- `split` 只是 `permuted_row` 的一个静态区间分段

这样能最小化对现有 `dispatch_permute` 路径的侵入。

## 5. 四层坐标系

建议把问题拆成 4 层坐标系，但只保留 2 层物理索引。

### 5.1 token space

```text
(src_rank, local_token_id)
```

或者全局写法：

```text
global_token_id = src_rank * N + local_token_id
```

### 5.2 dispatch space

```text
dispatch_row
```

含义：

- token 在目标 rank 上，按 rank 去重后进入 dispatch buffer 的行号

### 5.3 permute space

```text
(local_expert_id, expert_row)
```

或 flatten 后：

```text
permuted_row
```

含义：

- token 在最终 `permuted_x` 中的物理行号

### 5.4 split space

```text
(local_expert_id, split_id)
```

含义：

- 某个 expert 的 M 维逻辑子块
- 它只是 `permuted_x` 上的一段区间
- 不是新的物理 buffer 编号

## 6. 三段映射关系

本文建议的映射链条为：

```text
token
-> dispatched token
-> permuted token
-> split group
```

其中前两段是物理映射，最后一段是逻辑映射。

### 6.1 token -> dispatched token

对给定目标 rank `dst_rank`：

```text
dispatch_row =
    rank_base[src_rank, dst_rank] + token_idx_in_rank[token, dst_rank]
```

其中：

- `token_idx_in_rank[token, dst_rank]`
  - 表示当前 token 发往 `dst_rank` 后，在该目标 rank 内的去重序号
- `rank_base[src_rank, dst_rank]`
  - 表示目标 rank 上，来自所有更小 source rank 的 dispatched token 前缀和
  - 可由 `rank_prefix_matrix` 推导

也就是说，这一层回答的是：

```text
一个 token 在目标 rank 的 dispatch buffer 里排第几行
```

### 6.2 dispatched token -> permuted token

这一层保持现有 receiver 侧 atomic materialization 语义不变：

```text
expert = recv_topk_idx[dispatch_row, k]
expert_row = atomicAdd(expert_slot_idx[expert], 1)
permuted_row = expert_offs[expert] + expert_row
dispatch_to_expert_map[dispatch_row, k] = permuted_row
expert_tail_idx[expert] += 1
```

这一层回答的是：

```text
一个 dispatched token 命中的某个 local expert，
最终落到 expert-major permuted_x 的哪一行
```

### 6.3 permuted token -> split group

这一层不需要额外物化 row buffer，只需要静态 split 元数据：

```text
local_row = permuted_row - expert_offs[expert]
split_id = locate(local_row within expert e's split ranges)
```

也就是说：

```text
split_row == permuted_row
```

地址不变，只是 compute 调度时把这段 row 视作一个单独 group。

## 7. 推荐的数据结构

下面给出一组最小可落地的数据结构。

### 7.1 与当前设计一致、应继续保留的数据

#### `token_idx_in_rank [N, R] int32`

语义：

- 每个 token 对每个 rank 的去重后局部序号
- 若该 token 不发往该 rank，则为 `-1`

用途：

- `token -> dispatch_row`

#### `rank_prefix_matrix [R, R] int32`

语义：

- receiver 侧不同 source rank 的 dispatch row 前缀和

用途：

- 计算 `rank_base[src_rank, dst_rank]`

#### `recv_topk_idx [D, K] int32/int64`

语义：

- `dispatch_row` 上每个 topk slot 命中的 local expert id
- 未命中为 `-1`

用途：

- `dispatch_row -> local expert`

#### `dispatch_to_expert_map [D, K] int32`

语义：

- `dispatch_row` 的第 `k` 个命中最终落到哪个 `permuted_row`

用途：

- `dispatched token -> permuted token`

#### `expert_slot_idx [E_r] int32`

语义：

- receiver 侧对每个 local expert 的原子 slot 分配器

用途：

- materialize `permuted_x`

#### `expert_tail_idx [E_r] int32`

语义：

- 当前 expert 已经 ready 的前缀长度

建议把它明确理解为：

```text
ready_count per expert
```

而不是“最后一个 row id”。

### 7.2 新增的静态 split 元数据

#### `expert_offs [E_r + 1] int32/int64`

语义：

- 每个 local expert 在 `permuted_x` 中的绝对起始位置

即：

```text
expert e occupies rows [expert_offs[e], expert_offs[e + 1))
```

#### `num_splits_per_expert [E_r] int32`

语义：

- 每个 expert 切成多少个 split

可选两种策略：

1. 固定 `num_splits_m`
2. 固定 `split_rows_target`，按 `ceil_div(count[e], split_rows_target)` 决定

#### `split_base [E_r + 1] int32`

语义：

- flatten 后，每个 expert 的第一个 split group 在全局 group 空间中的起始编号

即：

```text
expert e's split groups are
[split_base[e], split_base[e + 1))
```

#### `split_offs [G_split + 1] int32/int64`

语义：

- flatten 后的 split group 在 `permuted_x` 上对应的绝对 row 区间

即：

```text
split group g occupies rows [split_offs[g], split_offs[g + 1))
```

#### `split_expert_id [G_split] int32`

语义：

- split group `g` 属于哪个 local expert

#### `split_lo_rel [G_split] int32`

语义：

- split group `g` 在所属 expert 内的相对起始行号

#### `split_hi_rel [G_split] int32`

语义：

- split group `g` 在所属 expert 内的相对结束行号

它同时可以作为 ready threshold。

## 8. split 的切分方式

### 8.1 固定 `num_splits_m`

若每个 expert 统一切成 `num_splits_m` 段，则对 expert `e`：

```text
split_lo(e, s) = floor(s * count[e] / num_splits_m)
split_hi(e, s) = floor((s + 1) * count[e] / num_splits_m)
```

优点：

- 形式简单
- 所有 expert 的 split 语义一致

缺点：

- 小 expert 可能切出许多很短甚至空的 split

### 8.2 固定目标 split 行数

更推荐的猜想是固定目标 split 大小，例如 `64` 或 `128`：

```text
num_splits_e = ceil_div(count[e], split_rows_target)
```

再按均匀方式切分该 expert。

优点：

- 小 expert 不会被过度切分
- 大 expert 会自动多切
- 更贴近实际 overlap 粒度

## 9. GroupedGEMM 的 group 语义如何变化

当前 `grouped_gemm` 的 group 语义本质上是：

```text
one group == one expert
```

本文猜想建议改为：

```text
one group == one (expert, split)
```

但注意：

- 输入 tensor 仍然是同一个 `permuted_x`
- weight tensor 仍然按 expert 取
- 变化的只是 group 元数据和 wait 条件

也就是 compute 侧看到的是一组更细的逻辑 group。

## 10. compute wait 条件的推荐改法

最重要的一点：

**不建议新增 `split_tail_idx`。**

而应继续复用：

```text
expert_tail_idx[expert]
```

再配一张静态表：

- `split_expert_id[g]`
- `split_hi_rel[g]`

那么 group `g` 的 wait 条件就是：

```text
expert_tail_idx[split_expert_id[g]] >= split_hi_rel[g]
```

换句话说：

- ready signal 仍然是 per-expert 的动态前缀长度
- split 只是静态阈值表

这是本文最关键的元数据设计判断。

## 11. 为什么不建议引入 `split_tail_idx`

如果把 ready signal 扩成：

```text
split_tail_idx[E_r, S]
```

会带来几个问题：

1. receiver 每写一行都要决定属于哪个 split，并更新额外 atomic
2. 多个 split 共享同一个 expert 的权重，但 ready 信号被碎片化
3. receiver 热路径更复杂
4. 现有 `expert_tail_idx` 已经能表达“前缀 ready”这一核心语义

而使用：

```text
expert_tail_idx + split_required_tail
```

则只需在 compute 侧多查两张静态表，receiver 完全可以保持当前 atomic materialization 模式。

## 12. 一个更具体的 group 定义

对于 flatten 后的 split group `g`：

```text
group_offs[g]     = split_offs[g]
group_offs[g + 1] = split_offs[g + 1]
group_expert[g]   = split_expert_id[g]
group_ready[g]    = split_hi_rel[g]
```

于是 compute 侧的逻辑变成：

```text
wait until expert_tail_idx[group_expert[g]] >= group_ready[g]
then compute rows [group_offs[g], group_offs[g + 1))
using weight of expert group_expert[g]
```

这里的关键点是：

- `group_offs` 决定数据区间
- `group_expert` 决定用哪个权重
- `group_ready` 决定等待阈值

## 13. 这套设计对映射链条的含义

把问题重新写成一句话：

### 13.1 token -> dispatched token

由：

- `token_idx_in_rank`
- `rank_prefix_matrix`

决定。

### 13.2 dispatched token -> permuted token

由：

- `recv_topk_idx`
- `expert_slot_idx`
- `expert_offs`
- `dispatch_to_expert_map`

决定。

### 13.3 permuted token -> split

由：

- `split_base`
- `split_offs`
- `split_expert_id`
- `split_lo_rel`
- `split_hi_rel`

静态决定。

因此本文的核心观点可以压缩成一句：

**物理映射只做到 `permuted_row`，`split` 不再生成第三套物理 row id，而是作为 `permuted_row` 上的逻辑 group 视图。**

## 14. 这种设计的优点

### 14.1 对 receiver 改动最小

receiver 仍然可以保持：

- `atomicAdd(expert_slot_idx)`
- 写 `dispatch_to_expert_map`
- 更新 `expert_tail_idx`

### 14.2 对 `dispatch_to_expert_map` 语义零破坏

现有：

```text
dispatch_row × topk -> permuted_row
```

的语义可以直接保留。

### 14.3 对 GroupedGEMM 更自然

compute 侧本来就在消费：

```text
permuted_x + group_offs + group_tail_idx
```

现在只需要把：

- `group_offs`
  从 per-expert 改成 per-split
- `group_tail_idx`
  的含义改为“查 owner expert 的 tail 是否超过静态阈值”

### 14.4 不需要第三套 buffer

不需要：

- split-major 新 buffer
- split-major 新 permute map

从而避免增加额外搬运和额外地址空间。

## 15. 风险与未验证点

本文仍有几个关键点尚未验证，因此必须标记为**猜想**。

### 15.1 expert 内顺序是否重要

当前 atomic materialization 决定了：

- expert 内 token 的最终顺序由到达顺序决定

本文默认假设：

- FFN expert 对 token 行顺序不敏感
- 只要是同一 expert 的连续前缀 ready，就可以计算

若后续某条路径要求稳定顺序，这个前提需要重新检查。

### 15.2 split 粒度如何与 BLOCK_M 对齐

如果 split 比 `BLOCK_M` 更小，compute 的 kernel launch 与调度开销可能变重。
如果 split 仍然接近 `BLOCK_M`，小 expert 场景的 overlap 可能提升有限。

### 15.3 compute kernel 的 owner-expert wait 改法是否足够干净

现有 grouped_gemm kernel 把：

```text
group_idx
```

直接当成 tail array 的下标使用。

若改成：

```text
group_idx -> owner_expert -> required_tail
```

需要确认 Triton kernel 中不会引入过高的额外访存或分支开销。

## 16. 建议的最小验证路径

若要验证本文猜想，建议按下面顺序推进：

1. 先在 Python 侧根据 `moe_recv_expert_counter` 构造 `split_offs / split_expert_id / split_hi_rel`
2. 保持 `dispatch_permute` 内核不变，仅把 compute 侧 group 定义改成 `(expert, split)`
3. 让 grouped_gemm 的 wait 条件改为：
   `expert_tail_idx[group_expert[g]] >= group_ready[g]`
4. 观察 `ep8`、平均 `128 token/expert` 时首批 group 的启动时间是否明显提前
5. 再决定是否需要继续把 split 信息下沉到 C++/HIP 路径

## 17. 总结

本文的最终猜想如下：

### 17.1 总体结论

当小 expert 场景下 `BLOCK_M` 偏大导致 overlap 很弱时，最值得改的不是
`token -> permuted` 这条映射本身，而是 compute 侧对 `permuted_x` 的 group 解释方式。

### 17.2 推荐设计

推荐保留：

- `token -> dispatch_row`
- `dispatch_row -> permuted_row`

这两层物理映射不变。

然后新增：

- `permuted_row -> (expert, split)` 的静态逻辑映射

并让 compute 用：

- `split_offs`
- `split_expert_id`
- `split_hi_rel`

去解释 `permuted_x`。

### 17.3 一句话版本

**猜想：最合适的 split-M overlap 元数据设计，不是把 expert-major 物理布局改成 split-major，而是在现有 expert-major `permuted_x` 之上增加一层 split 逻辑视图，并继续复用 `expert_tail_idx` 作为唯一动态 ready signal。**
