# MegaAttn 设计（FlyDSL，gfx950）

把 Ulysses attention 层的 “QKV GEMM → a2a → attention → a2a → o_proj GEMM” 放进**每个方向一个 kernel**：GEMM tile、attention tile 和交换 block 在同一次 launch 里按 flag 流水，前向和反向都做。前置工作是 GemmA2A / A2AGemm（`docs/ulysses_gemm_a2a.md`），任务描述见 `docs/mega_attn_prompt.md`。

## 1. 起点和上限

`bench_ulysses_a2a.py --fused`，Qwen3-30B-A3B，n = 4，mbs 1，一层 attention + dense MLP，每 rank median，最大 rank：

| 长度 | fwd 层 | bwd 层 | 合计 | 四个交换阶段（关键 rank） | attention fwd + bwd |
|---|---|---|---|---|---|
| 16K | 1387 | 2897 | 4284 µs | 249 + 220 + 255 + 420 = 1144 µs | 558 + 1443 = 2001 µs |
| 32K | 3658 | 8000 | 11658 µs | 453 + 368 + 489 + 645 = 1955 µs | 2156 + 5417 = 7573 µs |

交换阶段全部藏到 attention 后面的上限：16K 每层 -27%，32K 每层 -17%。attention 本身是交换的 2–4 倍长，时间上放得下。

## 2. 序列切分：条带（striped）

交换要藏在 attention 后面，两侧的要求相反：

- QKV 一侧：causal 下 q-block 只依赖它之前的 KV，所以 KV 要**按全局位置从前往后到达**，attention 才能边收边算。
- o_proj 一侧：rank r 的 o_proj 需要所有 rank 算完属于 r 的 q-block，所以 attention 的完成顺序要**在各 rank 之间轮转**，否则最后一个 rank 的输入全部压在最后。
- 交换本身要所有 XGMI 链路同时忙：按源 rank 串行拉取会让每段数据的到达慢 n - 1 倍。

contiguous 切分（rank r 持有 `[r·s/n, (r+1)·s/n)`）下三者冲突：各源并行拉取时，任意 q-block 都要等源 0..r-1 全部到齐，只有 rank 0 那段（causal 工作量的 1/n²）能提前开始；o_proj 侧最后一段的 owner 拿不到任何重叠。Megatron 的 zigzag（2n 段）能把前一半提前，约藏住一半的 QKV 交换。

**条带切分**：序列切成长 `cs` 的块（`cs` = pull 的 m-tile = attention 的 BLOCK_M 的倍数，默认 256），第 `c` 块归 rank `c % n`。

- 每个 rank 的本地 token 是第 `k·n + r` 块（`k = 0, 1, ...`）。MLP、norm、loss 逐 token，不受影响；RoPE 本来就在 a2a 之后按全局位置做。
- GemmA2A 的 pull 单元 `(m_tile, src)` 正好是全局第 `m_tile·n + src` 块；现有的 m-major、各源轮转的领取顺序不变，所有链路同时忙，KV 按全局位置顺序到达。
- q-block `c` 的 O 归 rank `c % n`，各 rank 的 o_proj 输入轮流到达。
- Primus 侧只改 CP 切分（`ulysses_fused_a2a_patches.py` 的 `_contiguous_cp_split` 换成按块轮转的 narrow / view）。

## 3. 前向 kernel 的角色

一个 launch，block 按派发顺序分角色（被等待的工作先派发）：

| 角色 | 数量 | 工作 | 等什么 | 产出 |
|---|---|---|---|---|
| QKV GEMM | 每 tile 一个 block | 本地 token 的 QKV GEMM，写本 rank 对称暂存 bank | — | 每 tile 一个 flag（现有） |
| pull | 少量 persistent | 按 `(m_tile, src)` 领单元，读远端暂存，写 q / k / v（全局位置） | 源 rank 的 GEMM tile flag | 每个全局块一个本地到达 flag |
| attention / o_proj worker | 其余 CU，persistent | 先按工作表领 attention item（head, q-block），做完再领 o_proj m-tile | attention：块 0..c 的到达 flag；o_proj：n 个源的 O 到达 flag | O、LSE 写普通张量（反向要用）；每个 (q-block, head 组) 一个 O-ready flag |
| push | 少量 persistent | 等某全局块所有本地 head 的 O-ready，把整行（本 rank 全部 head，`h/n·d` 连续）推到 owner 的 recv bank | O-ready flag | owner 上的到达 flag（A2AGemm 现有协议） |

工作表顺序：前面一小段按块号升序（覆盖交换窗口，保证到达即可算），之后按 causal 工作量降序（LPT，压住尾部）。交换窗口 ≈ 465 µs，32K attention ≈ 2.1 ms，所以绝大部分 item 领取时数据已到。

资源：一个 kernel 只有一种 block 大小，取 512 线程——GemmA2A / A2AGemm 本来就是 512 线程（`common.BLOCK_THREADS`），attention 保持 GQA merge（512 线程）的生产配置。（作为对照：关掉 GQA merge 的 256 线程 attention 单卡前向 16K +1.3%、32K +5.2%，输出逐位相同。）VGPR / LDS 取各角色最大值（attention 约 68 KB LDS，每 CU 2 个 block）。

死锁：GEMM block 不等任何东西，先派发；pull / push 是少量 persistent block，派发在 worker 之前；worker 只等 pull / push 和远端 worker 产出的数据，而远端的同角色 block 同样已驻留。

## 4. 反向

反向的 attention 是一组 kernel（`flash_attn_bwd.py`：odo 预处理、主 kernel、dq 归约），先把它们的依赖关系理清，再决定单 kernel 内怎么排。数据流：o_proj dgrad（GemmA2A）按块送来 dO → attention bwd 按块消费 → dq / dk / dv 按块 push 给 QKV dgrad（A2AGemm）。causal 反向里 dk / dv 依赖所有更靠后的 q-block，消费顺序与前向相反，条带切分下同样能流水，细节在前向完成后定。

## 5. 推进顺序（每步一轮，按 `.cursor/rules/iteration_rules.mdc`）

1. 条带切分：GemmA2A pull / A2AGemm 的行映射 + 测试，单独一轮，性能应不变。
2. 前向 v1：QKV GEMM + pull + attention worker 一个 kernel，O 照常写出，o_proj 仍用 A2AGemm。
3. 前向 v2：push + o_proj 并入。
4. 前向调优：工作表顺序、pull / push CU 数、512 线程（GEMM / pull 拆半 block，拿回 GQA merge 的 1–5%）。
5. 反向，同样的步骤。
6. 里程碑：e2e 训练（Primus，Qwen3-30B-A3B CP4，16K / 32K）。

约束：不用 CUDA graph；op 返回后无常驻 kernel；无全局 barrier、无 D2H 同步；每个自旋有超时诊断；容忍 rank skew；结果按位确定。
