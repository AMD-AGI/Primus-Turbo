# Ulysses GemmA2A / A2AGemm 融合 kernel 设计说明（第一阶段）

范围：bf16，单机 XGMI，gfx950，n ∈ {2,4,8}，前向 + 反向，sbhd / bshd，d ∈ {64,128}，hq、hkv 均被 n 整除。
记号：T = b·s/n 是本 rank 的 token 数（GEMM 的 M），N_r = (hq+2hkv)/n·d 是每个 rank 拿到的 QKV 列数，K_r = hq/n·d 是 o_proj 每个来源 rank 对应的 K 切片宽度。

## 1. 总原则

1. **交给 autograd 和 attention 的张量全部是普通 caching-allocator 张量**（q/k/v、dO、保存给 wgrad 的 O_gathered）。对称内存只放**单次调用内用完的临时数据**（staging / recv 缓冲 + flag）。这样 attention 读的永远是普通 coarse-grained 显存，`save_for_backward` 也不受对称缓冲复用的影响，CUDA graph 的内存池照常工作。对称堆沿用 `SymmetricMemory`（`hipMalloc`，coarse-grained，不用 uncached）。
2. **没有全局 barrier**：只用点对点的逐块 flag，flag 的值是单调递增的 epoch。
3. **op 返回时，本 rank 上不会留下任何常驻或自旋的 kernel**。每个融合 kernel 要等本 rank 需要的数据全部到齐才退出，然后就结束了。
4. 所有元数据（tile→rank 映射、拷贝计划）都在 host 上由形状静态算出并缓存。全程没有 device-to-host 同步。

## 2. push 还是 pull

| 融合 | 模式 | 原因 |
|---|---|---|
| GemmA2A（前向 QKV，反向 o_proj dgrad） | **pull** | GEMM epilogue 原样写到本地对称 staging，每个 tile 写完后置一个本地 flag。接收方的通信 CU 轮询远端 flag，从 peer 的 staging 读出数据，**直接按 a2a 之后的布局写进普通张量**。push 的话目的地只能是对称内存，还得再拷一次到普通张量（相当于又有一次 unpack）。 |
| A2AGemm（前向 o_proj，反向 QKV dgrad+wgrad） | **push** | 数据源（attention 的 O、dq/dk/dv）是 flash-attn 输出的普通张量，peer 拿不到它的地址。发送方的通信 CU **按 flash-attn 的原生 strided 布局原地读取**（所以没有 post_pack），写进目标 rank 的对称 recv 缓冲，然后对目标 rank 上的 flag 做 atomic。 |

实测（`benchmark/ops/training/bench_xgmi_read_write.py`，8 卡 all-to-all，每个 rank 进出的 GB/s；用的是 Triton 拷贝 kernel，每个 lane 一次只有一条 load 在飞）：

| CU 数 | push（6 MiB/peer） | pull（6 MiB/peer） | push（24 MiB/peer） | pull（24 MiB/peer） |
|---|---|---|---|---|
| 14 | 202 | 136 | 214 | 141 |
| 28 | 282 | 254 | 333 | 270 |
| 56 | 353 | 352 | 368 | 379 |
| 112 | 350 | 343 | 371 | 370 |

结论：
- 用到 56 个 CU 时，读和写都在 350–380 GB/s 饱和，两者打平。所以 GemmA2A 用 pull 可以成立。
- CU 少的时候 pull 吃亏，因为远端读的延迟更高。FlyDSL 实现里每个 lane 要让多条 load 同时在飞，把这部分差距补回来；`num_comm_cu` 的 autotune 范围定为 {24, 32, 48, 64}。
- push 到对称 recv、再由接收方拷出的写法保留为备选。
- 按这个带宽算，前向 QKV 侧每个 rank 发出 42 MiB，纯传输约 120 µs，和 GEMM 本身的时长在同一个量级。所以 CU 划分（通信 block 占多少 CU）是性能的主要旋钮。
pull 模式下，慢的 peer 可能在本卡跑 attention 时还在读本卡显存。单个 peer 最多读 T·N_r·2 B（Llama3.1-8B 32K 时约 6 MiB）。这个影响会用人为注入的 skew 量化，并报告 attention 的耗时变化。

## 3. tile → rank 映射

- **QKV 权重布局**：
  - Megatron 交错布局 `[ng, hq/ng+2, d]`：rank r 的 head 正好是第 `[r·ng/n, (r+1)·ng/n)` 个 query group，**在原始行序下就是一段连续的行**，所以不用重排。
  - concat 布局 `[q; k; v]`（benchmark 用的）：按 `[rank][q_r, k_r, v_r]` 重排一次，以 `(data_ptr, _version)` 为 key 缓存，权重更新后才重算。
  - 两种布局下 GEMM 都是普通 dense GEMM，staging 的列按 rank 排列（rank r 占 `[r·N_r, (r+1)·N_r)`）。
- **一个 N tile 跨两个 rank 也没关系**（比如 Wan 的 N_r=1920 不是 256 的整数倍）：pull 方只等覆盖自己列区间的那几个 tile 的 flag。pull 拷贝时按 constexpr 规则把列映射到 (q/k/v, local head, dd)，再把行映射到目的位置：sbhd 时每个 src 的数据是一整段连续行，bshd 时每个 (src, b) 是一段连续行。序列按 rank 连续切分，和 `AttentionCPA2AHelper` 一致。
- **A2AGemm 的 recv 缓冲**是行主序的 `[T, n·K_r]`，src 对应列块 src，**它本身就是 GEMM 的 A 矩阵**，本地切片也拷进来（1/n 的数据量）。GEMM 沿 K 按 src 切片，K 循环在每个切片开头等 flag `(src, m_tile)`。这需要给 `dense_mma_pipeline_bf16` 加一个 constexpr 位置的 K-slice wait hook，并放在该切片的第一次 prefetch 之前。
- **wgrad 都在融合 kernel 里，不用 side stream**（MI355X 上多 stream 重叠不可控）。tile 用生产的 FlyDSL MoE wgrad tile `grouped_gemm_bf16_variable_k_tile`（K 循环按 256 行分块，运行时循环），加了三个可选参数：`a_base`（A 用地址给出）、`k_hook`（每个 K 块的 A 读之前调用）、`persistent="staggered"`。
  - 反向 QKV：`dW[src 的行] = dQKV_src^T @ x` 按 src 天然可分，不需要跨 rank 归约。A2AGemm 的 tile 计数器上排 `K/256 × N/256` 个 wgrad tile，直接读 recv bank，每个 K 块只等它读到的 m-tile 里本 tile 那个 src 的 flag，沿 M 固定顺序累加，结果确定。gathered dqkv 不再拷出。wgrad tile 默认排在最前、16 个 push block；超过两波（2 × CU 数，GLM-4.7 的 1120）时改为 BM=256、32 个 push block、排在全部 dgrad tile 之后，否则 dgrad 要等几波 wgrad 领完，与交换串行。
  - 反向 o_proj：`dW_o = dY^T @ O_gathered` 不依赖交换，GemmA2A 里每个 block 一个 wgrad tile，派发在第 `num_lead + GEMM tile 数 / 8` 个 GEMM tile 之后。wgrad tile 少时排在全部 GEMM tile 之后会挡住尾部 pull（Qwen3-30B 32K 503 µs，对比 419 µs）；超过 CU 数（GLM-4.7 960、MiniMax-M2.5 288）时反过来，插在中间会把后面的 GEMM tile 推迟几波，所以排在最后。
  - 维度不是 256 的倍数时回退为 kernel 之后单独跑 wgrad。
  - 试过但回滚的做法：o_proj wgrad 用 mega `tn` tile 放进 grid（K 循环完全展开，kernel 内每 tile 比 variable-K tile 慢约 3 倍）；QKV wgrad 按 M 分块放 side stream、用 `hipStreamWaitValue64` 等待，并发 GEMM 拖慢交换（16K 403 → 457 µs）。
- **前向 O_gathered 的保存**：接收方的通信 CU 推完自己的数据后，把已经到达的切片从 recv 缓冲拷到普通张量，和 GEMM 重叠。

## 4. flag / parity 协议

- 每个 group 有一个设备端 epoch 计数器，**四个 op 共用**。每个 kernel 开始时读出 epoch e，使用 bank `e & 1`。最后一个退出的 block（exit counter 判断）把 epoch 写成 e+1，并把 exit counter 复位。这样不需要 host 参与，也不需要额外的 bump kernel，CUDA graph replay 也成立。
- flag 是 i64，写入值就是 e，等待条件是 `flag >= e`。flag 不复位、不分 bank，每种 op 有自己独立的 flag 区。
  - pull：flag 在**发送方本地**，接收方远端轮询。
  - push：flag 在**接收方**，由发送方 atomic 写入。
- **为什么双缓冲足够，不需要 ack 或 barrier**：rank 的第 e 次调用必须拿到每个 peer 在第 e 次调用里才产生的数据才能结束。所以当我进入第 e+2 次调用时，每个 peer 都已经开始了第 e+1 次调用，也就已经做完了第 e 次调用（包括它从我这里 pull 的读取）。因此 bank `e & 1` 可以安全覆盖。前提是**同一个 group 上的这些 op 在每个 rank 上都按 stream 顺序执行**（同一个 stream，或者用 event 串起来），文档和断言里都会写明。
- 内存序：
  - staging 和远端 push 用 `sc0|sc1|nt` 写，然后 `s_waitcnt`，再做 sys 作用域的 release，最后写 flag。
  - 读方先 acquire flag，再用绕过 L2 的 modifier 读数据。因为 recv/staging 每两个 epoch 复用一次，本卡 L2 里可能还有旧行，这样读能避开它们。
  - 这些都沿用 MegaMoE `ep_intranode.py` 的约定，并用“每个 epoch 之间往缓冲里写毒值”的压力测试验证。
- 所有自旋都用 `spin_timed_out`：超时后打印 rank、op、epoch、src、tile、flag 当前值和期望值，然后重新计时继续等。rank 之间的进度差只表现为等待时间，不会出错。

## 5. CU 划分（参照 `_block_role`）

grid 的排列顺序是：`num_lead` 个 GEMM block（默认 = CU 数，即第一个 wave）→ `num_pull_cu` 个 pull block（默认 16）→ 其余 GEMM block → `num_tail_pull` 个 tail pull block（默认 16）。
- 第一个 wave 结束之前没有可拷的数据，所以 pull block 排在第一个 wave 之后，等第一个 wave 退场再上 CU。tail block 排在所有 GEMM block 之后：它们开始运行时每个 GEMM tile 都已派发，占用 CU 也不会饿死自己要等的 tile。只要硬件按序派发 block，就不会死锁。
- pull unit 用 `state[2]` 原子计数器动态领取（领到的编号经一个 LDS 槽位广播给整个 block），按 `(m_tile, src=(me+i)%n)` 顺序，即就绪顺序。拷贝和顺序无关，不影响确定性。
- 实测（8B 32K n=8，`trace` 时间线）：第一个 wave 约在 100 µs 结束，pull 合计约 350–390 GB/s，已到 XGMI 读带宽上限。pull 并发再加大会拖慢对端的 GEMM：tail=48 或 pull=64 都变慢。下限约为“第一个 wave + 44 MB / 350 GB/s ≈ 230–245 µs”，实测约 281 µs。
- 试过但没有收益、已经移除的：XCD remap / GROUP_M，epilogue cache modifier 0/18/19，main 上的 dense tile（核心与 mega 快照相同），把本地列 tile 放到最后算，让最后一个 wave 的 GEMM block 跑完后去 pull（还会死锁）。
- GemmA2A：GEMM 按 m-major 光栅化，数据按 m 逐步就绪。pull block 的块序是 `(m_tile, src=(me+i)%n)`，让各 rank 在同一时刻从不同的 src 读，7 条链路都用上。
- A2AGemm（`a2a_gemm_kernel.py`）：前 `num_push`（默认 32）个 block 先动态领取 push unit `(m_tile, dst=(me+i)%n)`，把 dst 那段序列的行写进 dst 的 recv bank 第 `me` 个列块，再置 dst 的 flag `m_tile*n + me`；领完后这些 block 去算最后几个 tile（它们的行反正最晚到齐）。每个 GEMM tile 等齐本 m_tile 所有来源的 flag，做一次 system 级 acquire，然后跑普通的 K 循环。
  - 源数据可以是一个张量（o_proj：`G=1`）或 `(dq, dk, dv)` 三个张量，按权重行序（`G` 组，每组 `(nq, nk, nv)/G` 个头）写进 recv bank，所以 QKV dgrad 直接用原权重（concat 布局用 rank-major 重排后的权重），B 走 `nn` 布局。
  - 前向 o_proj 需要给反向留 O 时，每个 m_tile 的 n_tiles 个 block 各把 recv 行的一个列块拷到 `o_gathered`。反向 QKV 的 wgrad tile 直接读 recv bank（第 3 节）。
  - residual 在 epilogue 里按 fp32 加到累加器上，只做一次 bf16 舍入（`_residual_store_c`，要求 N % 256 == 0）。
  - `num_push` 扫过 16/32/48/64/96/128：o_proj 和 QKV dgrad 都是 32–48 最好。
- tile 形状固定 256×256，没有做 autotune（见遗留问题）。

## 6. 确定性

- GemmA2A 只有 GEMM 和纯拷贝，结果按位确定。
- A2AGemm 的 K 沿 src 排列（recv bank 的列块 = 来源 rank），每个 tile 等齐所有来源后按固定 K 顺序累加，结果按位确定。
  - 没有实现按到达顺序累加的 `deterministic=False`：实测（第 10 节）各阶段在关键 rank 上已不含等待，按到达顺序只能缩短非关键 rank 的等待，不缩短整层。
- wgrad 不跨 rank 归约，结果总是确定的。

## 7. 缓冲区生命周期和大小

- 对称堆的大小是 2 个 bank × max(T·(hq+2hkv)·d, T·hq·d) × 2 B，再加上 flag。
  - Llama3.1-8B、32K、n=8：每个 bank 48 MiB，共 96 MiB。
  - Llama3.1-70B、128K、n=8：每个 bank 320 MiB，共 640 MiB。
- 第一次（eager）调用时分配；之后需要扩容时，如果正处在 graph capture 中就直接报错（和 `get_symm_mem_workspace` 的行为一样）。
- 跨层、跨 micro-batch 复用由第 4 节的 epoch 和 bank 保证。

## 8. API 和 autograd

- `ulysses_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout="interleaved"|"concat")`：返回 `q, k, v`，形状为 `[s, b, h/n, d]`（sbhd）或 `[b, s, h/n, d]`（bshd）。
  - 反向：A2AGemm（dx 和 dW 在同一个 kernel 里）。
- `ulysses_a2a_o_gemm(o, w_o, residual, group, qkv_format)`：返回 y。
  - 反向：GemmA2A（dO 和用保存的 O_gathered 算的 dW_o 在同一个 kernel 里），`d_residual = dY`。
- 现有的 `flash_attn_usp_func` 不改。

## 9. 已确认的决策

- A2AGemm 按固定顺序累加（按位确定）。
- 两种权重布局都支持。A/B benchmark 的 baseline 和融合路径都用 Megatron 交错布局。
- wgrad 返回普通 bf16 `.grad`。`main_grad`（gradient_accumulation_fusion）放到遗留问题。
- 对称堆按第 7 节的整份双 bank 分配。
- A/B benchmark 从 Primus 复制到 `benchmark/ops/training/bench_ulysses_a2a.py`，在 Turbo 里改。

## 10. 结果

测量：`benchmark/ops/training/bench_ulysses_a2a.py --fused`，一层 attention + dense MLP 的 forward + backward，CUDA event 逐阶段计时，每个 rank 取 median。每轮开始前在 GPU 上排一个 `_sleep`，把 host 下发时间藏在后面（训练里前一层的 kernel 起同样作用）。

各 rank 的阶段时间差别主要来自等待：本机 GPU 2 和 GPU 4 的 attention / MLP 固定慢 5–10%，其他 rank 在下一次交换处等它们，所以“各 rank 平均”和“最大 rank”里都含有等待，unfused 的 all-to-all 和 fused kernel 一样。表里的“关键 rank”是该 phase 计算（attention + MLP）最多的 rank：别人等它，它自己的交换阶段不含等待，这一列才是 kernel 在关键路径上的开销。

<!-- RESULTS -->

### 端到端训练

Primus / Megatron 预训练 Qwen3-30B-A3B，1×8 MI355X，TP=1，CP=4（Ulysses），EP=8，DP=2，micro batch 1，GBS 16，mock 数据，第 3–10 步平均 ms/iter。接入方式是 Primus 的 `ulysses_fused_a2a_patches.py`（`PRIMUS_ULYSSES_FUSED_A2A=1`）：CP 切分改为 contiguous，RoPE 放到 a2a 之后按全长序列做，attention 用本地 `flash_attn_func`。`serial` 是同一路径但 GEMM 和 RCCL all-to-all 串行（`PRIMUS_ULYSSES_FUSED_A2A_IMPL=serial`）。三者 loss 在 1e-3 内一致。

| 长度 | base（Turbo `flash_attn_usp_func`） | serial | fused | fused 对 base | fused 对 serial |
|---|---|---|---|---|---|
| 16K | 3147 | 3094 | 2897 | -7.9% | -6.4% |
| 32K | 6583 | 6534 | 6103 | -7.3% | -6.6% |

## 11. 遗留问题

- `main_grad` / gradient_accumulation_fusion：wgrad 返回普通 bf16 `.grad`，没有融合 fp32 累加。
- tile 形状固定 256×256，没有 autotune；GEMM 核心（mega bf16 tile）在同形状上比 hipBLASLt 慢约 15%（例如 8B 32K QKV dgrad 单独 GEMM：hipBLASLt 173 µs）。
- Qwen3-30B-A3B（hidden 2048，n=4）交换量相对 GEMM 很大，受 XGMI 链路带宽限制（约 131 GB/s 每 rank），加速主要来自去掉 pack/unpack 和重叠。
- GemmA2A 的调试开关“跳过 GEMM、只 pull”在 n=8、M=4096、leading pull=16 时会挂住，不是 flag 自旋（超时诊断没有触发）；原因没有查清，开关已删除，正常路径不受影响。
- FlyDSL 磁盘缓存的 key 只哈希 launcher 可达函数的源码，不含类体：改了 epilogue 类而缓存没失效，曾经加载到旧 kernel。现在把类定义放在函数里；建议在 FlyDSL 上游修。
- 传给 FlyDSL kernel 的 4 维张量（带 size 为 1 的维）出现过错误偏移，原因未查清；现在统一按 2/3 维传入。
- residual epilogue 只支持 N % 256 == 0（所有目标模型都满足），否则报错。
- 限制：`M_pad*K`、`M*N`、`K*N` < 2^31（i32 偏移），`m_tiles*n` ≤ 32768（一个 flag region）。
- CUDA graph 不在支持范围内（不测试）。代码里保留了 capture 时不扩容 workspace、launcher 走原始路径的处理，但没有验证。
- concat 权重布局：前向用缓存的 rank-major 重排权重（多占一份权重显存），dW 再 scatter 回 `[q; k; v]`。
