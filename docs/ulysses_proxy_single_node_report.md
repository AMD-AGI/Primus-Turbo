# Ulysses GemmA2A / A2AGemm：多模型单节点 proxy 端到端测试报告

任务描述见 `docs/ulysses_proxy_single_node_prompt.md`。测试在两台 8×MI355X 节点（`smci355-ccs-aus-n01-33`、`smci355-ccs-aus-n02-21`）上并行进行：每个组合的 base / serial / fused 都在同一台节点上串行跑完，测量期间这台节点上没有其他任务，所有对比都只在同一台节点内进行。所有运行都不用 CUDA graph。

## 结论摘要

- **发布主推：Qwen3-235B-A22B，CP4 / EP8，32K。** 4 层 proxy 实测 fused 比 base 快 5.7%，推算完整 94 层模型快 6.9%（16K 为 5.0%）。两种外推口径在这个模型上结果一致；同结构的 Qwen3-30B-A3B 有完整模型实测（16K 快 8.9%，32K 快 5.5%），并用来验证了外推方法。需要说明的是，Qwen3 的收益里有 0.3–1.7 个百分点来自 patch 路径顺带换用的 FlyDSL attention，不来自 GemmA2A / A2AGemm 本身（见第 3 节）。
- **第二案例：GLM-4.7，CP4 / EP8。** 推算完整模型 32K 快 6.6%（16K 为 5.1%），每层节省在所有模型里最大（32K 每层每 micro-batch 3.6 ms）。GLM 两条路径用的都是 AITER attention，收益全部来自融合 kernel。MiniMax-M2.5 推算 16K / 32K 分别快 6.2% / 6.3%，partial RoPE 加 qk norm 的路径正确。
- **不建议主推 Llama3.1-8B。** 在默认配置下 16K 反而比 base 慢 9.0%。原因是 patch 路径的 `flash_attn_func` 默认选 FlyDSL attention，而它在 Llama 这个形状（每 rank 8 个 q 头 / 2 个 kv 头，16K）上比 AITER 慢 19%。强制 AITER 后 16K / 32K 分别快 4.0% / 4.1%。
- **CP=8：kernel 参数值得调，但收益上限只有约 1–2 个百分点，不建议作为发布前提。** GLM CP8 推算完整模型只快 0.5–3.9%（16K）和 3.5–3.8%（32K）。CP=8 时 RCCL all-to-all 的带宽约为 CP=4 的 2 倍，未融合的基线本身变快了；GLM 的融合 kernel 只消掉了 30–40% 的交换时间（CP4 下各模型为 46–59%）。其中 A2AGemm（前向 a2a + o_proj）在 n=8、16K 下比不融合更慢（GLM 0.93×，Llama 0.80×），这是一个确定的缺陷，应优先修复。
- **没有结构回退。** 所有 fused / serial 运行都打印了 `[Patch] ✓ Applied: megatron.ulysses_fused_a2a`，没有 `keeping the stock path`，probe 也确认实际调用了融合（或 serial）kernel。
- **数值。** 唯一超出 1e-3 量级的是 Llama3.1-8B 16K：第 12 步 loss 差 5.3e-2。对照实验表明，这是训练动态把 bf16 噪声放大了，不是 patch 算错：base 自己只改 CP 大小，第 12 步就会差 0.11–0.14（见 6.3 节）。
- **测试中发现并修复了一个 patch bug。** 开启 `overlap_param_gather` 时，fused 路径绕过了 `linear_qkv` / `linear_proj` 的 forward pre-hook，DDP 不会等待权重的 all-gather 完成。GLM 在第 3 步会 assert 崩溃；Qwen 不崩溃，但存在读到未更新权重的竞争。已在 Primus `feat/ulysses-fused-a2a` 修复（f9a298a49），本报告的数据全部用修复后的代码测得。

## 1. 测试设置

| 项 | 值 |
|---|---|
| 并行 | TP=1，PP=1，`micro_batch_size=1`，`global_batch_size=16`，mock 数据，12 步，16K / 32K |
| 训练配置 | `examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-cp4-pretrain.yaml`，模型由 `PRIMUS_MODEL` 选，CP / EP 由 `PRIMUS_CP` / `PRIMUS_EP` 选 |
| 每个 DP rank 的 micro-batch 数 | CP4：DP=2，8 个；CP8：DP=1，16 个 |
| 指标 | 第 12 步日志 `elapsed time per iteration (ms)` 的平均值（排除前 2 步），以及 `TFLOP/s/GPU`、`tokens/s/GPU`（harmonic mean）、`hip mem usage`、第 12 步 `lm loss` |
| 镜像 | `tasimage/primus:pr-1275`（节点上已有的 `xiaoming-dev` 容器；本地没有 prompt 指定的 `rocm/primus:v26.8`），torch 2.14.0+rocm10.1.0，`flydsl==0.3.4.1` |
| Turbo | `feat/ulysses-gemm-a2a` @ 60687e12。`primus_turbo/`、`csrc/` 无本地修改，base / serial / fused 用同一份。训练进程通过 `sitecustomize.py` 把这份 Turbo 放到 `sys.path` 最前，并用 probe 确认加载的是它 |
| Primus | `feat/ulysses-fused-a2a` @ f9a298a49，Megatron-LM 子模块 4b4acac9 |

### 每次运行的检查

`check.py` 对每个 (组合, 长度) 的三次运行检查以下各项，任何一项不过都会停下后面的组合：

- 跑满 12 步，没有 NaN 或跳过的步；
- 打印出的模型参数（`num_layers`、`hidden_size`、`num_attention_heads`、`num_query_groups`、`num_experts`、`rotary_percent`、`qk_layernorm`，以及 `add_qkv_bias`、`add_bias_linear`、`multi_latent_attention`、`attention_output_gate`）和预期一致；
- CP、EP、序列长度正确；
- base 没有打 patch，fused / serial 打了 patch，并且没有 `keeping the stock path`；
- probe 显示 fused 调用的是 `ulysses_qkv_gemm_a2a` / `ulysses_a2a_o_gemm`，serial 调用的是 `_serial_qkv_gemm_a2a` / `_serial_a2a_o_gemm`，都没有调用 `flash_attn_usp_func`；
- 三种实现第 12 步的 loss 差不超过 5e-3。

36 次运行中，只有 Llama3.1-8B 16K 因 loss 差 5.3e-2 被判停，6.3 节给出了解释，之后继续跑完了剩余组合。

### 节点分配

| 节点 | 主矩阵 | 补充运行 |
|---|---|---|
| n01-33 | q30b、glm_cp4、mm25、l8b 32K | q30b 4 层 / 8 层，glm_cp4 8 层，mm25 8 层；l8b 32K、q30b 32K 的 AITER 隔离 |
| n02-21 | q235、glm_cp8、l8b 16K | q235 8 层，glm_cp8 8 层；l8b 对照与消融；l8b 16K、q30b 16K（含同节点 base 重跑）、q235 16K / 32K 的 AITER 隔离；微基准 |

### 测试中修复的 patch bug

第一轮矩阵中，GLM-4.7 的 serial / fused 在第 3 步崩溃：`assert self.param_gather_handle is None`（`param_and_grad_buffer.py:400`）。

原因如下。开启 `overlap_param_gather` 时，Megatron DDP 会在每个带参数模块的 forward pre-hook 里，等待这个模块权重的 all-gather 完成（`finish_param_sync`）。fused forward 直接读取 `self.linear_qkv.weight` / `self.linear_proj.weight`，不经过模块调用，所以这两个 hook 从未触发。

修复方法是在读取权重之前，按 `nn.Module.__call__` 的方式手动执行这两个模块的 forward pre-hook（Primus f9a298a49）。修复后，GLM CP4 fused 16K 的 loss 是 11.13924，修复前 base 是 11.13908；GLM CP8 serial 16K 是 11.12283，base 是 11.12460。

修复前 Qwen3 没有崩溃，但存在同样的竞争。修复前的 q30b / q235 serial / fused 数据（保存在 `logs_v1_prefix/`）和修复后相差 0.7% 以内，全部已用修复后的代码重跑。

## 2. 结果总表

ms/iter 为每步平均耗时；显存列是 `hip mem usage`（GB）；"b / s / f"依次表示 base / serial / fused。

| 模型 | CP / EP | 长度 | base ms | serial ms | fused ms | fused/base−1 | fused/serial−1 | serial/base−1 | TFLOP/s/GPU b / s / f | tokens/s/GPU b / s / f | 显存 GB b / s / f | 第 12 步 loss b / s / f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B（完整 48 层） | 4 / 8 | 16K | 3011.7 | 2940.1 | 2744.7 | **-8.9%** | -6.6% | -2.4% | 408 / 418 / 448 | 10880 / 11145 / 11939 | 83.4 / 83.3 / 83.3 | 11.6226 / 11.6242 / 11.6242 |
| Qwen3-30B-A3B（完整 48 层） | 4 / 8 | 32K | 6179.1 | 6213.6 | 5837.2 | **-5.5%** | -6.1% | +0.6% | 603 / 599 / 638 | 10606 / 10547 / 11227 | 109.2 / 107.6 / 107.7 | 11.6143 / 11.6124 / 11.6123 |
| Qwen3-235B-A22B proxy | 4 / 8 | 16K | 598.4 | 605.1 | 574.9 | **-3.9%** | -5.0% | +1.1% | 673 / 666 / 701 | 54764 / 54154 / 56997 | 43.1 / 42.8 / 42.8 | 11.3108 / 11.3106 / 11.3106 |
| Qwen3-235B-A22B proxy | 4 / 8 | 32K | 1252.5 | 1252.6 | 1181.2 | **-5.7%** | -5.7% | +0.0% | 812 / 812 / 861 | 52324 / 52320 / 55484 | 49.3 / 49.3 / 48.4 | 11.3049 / 11.3045 / 11.3045 |
| GLM-4.7 proxy | 4 / 8 | 16K | 1029.0 | 1023.7 | 984.2 | **-4.4%** | -3.9% | -0.5% | 569 / 572 / 595 | 31846 / 32009 / 33293 | 56.0 / 55.8 / 56.0 | 11.1391 / 11.1392 / 11.1392 |
| GLM-4.7 proxy | 4 / 8 | 32K | 2009.1 | 2001.1 | 1893.6 | **-5.7%** | -5.4% | -0.4% | 740 / 743 / 785 | 32619 / 32750 / 34610 | 62.4 / 62.4 / 62.8 | 11.1342 / 11.1339 / 11.1339 |
| GLM-4.7 proxy | 8 / 8 | 16K | 1370.5 | 1373.3 | 1364.9 | **-0.4%** | -0.6% | +0.2% | 427 / 426 / 429 | 23910 / 23862 / 24007 | 53.6 / 53.3 / 53.1 | 11.1246 / 11.1228 / 11.1228 |
| GLM-4.7 proxy | 8 / 8 | 32K | 2335.8 | 2324.2 | 2264.2 | **-3.1%** | -2.6% | -0.5% | 637 / 640 / 657 | 28057 / 28197 / 28945 | 57.4 / 57.2 / 57.1 | 11.1356 / 11.1352 / 11.1352 |
| MiniMax-M2.5 proxy | 4 / 8 | 16K | 590.3 | 592.2 | 561.6 | **-4.9%** | -5.2% | +0.3% | 548 / 547 / 576 | 55513 / 55332 / 58343 | 52.2 / 52.0 / 52.0 | 11.7381 / 11.7379 / 11.7379 |
| MiniMax-M2.5 proxy | 4 / 8 | 32K | 1115.4 | 1122.6 | 1059.4 | **-5.0%** | -5.6% | +0.6% | 722 / 718 / 761 | 58756 / 58378 / 61860 | 57.3 / 57.0 / 57.2 | 11.7303 / 11.7301 / 11.7301 |
| Llama3.1-8B（完整 32 层） | 4 / 1 | 16K | 2153.4 | 2493.2 | 2347.2 | **+9.0%** | -5.9% | +15.8% | 881 / 761 / 808 | 15217 / 13143 / 13960 | 72.3 / 71.3 / 70.9 | 7.8107 / 7.8633 / 7.8637 |
| Llama3.1-8B（完整 32 层） | 4 / 1 | 32K | 4785.1 | 4886.2 | 4607.9 | **-3.7%** | -5.7% | +2.1% | 970 / 950 / 1007 | 13696 / 13412 / 14223 | 95.0 / 93.4 / 93.9 | 7.1292 / 7.1281 / 7.1287 |

- **没有 OOM 或失败。** 三种实现的显存最多相差 1.6 GB；fused 最多比 base 多 0.4 GB（GLM CP4 32K），其余组合不多于 base。
- **loss。** 除 Llama 16K 外，base 和 patch 路径之间第 12 步的最大差是 2.0e-3（q30b 32K），其次是 GLM CP8 16K（1.8e-3）和 q30b 16K（1.7e-3），其余都不超过 1.1e-3。serial 和 fused 的数据切分相同，两者最大只差 6e-4（Llama 32K），MoE 模型上不超过 2e-4。
- **run-to-run 噪声。** 修复前后各跑了一次的 base（代码完全相同）：CP4 的组合相差都在 0.5% 以内（q30b 16K 0.05%、q235 16K 0.42%、GLM CP4 16K 0.06%）；GLM CP8 16K 相差 1.6%（1393.0 对 1370.5 ms）。所以 GLM CP8 16K 的 −0.4% 在噪声范围内。
- **和 prompt 参考数据（v26.8 镜像）对比。** 本次镜像下三种实现都更快，其中 base 提升更多，所以相对收益偏低：q30b 16K 3012 / 2940 / 2745 对参考 3147 / 3094 / 2897，fused 比 base 快 8.9%（参考为 7.9%）；q30b 32K 6179 / 6214 / 5837 对 6583 / 6534 / 6103，快 5.5%（参考为 7.3%）；q235 proxy 16K 598 / 605 / 575 对 634 / 632 / 597，快 3.9%（参考为 5.8%）。

## 3. attention backend 对收益拆分的影响

prompt 把 `serial / base − 1` 定义为"路径重构（contiguous 切分 + 本地 flash-attn）的收益"。实际上，这一步在一部分模型上同时更换了 attention kernel：

- base 的 `flash_attn_usp_func` 固定调用 AITER（`attention_aiter_forward_impl` / `attention_aiter_backward_impl`）；
- patch 路径调用 `flash_attn_func`，它在 MI355X 上默认优先选 FlyDSL，不满足条件时才回退到 AITER。FlyDSL 的条件之一是 GQA 分组数必须是 2 的幂。

| 模型 | hq / hkv | patch 路径的 attention | base 的 attention |
|---|---|---|---|
| Qwen3-30B-A3B / Qwen3-235B-A22B | 8 / 16 | FlyDSL | AITER |
| Llama3.1-8B | 4 | FlyDSL | AITER |
| GLM-4.7 | 12 | AITER（回退） | AITER |
| MiniMax-M2.5 | 6 | AITER（回退） | AITER |

单卡 `flash_attn_func` 前向加反向，用的是每个 rank 的实际形状（全长序列、本地头数，sbhd 存储），数据见 `logs_bench/attn.log`：

| 每 rank 形状（q / kv 头） | 16K FlyDSL / AITER ms | 32K FlyDSL / AITER ms |
|---|---|---|
| Qwen3-30B CP4（8 / 1） | 1.93 / 2.31（0.84×） | 7.04 / 8.81（0.80×） |
| Qwen3-235B CP4（16 / 1） | 3.68 / 4.59（0.80×） | 13.75 / 17.39（0.79×） |
| Llama3.1-8B CP4（8 / 2） | **2.78 / 2.33（1.19×）** | 7.18 / 8.83（0.81×） |

为了把融合本身的收益分离出来，我用 `PRIMUS_TURBO_ATTN_BACKEND=AITER` 让 patch 路径也走 AITER，重跑了 serial 和 fused。这个开关不影响 base，因为 base 本来就固定用 AITER。q30b 16K 的 AITER 列用的是在同一台节点（n02-21）上重跑的 base（3036.1 ms），默认列用矩阵里 n01-33 的 base（3011.7 ms）。

| 模型 | 长度 | base ms | serial ms（默认） | fused ms（默认） | serial ms（AITER） | fused ms（AITER） | fused/base−1（默认） | fused/base−1（AITER） | fused/serial−1（AITER） | serial/base−1（AITER） |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama3.1-8B | 16K | 2153.4 | 2493.2 | 2347.2 | 2200.2 | 2066.3 | +9.0% | **-4.0%** | -6.1% | +2.2% |
| Llama3.1-8B | 32K | 4785.1 | 4886.2 | 4607.9 | 4854.8 | 4590.2 | -3.7% | **-4.1%** | -5.5% | +1.5% |
| Qwen3-30B-A3B | 16K | 3011.7 / 3036.1 | 2940.1 | 2744.7 | 3050.8 | 2817.2 | -8.9% | **-7.2%** | -7.7% | +0.5% |
| Qwen3-30B-A3B | 32K | 6179.1 | 6213.6 | 5837.2 | 6244.2 | 5854.8 | -5.5% | **-5.2%** | -6.2% | +1.1% |
| Qwen3-235B-A22B proxy | 16K | 598.4 | 605.1 | 574.9 | 604.6 | 578.2 | -3.9% | **-3.4%** | -4.4% | +1.0% |
| Qwen3-235B-A22B proxy | 32K | 1252.5 | 1252.6 | 1181.2 | 1265.5 | 1192.5 | -5.7% | **-4.8%** | -5.8% | +1.0% |

从这张表可以得出：

- **Llama 16K 变慢完全来自 attention backend。** patch 路径换成 AITER 后，fused 比 base 快 4.0%。端到端里 FlyDSL 比 AITER 每步多花 293 ms，折合每层每 micro-batch 约 1.1 ms，比单卡测得的 0.45 ms 还大。这和 GemmA2A / A2AGemm 无关，是 Turbo attention dispatch 在这个形状上选错了 kernel。
- **Qwen 的默认收益里，有一部分来自 FlyDSL attention 更快。** q30b 16K 为 1.7 个百分点，q235 proxy 32K 为 0.9 个百分点。
- **在相同 attention 下，路径重构本身略有开销（+0.5% 到 +2.2%）。** 这部分没有逐项拆开测。和 base 相比，patch 路径的不同之处有：a2a 之后在全长序列上做 RoPE；input RMSNorm 单独计算（已测得开销很小，见 6.4 节）；`linear_qkv` / `linear_proj` 的 wgrad 不走 TE 的融合累加。融合 kernel 本身（`fused/serial−1`）贡献了 4.4–7.7%。
- 对 GLM 和 MiniMax 而言，两条路径用的都是 AITER，第 2 节的 `serial/base−1`（−0.5% 到 +0.6%）就是纯粹的路径重构开销。

## 4. 每层每 micro-batch 节省与完整模型推算

### 方法

每层每 micro-batch 节省按 prompt 的公式计算：`s = (base − fused) / (层数 × 每个 DP rank 的 micro-batch 数)`，使用 4 层 proxy（q30b 和 Llama 是完整模型，直接用实测值）。完整模型每步节省 ≈ `s × 完整层数 × micro-batch 数`。

分母（完整模型 base 每步耗时）不能用"proxy 每步耗时 ÷ 4 × 完整层数"来近似：那样 embedding、输出层和优化器的固定开销也会跟着层数放大，推出来的相对收益恒等于 proxy 自己的收益。所以我对每个模型另外跑了 8 层的 base / fused（用 `--num_layers 8` 覆盖层数，其他配置不变），用 `T(L) = a + b·L` 两点拟合出固定开销 a 和每层耗时 b，再取 `a + b × 完整层数` 作为完整模型的 base。

用 Qwen3-30B-A3B 检验这个方法：用 4 层和 8 层的拟合去预测 48 层，并和实测的完整模型对比。

| 长度 | 实测 48 层 base / fused / 收益 | 拟合预测 48 层 base | 本节方法推算收益 | 参考：拟合斜率口径 | 4 层 proxy 自身收益 |
|---|---|---|---|---|---|
| 16K | 3012 / 2745 / -8.9% | 2966（−1.5%） | **-9.1%**（节省 269 ms，实测 267 ms） | -6.4% | -6.9% |
| 32K | 6179 / 5837 / -5.5% | 6183（+0.1%） | **-5.8%**（节省 360 ms，实测 342 ms） | -5.6% | -4.7% |

本节方法推算的收益和实测相差不超过 0.3 个百分点，而"拟合斜率口径"（用两个 fused 点也做拟合）在 16K 下低估了 2.5 个百分点。所以下表以本节方法为主，斜率口径仅供参考。两种口径的差别主要反映单次运行的噪声：斜率由两个相隔 4 层的点相减得到，对噪声更敏感。

### 结果

| 模型 | CP | 长度 | 每 DP rank micro-batch | proxy 实测收益 | 每层每 micro-batch 节省 ms | 完整层数 | 完整模型 base ms（拟合） | 推算每步节省 ms | **推算完整模型收益** | 参考：拟合斜率口径 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 4 | 16K | 8 | -6.9%（4 层） | 0.700 | 48 | 2966（实测 3012） | 269 | **-9.1%**（实测 -8.9%） | -6.4% |
| Qwen3-30B-A3B | 4 | 32K | 8 | -4.7%（4 层） | 0.938 | 48 | 6183（实测 6179） | 360 | **-5.8%**（实测 -5.5%） | -5.6% |
| Qwen3-235B-A22B | 4 | 16K | 8 | -3.9% | 0.734 | 94 | 11074 | 552 | **-5.0%** | -4.6% |
| Qwen3-235B-A22B | 4 | 32K | 8 | -5.7% | 2.228 | 94 | 24241 | 1676 | **-6.9%** | -6.8% |
| GLM-4.7 | 4 | 16K | 8 | -4.4% | 1.400 | 90 | 19930 | 1008 | **-5.1%** | -6.8% |
| GLM-4.7 | 4 | 32K | 8 | -5.7% | 3.609 | 90 | 39415 | 2599 | **-6.6%** | -8.4% |
| GLM-4.7 | 8 | 16K | 16 | -0.4% | 0.087 | 90 | 27188 | 126 | **-0.5%** | -3.9% |
| GLM-4.7 | 8 | 32K | 16 | -3.1% | 1.119 | 90 | 46331 | 1611 | **-3.5%** | -3.8% |
| MiniMax-M2.5 | 4 | 16K | 8 | -4.9% | 0.897 | 62 | 7121 | 445 | **-6.2%** | -5.1% |
| MiniMax-M2.5 | 4 | 32K | 8 | -5.0% | 1.750 | 62 | 13796 | 868 | **-6.3%** | -6.5% |
| Llama3.1-8B | 4 | 16K | 8 | +9.0%（完整模型） | −0.757（AITER：0.340） | 32 | 2153（实测） | −194（AITER：87） | **+9.0%**（实测；AITER -4.0%） | - |
| Llama3.1-8B | 4 | 32K | 8 | -3.7%（完整模型） | 0.692（AITER：0.761） | 32 | 4785（实测） | 177（AITER：195） | **-3.7%**（实测；AITER -4.1%） | - |

说明：

- 4 层和 8 层的原始数据见 `logs_layers/`。例如 q235 8 层 base / fused 为 1064.0 / 1019.1 ms（16K）、2274.2 / 2133.1 ms（32K）；GLM CP8 8 层为 2571.3 / 2517.2 ms（16K）、4382.1 / 4230.9 ms（32K）。
- GLM CP8 16K 的两种口径差得很远（−0.5% 对 −3.9%）：4 层 proxy 的收益只有 0.4%，而 8 层的收益是 2.1%，同时这一组 base 的 run-to-run 噪声就有 1.6%。所以 CP8 16K 的每层节省只能给出 0.09–0.76 ms 的区间。
- Qwen3 的推算值里包含了 FlyDSL attention 的贡献。只算融合（AITER 隔离）时，Qwen3-235B-A22B 推算完整模型为 16K −4.3%、32K −5.8%（每层每 micro-batch 节省 0.631 / 1.875 ms）。

和 prompt 中理论估算的对比（32K 完整模型）：

| 模型 | 理论估算 | 本次推算 / 实测 |
|---|---|---|
| Qwen3-235B-A22B | 约 7% | 6.9%（默认），5.8%（只算融合） |
| GLM-4.x CP4 | 约 8% | 6.6%（斜率口径 8.4%） |
| GLM-4.x CP8 | 约 9% | 3.5%（斜率口径 3.8%） |
| MiniMax-M2.5 | 约 7% | 6.3% |
| Llama3.1-8B | 约 9% | 3.7% 实测（AITER 为 4.1%） |

CP4 的 MoE 模型和理论估算基本吻合，差距主要来自本次镜像的 base 更快（见第 2 节）。CP8 和 Llama 明显低于估算，原因分别见第 5 节和 6.1 节。

## 5. 单层微基准（kernel 层面）

用 Turbo 的 `benchmark/ops/training/bench_ulysses_a2a.py --fused` 测一层 attention 加 dense MLP 的前向加反向。GLM-4.7 / MiniMax-M2.5 的形状通过 `bench_glm.py` 补进模型表（不修改 Turbo 文件），MLP 用激活专家宽度的 dense SwiGLU 代替。"未融合"是 Turbo USP 路径（pack + RCCL all-to-all + unpack），和 base 一样。两种实现用的是同一个 attention backend，所以这里只反映 GemmA2A / A2AGemm 的差别。表中是 max rank 的层总耗时，单位 µs；数据见 `logs_bench/layer_cp4.log`、`logs_bench/layer_cp8.log`。

| 形状 | CP | 长度 | 未融合层 fwd+bwd | 融合层 fwd+bwd | 每层节省 | 未融合交换（pack+a2a+unpack） | 节省 / 交换 | A2AGemm（fwd a2a+o_proj）加速 | RCCL a2a 带宽 |
|---|---|---|---|---|---|---|---|---|---|
| GLM-4.7 | 4 | 16K | 17128 | 15391 | 1737 | 3045 | 57% | 1.25× | 140 GB/s |
| GLM-4.7 | 4 | 32K | 45778 | 42335 | 3443 | 5845 | 59% | 1.31× | 145 GB/s |
| MiniMax-M2.5 | 4 | 16K | 9131 | 8164 | 967 | 1754 | 55% | 1.38× | 132 GB/s |
| MiniMax-M2.5 | 4 | 32K | 23315 | 21433 | 1882 | 3284 | 57% | 1.54× | 141 GB/s |
| Qwen3-30B-A3B | 4 | 16K | 4713 | 4111 | 602 | 1199 | 50% | 1.39× | 125 GB/s |
| Qwen3-30B-A3B | 4 | 32K | 12334 | 11108 | 1226 | 2212 | 55% | 1.47× | 136 GB/s |
| Llama3.1-8B | 4 | 16K | 9030 | 8403 | 627 | 1350 | 46% | 1.28× | 126 GB/s |
| Llama3.1-8B | 4 | 32K | 18696 | 17305 | 1391 | 2449 | 57% | 1.48× | 138 GB/s |
| GLM-4.7 | 8 | 16K | 8799 | 8441 | 358 | 1202 | **30%** | **0.93×** | 253 GB/s |
| GLM-4.7 | 8 | 32K | 22447 | 21575 | 872 | 2190 | **40%** | 1.12× | 287 GB/s |
| Llama3.1-8B | 8 | 16K | 5962 | 5814 | 148 | 661 | **22%** | **0.80×** | 191 GB/s |
| Llama3.1-8B | 8 | 32K | 11381 | 10956 | 425 | 1055 | **40%** | 1.17× | 234 GB/s |

（RCCL a2a 带宽取 pre / post 两次 all-to-all 的平均值。）

- CP4 下微基准的每层节省和端到端的每层每 micro-batch 节省是同一个量级。例如 GLM 16K / 32K 为 1.74 / 3.44 ms 对 1.40 / 3.61 ms，MiniMax 为 0.97 / 1.88 对 0.90 / 1.75。所以端到端收益确实来自融合 kernel。
- CP8 下 QKV 的 GemmA2A 仍有 1.5–1.8× 的加速，但 A2AGemm 在 16K 下比不融合更慢，反向两个融合阶段也只有 1.07–1.26×（CP4 为 1.20–1.62×）。与此同时，RCCL 在 n=8 时带宽接近翻倍。以 GLM / Llama 为例，16K 下交换在层内的占比从 CP4 的 17.8% / 14.9% 降到 CP8 的 13.7% / 11.1%，融合能省的空间本来就更小。

## 6. 结论

### 6.1 发布主推

推荐 **Qwen3-235B-A22B，CP4 / EP8，32K**。

- 推算完整模型收益为 6.9%（16K 为 5.0%），两种外推口径一致（6.9% / 6.8%）。
- 同结构的 Qwen3-30B-A3B 有完整模型实测（16K −8.9%，32K −5.5%），本次的外推方法在它上面的误差不超过 0.3 个百分点。
- 对外宣传时应写清楚口径：用户打开 `PRIMUS_ULYSSES_FUSED_A2A=1` 后看到的收益里，包含 patch 路径顺带换用的 FlyDSL attention。只算 GemmA2A / A2AGemm，32K 完整模型约为 5.8%。

**GLM-4.7 CP4** 适合作为第二个案例：32K 推算 6.6%，16K 推算 5.1%，每层节省最大，并且两条路径都用 AITER，收益全部来自融合 kernel。**MiniMax-M2.5 CP4**（6.2% / 6.3%）说明 partial RoPE 加 qk norm 的路径没有问题。

**Llama3.1-8B 不适合主推。**

- 在默认配置下 16K 比 base 慢 9.0%。这是 Turbo `flash_attn_func` 在 8 / 2 头、16K 的形状上默认选中了 FlyDSL，而它比 AITER 慢 19%。建议在 attention dispatch 里对这个形状改选 AITER，或者调优这个 FlyDSL kernel，这和本次的融合无关。
- 即使改用 AITER，收益也只有 4.0–4.1%，低于理论估算的约 9%。实测到的原因是路径重构本身有 1.5–2.2% 的开销，抵消了一部分融合收益（`fused/serial−1` 为 −5.5% 到 −6.1%）。此外，端到端每层每 micro-batch 的节省（AITER 下 0.34 / 0.76 ms）只有单层微基准的一半左右（0.63 / 1.39 ms）。这部分差距没有进一步拆分，可能和 dense 8B 在 DP=2 下的优化器、DP 通信与 attention 层争用有关。

### 6.2 CP=8 是否值得先调 kernel 参数

值得调，但不建议作为发布前提，优先级排在 CP4 发布之后。

- **现状。** GLM CP8 推算完整模型 16K 为 −0.5% 到 −3.9%（处在噪声范围内），32K 为 −3.5% 到 −3.8%。
- **上限。** 假设调优后 CP8 也能消掉 CP4 那样 57–59% 的交换时间，微基准下每层可多省约 330 µs（16K）和 400 µs（32K），折算到完整 GLM（90 层、16 个 micro-batch）约多 1.7 / 1.2 个百分点，也就是 16K 约 −2% 到 −5.5%、32K 约 −5%。理论估算的 9% 偏乐观，因为它没有考虑 n=8 时 RCCL 带宽翻倍。
- **确定的缺陷，应先修。** A2AGemm 在 n=8、16K 下比不融合还慢（GLM 0.93×，Llama 0.80×）。修好这一项至少能保证 CP8 不倒退，成本也比全面调参低。之后再调反向的两个融合 kernel。

### 6.3 回退与数值异常

- **结构回退：无。** 所有 serial / fused 运行都应用了 patch，没有 `keeping the stock path`，probe 显示每次都调用了融合（或 serial）kernel，没有任何一层回到 `flash_attn_usp_func`。GLM 和 MiniMax 的 attention 回退到 AITER，这是 Turbo attention backend 层面的回退（见第 3 节），patch 本身没有回退。
- **Llama3.1-8B 16K 的 loss 偏差（5.3e-2）是训练动态放大，不是计算错误。** 这个 dense 模型在 mock 数据上用 `lr_warmup_iters=2`，12 步内 loss 从 11.9 降到 7.8，grad norm 在 100–210 之间（clip 为 1.0），对微小的数值差异非常敏感。对照实验都是 base CP4 16K，只改并行方式，数学上是同一个训练：

  | 运行 | 第 1 步 loss | 第 12 步 loss | 与 base CP4 相比，第 12 步差 |
  |---|---|---|---|
  | base CP4 | 11.88879 | 7.81069 | 0 |
  | base CP4 重跑 | 11.88879 | 7.80844 | −2.3e-3 |
  | base CP2 | 11.88921 | 7.70410 | −0.107 |
  | base CP8 | 11.89202 | 7.66625 | −0.144 |
  | serial CP4（patch 路径） | 11.88847 | 7.86326 | +0.053 |
  | fused CP4（patch 路径） | 11.88846 | 7.86367 | +0.053 |

  patch 路径的偏差在 base 自身随 CP 变化的范围之内；它第 1 步的偏差（3.2e-4）也比 base CP8 的 3.2e-3 小。32K 下三者第 12 步的 loss 只差 1.1e-3。
- **MiniMax-M2.5 的 `bases:`。** `minimax_m2.5.yaml` 用 `bases: language_model.yaml`，而 Primus loader 只处理 `extends:`，所以 `language_model.yaml` 及其上层的默认值确实没有被继承。逐项对比 MiniMax 和 GLM 打印出的参数后，与 GLM 不同的只有以下几项，对本次测试都没有影响：
  - `add_position_embedding=True`：mcore 里由 `position_embedding_type == 'learned_absolute'` 决定，rope 模型不使用；
  - `use_rotary_position_embeddings=False`：只有为 True 时才会改写配置；
  - `mscale_all_dim=0.0`：只用于 YaRN；
  - 另有 14 个键（例如 `apply_layernorm_1p`、`model_type`）在参数里不存在，运行中没有被访问。

  注意力和 norm 相关的参数都和预期一致：RMSNorm、`rotary_percent=0.5`、`rotary_base=5e6`、`qk_layernorm=True`、`apply_rope_fusion=True`、无 bias。建议把这个文件改成 `extends:`，避免以后依赖默认值的改动在 MiniMax 上悄悄失效。
- **Llama 的 MoE 配置项。** 训练配置里的 MoE 相关项（`moe_router_force_load_balancing`、`use_turbo_deepep`、`turbo_sync_free_moe_stage` 等）没有导致 Llama 启动失败，所以没有另外复制配置。

### 6.4 其他记录

- 为了定位 Llama 变慢的原因，我做了不依赖 profiler 的消融（`logs_ctl/`），以下两项都可以排除：
  - 关闭 `overlap_param_gather` / `overlap_grad_reduce` 后，serial 仍比 base 慢 343 ms（2501.1 对 2157.7）；
  - 关闭 `gradient_accumulation_fusion` 后，仍慢 323 ms（2519.4 对 2196.6）。

  patch 路径里单独的 `F.rms_norm` 在单卡上测得前向加反向只要 107 µs（Llama 形状，见 `bench_norm.py`），也可以排除。
- 在这个容器里，rocprofv3 和 Megatron 的 torch profiler 都没能完成一次训练 profiling。rocprofv3 会注入 `primus-cli` 启动链里的 `tee`，导致训练无法启动；torch profiler 只抓 rank 0 时，rocprofiler-sdk 的 queue interposition 一直停在 "Async signal handler still waiting on signal"。日志在 `logs_prof/`。

## 附录

### A. 日志路径

所有日志都在共享存储 `/perf_apps/xiaoming/ulysses_proxy/` 下，两台节点都能访问。

| 目录 | 内容 |
|---|---|
| `logs/<前缀>_<实现><16k\|32k>.log` | 主矩阵 36 次运行，前缀为 `q30b`、`q235`、`glm_cp4`、`glm_cp8`、`mm25`、`l8b`，实现为 `base`、`serial`、`fused` |
| `logs_layers/<标签>_<base\|fused><16k\|32k>.log` | 层数补充 24 次，标签为 `q30bL4`、`q30bL8`、`q235L8`、`glm_cp4L8`、`glm_cp8L8`、`mm25L8` |
| `logs_ctl/` | Llama 对照：`l8b_base_cp4_rerun`、`l8b_base_cp2`、`l8b_base_cp8`；消融：`l8b_noovl_{base,serial}`、`l8b_nogaf_{base,serial}`；AITER 隔离：`l8b_aiter_{serial,fused}[32k]`、`q30b_aiter_{base,serial,fused}`、`q30b_aiter_{serial,fused}32k`、`q235_aiter_{serial,fused}{16k,32k}`，共 20 个 |
| `logs_v1_prefix/` | 修复 patch 之前的第一轮矩阵（Primus a4b8abde4），包括 GLM serial / fused 崩溃的日志 |
| `logs_validate/` | 修复后的验证：GLM CP4 fused 16K、GLM CP8 serial 16K |
| `logs_bench/` | `attn.log`（attention backend 单卡）、`layer_cp4.log`、`layer_cp8.log`（单层微基准） |
| `logs_prof/` | profiler 尝试（未成功） |
| `driver_n01.txt`、`driver_n02.txt`、`chain*_n0?.txt`、`ctl*_n02.txt`、`n02_batch2.txt`、`l8b32k_n01.txt` | 各批次的驱动输出（带时间戳） |

脚本（同一目录）：

| 脚本 | 作用 |
|---|---|
| `run_one.sh` | 容器内单次运行 |
| `run_matrix.sh` | 主矩阵驱动 |
| `check.py` | 逐组检查和汇总表 |
| `run_layers.sh` / `run_one_layers.sh` | 层数补充运行 |
| `run_ctl.sh` | 对照 / 隔离运行 |
| `layers.py` / `report_tables.py` | 外推和报告表格 |
| `bench_attn.py` / `bench_glm.py` / `bench_norm.py` | 微基准 |
| `sc/sitecustomize.py` | Turbo 路径和调用 probe |

### B. 版本与节点

| 项 | 值 |
|---|---|
| Primus-Turbo | `feat/ulysses-gemm-a2a` @ 60687e1262b4df4ed714c73e165e472e5024249a（测试所用代码；本报告的 commit 在它之上） |
| Primus | `feat/ulysses-fused-a2a` @ f9a298a49acbd39b2d4fc9ab72ac22dbdc5d1729（含 pre-hook 修复）；修复前的第一轮为 a4b8abde4 |
| Megatron-LM | 4b4acac9a1d28ea6829c8d4f566d75698a21249d |
| flydsl | 0.3.4.1 |
| torch | 2.14.0+rocm10.1.0 |
| 镜像 / 容器 | `tasimage/primus:pr-1275` / `xiaoming-dev`（两台节点相同） |
| 节点 | `smci355-ccs-aus-n01-33`、`smci355-ccs-aus-n02-21`（8×MI355X） |
