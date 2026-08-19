# MXFP4 Quant、GEAK stacked Quant 与 K256 紧凑 wgrad 优化在 Primus-Turbo PR #460 上的验证报告

## 结论

本报告覆盖三套必须按 exact source boundary 分开解读的证据：最初移植到
Primus-Turbo PR #460（下文简称 `primus-460`）的两项 Quant-only 优化、随后加入当前
branch 的 K256 wgrad producer/consumer 原子合约，以及 2026-08-19 新归档的 GEAK
stacked grouped-Quant winner。当前 branch head 只包含前两者；GEAK 新 patch 目前仅作为
已验证候选记录在本报告中，尚未集成到 branch 代码。

两项 Quant-only 优化在相同 PR #460 GEMM、相同 routing、相同进程和交替配对计时
条件下，三次独立 full-op session 的等权汇总结果为：

| 指标 | primus-460 原生 Quant | 优化后 Quant | 加速比 | 性能提升 |
|---|---:|---:|---:|---:|
| Quant | 3.293491 ms | 2.552228 ms | **1.290438x** | **+29.0438%** |
| Full-op | 8.312024 ms | 7.574293 ms | **1.097399x** | **+9.7399%** |

这里的“性能提升”定义为 `(baseline_latency / candidate_latency - 1) * 100%`。对应的延迟下降分别为 22.5070%（Quant）和 8.8755%（full-op）。

结论是：**Quant 优化在 primus-460 上仍然明确有效**。三次独立 session 的 full-op 加速均为 1.0932x～1.1019x，95% bootstrap CI 下界均大于 1，同 session A/A 噪声门限也全部通过。Fwd、dgrad、wgrad 三个 GEMM phase 汇总均在 ±0.12% 内，且两臂 GEMM 文件 SHA256 完全相同，因此该轮 full-op 收益可归因于 Quant 优化，而不是 GEMM 配置或实现差异。

K256-only 候选在 primus-460 上的另一组三 session 正式复测结果为：

| 指标 | K256-only 加速比 | 性能提升 |
|---|---:|---:|
| Quant collateral benefit | **1.045990x** | **+4.5990%** |
| Wgrad | **1.016496x** | **+1.6496%** |
| 六 GEMM aggregate | **1.005547x** | **+0.5547%** |
| Full-op | **1.020800x** | **+2.0800%** |

K256 的三个 full-op session 为 `1.021137x / 1.020317x / 1.020949x`，全部通过
CI、MDE99、route floor、correctness、zero fallback、cache 和 selector gate。当前 branch
把两项 Quant 优化与 K256 合并到同一源码，但这个**精确组合后的 branch head 尚未单独执行
一次 paired full-op A/B**，因此不能把 `1.097399x` 与 `1.020800x` 直接相乘后称为实测结果。

新归档的 GEAK stacked grouped-Quant candidate 在 48 个 route×shape case 上给出：

| 聚合指标 | 加速比 | 性能提升 |
|---|---:|---:|
| Workload-aligned time-weighted ratio-of-sums | **1.0206x** | **+2.06%** |
| Unweighted geomean | **1.02158x** | **+2.158%** |
| Arithmetic mean | **1.02164x** | **+2.164%** |

该数字是 **grouped Quant kernel-only** 结果，baseline geomean latency 为 `3.7122 ms`，
优化后约为 `3.634 ms`。它来自 BM=128、仍使用 512-aligned colwise padding 的旧
Quant-only source boundary，不包含 K256，也没有给出对应的 fwd、dgrad、wgrad 或 full-op
实测。因此不能把 `1.0206x` 与 `1.290438x`、`1.045990x` 或 `1.097399x` 相乘后称为
当前组合 branch 的实测收益。

## PR 状态与依赖

- branch：`zhitwang17:codex/quant-mxfp4-pr460`
- stacked base：`dev/kyle/gptoss-mxfp4-grouped-pr`
- `primus-460` PR：https://github.com/AMD-AGI/Primus-Turbo/pull/460
- `primus-460` head commit：`0f3972175fdbe3621d5bf67b23f2b15decfccbf3`
- 当前 fork branch head：`5a8860733570e651d26ff009330b5cf7df91e1cb`
- GEAK 新 Quant 证据 commit：`b9bf93f4f25365604b6ce38b6e753120e2efc003`

本 branch 以 #460 的 head branch 为 base，不重复包含 #460 本身的 grouped GEMM 调优。
PR #463 当前保持关闭；本次只更新 branch 内容，不 reopen PR。#460 合并或更新后，需要
重新 rebase 并复跑完整验证。GEAK 新 Quant patch 尚未应用到该 branch；本节新增的是
验证记录与后续集成边界，不代表 branch 已包含新 kernel code。

## 优化策略与代码范围

### 1. Grouped dual quant 的 BM 从 64 调整为 128

文件：`primus_turbo/flydsl/quantization/mxfp4_grouped_quant.py`

默认 `BM` 从 64 增大到 128，在不超出 LDS 预算且保持 tile 不跨 group 的前提下：

- 每次 launch 覆盖的有效行数翻倍；
- 摊薄 grouped metadata 和 kernel prologue 开销；
- 保持现有 `BK=256`、row/col 输出布局和数值语义不变。

### 2. Batched dual quant 避免四个完整输出 tensor 的全量清零

文件：`primus_turbo/flydsl/quantization/mxfp4_quant_kernel.py`

原实现只要存在 padding，就使用 `torch.zeros` 分配 row/col 的 data 与 scale 四个输出。优化后：

- 四个输出均使用 `torch.empty`；
- row-output 的 K-pad tail 已由 kernel 的 masked-zero load 完整写回，无需预清零；
- 仅对没有 producer thread 覆盖的 col-output N-pad suffix 显式执行 `zero_()`；
- 避免与有效输出规模等大的冗余 memset，同时保持 padding 区域为零的接口契约。

### 3. K256 紧凑 producer/consumer 原子合约

涉及文件：

- `primus_turbo/flydsl/quantization/mxfp4_grouped_quant.py`
- `primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py`
- `primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py`

该优化将 grouped colwise Quant 的每组 M span 从 512 对齐收紧为 256 对齐，并同步修改
wgrad consumer：

1. producer 输出 256-aligned `group_lens/group_offs`，减少 Quant grid 和 colwise 写回；
2. consumer 使用 raw `nval = span / 256`，不再假定 trip count 必为偶数；
3. runtime whole-loop 显式处理 zero、even pair 和 odd-256 MFMA-only tail，并 drain
   speculative preload，防止 odd span 越过 expert 边界。

这三部分是不可拆分的正确性合约：只改 producer/trip 会在 odd span 上产生错误；只改
runtime loop 而保留 512 padding 则路径基本休眠，没有可测收益。

### 4. GEAK stacked grouped-Quant candidate（新证据，尚未集成）

文件：`primus_turbo/flydsl/quantization/mxfp4_grouped_quant.py`

GEAK round-4 winner `r4_d0` 将两个正交 lever 合成一个 candidate：

1. **按 N 维折叠 launch grid。** Launch 从 `NBM × NBK` 收缩为 `NBM`；每个 padded-M
   block 的 WG 只读取一次 `GO[0..G]` 并计算一次 group metadata，然后在 WG 内循环所有
   `NBK` 个 N-block。这样原先每个 M×N tile 都重复执行的 group scan 被摊薄为每个
   M-block 一次，同时把只依赖 M-block 的 row-band SRD 创建移出 N-loop。
2. **COL fp4 直接写 global memory。** 删除约 16 KB 的 `ldsc` LDS staging buffer，COL
   half 将连续 fp4 vec4 直接 `buffer_store` 到 `COL_OUT`，从而删除 stage→barrier→read-back
   的 LDS 往返及对应 visibility barrier。N-loop 末尾的 WAR barrier 仍必须保留，防止下一
   N-block 的 tile load 覆盖仍在读取的共享 `buf`。

这两个 lever 单独运行时约处于同 session 噪声门限附近，GEAK 在 round 2–3 观察到 A/A
MDE99 在约 `0.00285–0.0068` 间波动，因此最终以 stacked candidate 把 point estimate
抬到噪声门限之上。

需要特别说明：GEAK final report 的摘要将 group-boundary lookup 描述成 `O(log G)`；但
commit 中归档的实际 patch 仍使用 `for g in range_constexpr(G)` 的 O(G=32) register scan。
实际代码收益来自“把 O(G) scan 从每个 M×N tile 减少到每个 M-block 一次”，本报告以
archived patch 为准，不把尚未出现在 patch 中的二分搜索算作已实现改动。

### 5. Padding 与 K256 回归测试

文件：`tests/pytorch/ops/test_quantization.py`

新增或保留三类覆盖：

- 3D batched dual-quant padding：覆盖 `(N,K)=(64,192)` 和 `(192,64)`；
- grouped Quant metadata：覆盖 zero、1、255、256、257、511、512、513 行 group，确认
  colwise lens/offs 严格按 256 对齐；
- FlyDSL full forward/backward：覆盖 zero、odd 和 even 256-block span，检查 fwd、dgrad、
  wgrad SNR。

## 测试版本与环境

| 项目 | 配置 |
|---|---|
| GPU | AMD Instinct MI355X，`gfx950`；物理 GPU 4 |
| 容器 | `geak_primus460_profile_gpu4_20260818` |
| 镜像 | `geak/rocm-pytorch-flydsl:0.2.4-rocm7.2.2` |
| PyTorch | `2.8.0+rocm7.2.2.git7f079cbb` |
| ROCm runtime | `7.2.53211-671d39a71e` |
| FlyDSL | `0.2.4` |
| Baseline | `primus-460` commit `0f3972175fdbe3621d5bf67b23f2b15decfccbf3`，原生 Quant |
| Candidate | Quant-only 与 K256-only 分别在同一 commit 上独立 A/B；当前 branch head 合并两者 |
| Public operator boundary | `grouped_gemm_fp4` 的 quantization + fwd + dgrad + wgrad |
| Backend 环境 | `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=FLYDSL`、`TURBO_GROUPED_GEMM_WITHOUT_PADDING=true`、`PRIMUS_TURBO_AUTO_TUNE=0` |

下列哈希来自 Quant-only paired A/B，证明该轮两臂使用完全相同的 PR #460 GEMM 源码：

| 文件 | Baseline/Candidate SHA256 |
|---|---|
| `gemm_mxfp4_kernel.py` | `c4643df5ae36d3f1514338d4a4d2f10b4b7788026f53f413d86f9da3a0a191c1` |
| `grouped_gemm_mxfp4_kernel.py` | `78a498c584c4abddd4b9407a67cdd4dae42de12fce419d40ec10858814329f51` |

用于 Quant-only 测试的 candidate Quant 源码 SHA256：

| 文件 | SHA256 |
|---|---|
| `mxfp4_grouped_quant.py` | `b7827da16587bf5c59a29b4a0a2c28c7fb75e27a44449e970a73c96912cb8008` |
| `mxfp4_quant_kernel.py` | `52e16bdd3b71cccb83ab9f9f3478be7b636b5803faff5a46b27f3cfc8ccbb597` |

当前 Quant + K256 组合 branch head 的相关源码 SHA256：

| 文件 | SHA256 |
|---|---|
| `mxfp4_grouped_quant.py` | `9e2001ac40a6647bda51a94c8fa523ba4a597b2a41653c6c1248c9bc0cc3b706` |
| `mxfp4_quant_kernel.py` | `52e16bdd3b71cccb83ab9f9f3478be7b636b5803faff5a46b27f3cfc8ccbb597` |
| `gemm_mxfp4_kernel.py` | `7814871974d464955aabf3696198be7da2e8608e0793616c63d10a52c39833e2` |
| `grouped_gemm_mxfp4_kernel.py` | `6d2cd68e75d31a45cb38fe995e515b85bd24d7e7c146f68702afd26673945e21` |

### GEAK stacked Quant 的 source boundary

- 证据 commit：`b9bf93f4f25365604b6ce38b6e753120e2efc003`；
- final report：`docs/technical-reports/2026-08-18_geak-primus460-quant-final-report.md`；
- archived patch：`docs/technical-reports/patches/2026-08-18_geak-primus460-quant-final.patch`；
- patch baseline blob：`42dfe9e197718f9e350a0e154f76a12a17423280`，精确对应
  `f35cb9e0715c2296fe9ee2b1a437ebe867744e5a:primus_turbo/flydsl/quantization/mxfp4_grouped_quant.py`；
- archived patch candidate index：`33745c4`；
- 该 baseline 已包含 `BM=128`，但仍为 512-aligned colwise padding；因此这轮证据不包含
  K256 `512→256` producer/consumer 合约，也不对应当前 `5a886073` branch head。

## Full-op shape、routing 与统计方法

- EP1 / G=32；每条 route 的 `total_M=131072`。
- 使用真实 GPT-OSS-20B 训练 capture 产生的 24 条加权 routing representatives。
- 每条 route 测试两个 grouped GEMM shape，共 48 个 route×shape cell：
  - GG1：`N=5760, K=2880`
  - GG2：`N=2880, K=2880`
- routing manifest SHA256：`d7c81df6ae0608124baeaad7ad83ed7e9bd8bce6765cc39669721072c6c83693`。
- 两臂在同一进程内动态绑定 Quant 源文件，public op runtime 和 PR #460 GEMM 保持不变。
- 每个 cell 先 warmup 1 次，再运行 8 个 ABBA/BAAB 交替 supercycle；每个 block 1 次调用。
- 正式 A/B 前后各做 2 个独立加载的 baseline A/A supercycle，用于估计同 session 的 99% MDE。
- CUDA event 计时；按 route 权重汇总 latency。
- 每个 session 做 10,000 次 stratified paired bootstrap，报告 95% CI。
- 三个 session 使用独立 seed；最终结果先在 session 内做 route-weighted latency，再对三个 session 等权平均，最后计算 baseline/candidate ratio。

K256-only 复测使用相同的 24-route × GG1/GG2 public-op 合同和三个 seed；正式结果来自
独立 git worktree，runner 在每个 session 前后校验 source SHA，排除了早期 mutable
workspace 污染的无效轮次。

GEAK stacked Quant 使用同样的 24 个 trace hash 和 GG1/GG2 两个 shape，形成 48-case
Quant kernel 表；其 headline metric 是按 workload time weight 计算的 ratio-of-sums，另报
unweighted geomean 与 arithmetic mean。它经历 4 个 round、共 8 个方向，最终 `r4_d0`
被标记为 VERIFIED winner。这套统计边界与上面的三 session public full-op paired A/B
不同，不能把两者的 gate 或聚合方式混写。

有效性门禁包括：sampled correctness、full-op CI 下界大于 1、收益大于同 session MDE99、每条 route 不低于 0.97x、零 fallback、计时 cache 稳定、selector contract 一致以及 source-selector receipt 完整。三次 session 的全部八项门禁均通过。

## Quant-only 原始性能数据

### 三次独立 session

| Session / seed | Quant 加速 | Full-op baseline | Full-op candidate | Full-op 加速 | Full-op 95% CI | MDE99 | 最差 route |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 / `20260811` | 1.291381x | 8.286491 ms | 7.552895 ms | 1.097128x | [1.094271, 1.097824] | 0.2310% | 1.091379x |
| S2 / `20261811` | 1.281402x | 8.278787 ms | 7.573263 ms | 1.093160x | [1.089346, 1.099120] | 0.7063% | 1.087461x |
| S3 / `20262811` | 1.298471x | 8.370794 ms | 7.596722 ms | 1.101896x | [1.096807, 1.103104] | 0.6418% | 1.087513x |

### 三个 session 等权汇总

| Phase | Baseline | Candidate | 加速比 | 性能提升 |
|---|---:|---:|---:|---:|
| Quant | 3.293491 ms | 2.552228 ms | **1.290438x** | **+29.0438%** |
| Fwd | 1.644042 ms | 1.645968 ms | 0.998830x | -0.1170% |
| Dgrad | 1.554053 ms | 1.554537 ms | 0.999689x | -0.0311% |
| Wgrad | 1.663773 ms | 1.664126 ms | 0.999788x | -0.0212% |
| Full-op | 8.312024 ms | 7.574293 ms | **1.097399x** | **+9.7399%** |

GEMM phases 的微小正负波动属于计时噪声；三者合计没有显示出候选侧的 GEMM 优化。Quant latency 减少约 0.7413 ms，full-op latency 减少约 0.7377 ms，两者一致。

### 单变量消融

| Candidate | Quant 加速 | Full-op 加速 | Full-op 95% CI 下界 |
|---|---:|---:|---:|
| 仅 BM=128 | 1.211688x | 1.074091x | 1.071391x |
| 仅 selective tail-zero | 1.054516x | 1.020838x | 1.019562x |

两项机制单独启用时均有显著收益，说明总体结果不是由单一偶然波动产生。

## K256-only 原始性能数据

| Phase | 加速比 | 三 session 范围/说明 |
|---|---:|---|
| Quant | **1.045990x** | 256-aligned producer 减少 padding/grid/write |
| Fwd | 0.999588x | 中性 control |
| Dgrad | 1.000308x | 中性 control |
| Wgrad | **1.016496x** | `1.0161x–1.0168x` |
| 六 GEMM aggregate | **1.005547x** | 收益集中在 wgrad |
| Full-op | **1.020800x** | `1.0203x–1.0211x` |

当前 routing 下，32 个 expert span 从 512 对齐缩到 256 对齐后，route-weighted padded M
从 `140394.667` 降到 `135412.000`，减少 `3.549%`；平均每条 route 有 `19.464` 个
odd-256 span 和 `2.281` 个 zero expert，因此 zero/even/odd runtime 分支都是真实路径，
不是只为 synthetic test 添加的死代码。

## GEAK stacked grouped-Quant 原始性能数据

| 指标 | Baseline | Candidate | 加速比 | 性能提升 |
|---|---:|---:|---:|---:|
| Geomean latency | 3.7122 ms | ≈3.634 ms | **1.02158x** | **+2.158%** |
| Workload-aligned ratio-of-sums | — | — | **1.0206x** | **+2.06%** |
| Arithmetic mean | — | — | **1.02164x** | **+2.164%** |

48 个 route×shape case 中，除 `ae810fc516cc9878/GG2 = 0.9988x` 这一项 near-tie 外，
其余 case 均不低于 `1.0x`。GG1（`N=5760`、NBK 更大）总体收益高于 GG2
（`N=2880`），与“折叠 N-grid、摊薄每个 M-block metadata scan”的机制一致。

GEAK 的 round 2–3 先分别验证两个 lever 可复现但单独接近噪声门限：grid-collapse 路径的
Quant phase 约为 `1.0130–1.0146x`，direct COL store 则在两个 shape 上都降低绝对 Quant
latency。round 4 将两者叠加后才稳定越过最差观测 MDE99，得到最终 `1.0206x` workload-
weighted winner。该表没有对应的 fwd、dgrad、wgrad 或 public full-op latency，不能从
Quant kernel ratio 直接推导完整算子收益。

## Correctness 与单元测试

- Quant-only full-op correctness：24 routes × GG1/GG2，共 **48/48 PASS**；
  baseline/candidate 的 fwd、dgrad、wgrad SNR 判定完全一致，fallback=0，selector/config
  receipt 两臂一致。
- K256-only 正式复测：**48/48 PASS**，最小 SNR `12.899 dB`，fallback=0。
- 当前 Quant + K256 组合 head 的新增 metadata 与 zero/even/odd forward/backward 测试：
  **4 passed, 6151 deselected**。
- 当前组合 head 的既有 dual-quant 回归：**130 passed, 3788 deselected**。
- 新增测试特别验证 batched 3D dual quant 的 row K-pad 与 col N-pad 都被物化为零。
- 新增 grouped Quant 256-alignment metadata 和 zero/even/odd wgrad 回归测试。
- 当前组合 head 尚未单独重跑 48-cell correctness harness；上面的两组 48/48 收据分别属于
  Quant-only 与 K256-only 冻结候选，不能替代组合 head 的 promotion gate。
- GEAK final report 将 stacked candidate 标记为 VERIFIED winner，并提供完整 48-case
  performance 表；这属于旧 512-padding Quant source boundary 的 kernel verification，
  不能替代将该 patch 与 K256 合并后的 correctness、paired full-op 或 training-step gate。

## 与 latest-new 约 +32% 结果、K256 及 GEAK 新 Quant 证据的关系

此前 `latest-new` 组合候选中记录的 Quant 结果约为：

- 3.260968 ms → 2.470426 ms
- 1.32000x，即约 +32.0%

两项 Quant-only 代码在 primus-460 上测得 1.290438x（+29.0438%）。差异主要来自
优化边界，而不是 PR #460 令 Quant 优化失效：

1. 原始 PR #463 commit 只包含 grouped dual quant 的 BM=128 与 batched dual quant 的
   selective tail-zero。
2. 本次 branch 更新已加入 K256 `512→256` 紧凑 padding 原子合约。它在最新 primus-460
   K256-only 复测中的 Quant collateral benefit 为 `1.045990x`，高于旧 parent 上的
   `1.021431x`；差异来自 routing 与 tile/grid 取整。
3. 旧的 exact latest-new 独立验证本身也出现过 1.281401x，与本次三个 session 的 1.281402x～1.298471x 区间一致。剩余差异可由 exact source boundary、routing/seed 和 session 间测量变化解释。
4. GEAK 新 winner 是在上述 BM=128 grouped-Quant 源码上进一步加入 grid collapse 与 direct
   COL store，独立给出 `1.0206x` workload-weighted Quant kernel 收益；其 baseline 仍为
   512 padding，因此不包含 K256 的 `+4.5990%` Quant collateral benefit。
5. Quant-only A/B 中 GEMM 哈希完全一致，证明最初两项 Quant 机制可以迁移；K256-only A/B
   则单独证明 producer/consumer 联合机制仍然有效；GEAK 48-case 表证明新的 stacked
   Quant kernel candidate 有效。三套证据的 source boundary 不同，精确合并仍需共同 A/B。

PR #460 的 GEMM 更快后，Quant 在 baseline full-op 中约占 39.62%，因此约 +29.04% 的 Quant throughput 提升最终转化为约 +9.74% 的 full-op throughput 提升，符合 Amdahl 分解。
GEAK 新 `1.0206x` 只能作为这一 Quant 路径上的额外独立证据；在没有 exact combined
measurement 前，不能将 `1.290438 × 1.045990 × 1.0206` 宣称为当前 branch 的 Quant
实测，更不能据此直接推导 full-op 或 training-step 加速。

## 复现命令

以下路径以本次测试容器内的 `/campaign` mount 为准。完整 paired harness 和原始 JSON 收据保存在实验工作区：

```text
/campaign/experiments/quant_on_pr460_20260818/bench/benchmark_gptoss20b_ep1_full_op_pr460.py
/campaign/experiments/quant_on_pr460_20260818/bench/summarize_sessions.py
/campaign/experiments/quant_on_pr460_20260818/results/
```

### Full-op correctness

```bash
python /campaign/experiments/quant_on_pr460_20260818/bench/benchmark_gptoss20b_ep1_full_op_pr460.py \
  --manifest /campaign/evidence/geak_runs/c5_direct_20260731/operator_campaign_20260803/evidence/real_training_routing/gptoss20b_primus_ep1_gbs32_mbs4_fp8_capture_20260811/pilot_v2/gptoss20b_primus_ep1_g32_initial_step0_matrix.json \
  --baseline-root /campaign/src/latest-460 \
  --candidate-root /campaign/experiments/quant_on_pr460_20260818/src/candidate \
  --extension-path /workspace/primus/primus_turbo/pytorch/_C.cpython-310-x86_64-linux-gnu.so \
  --output-json /campaign/experiments/quant_on_pr460_20260818/results/correctness.json \
  --seed 20260811 \
  --correctness-only
```

### Full-op paired performance

对 seed `20260811`、`20261811`、`20262811` 分别执行：

```bash
python /campaign/experiments/quant_on_pr460_20260818/bench/benchmark_gptoss20b_ep1_full_op_pr460.py \
  --manifest /campaign/evidence/geak_runs/c5_direct_20260731/operator_campaign_20260803/evidence/real_training_routing/gptoss20b_primus_ep1_gbs32_mbs4_fp8_capture_20260811/pilot_v2/gptoss20b_primus_ep1_g32_initial_step0_matrix.json \
  --baseline-root /campaign/src/latest-460 \
  --candidate-root /campaign/experiments/quant_on_pr460_20260818/src/candidate \
  --extension-path /workspace/primus/primus_turbo/pytorch/_C.cpython-310-x86_64-linux-gnu.so \
  --output-json /campaign/experiments/quant_on_pr460_20260818/results/full_op_s1.json \
  --seed 20260811 \
  --warmup 1 \
  --block-calls 1 \
  --supercycles 8 \
  --aa-supercycles 2 \
  --bootstrap-resamples 10000 \
  --confidence 0.95
```

聚合三个 session：

```bash
python /campaign/experiments/quant_on_pr460_20260818/bench/summarize_sessions.py \
  /campaign/experiments/quant_on_pr460_20260818/results/full_op_s1.json \
  /campaign/experiments/quant_on_pr460_20260818/results/full_op_s2.json \
  /campaign/experiments/quant_on_pr460_20260818/results/full_op_s3.json \
  --json /campaign/experiments/quant_on_pr460_20260818/results/summary.json \
  --csv /campaign/experiments/quant_on_pr460_20260818/results/session_phase_metrics.csv
```

### Quant 与 K256 单元测试

在已有 Primus-Turbo extension 可加载的环境中执行：

```bash
pytest -q tests/pytorch/ops/test_quantization.py \
  -k "mxfp4_with_trans or grouped_quantize_mxfp4_colwise"

pytest -q tests/pytorch/ops/test_grouped_gemm_fp4.py \
  -k "k256_zero_even_odd_wgrad"
```

## 原始收据

本次验证的原始文件位于宿主机：

```text
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/quant_on_pr460_20260818/results/correctness.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/quant_on_pr460_20260818/results/full_op_s1.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/quant_on_pr460_20260818/results/full_op_s2.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/quant_on_pr460_20260818/results/full_op_s3.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/quant_on_pr460_20260818/results/ablation_bm128.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/quant_on_pr460_20260818/results/ablation_tail_zero.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/quant_on_pr460_20260818/results/summary.json

/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/latest_new_gemm_on_pr460_20260818/rounds/round-6/artifacts/full_op_clean_s1.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/latest_new_gemm_on_pr460_20260818/rounds/round-6/artifacts/full_op_clean_s2.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/latest_new_gemm_on_pr460_20260818/rounds/round-6/artifacts/full_op_clean_s3.json
/home/zhitwang/GEAK/agent/workspace/mxfp4_moe_fullop_flydsl_gfx950_20260817/experiments/latest_new_gemm_on_pr460_20260818/rounds/round-6/artifacts/full_op_clean_summary/summary.json

/home/zhitwang/geak-mxfp4-gpt-oss-kb/docs/technical-reports/2026-08-18_geak-primus460-quant-final-report.md
/home/zhitwang/geak-mxfp4-gpt-oss-kb/docs/technical-reports/patches/2026-08-18_geak-primus460-quant-final.patch
```

GEAK 新证据的固定 GitHub commit 为：

```text
https://github.com/zhitwang17/geak-mxfp4-gpt-oss-kb/commit/b9bf93f4f25365604b6ce38b6e753120e2efc003
```

## 已知限制、风险与后续建议

- 当前性能结论针对 MI355X/gfx950、EP1/G32、GPT-OSS-20B 的两组训练 shape 和 24 条真实 routing representatives；其他 GPU、EP 配置、shape 和 routing 仍应单独验证。
- `BM=128` 依赖当前 grouped quant 的 LDS/tiling 约束；未来调整 tile 布局或扩大临时 LDS 使用时应重新检查资源预算。
- selective tail-zero 依赖 kernel 继续完整写入 row-output K-pad tail。若 producer mapping 或 masked-load/store 语义变化，必须保留本 PR 新增的 padding 回归测试。
- K256 必须保持 producer、consumer trip 与 runtime whole-loop 原子一致；任何一部分变化都要
  重跑 zero/even/odd correctness。
- GEAK grid-collapse patch 与 K256 都修改 `mxfp4_grouped_quant.py`，集成时必须以当前
  256-padding producer 为 base 手工迁移，不能直接把归档 patch 套到 `5a886073`；同时需要
  验证内部 NBK loop、direct COL store 与 256-aligned `lens/offs` 的组合正确性。
- Direct COL store 依赖当前 feature-major、M-contiguous 布局和连续 COL thread 的写合并；
  N-loop 末尾的 WAR barrier 不能随 `ldsc` visibility barrier 一起删除。
- GEAK final report 中的 `O(log G)` 描述与 archived patch 不一致；在实际 patch 改成二分
  search 之前，应将当前实现视为“每 M-block 一次 O(G=32) scan”。
- 当前 branch 的“BM128 + selective tail-zero + K256”精确组合尚未单独执行 promotion-grade
  paired A/B；进一步加入 GEAK grid-collapse + direct-store 后更没有 exact combined 数据。
  现有数字分别来自 Quant-only、K256-only 和 GEAK 旧 512-padding Quant exact-source 实验。
- 仍缺真实 GPT-OSS training-step transfer：目前证明的是完整 public operator replay 变快，
  尚未证明装回完整训练图后整步时间稳定下降。
- 本 branch 依赖 #460；#460 合并或更新后，需要重新 rebase，并至少复跑 Quant pytest、
  K256 zero/even/odd test、48-cell correctness 和一轮 paired full-op performance；若集成 GEAK
  新 patch，还应重新生成 source receipt，并单独报告 combined Quant kernel 与 full-op 数据。
