# B0 e2e 里的 attention 细粒度剖析：FlyDSL vs aiter ASM（2026-09-28，GPU0 / fa-g0）

配置同 `../e2e/RESULT.md` 顶部（Llama-3.1-8B BF16，MBS=GBS=4，seq 8192，32 层，nkfix GEMM 修复，`E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1`）。
FlyDSL = fwd r6 + bwd r20（flydsl 0.3.4.1）；另测了 fwd job 当前冠军 r11 和两个 r6 的单开关变体。

## 0. 结论

1. **每步差距 87 ms 的构成**（trace，step 50 fly / step 60 asm，同一进程）：fwd **+19.9 ms**，bwd **+71.0 ms**，ASM 独有的 GQA 求和 −3.9 ms。
   bwd 占 80%，fwd 占 20%。所有 attention kernel 与前后 kernel 的间隔都是 0.8–1.0 µs，与其它 kernel 没有重叠（单 stream），**没有 launch 或调度开销可省，差距全在 kernel 本身**。
2. **op 级结论在训练里反转有两个原因，二者可以相乘**：
   - **时钟**：训练里 attention 跑在 **1250–1690 MHz**（空闲是 2356 MHz），整板功耗一直顶在约 2.13 kW（PPT 读数，空闲时约 1.13 kW）。
     FlyDSL fwd 对 sclk 敏感：randn 输入下，2350 MHz 时比 ASM **快 16%**（0.84×），GEMM 刚跑完、时钟约 1300 MHz 时慢 20%（1.20×）。ASM fwd 基本不随 sclk 变化。
   - **数据**：op 级尺子用的是 N(0,1) 输入，score 标准差为 1。真实训练输入（step 43 dump 出来的 6 层 q/k/v）score 标准差是 21–53。
     r6 的投机 stale-max softmax（g14，`SPEC_STALE_MAX`）一旦触发，就要重算 QK 并重载 K/V。在真实数据上，32 行一组、按 64 列 KV tile 计，有 **13–25% 的 tile 步会触发**；randn 上是 **0%**。
     SQC_ICACHE_REQ 显示 fly fwd 在真实数据上多发射 12.8% 的指令，ASM 不变。
   - 用真实输入，在"刚跑完 GEMM"的时钟状态下，op 级测得 fly/asm = **1.45–1.67**（3 个进程之间相差 <1%）。训练里逐层的比值是 1.19–1.72，已经复现。
     同一批数据用分块尺子测是 1.15–1.30，用 randn 分块测是 1.02。
   - **fwd 差距还随训练步增长**：fly fwd 每步 50.2 → 55.3 → 59.2 ms（step 10/30/50），期间时钟不变；ASM 一直是 38.8–39.3 ms。
     RESULT.md 里"高档/低档 52.5→59.2"的差别，很可能至少有一部分是这个效应，而不是时钟档位。
3. **GEMM 修复后 attention 变慢，原因是时钟**，不是 L2 或共同调度：
   - 修复前，反向 GEMM 落在低功耗的 MT32x16x32 上，训练中 sclk 均值约 **1,820 MHz**；修复后约 **1,390 MHz**。
   - 变慢的只有对时钟敏感的 kernel：`k_dkdv` +18%，ASM bwd 主 kernel +17%，`k_delta` +85%。`k_dq`（对时钟不敏感）只 +2%。两边的 fwd 本来就跑在 fwd GEMM 之后，所以没变。
   - op 级复现（层仿真，反向 GEMM 用修复前/后两种布局）：sclk 1,856 → 1,385 MHz，`k_dkdv` 5.26 → 5.84，ASM 主 kernel 6.31 → 6.80，`k_dq` 不变。
4. **最大的一条可以直接交给 fwd job 的线索**：r6 关掉投机（`SPEC_STALE_MAX=False`，只改一行，输出与 r6 逐位相同）后：
   - 在真实输入、训练时钟下，fwd **快 13–23%**（1.59–1.60 ms 对 1.83–2.08 ms）；
   - 在 2350 MHz 下 **比 ASM 快 8–12%**；
   - 在 randn 上反而慢 5%，所以 job 的尺子一直选中了投机。
   - 估算每步省约 11 ms，fwd 差距从 20 ms 降到约 9 ms。r11 带着同一条投机路径，与 r6 相差在 1% 以内。
5. **bwd（71 ms）**：
   - `k_dq`（3.33 ms/层，约 107 ms/步）是 ASM 融合 kernel 里没有的额外工作。它对时钟完全不敏感（1350→2350 MHz 都在 3.1–3.4 ms），属于延迟/访存 bound：960 VGPR，每个 SIMD 只有 1 个 wave，16384 个单 wave 的 WG。
   - `k_dkdv` 对时钟的敏感度（2350→1350 MHz 慢 1.24×）高于 ASM 主 kernel（1.16×），是发射 bound。
   - 数据相关性：bwd 的 fly/asm 在所有输入和条件下都是 1.30–1.38。

本次卡上工作：2 次训练启动（4 层 smoke + 32 层 62 步），25 个 op 级进程，占卡约 30 分钟。每次之后 dmesg 都没有新的 amdgpu 行，没有挂卡。

## 1. 单步时间线（交付 1）

运行 `prof_t1`：32 层，62 步，schedule `asm,flyd,asm,flyd;asm,flyd,flyd,asm`（`flyd` = r6 原样，外加一次输入 dump）。profile 了 step 10/30/50（fly）和 20/40/60（asm）。
宿主机上同时跑时钟采样器（`tools/clksamp.sh`），读 hwmon 的 `freq1_input`（sclk）和 `power1_input`：数值变化时记一行，另外每 10 ms 一行心跳。SMU 数值大约每 14 ms 变一次，所以 kernel 级的时钟是约 10 ms 窗口的平均。
kineto 时间换成绝对时间用 `baseTimeNanoseconds + ts`。

| 项 | fly（s10 / s30 / s50） | asm（s20 / s40 / s60） |
|---|---|---|
| GPU span ms | 1640 / 1649 / 1661 | 1575 / 1587 / 1590 |
| FA path ms | 352.1 / 355.3 / 362.0 | 273.0 / 272.9 / 275.0 |
| attn fwd ms | 50.2 / 55.3 / 59.2 | 38.8 / 39.1 / 39.3 |
| attn bwd ms | 302.0 / 300.0 / 302.8 | 230.3 / 230.0 / 231.8（另有 GQA 求和 3.9） |

各层数据（s50 对 s60，`data/train_layers_s50_s60.txt`；完整逐 kernel 的 CSV 在 `data/train_layer_timeline.csv`）：

- **fwd**（按模型层序）：
  - fly：第 0 层 1.50 ms，其余层 1.71–2.06 ms；ASM 各层 1.18–1.26 ms；fly/asm = 1.19（第 0 层）到 1.72（第 7 层）。
  - sclk 在 fwd 开始时约 1,660–1,690 MHz（刚经过 optimizer 和步间空隙），到第 31 层降到约 1,240–1,290 MHz。两个 arm 的时钟曲线相同。
- **bwd**（按执行顺序，也就是模型第 31 层先跑）：
  - fly = `k_delta` 0.12 + `k_dkdv` 5.96 + `k_dq` 3.33 = 9.42 ms/层；ASM = odo 0.05 + 主 kernel 7.00–7.09 + dq_convert 0.07 = 7.18 ms/层。
  - 比值 1.26–1.39，与层无关。sclk 从约 1,250 升到约 1,430 MHz。
- **功耗**：所有 attention kernel 期间功耗都在 2,121–2,146 W，这是整板的天花板。空闲约 1,134 W，PPT cap 读数是 2,500 W；具体是哪个限制在起作用没有查。
- **邻居**：
  - fwd 前面是 `bfloat16_copy`（rope/contiguous），后面也是一个 `bfloat16_copy`；
  - fly bwd 的顺序是 `bf16→fp32 copy` → `k_delta` → `k_dkdv` → `k_dq`；
  - ASM bwd 的顺序是 `fill`（dq_acc 清零）→ odo → 主 kernel → dq_convert → GQA 求和。
  - 间隔都是 0.8–1.0 µs，重叠为 0。

## 2. 为什么 op 级结论在训练里反转（交付 2）

### 2.1 同一个进程里比较不同"工作点"（`tools/oppoint.py`，3 个进程，arm 顺序轮换，randn 输入）

各 kernel 的时间来自 kineto；sclk 来自宿主采样器（`data/opsum_p123.txt`）。

| 条件 | 含义 | fwd fly/asm（3 个进程） | fwd sclk（asm/fly） | bwd fly/asm（3 个进程） | bwd sclk |
|---|---|---|---|---|---|
| iso | 每次调用前空闲 150 ms | **0.842 / 0.846 / 0.845** | 2350 | 1.377 / 1.378 / 1.381 | 2345 |
| blk | 4 次不计时 + 9 次连续计时（分块尺子） | 1.016 / 1.021 / 1.025 | 1750–1900 | 1.306 / 1.311 / 1.310 | 1690–1810 |
| eburst | 紧跟 40 次 2.8 GB 的 elementwise | 1.049 / 1.048 / 1.046 | 1500–1740 | 1.341 / 1.350 / 1.346 | 1650–1740 |
| gburst | 紧跟 10 个 32768×4096×14336 GEMM | **1.198 / 1.191 / 1.189** | 1310–1550 | 1.311 / 1.310 / 1.312 | 1290–1400 |
| layer | 16 层训练仿真（nkfix 后的 GEMM 布局） | 1.204 / 1.195 / 1.214 | 1380–1410 | 1.295 / 1.297 / 1.297 | 1385–1434 |
| layerbad | 3 层，反向 GEMM 用修复前的 MT32x16x32 | 1.409 / 1.392 / 1.393 * | 1370–1920 | 1.375 / 1.368 / 1.370 | 1856 |

\* layer/layerbad 里的 q/k/v 来自仿真 GEMM，逐层的统计量不同，fly fwd 在第 1–3 层是 1.75–2.05 ms，其余层约 1.2 ms。这是第一次看到数据相关性，见 2.2。

**时钟敏感度**：用 blk/iso/eburst/gburst（bwd 另加 layer/layerbad）的所有调用拟合 `dur = a + b·(1000/sclk)`，3 个进程合并。

| kernel | @2350 | @1800 | @1350 | 1350/2350 | r² |
|---|--:|--:|--:|--:|--:|
| ASM fwd | 1.315 | 1.296 | 1.269 | **0.97**（不随 sclk 变化） | 0.01 |
| FlyDSL fwd r6（randn） | 1.190 | 1.313 | 1.487 | **1.25** | 0.28 |
| FlyDSL fwd r11 | 1.188 | 1.311 | 1.486 | 1.25 | 0.28 |
| ASM bwd 主 kernel | 5.983 | 6.383 | 6.952 | 1.16 | 0.67 |
| `k_dkdv` | 4.917 | 5.399 | 6.084 | **1.24** | 0.83 |
| `k_dq` | 3.365 | 3.257 | 3.103 | **0.92**（不敏感） | 0.16 |
| ASM odo / `k_delta` | 0.041 / 0.072 | | 0.057 / 0.107 | 1.41 / 1.49 | |

ASM fwd 对 sclk 不敏感，但对"条件"敏感：iso 1.22，blk 1.45，eburst 1.13，layer 1.00。它似乎取决于 q/k/v 是否刚被写过（驻留在 cache 里），或者 memory 侧的状态。这里只记录现象，没有查。

### 2.2 真实训练输入的回放（`tools/replay.py`，dump 自 step 43 的第 0/1/2/8/16/31 层，3 个进程）

`data/replay_p123.txt`，`data/replay_variants_p123.txt`；以下取 3 个进程的中位数，进程之间相差 <1%。

| 输入 | score 标准差 | 触发率：行 / 32 行组 | blk fly/asm | iso fly/asm | gb fly/asm（训练工作点） |
|---|--:|--:|--:|--:|--:|
| randn | 1.0 | 0 / 0 | 1.02 | 0.84 | 1.20 |
| L00 | 24.1 | 2.7% / 25.5% | 1.25 | 1.08 | 1.58 |
| L01 | 20.9 | 3.6% / 19.1% | 1.24 | 1.12 | 1.59 |
| L02 | 37.1 | 2.4% / 16.3% | 1.21 | 1.07 | 1.55 |
| L08 | 52.8 | 4.4% / 23.9% | 1.28 | 1.18 | 1.67 |
| L16 | 21.5 | 2.4% / 12.9% | 1.16 | 1.00 | 1.45 |
| L31 | 21.4 | 4.2% / 20.8% | 1.24 | 1.11 | 1.60 |

- 触发率的算法：batch 0，全部 32 个头，按 64 列 KV tile 升序走，stale max 超出 7 nat（`SPEC_TRIGGER = e^7`）就记一次触发。32 行组 = 同一个头连续 32 个 seq 行，是对 wave-uniform ballot 的近似。
- 训练里逐层实测是 1.19–1.72，和 gb 条件吻合。

**分解**：训练工作点下，fwd 的 fly/asm ≈ 0.84（2350 MHz、randn 时 fly 本来更快）× 1.43（时钟，gb/iso）× 1.32（数据，real/randn，gb 条件）≈ 1.58。

**fwd 变体**：r6 只改一个开关，其余不变；只编译检查无 spill、无 scratch；toy 形状和 prod 形状都先单独进程验过正确性。

| 变体 | 静态指令数 / v_wmma / v_nop / SGPR spill | 对 r6 的正确性 | gb 真实输入 ms（对 asm） | gb randn | iso 真实输入 | iso randn |
|---|---|---|---|---|---|---|
| r6（当前 e2e） | 6068 / 384 / 169 / 60 | — | 1.83–2.08（1.45–1.68） | 1.49（1.20） | 1.24–1.41（1.00–1.18） | 1.03（0.84） |
| r11（fwd job 冠军） | 5024 / 320 / 162 / 0 | — | 与 r6 相差 −1% 到 0 | 同 r6 | 同 r6 | 同 r6 |
| **nospec**（`SPEC_STALE_MAX=False`） | 4130 / 256 / 77 / 0 | **逐位相同**（prod L08 真实输入） | **1.59–1.60（1.26–1.29）** | 1.575（1.26） | **1.09–1.10（0.88–0.92）** | 1.08（0.89） |
| nodefer（`ENABLE_DEFER_RESCALE=False`） | 4133 / 256 / 81 / 0 | 与 r6 的 SQNR 60.6 dB | 1.69–1.71（1.35–1.37） | 1.70（1.36） | 1.16–1.17 | 1.16 |

- r11 的 prod 路径：与 r6 相比，只有小 grid 门控（m32x2）和"投机只用在无 mask 的 tile 上"两处不同（`arms/fwd_r11.sha256`）。prod 走的仍是 m32x8，结果与 r6 相差在 1% 以内。
- 这个对照说明 fwd job 的尺子（randn）系统性地奖励了 g14 投机：randn 上 −5%，真实输入上 +13–23%。

## 3. GEMM 修复后 attention 变慢 10–16% 的原因（交付 3）

| kernel | 修复前（p2a s22/s44） | 修复后（n5 + prof_t1） | 变化 | 时钟敏感度（1350/2350） |
|---|--:|--:|--:|--:|
| `k_dkdv` | 5.049 | 5.95–5.99 | **+18%** | 1.24 |
| ASM bwd 主 kernel | 6.008 | 7.00–7.09 | **+17%** | 1.16 |
| `k_delta` | 0.063 | 0.117 | +85% | 1.49 |
| ASM odo | 0.045 | 0.049 | +9% | 1.41 |
| `k_dq` | 3.272 | 3.30–3.35 | +2% | 0.92 |
| fly fwd | 1.636 | 1.57–1.84（随训练步增长） | 同一步数下约 0 | 1.25 |
| ASM fwd | 1.225 | 1.209–1.229 | 0 | 0.97 |

- 训练中的 sclk（5 s 采样，busy>90%）：修复前 p2a/p2b 为 1,826/1,811 MHz，修复后 n5/n6 为 1,383/1,394 MHz。
  prof_t1 的高频采样显示，bwd 阶段是 1,250–1,430 MHz，功耗一直顶在约 2.13 kW。
- 修复前，反向 GEMM 是低功耗的 MT32x16x32 回退 kernel（52–79 TF/s），功耗到不了上限，所以时钟停在约 1.85 GHz。
  修复后，反向 GEMM 换成 MT256（约 1.5 PF/s），功耗顶到上限，bwd 阶段的时钟掉到约 1.4 GHz。
  fwd GEMM 修复前后都是 MT256，所以两边的 fwd attention 都没变。
- 变慢的幅度按各 kernel 的时钟敏感度排序：`k_dq` 不敏感所以不变，这一点排除了 L2/MALL 驻留和共同调度这两种解释。
  op 级 layer 与 layerbad 对比（同一个进程，只换反向 GEMM 的布局）复现了方向和量级：ASM 主 kernel +8%，`k_dkdv` +11%，sclk 1,856 → 1,385 MHz。

## 4. 计数器与静态资源（交付 4）

rocprofv3 1.3.2（容器 `/opt/venv`）：
- `--kernel-trace` 和 `--stats` 在这套环境上**不写任何文件**，与 bwd job 第 5/6 轮的记录一致，所以 dispatch 时间用的是 kineto。
- `--pmc` 可以用。但 gfx1250 上列出来的 110 个计数器里，只有 `SQ_WAVES`、`SQ_CYCLES`、`SQ_BUSY_CYCLES`、`SQ_ITEMS`、`GRBM_*`、`SQC_ICACHE_*` 返回非 0。
  `SQ_INSTS_VEC32_VALU_WMMA`、`SQ_INST_CYCLES_VALU_WMMA`、`SQ_VALU_WMMA_FLOP_*`、`SQ_WAVE_CYCLES`、`CHA_*`、`CHC_*`、`GL1A_*`、`GL1C_*`、`SPI_RA_REQ_NO_ALLOC`、`CPC_*` 在所有 kernel 上都是 0。
- **所以 VALU/SALU/LDS busy、LDS bank conflict、L2 命中率、fetch/write 字节在这张卡上测不到**，下表只给能测的量。

测量方式：prod 形状，L16 真实输入（ICACHE 另测了 randn），每个 arm 5 次，每组计数器一个进程。汇总在 `data/pmc_summary.txt`，静态资源在 `data/static_resources.txt`。

| kernel | WG × wave/WG | VGPR / SGPR / LDS B / spill | wave/SIMD | SQ_WAVES | SQ_BUSY_CYCLES | SQC_ICACHE_REQ 真实 / randn | icache miss | profiler 下 ms |
|---|---|---|--:|--:|--:|--:|--:|--:|
| ASM fwd | 4096 × 4 | 1024 / 106 / 327680（声明值）/ 0 | 1 | 16,384 | 3.68e8 | 1.131e8 / 1.131e8 | 0.02% | 1.15–1.23 |
| fly fwd r6 | 4096 × 8 | 448 / 107 / 327680 / 0（60 个 SGPR spill 到 VGPR lane，无 scratch） | 2 | 32,768 | 5.90e8 | **1.353e8 / 1.200e8（+12.8%）** | 0.00% | 1.28–1.36 / 1.11 |
| ASM odo | — | 35 / 50 / 0 / 0 | | 32,768 | 2.61e7 | 1.05e6 | 1.1% | 0.046 |
| ASM bwd 主 kernel | 4096 × 4 | 1024 / 128 / 327680 / 0 | 1 | 16,384 | 2.887e9 | 2.905e8 / 2.905e8 | 0.00% | 5.97–6.31 |
| ASM dq_convert | — | 32 / 48 / 0 / 0 | | 65,536 | 3.48e7 | 3.5e4 | 5% | 0.07 |
| `k_delta` | 32768 × 8 | 40 / 21 / 0 / 0 | | 262,144 | 2.76e7 | 2.87e4 | 11% | 0.05 |
| `k_dkdv` | 8192 × 1 | 904 / 79 / 70656 / 0 | 1 | 8,192 | 2.398e9 | 2.960e8 / 2.960e8 | 0.00% | 4.99–5.16 |
| `k_dq` | 16384 × 1 | 960 / 89 / 8704 / 0 | 1 | 16,384 | 1.540e9 | 1.419e8 / 1.419e8 | 0.00% | 3.12–3.26 |

- fly bwd 的 SQ_BUSY_CYCLES 合计 3.97e9，ASM 是 2.95e9，比值 1.35，与时间比一致。
- fly bwd 的指令请求 4.38e8，ASM 是 2.92e8，多 1.50×。
- fly fwd 在真实数据上的指令请求是 ASM 的 1.20×，randn 上是 1.06×。
- 静态 ISA：
  - r6 fwd：6068 条指令，其中 v_wmma 384，v_nop 169，v_readlane/writelane 143（SGPR spill）。
  - nospec：4130 条指令，其中 v_wmma 256，v_nop 77。
  - `k_dkdv`：3168 条（v_wmma 128）；`k_dq`：3567 条（v_wmma 192）。
  - ASM 的 TDM 指令 objdump 解不出来，逐 256 KV 的 ASM/Fly 对照见 `../../0925__flydsl/fwd-isa/REPORT.md`：v_nop/SALU 13/58 对 156/188，barrier 1.5 对 4。
- rocprofv3 报告的 dispatch VGPR 是静态值的一半（例如 ASM 报 512，静态是 1024），是分配粒度的写法，以静态值为准。

## 5. 如何补上每步约 87 ms（交付 5，按预计收益排序，可直接作为 hint）

| # | job | 杠杆 | 证据 | 预计每步 |
|---|---|---|---|---|
| 1 | bwd | **`k_dq` 是延迟 bound，不是发射 bound** | 3.1–3.4 ms 与 sclk 无关（1350–2350 MHz）；每 SIMD 1 wave（960 VGPR）；16384 个单 wave WG；它是 ASM 融合 kernel 里没有的第二遍（重算 QK 和 dP）。方向：把 VGPR 压到 ≤512 换取 2 wave/SIMD，或更深地预取 K/V（TDM），或寻找不需要原子操作的办法把 dq 并进 dkdv（split 路径） | `k_dq` 从 3.33 降到 2.2 ms/层可省约 **36 ms** |
| 2 | fwd | **关掉 g14 投机 `SPEC_STALE_MAX`，或只在不触发时才走投机** | 真实输入、训练时钟下 nospec 比 r6 快 13–23%，输出逐位相同；在 2350 MHz 下比 ASM 快 8–12%；randn 上慢 5%（尺子因此选错）。r11 带着同一条投机路径 | 约 **11 ms** |
| 3 | fwd + bwd | **尺子换成训练工作点**：输入用真实 q/k/v（`/home/lihuzhan/_prof_dump/qkv_call0{672,673,674,680,688,703}.pt`，step 43 的第 0/1/2/8/16/31 层，[B,S,H,D] bf16），条件用"紧跟 GEMM 突发"（10 个 32768×4096×14336）或至少 blk | 在 randn + blk 下 fwd 看起来是 1.02×，训练里是 1.45–1.72×。bwd 在各条件下稳定（1.30–1.38），换尺子影响小 | 不直接省时间，但能避免再选出类似 #2 的"赢家" |
| 4 | fwd | **降低对时钟的敏感度**：减少 v_nop、SALU、barrier、permlane | fly fwd 1350/2350 = 1.25×，ASM 是 0.97×。nospec 在训练时钟下仍是 ASM 的 1.26–1.29×，在 2350 MHz 下是 0.89× | 做完 #2 之后剩下的约 9 ms |
| 5 | bwd | **`k_dkdv` 是发射 bound**：减少 VALU/SALU/nop | 1350/2350 = 1.24×，ASM 主 kernel是 1.16×；在训练时钟下 5.96 ms，在 2350 MHz 下 4.92 ms | 若敏感度降到 ASM 的水平，约 **12 ms** |
| 6 | bwd | 把 `k_delta` 融进 `k_dkdv` 的开头 | 0.117 ms/层（时钟敏感，1.49×） | 约 3.7 ms（ASM 路径自身有 GQA 求和 + fill 共 6 ms，fly 已经省掉了） |

- 系统层面（与 kernel 无关）：attention 被整板约 2.13 kW 的功耗上限压在 1.25–1.45 GHz。任何降低 GEMM 功耗的改动都会让时钟敏感的 kernel 变快，例如 `NKFIX_CHECK=0` 可以少 33 ms/步的 reduce。这对 fly 的好处大于对 ASM。
- fwd 的差距随训练步增长（step 10/30/50 分别是 +11.4/+16.3/+19.9 ms）。更长的训练里，#2 的收益可能更大。

## 6. 文件

- `tools/`：
  - `clksamp.sh`、`clksamp2.sh`（加了 fclk）：宿主时钟采样器；
  - `run_op.sh`：一个卡上进程，负责 lock、采样器和 dmesg 检查；
  - `oppoint.py` / `opana.py` / `opsum.py`：工作点实验；
  - `replay.py` / `rpsum.py` / `rpvsum.py`：真实输入回放；
  - `timeline.py` / `train_layers.py`：训练 trace 的逐层时间线；
  - `pmcrun.py` / `pmc_all.sh` / `pmcsum.py`：计数器；
  - `varcheck.py`：变体正确性；`flyd_check.py`：dump arm 检查。
- `arms/flyd/impl.py`：r6 原样，加上 `PROF_DUMP_CALLS` 输入 dump。
- `arms/variants/*.diff`：nospec/nodefer 相对 r6 的一行改动。
- `arms/fwd_r11.sha256`：r11 副本来源（fwd job 的 `op/current`，05:37 拷贝）。
- `data/`：本报告所有表格的原始汇总，以及 `prof_t1.clk.gz`（高频时钟）。
- `runs/*.log`：每个卡上进程的日志；kineto trace（16–66 MB）和 rows json 只留在本地，不提交。
- 训练产物是 run_e2e.sh 写的，在 `../e2e/`：`logs/e2e.{prof_s1_4L,prof_t1}.log`、`traces/prof_t1/iteration_*`。输入 dump 在 `/home/lihuzhan/_prof_dump/`（6 × 约 390 MB，不在 git 里）。
