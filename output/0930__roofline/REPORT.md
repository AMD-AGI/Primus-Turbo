# gfx1250（MI455X）attention roofline：microbench 实测（2026-09-30，A0 heliosr-1b114-c07-1）

形状：Llama-3.1-8B b4 s8192 hq32 hkv8 d128 bf16 causal。FLOP：fwd 2.199292e12，bwd 5.498229e12。
机器状态：09-29 刷新固件之后（VBIOS 700E，SMU 125.12.0），sclk 最高 2400 MHz（见 `../0928__a0_repro/REPORT.md` §6）。
代码在 `mb/`（`mb1.hip` 手写 HIP；`gen_mb{2,3,4,5}.py` 生成内联汇编 kernel；`mb6.hip` 测访存）；日志在 `runs/`；上卡脚本是 `tools/run_mb.sh`（锁、KFD 检查、时钟采样、dmesg 检查）。
本次共 13 个卡上进程，全部 rc=0，dmesg 无 GPU 错误（PMC 冠军测量的问题见 §6）。

## 0. 结论

| 参数 | 实测值 | 之前的说法 |
|---|---|---|
| **bf16 WMMA 16x16x32 发射周期 W** | **8.0 cycle**，即每 SIMD 每 cycle 2048 FLOP | 1024（错）/ 2048（推断，没实测过） |
| WMMA 依赖链延迟 | 16 cycle，每个 SIMD 需要 ≥2 条独立累加链才能跑满 | 未知 |
| 满载 WMMA 时的时钟 | **约 1.66–1.75 GHz**，整板约 2.1 kW（空闲 2.36 GHz）；整卡实测 **3.45 PF/s** bf16 | "1002.7 TF/s roof"（错） |
| v_exp_f32 | 约 2 cycle/条（每 SIMD 每 cycle 16 lane） | LLVM 表推断为 2 |
| v_fma_f32 / v_pk_fma_f32 | 1 cycle/条（32 / 64 lane） | |
| **LDS 读** | **每个 64 KB 段 256 B/clk/CU**；读分散到两段时 **512 B/clk/CU**；`ds_load_tr16_b128` 和 `ds_load_b128` 相同 | 未知（256 或 512） |
| **WMMA ↔ DS 切换** | **每次约 29 cycle，是每个 wave 自己的停顿**，同一 SIMD 上的第二个 wave 可以把它藏住 | 未知 |
| 同一 wave 内和 WMMA 同时发射 | 每条 WMMA 可藏约 4 条普通 VALU（每条 +0.2 cycle）、约 2 条 v_exp；超出后每条 VALU 约 +1 cycle，每条 exp 约 +1.6；packed 指令约 2 倍代价；v_nop 每条 +0.75 | 按"约 1790 个免费槽"估计（偏乐观） |
| HBM | **读 18.5、写 18.6、copy 16.6 TB/s**（标称 23.3） | 4.39 / 6.46（旧的限频 A0） |
| L2/MALL | 工作集 64–256 MB 时 36–37 TB/s，16 MB 时 60 TB/s | 未知 |
| 计数器 | `s_get_shader_cycles` = sclk cycle；realtime = 100 MHz；**`GRBM_GUI_ACTIVE`/8 = kernel 的 sclk cycle 数**（误差 1–3%）；`SQ_BUSY_CYCLES` = 32×GRBM | 时钟域未知 |

## 1. WMMA（`mb1`，256 WG × 1/2/4 wave/SIMD，占满全部 1024 个 SIMD）

| kernel | 1 wave/SIMD | 2 wave | 4 wave（按 event 时间） |
|---|--:|--:|--:|
| 1 条依赖链 | 16.02 cycle/WMMA | 9.02 | 8.00 |
| 2 / 4 / 8 / 16 条独立链 | 8.25 / 8.19 / 8.19 / 8.19 | 8.12 / 8.09 / 8.09 / 8.09 | 事件时间相同 |

- 多出来的 0.1–0.2 是每 16 条 WMMA 3–4 条的循环开销。gfx1250 只有 16x16x32 一种 dense bf16 形状，所以 **W = 8**。
- 满载时 kernel 内计数器给出 f ≈ 1.66–1.75 GHz（cycle/realtime 与 cycle/event 一致）；exp/fma 这类 kernel 能跑在 2.25–2.29 GHz。
  **WMMA 密集的 kernel 会把时钟拉到约 1.7 GHz**（整板约 2.1 kW），这就是 A0/B0 上 attention 的"功耗 roofline"。

## 2. 同一 wave 内的同时发射（`mb2`：每条 WMMA 后跟 N 条同类指令，8 条独立链，内联汇编精确控制顺序）

每条 WMMA 基线是 9.64 cycle（1 wave），这个循环里的回跳开销约 13 cycle/轮；下表是每条指令的**额外**代价（cycle/条，1 wave / 2 wave）：

| 指令 | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| v_fma / v_mul | 0.25 / 0.17 | 0.19 / 0.15 | 0.25 / 0.18 | 0.83 / 0.83 |
| v_add / v_max / v_cvt_pk_bf16 | 0.25 / 0.09 | 0.19 / 0.07 | 0.22 / 0.09 | 0.73 / 0.70 |
| v_exp_f32 | 0.25 / 0.10 | 0.38 / 0.12 | **1.22 / 1.08** | **1.61 / 1.57** |
| v_pk_fma / v_pk_mul | 0.25 / 0.72 | 0.31 / 0.59 | 0.50 / 0.54 | 1.02 / 1.00 |
| v_nop | **0.75** / 0.25 | 0.44 / 0.16 | 0.34 / 0.14 | 0.80 / 0.69 |
| s_add（SALU） | 0.12 / 0.05 | 0.19 / 0.07 | 0.22 / 0.08 | 1.11 / 0.95 |

- 每条 WMMA 的"影子"里能藏约 4 条普通 VALU、约 2 条 exp。超出这个量的指令基本按 1 条/cycle（exp 约 1.6–2 cycle/条）串行执行。
- 同一 SIMD 上 2 个 wave：N≤4 时代价降到约 0.1，但 N=8 时几乎不变，因为整个 SIMD 的发射带宽才是上限。
- `mb1 roles`：一个 wave 只做 WMMA，同 SIMD 的另一个 wave 只做 exp。两者几乎完全重叠：WMMA wave 的耗时不变，总时间只增加约 7%。

## 3. LDS（`mb3`、`mb4`、`mb5`）

纯读（每个 wave 16 条 `ds_load_b128`/`ds_load_tr16_b128`，dscnt lag 8–32）：

| 地址范围 | 1 wave/SIMD | 2 wave | 4 wave |
|---|--:|--:|--:|
| 所有 wave 都读同一个 64 KB 段 | **255.9** B/clk/CU | 255.8 | 280 |
| wave 按奇偶分到两个段 | 324 | **511.3** | 443 |

**WMMA 与 ds_load 交错**：lag 8/16/24/32 结果完全相同（每条 WMMA 37.94 cycle），说明不是带宽或延迟问题，而是发射层面的切换代价。拟合（`mb5`，14 种排法，1 wave）：

  `每段耗时 = 8·nW + 29 + 1·nD`（nW 条 WMMA 连续，后面跟 nD 条 ds_load；± 10%）

| 排法（W:D） | 1 wave cycle/WMMA | 2 wave |
|---|--:|--:|
| 1:1 | 37.9 | 19.4 |
| 2:2 | 23.9 | 12.1 |
| 4:4 | 17.5 | **8.9** |
| 8:8 | 13.8 | 10.5 |
| 16:1 | 10.7 | 9.0 |
| 1:4（ASM 的写法） | 41.0 | 33.1 |

- 在中间插一条 VALU 或 s_nop 没有影响；2 个 wave 时逐条交错正好快 2 倍，说明这是**每个 wave 的延迟**，而不是 SIMD 资源被占用。
- 对 kernel 设计的含义：1 wave/SIMD 时，LDS 读要攒成少数几段（ASM 的 "W DDDD" 排法每段都要付这个代价）；2 wave/SIMD 时可以互相遮盖。

## 4. 访存（`mb6`，4096 WG × 256 线程，grid-stride 循环，读用 4 路展开）

| 工作集 | 读 TB/s | 写 TB/s | copy TB/s |
|---|--:|--:|--:|
| 16 MB | 60.6 | 32.0 | 43.5 |
| 64 MB | 36.2 | 32.3 | 38.0 |
| 134 MB（prod 形状 K+V 总量） | 36.8 | 32.0 | 29.7 |
| 256 MB | 37.1 | 21.9 | 16.6 |
| 1–4 GB（HBM） | **18.3–18.8** | **18.6–18.8** | **16.6** |

## 5. 用实测参数重算 roofline（每 SIMD 的 cycle 数）

| 项 | fwd | bwd |
|---|--:|--:|
| 矩阵下限（算法所需，W=8） | 1.049e6 | 5-GEMM 2.622e6；7-GEMM（fly 的 split 做法）3.690e6 |
| fly 分块的矩阵下限 | 1.057e6（+0.77% 对角线） | — |
| exp | 每条 WMMA 约 1 条：在"每条 WMMA 可藏 2 条"的范围内，额外约 +3% | recompute，同量级 |
| 其他 VALU | 约 3 条/WMMA，在"可藏 4 条"的范围内，额外约 +8% | 约 2.5 条/WMMA |
| LDS | 每条 WMMA 读 512 B，正好是 1 个段的 256 B/clk/CU；K/V 分到两个段之后只占 50% | |
| 实际可达的下限 | **约 1.15–1.2e6 cycle** | 5-GEMM 约 2.9e6；7-GEMM 约 4.1e6 |
| 换算成时间 @1.7 GHz（满载时钟）/ @1.35 GHz（训练） | **0.68–0.71 / 0.85–0.89 ms** | 5-GEMM 1.7 / 2.15 ms；7-GEMM 2.4 / 3.0 ms |

L2 侧：ASM fwd 从 L2 读 17.7 GB 的 K/V，按 37 TB/s 算需要 0.48 ms；fly 读 8.66 GB，需要 0.23 ms，都不是瓶颈。
bwd k_dkdv 请求 71 GB，按 37 TB/s 算需要 1.9 ms，而 k_dkdv 本身约 4.3 ms，也就是 L2 带宽用了 45%，接近成为瓶颈之一。

## 6. 冠军在这把尺子上的位置（A0，prod，randn，rocprofv3 `--pmc GRBM_GUI_ACTIVE`，每个 kernel 17 次 dispatch 取中位数）

`runs/pmc/{fwd3,bwd3}`，`tools/pmcsum.py`；harness 同进程测得 r13ns 1.317 / ASM 1.262 ms，r29 6.856 / ASM 6.038 ms（profiler 开销使时间略长）。

### fwd

| kernel | cycle/SIMD | 占矩阵下限 1.049e6 | 占"实际可达下限" 1.15–1.2e6 | kernel 内有效时钟 | 时间（profiler 下） |
|---|--:|--:|--:|--:|--:|
| fly r13ns（m32x8，2 wave/SIMD，232 VGPR） | 1.985e6 | **53%** | 58–60% | **1.70 GHz** | 1.166 ms |
| ASM（1 wave/SIMD，512 VGPR） | 1.452e6 | **72%** | 79–83% | **1.36 GHz** | 1.070 ms |

- **ASM 比 fly 少用 27% 的 cycle，但有效时钟低 20%**：ASM 的 WMMA 更密、功耗更高，被功耗上限压了频率。所以按时间算只快 9%。
  这解释了 profile 报告里"ASM fwd 对 sclk 不敏感"的现象：它一直跑在功耗上限，时钟由功耗决定，而不是由 DPM 档位决定。
- 所以对 fwd 来说，**真正的 roofline 是"功耗 × 每 FLOP 能耗"**。在 cycle 上逼近 ASM，同时会把自己的时钟压下来；时间上的收益只有 cycle 收益的一部分。
  以 ASM 为参照：cycle 减少 27%，时钟降 20%，时间只快 9%。
- fly 还有 cycle 空间：1.985e6 → ASM 的 1.45e6 → 可达下限约 1.15e6。

### bwd

| kernel | cycle/SIMD | 下限 | 占下限 | 每条实际发射的 WMMA 的 cycle | 有效时钟 |
|---|--:|--:|--:|--:|--:|
| ASM 主 kernel | 7.71e6 | 5-GEMM 2.62e6 | **34%** | 23.2 | 1.64 GHz |
| fly k_dkdv（1 wave/SIMD，4G） | 7.26e6 | 2.10e6 | 29% | 27.6 | 1.76 GHz |
| fly k_dq（k_dqg，3G） | 4.00e6 | 1.57e6 | 39% | 20.2 | 1.80 GHz |
| fly 合计（+ k_delta 0.18e6） | 11.44e6 | 7-GEMM 3.69e6 / 5-GEMM 2.62e6 | 31% / 23% | 24.4 | |

- 每条 WMMA 的效率两边基本相同（24.4 对 23.2），fly 对 ASM 的差距主要来自 7 对 5 个 GEMM。但**两边都只到矩阵下限的约 1/3**：每条 WMMA 有 16 个 cycle 花在别处。
- **k_dkdv 每轮的 cycle 预算**（`tools/wd_switch.py` 对 `0927__flydsl/bwd/census/dump0341/champ/k_dkdv_0`：64 WMMA、80 DS、515 其他指令、7 次 WMMA→DS 切换）：

  | 项 | cycle/轮 | 依据 |
  |---|--:|---|
  | WMMA | 512 | W=8（§1） |
  | 其他 VALU/SALU | 约 360 | 其中约 256 条藏在 WMMA 影子里（每条约 0.2），其余约 1/条（§2） |
  | ds 指令 | 约 80 | |
  | WMMA↔DS 切换 | 约 203 | 7 × 29（§3），1 wave/SIMD，无法遮盖 |
  | 循环回跳 | 约 13 | §2 |
  | **模型合计** | **约 1170** | |
  | 实测 | **1766** | 7.26e6 / 4112 轮 |
  | 未解释（主要是全局预取的 loadcnt 等待） | 约 600 | 与 facts 里 r18 的"一个 loadcnt 等待点约 470 cycle"一致 |

  也就是说，k_dkdv 的 1766 cycle 大致是 WMMA 512 + 发射 450 + 切换 200 + 访存等待 600。
  一旦能做到 2 wave/SIMD，切换和等待这两项（约 800 cycle，占 45%）都有可能被遮盖；但 1 wave/SIMD 是受 VGPR 限制的现状（`bwd-history` 的 4-wave barrier 问题）。
- fwd 对照：fly 的 fwd 热循环每轮只切换 2 次（32W / 32D 一段），而且是 2 wave/SIMD，切换代价能被遮盖。

## 7. 对优化方向的含义

1. **fwd**：cycle 上还有约 1.7 倍的空间（1.985e6 → 约 1.15e6），但时间上的收益会被功耗打折。
   一个 cycle 更少的 kernel 会把时钟拉低（ASM：cycle −27%，时钟 −20%）。要按 **cycle 和 J/FLOP** 两把尺子一起评估，而不是只看时间。
   具体方向：v_nop 每条代价 0.75，最值得消除；每条 WMMA 超过 4 条的 VALU 和超过 2 条的 exp 基本要串行付费，所以 softmax 的 VALU 要和 WMMA 均匀交错。
2. **bwd**：每条 WMMA 23–24 cycle 对 8，差距里约 45% 是切换和访存等待。它们只在 1 wave/SIMD 时才暴露，所以主要杠杆是
   (a) 在 k_dkdv 里减少 WMMA↔DS 切换（把 DS 攒成段，7 次降到 2 次可省约 145 cycle/轮，约 8%），
   (b) 让预取等待不落在关键路径上，
   (c) 想办法做到 2 wave/SIMD。
   7 对 5 个 GEMM 的结构差距（1.4 倍）另算。
3. **LDS**：fwd 每条 WMMA 读 512 B，正好是单个段 256 B/clk/CU 的上限。K 和 V 必须放在不同的 64 KB 段，否则 LDS 就会和矩阵一起成为瓶颈。可以检查 fly fwd 的 K/V 布局。

## 8. PMC 冠军测量踩的坑

harness 的 warmup 在 1 秒内不做同步地连续提交 kernel，rocprofv3 `--pmc` 下每次 dispatch 都要串行采集计数，于是积压几万次 dispatch，看起来像卡死（两次 timeout，卡没有问题）。解决办法：`--warmup-seconds 0`。

## 9. M8：fwd 冠军的光速消融（2026-09-30，A0）

arm 是 r13ns 的拷贝，每个只加一个开关（`m8/build_arms.py`，每处改动都用断言确认唯一命中；不动任何 index 或地址计算；输出结果本来就是错的，只看时间）。
所有 arm 都通过了 compile-only 检查（0 spill，0 scratch），ISA 统计确认开关生效（`m8/dump/*`：noexp 和 nosm 的 exp 为 0，nobar 的 barrier 从 12 降到 4，WMMA/DS 条数全部保持 256/304）。
toy（proxy 形状）单独一个进程，开 `AMD_SERIALIZE_KERNEL=3`，全部通过；prod 分计时 / 非 causal / PMC 三个进程，全部 rc=0，dmesg 干净。
cycle 数用 `m8/m8drive.py` 测：每个 arm 前先发一个 grid 不同的标记 kernel，用来切分 dispatch 序列；每个 arm 丢掉前 3 次，取 7 次的中位数（`m8/m8sum.py`）。

| arm | 改动 | cycle/SIMD（PMC） | 相对 base | 有效时钟 | harness 计时 ms |
|---|---|--:|--:|--:|--:|
| base | r13ns 原样 | **1.979e6** | 1 | 1.46 GHz | 1.377 |
| noexp | exp2 换成一条 v_mul | 1.936e6 | −2.2% | 1.60 | 1.333 |
| nosm | 去掉 max/exp/sum/rescale，P = cvt(S) | 1.612e6 | **−18.6%** | 1.62 | 1.113 |
| nomask | 所有 tile 走无 mask 的主体 | 1.978e6 | −0.1% | 1.61 | 1.358 |
| nobar | 每个 tile 去掉 WG barrier（保留 TDM wait） | 1.883e6 | −4.9% | 1.61 | 1.307 |
| wl | nosm + nomask：只剩 WMMA + LDS + TDM + barrier | **1.603e6** | −19.0% | 1.61 | 1.118 |
| ASM | | 1.430e6 | 0.722 | **1.24** | 1.264 |
| base，非 causal | 2 倍 FLOP | 3.814e6（矩阵下限 2.097e6 的 55%） | | 1.58 | 2.704 |
| ASM，非 causal | | 2.490e6（**84%**） | | 1.20 | 2.459 |

**fly fwd 的 cycle 构成**（每 SIMD，合计 1.979e6）：

| 项 | cycle | 占比 | 依据 |
|---|--:|--:|---|
| 矩阵下限（fly 分块） | 1.057e6 | 53% | W=8 |
| **骨架开销**：LDS/TDM/同步/prologue/epilogue | **0.546e6** | **28%** | wl − 矩阵下限 |
| 其中 per-tile barrier | 约 0.096e6 | 5% | nobar |
| 其中 causal 的不均衡和尾部 | 约 0.07e6 | 4% | base − 非 causal/2 = 1.979 − 1.907 |
| **softmax**（max/sum/cvt/rescale/permlane 及依赖等待） | **0.376e6** | **19%** | base − nosm |
| 其中 exp 本身（trans） | 约 0.043e6 | 2% | noexp |
| mask | 约 0 | 0% | nomask |

结论：
1. **最大的一块不是 softmax，而是骨架**：去掉全部 softmax 之后（wl，1.603e6），fly 仍然比**完整的** ASM（1.430e6）多用 12% 的 cycle。
   ASM 的非 causal 主体能跑到矩阵下限的 84%，fly 只有 55%。
   fly 每个 SIMD 2 个 wave，每个 tile 每个 wave 从 LDS 读 32 KB（每条 WMMA 512 B），刚好顶到单个 LDS 段的 256 B/clk/CU（§3）；再加上 TDM 写入，LDS 需求约 288 B/clk。
   K 和 V 的 ping-pong 缓冲是相邻的（`_v_lds_buf = K base + k_blk_bytes`），很可能落在同一个 64 KB 段里，这一点下一步要核实。
   如果确实在同一段，**把 K 和 V 分到不同的段**，理论上能把 LDS 从瓶颈上拿掉。
2. **softmax 占 19%，但 exp 本身只占 2%**：这和 §2 的结论一致（每条 WMMA 后面跟 1–2 条 exp 几乎免费）。softmax 的代价在 max/sum 的归约树、permlane、cvt 和 rescale 这些 VALU，以及它们和 WMMA 之间的依赖等待。
3. per-tile barrier 占 5%，causal 的不均衡占 4%，mask 基本免费。
4. **功耗和时钟**：ASM 的有效时钟只有 1.2–1.24 GHz，fly 各 arm 是 1.46–1.62 GHz。所以 wl 在时间上已经比 ASM 快 14%（0.993 对 1.152 ms），但在 cycle 上仍比 ASM 慢 12%。
   减少 cycle 的改动会提高 WMMA 密度，进而拉低时钟；这一点在评估时必须和 cycle 一起看。
