# A0 复现 B0 今天的 fwd / bwd 相对提升（2026-09-28，heliosr-1b114-c07-1，fa-repro）

分支 `dev/lhz/flydsl-attn-b0` @ `cf33d10d`（只 checkout，没有提交）。ssh 到 B0 因 host key 变更失败，没有绕过；
全部树来自 git（`output/0927__b0/champions/` 等）和 A0 本地（bwd r20 = A0 job `rounds/020/op`，
`bwd_r20_0341` = `output/0925__flydsl/bwd341/op0341`），md5 与 HANDOFF 一致。没有真实 q/k/v dump
（`/home/lihuzhan/_prof_dump` 不在 A0），所以 op 级只有 randn 尺子（ruler 1）。

A0 状态：sclk 只有 500/1100 MHz 两档（VR 限频），负载下 ~1010–1060 MHz；B0 分块尺子下约 2155 MHz。

## 1. 单 kernel（blocked ruler，prod 每种顺序一个进程，A/A 副本同进程）

### fwd（ms，越小越快；比值为同进程）

| shape | base | A0 r11 | r6 | r13 | **r13ns（今天）** | ASM | r13ns / r6 | r6 / ASM | r13ns / ASM | A/A |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prod o1 | 2.395 | 2.008 | 1.919 | 1.919 | 2.023 | 1.564 | 1.054 | 1.227 | 1.293 | 0.15% |
| prod o2（反序） | 2.329 | 1.977 | 1.889 | 1.889 | 1.992 | 1.543 | 1.055 | 1.224 | 1.291 | 0.01% |
| proxy | 0.2474 | 0.1480 | 0.1449 | 0.1440 | 0.1486 | 0.1330 | 1.026 | 1.089 | 1.117 | 0.1% |
| fast | 0.0404 | 0.0327 | 0.0441 | 0.0358 | 0.0320 | 0.0303 | **0.727** | 1.454 | 1.057 | 0.2% |

与 B0 对照：
- r6 相对 baseline：A0 +23–25%，B0 约 +23% —— **一致**。
- r13 相对 r6（prod）：持平 —— **一致**。fast 上 r13 比 r6 快 23%（B0 r13 fast 是 r6 的 1.66x）—— 方向一致，幅度小。
- r13ns 相对 r13（randn prod）：A0 慢 5.4%，B0 慢 4.8% —— **一致**（关投机在 randn 上本来就亏，收益只在真实数据上）。
- **FlyDSL vs ASM 差距 A0 大得多**：r13/ASM A0 1.225 vs B0 1.03；r13ns/ASM A0 1.29 vs B0 1.08。
  A0 的 1.29 恰好等于 B0「真实数据 + GEMM burst（训练时钟 ~1300 MHz）」下的 1.25–1.29，符合 fwd 对时钟敏感的结论。

### bwd（ms）

| shape | r20 | r19h | **r29（今天）** | ASM | r19h / r20 | r29 / r20 | r20 / ASM | r29 / ASM | A/A |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prod o1 | 10.600 | 10.826 | 11.010 | 7.654 | **1.021** | **1.039** | 1.385 | 1.438 | 0.08% |
| prod o2（反序） | 10.605 | 10.837 | 11.015 | 7.666 | **1.022** | **1.039** | 1.383 | 1.437 | 0.01% |
| proxy | 0.7818 | 0.7770 | 0.7959 | 0.6016 | 0.994 | 1.018 | 1.300 | 1.323 | 0.3% |
| fast | 0.0949 | 0.0946 | 0.0941 | 0.1099 | 0.997 | 0.991 | 0.864 | 0.856 | 0.2% |

**B0 的 bwd 提升在 A0 上不成立，方向相反**：B0 r19 比 r20 快 1.7%、r29 比 r19h 再快 2.7%（prod 80.4% of ASM）；
A0 上 r19h 慢 2.2%，r29 慢 3.9%（69.5% of ASM；r20 72.2%）。两种顺序一致（差 <0.1%），ASM 两进程一致。

逐 kernel（`tools/kbench.py prod blk`，sclk ~1010 MHz）：

| arm | k_dq 单独 | 整个 op | 整个 − k_dq（≈ k_dkdv + k_delta） |
|---|--:|--:|--:|
| r20 | 3.571 | 10.509 | 6.94 |
| r19h | 3.570 | 10.747 | 7.18（+3.4%） |
| r29（u2n） | 3.872（**+8.5%**） | 10.924 | 7.05 |

- u2n（k_dq kv 循环 unroll x2）在 A0 上让 k_dq 慢 8.5%（B0 训练时钟下 −7.8%）。
- r19h 的 k_dkdv 部分慢约 3%（B0 r19 快 1.7%）。
- 注意：A0 在 ~1010 MHz 下 r20 的 k_dq 是 3.57 ms，比 B0 在 ~1300 MHz（GEMM burst）下的 4.05 ms 还快，所以
  A0/B0 的差别不能只用 sclk 解释（功耗/内存侧工作点不同），这点还没查清。

## 2. e2e（Llama-3.1-8B 32L，MBS=GBS=4，seq 8192，nkfix 开，同进程 3 arm 按步交替）

arm：asm；old = fwd r6 + bwd r20（B0 早上 e2e 的起点）；new = fwd r16(r13ns) + bwd r29（今天的冠军）。
schedule `asm,old,new,asm,new,old`（每对 arm 两种相邻顺序都有），90 步，每 11 步 profile 一次（三个 arm 都覆盖）。

| 进程 | 状态 | old/asm | new/asm | **new/old** | step ms asm / old / new |
|---|---|--:|--:|--:|--:|
| a0_p3a | 有效，loss 全程有限（12.25951 → 3.50509） | 1.0671（相邻 17 对） | 1.0647（16 对） | **0.9987**（17 对；4 个完整周期 0.9990） | 1947 / 2070 / 2071 |
| a0_p3b（反序） | **作废**：step 4（new arm）grad_norm inf，step 5 起 loss nan，86 步 nan | — | — | — | — |
| a0_p4a（2x2 分解 fwd×bwd） | **挂卡**：step 2（old arm 第一次调用）hipErrorLaunchFailure → MES 不可恢复 | — | — | — | — |
| B0 对照 | | 1.0475 / 1.0500 | 1.0323 / 1.0328 | ≈0.985 | 1579 / 1656 / 1631 |

- **e2e 上 A0 看不到今天的提升**：new/old = 0.999（B0 约 0.985，即 −1.5%）。这和 op 级结果一致：
  fwd 的 r13ns 在真实数据上应该赚（B0 约 −10 ms/步），但 bwd r29 在 A0 上比 r20 慢约 4%（op 级 +0.4 ms/调用 × 32 层 ≈ +13 ms/步），两者大体抵消。
  因为 A0 的 kineto trace 每个 profile 步只记录了 4 个 GPU kernel（CPU 侧 `e2e::attn_*` 范围都在），trace 拆不出 fwd/bwd，
  所以准备用 2x2 分解（`new/old` = fwd r16 + bwd r20，`old/new` = fwd r6 + bwd r29）直接量——这一进程在第 2 步挂卡。
- 单步 FlyDSL/ASM 差距 A0 6.5–6.7%，B0 3.2–5.0%，与 op 级 A0 差距更大一致。

## 3. 卡的事件

- 11:55:12 起 a0_p4a，step 1（asm）正常；step 2 是 old arm 第一次调用 → `hipErrorLaunchFailure` → dmesg
  `MES failed to respond ... MES might be in unrecoverable state, issue a GPU reset` / `GPU reset begin!`。
  属于不可恢复类，需要 AC 断电。KFD 里还留着 2 个进程（80337、80661），没有去 kill，也没有再探测卡。
- old arm 在 p3a/p3b 已经跑了几十步都正常；前一个进程 p3b 出现了不明原因的 inf/NaN。更像卡的状态先坏（NaN）后挂，
  而不是 old arm 代码越界。但不能排除。
- 另外：dmesg 里有开机时记录的上一次启动的 CPU 致命 L3 取指错误（BERT，Uncorrected，Processor Context Corrupt），
  以及本次开机后约 11:24 一条 CPU 8 的 L3 corrected MCE —— 这台机器的 CPU 侧也在报硬件错误，建议告诉机器管理员。
- 本次共 9 个 op 级卡进程 + 1 个 opcheck + 3 个训练启动；挂在第 3 个训练启动。

## 4. 文件

- `tools/run_op.sh`（锁 /tmp/a0-gpu0.lock、每进程新 JIT 缓存、sclk 采样、dmesg 检查），`tools/kbench.py`（sclk 指向 card1），
  `tools/e2e_arms3.py`（N arm 稳态分析），`run_e2e_a0.sh`（B0 launcher 的 A0 副本：fa-repro、card1、dmesg 过滤），`e2e_drive.sh`。
- `arms/`：fwd_{base,a0r11,r6,r13,r13ns,r13ns_aa}，bwd_{r20,r19h,r29,r29_aa}；e2e arm 在 `output/0927__b0/e2e/arms/`（gitignored）。
- `runs/`：每个进程的 log / json / clk；e2e 日志在 `output/0927__b0/e2e/logs/e2e.a0_p3{a,b}.log`、`e2e.a0_p4a.log`。

## 5. 第二次开机（12:10）之后：2x2 分解 e2e 与第二次挂卡

### 5.1 a0_p4a（24 层，100 步，无 profile，12:24–12:27，干净）

显存峰值 70.98%（32 层时 89%）；loss 全程有限（12.21202 → 3.50218），nkfix `nonfinite_events: []`，dmesg 无 GPU 行。

| arm（fwd / bwd） | 单步 ms（中位数，n=18–19） | / ASM |
|---|--:|--:|
| ASM | 1483.4 | 1 |
| old（r6 / r20） | 1578.8 | 1.064 |
| new（r16 / r29，今天冠军） | 1579.7 | 1.065 |
| r16 / r20 | **1569.9** | **1.058** |
| r6 / r29 | 1590.5 | 1.072 |

- fwd r6 → r16(r13ns)：−8.9 / −10.8 ms/步（9 个完整周期比值 0.9936 / 0.9937）—— **与 B0 一致**（B0 32 层约 −10 ms）。
- bwd r20 → r29：+11.7 / +9.8 ms/步（1.0062 / 1.0063）—— **与 B0 相反**（B0 −15 ms），与 op 级 +0.41 ms/调用 × 24 层 ≈ +9.8 ms 吻合。
- 两者抵消 → new/old = 1.000。A0 上最快的组合是 fwd r16 + bwd r20。

### 5.2 a0_p4b 挂卡（12:32）

- 12:32:33 启动（p4a 结束后 KFD 为空并冷却 300 s）；12:32:40 Training starts；12:32:44 第一次 attention 调用
  （step 1 = `old/new`，即 fwd r6 + bwd r29，进程里第一次用 FlyDSL）；**12:32:50** dmesg
  `MES(0, 0) failed to respond to msg=INVALIDATE_TLBS` → `MES might be in unrecoverable state` → `GPU reset begin!` →
  最后 `wait for reset ack`。不可恢复，需要 AC 断电。KFD 残留 11714、12041，没有 kill。

### 5.3 原因分析（两次开机共 5 个训练进程）

| 进程 | 层数 / 显存峰值 | 第一个 attention arm | 首次 attention 时的 HIPBLASLT_TENSILE_LIBPATH | 结果 |
|---|---|---|---|---|
| a0_p3a（开机 1 第 1 个训练） | 32 / 89.0% | asm（FlyDSL 在 step 2 才首次调用） | 镜像库 | 正常 |
| a0_p3b（第 2 个） | 32 / 88.6% | **new（FlyDSL 在 step 1）** | **宿主机库 ~/.local/hipblaslt-gfx1250** | step 4 grad inf，之后 86 步 NaN；非有限值全在 bwd GEMM（dgrad/wgrad）输出 |
| a0_p4a 32L（第 3 个） | 32 / 89% | asm | 镜像库 | step 2（old 第一次调用）hipErrorLaunchFailure → **挂卡** |
| a0_p4a 24L（开机 2 第 1 个） | 24 / 71.0% | asm | 镜像库 | 正常 |
| a0_p4b 24L（第 2 个） | 24 / ~71% | **old/new（FlyDSL 在 step 1）** | **宿主机库** | step 1 **挂卡** |

1. **最符合证据的原因：FlyDSL fwd 树的 `_env.py` 在 import 时把 `HIPBLASLT_TENSILE_LIBPATH` 改成宿主机库**
   （`e2e/arms/fwd_r6/_env.py:26`，r16 同样）。fwd arm 在 step 1 就被加载时，这个改动发生在 step 1 的反向之前；
   反向 GEMM 可能用宿主机库（2026-08-07 的构建，"只验证过方阵 GEMM"）去加载新的 kernel。B0 的 run_e2e.sh 注释里记录过：
   用宿主机库的第一次运行 p1a_turbo 在 step 1 挂住。
   两个"FlyDSL 在 step 1"的进程一个 NaN（坏值恰好出在 bwd GEMM）、一个挂卡；两个"asm 先跑"的进程都正常。
   **没有验证的部分**：hipBLASLt 是否会在进程中途重新读这个环境变量（例如反向线程新建 handle 时）没有在卡上证实；
   B0 的 fin_p2b（FlyDSL 先跑）同样的设置却正常，可能 B0 的宿主机库不同。
2. p4a-32L 是 asm 先跑却也挂了：它紧跟在 p3b（86 步 NaN）之后，显存 89%（A0 以前在 88.30% 时 SIGBUS 并需要 AC 断电）。
   更像是"上一个进程把卡弄坏了 + 显存太满"，不能归到原因 1。
3. **宿主机 CPU 硬件错误**（独立问题，要报给机器管理员）：CPU 8 的 MC60 bank 一直在报 corrected L3 错误（mem/io: IO），
   与 GPU 无关（amdgpu 加载之前就有）：09-25 那次开机 163 次，09-28 02:15 开机 19 次，08:13 开机 16 次，本次开机 2 次。
   **08:13 和 02:15 两次开机都记录了上一次运行的 fatal 错误**（BERT：CPU L3 取指 Uncorrected，Processor Context Corrupt），
   即今天早上的重启是主机崩溃。两次挂卡前 3.7 min / 46 s 各有一次 MCE，但 MCE 平均每 5–30 min 一次，这个时间接近可能是巧合。
4. 已排除：显存（p4b 只有 ~71%，而且死在 step 1）；启动过快（KFD 为空并冷却 120/300 s）；JIT 缓存串用（每个进程新目录）。

### 5.4 下次开机后怎么跑（需要确认）

- e2e 用的 fwd 树副本（`output/0927__b0/e2e/arms/fwd_*`，gitignored，不是冠军源码）去掉 `_env.py` 里对
  `HIPBLASLT_TENSILE_LIBPATH` 的改写（fwd attention 本身不用 hipBLASLt），并在 shim 里断言整个进程 GEMM 只用镜像库。
- schedule 固定让 asm 先跑 1 步；24 层；每个开机只跑 1 个训练进程后先看 dmesg/MCE 再决定下一个。
- 还没跑完的只有 p4b（反序的 2x2 分解，用来确认 p4a 的结论）。p4a 的结论已经清楚，p4b 可以不跑。

## 6. 2026-09-30 复测：A0 换固件/驱动后（heliosr-1b114-c07-1，fa-repro）

与 §1 完全相同的命令、arm 目录和 blocked 尺子（`runs/*_0930.*`）；arm md5 未变（r13ns m32x8 `370769c9`，r29 kernels `37f37052`）。
5 个卡进程全部 rc=0，dmesg 无 GPU 错误行，KFD 前后为空。随机输入（ruler 1），没有真实 dump。

### 6.1 fwd prod（ms；o1 / o2 两种顺序，A/A 0.07% / 0.02%）

| arm | 09-28 | **09-30** | 提速 | 09-30 / ASM |
|---|--:|--:|--:|--:|
| base | 2.395 / 2.329 | 1.414 / 1.413 | 1.69× / 1.65× | 1.125 / 1.128 |
| A0 r11 | 2.008 / 1.977 | 1.333 / 1.329 | 1.51× / 1.49× | 1.061 / 1.060 |
| r6 | 1.919 / 1.889 | 1.292 / 1.300 | 1.49× / 1.45× | 1.028 / 1.037 |
| r13 | 1.919 / 1.889 | 1.296 / 1.302 | 1.48× / 1.45× | 1.031 / 1.039 |
| **r13ns（冠军）** | 2.023 / 1.992 | **1.353 / 1.356**（1626 / 1622 TF/s） | **1.50× / 1.47×** | **1.076 / 1.082** |
| ASM | 1.564 / 1.543 | **1.257 / 1.253**（1750 / 1755 TF/s） | 1.24× / 1.23× | 1 |

sclk 见证 1757–1928 / 1992–1902 MHz（09-28 为 1020–1061）。r13ns/ASM 从 1.29 降到 1.08，与 B0 blocked 尺子的 1.08 一致。

### 6.2 bwd prod（ms；A/A 0.06% / 0.02%）

| arm | 09-28 | **09-30** | 提速 | / r20 | / ASM |
|---|--:|--:|--:|--:|--:|
| r20 | 10.600 / 10.605 | 6.571 / 6.577 | 1.61× | 1 | 1.194 / 1.197 |
| r19h | 10.826 / 10.837 | 6.575 / 6.584 | 1.65× | 1.001 / 1.001 | 1.195 / 1.198 |
| **r29（冠军）** | 11.010 / 11.015 | **6.583 / 6.596**（835 / 834 TF/s） | **1.67×** | 1.002 / 1.003 | **1.196 / 1.201** |
| ASM | 7.654 / 7.666 | **5.503 / 5.494**（999 / 1001 TF/s） | 1.39× | | 1 |

sclk 1872–2031 / 2024–1980 MHz。r29 达到 ASM 的 83.6% / 83.3%（09-28 A0 69.5%，B0 80.4%）。
09-28 A0 上 r19h/r29 比 r20 慢 2.2% / 3.9% 的现象消失了，三者现在相差在 0.3% 以内；B0 上的提升（约 −4%）在 A0 上同样没有出现。

逐 kernel（`kbench.py prod blk`，sclk 约 1740 MHz）：

| arm | k_dq | 整个 op | 整个 − k_dq |
|---|--:|--:|--:|
| r20 | 2.217（09-28 3.571） | 6.503（10.509） | 4.286 |
| r19h | 2.214 | 6.520 | 4.306 |
| r29 | 2.290（**+3.3%**，09-28 +8.5%） | 6.544 | 4.255（−0.7%） |

### 6.3 A0 改了什么（依据：dpkg.log、`last`、dmesg、以往 session 的 dmesg 摘录）

代码和尺子都没变，变的是平台：

| | 09-28 复现时 | 现在 | 何时 |
|---|---|---|---|
| VBIOS | `113-M4500001-630A`，build 00195379，2026/07/25 | **`113-M4500001-700E`，build 00204667，2026/09/25** | 09-29 06:25 UTC 的 dmesg 还是 630A，之后被刷新 |
| SMU fw | 125.7.1（0x027d0701） | **125.12.0（0x027d0c00）** | 随 VBIOS/固件包 |
| amdgpu-dkms（+ firmware 包） | 7.1.1.31300009-2397345 | **7.1.0.31300009-2411946** | 09-28 19:40 换成 -2410994，09-29 16:09 换成 -2411826，21:28 换成 -2411946（21:41、22:13 又重装） |
| sclk DPM 档位 | 500 / 1100 | **500 / 2356 / 2400** | |
| fclk DPM 档位 | 1100 | **1250 / 1900** | |
| mclk | 1900 | 1900 | 未变 |
| 负载下 sclk | 1010–1060 MHz | 1740–2030 MHz | |

- 操作人：用户 `asierrag`，09-28 19:02–21:40 和 09-29 16:07–21:12 多次登录（10.236.11.241）。
  09-29 19:08–22:19 之间主机重启约 10 次，符合刷固件加多次冷启动的过程。
- 开机时的 `WARN: GPU is throttled, expect performance decrease. VR.` **仍然会打印**，但 DPM 表不再被截断到 1100 MHz。
  所以应该是新的 VBIOS/PMFW 改变了限频处理方式，不是 VR 硬件本身修好了。
  驱动和 VBIOS 在同一时段一起换，不能严格分开归因。不过 DPM 档位表由 PPTable/PMFW 决定，所以以 VBIOS/PMFW 为主因。
- 提速小于时钟比（约 1.75×）：ASM fwd 1.24×，bwd 1.39×；FlyDSL fwd 1.48×，bwd 1.67×。这符合 profile 报告的结论：FlyDSL 对 sclk 更敏感，ASM fwd 基本不随 sclk 变化。
- 旁注：本次开机以来 dmesg 有 56 次 `pcie_pl` 可纠正错误（非致命），CPU 侧 L3 MCE 仍在报。

### 6.4 影响

- A0 现在的工作点接近 B0 的 blocked 尺子。09-28 的 A0 绝对数、"A0 上 bwd r19h/u2n 退化"的结论都属于旧的 1100 MHz 状态，作废。
- 可以恢复在 A0 上跑 job 和 e2e。A0 上 bwd 冠军可以用 r29（与 r20 持平），fwd 用 r13ns。
- skill 与 HANDOFF 里的 "A0 VR 限频 ~1.0–1.1 GHz" 需要更新。
