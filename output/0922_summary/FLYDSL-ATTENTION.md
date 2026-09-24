# 基于 FlyDSL 的 attention 优化总结

数据截止 2026-09-22 16:10 UTC（作业已停、节点已交还）· 机器 A0 `heliosr-1b114-c07-1`
（单卡 gfx1250 / MI455X，VR 限频 1100 MHz，窗口内漂到 967–1069 MHz）

---

## 0. 环境（汇报时要点名的那几项）

| 项 | 值 |
|---|---|
| 宿主 | `heliosr-1b114-c07-1`（A0），单卡，`gpu_pool` 为空 → deep round 无法并行 |
| 容器 / 镜像 | `fa-repro` / `fa-tune:deps` —— 即 `amdprimus/amdprimus:gfx1250-20260910` **+1 层**（本地派生，建于 2026-09-13） |
| torch | `2.11.0+rocm7.14.0a20260625`，torch_hip `7.14.60850`，rocprofv3 `1.3.2` |
| **flydsl** | 镜像自带 **0.2.4**；实际使用 **0.3.2**，侧装在 `~/.local/flydsl032` 并 prepend `sys.path`，**未覆盖镜像** |
| aiter | `/home/lihuzhan/code/aiter-src` @ `6963ae9d`（shallow clone） |
| Primus-Turbo | 分支 `dev/lhz/flydsl-attn`，从 main `c1325c7e` 起 |
| 模型 / 配置 | Llama-3.1-8B：d=128、GQA ratio=4、causal bottom-right、BSHD、bf16、无 sink、无 sliding window |
| shape（三个都计分） | fast `1×1024×1024×8/2×128`；proxy `1×4096×4096×32/8×128`；**prod `4×8192×8192×32/8×128`** |
| 环境变量 | `TORCH_BLAS_PREFER_HIPBLASLT=0`（镜像缺 gfx1250 的 hipBLASLt Tensile 库） |

> **版本冲突是硬的**：aiter 的 gfx1250 前向要 flydsl **0.3.2**，而 0.3.2 删掉了
> `flydsl.expr.buffer_ops`，Primus-Turbo 自己的 FlyDSL 树无条件 import 它 →
> **0.3.2 下 `import primus_turbo` 直接抛异常**。本作业的 baseline 因此只 import torch + aiter，
> **绝不 import primus_turbo**。

---

## 1. 前向：先做了一次"值不值得做"的判定

拿 aiter 自己的 gfx1250 FlyDSL 前向 vs 它自己的预编译 ASM 前向，**两边都成熟**，ABAB 交替、
每次 256 MiB L2 flush、n=20、带时钟见证：

| 臂 | median | median TF/s | best | best TF/s | sd | SQNR out |
|---|--:|--:|--:|--:|--:|--:|
| FlyDSL | **2.3732 ms** | **926.6** | 2.3282 | 944.5 | 2.68% | 51.63 dB |
| ASM | **1.5691 ms** | **1401.5** | 1.5552 | 1414.0 | 1.09% | 53.62 dB |

（前向 FLOP = `2·b·hq·s²·d` = 2.199e12，生产形状 `b=4 s=8192 hq=32 hkv=8 d=128` causal。
**FlyDSL 前向当前就是这个数：2.3732 ms / 926.6 TF/s** —— 这是 aiter 自己的 FlyDSL 前向，
我们没有对它做任何优化，测它只是为了判定要不要走这条路。）

**ASM 快 1.51×。** 测量链交叉验证：ASM 这次 1.5691 与 0915 冠军的 1.572 差 0.2%，说明两组数可并列。
顺带确认 LSE 是**自然对数**（ratio 1.0000，81.85 dB），与 fused backward 配对无需转换。

**决策**：落在 `PLAN.md` 的中间档 → **继续，但按可维护性而非性能立项**。
FlyDSL 前向本身**不做优化**，前向已由 ASM 拿下。剩下的立项理由只有结构性三条：
ASM 反向的 GQA 越界写、gfx1250 反向资产只有 gfx950 的 1/25（6 vs 124 个 `.co`）、
**gfx1250 完全没有 varlen ASM 反向**（第三条是"有没有"，不是"快多少"）。

---

## 2. 反向：三个内核从零写出来（2026-09-17）

gfx1250 **没有 MFMA**，gfx950 的 FlyDSL 反向模板不可移植：
`llvm.amdgcn.mfma.f32.32x32x16.bf16`、`ds.read.tr16.b64`、`permlane32.swap` 在 gfx1250 上
全部 cannot-select；LDS 转置要换成 `ds_load_tr16_b128`，**寄存器落位不同**。
Primus-Turbo 自己的 FlyDSL attention 树里有 **6 个 kernel 各自硬断言 gfx950**。
所以是**逐 kernel 重写**，不是改一个 arch 判断。

| 内核 | 作用 | 首次验证 SQNR |
|---|---|--:|
| `odo_gfx1250.py` | `delta = rowsum(dO*O)` | 159.24 / 156.49 dB |
| `dkdv_loop_gfx1250.py` | dV + dK，运行时 q 循环 + causal | 141.9 / 141.0 dB |
| `dq_gfx1250.py` | `dQ = dS·K` | 144.90 dB |

**三个都是一次通过**，每个都带 NaN 预填 + 全 `isfinite` 覆盖检查再算 SQNR。
写之前先用探针把三件事钉死：WMMA fragment 布局（对 torch 验证）、跨 d-tile 的累加链、
`ds_load_tr16_b128` 的语义。并顺手发现 gfx942 模板的"免费操作数"技巧
**在 dq 上成立（151.32 dB 实测）、在 dkdv 上不成立** —— 方向不同结论不同。

顺带修掉 main 上两个真实缺陷：flydsl 名义可选却被无条件 import（缺它整个
`import primus_turbo.pytorch` 失败，连 gfx1250 唯一可用的 Triton 路径也被拖下水）；
`_flydsl_common_ok` 用 `>= _GFX950` 判断而 gfx1250 报 `(12,5)` —— **它能通过**，只是碰巧被 GQA 门挡住。

---

## 3. op-evolve 自动优化：round 0 → round 8

作业 `gfx1250-flydsl-attn-bwd-20260917-115934`，`max_rounds: 12`，**单卡、全 fast round**。
判据：同一 session 内回文交错重测，**吞吐提升即接受**；精度 dq/dk/dv **各自** ≥ 50 dB；
**无 atomic、200 次逐位确定性**（这是源码版反向的立项理由之一，必须守住）。
对标 `beat` = aiter 预编译 ASM 反向（`dkdv_heads="q"` + host 规约），**每轮同 session 重测**，
因为这张卡会漂（1100 → 967 MHz 在一个计时窗口内发生过）。

**最终状态**：`best_round 8`，`op/current` 是 round 8 的码；round 9 已跑但**未记账**；
节点于 09-22 16:10 交还他人（`.stop` 哨兵已写、持卡令牌已删、容器已停）。

### 3.1 prod 形状曲线（bwd only，FLOP = 5·b·hq·s²·d = 5.498e12）

| round | prod ms | TFLOP/s | 对基线 | 对 ASM(beat) | 同 session gain |
|---|--:|--:|--:|--:|--:|
| 0（基线） | 96.05 | 57.2 | — | 0.081 | — |
| 1 | 59.36 | 92.6 | 1.62× | 0.131 | 1.342× |
| 2 | 38.39 | 143.2 | 2.50× | 0.203 | 1.921× |
| 3 | 26.50 | 207.5 | 3.63× | 0.294 | 1.297× |
| 4 | 16.61 | 331.0 | 5.78× | 0.470 | 1.512× |
| 5 | 16.39 | 335.4 | 5.86× | 0.476 | 1.154× |
| 6 | 15.52 | 354.3 | 6.19× | 0.495 | 1.049× |
| 7 | 14.37 | 382.7 | 6.69× | 0.534 | 1.093× |
| **8（最终冠军）** | **12.80** | **429.4** | **7.50×** | **0.609** | 1.030× |
| 9 | 未记账 | — | — | — | — |

**与 ASM 反向的差距：12.6× → 1.64×。** beat 在同口径下是 7.80 ms / 704.6 TF/s
（op-evolve 只计三次 kernel 发射，不含 autograd 管路，故与产品路径的 10.160 ms 不同口径）。
精度全程 **52.52–52.83 dB**（闸门 50），fast 形状 200 次逐位一致。

> ⚠ **round 7 与 round 8 之间有一条 session 边界，绝对值不可跨它引用。**
> 同一份 round-7 代码（md5 `3ad59da722a4`）在 round 7 的 session 测 **382.73**、
> 在 round 8 的 session（AC-cycle 之后）测 **404.06**，**+5.6% 而代码没动**；
> 同期 beat 在 prod 上只动了 −1.7%。**同 session 的真实增益是 1.063×，不是 429.4/382.7 = 1.12×。**

### 3.2 每轮做了什么

| round | 候选 | 改动 | prod 效果 |
|---|---|---|--:|
| 1 | `g01` | **causal tile 跳过**：按 causal 结构在运行时算循环次数，不再遍历全部 KV tile。**位级恒等** | 1.62× |
| 2 | `g07 + g06` | `k_dkdv` **32 深收缩**（一次 staging 32 个 query 行，WMMA 不再乘 padding 零）+ **`BLOCK_KV` 16→32**。同时把所有 global buffer descriptor 的 `num_records` 从平坦的 1 GiB 改成**真实字节范围**，修掉一个会走到未映射页的越界读 | 2.50× |
| 3 | `g13` | `k_dq` 的 **`BLOCK_Q` 32→64**：一个 wave 从 32 个 query 行变 64，把 K/V 全局载入和 K 的 LDS `ds_load_tr16_b128` 转置摊到两倍 query 上。VGPR 457→826，零 spill | 3.63× |
| 4 | `g09 + g16` | 两条都打在 `k_dkdv`（占 prod 的 **75.2%**）：**删掉 Q/dO staging 循环**直接写 LDS tile + **LDS padding**（`X_ROW_B` 256→272 B、`S_ROW_B` 64→80 B）消 bank conflict。两者强超加性 | **5.78×** |
| 5 | `g12 + g17` | **删掉 `k_dq` 的 K staging 循环** + 三次发射改走 `flyc.compile()`（发射路径开销） | 5.86× |
| 6 | `armMNS` | **causal mask 特化 + grid 轴交换**：KV 循环拆成 full 段（不带 mask）与 mask 段，循环次数在 kernel 内算 | 6.19×（+5.4%） |
| 7 | `g21` | `k_dkdv` **把下一迭代的 Q/dO 预取提到本迭代体顶部** | 6.69×（+8.3%） |
| **8** | `g23 + g24` | **删掉 `k_dkdv` 和 `k_dq` 里的 `fx.barrier()`**。机制：`block=(32,1,1)` 是单 wave，`s_barrier` 语义上是空操作、LLVM 本就会删；但后端在删它之前会插一条保守的全计数器等待 `s_wait_loadcnt_dscnt 0x0`，其 `loadcnt` 那一半在这里**没有任何依赖**，正是把 `g21` 的预取覆盖截断到 299 条指令的元凶 | **7.50×**（同 session +6.3%） |
| 9 | — | **无 GPU 静态多 wave 筛**（见 §3.5），未记账 | — |

`k_dkdv` 是延迟受限、`k_dq` 是计算受限 —— round 8 的机制边界被同会话五臂 sweep 证实：
删 barrier 在 `k_dkdv` 上值 **1.046×**，在 `k_dq` 上只值 **1.015×**，合并 **1.063×**。

### 3.3 round 5 的一个诚实注记

记账上的 1.15 倍 **是被 fast 形状抬起来的**：接受判据取三形状增益的**算术平均**，而 fast 只有
0.35 ms、被发射开销主导，却与 26 ms 的 prod 同权重。单臂 prod 实测（基线 331.07）：
`g12` 335.94、`g17` 333.45、合并 336.32 —— **round 5 在 prod 上的真实收益约 +1.6%**。

同类问题在 round 8 换了个面孔：**`score` 从 0.42321 跳到 0.60244，但那不是进步** ——
本次 beat 在小形状上测得异常低（fast 21.83 vs 50.70 即 −57%，proxy 461.57 vs 573.31 即 −19.5%，
prod 正常）。**不要把 score 这一列跨这个边界作图；排名看 `gain` 和 prod。**

### 3.4 挂卡与恢复（round 8）

round 8（`g04` / 4-wave 重构）运行中，卡于 14:26 挂死：
`MES(0,0) failed to respond` 从 `INVALIDATE_TLBS` 一路升级到 `RESET`/`RESUME`，
随后 `GPU reset begin!` + **MES 0–7 的 `REMOVE_QUEUE` 无界级联** —— 属不可恢复档。

**没有做任何恢复尝试**：不 `modprobe -r`（上次让整机掉 SSH 两小时）、不 `kill -9`
（每次都会多留一个不可杀的 D-state 进程，把可恢复推向不可恢复）、不做进一步探测
（挂卡时 `rocm-smi` / `pgrep` / `docker stop` 会自己挂住）。
**14:40:38 完成 AC-cycle**，重启后全检通过：`amdgpu` 已加载、KFD 空、零故障签名、
`git fsck` 通过、4096³ bf16 matmul 5.071 ms 有限结果。round 8 随后手工收口。

> **闸门本身才是挂卡源。** 补验 armAB 时 `validation.py` 在 proxy 阶段以
> `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` 中止，dispatch 签名
> `workgroup=[256,1,1] group_seg_size=6144` —— 那是 `op/eager/` 里的 hipBLASLt/Tensile GEMM，
> **不是我们任何一个 kernel**（k_dkdv 32 线程 / 22528 B、k_dq 32 / 8704、k_delta 256 / 0），
> 且与 round 3 中止时的签名完全相同。
> **每轮的固定挂卡风险有很大一部分来自闸门里的 fp32 参考实现，而不是被测的 kernel。**
>
> 本轮的绕法：armAB 对 incumbent **逐位相同**（prod 上 134,217,728 个元素零差异），
> 正确性由继承得到，**比重测 SQNR 更硬**（闸门有 2.5 dB 余量，逐位没有余量）；
> determinism 单独跑（它不需要 eager 参考），200 次通过。

### 3.5 round 9：只做了静态筛，未记账

round 9 做的是**不需要 GPU** 的多 wave 静态筛：把 `DKDV_THREADS` 从 32 抬到 64 / 128，
在三棵基线树上各筛一遍，共 9 个 build。结果：

| | 32 线程 | 64 线程 | 128 线程 |
|---|--:|--:|--:|
| `k_dkdv` VGPR（三棵树） | 988 / 956 / 632 | 984 / 952 / 636 | 984 / 952 / 634 |
| spill | 0 | 0 | 0 |
| LDS | 22,528 B | 22,528 B | 22,528 B |
| `s_barrier` | 0 | 8 | 8 |

**关键读数：抬线程数并不重新分配寄存器**（每 wave VGPR 变化在 ±4 以内），
而多 wave 需要的 `s_barrier` 如期回来了。
也就是说 **`g04` 不是一个旋钮，是一次真正的重构** —— 这与 §4 的两条阻塞一致。

节点在此之后交还：`.stop` 模块边界哨兵已写（文件，非信号）、supervisor 全程从未启动、
持卡令牌 `~/gfx1250.owner` 已删、定时任务与监控已全停。机器于当晚 AC-cycle，现由他人使用。

> **一处记录瑕疵，如实记下**：`ROUND3-CLOSEOUT.md` 被截断在 16,384 B（末尾是填充空白），
> `HANDOVER.md` 引用的"round 9 段"**实际不在文件里** —— 写入过程被挂卡打断。
> 本节的 round 9 内容是从 `r9probe/` 的 9 份 screen JSON 与 `round9.log` 重建的，不是从那一段读到的。

---

## 4. 已知的天花板与剩余方向

- **一个被证伪的模型，记下来免得再被引用**：hint `h8` 断言"我们钉在 HBM 屋顶上"
  （107 GB / 6.46 TB/s = 16.56 ms ≈ 当时实测 16.39 ms）。**强形式错了** ——
  round 6 的 prod 已是 15.52 ms、round 7 是 14.36 ms，**都低于那个下界**。
  107 GB 是 load 指令请求字节的正确计数，但实际到 HBM 的流量低于它（cache 吸收一部分），
  所以并未贴顶，还有延迟/访存并行度余量 —— round 6/7/8 吃的就是这块。
  **方向仍成立**：冗余流量是最大杠杆（**aiter 的字节数是我们的 1/5.6**），
  但不要再引用"101% of roof"。
- **`k_dkdv` 仍把 16 宽的收缩 padding 到 WMMA 的 32**，浪费一半 GEMM。
- **`r1.i4.g04`（多 wave workgroup，`BLOCK_KV` 32→128）现在有两条独立的阻塞**（hint `h13`）：
  1. **寄存器**：缺 462 个。`k_dkdv` 因 `g21` 的预取已到 **956–988 / 1024**。
     历史规律是 VGPR ≤ 951 的 21 个变体零 spill、= 1024 的 14 个全 spill，**中间没有过渡态**；
     而 gfx1250 上 spill 的 build **第一次发射就挂起，代价是一次断电**。
  2. **同步**：round 8 删掉的正是 4-wave 需要的那两个 barrier。若直接把 block 改成 128，
     kernel 会**静默竞态** —— 而竞态可能通过 SQNR 闸门（causal 跳过区真值本就接近零）、
     也可能在固定调度下 200 次逐位稳定，**两道现有闸门都不保证能抓到**。

  两者都是由各自正确的轮次造成的（`g21` +8.3%、`g23+g24` +6.3%）。合起来，
  它们把 kernel 爬进了一个**局部最优**：流量分析认为唯一必要的结构性改动，现在装不下了。
  **要走 `g04`，可能得先吐回 `g21` 和/或 `g23+g24`。这笔账该由下一轮显式定价。**
- **精度余量按设计就薄**：返回 bf16 梯度把 SQNR 封在 52–53 dB，对 50 dB 闸门只有约 2.5 dB
  余量，任何引入近似的一轮会很快撞门。
- **SQNR 在这个 op 上基本看不出东西**（连改了累加次序的臂都读同样的 52.61/52.65/52.83）
  —— 能干活的是**逐位比对**。这也是 round 8 在闸门挂掉时仍能交付的原因。

## 5. 对"能不能追平 ASM"这个问题的当前回答

立项时的判断是"最可能持平，买的不是速度而是源码可控 + GQA 根治 + varlen 能力"。
八轮之后：**7.50× 基线（57.2 → 429.4 TF/s），与 ASM 的差距从 12.6× 收到 1.64×**，
还没有出现证伪"能追平"的证据，但也还没追平。

剩下的路是清楚的但不便宜：唯一还能指望数量级改善的 `g04` 被寄存器和同步双重堵住，
而拆掉它要先吐回两轮各自正确的收益。**下一轮应当把"吐回 g21/g23+g24 换 g04"这笔账显式定价**，
并把交付标准写成"证伪 g04 也算合格"。

卡的账目：全程 **2 次挂卡需 AC-cycle**（09-21 16:01、09-22 14:26），
其中第二次发生在 round 8 运行中，已恢复并手工收口；
**另有一条重要发现 —— 挂卡风险有很大一部分来自闸门里的 hipBLASLt 参考实现，而非被测 kernel。**
