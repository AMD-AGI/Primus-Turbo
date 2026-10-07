## (c) 优化过程每一轮的进展

本节先用一张里程碑速览（c.2a）给出主线，再把 09-10 到 10-05 的每个冠军变化点按时间排成一张总表（c.2），最后给当前最优版本（c.3）和进展要点（c.4）。每一轮单独成行的明细见 `rounds.xlsx`（198 行，不合并；第 2 个 sheet 单独列出有提升的轮次）。路径若无前缀，均相对 `Primus-Turbo/output/`；`OE:` 表示 `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/`。

### c.1 口径说明

**机器与时钟阶段。** 绝对值只在同一阶段内比较；跨阶段只比较同进程比值。

| 代号 | 时间 | 机器 / 固件 / 驱动 | 负载时钟 | 说明 |
|---|---|---|---|---|
| A0-H | 09-04 及之前 | A0 未限频（这一阶段的固件/驱动版本没有单独记录） | 09-04 负载下 1699–1703 MHz | 本节没有这一阶段的 attention 测量。c1325c7e 提交信息里的 220.6 TF/s（b=2 s=8192）测量条件不明：提交信息只写 "seq 8192, B2 H32 D128, fwd+bwd"，没写是哪台机器、hkv、是否 causal 和时钟。09-13 同 kernel 在限频卡上 b2 实测 129.4，比值 1.70×；当天按估算的约 1.65× 时钟比判定它是在未限频的卡上测的，这个判定没有定论（见 a.2 #11）[0913__opt_plan__claude/phase1/RESULTS.md §0,§2；git show -s c1325c7e] |
| A0-T | 09-10..09-28 | A0 = heliosr-1b114-c07-1，单卡；VBIOS 113-M4500001-630A，SMU 125.7.1，amdgpu-dkms 7.1.1.31300009-2397345 | sclk DPM 只剩 500/1100 MHz（VR 限频）；op 级约 0.94–1.07 GHz（09-24 prod 窗口实测 998–1029 MHz） | 最早的限频记录是 09-10 22:38，09-11 由 op-evolve Phase-1 探针写成 HARDWARE-ISSUE.md；09-05..09-09 没有记录 [phase0/PLATFORM-ESCALATION.md:11-28,98-101]。09-29 之后，这一阶段的绝对数全部作废 |
| B0-14 | 09-14 | B0 = ctheliosp-1b112-a37-1，4×gfx1250（GPU0 降级，只用 GPU1–3）；驱动 / VBIOS / SMU 版本未记录 | 2133–2244 MHz | 与 A0 用同一镜像；e2e 与 A0 09-13 一样用 torchtitan v0.2.2（Primus 子模块 09-10 起固定在 73a0e6979）。`0914__campaign/RESULTS.md:65` 把 A0 09-13 记成 0.1.0，是笔误（见 a.1） |
| B0-28 | 09-27..09-28 | B0，每卡一个快照容器 fa-g0..g3；09-28 GPU1 wedge，交还时仍未恢复；驱动 / VBIOS / SMU 版本未记录 | op 级（分块尺子）约 2.15 GHz；nkfix 后训练中约 1.34–1.39 GHz（整板功耗墙约 2.13 kW） | 09-27/28 的 B0 会话不在本机，数字只来自文档和账本 |
| A0-R | 09-29..10-01 | A0 刷固件：VBIOS 700E，SMU 125.12.0，amdgpu-dkms 7.1.0-2411946 | sclk DPM 500/2356/2400；op 级 1.74–2.03 GHz；满载 WMMA 约 1.4–1.8 GHz（2500 W 功耗墙）；训练中约 1.5 GHz | 开机仍打印 VR 告警，但 DPM 不再截到 1100 MHz。09-30 开机后 dmesg 有 56 次 pcie_pl 可纠正错误 [0928__a0_repro/REPORT.md §6.3] |
| A0-D | 10-01 起 | A0，amdgpu-dkms 7.1.0-2412954 | 同 A0-R；GEMM 突发后（训练工作点）约 1.28 GHz | randn 基线与 A0-R 一致（bwd s6 = ASM 的 104.1%，fwd 时间比 1.079）。10-02 GPU RAS 在 07:58:50、08:09:46 各报一批 pcie_pl 可纠正错误（57+58 条）。主机 CPU L3 取指致命错误（BERT）导致的重启跨阶段出现：09-28 两次、10-01 20:41，与 GPU 无关 [1002__e2e/E2E-PLAN.md §1,§7；0928__a0_repro/REPORT.md §5.3] |

软件栈：容器 fa-repro（镜像 fa-tune:deps，即 amdprimus/amdprimus:gfx1250-20260910 加 torchtitan 依赖），torch 2.11.0+rocm7.14.0a20260625，triton 3.6.0，aiter-src @6963ae9，torchtitan v0.2.2（73a0e6979）。flydsl 有三个版本并存：
- 0.2.4：镜像自带，产品分支固定用它；
- 0.3.2：A0 bwd job 冠军和手工 s1–s6 用它；
- 0.3.4.1：09-25 起 fwd job、B0、e2e 用它。

**尺子**
1. **tune_attention.py**（09-13..09-16）：一层 fwd+bwd，CUDA event 取中位数，iters 20，每次冲刷 256 MiB L2；四张量 SQNR 门（out/dq/dk/dv ≥50 dB）。报 fwd / bwd / total ms，TF/s 按 fwd+bwd 合计计算。
2. **op-evolve benchmark.py**，逐次交错（A0 两个 job 09-17..09-27；B0 fwd r5–r9；B0 bwd r24–r26）：
   - 每个 arm 取 51 次中位数，palindromic 交错，ASM beat 与候选在同一进程。
   - 三个形状：fast (1,1024,1024,8,2,128)、proxy (1,4096,4096,32,8,128)、prod (4,8192,8192,32,8,128)；**只按 prod 排名**。
   - gain = 本轮代码与历史最佳代码在同一会话重测后的比值。min_gain：bwd 先是 0（B0 09-28、A0 09-30 改为 0.007），fwd 为 0.007。
   - fwd job 从 r9 起不再与 beat 同进程（h28）。r10 因三形状算术平均 1.0022 被否后，operator 加了 gain_weights（prod 1 / proxy 0.25 / fast 0），r11 起生效 [a8fa47bf]。A0 bwd job 从 r25 起也加 gain_weights，fast 改按 min 计。
3. **分块计时**（blocked，h40/h41/h66）：B0 fwd 从 r10 的 act 起（act 在 09-28 03:16 UTC 已按 blocked 9 + lead 4 计时 [0927__b0/fwd/rounds/010/act.md]，而 ruler/REPORT.md 记 harness 约 03:40 UTC 换装）；B0 bwd 从 r27 起（progress.md 的 r27 已报分块 beat 817.8，而 ruler/bwd/REPORT.md 记安装时间约 05:00 UTC）。两处的安装时间记录都比实际使用晚，以轮次为准。之后 A0 一直沿用。每个 arm 先跑 4 次不计时，再连续计时 9 次，轮间 palindromic。审计发现逐次交错会把 FlyDSL/ASM 比值压低：fwd 约 25%，bwd 约 3.4%。改为分块后 A/A 误差为 fwd ±0.19%、bwd ±0.05%。
4. **真实数据与训练工作点**（h47）：输入用 B0 step 43 导出的 6 层真实 q/k/v。“gb”指每次计时前先跑 10 个 bf16 32768×4096×14336 GEMM，把时钟压到训练工作点（A0 约 1.28 GHz）。FlyDSL 对时钟敏感，ASM 几乎不敏感，所以 randn/blk 和 real/gb 两种条件下的 FlyDSL/ASM 比值可以差 25% 以上。
5. **e2e**：Llama-3.1-8B BF16，32 层（09-15/16 部分实验为 8 层），MBS=GBS=4，seq 8192，每步 32,768 token，单卡，不开 activation checkpoint，不开 compile；tps = tokens/s。
   - 09-16 起，比较 attention 的前提是 nkfix（运行时改写 aten::mm 的 GEMM 布局）。
   - NKFIX_CHECK=1 每步约多 33 ms（10-02 的 e2e 开着它，10-05 的教程关掉了它）。
   - torchtitan 的 mfu 列按 A100 峰值计算，不可用。

**FLOP 口径**
- 当前口径（op-evolve `tools/op_flops.py`）：fwd 2.199292e12，bwd 5.498229e12（5-GEMM 名义计数，causal 取 (s+1)/(2s)），fwd+bwd 7.697e12；ms = FLOP / TF/s。表中标 derived 的值都是这样换算的。
- 09-13..09-16 文档的 TF/s 是 fwd+bwd 合计（7.697e12 / total ms）。0913/0914 文档另按 7-GEMM 实发口径报过 bwd（例如 B0 vendored 融合 bwd 748.4 vs 534.6 TF/s），本节不用这个口径。09-14 B0 Triton op-evolve job 只计 bwd。
- 09-23 的 STAGE2-FWD-SWEEP 用 fwd ≈2.233e12（旧 harness，TF/s 偏高约 1.5%）。09-17 DKDV-FIRST-TIMING 的 5.8 TF/s 是自定义的 useful 口径（单头），不可比。
- 我方 1-wave bwd（r0 到 s6）实际发 7 个 GEMM，是算法量的 1.4076×；aiter ASM 只发 1.0155×。本节一律按算法量计。所以 s6 的“占 ASM 103.8%”，是在多做约 39% 矩阵运算的情况下达到的。

**“占 ASM”**：ASM 时间 ÷ 本行时间（等于本行 TF/s ÷ ASM TF/s）。ASM 优先取同进程或同会话的测量；大于 100% 表示比 ASM 快。09-14 之前还没有 ASM，记“—”。op-evolve 的轮次用该轮同会话的 beat。

**为什么跨阶段不可比（实测例子）**
- **同代码、同尺子，只换平台**（A0-T 09-28 → A0-R 09-30）：
  - FlyDSL fwd r13ns 2.023→1.353 ms（o1 顺序；反序 o2 为 1.992→1.356），提速 1.47–1.50×（两种顺序，均值约 1.48×）；ASM fwd 1.564→1.257（1.24×；o2 为 1.23×）；
  - FlyDSL bwd r29 11.010→6.583（1.67×），ASM bwd 7.654→5.503（1.39×）。
  - 连比值都变了：bwd r29 从 ASM 的 69.5% 变成 83.6%。
  - e2e 层面没有干净的同口径倍数。教程 §7 的 ASM attention + nkfix 1,947→1,351 ms（1.44×）不纯是平台差异：1,947 出自 09-28 已作废的 a0_p3a 进程，同进程的 FlyDSL arm 从 step 2 起加载，fwd 树的 `_env.py` 随即把 `HIPBLASLT_TENSILE_LIBPATH` 改指宿主库（hipBLASLt 是否中途重读这个变量未证实）；1,351 是 10-02 的读数（两边都是 NKFIX_CHECK=1），两次之间驱动也从 7.1.1-2397345 换成 7.1.0-2412954（中间经过 -2410994、-2411826、-2411946）[0928__a0_repro/REPORT.md §2,§5.3,§6.3；1002__e2e/E2E-PLAN.md §1]。
- **同机同代码，只换尺子**：B0 fwd 从交错改为分块后，占 ASM 从约 85% 跳到约 96%，kernel 没有变。r16 起冠军 r13ns 在 randn 上按设计读数低约 4.8%。
- **同代码，只换输入和工作点**：A0 10-02，bwd s6/ASM 时间比 blk 0.972、gb 1.038；fwd r16/ASM blk 1.081、gb 1.349。
- **AC-cycle 前后会漂**：A0 09-22 同一份 r7 代码，404.06 vs 382.73 TF/s（+5.6%）。同会话噪声地板 0.24–0.66%，跨会话漂移约 1.5%。

**各阶段的 ASM 基准（bar）**

| 阶段 | ASM fwd（prod） | ASM bwd（prod） | 尺子 | 来源 |
|---|---|---|---|---|
| A0-T 09-15 | 1.549–1.572 ms | 10.160 ms（**已撤回**：autograd shim，每次调用都 hipModuleLoad 并新分配约 1 GiB scratch） | tune_attention.py | 0915__opt/RESULTS.md；baselines.md §5 |
| A0-T 09-17 | 1.5691 ms（n=20，967 MHz） | 7.6134 ms / 722.2 TF/s（op-evolve setup 同会话） | stage1_fwd_ab / benchmark.py | 0917__flydsl/STAGE1-FWD.md；OE:…a0-stale-0930/job_context/history/review/op_setup_v000.md |
| **A0-T 09-24（引用值）** | **1.5724 ms / 1398.67 TF/s**（n=51） | **7.6766 ms / ≈716 TF/s**（n=224 raw；7.6769 n=146） | benchmark.py | 0924__flydsl/bar-census/fwd_anchor.json；0924__flydsl/DAY-SUMMARY.md |
| A0-T job 每轮 beat | 1395–1416 TF/s（fwd r1–r12） | 704–720 TF/s（bwd r1–r23） | 同会话 | OE: 两个 job 的 state.yaml |
| A0-T 09-28 | 1.543–1.564 ms | 7.654–7.666 ms | 分块 | 0928__a0_repro/REPORT.md §1 |
| B0-14 | 1.410 ms（安静 1.396） | 当天未测 | tune_attention.py | 0914__campaign/RESULTS.md |
| B0-28 交错 | 1751–1797 TF/s（1.22–1.26 ms，偏快） | 852 TF/s（6.45 ms，r24） | 交错（有偏差） | 0927__b0/fwd,bwd/progress.md；bwd/rounds/024/act.yaml |
| B0-28 分块 | 1549–1562 TF/s（1.41–1.42 ms） | 816.8–820.7 TF/s（6.70–6.73 ms） | 分块 | 同上 |
| A0-R 09-30 | 1.253–1.257 ms（1750–1755 TF/s） | 5.476–5.517 ms（约 1000 TF/s） | 分块 randn | 0928__a0_repro/REPORT.md §6；0930__bwd/PROGRESS.md |
| A0-R 09-30 产品 bench | 1.187 ms | 5.360 ms | 产品 bench，同进程 | 0930__port/PR_BODY.md |
| A0-D 10-02 | randn blk 1.190–1.262；真实 blk 1.209；gb 1.137 | randn blk 5.480–5.513；真实 blk 5.475；gb 5.530 | 分块 / 真实数据 / gb | 1002__e2e/RESULT-realab.md；1002__e2e/E2E-PLAN.md §1 |
| A0-D 10-02 e2e（训练中，每步 32 层） | 34.7 ms/步 | 166.5 ms/步 | CUDA event | 1002__e2e/RESULT-e2e.md |

**轮次编号**
- bwd 的 round 24–27 有两套。一套是 B0 在 09-27/28 跑的 r24–r32，记作“B0 r24”等，账本只在 `0927__b0/bwd/`。另一套是 A0 在 09-30 和 10-02 于 live 目录续跑的 r24–r27，记作“A0 r24”等。
- A0 bwd job r0–r23 的账本在 `OE:gfx1250-flydsl-attn-bwd-20260917-115934.a0-stale-0930/` 和 live 目录中逐项相同；r23 的 09-30 收口只记在 live 目录。
- fwd 有两个 job：A0 的 `gfx1250-flydsl-attn-fwd-20260925-114644`（r0–r13），以及 B0 的 `gfx1250-flydsl-attn-fwd-b0-20260927`（克隆自 A0 的 r4，在 B0 上跑了 r5–r20）。
- state.yaml 里按形状记的 champions 指针与 best_round 不一定相同。A0 bwd job：best_round=24，champions={prod: 25, proxy: 25, fast: 24}，r25 虽被否，却是 prod/proxy 的 best-ever 记录。A0 fwd job：best_round=11，champions={prod: 12, proxy: 11, fast: 11}。续跑时，这些被否轮次的读数会被当作 best-ever 门槛；B0 fwd 的 r9、r19 就是这样被挡住的。

### c.2a 里程碑速览

下表从 c.2 里挑出主线上的 14 个节点。序号就是 c.2 的行号，数字都取自 c.2 对应行，口径与 c.2 相同；跨机器、跨固件阶段的绝对值不能直接比（见 c.1）。“—”表示该行没有这一项。

| 序号 | 日期 | 里程碑 | fwd ms / TF/s | bwd ms / TF/s | 占 ASM | e2e | 机器/口径 |
|---|---|---|---|---|---|---|---|
| 2 | 09-13 | 优化前基线：树内 Triton 出厂（R0；main 原样 58.424 ms 见 #1） | 10.673 / — | 48.969 / — | — | 32L 133.7 s/步，245 tps（#9；GEMM 走 rocBLAS，attention 很可能没进 e2e） | A0-T，tune_attention；合计 59.642 ms，129.0 TF/s（合计） |
| 7–8 | 09-13 | A0 Day-1 Triton 终点：两个 config 旋钮 + vendor aiter 融合 bwd + shape gate | 4.148 / — | 20.201 / — | — | — | A0-T，tune_attention；合计 24.349–24.435 ms，315.0–316.1 TF/s（合计），相对出厂 2.44× |
| 14 | 09-14 | B0 09-14 冠军：ASM fwd + Triton 融合 bwd（in-thread transpose） | 1.396 / —（ASM） | 8.840 / — | fwd = ASM | — | B0-14 GPU1，tune_attention；合计 10.235 ms，752.0 TF/s（合计），相对出厂 2.75× |
| 17、43、47 | 09-15..09-24 | aiter ASM bwd 接入（opt-in，默认关；ASM fwd 已在 09-14 #12 接入）；09-24 定 ASM bar | 1.5724 / 1398.67（ASM bar） | 7.6766 / ≈716（ASM bar；09-15 的 10.160 ms 是 shim 伪数，已撤回） | 100% | nkfix 后 8L ASM bwd 开/关 +12.72%（#21） | A0-T，op-evolve benchmark.py；fwd n=51，bwd n=224 普查 |
| 18–22 | 09-15..09-16 | GEMM 修复：hipBLASLt 路径 + nkfix（e2e 的头号杠杆） | — | — | — | A0 32L 244→2,027 tps（#18）；A0 8L nkfix v3 37,746 tps，6.16×（#20）；B0 32L nkfix 后 ASM 臂 20,718 tps（#64） | A0-T 8L / 32L；B0-28 32L |
| 24、26 | 09-17 | 转向 FlyDSL：fwd 首测 aiter FlyDSL m32x8；bwd 从零手写，A0 bwd job r0 | 2.3732 / 926.7 | 96.05 / 57.24 | fwd 66.1%；bwd 7.9% | — | A0-T；fwd 用 stage1_fwd_ab n=20（同会话 ASM 1.5691 ms），bwd 用 benchmark.py 逐次交错（同会话 beat 7.6134 ms） |
| 42、52 | 09-24 / 09-27 | 限频期冠军：bwd A0 r20，fwd A0 r11 | 1.979 / 1111.15（r11） | 10.75 / 511.42（r20） | fwd 78.9%（对同会话 beat 1407.8；对 09-24 bar 为 79.4%）；bwd 71.4%（对 09-24 bar 为 0.71–0.72×） | — | A0-T，op-evolve benchmark.py 逐次交错，同会话 beat |
| 58、62 | 09-28 | B0 冠军：fwd r16 = r13ns（关投机 softmax），bwd r29（u2n） | randn ≈1.51–1.52 / ≈1443–1455 | 8.375 / 656.5 | fwd randn ≈93%；bwd 80.4% | 见 #66 | B0-28，分块尺子，randn |
| 66 | 09-28 | B0 e2e 终版：fwd r16 + bwd r29 | — | — | — | fly/asm 1.0323 / 1.0328（fly 1,631/1,633 ms，ASM 1,579/1,583 ms） | B0-28 GPU0，32L，nkfix，NKFIX_CHECK=1 |
| 69 | 09-30 | A0 刷固件后同尺子复测（代码未变，新基线） | r13ns 1.353 / 1626（ASM 1.257 / 1750） | r29 6.583 / 835（ASM 5.503 / 999） | fwd 92.9%；bwd 83.6%（同代码 09-28 为 77.3% / 69.5%，见 #67，已作废） | — | A0-R，分块，randn |
| 77 | 09-30 | s6：当前 bwd 冠军（TDM 3 级 ring + 下一轮 B 操作数预读 + VALU 精简） | — | 5.300（反序 5.295）/ 1037–1038 | bwd 103.8% | 见 #81 | A0-R，分块，randn，同进程 ASM |
| 78 | 09-30 | 产品分支移植：fwd r16 + bwd r29 → flydsl 0.2.4 | 1.292 / 1702 | 6.618 / 831 | fwd 91.9%；bwd 81.0% | 见 #87 | A0-R，产品 bench，同进程 ASM 1.187 / 5.360 ms |
| 81 | 10-02 | A0 e2e：fwd r16 + bwd s6，单步与 ASM 持平 | 每步慢 9.2 ms | 每步快约 4 ms | 单步 0.9990 / 0.9985× | fly 1349.4/1349.9 ms，24,283/24,275 tps；ASM 1350.6 ms，24,262 tps | A0-D，32L，nkfix，NKFIX_CHECK=1，训练 sclk 约 1.50 GHz |
| 87 | 10-05 | 产品分支教程 e2e：fwd r16 + bwd r29（flydsl 0.2.4） | — | — | 折算到同口径约慢 3%（推算，未实测） | 1,361.9 ms/步，24,061 tps（NKFIX_CHECK=0） | A0-D；与 #81 的 NKFIX_CHECK=1 不同口径 |

### c.2 总表

单行“算子耗时 ms”里写成 a / b / c 的，依次是 fwd / bwd / total。TF/s 列的每个数后面都注明口径：“（合计）”= 7.697e12 ÷ fwd+bwd 总 ms，是 tune_attention 和 Primus bench 的报法，阶段 1–2 除 #13 外都是它；“（bwd）”“（fwd）”是单方向，分别按 5.498229e12、2.199292e12 计，#13 的 B0 Triton job 和阶段 3 起都是单方向。合计口径的数与单方向的数不能直接比。被否决的连续轮次合并成一行，每轮明细见 rounds.xlsx。

| 序号 | 日期 | 方向 | 轮次 | 算子耗时 ms | TF/s | 占 ASM | e2e 耗时/吞吐 | 机器 | 优化内容 | 备注（驱动/固件/口径） |
|---|---|---|---|---|---|---|---|---|---|---|
| **阶段 1 · Triton 基线与调优（A0-T，09-10..09-13；TF/s 为 fwd+bwd 合计 7.697e12）** | | | | | | | | | | |
| 1 | 09-10/11 | fwd+bwd | 优化前基线：main 的 turbo:TRITON 后端（c1325c7e，#481） | 58.424（同次 flex 31.508，sdpa FLASH 96.581） | 131.7（合计，bench 自报） | — | — | A0-T | 无（main 原样，需显式指定 Triton 后端） | 测时卡已在 VR 限频下（sclk 500/1100 MHz；最早的限频记录是 09-10 22:38，09-11 才写成报告）。默认派发会落到 CK，bwd 报 invalid argument。Primus bench_attention.py，log 为 Primus 仓库 `output/0910__fa_opt/attn_bench_gfx1250.log`（目录名 0910，文件时间 09-11 06:29 UTC）。09-13 同法复测 60.209 [0913__opt_plan__claude/phase1/RESULTS.md §1；phase0/PLATFORM-ESCALATION.md] |
| 2 | 09-13 | fwd+bwd | R0 出厂（tune_attention 基线） | 10.673 / 48.969 / 59.642 | 129.0（合计） | — | — | A0-T | 树内 Triton 的出厂单一 config（num_stages=1、num_warps=4，照搬 CDNA） | 同会话 flex 31.337 ms，turbo 慢 1.9×。镜像 amdprimus gfx1250-20260910 [0913__opt_plan__claude/phase1/BAKEOFF.md] |
| 3 | 09-13 | fwd | R1 | fwd 10.781→4.227（total ≈52.8） | ≈146（合计） | — | — | A0-T | fwd num_stages 1→2：fwd 快 2.42–2.55×，SQNR 逐位不变 | [phase1/RESULTS.md §3] |
| 4 | 09-13 | bwd | R2（f8c45dee） | 4.227 / 31.477 / 35.704（bake-off 复测 36.405） | 215.6；bake-off 复测 211.4（合计） | — | — | A0-T | bwd num_warps 4→2，bwd 快 1.54×；waves_per_eu=2/4 时掉入性能悬崖 | 只改 config，1000 次逐位确定。R3 的三处 dkdv 源码改动被否（mask-skip 反而慢 9%）[phase1/RESULTS.md §3,§7] |
| 5 | 09-13 | fwd+bwd | R4 Round-0 bake-off | aiter+2 knobs 31.207；flex 31.337；aiter 出厂 34.565；turbo 36.405 | 246.6 / 245.6 / 222.7 / 211.4（合计） | — | — | A0-T | 同会话、同门槛对比：树内实现输在 bwd（32.252 vs 27.888），于是转向 aiter 融合 bwd 结构 | 09-11 的 aiter 24.347 ms / 316.1 已撤回：上游 aiter 改过 bwd 配置，复测为 31.207 [phase1/BAKEOFF.md §1,§4] |
| 6 | 09-13 | bwd | R5–R7（纯 aiter 调参，作参考上界） | 25.283→21.672（bwd 18.416） | 304.4→355.1（合计） | — | — | A0-T | LSE-ABI 解耦（5eba2cf4）；非对称 tile 被证伪（64×64 已经最优）；aiter 融合 bwd 的 N1/M2 128→256（1.26×），BLK_SLICE_FACTOR 2→1（1.20×） | 首轮 sweep 因 import 绑定，override 没生效，作废。这个上界没有入树 [phase2/DECISIONS.md D1–D9] |
| 7 | 09-13 | bwd | R8：vendor 融合 bwd（1cb2e183） | 4.148 / 20.201 / 24.349 | 316.1（合计） | — | — | A0-T | vendor aiter 的 mha_onekernel_bwd（pin ffa945f9），配置硬编码，由 turbo 自己的 fwd 驱动 | [commit 1cb2e183；session 541e7bc3] |
| 8 | 09-13 | bwd | R9 shape gate（03a76f61→8dd2fd32），Day-1 终点 | 24.435 | 315.0（合计） | — | — | A0-T | 按并行工作量决定是否走融合 bwd，按 seqlen 选 tile；51 个 gfx1250 测试通过（main 上一个都不运行） | 相对出厂 2.44×，相对 flex 1.28× [0913__opt_plan__claude/PROGRESS.md] |
| 9 | 09-13 | e2e | 首次跑通 e2e | — | — | — | 32L 133.7 s/step，245 tps（flex 244） | A0-T | 补齐 torchtitan 依赖后，第一次跑完 step | GEMM 走 rocBLAS（27.4 TF/s）。converters: [] 导致 turbo attention 没进 e2e，两臂很可能都是 flex。当日误判为“缺 gfx1250 Tensile 库”，09-15 更正为路径错位 [phase1/RESULTS.md §10；0914__campaign/HANDOFF.md §4]。整条 attention 路径 35.7 ms × 32 = 1.14 s，不到单步的 1%（约 0.85%）[phase1/RESULTS.md:266]。当天约 13:19 UTC（13:30 发现）第 5 次 wedge（前 4 次在此前八天内，见 HARDWARE-ISSUE.md），需要重启主机 [0913__opt_plan__claude/phase2/INCIDENT-2026-09-13-wedge.md；PROGRESS.md:36-38] |
| **阶段 2 · ASM 接入（B0-14 09-14；A0-T 09-15..09-17；TF/s 除 #13 只计 bwd 外，都是 fwd+bwd 合计）** | | | | | | | | | | |
| 10 | 09-14 | fwd+bwd | B0 复现 Day-1 阶梯 | main@c1325c7e 30.258；出厂@1cb2e183 28.121；配置调优 18.916；vendored 融合 12.994；纯 aiter 上界 11.269 | 254.4 / 273.7 / 406.9 / 592.3 / 683.0（合计） | — | — | B0-14 | 无新代码，换到满频机重测 | 换机到 B0。七行加速 1.88–2.12×（报告按五行 like-for-like 取中位 1.93×），名次和 SQNR 都一致。这是 A0-VR→B0 的跨卡比，不是 A0 自身的限频代价：A0 自身刷固件前后，同代码同尺子 op 为 1.24–1.67×（o1 顺序），e2e 1.44×（不纯，见 c.1）。09-13 按时钟估的约 1.65× 与它不是同一个量，不再称作废 [0914__repro__c07/FINAL-TABLE.md；0914__repro__c07/RESULTS.md:126-130] |
| 11 | 09-14 | fwd+bwd | B0 step1（566e7798） | 2.579 / 9.867 / 12.412 | 620.5（合计） | fwd 54.7% | — | B0-14 | 融合 bwd 不设 waves_per_eu，fwd num_warps=2（+4.28%）；修掉 harness 把 0 当非法值的 bug | 0915 xlsx 按 ledger 中位数改记为 12.454 [0914__campaign/RESULTS.md] |
| 12 | 09-14 | fwd | B0 step2：T7 ASM fwd（0e5cb743） | 1.410 / 9.786 / 11.198 | 687.3（合计） | fwd = ASM | — | B0-14 | fwd 换成 aiter 预编译的 fmha_fwd_with_sink_asm（.co），fwd 快 1.82×，越过纯 aiter 上界 11.269 | 四卡同时负载；200 次逐位确定 [0914__campaign/RESULTS.md] |
| 13 | 09-14 | bwd | B0 Triton op-evolve job r0→r1（…bwd-20260914-015415） | bwd 10.017→8.683 | 548.9→633.2（bwd） | — | — | B0-14 GPU1 | r1：waves_per_eu 1→0，加 TRITON_HIP_USE_IN_THREAD_TRANSPOSE，同会话 1.153× | r2 gain 0.934（act.yaml 记 vs_champion 0.9345；另因 FastError 记 failed）、r4 gain 0.925 被否；r3 没有产出候选就放弃。09:44 UTC 用户中断，job 停在 r4，best_round=1。beat = flex bwd 11.68 ms [0914__campaign/op-evolve-artifacts/…/job_context/state.yaml、progress.md] |
| 14 | 09-14 | bwd | **B0 step3：ITT 入树（a009599b），09-14 冠军** | 1.396 / 8.840 / 10.235（四卡满载 10.299） | 752.0（合计） | fwd = ASM | — | B0-14 GPU1 | 融合 bwd 打开 in-thread transpose，bwd −9.7%；相对出厂 2.75×，相对 flex 1.50× | 安静条件复测，n=1 [0914__campaign/HANDOFF.md §15] |
| 15 | 09-14 | e2e | B0 e2e：GEMM 绕行 | — | — | — | flex eager 13.70 s，2,392 tps → compile + Triton GEMM 3.41 s，9,602 tps（4.01×） | B0-14 GPU3 | 让 inductor 强制走 Triton GEMM，绕开 hipBLASLt（113 vs 1190 TF/s） | torchtitan 0.2.2。当日的“turbo attention 13,204 tps / 1.459×”已撤回：镜像包遮蔽了 checkout，加上 converters: []，attention 实际是 flex [0914__campaign/RESULTS.md §10,§17] |
| 16 | 09-15 | fwd+bwd | A0 复现 B0 冠军 | 出厂 55.785 → 1.552 / 17.735 / 19.285 | 138.0→399.1（合计） | fwd = ASM | — | A0-T | 同一份代码回 A0：2.893×（B0 为 2.748×），SQNR 吻合到 9 位有效数字；ASM fwd 单独值 2.74×（4.253→1.549） | 回到 A0。新增 PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD 开关 [0915__repro__c07/RESULTS.md；0915__opt/RESULTS.md] |
| 17 | 09-15 | bwd | T2：aiter ASM bwd 接入（opt-in） | 1.572 / 10.160 / 11.726 | 656.4（合计，作废） | bwd = ASM | ASM bwd 开相对关：32L 开了反而慢 6.78%（1,858 vs 1,984 tps，n=1）；8L tps −0.28%（n=9，0.15× sem，不显著）。32L 的 6.78% 先归因于显存压力，当晚 n=9 复测后撤回这个归因：本机 e2e 跨运行噪声约 3.9% 且多模态，6.78% 落在模态间距内 [0915__opt/E2E-AB.md:101,182,232] | A0-T | 手写 ASM bwd launcher。发现 aiter gfx1250 ASM bwd 在 GQA 下按 q head 越界写 dk/dv，改为 dkdv_heads=q 加 host 规约；资格门 seqlen≥2048，默认关 | **10.160 ms / 541 TF/s 已撤回**（autograd shim 没传 hip= 和 scratch=），真实 ASM bwd bar 是 7.6766 ms（09-24）。当日 modprobe -r 导致整机失联，做了 AC-cycle [0915__opt/RESULTS.md；baselines.md §5] |
| 18 | 09-15 | e2e | e2e 打通，修正 hipBLASLt 路径 | — | — | — | 32L 134 s→17 s/step，244→2,027 tps（8.3×，3 步，n=1）；20 步稳态 1,984 tps | A0-T | TurboAttention.forward 接受 enable_gqa，第一次真正用上 turbo attention；HIPBLASLT_TENSILE_LIBPATH 指向镜像的 library/gfx1250（GEMM 27.64→68.74 TF/s） | GEMM 路径从 rocBLAS 改为 hipBLASLt [0915__opt/BLAS-FINDING.md；0915__opt/E2E-AB.md] |
| 19 | 09-15 | e2e | nkfix v1（dgrad） | — | — | — | 8L 5.35→2.82 s/step，6,128→11,604 tps（1.894×，n=9） | A0-T | 查明 hipBLASLt 的 NN（dgrad）和 wgrad 布局缺纯 bf16 调优库，落到 MT32x16x32 GEMV tile；用 TorchDispatchMode 把 B 改成 N-major | e2e 方差 3.91%→0.42%，原来的三模态噪声也来自这里 [0915__opt/GEMM-NN-FINDING.md] |
| 20 | 09-16 | e2e | nkfix v3（wgrad） | — | — | — | 8L 867 ms/step，37,746 tps（6.16×，n=5） | A0-T | wgrad 的 A 改连续、B 改 N-major（rule 2 只改 A，命中 0 次，被否） | 原报 38,043 / 6.21× 含一次 NaN 运行，0917 更正 [0915__opt/GEMM-WGRAD-FINDING.md；PROGRESS-REPORT-0917.html] |
| 21 | 09-16 | e2e | ASM bwd e2e 重做 A/B | — | — | — | 8L ON 42,534 / OFF 37,734 tps（+12.72%） | A0-T | GEMM 修好后，attention 占单步从 3.3% 升到 20.8%，ASM bwd 才有 e2e 收益 | 原报 +14.40% 已更正；仍默认关 [0915__opt/BOTTLENECK-SHIFT.md] |
| 22 | 09-16 | e2e | nkfix rule 3：wgrad 改走 FlyDSL GEMM | — | — | — | 8L 707 ms，46,374 tps（+9.26%，累计 7.57×）；32L ≈2.33 s，14,050 vs 12,340 tps（+13.9%，相对 1,984 为 7.08×，n=1） | A0-T | wgrad 改调同事的 FlyDSL gfx1250 WMMA GEMM（离线建表），省掉每步 137 ms 的转置拷贝 | 带 nkfix 的运行有 21% 出现 loss NaN（p=0.004），没解决，rule 3 默认关。8L 49,878 和 32L 11,608 / 5.85× 都是 NaN 运行，已作废 [0915__opt/NAN-FINDING.md；0915__opt/RESULT-32L.md]。AC 断电按 d.1.4 去重统计：09-15 共 5 次（07:18 前整机失联、~08:20 SIGBUS、09:00 8L worker、11:16 noconv、13:07 nk4），09-16 共 9 次（02:26、07:35、07:56、08:20、09:18、11:17、11:57、12:44 有用户确认，~13:15–13:30 为推断）。DAY-0916-SUMMARY 记的“09-16 五次”（14 次启动有 3 次死在启动阶段，21%）只是 08:20 之后那一段 [d.1.4；0915__opt/DAY-0916-SUMMARY.md §四；0915__opt/INCIDENT-2026-09-15-machine-death.md；session b596bddb、11052787] |
| 23 | 09-16 | fwd/bwd | Triton 侧收尾扫描 | bwd 两 kernel 9.10→7.47（−17.8%）；fwd 1.424→1.371（−3.73%） | — | bwd 36.7% | — | A0-T | proxy s4096 口径。Triton bwd 仍比 ASM 慢 2.73×，这条路线关闭 | 09-17 BACKEND-STRATEGY 建议不移植 FlyDSL，用户同日决定转向 FlyDSL [0915__opt/TWOKERNEL-SWEEP.md；FWD-SWEEP.md；BACKEND-STRATEGY.md] |
| **阶段 3 · FlyDSL 移植（A0-T，09-17；从这里起 TF/s 分方向）** | | | | | | | | | | |
| 24 | 09-17 | fwd | Stage1：aiter FlyDSL m32x8 fwd | FlyDSL 2.3732 / ASM 1.5691 | 926.7 / 1401.6（fwd） | 66.1% | — | A0-T（窗口 967 MHz） | flydsl 0.3.2 不加任何 shim 就能跑通。“前向模板有竞争力”被证伪：比 ASM 慢 1.51× | flydsl 0.3.2 侧装（与镜像的 0.2.4 不能同进程）；n=20，ABAB 交替 [0917__flydsl/STAGE1-FWD.md；530dd77e] |
| 25 | 09-17 | bwd | Stage3 bring-up 和 dkdv 首测 | dkdv_loop 单头 S=4096：1.475 | 5.8（bwd，自定义 useful 口径） | “1/93” 已撤回 | — | A0-T | 用 6 个探针排除未知量；odo/dkdv/dq 三个 kernel 首跑即通过（141–159 dB） | “慢 93×”已撤回：单头的 grid 填不满卡，同一 kernel 在 prod 形状是 57.2 TF/s [0917__flydsl/DKDV-FIRST-TIMING.md；HANDOFF-0917.md] |
| 26 | 09-17 | bwd | A0 bwd job r0（setup 基线） | 96.05 | 57.24（bwd） | 7.9% | — | A0-T | 从零手写 k_delta_bshd + k_dkdv + k_dq：每个 WG 一个 wave32，GQA 在寄存器里规约，无 atomic，结果确定 | job gfx1250-flydsl-attn-bwd-20260917-115934 于 11:59 发车，同会话 beat 7.6134 ms / 722.2 [OE:…a0-stale-0930/job_context/history/review/op_setup_v000.md]。r1 首次尝试已测到 armA（causal tile 跳过）prod 58.272 ms / 94.4 TF/s（同次 cur 57.4、beat 7.608 ms / 722.7，13.1%），armB（32 深收缩）64.8 TF/s；13:17 交机，r1 未判即中断，09-21 重跑 [OE:gfx1250-flydsl-attn-bwd-20260917-115934/rounds/001/_scratch/benchAB.json] |
| **阶段 4 · bwd op-evolve（A0-T，r1–r23，09-21..09-25；ms 按 5.498229e12/TF/s 换算；占 ASM 用同会话 beat）** | | | | | | | | | | |
| 27 | 09-21 | bwd | A0 r1 | 59.36 | 92.62（bwd） | 12.9% | — | A0-T | g01：causal tile 跳过（gain 1.342） | sclk 1055–1100 [OE:gfx1250-flydsl-attn-bwd-20260917-115934/job_context/state.yaml（下同）] |
| 28 | 09-21 | bwd | A0 r2 | 38.39 | 143.23（bwd） | 19.9% | — | A0-T | g07 32 深收缩，buffer descriptor 的 num_records 改用真实 extent（修越界读）；g06 BLOCK_KV 16→32（gain 1.921） | |
| 29 | 09-21/22 | bwd | A0 r3（手工收口） | 26.50 | 207.47（bwd） | 29.4% | — | A0-T | g10/g13 k_dq BLOCK_Q 16→32→64；g11 直接存 bf16（gain 1.297） | 09-21 两次 wedge（约 14:12、16:01:54），09-22 AC 后补测；本轮 beat 704.69 偏低 |
| 30 | 09-22 | bwd | A0 r4 | 16.61 | 330.98（bwd） | 46.3% | — | A0-T | g09 删掉 Q/dO 的 staging，直接写 LDS；g16 LDS padding（行距 256→272 B），消除 64 路 bank 冲突（gain 1.512） | |
| 31 | 09-22 | bwd | A0 r5 | 16.39 | 335.42（bwd） | 46.8% | — | A0-T | g12 删掉 k_dq 的 K staging；g17 改用 flyc.compile() 发射（gain 1.154） | gain 主要来自 fast（三形状算术平均），prod 实际约 +1.3%（act.yaml vs_champion 1.0132；335.42 对同会话 incumbent 331.07） |
| 32 | 09-22 | bwd | A0 r6 | 15.52 | 354.29（bwd） | 49.5% | — | A0-T | g19/g20 把循环拆成 mask-free 段和 masked 段；g03 交换 grid 轴（gain 1.049） | |
| 33 | 09-22 | bwd | A0 r7 | 14.37 | 382.73（bwd） | 53.4% | — | A0-T | g21：k_dkdv 跨迭代预取 Q/dO（gain 1.093；r19 删掉它会 −17.27%） | |
| 34 | 09-22 | bwd | A0 r8（手工收口） | 12.80 | 429.41（bwd） | 60.9% | — | A0-T | g23/g24 删掉单 wave 的 fx.barrier()，连带去掉保守的 s_wait_loadcnt_dscnt 0x0（gain 1.030） | 14:26 wedge 后 AC。同一份代码 AC 前后 382.73→404.06（+5.6%），绝对值不跨这条边界比较 |
| 35 | 09-22/23 | bwd | A0 r9（手工收口） | 11.43 | 480.96（bwd） | 66.8% | — | A0-T | g25 合并 carried-state；g26 k_dq 预取（k_dq 5.17→3.99 ms）（gain 1.119） | 16:01 在 d_final wedge。断电截断了 255 个 git 对象，由此定下“每轮结束都 push”的规则 |
| 36 | 09-23 | bwd | A0 r10–r11，均未接受 | 12.77 / 13.08 | 430.61 / 420.23（bwd） | 59.8% / 58.4% | — | A0-T | 指令级调度（g27/g28/g32）；s_set_vgpr_msb 的 bank 税（g36/g37）（gain 0.959 / 0.929） | r11 09:21 wedge（B 类） |
| 37 | 09-23 | bwd | A0 r12 | 11.02 | 499.15（bwd） | 69.9% | — | A0-T | g40 k_dq longest-first 派发（prod +4.91%）；g39 LDS 段分离（null）（gain 1.091） | max_rounds 12 用完，提到 40。当天查明 hipBLASLt 的真因：HIPBLASLT_TENSILE_LIBPATH 从没设过，sitecustomize 又把 PREFER_HIPBLASLT 钉成 0（_env.py 的 setdefault 空转了 12 轮）；改为显式赋值并指向宿主库（fp32 参考此前已改用 refcache）[0924__flydsl/REFCACHE-PREMISE-GONE.md] |
| 38 | 09-23 | bwd | A0 r13–r14 | 11.05 / 11.02 | 497.39 / 499.14（bwd） | 70.0% / 69.5% | — | A0-T | g42 k_dkdv 在 q 轴做确定性 split-K（fp32 partial，按固定顺序归约）；g44 nsp 规则；g45 k_redsp 归约 kernel（gain 1.370 / 1.111）。只提升 fast/proxy，prod 走 nsp=1，没动 | r14 首次尝试时 13:35 wedge：五个变体之一越界读，触发 fault storm（IH ring overflow）后 MES 不可恢复（C 类），这次尝试没有留下读数，AC 后重跑；之后 dmesg_restrict 改为 0 [0923__flydsl/hint.md h22] |
| 39 | 09-24 | bwd | A0 r15（被误晋升后回滚） | 11.57 | 475.14（bwd） | 66.5% | — | A0-T | g47 sched_group_barrier：prod −3.1%（vs_champion 0.9688），被 fast +6% 抵消 | 三形状算术平均 gain 1.0035 加 min_gain 0，让它被晋升为冠军；框架另因 FastError（g46/g47 未写进 facts/dead_ends）把本轮记为 failed，但晋升已经发生。用 20baa45e 手工回滚到 r14（best_round 和三个形状的 champions 都改回 14）。回滚后的 state.yaml 记 accepted:false / outcome failed，只看账本已看不出它曾被晋升 |
| 40 | 09-24 | bwd | A0 r16–r17（r17 是第一个 deep 轮） | 10.95 / 10.83 | 502.03 / 507.79（bwd） | 69.8% / 70.6% | — | A0-T | g51 把 LSE/delta 载入并入 g21 预取，加 sched_barrier(0)；g52 k_dq nsp_q split-K，只在 grid 欠填时启用（gain 1.005 / 1.159） | r17 的 PC sampling 在 prod 形状 fault（卡存活），此后永久禁用 |
| 41 | 09-24 | bwd | A0 r18–r19，均未接受 | 11.71 / 10.88 | 469.56 / 505.29（bwd） | 65.3% / 70.2% | — | A0-T | g56 改用 async LDS 载入 −6.84%；g59 WMMA reuse null；g60 删掉 g21 预取 −17.27% | r18 08:58 在 validation 单进程跑多形状时 wedge，之后改为一个形状一个进程 |
| 42 | 09-24 | bwd | **A0 r20：限频期 bwd 冠军** | 10.75 | 511.42（bwd；r22 同会话复测 516.87） | 71.4%；对 09-24 bar 为 0.71–0.72× | — | A0-T | g62 k_dkdv 的 Q/dO 预取深度 1→2（+1.65%）；g61 拆分累加链（null）；VGPR 904（gain 1.0016） | 靠 min_gain 0 才过；shape key 改名导致 champions 分叉（h35） |
| 43 | 09-24 | bwd | ASM bar 普查 | ASM 7.6766（n=224 raw；7.6769 n=146） | ≈716（bwd） | 100% | — | A0-T | 撤回 10.160 ms / 541 TF/s。剩余差距 = 7 vs 5 GEMM 的结构因子 1.386× × 调度因子 1.00–1.01× | 之前说的“0.92×”作废，实际是 0.70–0.72× [0924__flydsl/DAY-SUMMARY.md] |
| 44 | 09-24/25 | bwd | A0 r21–r23，均未接受 | 13.34 / 15.20 / 10.94 | 412.10 / 361.73 / 502.44（bwd） | 57.8% / 50.3% / 70.1% | — | A0-T | k_dq 预取 1→2：−19.33%；打散载入：−21.93% / −30.0%；g71 删回 g62 的第二级预取 −1.74%，反证 g62 是真实机制 | r23 于 09-30 在 A0 手工收口，判为 REJECTED |
| 45 | 09-25 | bwd | 手工 4-wave 探索（job 外） | G2 4-wave 16.31 vs 冠军 10.84 | 337.02 vs 507.34（bwd） | 47.2% | — | A0-T | G0 原子 scope；G1b 4-wave BLOCK_KV=128 骨架（UT 15/15）；S1/S2 均 null；barrier-free 探针 1.449×（就算 barrier 免费，上限也只到冠军的 0.950×） | 当日 dq 合同放宽为 run-to-run ≥70 dB，dk/dv 仍要 200 次逐位 [bwd-history.md §5,§6] |
| **阶段 5 · fwd op-evolve（A0-T，09-23..09-27；fwd FLOP 2.199292e12）** | | | | | | | | | | |
| 46 | 09-23 | fwd | Stage2 扫参（aiter FlyDSL fwd） | 出货点 2.356–2.400 | 930–948（fwd，旧口径 ≈2.233e12） | 65.7–66.1% | — | A0-T | n_block 64→128/256 分别慢 3.2×/9.9×（VGPR 溢出）；(BLOCK_M, n_block) 联扫证明出货点是局部最优；4-wave 在 aiter 代码里走不通；O_VARIANT v1 +1.90%（09-25 没复现） | [0923__flydsl/STAGE2-FWD-SWEEP.md] |
| 47 | 09-24 | fwd | fwd bar 锚定 | FlyDSL 2.4005 / ASM 1.5724 | 916.16 / 1398.67（fwd） | 65.5% | — | A0-T（sclk 1011→989） | 1.5266× 的差距全部是效率问题 | [0924__flydsl/bar-census/fwd_anchor.json] |
| 48 | 09-25 | fwd | A0 fwd job r0（20260925-114644） | 2.352 | 935.04（fwd） | 66.9% | — | A0-T | 基线是 vendored 的 aiter FlyDSL fwd（op0341）。flydsl 0.3.2→0.3.4.1 的 A/B 结果中性（fly/asm 0.664–0.666 vs 0.657–0.664），ISA 逐字节一致 | fwd 改用 flydsl 0.3.4.1（bwd 仍用 0.3.2）；r0 的数取自 r1 同会话的 incumbent [0925__flydsl/fwd341/ab_prod.log；OE: fwd state.yaml] |
| 49 | 09-25 | fwd | A0 r1（被否）→ r2（接受） | 2.166→2.160 | 1015.37→1017.97（fwd） | 72.7%→73.0% | — | A0-T | g01：因果 q-tile 按 longest-first 调度（prod +8.6–8.9%） | r1 被否只因两个框架 bug：50 dB 精度门（基线自己才 49.82–49.99）和 fast band 0.993。h20 重新提交后接受 [d64933d4] |
| 50 | 09-25/27 | fwd | A0 r3（被否）→ r4（接受） | 2.170→2.032 | 1013.60→1082.09（fwd） | 72.5%→77.4% | — | A0-T | r3：s_prefetch_inst / s_setprio 为 null，但探针证明 softmax 的 exp 占 prod 13.4%。r4：exp 参数用 v_pk_fma_f32，row-sum 用 v_pk_add_f32（两者叠加超过单独之和，+6.3%） | r4 在 09-25 交机时中断，09-27 重跑 [53ca89c9] |
| 51 | 09-27 | fwd | A0 r5–r10，均未接受 | 2.409 / 2.064 / 2.104 / 2.012 / 2.072 / 2.001 | 912.99 / 1065.79 / 1045.27 / 1093.10 / 1061.58 / 1099.24（fwd） | 65.4–77.8%（逐轮 65.4 / 76.1 / 74.7 / 77.8 / 75.2 / 77.6，各用本轮同会话 beat） | — | A0-T | r5（deep）QK/softmax 软件流水 −15.2%；r6 per-lane row-sum −1.9%，发现 post-ASM 的 I$ 惩罚；r7 barrier 减半原型 −3.5%（删 barrier 的上限是 +4.5–5.2%）；r8 nodelay +0.6%，被 post-beat 伪影否掉；r9 合并提交，输；r10 nodelay 单独 +0.83%，被三形状平均稀释，又被硬编码的 GATE_DB=50 挡住 | 尺子变了：r9 起不与 beat 同进程（h28）；r10 被否后加 gain_weights，r11 起生效 [a8fa47bf]；r9 之后的 fast/proxy 绝对值不能与 r1–r8 比 |
| 52 | 09-27 | fwd | **A0 r11：限频期 fwd 冠军** | 1.979 | 1111.15（fwd） | 78.9%（对同会话 beat 1407.8；对 09-24 bar 1398.67 为 79.4%） | — | A0-T | nodelay（amdgpu-enable-delay-alu=False），加 g30 按形状切到 R=1（BLOCK_M=128），fast 55.2→65.4，超过 ASM 的 64.5（gain 1.0074） | 相对 r0 +18.8% [e5d57d49] |
| 53 | 09-27 | fwd | A0 r12（被否）、r13（中断），以及 CPU 原型 | 1.979 | 1111.36（fwd） | 79.0% | — | A0-T | r12：R=1 路径做 split-KV（fast 0.987），关闭。另有 h21–h25 的 compile-only 原型，以及 ASM 结构 P2 设计（800 VGPR、0 spill，预测 +8–20%，没上卡） | 15:21 交机，fwd 迁到 B0 [0927__flydsl/asm-structure/DESIGN.md]。r12 被否，但 state.yaml 的 champions 记 prod=12（1111.36，与 r11 持平），best_round=11 [OE: fwd state.yaml] |
| **阶段 6 · B0 四卡攻关（B0-28，09-27..09-28）** | | | | | | | | | | |
| 54 | 09-27 | fwd | B0 fwd r5（失败）→ r6（接受）（job fwd-b0-20260927，克隆自 A0 r4） | 1.443 | 1524.1（fwd） | 84.8%（交错，偏低） | — | B0-28 GPU0 | 投机 softmax（SPEC_STALE_MAX）：跳过 row-max 和 rescale，用 ballot 回退到精确 softmax；+5.7%（分块复核为 +4.6%） | 四卡隔离成 fa-g0..g3；r5 的 reviewer 报 401（B0 上没有 codex 账号）；r4 在 B0 上为 1441.7 / 80.2%。这个投机后来在真实数据上是负收益，r16 关掉了 [0927__b0/fwd/progress.md] |
| 55 | 09-27/28 | fwd | B0 fwd r7–r9（被否）→ r10 refactor h39 | 1.453 / 1.457 / 1.466 | 1513.7 / 1509.8 / 1500.6（fwd） | 84.3–85.7%（交错） | — | B0-28 GPU0→GPU2 | 把 r9 的 BG 提为冠军：m16x8 小 grid gate（fast +27.5%）加 g18（masked tile 不做投机） | r9 只是被未发布 arm 的 best-ever 记录挡住。09-27 15:43 容器被他人删除，停机；09-28 GPU1 wedge，fwd 移到 GPU2 |
| 56 | 09-28 | fwd+bwd | 尺子审计：交错改为分块（h40/h41/h66） | — | — | 交错让 FlyDSL/ASM 比偏低：fwd 约 25%，bwd 约 3.4% | — | B0-28 GPU0 | 撤回两个假赢：lab2-L12 +6.4%→−0.4%（分块下 r6/L12 0.9964）；lab3-L21 +6.3%→约 +0.3%（lab3 VERIFY；分块下 bnegg 对 r6 +0.6%，在噪声边缘）。fwd r6 的 +5.7% 下修为 +4.6%，仍是真实收益（OP-EVOLVE-SUGGESTIONS 把这次高估也算作第三个“假赢”） | A/A：fwd ±0.19%，bwd ±0.05%；新旧尺子的绝对 ms 不可比。审计报告称“r6/ASM 稳态 1.035（r6 更快）”，与 job 分块尺子的 96.5%、A0 09-30 的 r6/ASM 1.028 矛盾，未解决 [0927__b0/ruler/REPORT.md] |
| 57 | 09-28 | fwd | B0 fwd r10 deep（被否）→ r11、r13 接受 | 1.474 / 1.466 / 1.463 | 1491.8 / 1500.7 / 1502.8（fwd） | 95.8 / 96.4 / 96.5%（分块 beat ≈1557） | — | B0-28 GPU2 | r10：L12 occ2 4-wave −0.7%，判死；r11：欠填 grid 改用 2-wave m32x2（fast +8.4%）；r13：小 grid 改 4-wave 并切分 d（fast +5.6%，fast 达到 ASM 的 105.5%） | prod 基本没动，增益都在 fast。占 ASM 从 85% 到 96% 是尺子变化。r12/r14/r15 被否（host 路径 null；TDM multicast 使 proxy −5.4%；分 lane row-sum null） |
| 58 | 09-28 | fwd | **B0 fwd r16 refactor h44：r13ns，当前 fwd 冠军** | randn ≈1.51–1.52 | randn ≈1443–1455（fwd） | randn ≈93%；真实数据 + GEMM 突发下 r13ns/ASM 时间比 1.25–1.29 | e2e fwd 50–61 → 48–49 ms/步 | B0-28 | 关掉投机 softmax（SPEC_STALE_MAX=False）：randn 慢 4.8%（预期内），真实 dump 快 8–15%，真实数据 + GEMM 突发快 12–23% | 线索来自 profile 和 fwd-nospec lab。r16 起 randn 读数比 r13 低约 4%，前后不能直接比 [0927__b0/fwd-nospec/REPORT.md；df9aa3e8] |
| 59 | 09-28 | fwd | B0 fwd r16opt–r19 均未接受；r20 中断 | 1.518 / 1.510 / 1.498 / 1.500 | 1448.9 / 1456.9 / 1467.8 / 1466.6（fwd） | 93.1–94.8%（randn） | — | B0-28 GPU2 | 逐行独立的 deferred rescale；用 ones-WMMA 算 row-sum；row-sum 重构（只在 randn 上有收益）；r19 = r18 的 g64 加 RESCALE_THRESHOLD 8→24（对冠军：h47 尺子即真实 dump 加每次计时前 GEMM 突发，+1.5%，18/18 读数高于 A/A；randn +1.7%），只被 r18 的 best-ever 记录挡住，refactor h48 没执行 [0927__b0/fwd/rounds/019/opt.md] | r18 起同时报真实 dump 尺子（h47）。r20 的 act 在卡上算 fp32 hipBLASLt 参考 GEMM，导致 GPU2 page fault，硬停，补 h50 |
| 60 | 09-27/28 | bwd | B0 r24–r27 均未接受（B0 编号） | 8.607 / 8.931 / 8.591 / 8.573 | 638.8 / 615.6 / 640.0 / 641.3（bwd） | 75.0 / 74.4 / 76.5 / 78.4% | — | B0-28 GPU1→GPU3 | unroll −4.5%；4-wave 的 X1/X2 判死；反向扫描做 L2 对齐 −1.9%；h33 clamp null。r27 的 geomean 被 fast 的 1.66× 拉高，误判 target_met，job 自行结束 | 起点是 A0 r20，在 B0 上约为 ASM 的 0.74–0.76×。r24–r26 用交错尺子。h69：validation 改为 proxy 和 prod 各自 ≥ beat |
| 61 | 09-28 | bwd | B0 r28 refactor h68：r19h | 8.583 | 640.6（bwd） | ≈78% | — | B0-28 GPU3 | lab-bwd-r19：r19 加 h33 的三处 clamp（越界预取 1,580→0，VGPR 904→724）；r19 在 prod 上比 r20 快 1.7% | r28 opt 为 null（0.9967）；min_gain 0→0.007 |
| 62 | 09-28 | bwd | **B0 r29：u2n，B0 bwd 冠军** | 8.375 | 656.5（bwd） | 80.4%（分块 beat 817.0） | 见 #66 | B0-28 GPU3 | lab-kdq：k_dq 的 full kv 循环展开 ×2（u2n），去掉回边上 65 条 v_mov_b64 的寄存器轮转（k_dq 3.185→2.990 ms）；加 g86，dK/dV 的 epilogue 经 LDS ring | prod +2.7%，proxy +2.8%；训练工作点下 r29/r19h −3.2% [0927__b0/bwd/progress.md；9aa21d96] |
| 63 | 09-28 | bwd | B0 r30–r32 均未接受；FUSED5 判死 | 8.377 / 8.419 / 8.641 | 656.4 / 653.1 / 636.3（bwd） | 80.0 / 79.8 / 77.8% | — | B0-28 GPU3 | dQ epilogue 经 LDS、常数外提、寄存器承载 dV/dK，全是 null 或变慢；lab3 FUSED5（把 dQ 的 fp32 原子累加融进 k_dkdv）21.9 ms，只有冠军的 0.39× | 12:45 UTC 两个 loop 停止，冠军快照 5188c9da 交接给 A0 |
| 64 | 09-28 | e2e | B0 e2e：修复前 → nkfix_b0 | — | — | — | 修复前：ASM 17,049 ms，低档中位数 1,922 tps（高档 2,016–2,022），fly(r6+r20) 1,914（1.0045）。nkfix 后：ASM 1,582/1,576 ms，20,718/20,792 tps；fly 19,762/19,809（1.0475/1.0500）；turbo Triton 2,250 ms / 14,563 | B0-28 GPU0 | 移植并重写 nkfix，加 transpose_triton：997 个 GEMM 全走 MT256x256x128。GEMM 每步 15.6–16.5 s（占 96%）→0.83–0.86 s（约 53%），约 18–20×。ASM 臂 20,718 tps 是修复前高档 2,016–2,022 的 10.3×（对低档中位数 1,922 为 10.8×） | 8 次运行 0 NaN [0927__b0/gemm/REPORT.md §0；0927__b0/e2e/RESULT.md §0]。nkfix 后训练 sclk 均值约 1,820→1,390 MHz（整板约 2.13 kW 功耗墙），attention kernel 慢 10–16%（k_dkdv +18%，ASM bwd 主 kernel +17%，k_dq +2%）[0927__b0/profile/REPORT.md:11,21,§3] |
| 65 | 09-28 | fwd+bwd | B0 训练中 attention 剖析（e0e42ad7） | 训练中每层（nkfix 后）：fly fwd 1.57–1.84 / ASM fwd 1.21–1.23；k_dkdv 5.95–5.99 + k_dq 3.30–3.35 / ASM bwd 主 kernel 7.00–7.09 | — | — | fly（fwd r6 + bwd r20）每步比 ASM 慢 87 ms：fwd +19.9、bwd +71.0、ASM 独有的 GQA 求和 −3.9 ms | B0-28 GPU0（训练中 attention 跑在 1250–1690 MHz，整板约 2.13 kW） | 解释 op 级结论为何在训练里反转，两个因子相乘。①时钟：randn 下 FlyDSL fwd 在 2350 MHz 为 ASM 的 0.84×，GEMM 刚跑完（约 1300 MHz）时为 1.20×，ASM fwd 基本不随时钟变。②数据：step 43 导出的真实 q/k/v，score 标准差 21–53，r6 的投机 softmax 在 13–25% 的 tile 步回退重算（randn 上为 0%）。真实输入 + GEMM 后时钟下，op 级 fly/asm 为 1.45–1.67。bwd 差距主要在 k_dq（3.33 ms/层，约 107 ms/步，ASM 融合 kernel 没有这份工作） | 由此出了 fwd 的 nospec lab → h44（r16 = r13ns）；bwd 随后的 lab-kdq 以 k_dq 为对象 → u2n（r29）。profile 对 k_dq 的定性（延迟/访存 bound、对时钟不敏感）后被 lab-kdq 修正：第一轮判为发射 bound，第二轮在 1272–1352 MHz 下测得 k_dq 约 1.36× 时钟敏感；这与 profile 里训练中 k_dq 只慢 2% 的读数仍未对上 [0927__b0/profile/REPORT.md §0,§3；0927__b0/lab-kdq/REPORT.md] |
| 66 | 09-28 | e2e | **B0 e2e 终版：fwd r16 + bwd r29** | — | — | — | fly 1,631/1,633 ms、20,091/20,070 tps，对 ASM 1,579/1,583 ms、20,747/20,697 tps，为 1.0323/1.0328。各版同进程 fly/asm：r6+r20 1.0475/1.0500 → r13（投机）+r20 1.0454/1.0447 → r13ns+r20 1.0403/1.0408 → 本版 [0927__b0/fwd-nospec/REPORT.md §3] | B0-28 GPU0 | 这一步的收益全部来自 bwd（每步 −15 ms）；剩余差距 bwd 约 52 ms、fwd 约 10 ms/步 | NKFIX_CHECK=1；sclk 1342/1371 MHz。PR 正文引用的 e2e 1.032 就是这个数 [0927__b0/e2e/RESULT-final.md；64d59269] |
| **阶段 7 · A0 换固件前后复测（09-28..09-30）** | | | | | | | | | | |
| 67 | 09-28 | fwd+bwd | A0 复现 B0 冠军（作废） | fwd r13ns 2.023 / ASM 1.564；bwd r29 11.010 / ASM 7.654 | 1087（fwd）/ 499（bwd） | fwd 77.3%；bwd 69.5% | 32L asm/old/new 1947 / 2070 / 2071 ms；24L（第二次开机）ASM 1483.4 / r16+r20 1569.9 ms（均作废） | A0-T（最后一次） | 同样的分块尺子；B0 上的 bwd 提升在 A0 上方向相反（r19h −2.2%，r29 −3.9%） | 两次开机共 5 个训练进程：开机 1 的 p3a 正常、p3b 从 step 4 起 NaN、p4a-32L 11:55 挂卡；AC 后开机 2 的 p4a-24L 正常，p4b-24L 在 12:32:50 第 1 步挂卡。两次挂卡各要一次 AC。最可疑的原因（未证实）：FlyDSL fwd 树的 _env.py 在 import 时把 HIPBLASLT_TENSILE_LIBPATH 改指宿主库 ~/.local/hipblaslt-gfx1250；第 1 步就用 FlyDSL 的两个进程一个 NaN、一个挂卡。10-02 的 h85“只用镜像 hipBLASLt”沿用了这条嫌疑。09-30 复测后整组作废 [0928__a0_repro/REPORT.md §1,§2,§5.3,§6.4；1002__oe/incident/WEDGE-1002.md:13] |
| 68 | 09-29 | — | A0 刷固件、换驱动（管理员操作） | — | — | — | — | A0 | — | VBIOS 630A→700E；SMU 125.7.1→125.12.0；amdgpu-dkms 7.1.1-2397345→7.1.0-2411946；sclk DPM 500/1100→500/2356/2400；fclk 1100→1250/1900。过程：09-28 19:39 UTC 另一用户卸掉 7.1.1-2397345，19:40 装上 7.1.0-2410994（21:40–21:42 又重装同版本）；重启后 PSP `ID_LOAD_TOC failed (0x11)`，probe 报 −62，卡无法初始化，09-29 06:40 AC 断电后依旧。09-29 16:09 换成 -2411826，21:28 换成 -2411946（21:41、22:13 又重装）；VBIOS 在 09-29 06:25 之后刷成 700E；19:08–22:19 主机重启约 10 次 [0928__a0_repro/REPORT.md §6.3；session 2dafe0d2] |
| 69 | 09-30 | fwd+bwd | **A0-R 同尺子复测（新基线）** | fwd r13ns 1.353 / ASM 1.257；bwd r29 6.583 / ASM 5.503 | 1626 / 1750（fwd）；835 / 999（bwd） | fwd 92.9%（反序 o2 92.4%）；bwd 83.6%（o2 83.3%，5.494/6.596） | — | A0-R（1.74–2.03 GHz） | 代码没变：FlyDSL fwd 快 1.47–1.50×（两种顺序，均值约 1.48×），ASM fwd 1.23–1.24×，FlyDSL bwd 1.67×，ASM bwd 1.39×；r20、r19h、r29 三者相差 <0.3% | 9b8f7397（提交标题写 bwd 83.5%，即两种顺序的均值）。同日跑了 roofline 微基准（W=8 cyc/WMMA；LDS 256 B/clk/段；WMMA↔DS 切换约 29 cyc；满载约 1.7 GHz）和 M8 fwd 消融（骨架开销 28%，softmax 19%）；新固件下 ATT 可用 [0930__roofline/REPORT.md；0930__bwd/probe/P1-RESULTS.md] |
| **阶段 8 · bwd 手工攻关（A0-R，09-30；同进程跑 r29 + r29_aa + ASM，分块尺子，randn，判定阈值 0.5%）** | | | | | | | | | | |
| 70 | 09-30 | bwd | base r29 → a01 c1 | 6.579→6.504 | 835.7→845（bwd） | 83.6%→84.2% | — | A0-R | c1：DQ_U2=False（新固件上 k_dq 展开 ×2 反而是负收益，关掉后 −0.85%） | A/A 0.01% [0930__bwd/PROGRESS.md（下同）] |
| 71 | 09-30 | bwd | a02–a08（被否 / 仅信息 / 待叠加） | ku2 7.099；tdm 6.628；tdm2 6.503 | — | 77.3–86.2% | — | A0-R | 消融 A2（去掉 Q/dO 全局加载，上限 −18.5%）指出主杠杆；trorder −0.89%、tailpf −3.33%、streams −0.89%、soffset −0.65% 留待叠加；只上 TDM ring、不把操作数读回寄存器反而 +1.7%；abl_l2 证明 L2 带宽不是瓶颈 | 每一步都由 ATT 指导 |
| 72 | 09-30 | bwd | a09 s1 → a10 s2 | 6.329→6.304 | 869→872（bwd） | 86.9%→87.4% | — | A0-R | s1 = tailpf + trorder + streams（k_dqg 放到 side stream，与 k_dkdv 并发）；s2 = s1 + soffset | 结果逐位等价 [7f42a5da] |
| 73 | 09-30 | bwd | a11 dkdv_tdm3 → a12 s3 | 5.856→5.769 | 939→953（bwd） | 94.0%→95.4% | — | A0-R | k_dkdv 用 TDM 3 级 LDS ring 装 Q/dO，并把下一轮的 B 操作数提前读回寄存器（每轮 WMMA↔DS 切换 7→2）；s3 = tdm3 + dqg_ts + side stream | 逐位 [12bbd6c9；4080e84a] |
| 74 | 09-30 | bwd | a14 s3_trim → a16 s4 | 5.508→5.532（复测 5.494） | 998→994（bwd） | 99.8%→99.7% | — | A0-R | 精简 VALU、v_nop 和地址计算（DS 段地址 VALU 20→2，v_nop 23→9）；计数器去掉除法（热循环 SALU 105→60） | [552dfb4d] |
| 75 | 09-30 | bwd | a13/a15/a17/a18/a20：w4f 4-wave 融合 dQ（均未接受） | 15.857→9.529→6.630→6.509（r4b）；relax_noatom 4.241（noatom 4.341；两者都不写 dQ 原子，输出错误，只作上限） | 347→845（bwd） | 34.6%→84.5% | — | A0-R | 5 个 GEMM，split barrier，dQ 用 fp32 SCOPE_DEV 原子累加；结果正确。dQ 原子写吃掉约 5 ms；按 ASM 的 fragment 顺序写满 128 B line 后，9.27→6.63 ms | 受整芯片原子吞吐限制，暂停 |
| 76 | 09-30 | bwd | a19 s5（dqg_tdm） | 5.338 | 1030（bwd） | 103.1%（proxy 112.8%） | — | A0-R | k_dqg 也改走 TDM 3 级 ring 并在寄存器里预取（VGPR 1020→878）；1-wave 路径第一次快过 ASM | 200 次逐位一致 [a94ccebe] |
| 77 | 09-30 | bwd | **a21 s6：当前 bwd 冠军** | 5.300（反序复核 5.295） | 1037–1038（bwd） | 103.8% | 见 #81 | A0-R | s5 再精简 k_dqg 的 VALU（热循环 VALU 290→194） | 相对 r29 −19.5%。dq 与 s5 差 81 dB（packed 运算顺序不同），SQNR 仍为 52.5 dB；只测过 randn [7cd15aef] |
| **阶段 9 · A0 e2e 与产品移植（09-30..10-02）** | | | | | | | | | | |
| 78 | 09-30 | fwd+bwd | 产品分支移植（dev/lhz/llama31_attn_opt，基于 origin/main 8cda13c7） | fwd 1.292 / ASM 1.187；bwd 6.618 / ASM 5.360 | 1702（fwd）/ 831（bwd） | fwd 91.9%（1.088×）；bwd 81.0%（1.235×） | — | A0-R | 把 fwd r16 + bwd r29 移植到 main 固定的 flydsl 0.2.4：c36cc124（gfx950 gate 的 >= 改为 ==）、5658a443（kernel 包 + 兼容层）、e05b67fd（FLYDSL 后端路由）、5a8446b5（测试）；API 清理 1f74e662/58d5d798（同日）：11 个 kernel 在两个 flydsl 版本下指令流都不变，60 个 UT 通过，prod 复测 fwd 1.283 / bwd 6.620 ms（清理前 1.292 / 6.618；这两个数只见于会话记录，没有对应的 ASM 读数）[session fdc2534d]。对 Triton 的 36 个形状几何平均：causal fwd 4.13×，bwd 3.37× | 0.2.4 把 async LDS→global 的 O store 地址编码错了，一上卡就 fault，改用 buffer_store，代价 fwd 约 1.3%、bwd 约 1.8%。已 push，PR 未开 [0930__port/PR_BODY.md] |
| 79 | 10-02 | fwd+bwd | 新驱动下 randn 复测；s6 移植到 0.3.4.1 | bwd s6 5.295 / ASM 5.513；fwd r13ns 1.362 / ASM 1.262 | 1038（bwd）/ 1615（fwd） | bwd 104.1%；fwd 92.7% | — | A0-D | s6 在 0.3.2 和 0.3.4.1 下的 ISA、LLVM IR、gpu.binary 逐字节相同（fd6dce02） | amdgpu-dkms 7.1.0-2412954（10-01 换）；上一次开机在 10-01 20:41 以 CPU L3 取指致命错误（BERT）重启结束；当天 GPU RAS 在 07:58:50、08:09:46 各报一批 pcie_pl 可纠正错误（57+58 条）[1002__e2e/E2E-PLAN.md §1] |
| 80 | 10-02 | fwd+bwd | 真实数据 op A/B（realab） | blk：fwd r16 1.308 / ASM 1.209，bwd s6 5.324 / ASM 5.475；gb：fwd 1.535 / 1.137，bwd 5.743 / 5.530 | — | blk：fwd 92.5%，bwd 102.9%；gb：fwd 74.1%，bwd 96.3% | — | A0-D（gb ≈1.28 GHz） | 用 B0 step 43 导出的 6 层真实 q/k/v。在训练工作点上 ASM 几乎不变，FlyDSL 两个方向都变慢 | gb 下 s6/r29 为 0.774 [1002__e2e/RESULT-realab.md；9108fd7c] |
| 81 | 10-02 | e2e | **A0 e2e 三臂对比：fwd r16 + bwd s6，当前 e2e 最佳** | — | — | 每步 bwd 快约 4 ms（相邻配对 4.50 / 4.56，即正文的 4.5；两臂中位数相减 3.8–4.4，RESULT-e2e.md 表四舍五入后为 3.9–4.3），fwd 慢 9.2 ms | fly 1349.4/1349.9 ms，24,283/24,275 tps（ASM 的 0.9990/0.9985×）；ASM 1350.6 ms，24,262 tps；flyr29 1389.6/1388.5（1.027） | A0-D | s6 相对 r29 每步省下 attention bwd 约 53 ms，与 op 级预测一致 | 32L，nkfix，NKFIX_CHECK=1，2 个进程 × 92 步；训练 sclk 中位数约 1.50 GHz；0 次非有限值。attention 合计 fly 每步多约 4.3–5.2 ms（相邻配对 +4.89 / +4.27；两臂中位数相减 +4.6 / +5.2），单步却快 0.7–2.0 ms（相邻配对 1.4 / 2.0；两臂中位数 1.2 / 0.7）：两进程的 Δ单步−ΔFA 都是 −6.3 ms，超出 E2E-PLAN §4 判定 2 的 5 ms 门限（三对比较都超出；s6 对 r29 是 attention −53 ms、单步 −37.0 至 −39.1 ms）。这部分在 attention 以外或是噪声，RESULT-e2e.md 没有讨论，也没有拆解（A0 的 kineto trace 无效）[1002__e2e/RESULT-e2e.md；1002__e2e/e2e/runs/analysis.1002_095504.txt §2；5497446d] |
| **阶段 10 · A0 bwd job r24–r27（A0 编号）、gb 尺子设计与 10-02 挂卡** | | | | | | | | | | |
| 82 | 09-30 | bwd | A0 r24（deep；refactor h75 采用 s6） | 5.295 | 1038.34（bwd） | 104.0% | — | A0-R | 候选 g77 DQT_VT_KEEP（只改寄存器分配），在 prod/proxy 上 null；deep profiling 第一次跑通 ATT，判定瓶颈是功耗墙（操作数全零时间 −34%） | 被接受的“1.4263×”是 fast 中位数噪声造成的假阳性（prod 0.9986），随后 fast 改按 min 计分，加 gain_weights 1/0.25/0（4d61867f）。best_round=24，即 s6 + DQT_VT_KEEP |
| 83 | 09-30 | bwd | A0 r25 | 5.292 | 1039.02（bwd） | 103.9% | — | A0-R | g75 DQT_DSADDR prod −0.19%；g73（dQ 用 packed bf16 原子）在 CPU 数值预筛中判死 | 加权 gain 1.0033 < 1.007，被否；14:34 停在 r26 开始前，交机 [070c8895]。r25 虽被否，state.yaml 的 champions 仍把 prod/proxy 记为 25（1039.02 / 1002.18），fast 记为 24，best_round=24；续跑时 r25 的读数就是 best-ever 门槛 |
| 84 | 10-02 | bwd | A0 r26 | 6.486（w4f_r4b） | 847.73（bwd） | 84.7% | — | A0-D | 改走融合 w4f：noatom 上限 4.23 ms，dQ 原子池占 2.24 ms（34.7%）；F=2 加宽在编译门判死（spill 616） | 被否，又因 FastError 记为 failed |
| 85 | 10-02 | fwd+bwd | gb 尺子设计；fwd job 从 B0 备份恢复（5696ae53，两者都没上线） | — | — | — | — | A0-D（只做 CPU 侧准备，没有用卡） | gb 尺子（训练工作点）：每次计时前跑 10 个 bf16 32768×4096×14336 GEMM，proxy/prod 按 gb 中位数计分，fast 仍用 blk。依据是同日 A0 的时间比（blk / gb / e2e）：bwd s6/r29 0.806 / 0.774 / 0.754，r29/ASM 1.206 / 1.341 / 1.290，fwd r16/ASM 1.081 / 1.349 / 1.263，这三项 gb 更接近训练；例外是 s6/ASM（0.972 / 1.038 / 0.974），gb 偏悲观约 6%。fwd job（B0 的 gfx1250-flydsl-attn-fwd-b0-20260927）从 0928 备份解出：best_round 16（r13ns），round 20 的 act 待在 A0 上重做，h48（采用 r19 的 op）在 round 21 开头执行 | 两者都只是备好，上线需用户批准。A0 上 fwd 的 yaml 没有 gain_weights，验收仍是三形状算术平均。同一时段 bwd job 被另一会话在 10:21:42 resume 跑 r26 [1002__oe/RULER.md §0,§1；1002__oe/FWDJOB.md §0] |
| 86 | 10-02 | bwd | A0 r27（未完成） | — | — | — | — | A0-D | opt agent 只为读 VGPR，就在 prod 形状直接跑了两个从未上过卡的变体（A_g74、B_g82），还用了宿主 hipBLASLt 库。WEDGE-1002.md 和 1ee0dd59 的提交标题都把它们记为 w4f 融合变体，是误记。核对 rounds/027/_scratch/arms 源码，两者都是 s6 原样拷贝后加 cluster multicast：A_g74 在 k_dkdv 里让同一 (bat, hkv) 的 4 个相邻 KV band 共享 Q/dO tile（DKDV_CLUSTER=4），B_g82 在 k_dqg 里让同一 GQA 组的 4 个 q head 共享 K/V tile（DQG_CLUSTER=4）。探针脚本是沿用 r26 w4f 的 build_probe.py | 11:40:59 UTC MES 不可恢复，挂卡，需 AC 断电；新增 h85：先 toy 后 prod，读 VGPR 一律 compile-only，只用镜像自带的 hipBLASLt [1002__oe/incident/WEDGE-1002.md；1ee0dd59] |
| **阶段 11 · 教程（10-05）** | | | | | | | | | | |
| 87 | 10-05 | e2e | 产品分支教程 e2e（docs/gfx1250_llama31_8b_e2e） | — | — | — | 1,361.9 ms/step，24,061 tps（按提交版本复跑 1,360 / 24,099）；漏掉 transpose_triton 时 1,689 / 19,406 | A0-D（sclk 最高 2.36 GHz） | 分支版 fwd r16 + bwd r29（flydsl 0.2.4），加 nkfix（NKFIX_CHECK=0）和镜像 hipBLASLt；附 Dockerfile、启动脚本、Primus hook patch | 20 步墙钟 53 s。这是 NKFIX_CHECK=0，不能直接对 #81 的 ASM 1,350.6 ms（NKFIX_CHECK=1）读成“只慢 0.8%”：按 CHECK=1 约 +33 ms/步（B0 实测，教程沿用）折算，ASM 在 CHECK=0 下约 1,318 ms，分支约慢 3%（推算，未实测）；bwd 版本（r29 对 s6）也不同。教程 §7 称固件更新前后同代码同方法，ASM attention + nkfix 的 e2e 1,947→1,351 ms（1.44×），这个倍数不纯是平台差异（见 c.1）[wt-llama31/docs/gfx1250_llama31_8b_e2e/README.md §3.2,§7；0927__b0/gemm/REPORT.md:83] |

### c.3 当前最优版本

| 项 | 版本 | 算子 ms | TF/s | 对 ASM | e2e | 机器 · 尺子 | 代码位置 / commit | 日期 |
|---|---|---|---|---|---|---|---|---|
| fwd 冠军 | r16 = r13ns（B0 fwd job 在 round 16 由 refactor h44 晋升；即 B0 r13 关掉投机 softmax） | 1.353 / 1.356 | 1626 / 1622 | 92.9% / 92.4%（时间比 1.076 / 1.082）；10-02 真实数据 blk 1.081，训练工作点 gb 1.349 | 在 #81 中每步 fwd 比 ASM 慢 9.2 ms | A0-R 09-30，分块，randn，同进程 ASM 1.257 | `0927__b0/champions/fwd_r16_r13ns`（5188c9da，df9aa3e8）；产品分支 `primus_turbo/flydsl/attention/gfx1250/`（5658a443，flydsl 0.2.4 下 1.292 ms，1.088×） | 09-28 晋升，09-30 复测 |
| bwd 冠军 | s6 = s5 + k_dqg VALU 精简（手工攻关；A0 job r24 = s6 + DQT_VT_KEEP，1038.34 TF/s） | 5.295–5.300 | 1037–1038 | 103.8%（ASM 5.497–5.500）；10-02 randn 104.1%；真实 blk 0.972（≈102.9%），gb 1.038（≈96.3%） | 在 #81 中每步 bwd 比 ASM 快约 4 ms（相邻配对 4.50 / 4.56，即正文的 4.5；两臂中位数相减 3.8–4.4），比 r29 快约 53 ms | A0-R 09-30 / A0-D 10-02，分块 | `0930__bwd/armsrc/s6`（7cd15aef，flydsl 0.3.2）；0.3.4.1 版 `1002__e2e/arms_src/bwd_s6_0341`（fd6dce02）。**还没进产品分支**，分支里是 r29：6.618 ms，81.0% | 09-30 |
| e2e 最佳 | fwd r16 + bwd s6（flydsl 0.3.4.1）+ nkfix（NKFIX_CHECK=1） | — | — | 单步为 ASM 的 0.9990 / 0.9985× | 1349.4 / 1349.9 ms/step，24,283 / 24,275 tps（ASM 1350.6 ms，24,262 tps） | A0-D 10-02，32L，训练 sclk 约 1.50 GHz | `1002__e2e/RESULT-e2e.md`（5497446d） | 10-02 |
| 产品分支现状 | fwd r16 + bwd r29（flydsl 0.2.4）+ nkfix（NKFIX_CHECK=0） | fwd 1.292 / bwd 6.618 | 1702 / 831 | fwd 91.9%，bwd 81.0% | 1,361.9 ms/step，24,061 tps（10-05，NKFIX_CHECK=0）。不能直接对 ASM 的 1,350.6 ms（10-02，NKFIX_CHECK=1）读成“只慢 0.8%”：按 CHECK=1 约 +33 ms/步（B0 实测，教程沿用）折算，ASM 在 CHECK=0 下约 1,318 ms，分支约慢 3%（推算，未实测）。#81 的 flyr29 臂同为 fwd r16 + bwd r29，为 ASM 的 1.027×，但那是 flydsl 0.3.4.1 构建、NKFIX_CHECK=1，不是分支本身；分支在 0.2.4 上用 buffer_store O writer，相对 0.3.4.1 约慢 fwd 1.3%、bwd 1.8%，与约 3% 的推算大体相符 [1002__e2e/RESULT-e2e.md；0930__port/PR_BODY.md；wt-llama31 README §3.2,§7] | A0-D | `dev/lhz/llama31_attn_opt`（c36cc124..58d5d798；教程 7e88d891/28e09a01/b8c543a9） | 09-30 / 10-05 |

### c.4 进展曲线要点

- **e2e 的头号杠杆是 GEMM，不是 attention。**
  - hipBLASLt 路径修正让 A0 32L 从 244 到 2,027 tps（8.3×）。nkfix 又带来 A0 8L 6.16×；B0 32L 上 ASM 臂 20,718 tps，是修复前高档 2,016–2,022 tps 的 10.3×（对低档中位数 1,922 为 10.8×），其中 GEMM 每步 15.6–16.5 s→0.83–0.86 s，约 18–20×。
  - 修好之前，attention 只占单步不到 1% 到 3.3%：A0 09-13 32L 走 rocBLAS 时整条 attention 路径 35.7 ms × 32 = 1.14 s，约 0.85%（phase1/RESULTS.md:266）；A0 8L 走 hipBLASLt、无 nkfix 时 3.3%（BOTTLENECK-SHIFT）；B0 09-28 nkfix 前为 1.4–5.4%（0927__b0/e2e/RESULT.md:37）。ASM bwd 的 e2e A/B 因此被 e2e 噪声淹没（8L n=9，−0.28%，不显著）。修好之后 attention 占 15–22%（A0 8L 20.8%，B0 17–22%，A0-D 10-02 约 15%），op 级改进才在 e2e 里看得见。
- **Triton 阶段：A0 一天 2.44×，B0 一天 2.75×。**
  - A0-T 上 59.642→24.435 ms，靠两个 config 旋钮加 vendor aiter 融合 bwd。
  - B0-14 上先复现 Day-1 阶梯（28.121→12.994 ms），再靠 waves_per_eu、ASM fwd（fwd 1.82×）和 in-thread transpose 降到 10.235 ms，相对出厂 2.75×。
  - waves_per_eu 在限频机上测不出收益，满频下值 4.2%：机器状态本身决定哪些杠杆有效。
- **ASM fwd 是 fwd 方向单次最大的一跳**（A0 4.253→1.549 ms，2.74×），此后一直作为 bar。ASM bwd 的 bar 先被 10.160 ms 的 shim 伪数把耗时高估了 32%（等于低估了 ASM 的速度）；09-24 普查定为 7.6766 ms，FlyDSL 的“0.92×”随之更正为“0.70–0.72×”。
- **FlyDSL bwd 从零做到约 0.71×**（A0-T，20 轮，57.24→511.42 TF/s，8.9×；对 09-24 bar 716 TF/s 为 0.714，按 r22 同会话复测 516.87 为 0.72）。
  - 按 prod 冠军 TF/s 逐轮相比（各轮在自己会话里的读数），最大的四跳是 r1 1.62×（57.24→92.62，causal tile 跳过）、r4 1.60×（207.47→330.98，LDS padding 消除 64 路 bank 冲突）、r2 1.55×（地址钳位 + 32 深收缩 + BLOCK_KV 32）、r3 1.45×（k_dq BLOCK_Q 64）。
  - 框架的三形状 gain 排序不同：r2 1.921、r4 1.512、r13 1.370、r1 1.342、r3 1.297。r13 的 gain 全部来自 fast/proxy，prod 没动。
  - r12 之后进入效率期，上限被 7 vs 5 GEMM 的结构因子 1.386× 卡住，r13–r23 在 prod 上只多出约 2–3%。
- **fwd 的收益主要来自调度和 softmax 指令。**
  - A0-T 上 935→1111 TF/s（+18.8%），由 longest-first 调度、packed exp/row-sum、nodelay 三项组成。
  - B0 上投机 softmax 在 randn 上 +4.6%，但真实数据和训练时钟下反而更慢。r16 关掉投机（r13ns）后，真实数据快 8–15%，真实数据加训练时钟快 12–23%。同一个改动，在两把尺子上方向相反。
- **尺子和判据的修正，本身“改写”了一部分进度。**
  - B0 从交错改为分块后，fwd 占 ASM 从约 85% 变成约 96%。同时撤回了两个假赢（lab2-L12 +6.4%→−0.4%，lab3-L21 +6.3%→约 +0.3%），fwd r6 的 +5.7% 下修为 +4.6%（仍为真实收益）。
  - r15 的算术平均、B0 r27 的 geomean 误判 target_met、A0 r24 的 fast 中位数假阳性，三次框架判据错误都是手工纠正的。
- **A0 刷固件让同一份代码快了 1.24–1.67×（o1 顺序；反序 o2 下 ASM fwd 1.23×、FlyDSL fwd 1.47×），FlyDSL 受益更多**：bwd r29 从 ASM 的 69.5% 升到 83.6%，fwd r13ns/ASM 从 1.29 降到 1.08。09-29 之前 A0 的绝对数全部作废。e2e 没有干净的同口径倍数（1.44× 不纯，见 c.1）。
- **bwd 手工攻关一天从 83.6% 做到 103.8%**（09-30，r29 6.58→s6 5.295 ms，−19.5%）。
  - 核心杠杆是 TDM 3 级 LDS ring，加上把下一轮 B 操作数提前读回寄存器。两步都是整个 op 的变化：只换 k_dkdv 的 dkdv_tdm3 使整个 op 相对 r29 −11.17%（5.856 vs 6.592 ms）；k_dqg 也换成 TDM（s5）后，整个 op 相对 s4 再 −2.1%。消融 A2 先给出了上限，ATT（刷固件后才可用）再定位到具体指令。
  - s6 仍是 7-GEMM 结构，多做约 39% 矩阵运算，依然快过 ASM。
  - 5-GEMM 融合版 w4f 不做 dQ 原子累加时（relax_noatom）只要 4.241 ms（ASM 的 129.6%；输出错误，只作上限），但 dQ 原子写要吃掉约 5 ms，所以暂停。
- **e2e 从落后 4.8–5.0% 追到持平。**
  - B0 上 fly/asm 1.0475（r6+r20）→1.0454（换 fwd r13，仍带投机）→1.0403（换 fwd r13ns）→1.0323（换 bwd r29）。A0-D 再换成 s6 后为 0.999，每步 bwd 省 53 ms。B0 和 A0-D 的绝对 ms 不可比，这里只比同进程比值。
  - 剩余差距在 fwd：训练工作点下 r16/ASM 1.349，每步慢 9.2 ms。
  - 产品分支仍是 r29，s6 还没移植进去。同组合（fwd r16 + bwd r29）在 #81 中为 ASM 的 1.027×，但那是 flydsl 0.3.4.1 构建；分支本身（0.2.4）只测过教程 e2e 1,361.9 ms/步，那是 NKFIX_CHECK=0，按 +33 ms/步折算到同口径约比 ASM 慢 3%（推算，未实测）。下一步是 fwd，以及把计分尺子换成训练工作点的 gb 尺子（10-02 已设计，未安装）。
