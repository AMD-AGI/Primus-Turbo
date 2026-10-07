## 附录 A 口径与可比性

### A.1 机器、时钟阶段与驱动/固件

四节对同一阶段的叫法不同：(a) 用 A0-VR / A0-RF / B0，(b) 用 E1–E5，(c) 用 A0-H / A0-T / B0-14 / B0-28 / A0-R / A0-D，(d) 用 A0-限频 / A0-新固件 / B0。对照如下。

| 阶段 | 各节记号 | 机器 | 日期（UTC） | VBIOS / SMU / 驱动 | 负载时钟 | 可比性与备注 |
|---|---|---|---|---|---|---|
| A0 未限频 | (c) A0-H | A0 `heliosr-1b114-c07-1`，独占 1× gfx1250（MI455X，256 CU，wave32，每 CU 320 KB LDS，432 GiB HBM，PPT0 功耗上限 2500 W） | 09-04 及之前 | 未单独记录 | 09-04 负载下 1699–1703 MHz（上限 2400） | 这一阶段没有本项目的 attention 测量 |
| A0 VR 限频 | (a) A0-VR；(b) E1（09-10..09-17）、E3（09-17..09-27）、E3′（09-28）；(c) A0-T；(d) A0-限频 | A0 | 09-10..09-28（最早的限频记录是 09-10 22:38） | VBIOS `113-M4500001-630A`；SMU fw 125.7.1；amdgpu 7.1.1.31300009（dkms 7.1.1-2397345）；内核 6.14.0-37-generic，cmdline 带 `modprobe.blacklist=amdgpu` | sclk DPM 只剩 500/1100 MHz；负载约 0.94–1.07 GHz（09-24 prod 计时窗口实测 998–1029 MHz） | 09-29 之后，这一阶段的绝对数全部作废，只能引用同进程比值 |
| B0 09-14 | (b) E2；(c) B0-14 | B0 `ctheliosp-1b112-a37-1`，4× gfx1250（GPU0 降级，当天只用 GPU1–3） | 09-14 | 未记录（与 A0 同镜像） | 2133–2244 MHz | 四卡同时负载时，单卡计时漂移最高 50% |
| B0 09-27/28 | (b) E4；(c) B0-28；(d) B0 | B0，每卡一个快照容器 fa-g0..g3；09-28 GPU1 挂死，交还时仍未恢复 | 09-27..09-28 | 未记录 | 空闲 2356 MHz；op 级（分块尺子）约 2.15 GHz；nkfix 后训练整步约 1.34–1.39 GHz（终版运行 1,342 / 1,371 MHz），训练中 attention 窗口 1.25–1.69 GHz（整板约 2.13 kW 功耗墙） | B0 09-27/28 的会话不在本机，数字只来自文档和账本 |
| A0 刷固件后 | (a) A0-RF；(b) E5；(c) A0-R；(d) A0-新固件 | A0 | 09-29..10-01 | VBIOS 700E；SMU 125.12.0；amdgpu-dkms 7.1.0-2411946 | sclk DPM 500/2356/2400；op 级 1.74–2.03 GHz；满载 WMMA 约 1.4–1.8 GHz（2500 W 功耗墙）；训练中约 1.5 GHz | 开机仍打印 VR 告警，但 DPM 不再截到 1100 MHz；ATT 可用 |
| A0 新驱动 | (b) E5（10-01 起）；(c) A0-D；(d) A0-新固件 | A0 | 10-01 起 | VBIOS 700E 不变；amdgpu-dkms 7.1.0-2412954 | 同上；GEMM 突发后（训练工作点）约 1.28 GHz；e2e 整步 sclk 中位约 1.50 GHz | randn 基线与 09-30 一致（bwd s6 为 ASM 的 104.1%，fwd 时间比 1.079） |

**软件栈**：容器 `fa-repro`（镜像 `fa-tune:deps`，即 `amdprimus/amdprimus:gfx1250-20260910` 加 torchtitan 依赖），torch `2.11.0+rocm7.14.0a20260625`，triton 3.6.0，HIP 7.14，python 3.12.3；aiter-src 09-13 为 @`ffa945f9`，09-17 起为 @`6963ae9d`；torchtitan v0.2.2（`73a0e6979`）。flydsl 有三个版本并存：
- 0.2.4：镜像自带，Primus-Turbo main 和产品分支固定用它；
- 0.3.2：aiter 固定的版本（`~/.local/flydsl032`），A0 bwd job 和手工 s1–s6 用它；
- 0.3.4.1：`~/.local/flydsl0341`，09-25 起 fwd job、B0、e2e 用它。
0.3.x 删掉了 `flydsl.expr.buffer_ops`，所以和 primus_turbo 不能在同一个进程里用。

### A.2 跨阶段只能比比值，而且比值本身也会变

| 对照 | 倍数 | 说明 | 出处 |
|---|---|---|---|
| A0-VR → B0，同一份 Triton 类代码 | 5 行 like-for-like 1.88–2.01×（中位 1.93×）；FINAL-TABLE 全 7 行 1.88–2.12×（中位 1.97×） | 两张卡之间的比值。B0 自身也有 VR 警告（代价约 9%），所以这不是 A0 自己的限频代价；09-14 文档把它写成"VR 限频代价实测约 1.93×，1.65× 作废"，这个说法不成立 | (a) 记号与口径 ①；(d) d.3；`0914__repro__c07/RESULTS.md:32-38、124-130`；`0914__repro__c07/FINAL-TABLE.md` |
| A0-VR → B0，ASM fwd | 1.118×（1.572 对 1.406 ms） | 两边都是干净的 ASM fwd，成立；ASM fwd 几乎不随 sclk 变 | (d) d.3；`0915__opt/JIRA-TRACE-ANALYSIS.md` §五 |
| A0-VR → B0，"ASM attention 合计" | 1.12–1.15×（bwd 1.150、total 1.146；A0 11.726 ms 对 B0 10.236 ms） | **不成立**：bwd 一项比的是 A0 的 shim 伪影 10.160 ms 和 B0 的融合 Triton bwd 8.835 ms（09-14 B0 还没有 ASM bwd）；同代码的融合 Triton bwd 约为 2.01×。三项里只有上一行 fwd 的 1.118× 成立 | `0915__opt/JIRA-TRACE-ANALYSIS.md` §五；(a) 记号与口径 ②；(b) b.4 末行；(d) d.3 |
| A0 刷固件前 → 后，同代码同尺子（op） | FlyDSL fwd r13ns 1.47–1.50×（o1 2.023→1.353 ms 为 1.50×，o2 1.992→1.356 ms 为 1.47×，均值约 1.48×）；ASM fwd 1.24×（o2 1.23×）；FlyDSL bwd r29 1.67×；ASM bwd 1.39× | 负载时钟比约 1.75×，实测提速都低于它；FlyDSL 对 sclk 更敏感。驱动也同时变了（7.1.1-2397345 → 7.1.0-2411946），不能与固件严格分开 | (a) a.1；(c) c.1；`0928__a0_repro/REPORT.md` §6.1–§6.3 |
| A0 刷固件前 → 后（e2e，ASM + nkfix） | 1,947 → 1,351 ms，1.44× | 教程称"同代码同方法"，两边也都是 `NKFIX_CHECK=1`，但这个倍数不纯是平台（固件/时钟）差异：① 1,947 ms 是 09-28 a0_p3a（ASM / r6+r20 / r16+r29 三臂按步交替，90 步）里 ASM 臂的中位数；该进程 step 2 起加载的 FlyDSL fwd 树 `_env.py` 把 `HIPBLASLT_TENSILE_LIBPATH` 改指到宿主库（hipBLASLt 会不会中途重读没有证实），同批另有 1 次 NaN、2 次挂卡，09-28 的 A0 绝对数已整批作废；② 两次之间驱动从 7.1.1-2397345（09-28）经刷固件时的 7.1.0-2411946，换成了 10-02 的 7.1.0-2412954；③ 10-02 用的是另写的套件 `1002__e2e/e2e/`（fwd 树副本 `fwd_r16_imglib` 删掉了 LIBPATH 改写，加 BLAS guard）。两边相同的是 nkfix_b0 + `NKFIX_CHECK=1`、32L 配置模板和三臂按步交替的做法 | (a) a.1；(d) d.2.1；`0928__a0_repro/REPORT.md` §2、§5.3、§6.3–§6.4；`1002__e2e/RESULT-e2e.md`、`E2E-PLAN.md` §1–§3；wt-llama31 README §7 |
| 同一把分块尺子、同一批 arm，跨时钟阶段 | r13ns/ASM 1.29（E3′）→ 1.08（E5）；r29/ASM 1.44 → 1.20；bwd r29 占 ASM 69.5% → 83.6% | 比值本身也不能跨时钟阶段迁移 | (b) 开头；(c) c.1 |
| 同机同代码，只换尺子 | B0 fwd 交错 → 分块：占 ASM 约 85% → 约 96% | kernel 没有变 | (c) c.1 |
| 同代码，只换输入和工作点（A0 10-02） | bwd s6/ASM blk 0.972 → gb 1.038；fwd r16/ASM blk 1.081 → gb 1.349 | FlyDSL 对时钟敏感，ASM 几乎不敏感 | (b) b.2；(c) c.1；(d) d.3 |
| 09-13 按 MAX_CLK 估算的限频代价 | 1.65× | 时钟比估算，不是实测（原文以 MAX_CLK 1100 MHz 对 09-04 负载 1699–1703 MHz 得出约 1.65，没给算式；按这两个数算约 1.55）。09-14 拿跨卡比 1.93× 判它"作废"不成立，本报告不再称作废；A0 自身的实测见上面两行（op 1.24–1.67×，e2e 1.44×） | (a) a.1；(d) d.3；`0913__opt_plan__claude/phase0/PLATFORM-ESCALATION.md` |

### A.3 FLOP 口径

- **现行口径**（op-evolve `tools/op_flops.py`）：每次调用 fwd 2.199292e12、bwd 5.498229e12（5-GEMM 名义计数，causal 取 (s+1)/(2s)），fwd+bwd 7.697e12；ms = FLOP ÷ TF/s。(b) 表中带 `*`、(c) 表中标 derived 的值都是这样换算的。
- **09-13..09-16 的文档**：只报 fwd+bwd 合计（7.697e12 ÷ 总耗时）。(a)(b) 表里的分方向 TF/s 是按单向耗时换算的。
- **7-GEMM 实发口径**（本报告不用）：树内 vendored 融合 bwd 发 7 趟 GEMM，issued 7.6975e12，是名义的 1.400×；我们的 FlyDSL bwd（r0 到 s6）issued 7.7395e12，是名义的 1.40764×（含对角块粒度 ×1.00546）；aiter ASM 只发名义的 1.0155×。所以 s6 的"占 ASM 103.8%"是在多做约 39% 矩阵运算的情况下达到的。0913/0914 文档曾按实发口径报 bwd（例如 B0 vendored 融合 bwd 748.4 对名义 534.6 TF/s）。
- **其它旧口径**：09-23 STAGE2-FWD-SWEEP 用 fwd ≈2.233e12（旧 harness，TF/s 偏高约 1.5%）；09-17 DKDV-FIRST-TIMING 的 5.8 TF/s 是单头的自定义 useful 口径，不可比；09-14 B0 的 Triton op-evolve job 只计 bwd。
- **GEMM**：TF/s = 2MNK ÷ 耗时。
- **e2e**：tokens/s 是单卡吞吐；TFLOP/s 取 torchtitan 自报值，或按 torchtitan 口径由 tps 换算（32 层约 57.9 GFLOP/token，8 层约 16.84）；torchtitan 的 MFU 以 A100 的 312 TF/s 为分母，全部不可用。
- **比值方向**：(b) 的"相对 ASM" = 本行耗时 ÷ ASM 耗时（>1 表示比 ASM 慢）；(c) 的"占 ASM" = ASM 耗时 ÷ 本行耗时（>100% 表示比 ASM 快）。两者互为倒数，引用时要注意。

### A.4 尺子

| 尺子 | 用在哪里 | 做法 | 已知偏差 |
|---|---|---|---|
| tune（`tools/gfx1250/tune_attention.py`） | 09-13..09-16 的 op 阶梯（(b) 的 E1/E2） | 一层 fwd+bwd；CUDA event 取中位；iters 20 / warmup 5；每个 rep 冲刷 256 MiB L2；out/dq/dk/dv 四张量 SQNR ≥50 dB | 计时含 autograd 管路；ASM bwd 走 autograd shim 时的读数（10.160 ms）是伪影 |
| bench0910 / bat | 09-10 Primus 侧 bench；09-30 产品分支的 `bench_attention_turbo.py` | 仓库自带 bench | 自带的正确性检查不可信：0910 版对所有后端都报同一个 max-abs；0930 版在非因果 s8192 上 dq 参考只有约 12 dB，CPU fp32 参考实为 52 dB |
| E2_bench（op-evolve phase-1） | 09-11 | warmup 5，iters 20，重复 3 次 | — |
| op-evolve benchmark.py，逐次交错 | A0 两个 job（09-17..09-27）；B0 fwd r5–r9；B0 bwd r24–r26 | 只计 kernel 发射；每个 arm 51 次取中位；回文交错；ASM beat 与候选同进程；三个形状 fast/proxy/prod，只按 prod 排名 | FlyDSL/ASM 比值偏低（fwd 约 25%，bwd 约 3.4%）；ASM 之后的 I-cache 惩罚（h28 起 beat 单独一个进程） |
| 分块（blocked，h40/h41/h66） | B0 fwd 从 r10 的 act 起，B0 bwd 从 r27 起；之后 A0 一直沿用 | 每个 arm 先跑 4 次不计时，再连续计时 9 次；块间回文；每个排名进程带一个 A/A 副本 | A/A 误差 fwd ±0.19%、bwd ±0.05%；A/A 偏离超过 0.5% 整个进程作废；新旧尺子的绝对 ms 不可比 |
| 真实数据（h45/h47） | B0 fwd r18 起；A0 10-02 的 realab | B0 训练 step 43 导出的 6 层真实 q/k/v（`/home/lihuzhan/_prof_dump/qkv_call0*.pt`） | 真实数据的 score 标准差为 21–53（randn 约为 1）；投机 softmax 在 13–25% 的 tile 步回退重算 |
| gb（训练工作点） | A0 10-02 的 realab；op-evolve 计分尚未安装 | 每次计时前先跑 10 个 bf16 32768×4096×14336 GEMM，把时钟压到训练工作点（A0 约 1.28 GHz） | 4 项比值里有 3 项比 blk 更接近 e2e；s6/ASM 是例外，gb 偏悲观约 6% |
| e2e | 09-13 起 | Llama-3.1-8B BF16 32 层（09-15/16 部分为 8 层），MBS=GBS=4，seq 8192，单卡，AC none，compile 关；09-16 起以 nkfix 为前提；同进程 ABBA/BAAB 或按步交替；剔除 step 1–7 和 profile 步 | `NKFIX_CHECK=1` 每步约多 33 ms；出现 NaN/inf 的运行整次作废；e2e 里 ASM 每层比两把 op 尺子测得都快（未解释） |

### A.5 噪声地板

- **op**：本算子同 session 的噪声地板为 0.24–0.66%（09-24 之前曾误用 HipKittens GEMM 阶梯的 1.57%）；fwd 跨 session 漂移约 1.5%；同一份 r7 代码在 AC 断电前后读到 382.73 → 404.06 TF/s（+5.6%）；分块 A/A 为 fwd ±0.19%、bwd ±0.05%；验收的 min_gain 为 0.007（bwd 早期为 0）。
- **fast 形状**：每次约 55 µs，受 launch 开销主导。跑同一个 kernel 的两个臂，min 只差 0.05%，median 可以差 56%，所以后来 fast 改按 min 计分，权重设为 0。
- **e2e**：A0-VR 上 nkfix 之前呈三模态，跨运行 sd 3.91%（簇间 1.8% / 6.8%）；nkfix 之后 sd 0.42%；B0 有一个进程中途整体掉档约 5%（2,015 → 1,918 tps，原因没查）；步数从 20 减到 10 时，step 4–8 的中位数差 ≤0.22%。
- **卡上争用**：B0 09-14 四卡满载时单卡计时漂移最高 50%；邻卡跑持续 GEMM 会让被测卡的 attention 慢 3–11×。

### A.6 各节之间的不一致：已统一与未定

第一版在这里列了 11 条四节之间互相矛盾的记录。本版把其中能核实的几条改进了正文，列在"已统一"下（若某处仍残留旧说法，以这里为准）；其余的要么还没核实，要么是口径不同、算不上矛盾，列在"未定或口径不同"下。

**已统一**

- **A0 09-13 e2e 的 torchtitan 版本**：v0.2.2（Primus `third_party/torchtitan` 固定在 `73a0e6979`）。`0914__campaign/RESULTS.md:65` 写的 "0.1.0" 是笔误。
- **"VR 限频代价实测约 1.93×"**：1.88–2.01×（中位 1.93×）是 A0-VR→B0 的跨卡比（B0 自身也有约 9% 的 VR 代价），不是 A0 自身的限频代价。A0 自身刷固件前后为 op 1.24–1.67×、e2e 1.44×（后者另有附加差异）。09-13 的 1.65× 是时钟比估算，不再称作废（见 A.2）。
- **"ASM attention 跨机只差 1.12–1.15×"**：只有 fwd 的 1.118×（1.572 对 1.406 ms）成立；bwd 一项比的是 shim 伪影 10.160 ms 对融合 Triton 8.835 ms，不成立（见 A.2、d.3）。
- **`c1325c7e` 提交信息里的 220.6 TF/s**：测量条件不明（提交信息没写 hkv、是否 causal、时钟），220.6 / 129.4 = 1.70× 和哪种时钟比相符都没有定论。不再说"是在未限频的卡上测的"。
- **AC 次数**：以 d.1.4 为准。A0 因挂卡人工断电/重启 27 次，算上 09-29 那次无效 AC 为 28 次；其中 09-15 五次、09-16 九次（UTC）。
- **刷固件后 FlyDSL fwd 的提速**：1.47–1.50×，o1 顺序 2.023→1.353 ms 为 1.50×，o2 顺序 1.992→1.356 ms 为 1.47×，均值约 1.48×（`0928__a0_repro/REPORT.md` §6.1）。

**未定或口径不同**

1. **09-14 B0 "turbo vs flex 1.459×" 的来源**：(b) b.3 的撤回说明写"差异来自其它 turbo patch（float8 / mx linear）"；(a) a.3 #2、#11 认为这个归因存疑（按本机 Primus 代码，`converters: []` 下看不出这两个 patch 怎么生效），能确定的只是它不来自 attention。未定。
2. **aiter Triton MHA 出厂读数（09-13 34.565 ms 对 09-15 28.035 ms）**：(b) b.2.1 引用 0915 的判断，认为 09-13 那一行偏差 +23%、不可信；(a) a.2 #12 认为更可能异常的是 09-15 那次（它的 fwd 3.725 ms 像是调优过的前向）。未复测，未定。
3. **B0 训练中的时钟**：(b) E4 写 1.25–1.45 GHz，(c) B0-28 写 1.34–1.39 GHz，(d) 写 1.25–1.69 GHz。前者偏向整步 sclk 均值，后者是训练中 attention 窗口的读数，口径不同，未逐条核对。
4. **A0-VR 的负载时钟**：各节分别写 0.94 / 0.95 / 0.97–1.07 GHz，另有单点采样 943 MHz。都指 VR 限频下的负载时钟，差异来自测量窗口不同。
5. **L21 的真实增益**：REPORT-0928.html 写 +0.3%，ruler/REPORT.md 的分块读数是 +0.6%（(d) 末尾已列）。未定。
6. **10-02 e2e 的 fwd/bwd 拆分（已统一写法，仍有一项未拆解）**：`1002__e2e/RESULT-e2e.md` 正文的"bwd 每步快 4.5 ms"是相邻配对差的中位数（两进程 −4.50 / −4.56 ms）；两臂各自中位数相减是 3.8–4.4 ms（166.54−162.15、166.41−162.62；RESULT-e2e.md 表四舍五入后为 3.9–4.3）。两者不矛盾，本报告统一写"约 4 ms"。仍未拆解的是：fly 的 attention 合计每步比 ASM 多约 4.3–5.2 ms，单步却快 0.7–2.0 ms，两进程的 step − FA 都是 −6.3 ms，超出 E2E-PLAN §4 判定 2 的 5 ms 门限；s6 对 r29 同样如此（attention −53.0 ms，单步只快 37.0–39.1 ms）。这部分差额落在 attention 以外或计时噪声里 [`1002__e2e/e2e/runs/analysis.1002_095504.txt`]。

## 附录 B 术语表

| 术语 | 含义 | 见 |
|---|---|---|
| A0 / B0 | A0 = `heliosr-1b114-c07-1`（本机，单卡 gfx1250）；B0 = `ctheliosp-1b112-a37-1`（4× gfx1250，同一 XGMI hive） | 附录 A.1 |
| 时钟阶段代号 | A0-VR / A0-T / A0-限频 / E1·E3·E3′ = A0 刷固件前；A0-RF / A0-R / A0-新固件 / E5 = 刷固件后；A0-D = 10-01 换驱动之后；E2 / B0-14 = B0 09-14；E4 / B0-28 = B0 09-27/28 | 附录 A.1 |
| VR 限频 | A0 在 09-10..09-28 期间每次 amdgpu init 都打印 `GPU is throttled ... VR`，sclk DPM 只剩 500/1100 MHz；09-29 刷 VBIOS/SMU 后解除 | (a) a.1 |
| ASM（aiter ASM） | aiter 为 gfx1250 预编译的汇编 kernel（`.co`）。fwd 为 `fmha_bf16_pertokenBf16_hd128_128x256_mask.co`（入口 `fmha_fwd_with_sink_asm`）；bwd 为 odo → `bwd_hd128_bf16_causal_br_a32_pssk` → dq_convert 三个 kernel。本项目所有"相对 ASM / 占 ASM"的基准 | (b) b.1、b.4 |
| beat / bar | beat：op-evolve 里与候选同会话测的 aiter ASM 臂；bar：引用的 ASM 基准值（A0-VR 09-24 普查 fwd 1.5724 ms、bwd 7.6766 ms；各阶段的 bar 见 (c) c.1） | (c) c.1 |
| 10.160 ms（shim 伪影） | 09-15 用 autograd shim 测的 ASM bwd：每次调用做 3 次 hipModuleLoad、新分配约 1 GiB scratch，另含约 1.4 ms autograd 开销。09-24 撤回，真实 bar 为 7.6766 ms | (b) b.4；(d) d.3 |
| GQA 越界写 / `dkdv_heads=q` | aiter gfx1250 ASM bwd 的 grid 是 (kv_tiles, nhead_q, batch)，却按 q head 去索引 kv 尺寸的 dk/dv，G=4 时 4 个 WG 无同步地写同一个 tile。绕法是按 q head 分配 dk/dv，再在 host 端规约 | (a) a.2 #13；(b) b.4 |
| 7-GEMM / 5-GEMM | 我们的 1-wave bwd 为了 dq 确定（不用原子）要发 7 个 GEMM 当量，aiter ASM 用 dq 原子累加只发 5 个；结构因子 1.386–1.408× | (b) b.4；附录 A.3 |
| SQNR 门 | 输出与 fp32 参考的信噪比。out/dq/dk/dv 四张量各自 ≥50 dB（fwd 门后来降到 49 dB）；dk/dv 要求 200 次逐位一致，dq 自 09-25 起改为 run-to-run ≥70 dB | (d) d.5、d.6 |
| nkfix | e2e 中绕开 hipBLASLt 坏 tile 的运行时补丁：用 `TorchDispatchMode` 拦截 `aten::mm`，把反向 GEMM 的操作数改成前向那种物理布局（dgrad：`mm(A, B.t().contiguous().t())`；wgrad：`mm(A.contiguous(), B.t().contiguous().t())`，wgrad 的判据要先于 dgrad），从而命中 `MT256x256x128`，而不是 GEMV tile `MT32x16x32`。版本：A0 v1（09-15，只改 dgrad）；v3（09-16，加 wgrad；规则 3 让 wgrad 走 FlyDSL GEMM 离线表，默认关）；B0 `nkfix_b0.py`（09-28 重写，Triton 分块转置 `transpose_triton.py`、按输出维分块、`NKFIX_CHECK` 非有限值检查，教程用这一版）。开销每步约 70 ms，`NKFIX_CHECK=1` 再加约 33 ms | (d) d.2.2 |
| MT32x16x32 / MT256x256x128 | hipBLASLt（Tensile）kernel 的 macro tile。前者是 GEMV 用的小 tile，反向 GEMM 落到它时只有 50–80 TF/s；后者是 nkfix 之后全部命中的大 tile | (d) d.2 |
| refcache | 把精度门和 benchmark 用的 fp32 参考预先在 CPU 上算好并缓存（prod 10m26s、1.1 GB），避免在卡上跑 fp32 Tensile 参考（A 类挂卡的来源）；sha 对不上时直接判 FAIL | (d) d.1.3、d.5 |
| prod / proxy / fast | op-evolve 的三个计分形状 (B, Sq, Skv, Hq, Hkv, D)：prod (4,8192,8192,32,8,128)，即 Llama-3.1-8B 训练形状；proxy (1,4096,4096,32,8,128)；fast (1,1024,1024,8,2,128)。只按 prod 排名（h7）；fast 是 launch-bound，后来权重设为 0 | (c) c.1 |
| 逐次交错尺子 | op-evolve benchmark.py 原来的计时法：各 arm 逐次交替、回文排列，每个 arm 51 次取中位。每次调用继承上一个 arm 留下的功耗/时钟状态，FlyDSL/ASM 比值系统性偏低（fwd 约 25%，bwd 约 3.4%） | (d) d.3 |
| 分块尺子（blocked ruler，blk） | 09-28 起的计时法（h40/h41/h66）：每个 arm 先跑 4 次不计时，再连续计时 9 次，块间回文；每个排名进程带 A/A 副本（fwd ±0.19%、bwd ±0.05%），A/A 偏离超过 0.5% 整个进程作废。blk 也用来指"分块尺子下的读数"，与 gb 相对 | (c) c.1；(d) d.3 |
| 真实数据尺子（real-dump ruler） | 用 B0 训练 step 43 导出的 6 层真实 q/k/v 代替 randn（h45/h47）。真实 score 标准差 21–53，randn 约为 1 | (d) d.3 |
| 训练工作点（training clock）/ gb 尺子 | 训练中 GEMM 把整板功耗顶满，attention 跑在比 op 测试低得多的时钟上（B0 训练中 1.25–1.69 GHz；A0 在 GEMM 突发后约 1.28 GHz）。gb 尺子在每次计时前先跑 10 个 bf16 32768×4096×14336 GEMM 来复现这个工作点。ASM 几乎不随时钟变，FlyDSL 会变慢。gb 已设计，op-evolve 计分尚未安装 | (d) d.3；`1002__oe/RULER.md` |
| wedge（挂卡）类别 A/B/C | 卡挂死后驱动 reset 完不成，只能 AC 断电恢复。按 dmesg 首行分：**A** 是卡上 fp32 Tensile 参考 GEMM 的地址越界写（sub-4GB 截断地址、PF 0x5 / RW 0x1）；**B** 是首条为 `INVALIDATE_TLBS` 超时、此前没有任何内存故障；**C** 是候选 kernel 越界读引发 `GC_UTCL2` 故障风暴，伴随 `IH ring buffer overflow`。另有 S（启动期 MES 故障族）、K（已知操作诱因）、R（资源/进程）、F（进程级可恢复故障，不算挂卡）、D（降级）、X（驱动/固件等非 kernel 原因） | (d) d.1.1 |
| AC cycle（AC 断电） | 用户本人到机器前给整机断电再上电，是挂卡后唯一的恢复手段（`modprobe -r amdgpu` 会让整机失联）。恢复后要 `sudo modprobe amdgpu`、把 `dmesg_restrict` 设回 0、确认 KFD 为空、`git fsck` | (d) d.1.3 |
| MES / KFD | MES 是 GPU 上的硬件调度微码，挂卡签名多为 `MES failed to respond`、`wait for reset ack`；KFD 是 `/dev/kfd` 计算驱动接口，挂卡后进程的引用不会归零 | (d) d.1 |
| TDM | Tensor Data Mover：gfx1250 的异步张量搬运单元，按描述符把 global 数据搬进 LDS，FlyDSL 里对应 `make_tdm_atom` / `tdm_ops`。s3–s6 的主杠杆就是"TDM 3 级 LDS ring + 把下一轮 B 操作数提前读回寄存器" | (c) #73、#76 |
| WMMA | gfx1250（wave32）的矩阵乘累加指令，取代 CDNA 的 MFMA。roofline 实测每条 WMMA 8 cycle，WMMA↔DS 切换约 29 cycle | (c) #69；`0930__roofline/REPORT.md` |
| MFMA / `ds_read_tr16_b64` / `permlane32_swap` | CDNA（gfx950）专用指令，在 gfx1250 上都 "Cannot select"，所以 gfx950 的 FlyDSL / HipKittens attention 移植不过来 | (a) a.2 #6；(b) b.2.3 |
| ATT | rocprofv3 的线程追踪，能给出逐指令的 Hitcount / Latency / Stall。A0 在 09-29 刷固件之后才对 FlyDSL JIT 和 ASM `.co` 可用 | (d) d.4 |
| ITT | `TRITON_HIP_USE_IN_THREAD_TRANSPOSE`，09-14 op-evolve 找到的 Triton 编译开关，让融合 bwd 快约 9.7%（`a009599b`） | (c) #13、#14 |
| k_dkdv / k_dqg | 我们的 FlyDSL bwd 由几个 kernel 组成：k_delta（odo 预处理）；k_dkdv（按 KV tile 计算 dK/dV，GQA 在寄存器里规约，不用原子）；k_dq（计算 dQ；r29/s6 系列里 proxy/prod 形状走 k_dqg，toy/fast 形状走 split-K 版 k_dq_sp）；k_redsp 是 split-K 的归约 kernel。s1 起 k_dqg 放在 side stream 上，与 k_dkdv 并发 | (c) #26、#38、#72 |
| r20 | A0-VR bwd job 的冠军（k_dkdv 的 Q/dO 预取深度 1→2），511.42 TF/s；也是 B0 bwd 的起点 | (c) #42 |
| r19h | r19 加上 h33 的三处 clamp（越界预取 1,580→0，VGPR 904→724），由 B0 r28 的 refactor h68 晋升 | (c) #61 |
| u2n | k_dq 的 full KV 循环展开 ×2，去掉回边上 65 条 `v_mov_b64` 的寄存器轮转（k_dq 3.185→2.990 ms，h70）。A0 刷固件后反而是负收益，s 系列从 c1 起关掉（`DQ_U2=False`，−0.85%） | (c) #62、#70 |
| r29 | B0 bwd 冠军 = r19h + u2n + g86（dK/dV 的 epilogue 经 LDS ring），B0 上 656.5 TF/s（ASM 的 80.4%），A0 刷固件后为 ASM 的 83.6%。产品分支里的 bwd 就是它 | (c) #62、#69 |
| r13ns / r16 | B0 fwd job 第 13 轮的 kernel 关掉投机 softmax（`SPEC_STALE_MAX=False`）得到 r13ns，在第 16 轮由 refactor h44 晋升为冠军，所以 r16 = r13ns。它是当前的 fwd 冠军，也是产品分支里的 fwd | (c) #58 |
| s1–s6 / s6 | 09-30 A0 手工攻关的阶段冠军：s1 = tailpf + trorder + streams；s2 = s1 + soffset；s3 = k_dkdv 的 TDM 3 级 ring（dkdv_tdm3）+ dqg_ts + side stream；s4 = 精简 VALU、v_nop、地址计算，计数器去掉除法；s5 = k_dqg 也走 TDM ring（dqg_tdm）；s6 = s5 + k_dqg 的 VALU 精简。s6 为 5.295–5.300 ms、ASM 的 103.8%，是当前 bwd 冠军（flydsl 0.3.2；在 0.3.4.1 下 ISA 逐字节相同），还没进产品分支 | (c) #70–#77 |
| w4f | 4-wave 融合 bwd：5 个 GEMM、split barrier、dQ 用 fp32 `SCOPE_DEV` 原子累加。结果正确，但受整芯片原子吞吐限制，最好 6.509 ms（r4b，09-30），A0 r26 为 6.486 ms；不写 dQ 原子的 relax_noatom 上限为 4.241 ms（输出错误，只作上限）。已暂停。10-02 r27 挂卡的两个变体常被误记为 w4f，实为 s6 基底的 cluster multicast（A_g74、B_g82） | (c) #75、#84；(d) d.1.1 |
| h-hints（hNN） | operator 写给 op-evolve agent 的规则或指令，放在 job 的 `hint.md`，按 `h<N>` 编号。解析器只读首格为 `h<N>` 的索引表行，所以 h83 因为缺索引行，到 10-02 才生效。**编号按文件独立**，不同文件里的同号 hint 不是同一条（例如 A0 bwd 的 h41 讲 atomic scope，B0 的 h41 讲分块计时）。文件位置：A0 bwd 在 `0923__flydsl/hint.md`，A0 fwd 在 `0925__flydsl/fwd-job/hint.md`，B0 在 `0927__b0/{fwd,bwd}-hint.md`，A0 09-30 之后（h83/h85）在 bwd job 的 `job_context/hint.md`。常被引用的有：h7（只按 prod 排名）、h18（aiter 的 dq 原子过不了确定性门）、h28（beat 单独一个进程）、h40/h41/h66（分块计时）、h44（采用 r13ns）、h46/h72（FlyDSL JIT 缓存 key 漏掉模块常量）、h47（真实数据尺子）、h50（禁止在卡上算 fp32 参考）、h68/h70（r19h、u2n）、h69（proxy 和 prod 各自 ≥ beat）、h83（fast 按 min 计分，gain_weights 1/0.25/0）、h85（新 kernel 先 toy 后 prod，读 VGPR 只用 compile-only，只用镜像 hipBLASLt） | (d) 阅读约定、d.5 |
| op-evolve | 同事开发的自动持续优化算子框架（`/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`）。一个 job 按轮推进，账本在 `artifacts/<job>/rounds/NNN/` 和 `job_context/state.yaml`。普通（fast）轮为 `1-opt`（agent 提出并实现候选）→ `2-reflect`；deep 轮为 `1-profiling` → `2-plan` → `3-act` → `4-reflect`；operator 的 refactor 记在 `0-refactor`。gain 是本轮代码与历史最佳在同一会话重测后的比值，验收看 min_gain（0.007）和 gain_weights；FastError 指某个想法没写进 facts/dead_ends 时整轮被记为 failed | (c) c.1；(d) d.5 |
| champion（冠军） | 一个 job 当前被接受的最佳代码（`state.yaml` 的 best_round，代码在 `op/current`）。注意 state.yaml 按形状记的 champions 指针会记下被否轮次的 best-ever 读数，与 best_round 不一定相同，续跑时会挡住后面的轮次 | (c) c.1；(d) d.5 |

## 附录 C 主要来源索引

路径若无前缀，都相对 `Primus-Turbo/output/`；`OE:` = `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/`；`skill:` = `Primus-Turbo/.claude/skills/gfx1250-attn-campaign/`；`WT:` = worktree `/home/lihuzhan/code/2026_0903__turbo/wt-llama31`（分支 `dev/lhz/llama31_attn_opt`）；`Primus:` = `/home/lihuzhan/code/2026_0828__primus/Primus/`。

**总入口与状态**
- skill:`SKILL.md`，以及 skill:`references/{baselines,bwd-history,fwd,env-and-pitfalls,op-evolve-ops,flydsl-api,kyle-learnings}.md`；`~/.claude/skills/gfx1250-card-safety/SKILL.md`；memory 目录 `~/.claude/projects/-home-lihuzhan-code-2026-0903--turbo-Primus-Turbo/memory/`。
- op-evolve job 目录：
  - A0 bwd：`OE:artifacts/gfx1250-flydsl-attn-bwd-20260917-115934.a0-stale-0930/`（r0–r23）；live 目录 `OE:artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/`（r0–r27，其中 r24–r27 是 A0 09-30 之后的编号）。
  - A0 fwd：`OE:artifacts/gfx1250-flydsl-attn-fwd-20260925-114644/`（r0–r13）。
  - B0 fwd：`OE:artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/`（r0–r20，10-02 从 B0 备份恢复）。
  - B0 bwd（B0 编号 r24–r32）：账本在 `0927__b0/bwd/`；完整目录只在 B0 备份包 `/home/lihuzhan/code/0928__bak_b0.tar.gz` 里，至今没有解包（见 `0928__bak_b0/MANIFEST.md`）。
  - B0 09-14 Triton job：`0914__campaign/op-evolve-artifacts/`。
- 各日总报告：`0913__opt_plan__claude/index.html`、`0914__campaign/report.html`、`0915__opt/PROGRESS-REPORT.html`、`PROGRESS-REPORT-0917.html`、`0922_summary/ASM-ATTENTION.md`、`0927__b0/REPORT-0928.html`、`0928__a0_repro/REPORT.md`、`0930__bwd/REPORT.md`、`0930__roofline/REPORT.md`、`1002__e2e/RESULT-e2e.md`、WT:`docs/gfx1250_llama31_8b_e2e/README.md`。

**(a) 优化前 main 的问题**
- `0913__opt_plan__claude/`：`PROGRESS.md`、`phase0/PHASE0-STATUS.md`、`phase0/PLATFORM-ESCALATION.md`、`phase1/RESULTS.md`、`phase1/BAKEOFF.md`、`phase2/DECISIONS.md`、`phase2/INCIDENT-2026-09-13-wedge.md`、`phase2/ledgers/wq.jsonl`。
- `OE:output/0911__fa_gfx1250_phase1/`：`E1.md`、`E2.md`、`E3.md`、`E4.md`、`HARDWARE-ISSUE.md`、`E1.gemmchk.log`。
- `0914__campaign/{RESULTS,HANDOFF}.md`；`0914__repro__c07/{RESULTS,FINAL-TABLE}.md`；`0915__repro__c07/RESULTS.md`。
- `0915__opt/`：`BLAS-FINDING.md`、`GEMM-NN-FINDING.md`、`GEMM-WGRAD-FINDING.md`、`E2E-AB.md`、`JIRA-TRACE-ANALYSIS.md`、`VENDOR-REPORT-hipblaslt.md`、`BACKEND-STRATEGY.md`。
- `0917__flydsl/{SURVEY,VENDOR-REPORT-aiter-gfx1250,STAGE1-FWD,GQA-WORKAROUND-COST}.md`；`0927__b0/e2e/{RESULT,RESULT-final,RECON,VERIFY}.md`；`0928__a0_repro/REPORT.md`；`0930__port/PR_BODY.md`、`0930__port/runs/bench_turbo_{triton,flydsl}.csv`；WT:`docs/gfx1250_llama31_8b_e2e/README.md`。
- Primus 侧：`Primus:output/0910__fa_opt/attn_bench_gfx1250.log`、`Primus:examples/torchtitan/configs/MI455X/repro_l8b_*.yaml`、`Primus:primus/configs/modules/torchtitan/pre_trainer.yaml`。
- git：`c1325c7e`（#481）、origin/main `8cda13c7`、`f5e1f18b`（#501）、`9f05cd85`（#509）、`eeb9e8d7`（#508）。

**(b) 各后端性能与 ASM 的问题**
- `0913__opt_plan__claude/phase1/BAKEOFF.md`、`phase2/{DECISIONS,T2-ASM-BACKWARD-SPEC,T7-ASM-FORWARD}.md`。
- `0914__repro__c07/{FINAL-TABLE.md,RESULTS.md,flex_ledger.jsonl}`；`0914__campaign/{RESULTS.md,HANDOFF.md,t7/soak_cooled.jsonl}`；`0914__hk_udna1_gate/`。
- `0915__opt/`：`RESULTS.md`、`E2E-AB.md`、`BOTTLENECK-SHIFT.md`、`FLYDSL-AB.md`、`RESULT-32L.md`、`NAN-FINDING.md`、`PERF-CO-CLOSED.md`、`TWOKERNEL-SWEEP.md`、`status.json`；`0915__repro__c07/RESULTS.md`。
- `0917__flydsl/{STAGE1-FWD,VENDOR-REPORT-aiter-gfx1250,GQA-WORKAROUND-COST}.md`；`0922_summary/ASM-ATTENTION.md`；`0924__flydsl/DAY-SUMMARY.md`、`0924__flydsl/bar-census/{fwd_anchor,fwdbwd_anchor}.json`；`0925__flydsl/{AITER-5GEMM-STUDY.md,fwd-isa/REPORT.md,fwd341/ab_prod.log}`。
- `0927__b0/`：`REPORT-0928.html`、`fwd/progress.md`、`bwd/progress.md`、`ruler/REPORT.md`、`ruler/bwd/REPORT.md`、`fwd-nospec/REPORT.md`、`e2e/{BUILD,RESULT,RESULT-final,RECON}.md`、`gemm/REPORT.md`、`profile/REPORT.md`、`OP-EVOLVE-SUGGESTIONS.md`。
- `0928__a0_repro/REPORT.md`；`0930__bwd/{REPORT.md,PROGRESS.md,notes/report_asm.md}`；`0930__roofline/REPORT.md`；`0930__port/{PR_BODY.md,runs/}`；`1002__e2e/{E2E-PLAN.md,RESULT-realab.md,RESULT-e2e.md}`、`1002__e2e/e2e/runs/TABLE.1002_095504.md`；skill:`references/baselines.md`。

**(c) 每一轮的进展**
- `rounds.csv`（= `parts/rounds.csv`）；上面列出的各 op-evolve job 的 `job_context/state.yaml`、`rounds/NNN/{1-opt,3-act}/act.yaml`、`timing.yaml`、`opt.md`。
- `0923__flydsl/{hint.md,STAGE2-FWD-SWEEP.md}`；`0924__flydsl/DAY-SUMMARY.md`；`0925__flydsl/fwd-job/hint.md`；`0927__flydsl/asm-structure/DESIGN.md`。
- `0927__b0/{fwd,bwd}/progress.md`、`0927__b0/fwd/rounds/`、`0927__b0/lab-kdq/REPORT.md`、`0927__b0/lab-bwd-r19/REPORT.md`、`0927__b0/champions/`。
- `0930__bwd/{PROGRESS.md,REPORT.md,armsrc/}`、`0930__bwd/probe/P1-RESULTS.md`；`0930__roofline/REPORT.md`；`1002__oe/{RULER.md,FWDJOB.md,incident/WEDGE-1002.md}`；`1002__e2e/arms_src/bwd_s6_0341`；skill:`references/{bwd-history,fwd}.md`。

**(d) 问题与解决方案**
- 挂卡：`OE:output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md`；`0913__opt_plan__claude/phase2/INCIDENT-2026-09-13-wedge.md`；`0915__opt/{MES-WEDGE,INCIDENT-2026-09-15-machine-death,FLYDSL-WEDGE,DAY-0916-SUMMARY,FWD-SWEEP}.md`；`0922__flydsl/{ROUND3-CLOSEOUT,HANDOVER}.md`；`0923__flydsl/wedge4/ANALYSIS.md`；`0924__flydsl/wedge-rootcause/`；`0927__b0/{LAB-RULES,HANDOFF-A0}.md`；`1002__oe/incident/WEDGE-1002.md`、`OE:artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/rounds/027/{_scratch/arms,_scratch/build_run,1-opt/raw}/`（10-02 挂卡的实际变体与探针）；`~/.claude/skills/gfx1250-card-safety/SKILL.md`。
- GEMM / hipBLASLt：`0915__opt/{BLAS-FINDING,GEMM-NN-FINDING,GEMM-WGRAD-FINDING,PROFILE-POST-NKFIX,NKFIX-NAN-RATE,NAN-FINDING,VENDOR-REPORT-hipblaslt}.md`；`0922__flydsl/gate-patch/README.md`；`0923__flydsl/STAGE2-S0-PROBE.md`；`0924__flydsl/REFCACHE-PREMISE-GONE.md`；`0927__b0/gemm/REPORT.md`、`0927__b0/profile/REPORT.md`、`0927__b0/e2e/{RESULT,VERIFY}.md`；`1002__e2e/E2E-PLAN.md`。
- 尺子、框架、工具链、正确性：`0927__b0/ruler/REPORT.md`、`0927__b0/OP-EVOLVE-SUGGESTIONS.md`、`0927__b0/{fwd,bwd}-hint.md`、`0927__b0/interference.md`；`0925__flydsl/{PARITY-STRATEGY,AITER-5GEMM-STUDY}.md`、`0925__flydsl/fwd-job/{hint,NOTES}.md`；`0923__flydsl/{hint,CAMPAIGN-FINAL}.md`、`0923__flydsl/poison-fix/README.md`；`0924__flydsl/kdq-enumeration/SYNTHESIS.md`；`0917__flydsl/API-DELTA.md`；`0921__flydsl/A0-A1-ZERO-GPU-COMPILE.md`；`0930__port/API_AUDIT_SUMMARY.md`；`0930__bwd/{PLAN.md,oejob/README.md}`；`1002__oe/{RULER,FWDJOB}.md`；skill:`references/{env-and-pitfalls,op-evolve-ops,flydsl-api}.md`。
- 基础设施与协作：`0927__b0/{README,STOPPED}.md`；`0928__bak_b0/MANIFEST.md`；memory `push-after-every-round.md`、`e2e-monitoring-style.md`。

**会话记录**（原始 JSONL 在 `~/.claude/projects/-home-lihuzhan-code-2026-0903--turbo-Primus-Turbo/<id>.jsonl`；去掉工具输出的文本版在 `_work/sessions/`。B0 09-27/28 的会话不在本机）

| 会话 id | 时间范围（UTC） | 主要内容 |
|---|---|---|
| 541e7bc3 | 09-13 02:24 – 09-14 03:03 | 制定自动多轮优化计划；A0 Day-1：main bring-up、bake-off、Triton 调优、e2e 首次跑通 |
| b596bddb | 09-15 01:15 – 09:04 | 把 B0 09-14 的工作备份到 A0；A0 复现 B0 冠军；接入 ASM bwd；09-15 早上整机失联 |
| 11052787 | 09-15 09:05 – 09-17 07:37 | e2e A/B；hipBLASLt 路径与 nkfix；NaN 普查；09-16 多次挂卡 |
| e8ec014a | 09-17 07:52 – 08:00 | FlyDSL attn 开发计划（很短，随后由 c4b79aa6 接续） |
| c4b79aa6 | 09-17 08:01 – 09-21 09:57 | FlyDSL 调研、Stage1、bwd kernel bring-up、op-evolve bwd job 发车 |
| a9a96fef | 09-21 09:58 – 09-25 10:28 | bwd op-evolve r1–r23 无人值守运行；挂卡分类与防护 |
| 03a497e0 | 09-22 00:57 – 00:58 | 一次 deep-research workflow 调用，没有实质内容 |
| 1406f036 | 09-22 12:13 – 09-24 00:49 | 组会汇报用的 0922 总结（`0922_summary/`） |
| 0a07052f | 09-25 10:29 – 09-27 15:25 | flydsl 0.3.4.1；fwd op-evolve job；把经验整理进 skill |
| d2b7f402 | 09-28 11:02 – 12:44 | 在 A0 复现 B0 当天的 fwd/bwd 相对提升和 e2e |
| 2dafe0d2 | 09-29 06:24 – 06:53 | 换驱动后卡在 rocm-smi 里看不到、无法初始化 |
| c28b7a9f | 09-30 02:23 – 07:22 | roofline 分析（多机对比、M8 消融） |
| fdc2534d | 09-30 03:23 – 10-05 05:06 | 产品分支 `dev/lhz/llama31_attn_opt` 移植、API 审计、e2e 教程 |
| 69864fc9 | 09-30 07:37 – 10-05 01:09 | bwd 手工攻关 s1–s6；A0 bwd job r24–r27；10-02 解包真实 dump、处理挂卡 |
