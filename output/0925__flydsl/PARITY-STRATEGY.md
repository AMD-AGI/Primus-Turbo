# gfx1250 FlyDSL attention backward — 第 23 轮作战计划

> 本轮零卡时。以下每个数字要么是我这一轮用 CPU 重新枚举的，要么带 file:line。凡是我没能确定的，都写明了。

---

## 0. 本轮新确立的三件事（先说，因为后面全靠它们）

**(1) aiter 的 BLOCK_KV 是 128，不是 256。** `job_context/profiling/beat/kernel.yaml:112-114` 从 4096 个 workgroup 反推出 "BLOCK_KV = 256 at both" —— 它漏掉了 causal 的 grid 折半：`/home/lihuzhan/code/aiter-src/csrc/cpp_itfs/mha_bwd.cu:715-718` `if((mt==1)||(mt==2)) gdx=(gdx+1)/2`，而 `hsa/gfx1250/fmha_v3_bwd/fmha_bwd_dqdkdv.csv:3` 的 tile 列是 `...,32,128` = ts_qo 32 / ts_kv 128。一个 workgroup 拿一**对** kv block（近对角 + 远对角）做负载均衡，所以 4096 = (8192/128/2)×32×4。这条纠正会同时改掉 lessons 角度的 1.0311 粒度因子和 structural 角度的 727.3–738.5 区间。

**(2) 我亲自数过 aiter 的反汇编，TDM 是真的。** `/tmp/aiter_dis/bwd_hd128_bf16_causal_br_a32_pssk.s`：`v_wmma` 864、`s_barrier_signal` 26、`.long 0xd031` **40**、`buffer_atomic_add_f32` 514、`buffer_load` **0**。structural 角度说"反汇编里根本没有 TDM、.err 是 0 字节所以反汇编干净"是错的（objdump 对不认识的 opcode 吐 `.long` 且不写 stderr）；任务书里那个 40 是对的。

**(3) 多波没有寄存器障碍，这一点树里已经测过。** `hint.md:519-527`（h14）：compile-only 探针只抬 `known_block_size` 和 launch block，VGPR 从 32 线程到 128 线程基本持平 —— 988/984/984、956/952/952、632/636/634，**spill 全 0**，"There is no register blocker"。`dead_ends.md:18-22`（g04）那句 "cannot fit its accumulator floor" 说的是**单波替代方案**，不约束 4 波。而且在 BLOCK_KV=128 / 4 波下，每个 wave 仍只拿 32 个 kv，累加器还是 32×128×fp32×2/32 = **256 VGPR，和今天完全一样**，4 个波买到的寄存器解脱确实是 0，但代价也是 0。

---

## 1. 第 13–22 轮真正教了什么（可执行的规则）

**诚实的那条先说：22 轮里，能够关掉 39% 的 arm 数量是 0。**

我按 `rounds/0{13..22}/*-opt/act.yaml` 清点：13 轮起共 31 个 arm 条目，其中消耗 prod 卡时的候选 arm 16 个（g42 g43 g44 g45 g47 g49 g51 g52 g56 g59 g60 g61 g62 g63 g66 g68）。按它们自己 `expected:` 块里写下的天花板分类：

| 类别 | 数量 | 说明 |
|---|---|---|
| 天花板**恰好为 0**（构造使然） | 4 | g42/g44/g45 在 prod 走 `nsp=1` 的未改路径；g52 的 prod 内核被要求逐字节相同 |
| 预注册为**预期亏损**的对照 | 2 | g66、g68 |
| 天花板 ≤ 5% | 5 | g47 g49 g51 g59 g62 |
| 天花板 > 5% 但建立在已退役工具上 | 4 | g56(~17%，静态 stall 模型)、g61(+8.7%，语料数)、g43(+2~7%)、g60(设计的证否) |
| **攻击结构因子的** | **0** | — |

也就是说：**十轮、十六个 arm，没有一个的天花板量级对得上 39%。** 净结果 +2%（`hint.md:2972` 的 505.85 / 507.13 / 516.87）正是这个分布的期望值。

### 规则 1 —— arm 必须先申报它攻击的是差距分解里的哪个因子，以及那个因子的总大小
差距是 **1.386× 结构 × 1.00–1.01× 其它**（§2）。一个落在 1.01× 那个因子里的 arm，天花板就是 +1%，无论它做得多完美。按天花板排序，不按机制可信度排序，且天花板必须是算术而不是叙事。这条规则回溯适用会**禁掉第 15 轮以后每一个上卡的 arm**。

### 规则 2 —— 静态 ISA 指标不是排序信号，五次实测为证
g49 发射槽 678→640、full drain 2→1、全部改善，**-16.25%**；g56 删掉了被判定占 562 个 stall cycle 中 470 个的那条指令、每一个静态数都变好，**-6.84%**（`findings/facts.md:224-227`）；g60 VGPR -88、`s_set_vgpr_msb` -57、`v_nop` -58、spill 0，**-17.27%**；g53 一个 634 指令的 body 比 675 指令的慢 13%。第 19 轮还把模型升级了（`attrib_ss5.py`），升级后它把 g56 判成"快 20%"，而 g56 已经实测慢 6.84%（`dead_ends.md:280-292`）。**只能当 build gate 用。**

### 规则 3 —— 数字是产生它的那个内核和 harness 的属性，不是机制的属性
这个 campaign 已经因此翻车五次：g61（语料的 "+8.7% on a dK/dV body" → NULL）、g60（gfx950 的 `v_accvgpr` 类比 → -17.27%）、g43（per-XCD L2 → 这颗芯片只有一个 device-wide L2 → NULL）、g63（跨内核搬 prefetch → -19.33%）、以及 **1.57% 地板本身**：`hint.md:2963` "The 1.57% floor is a GEMM's floor, not this operator's"，源头是 HipKittens bf16 GEMM ladder @8192³。**本 operator 自己的同 session 地板是 0.40–0.66%**（`hint.md:2971-2975`）。以后所有 delta 用 0.40–0.66% 判，不要再用 1.57%。

> 这轮的五个角度里，有三个又犯了这条：一个用 1.57% 判 g62，一个把 h30 为 k_dq（KV_STEP=32）算的 32768 B LDS 搬给 BLOCK_KV=128 的 k_dkdv（差 4 倍），一个把 aiter 的 dK/dV host reduce 的 0.52 ms 当成它的 dQ 开销（真值是 `beat/kernel.yaml` gqa-reduce 111269 ns×2 = 0.2225 ms = 2.88%）。

### 规则 4 —— 零卡闸门是这个 campaign 收益率最高的做法，要加强不是削弱
g46、g48、g53、g65、g67-form1、g54、g55 七个候选在 15–22 轮零成本死掉。但**闸门要打在会失败的量上**：g14 是"982 VGPR、spill 0"通过了"spill 0 且 vgpr ≤ 1024"这个闸门，然后实测慢 2.76 倍（`dead_ends.md:30-33`）。spill 闸门只能证否"装不下"，不能证明"会快"。

### 规则 5 —— 减法探针（P1/P2/P3/P70 这类）极其便宜，每轮至少跑一个
round 20 的三个探针 20 分钟卡时就杀掉了活了一整轮的 LDS-port 假说；P70 从上方 bound 了整条"收紧 k_dq"的轴。round 21 的 reflect 把"没跑探针"列为它唯一的 process error。

### 规则 6 —— 预期值要在测量之前落盘并 commit
`hint.md` h39 §4：round 22 那次被表扬的预注册，`act.yaml` 的 mtime 14:23:51 晚于评分测量 14:06:47，所以是自述而非可证。一行成本：`expected` 写单独文件，跑之前 commit。

---

## 2. 39% 在哪里（分解，逐项标注测量 / 推断）

### 2.1 结构因子：1.386×，**枚举**，我这一轮从源码重算过

我用 CPU 枚举 bottom-right causal 的活 tile：

| 量 | 值 | 来源 |
|---|---|---|
| 活的 (32q,32kv) tile | 32,896 | 枚举；`kernels.py:544-546` 的 `_qsf` 界 |
| 活的 (64q,32kv) tile | 16,512 | 枚举；`kernels.py:937` 的 `_lim` 界 |
| 我方 issued WMMA / (b,hq) | 32896×64 + 16512×96 = **3,690,496** | `kernels.py:343` "64 WMMA of 643"、`:789` "96 WMMA out of 782"，ISA 印证 128 / 192（两个 body） |
| 算法 WMMA 当量 | 5×33,558,528×128/8192 = **2,621,760** | `op-evolve/tools/op_flops.py:163-168`；FLOP 5.498229e12 = `_final.yaml:186` |
| **我方 issued/algorithmic** | **1.407641** | |
| 活的 (32q,128kv) tile | 8,320 | 枚举 |
| aiter issued WMMA / (b,hq) | 8320×320 = 2,662,400 | 每 stage 每 wave 80 WMMA × 4 wave = 320 = 5×524288/8192 |
| **aiter issued/algorithmic** | **1.015501** | |
| **结构因子** | 1.407641 / 1.015501 = **1.38617** | |

7-vs-5 的读数 **成立**，并且现在两边都是枚举出来的（我方从我们自己的 ISA，aiter 从它自己的 `.co` 反汇编）。多出来的 2 个 GEMM 就是 k_dq 里重算的 S=QK^T 和 dP=dO·V^T。

### 2.2 调度残差：1.000–1.010×，**已经没了**

把两边放进**同一个归一化**（issued 矩阵 FLOP ÷ 墙钟时间）：

| | issued 矩阵速率 |
|---|--:|
| 我方 @ 冠军 511.42 | 511.42 × 1.407641 = **719.9 TF/s** |
| 我方 @ r22 sweep 516.87 | **727.6 TF/s** |
| **aiter @ 716 TF/s** | 716 × 1.015501 = **727.1 TF/s** |

**残差 = 1.010（对冠军）/ 0.999（对 r22 再测）。** 差距 = 1.386 × 1.00~1.01。

这条裁定了角度之间最大的分歧：
- fusion-multiwave 角度的 headline "我们比 bar 快 1.6%" —— **错**，它拿我方的 *issued* 比 aiter 的 *algorithmic*（苹果比橘子）。
- 它的 lens-1 反驳"bar 领先 6.4%" —— 也错，用的是 round-17 的 11.32 ms。
- 正确答案是 **两边在 1% 以内，落在本 operator 0.40–0.66% 地板的 1.5–2.5 倍**，实质持平。
- 语料里 `route.md:816-817` 的 "1.408 × 1.040 = 1.464" 已经**作废**：那个 1.040 是对 round-17 的 494 TF/s 冠军算的，18–22 轮的 +2% 正好把它花光了。

**推论：`findings/facts.md:219-227` 的 "如果 k_dkdv 达到 k_dq 的 per-element 效率，prod 就是 ~7.29 ms ≈ 754 TF/s，不用碰 fusion 问题" 必须撤回。** 754 TF/s 意味着 issued 矩阵速率 1061 TF/s —— 比同一颗芯片上的手写 ASM 快 46%。五轮定向攻击那个被点名的 470-cycle 站点净收益 +1.65%，而唯一直接打它的 g56 输了 6.84%。这个数字不是"还没拿到的 headroom"，它从来不存在。

（**测量 / 推断**：1.407641 和 1.015501 是枚举；719.9/727.1 里的时间是测量，但 511.42 和 716 来自不同 session，带约 ±1% 的 session 间松动；k_dkdv/k_dq 的 62.93%/32.13% 拆分只有一份 profile，`rounds/017/1-profiling/kernel.yaml:40,:56`，且那份 profile 记的 k_dkdv vgpr 是 368 而冠军是 904 —— 是**推断**，不是测量。）

### 2.3 功耗 / 时钟：**不是差距项**
`hint.md` h39 §1 的 264 样本 trace 是真的（185/264 ≤ 1030 MHz，对空载 1100）。但 bar 是同 run 测的（`_final.yaml` `beat_measured_same_run: true`），droop 同时作用于两边，比值不变。它改绝对 TF/s，不改 operator 目标所用的那个比值。**不要为它花一轮。**

### 2.4 剩下的 1% 从哪来
矩阵内核对矩阵内核：我方 7.7392e12 issued FLOP / 10.2198 ms = **757.3 TF/s**，aiter pssk 5.5833e12 / 7.2013 ms = **775.3 TF/s** —— bar 领先 2.4%。这 2.4% 是因为我们的非 WMMA 指令流是它的 1.7 倍：k_dkdv 每 body 128 条 `v_mov_b64_e32`（FlyDSL 非旋转 carried register 的产物，`dead_ends.md:93-96` 的 g37），aiter 在 8 个 stage 里 **0** 条。**这 2.4% 不值一轮。**

---

## 3. 排序后的计划

> **硬规则：只要 39% 的差距还开着，天花板 < 5% 的 arm 不许占卡时槽位，除非它是免费的。** 这条写进 round 23 的 act.yaml。

### 先算出"追平"到底要求什么 —— 平价账本

| 项 | ms | 来源 |
|---|--:|---|
| bar 总时间 | 7.679 | 5.498229e12 / 716e12 |
| 我方非矩阵内核（k_delta 等） | −0.531 | 100−95.06% × 10.751（`rounds/017/1-profiling/kernel.yaml:40,:56`） |
| dq_acc 清零 + dq convert | −0.18 | 按 aiter 同类的 1.22%+1.18%（`beat/kernel.yaml:180,:191`） |
| **⇒ 融合主内核预算** | **6.97** | |
| 今天的 k_dkdv | 6.766 | 62.93% × 10.751 |

融合后的 body 每个 (32q,32kv) tile 从 64 WMMA 变成 80（dQ += dS·K 是 16 条，dS 已经在寄存器里，`kernels.py:445-452`；K 在 k_dkdv 里是循环不变量）。

> ### **平价条件，一句话：融合后的 k_dkdv 必须用 ≤ +3.0% 的时间吃下 +25% 的 WMMA。**

这是整个 campaign 剩下的全部内容。它是可测的（见 §5），而且**从来没有人测过** —— `hint.md:2192-2194`（h28 假设 1）到今天还挂着 "Issue efficiency is assumed unchanged after fusing. **Not measured.**"

### 路线 R0 —— determinism gate 的裁决（operator 决定，零成本，**先做**）

**这是唯一的 gate，它决定其余所有路线是否存在。**

我这一轮把确定性账算完了，两个朝向都算了，结论是**枚举级的**：

| 形态 | dQ 或 dK/dV partial 流量 | 时间 @ 3.02 TB/s | 对比 GEMM 节省 2.93 ms |
|---|--:|--:|---|
| KV-outer, BLOCK_KV=32, 确定性（写+读） | 138 GB | 45.7 ms | 死（= `dead_ends.md:156-166` 的 g54） |
| KV-outer, BLOCK_KV=128, 确定性 | 34.9 GB | 11.6 ms | 死 |
| KV-outer, BLOCK_KV=256, 确定性 | 17.7 GB | 5.9 ms | 死 |
| KV-outer, BLOCK_KV=512, 确定性 | 8.9 GB | 2.9 ms | 勉强打平 —— 但累加器要 1024 VGPR/wave，装不下 |
| **Q-outer, BLOCK_Q=256, 确定性**（dK/dV 进 workspace，放弃 GQA 寄存器归约） | 35.4 GB | 11.7 ms | 死 |
| **KV-outer, BLOCK_KV=128, 原子**（= aiter 的形态） | 17.45 GB 单程 | 被主内核吸收 | **可行，bar 就是活证明** |

3.02 TB/s 是本树里这张卡上唯一一次真正的流式归约测量（`beat/kernel.yaml:119-128`，0.671 GB / 2×111269 ns）。用 1.19 TB/s（k_redsp）更死，用 4.39 TB/s（乐观 roof）仍然死。

**⇒ 不存在任何可建造的确定性融合几何。追平必须放宽 determinism gate。**

两份文本不一致，这件事必须摆到 operator 面前：
- `history/v000_original.yaml:94-97`（= `_final.yaml:131-134`，**用户原话**）："every output element is written exactly once. Keep that ... 200-run bitwise determinism is a cheap gate on it."
- `_final.yaml:135-138`（`[resolved]` 加的）："no atomics on any output, and dq/dk/dv bitwise identical across 200 consecutive runs. **A round that introduces a split-k or atomic reduction fails this gate even if it is faster.**"
- `op/validation.py:199+` 的 `check_determinism` 只跑 200 次 bitwise 比对，没有任何 atomic / split-k 检测器。
- 而 **split-k 每一轮都在出货并通过**：`op/current/impl.py:166-168` 的 `while _wgs*nsp < 2048 and nsp < 16`，在 gate 所在的 `fast` shape 上推出 **nsp=16**，走 `k_dkdv_sp` + `k_redsp`（固定顺序归约，逐位可复现）。

**该问 operator 的问题（precise form）：** 禁令针对的是 *split-k 这个类别*（那么它自 round 13 起每轮都在违规，而 gate 看不见），还是 *非确定性*（那么固定顺序 split-k 合法、fp32 原子不合法，且平价不可达）？**如果答案是后者，诚实的回答是：在当前契约下这个内核追不上 aiter，天花板是 ~517 TF/s = 0.72×。**

成本：0。**在 operator 回答之前不要开 round 23 的任何 build。**

---

### 路线 R1 —— dQ GEMM 成本探针（见 §5，**唯一要花卡时的**）
价值：把整条链上唯一没被测量过的假设变成一个数。成本：一次 benchmark 槽位。**不需要 gate 裁决、不需要原子、不需要 barrier、不需要多波。**

---

### 路线 R2 —— 4 波 KV-outer 融合（gate 开了才走）

**形态**：block=(128,1,1)，BLOCK_KV 32→128（4 波 × 32 kv/wave，每波累加器仍是 256 VGPR），Q/dO staging 四波共享（今天每波独吞 70656 B，之后是一份），dQ 用 `buffer_atomic_add_f32` 打进 fp32 `dq_acc` + 一个 convert 内核。**这逐字就是 aiter 的形态。**

**第一步（零卡时，4 次 COMPILE_ONLY）**：
1. 把 `kernels.py:453` 和 `:492` 的两个 `fx.barrier()` 加回来（h30 引的 `:816, :838` 是 round-17 的陈旧行号，今天那四处注释在 **453 / 492 / 855 / 877**），每个前面配显式 `rocdl.s_waitcnt(WAIT_LGKM)`，**保持 block=(32,1,1)**，diff ISA。
2. 加 h13 第 2 步的第三道 barrier（共享 Q/dO staging store 与 GEMM1 之间）。
3. 翻 `known_block_size` 和 launch block 到 [128,1,1]，BLOCK_KV=128。
4. 加 dQ GEMM，dQ 按 16-query 半块累加（64 VGPR）而不是整块（128 VGPR）。

**闸门（按顺序，任何一条不过就停）**：
- `.vgpr_spill_count == 0` 且 `.vgpr_count ≤ 1024`
- `.group_segment_fixed_size ≤ 81920`（≥ 4×70656 说明 Q/dO 被复制了四份，共享根本没发生）
- loop body 里 `buffer_load_b128 == 8`（是 32 就是零共享）
- 每个 body `s_barrier` 恰好 3（在 block=(32,1,1) 下这个检查无意义，ISA 读出来永远是 0）
- body 里 `v_wmma == 80`（不是 64，不是 96）

**杀手风险，以及它的解药**：barrier 会让后端发一条保守的全计数器 `s_wait_loadcnt_dscnt 0x0`，把 g21/g62 押在 LOADcnt 上的 prefetch cover 从 ~603 条砍到接近 0（`pool.md:29-35` 的 standing model；g68 把 cover 砍到 41 输 30.32%，g66 输 21.93%，两者都是等寄存器纯重排）。

**但 aiter 同时有 26 对 barrier 和巨大的 cover。** 原因在 §0(2)：它的 tile 流量**一条 `buffer_load` 都没有**，40 条 TDM 直接落 LDS、退休在独立的 TENSORcnt 上，barrier 的全计数器 wait 排不干它。**barrier 本身不致命；barrier 加上"prefetch 住在 VGPR 里、挂在 LOADcnt 上"才致命。** 这解释了 structural 角度和 lessons 角度在 barrier 上的对立 —— 两边各对一半。

**⇒ 致命风险有确定的解药，就是 R3。**

**轮次预算**：1 轮零卡（4 次编译）+ 1 轮上卡。

---

### 路线 R3 —— 把 tile 载入改成 TDM（`tensor_load_to_lds`）

**不是独立的赢法，是 R2 的前置。** 它同时：把 cover depth 从 VGPR 解耦（aiter：~1339 条 cover、0 VGPR 代价、循环内**从不**完整排空）；释放 32 条 `buffer_load_b128` 的目的寄存器 + g62 的 188 VGPR 二级 cover；删掉每 body 128 条 `v_mov_b64_e32`；让 4 条 LSE/delta `buffer_load_b32` 拿到自己的计数器（这才是 g56 该做的那一半 —— 它挪错了另一半，输了 6.84%）。

**可行性已证**：FlyDSL 0.3.2 有 `flydsl/expr/rocdl/tdm_ops.py`、`cluster_load_async_to_lds`、`s_wait_tensorcnt`；树内的 gfx1250 **forward** 内核已经在用（`~/.claude/skills/flydsl-gfx1250/SKILL.md:162-175`）。

**绝对不要单波单独建它**：单波 workgroup 里 TDM 就是一个 wave 把自己的私有 tile 绕 LDS 走一圈，正是 g09 删掉的那个 round trip。**只在 R2 的 4 波形态里建。**

**硬风险**：`SKILL.md:175` 记录的 —— gather 不做 `addr64` carry-safe 更新会在大 tensor 上**硬挂**。挂一次 = 一个 AC 电源循环。按 `gfx1250-card-safety` 的规程排，不要在无人值守窗口首跑。

---

### 不要走的路线（已定价为负，记进 dead_ends 免得下轮重开）

| | 为什么 |
|---|---|
| 任何确定性 dQ/dK partial workspace | §R0 的表，两个朝向、四个 BLOCK 尺寸全负，装得下的付不起、付得起的装不下 |
| 用 bf16 workspace 传 dS 代替在 k_dq 重算 | 4.2955e9 个 dS 元素 × 2 B × 2（写+读）= 17.18 GB；@3.02 TB/s = 5.7 ms 对 3.07 ms 的节省。负。任何跨内核物化 O(Sq×Skv) 中间量都是这个结论 |
| 8 波 workgroup | 8 个 wave32 落在 4 个 SIMD = 2 wave/SIMD ⇒ 每波上限 512 VGPR；我们在 904/960。h29 表里唯一为正的那一行不可建 |
| 单波融合 | 每波 dQ 瞬态 128 VGPR + 904 = 1032 > 1024，且 BLOCK_KV=32 的 dQ 流量是 aiter 的 8 倍（69 GB vs 17.45）。structural 角度把它叫成 "aiter's exact shape" 是把两条路的存在性证明张冠李戴了 —— aiter 是 4 波 / 26 barrier / 320 KiB LDS |
| dK/dV 累加器常驻 LDS | `rocdl.wmma_f32_16x16x32_bf16` 的 C/D operand 是 VGPR（`kernels.py:486-491`）。"常驻 LDS" = 每条输出 WMMA 前后各走一次 LDS，body 里 32 条输出 WMMA × 8 dword × 2 = 512 条额外 LDS dword-op，加在今天只有 80 条 LDS op 的 body 上。而且在 KV-outer/4 波下根本不需要：每波仍只拿 32 kv，累加器还是 256 VGPR。**h30 点名的那条路是伪路** |

---

## 4. 停止清单

**已被测量关闭的轴，不要再开：**

| 轴 | 证据 |
|---|---|
| compute / matrix ILP | g61 加 4 路矩阵 ILP = NULL（+0.27% 对 0.29% 地板）；`dead_ends.md:294-309` |
| 发射上限 | g60 删 115 个发射槽 + 88 VGPR、spill 0，**-17.27%** |
| LDS 端口压力 | round 20 的 P3 删光全部 80 条 LDS op、WMMA 不变，**-8.19%**。那些 LDS op 是 cover 不是成本 |
| Q/dO 的 LDS round trip | P1 **-6.98%** |
| occupancy / BLOCK_KV（单波） | g55 的 census（395 live-in 里 384 是 256-VGPR 累加器 + 128-VGPR prefetch tuple）+ g14（982 VGPR spill 0，慢 2.76 倍） |
| prefetch 深度与位置，两个内核两个方向 | g62 +1.65%、g63 -19.33%、g66 -21.93%、g68 -30.0%、P70 -10.86% |
| `sched_barrier` / `sched_group_barrier` | 上卡 0 胜 5 负；且 `sched_barrier(0)` 是**边界**不是**夹子**，围住 load group 反而摊得更开（`dead_ends.md:437-448`） |
| WMMA operand reuse hint | g59 编码正确（56/128 `matrix_a_reuse`）、算术逐位相同、买到 +0.08% |
| XCD locality | g43 NULL，这颗芯片只有一个 device-wide L2 |
| 源码重排 | g22 ISA 逐字节相同，g65 指令级相同 |
| 静态 ISA 指标作为排序信号 | 四次错判（规则 2） |
| 功耗 / 时钟 | 是真的，但同 run 测量下对两边同时作用，不是差距项 |

**不要再造的 arm 家族：** 纯重排（g66/g68 把这条轴的符号钉死了）、删除类（本内核 6 战 6 负，那些指令在做 cover）、手写 graded wait（g46 发现编译器已经在发降序 partial wait，而剩下那条 `s_wait_loadcnt 0x0` 的消费者是 128 条 `v_mov_b64` 把每个 prefetch 目的寄存器都拷一遍，所以 0x0 是**精确**的不是保守的）、以及任何以 `attrib*.py` 输出为主要依据的 arm。

**还要停掉一个习惯**：不要再引 1.57% 地板（§规则 3）。

---

## 5. 那一个实验

> **dQ-GEMM 成本探针。要花卡时，但只要一个 benchmark 槽位，且不需要 gate 裁决、不需要原子、不需要 barrier、不需要多波。**

**做法**：在今天的单波 k_dkdv 里，给每个 (32q,32kv) tile 加上 dQ += dS·K 的 **16 条 `v_wmma`**（dS 在 `kernels.py:445-452` 已经在寄存器里，K 在这个内核里是循环不变量，prologue 里 stage 并转置一次），累加进 16-query 半块（64 VGPR），循环外存进一个 dummy 缓冲以防 DCE。**不接 dQ 的真实输出，不改 k_dq，不改正确性路径。** 然后同 session palindromic A/B 测 k_dkdv 的时间。

**它测的是什么**：整条平价链上唯一一个从 h28 到今天一直标着 "not measured" 的量 —— **融合后的 body 能不能吃下 +25% 的 WMMA 而不按比例付时间。**

**预注册的判读**（跑之前写进文件并 commit）：

| k_dkdv 时间变化 | 含义 |
|---|---|
| **≤ +3.0%** | 平价条件满足。融合 = 703–719 TF/s = 0.98–1.00× bar。全力走 R2+R3 |
| +3.0% ~ +7.6% | 融合落在 0.90–0.98×。仍是唯一量级对的路线，但单靠它追不平，必须叠加 4 波的 Q/dO staging 摊销 |
| **> +7.6%** | 加的 WMMA 一点都没被现有的 stall cycle 吸收。**融合达不到平价，campaign 应当在 0.72× 停下并如实上报** |

+7.6% 是"纯发射、零吸收"的上界：16 条 WMMA × 8 cycle = 128 cycle / 1685 cycle 的 body 预算（`facts.md:224`，该预算出自已退役的模型，所以只当刻度不当结论）。

**为什么是它而不是别的**：R2 的 COMPILE_ONLY 闸门只能证否"装不下"，而 h14 已经说了装得下 —— 那些闸门**已知会通过**，不产生新信息（这是 lessons 角度的路线 A 的致命缺陷）。真正未知的是速率，而速率只有卡能回答。这个探针用最小的代价、在不触碰 determinism gate 的前提下、把那个未知变成一个数。

**卡安全**：不涉及 TDM、不涉及 barrier、不涉及 4 波，没有挂卡路径。可以在无人值守窗口跑。

---

## 6. 诚实的赔率

**能不能到 716 TF/s？—— 取决于一个非技术的裁决，和一个从未被测过的数。**

**若 determinism gate 维持"禁原子"：概率 < 5%。** 天花板是把 1.01× 的残差吃干净 = 516.6 TF/s = **0.72× bar**。所有确定性融合几何都被 §R0 的表枚举掉了。这种情况下 operator 该知道的是：**这个目标在当前契约下不可达**，而契约是他自己可以改的。

**若 gate 放开到"固定顺序归约 + fp32 原子累加 dQ"（= aiter 的做法）：**

| 结果 | 概率 | 需要为真的条件 |
|---|--:|---|
| ≥ 1.00× bar（716+） | **20–25%** | 融合 body +25% WMMA 只付 ≤+3.0%；4 波 Q/dO 共享再省下一点；TDM 让 barrier 不砍 cover |
| ≥ 0.95× bar（~680） | **55–65%** | 上面前两条中的一条成立 |
| ≥ 0.90× bar（~645） | **~80%** | 仅结构因子落地，issue 效率打八折 |
| 回到今天（0.72×）或更差 | ~15% | barrier + 原子争用吃掉全部结构收益，或 TDM 挂卡吃掉轮次 |

**必须为真的三件事，按不确定性排序：**
1. **融合 body 的 issue 效率。** 唯一未测量项，§5 的探针直接测它。有一个正面的先验：round 20 的 P1/P3 和 g60 都显示这个 body **删活反而更慢** —— 它有发射气泡，而新加的 16 条 WMMA 不需要任何新的数据搬运（dS 在寄存器、K 是循环不变量），所以它们是能填气泡的那种活。这是论证，不是测量，而这个 campaign 在这类论证上错过四次。
2. **barrier 不砍 cover。** 有确定的机制解药（TDM），有卡上的存在性证明（aiter：26 对 barrier + ~1339 条 cover + 716 TF/s），但在我们的代码上没人量过 barrier 今天把 cover 砍到多少 —— 那是一次 CPU 编译 + 一次 `chain.py` 读数，零卡时，应该和 §5 并行做。
3. **原子 dQ 的争用可承受。** aiter 用 514 条 `buffer_atomic_add_f32` 付同样的 17.45 GB 并把它吸收在 7.2 ms 的主内核里；这张卡上 fp32 device-scope 原子的实际吞吐**全树没有一条记录**。如果 R2 的 BLOCK_KV 不是 128 而退回 32，流量变成 69 GB、是 aiter 的 4 倍，那一项会独自杀死路线。**所以 BLOCK_KV=128 不是调优参数，是可行性前提。**

**一句话给 operator**：过去五轮的 +2% 不是运气不好，是那 1.01× 的因子本来就只有 1% 可拿，而十六个 arm 里没有一个瞄准过那 1.386×。现在只剩两个动作有意义 —— **回答 determinism gate 的问题**，和**跑 §5 的探针**。前者零成本，后者一个槽位。两个都做完之前，不要再花任何卡时。