# 0913–0917 回顾：当前最优版本是什么，以及下一个后端选哪个

日期 2026-09-17 · 全部结论标注证据来源 · **未使用 GPU**

---

# 问题一：当前最优版本是不是基于汇编的？

## 1.1 是的，但要看清它由什么构成

单层（`b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`）：

| 档位 | fwd | bwd | total | 这一级的来源 |
|---|--:|--:|--:|---|
| turbo 出厂 `1cb2e183` | 10.651 | 45.134 | 55.785 | — |
| 强制 Triton 前向 + 融合反向 | 4.166 | 17.835 | 22.001 | **我们的派发层改动** |
| ASM 前向 + 融合反向 | 1.561 | 17.686 | 19.239 | **接通 aiter 预编译前向** |
| **ASM 前向 + ASM 反向（冠军）** | **1.572** | **10.160** | **11.726** | **接通 aiter 预编译反向** |

**注意口径**：出厂那行 n=3、事故前、且来自另一个 worktree（`wt-bakeoff`），
后两行是 AC-cycle 后 n=5 —— 见 `LEDGER-AUDIT.md` §3，这个 4.757× 不是严格同会话。

## 1.2 这段 ASM 是谁的、什么时候的

**是 aiter 的预编译资产，不是我们写的。** 位置 `aiter-src/hsa/gfx1250/fmha_v3_bwd/`，
六个 ELF 对象：

```
bwd_hd128_bf16_a32_pssk.co              56168 B
bwd_hd128_bf16_a32_pssk_perf.co         53872 B   ← 不在选择表里
bwd_hd128_bf16_causal_br_a32_pssk.co    61688 B   ← 我们用的
bwd_hd128_bf16_causal_br_a32_pssk_perf.co 59392 B ← 不在选择表里
bwd_hd128_dq_convert_bf16.co             6256 B
bwd_hd128_odo_bf16.co                   10840 B
```

**开发时间：本地证据答不了。** `aiter-src` 是 **shallow clone（`git rev-list --count HEAD` = 1）**，
所有文件的"首次出现"都指向那一个克隆提交（2026-09-13）；`.co` 的 ELF note 里只有
`Linker: AMD LLD 23.0.0`，没有构建日期。**要准确回答需要去 aiter 的 GitHub 查
`hsa/gfx1250/fmha_v3_bwd` 的提交历史。**

有一个间接但有力的旁证说明**它很新、很不成熟**：同目录下 gfx950 有 **26+ 个变体**
（a16/a32 × causal/br × psskddv × group × swa 三种舍入 × fp16/bf16 × dq_shuffle/dq_convert），
**gfx1250 只有 6 个**，且**只有 a32、只有 pssk（没有 `ddv`）**。
MI355 在 trace 里实际跑的 `bwd_hd128_bf16_causal_br_a16_psskddv` 在 gfx1250 上**根本不存在**。

> **撤回**：`JIRA-TRACE-ANALYSIS.md` 第四节把"试试 a16"列为线索。**不成立** ——
> a16 是 gfx950 的资产，gfx1250 侧没有。该节已需修正。

## 1.3 我们是不是只是调用了？不是。但我们也确实没碰汇编本身

**做了的三件事，都不是"调用"：**

**（a）根本没有可调用的入口。** aiter 的 C++ host 在五处特判 gfx1250，**Python wrapper 没有**
（`can_impl_fmha_v3_bwd` 只认 gfx942/gfx950）。所以我们从 ELF metadata 读出 kernarg 布局，
自己写了发射器：`asm_bwd_abi.py`(136) + `asm_bwd_launcher.py`(185) +
`_asm_bwd_kernargs.py` + `attention_asm_bwd_impl.py`(353)，约 **674 行**。
两种 packing 约定共存（dqdkdv 每字段 16 B 对齐、odo 紧凑打包），是这里的主要坑。

**（b）我们在这个现成件里找到并绕过了一个正确性 bug。** 这是最实质的一项：

> 内核按 **q head** 索引写进 **kv 尺寸**的缓冲区 —— **越界写**。
> GQA ratio=4 下 `ratio` 个 workgroup 争同一块 dk/dv tile 且无同步。

表现取决于越界落在哪里：s=1024 落进别的分配 → **dk/dv 静默损坏到约 −0.3 dB 而 dq 仍然正确**；
s=256 跨页 → 进程 fault（dmesg 零记录，不伤卡）。
我们的定位过程记录在 `RESULTS.md`：三个实验（ratio=1 全对 / ratio=4 按 kv head 塌掉 /
ratio=4 按 q head + host 规约 全对）钉死了它，并排除了"累加 vs 赋值"假设。

**换句话说：aiter 发布的这个 gfx1250 ASM 反向，在 GQA 下直接算错 —— 而 llama3.1-8B 正是 GQA=4。**
不做这一步，它对我们的工作负载不可用。

**（c）资格门与两个每调用开销。** 资格门判据是 **seqlen ≥ 2048**（不是并行度 —— 这条我错过两次，
第一次从单个样本外推 `b*hq ≥ 32`，第二次差点整个拿掉而那组数据是被污染的）；
修掉了每次反向重新 `hipModuleLoad` 三个 `.co`（**每步 96 次加载**）；加了形状级 scratch 缓存
（进程级共 1.000 GiB，不随层数增长 —— 这一条也纠正过一次 45× 的错误外推）。

**没做的**：**我们一行汇编都没有改过。** 内核的 VGPR 用满 **1024**、LDS **320 KiB**、
workgroup 1024、wave32 —— 资源已经饱和。

## 1.4 这个后端还有多少余地

按可信度排序：

**（i）`_perf` 变体 —— 已测，不是 drop-in，但线索明确。**
两个 `_perf.co` **不在 aiter 的选择表 `fmha_bwd_dqdkdv.csv` 里**（该表只有 2 行，都是非 perf），
ELF 元数据与非 perf **完全相同**（同符号名、同 1024 VGPR、同 320 KiB LDS、同 704 B kernarg），
只有代码体积差 2296 B。我们在 gqa2k 上测过：**dk/dv 与 shipped 版逐位相同，但 dq 只有 5.84 dB**
（不是垃圾，是"部分正确"，符合贡献缺失）——**大概率 dq pass 需要不同的 grid 或 split**。
**代价**：几小时的 bring-up；**收益**：未知，但既然 AMD 把它命名为 `_perf` 且单独发布，值得一试。

**（ii）去掉 GQA 绕法的代价。** 我们现在按 q head 分配 dk/dv（4×）再 host 规约。
gfx950 的 `psskddv` 变体不需要这一步。**这部分开销目前没有单独计量过** ——
应该量一下，它可能是对 MI355 那 1.6× 差距的一大块。

**（iii）向 aiter 团队报这个 GQA bug，并要一个 gfx1250 的 `psskddv` / a16 变体。**
**这是杠杆最高的一项，成本几乎为零**（我们已经有完整的复现与三实验定位）。
gfx1250 的资产集只有 gfx950 的 1/4，这不是我们能靠调用补上的。

**（iv）自己改汇编。** 手写 gfx1250 汇编、在 VGPR 已满的内核上做调度 ——
**技能门槛最高、杠杆最低**，且与 (iii) 重复。**不建议。**

**一个必须先修的测量问题**：四个测量入口（`tune_attention.py:529`、`t2_bringup.py:224/266`、
`perfdq_diag.py:18`）调 `asm_backward` 时**都没传 `hip=` 和 `scratch=`**，
即它们测的是**没打补丁的路径**；只有产品调用点两个都传。
独立 bring-up 测反向 **8.68 ms**，harness 里是 10.098 —— 差的 1.4 ms 是 autograd 管路
（ctx 存取、`do.contiguous()`）。对外引用 harness 的数是对的（该算进去），
**但与 JIRA trace 比较时要注意：trace 是纯 kernel 时间，不含 autograd 管路。**
同口径比应是 **~8.7 vs MI355 的 5.381 = 1.62×**，而不是 `JIRA-TRACE-ANALYSIS.md` 里写的 1.89×。

---

# 问题二：下一个后端选哪个 —— Triton 还是 FlyDSL？

## 2.1 Triton：配置面已经关闭，而且结构上到不了我们现在的位置

三条 Triton 路径今天/前几天全部扫过：

| 路径 | 结论 | 证据 |
|---|---|---|
| vendored 融合反向 | 九个轴全测，`256/32/32/4-warps` **严格局部最优**，邻居更慢（有的 2×） | 0914 九轴 + op-evolve 4 轮只接受 1 轮 |
| 两 kernel 反向 | 扫出真实 **18%**，但**仍比 ASM 慢 2.7×** | 0916，33 点 + 3 对复现 |
| 前向 | **−3.73%**（15.8× sem），只值 e2e 的 0.15% | 0916，73 点 + 4 轮复现 |

**更关键的是结构上界。** `PLAN-4GPU-TOMORROW.md` 的分析给出：
"调度良好的 Triton 在 AMD 上落在 issue-slot wall 的 1.3–1.5×，即**反向 12.5–14 ms / 总 16.7–18.2 ms**。
**低于约 13 ms 总时间就不是 Triton 了。**"

**我们现在是 11.726 ms。** 也就是说 **Triton 不是"提升空间较小"，而是结构上就到不了我们已经在的位置。**
继续投 Triton attention，最好的结果也是回退。

（有一条相反方向的证据要记下：那份文档预测调度良好的 Triton 是 wall 的 1.3–1.5×，
而我们两个 Triton kernel 实测是 2.0× 和 2.2–2.4× —— **离墙比预想的更远，说明 Triton 内部还有空间**。
但今天的扫描证明那个空间**不在配置面上**，要拿到它得**重写 kernel**，不是调参。）

## 2.2 FlyDSL attention：被阻塞，而且它的优势恰好在我们已经赢了的那一半

**阻塞是硬的**：`primus_turbo/flydsl/attention/` 下每个 builder 都硬断言 gfx950 ——
`flash_attn_fwd.py:71` 直接 raise *"requires gfx950+ (uses `ds_read_tr16_b64`)"*，
反向的 odo / lse-transpose / dq-reduce / slot-reduce / a16-unpermute **五个 kernel 各断言一次**。
`ds_read_tr16_b64` 是 CDNA4 的 LDS 转置读，gfx1250 对应的是 `ds_load_tr16_b128`，
**寄存器落位不同** —— 这是逐 kernel 移植，不是改一个 arch 判断。

**但真正决定性的是这个数**，同事在 gfx950 上测的 FlyDSL vs AITER，**正是我们的形状**
（`dev/kyle/flydsl-attn-gqa4`，B=2，D=128，bf16 causal，min of 20）：

```
  shape                  G   fwd fly/aiter    bwd fly/aiter    fwd+bwd
  Llama-2-7B    32/32    1   0.304 / 0.566    1.439 / 1.153      +1.4%
  Llama-3.1-8B   32/8    4   0.985 / 2.066    3.408 / 3.866     -25.9%
  Llama-3.1-70B  64/8    8   1.915 / 4.096    6.573 / 7.546     -22.0%
```

llama3.1-8B 那行：**前向 2.10×，反向只有 1.13×。**

**FlyDSL 相对 AITER 的优势压倒性地在前向 —— 而前向我们已经是五机对照里最快的**
（1.572 ms，比 MI355 的 AITER 快 2.42×，比 MI455 的 Flex 快 2.10×）。
**我们缺的是反向，而 FlyDSL 在反向上只给 1.13×。**

再加一条同事自己的限定：`llama8b-flydsl-fa-pair`（0915，D=128）的 commit 原文写着
*"Isolated pair was 9.23 vs AITER a16 9.29; **this is not a training keep**"* ——
即便在它的目标平台上，那一对也只与 AITER 打平。

**结论：把 FlyDSL attention 移植到 gfx1250（逐 kernel、重写 LDS 转置路径）
换来的是我们已经有的前向，加上 1.13× 的反向。性价比很差。**

## 2.3 FlyDSL GEMM：这才是已经被证明的那条

不是 attention，但它是**唯一在 gfx1250 上实测有效的 FlyDSL 路径**：

| | 规则 3（FlyDSL wgrad） | 对照 | 效应 |
|---|--:|--:|--:|
| 8 层 | 46,374（n=3，sd 0.60%） | 42,445 | **+9.26%** |
| 32 层生产配置 | **14,050（7.08×）** | 12,340 | **+13.9%** |

机制已由 profile 证实（nkfix 的 187 ms 转置消失，非摊薄），
六个形状 SQNR 全部 345.2 dB，离线表在 8 层与 32 层通用。

**而且它可能同时解决一个更严重的问题。** 全量扫描 74 次运行发现：
**带 nkfix 的运行 21% 出现 `loss: nan`，不带 nkfix 的 35 次 0 次**（Fisher 单尾 p = 0.004）。
nkfix 的规则 1/2 是当前整条链上最大的收益来源，但它**每五次运行损坏一次**。
FlyDSL GEMM 是**替代 nkfix 的正路**（真修复而非绕路）——
如果 dgrad 也能走 FlyDSL，nkfix 可以整个退役。
代价是 dgrad 上 FlyDSL 慢（820–849 vs nkfix 绕路的 1238–1627 TF/s），
**但一个慢 1.5× 而正确的路径，胜过一个快而每五次崩一次的路径。**

---

# 建议的优先级

| 顺位 | 动作 | 预期收益 | 成本 | 依据 |
|---|---|---|---|---|
| **1** | **定位 nkfix 的 21% NaN**；若确认在规则 1/2，评估 dgrad 也走 FlyDSL | 决定当前 7.08× 能否交付 | 约 5 次运行 ≈ 一次 AC-cycle | Fisher p=0.004 |
| **2** | **向 aiter 报 GQA 越界写 bug**，并申请 gfx1250 的 `psskddv` / a16 变体 | 直接对上 MI355 的 1.6× 差距 | **近乎为零**（复现与定位已完备） | §1.3(b)、§1.2 资产对比 |
| **3** | 测 `_perf` 变体，修它的 dq pass | 未知，但 AMD 单独发布了它 | 几小时 bring-up | §1.4(i) |
| **4** | 计量 GQA 绕法（per-q-head + host 规约）自身的开销 | 界定 §1.4(ii) 的上限 | 一次算子级测量 | 从未单独量过 |
| **不建议** | Triton attention 继续投入 | **结构上界 16.7–18.2 ms，低于我们现在的 11.726** | — | §2.1 |
| **不建议** | 移植 FlyDSL attention 到 gfx1250 | 换来已有的前向 + 1.13× 反向 | 逐 kernel 重写 | §2.2 |

**关于"如何体现我们自己的优化工作"这一点，我的看法**：
目前最拿得出手的**不是**"我们选了哪个后端"，而是 §1.3(b) 那件事 ——
**我们在 AMD 自己发布的 gfx1250 ASM 资产里找到了一个会静默损坏 GQA 梯度的越界写，
并给出了完整定位与可用的绕法。** 这对 MI455 上所有跑 GQA 模型的人都成立。
第 2 项（上报 + 申请变体）把这件事变成对整条产品线的贡献，而成本几乎为零。
