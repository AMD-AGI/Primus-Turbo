# hipBLASLt 在 NN 上没有纯 GEMM 调优库，e2e 因此损失约一半吞吐

本机 8 层配置端到端 **6,327 → 12,113 tps（1.91×）**，MFU **34% → 65.4%**，
单改一件事：把 `aten::mm` 里 B 为连续 `(K,N)` 的调用换成 B 为 `(N,K)`-major 的等价形式。

## 怎么发现的

step 10 的 torch profile，单步 5.2 s：

| | ms | 占比 |
|---|--:|--:|
| **Tensile (hipBLASLt) GEMM** | **4877** | **94%** |
| attention（`bwd_kernel_causal` + `attn_fwd`） | 171 | 3.3% |
| 其余全部（elementwise / optimizer / reduce / softmax） | ~140 | 2.7% |

GEMM 的 4877 ms 里 **4727 ms（97%）** 花在 macro tile **`MT32x16x32`** 的 solution 上 ——
单次调用 592–681 ms，grid 高达 **6567 万**个 64 线程 workgroup。
同一批形状里走 `MT256x256x128` 的部分，单次只要 0.796 ms。

逐形状对照（同形状、不同 solution）：

| 形状 | `Ailk_*` MT32x16x32 | `Alik_*` MT256x256x128 | 倍数 |
|---|--:|--:|--:|
| `[32768,4096]×[4096,14336]` | 51.0 ms/次 | 3.1 ms/次 | 16.5× |
| `[32768,14336]×[14336,4096]` | 51.6 ms/次 | 2.4 ms/次 | 21.5× |
| `[32768,4096]×[4096,4096]` | 14.6 ms/次 | 0.82 ms/次 | 17.8× |

另有 `[32768,128256]` / `[128256,32768]`（128256 = 词表）的 lm_head dgrad+wgrad，
单步合计 **1273 ms，占 26%**。

## 根因：NN 组合缺少纯 bf16 GEMM 的调优库

`HIPBLASLT_LOG_LEVEL=4` + `TENSILE_DB=0x6`，同一个 M/N/K、同样 76 MB workspace，
**唯一差别是 transpose 组合**：

```
CASE A  torch.mm(a, b)          transA=OP_N transB=OP_N
        ProblemMap: Contraction_l_Ailk_Bljk...     sol-tag = GridBased
        Object key: 4096, 32768, 1, 14336
        最近邻 ->  M=1024, N=1, B=1, K=512          <-- N 差了三万倍
        Considered 11 (31.43%) of entries          ->  sol-idx 104  (MT32x16x32)

CASE B  torch.mm(a, bT.t())     transA=OP_T transB=OP_N
        ProblemMap: Contraction_l_Alik_Bljk...     sol-tag = Prediction
                                                   ->  sol-idx 237  (MT256x256x128)
```

`MT32x16x32` 正是为 `N=1` 这类 GEMV 式细长形状设计的 tile，被套用到了 N=32768 上。

**物证**在库文件本身。`library/gfx1250/` 下按 transpose 组合分文件：

```
NN (Ailk_Bljk) 的 bf16 库：        TN (Alik_Bljk) 的 bf16 库：
  BB_BB_Bias_UA_Type_BB_HPA          BB_BB_Bias_UA_Type_BB_HPA
  BB_BB_HA_Bias_Grad_UA_...          BB_BB_HA_Bias_Grad_UA_...
  BB_BB_HA_Bias_SAV_UA_...           BB_BB_HA_Bias_SAV_UA_...
  BB_BB_HA_Grad_UA_...               BB_BB_HA_Grad_UA_...
                                     BB_BB_UA_Type_BB_HPA         <-- 纯 GEMM
                                     BB_BB_UA_Type_BB_HPA_CU96    <-- CU 变体
                                     BB_BB_UA_Type_BB_HPA_CU192   <-- CU 变体
```

**`BB_BB_UA_Type_BB_HPA`（不带 Bias/Grad/SAV 的纯 bf16 GEMM 调优库）TN 有三份，NN 一份没有。**
文件数整体也不对等：`Alik_Bljk` 68 个，`Ailk_Bljk` 40 个。

于是纯 GEMM 调用走 NN 时只能落到带 Bias/SAV 的库上，而那些库只有稀疏的 `GridBased` 查找表、
覆盖不到大 N 区域 —— `Prediction` 模型是 TN 才有的。

这与 `BLAS-FINDING.md` 里的 `HIPBLASLT_TENSILE_LIBPATH` 错位是**同一家族的打包/覆盖缺陷**，
不是 gfx1250 的能力问题：同一块卡、同一时钟，NT 路径实测 **1502 TF/s**。

## 变通修法（已验证）

`aten::mm` 在 B 为连续 `(K,N)`、bf16/fp16、B 大于阈值时，改传 `b.t().contiguous().t()` ——
逻辑上同一个 `(K,N)`，物理上 N-major。实现见 `bin/nkfix.py`（`TorchDispatchMode`）。

> 注意：不能用 `torch.library` 覆盖 `aten::mm.default` —— 捕获的"原始"函数会解析回覆盖本身，
> fallback 路径无限递归炸栈。`TorchDispatchMode` 自带重入保护。

微基准（**已含每次转置的开销**）：

| 形状 | 原 | 改后 | 净收益 |
|---|--:|--:|--:|
| dgrad mlp | 55.5 ms / 69 TF | 2.36 ms / 1628 TF | **20.1×** |
| wgrad mlp | 52.8 ms / 73 TF | 3.25 ms / 1183 TF | 12.7× |
| lm_head | 678 ms / 51 TF | 19.8 ms / 1742 TF | **29.1×** |

**数值零代价**：两条路径对 fp32 参考的 SQNR 逐位相同
（K=14336/32768/128256 上 55.60 / 55.62 / 55.62 dB，门槛 50 dB）；
两者互差 0.3–0.4% 值域，是累加顺序，不是偏差。

端到端（8 层配置，`repro_l8b_turbo_conv_8L_fast.yaml`）：

| | tps | MFU |
|---|--:|--:|
| 对照 | 6,327 | ~34% |
| **nkfix ON** | **12,113** | **65.4%** |

端到端 1.91× 低于微基准的倍数，因为只修了 B 连续那一类，且 GEMM 修好后其余部分开始显现。
**n=1，尚未按 3.9% 的跨运行噪声地板做重复验证**（见 `E2E-AB.md`）——
但 1.91× 远超那个地板，方向不存疑，精确幅度待定。

## 上报建议

两条，都是镜像侧的：

1. **`Ailk_Bljk`（NN）缺 `BB_BB_UA_Type_BB_HPA` 及其 CU 变体。** 影响所有走 NN 的 bf16 GEMM ——
   在 PyTorch 训练里就是**全部 Linear 的 dgrad**，即每个模型每步的大头。
2. `hipblaslt-bench` 在这个镜像里用不了：它要 `TensileLibrary_lazy_gfx1250.dat`，
   而这里是"非 lazy"布局（每个 problem type 一个独立 `.dat.zlib`，无总索引）。
   这挡住了用户自己做 tuning 或验证 solution 的路。

## 这条发现同时更正了此前的判断

`BLAS-FINDING.md` 说"hipBLASLt 68.7 vs Triton 897，还有 13×"——
那是在 **NN 8192³ 方阵**上测的，而它恰好命中了这个缺陷。
所以 13× **在 NN 上是真的**，但它不是"hipBLASLt 慢"，是"NN 没调优库"。
同一块卡同一时钟，NT 上 hipBLASLt 是 **1502 TF/s，比那个 Triton 数字还快 1.67×**。

---

## 重复验证（n=9）：1.894×，而且三模态就此消失

`repro_l8b_turbo_conv_8L_fast.yaml`，种子已钉，步数 10，取 step≥4 中位：

| | n | 均值 tps | 跨运行 sd | 模态 |
|---|--:|--:|--:|---|
| 未打 patch | 9 | 6,128.1 | **3.91%** | 5822×2 \| 5936×2 \| 6327×5 |
| **nkfix ON** | 9 | **11,604.4** | **0.42%** | **11604×9（单模态）** |

**加速 1.894×；跨运行方差降低 9.3×。**

### 这顺带解掉了当天最大的一个谜

`E2E-AB.md` 花了一整天刻画"本机 e2e 跨运行噪声约 3.9% 且三模态"，并据此推翻了此前全部 n=1 的结论。
现在因果清楚了：**三模态的源头就是 NN 路径上 `GridBased` 最近邻查表的 solution 选择。**
那张表覆盖不到 (M=4096, N=32768, K=14336) 这片区域，最近邻落在 `N=1`，
不同进程可能落到不同的 solution 上 —— 绕开它，散布从 3.91% 降到 0.42%，模态消失。

这也解释了为什么两个 A/B 臂的方差几乎相同（3.88% / 3.91%）：
它们共用同一条 GEMM 路径，而模态不来自 attention，来自 GEMM。

**同时撤回一条此前的排除结论。** 早先"GEMM 层已排除为模态源"是用
**NT 布局和 8192³ 方阵**测的（跨进程 0.19%/1.2%/0.36%，极稳）——
那不是 e2e 实际走的路径。改用 e2e 真实的 NN 形状重测，跨进程变异 14–23%：

```
dgrad_mlp  55.80–66.81 ms   (~20%)
wgrad_mlp  55.06–67.59 ms   (~23%)
lm_head    691.7–790.6 ms   (~14%)
```

## 未决：0915 那次 `MES ring buffer is full` 挂卡

nkfix 连续第 4 次运行时卡 wedge（`MES(0,0) ring buffer is full` →
`MES(4,0) failed to respond` → `failed to reg_write_reg_wait`），进程成僵尸、
`torch.cuda.init()` 挂住，需要人工 AC-cycle。

**AC-cycle 后以 `E2E_COOLDOWN=20`（此前是 10）连续跑 6 次 nkfix ON，未能复现**：
6/6 成功，零 MES 报错，零 KFD 残留。

所以 **nkfix 与那次挂卡的因果没有建立**。两种解释都还站得住：
cooldown 10 秒不足以让卡排空，或者那是个随机事件。
当天另一次挂卡（converter/inductor）机制完全不同，而 `ring buffer is full`
这个签名当天只出现过那一次。**不要据此认为 nkfix 安全，也不要据此认为它有问题。**
跑长任务前应先补足样本。

## 适用范围

以上全部在 **8 层**配置上测得。32 层生产配置未测 ——
那个配置显存占用 88%，而 nkfix 的转置会增加瞬时分配。
`bin/nkfix_cached.py` 是一个带转置缓存的变体（权重一步之内不变，可复用），
用 `_version` + weakref 做 key 以避开分配器地址复用的陷阱，**尚未验证**。
