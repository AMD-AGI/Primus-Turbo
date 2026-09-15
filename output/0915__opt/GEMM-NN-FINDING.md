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
