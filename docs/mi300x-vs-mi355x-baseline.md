# MI300X vs MI355X vs GB200 Baseline 性能对比

> **数据来源**
> - MI300X: Primus-Turbo 本地 benchmark (2026-04-02, BF16)
> - MI355X / GB200: Primus-Turbo 2026 Q1 Summary Report (`docs/report.pdf`)
> - 峰值 TFLOPS: 各厂商官方规格 (dense, non-sparsity)

---

## 1. 硬件规格对比

| 规格 | MI300X (CDNA3) | MI355X (CDNA4) | GB200/B200 (Blackwell) |
|------|:-:|:-:|:-:|
| **BF16 Peak** | 1,307 TFLOPS | 2,500 TFLOPS | ~2,250 TFLOPS |
| **FP8 Peak** | 2,615 TFLOPS | 5,000 TFLOPS | ~4,500 TFLOPS |
| **MXFP4 Peak** | N/A | 10,100 TFLOPS | ~6,750 TFLOPS (FP4) |
| **HBM 容量** | 192 GB (HBM3) | 288 GB (HBM3E) | ~192 GB (HBM3E) |
| **HBM 带宽** | 5.3 TB/s | 8.0 TB/s | ~8.0 TB/s |
| **CU 数量** | 304 | 256 | 160 SM |
| **架构** | gfx942 | gfx950 | sm_100 |

**理论算力倍率 (相对 MI300X):**

| 精度 | MI355X / MI300X | GB200 / MI300X |
|------|:-:|:-:|
| BF16 | **1.91x** | **1.72x** |
| FP8 | **1.91x** | **1.72x** |
| HBM BW | **1.51x** | **1.51x** |

---

## 2. GEMM 性能对比

### 2.1 GEMM BF16

> Workload: Llama-2-7B/70B, Llama-3.1-8B/405B, Qwen2.5-7B/72B, Mistral-7B

| 指标 | MI300X Turbo | MI355X ROCm-TE | MI355X Turbo Best | GB200 NV-TE |
|------|:-:|:-:|:-:|:-:|
| **Avg Fwd (TFLOPS)** | 607.4 | 1,524.8 | **1,534.6** (HipBLASLt) | 1,434.4 |
| **Avg Bwd (TFLOPS)** | 578.1 | 1,339.4 | **1,351.1** (AutoTune) | 1,310.2 |
| **Fwd 利用率** | 46.5% | 61.0% | **61.4%** | 63.7% |
| **Bwd 利用率** | 44.2% | 53.6% | 54.0% | 58.2% |

```
BF16 Fwd 实测倍率:
  MI355X Turbo / MI300X = 1534.6 / 607.4 = 2.53x  (理论 1.91x → 超线性!)
  GB200 NV-TE / MI300X  = 1434.4 / 607.4 = 2.36x
  MI355X Turbo / GB200  = 1534.6 / 1434.4 = 1.07x  (MI355X 领先 7%)
```

**MI355X Turbo 各后端 GEMM BF16：**

| 后端 | Fwd TFLOPS | Bwd TFLOPS |
|------|:-:|:-:|
| HipBLASLt | **1,534.6** | 1,350.3 |
| CK | N/A | N/A |
| Triton | 1,146.9 | 1,225.5 |
| AutoTune | 1,532.0 | **1,351.1** |
| ROCm-TE | 1,524.8 | 1,339.4 |

> **观察**: HipBLASLt 在 Fwd 上最强, Triton 落后 ~25%。AutoTune 选择了 HipBLASLt。

---

### 2.2 GEMM FP8 (Tensorwise Scaling)

| 指标 | MI355X ROCm-TE | MI355X Turbo Best | GB200 NV-TE |
|------|:-:|:-:|:-:|
| **Avg Fwd (TFLOPS)** | 1,965.6 | **2,297.8** (HipBLASLt) | 2,081.9 |
| **Avg Bwd (TFLOPS)** | **2,254.0** | 1,884.5 (Triton) | 2,204.1 |
| **Fwd 利用率** | 39.3% | **45.9%** | 46.3% |

```
FP8 TW Fwd:
  MI355X Turbo / GB200 = 2297.8 / 2081.9 = 1.10x  (MI355X 领先 10%)
```

**MI355X Turbo 各后端 GEMM FP8 TW：**

| 后端 | Fwd TFLOPS | Bwd TFLOPS |
|------|:-:|:-:|
| HipBLASLt | **2,297.8** | 1,354.1 |
| CK | 1,675.4 | 160.5 ⚠️ |
| Triton | 1,878.1 | **1,884.5** |
| AutoTune | 2,295.2 | 1,866.0 |

> **观察**: CK Bwd 异常低 (160T)，可能存在 bug。HipBLASLt Fwd 最优但 Bwd 差。Triton Bwd 最佳。

---

### 2.3 GEMM FP8 (Blockwise Scaling)

| 指标 | MI355X Turbo Triton | MI355X Turbo AutoTune | GB200 NV-TE |
|------|:-:|:-:|:-:|
| **Avg Fwd** | 1,369.9 | 1,350.8 | 1,944.1 |
| **Avg Bwd** | 1,009.1 | 1,006.7 | 2,062.8 |

> **观察**: Blockwise FP8 是 DeepSeek 训练使用的量化方式。MI355X Turbo 在此项落后 GB200 **~30% Fwd / ~51% Bwd**，有较大优化空间。

---

### 2.4 GEMM MXFP8 / MXFP4 (仅 Llama2-7B)

| 精度 | 指标 | MI355X Turbo Best | GB200 NV-TE |
|------|------|:-:|:-:|
| **MXFP8** | Fwd | 1,012.0 (AutoTune) | **1,947.1** |
| | Bwd | 1,121.1 (AutoTune) | **1,864.3** |
| **MXFP4** | Fwd | **2,346.7** (Aiter) | 2,127.2 |
| | Bwd | **2,490.2** (Aiter) | 2,390.8 |

> **观察**: MXFP8 严重落后 GB200（约 52% Fwd）；MXFP4 通过 Aiter 后端反超 GB200 ~10%。

---

## 3. Grouped GEMM 性能对比 (MoE 核心)

### 3.1 Grouped GEMM BF16

> Workload: DeepSeek-V2/V3, Qwen3, Mixtral, Grok, Kimi-K2 等 MoE 模型

| 指标 | MI300X Turbo | MI355X ROCm-TE | MI355X Turbo Best | GB200 NV-TE |
|------|:-:|:-:|:-:|:-:|
| **Avg Fwd (TFLOPS)** | 475.4 | 904.4 | **1,098.7** (Triton) | 1,133.2 |
| **Avg Bwd (TFLOPS)** | 377.4 | 501.1 | **922.3** (Triton) | 932.0 |
| **Fwd 利用率** | 36.4% | 36.2% | **43.9%** | 50.4% |

```
BF16 Grouped GEMM Fwd 实测倍率:
  MI355X Turbo / MI300X = 1098.7 / 475.4 = 2.31x  (理论 1.91x → 超线性)
  GB200 NV-TE / MI300X  = 1133.2 / 475.4 = 2.38x
  MI355X Turbo / GB200  = 1098.7 / 1133.2 = 0.97x  (MI355X 落后 3%)
```

**MI355X Turbo 各后端 Grouped GEMM BF16：**

| 后端 | Fwd TFLOPS | Bwd TFLOPS |
|------|:-:|:-:|
| HipBLASLt | 940.1 | 659.2 |
| CK | 1,065.6 | 746.3 |
| **Triton** | **1,098.7** | **922.3** |
| AutoTune | 1,084.4 | 905.2 |

> **关键发现**: Grouped GEMM 场景下 **Triton 是最优后端**，超越 HipBLASLt/CK，这与 GEMM (HipBLASLt 最优) 形成鲜明对比。

---

### 3.2 Grouped GEMM FP8

| 量化方式 | 指标 | MI355X Turbo Best | GB200 NV-TE | 差距 |
|----------|------|:-:|:-:|:-:|
| **Tensorwise** | Fwd | **1,203.1** (AutoTune) | 992.7 | **MI355X +21%** |
| | Bwd | **1,304.9** (Triton) | 1,223.5 | MI355X +7% |
| **Rowwise** | Fwd | 859.7 (Triton) | N/A | — |
| | Bwd | 1,179.6 (Triton) | N/A | — |
| **Blockwise** | Fwd | 653.5 (Triton) | **968.8** | MI355X -33% |
| | Bwd | 557.0 (Triton) | **1,112.8** | MI355X -50% |

> **观察**:
> - Tensorwise: MI355X 大幅领先 GB200 (+21% Fwd)
> - Blockwise: MI355X 显著落后 GB200，与 GEMM FP8 Blockwise 趋势一致

---

## 4. 总览：MI355X Turbo Best vs GB200 NV-TE

| 算子 | 精度 | MI355X Turbo Fwd | GB200 NV-TE Fwd | MI355X 胜负 |
|------|------|:-:|:-:|:-:|
| GEMM | BF16 | **1,534.6** | 1,434.4 | ✅ **+7%** |
| GEMM | FP8 TW | **2,297.8** | 2,081.9 | ✅ **+10%** |
| GEMM | FP8 BW | 1,350.8 | **1,944.1** | ❌ **-30%** |
| GEMM | MXFP8 | 1,012.0 | **1,947.1** | ❌ **-48%** |
| GEMM | MXFP4 | **2,346.7** | 2,127.2 | ✅ **+10%** |
| Grouped GEMM | BF16 | 1,098.7 | **1,133.2** | ❌ -3% |
| Grouped GEMM | FP8 TW | **1,203.1** | 992.7 | ✅ **+21%** |
| Grouped GEMM | FP8 BW | 653.5 | **968.8** | ❌ **-33%** |

**MI355X vs GB200 战绩: 4 胜 / 4 负**

---

## 5. MI300X → MI355X 实际性能提升

| 算子 | 精度 | MI300X (TFLOPS) | MI355X Best (TFLOPS) | 实际倍率 | 理论倍率 |
|------|------|:-:|:-:|:-:|:-:|
| GEMM Fwd | BF16 | 607.4 | 1,534.6 | **2.53x** | 1.91x |
| GEMM Bwd | BF16 | 578.1 | 1,351.1 | **2.34x** | 1.91x |
| Grouped GEMM Fwd | BF16 | 475.4 | 1,098.7 | **2.31x** | 1.91x |
| Grouped GEMM Bwd | BF16 | 377.4 | 922.3 | **2.44x** | 1.91x |
| Attention Fwd | BF16 | 456.1 (AITER) | — | — | 1.91x |

> **分析**: 所有 BF16 算子在 MI355X 上的实际加速比都 **超过理论倍率 1.91x**，
> 达到 2.3-2.5x。原因可能包括:
> 1. MI355X CDNA4 架构改进（更高效的 Matrix Core 调度）
> 2. MI355X 更高 HBM 带宽 (8.0 vs 5.3 TB/s) 缓解内存瓶颈
> 3. MI300X baseline 未充分利用硬件（利用率 36-47% vs MI355X 44-61%）

---

## 6. 利用率对比（Fwd）

```
                  MI300X      MI355X     GB200
GEMM BF16:        46.5%      61.4%      63.7%    ← GEMM 整体最高
Grouped GEMM BF16: 36.4%     43.9%      50.4%    ← 优化空间最大
GEMM FP8 TW:       —         45.9%      46.3%
GEMM MXFP8:        —         20.2%      43.3%    ← MI355X MXFP8 严重偏低
GEMM MXFP4:        —         23.2%      31.5%    ← 新精度，均偏低
```

---

## 7. 关键发现 & 优化建议

### 强项（MI355X 领先 GB200）
1. **GEMM BF16/FP8 TW** — HipBLASLt 后端高度优化
2. **GEMM MXFP4** — Aiter 后端在新精度上表现优秀
3. **Grouped GEMM FP8 TW** — Triton 后端领先 GB200 21%

### 短板（MI355X 落后 GB200）
1. **FP8 Blockwise (GEMM + Grouped GEMM)** — 落后 30-50%，DeepSeek 训练常用
2. **MXFP8** — 落后 ~48%，HipBLASLt 和 AutoTune 均表现不佳
3. **Grouped GEMM BF16** — 微幅落后 3%

### MI300X 优化方向
1. **GEMM BF16**: 利用率 46.5%，MI355X 达到 61%，说明 MI300X 还有 ~30% 提升空间
2. **Grouped GEMM BF16**: 利用率 36.4%，MI355X 上 Triton 是最优后端 → MI300X 也应优先优化 Triton Grouped GEMM
3. **Attention**: MI300X AITER 利用率 34.9%，需关注 MI355X 上的 Attention 数据（报告中未包含）
