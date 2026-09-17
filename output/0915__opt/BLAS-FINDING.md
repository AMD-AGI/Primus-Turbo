# hipBLASLt 在这个镜像里不是慢，是路径错位

## 现象

任何 matmul 抛 `HIPBLAS_STATUS_INVALID_VALUE`（出自 `hipblasLtMatmulAlgoGetHeuristic`），
伴随 `rocblaslt error: Cannot read ".../hipblaslt/library/TensileLibrary_lazy_gfx1250.dat"`。
唯一的绕法是 `TORCH_BLAS_PREFER_HIPBLASLT=0` 退回 rocBLAS。

今天这个问题咬了三处：T2 bring-up 的 fp32 参考、第一次 e2e、以及 `gemm_roof.py` 自己。
`tune_attention.py` 在 import torch 之前就设了那个变量，所以**整个 campaign 的 op 级测量从未见过它** ——
这也是它没被写进任何 runbook 的原因。

## 根因

加载器在 `library/` 下找 `TensileLibrary_lazy_gfx1250.dat`，而 402 个文件（含该索引）
全在 `library/gfx1250/` **子目录**里：

```
找的：  _rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/TensileLibrary_lazy_gfx1250.dat
实际：  _rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250/TensileLibrary_lazy_gfx1250.dat.zlib
```

`library/` 下只有一个条目，就是 `gfx1250/`。这是镜像的打包问题，不是 gfx1250 的能力问题。

## 修法（不改文件）

```bash
-e HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
```

注意**不要**用 `_rocm_sdk_devel/lib/hipblaslt/library/gfx1250`，那条路径会 core dump。

## 实测（8192³ bf16，1100 MHz）

| 后端 | TFLOP/s | 耗时 | |
|---|--:|--:|---|
| rocBLAS（`TORCH_BLAS_PREFER_HIPBLASLT=0`，当前 e2e 走的） | 27.64 | 39.78 ms | backend=Cublas |
| **hipBLASLt（修正路径后）** | **68.74** | **15.99 ms** | backend=Cublaslt，**2.49×** |
| Triton GEMM（BM128/BN256/BK32/w8/s3） | 897.53 | 1.225 ms | 对 rocBLAS **32.5×** |

## 与既有文档的出入

`tools/gfx1250/tune_attention.py` 的注释记的是"镜像里有 46 个 gfx1250 bf16 Tensile 方案、
torch 报 backend 是 Cublaslt、什么都不抛，保留这个设置的理由是 hipBLASLt 在这块卡上就是慢"。

在**这个镜像**上不是这样：hipBLASLt 整个不可用，会抛异常，直到把路径覆盖掉。
两件事要分开 —— 一个是打包缺陷（可修，2.49×），一个是性能差距（真实，修完仍离 Triton 13×）。

## 端到端实测：8.3×

同配置、同步数、只换 BLAS 后端（llama-3.1-8B，b=4 s=8192，单卡 1100 MHz）：

| | 单步墙钟 | tps | mfu |
|---|--:|--:|--:|
| rocBLAS | 135 / 134 s | 244 | 4.53% |
| **hipBLASLt（修正路径）** | **17 / 17 s** | **2,027** | **37.62%** |

**单步 7.9×，吞吐 8.3×。**

注意这个比值**远大于**微基准预测的 2.49×（8192³ bf16 上 27.64 → 68.74）。
说明 rocBLAS 在模型实际的 GEMM 形状上比在方形基准上差得多 ——
**微基准的比值不能直接外推到端到端**，这是今天第三次撞上同类问题。

## 影响

e2e 单步 134 秒，其中 attention 全部 32 层约 374 ms，**占 0.28%**。
按 27.6 TF/s 算，llama-3.1-8B 每步 `6 × 8e9 × 32768 = 1.57e15` FLOPs 需要 57 秒，量级吻合 ——
**e2e 是 GEMM-bound**。修掉路径预计 134 → 约 54 秒，仍然 GEMM-bound。

修好之后单步 17 秒，attention 占比从 0.28% 升到约 **2.2%** —— 仍然小，
但 ASM 反向省下的约 260 ms 约占 1.5%，已经进入 20 步取稳态可测的范围。

**剩下的差距仍在 GEMM**：hipBLASLt 68.7 vs Triton 897 TF/s，还有 13×。
把 GEMM 钉到 Triton 需要 `torch.compile`，而配置自己的注释记载
inductor 对 TransformerBlock 做 autotune 时会抛 `hipErrorLaunchFailure` **并打死 GPU**。
这个矛盾未解，今天不碰。
