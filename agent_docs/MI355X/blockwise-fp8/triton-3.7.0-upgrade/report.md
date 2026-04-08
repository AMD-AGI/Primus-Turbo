# Triton 3.7.0 升级性能报告 — Blockwise FP8 GEMM on MI355X

## 概要

| 指标 | Triton 3.5.1 (旧) | Triton 3.7.0 (新) | 变化 |
|---|---|---|---|
| **Avg Forward TFLOPS** | 1293.14 | 1525.97 | **+18.0%** |
| **Avg Backward TFLOPS** | 974.06 | 1225.25 | **+25.8%** |
| 精度 | 192/192 PASS | 192/192 PASS | 无回归 |

## 版本信息

- **旧 Triton**: `pytorch-triton-rocm 3.5.1+gitbfeb0668`
- **新 Triton**: `triton 3.7.0+git5f968786` (from `/shared_nfs/yaoc/agent_work/triton`)
- **Branch**: `agent-optimize`
- **GPU**: AMD Instinct MI355X (gfx950, CDNA4)

## 按模型分组对比

| Model | Fwd 提升 | Bwd 提升 |
|---|---|---|
| Llama-3.1-405B | **+23.8%** | **+29.4%** |
| Qwen2.5-72B | **+21.4%** | **+28.0%** |
| Llama-2-70B | **+21.1%** | **+26.8%** |
| Qwen2.5-7B | +16.6% | +25.0% |
| Llama-2-7B | +15.0% | +23.0% |
| Llama-3.1-8B | +14.5% | +25.5% |
| Mistral-7B | +14.5% | +21.6% |

## 关键观察

1. **大模型收益更显著**: 405B/72B/70B 模型的提升 (21-30%) 高于 7B/8B 模型 (14-25%)，
   说明新 Triton 在大矩阵的寄存器分配和指令调度上优化更明显。

2. **Backward 比 Forward 提升更大**: Bwd 平均 +25.8% vs Fwd +18.1%，
   暗示 TN layout (backward) 的代码生成在新版本中改善更多。

3. **全部 84 个 case 均有提升**: 最小提升 +9.7% (Mistral-7B MBS=1 M=4096 N=4096 K=4096 Bwd)，
   最大提升 +34.0% (Qwen2.5-72B MBS=2 M=16384 N=8192 K=29568 Bwd)。

4. **零精度回归**: 192/192 blockwise 测试全部通过。

## 注意事项

新 Triton 3.7.0 的 editable install 存在 namespace package 冲突：
- 需要设置 `PYTHONPATH="/shared_nfs/yaoc/agent_work/triton/python:$PYTHONPATH"` 才能正确导入
- 直接 `import triton` 会找到不完整的 namespace package（缺少 `triton.jit`）
- 建议正式安装时使用 `pip install .` 而非 editable mode
