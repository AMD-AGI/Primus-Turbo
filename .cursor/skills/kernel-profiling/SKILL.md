---
name: kernel-profiling
description: >-
  GPU kernel 性能分析和瓶颈诊断工作流。
  当用户需要分析 kernel 性能、诊断性能瓶颈、使用 rocprof/omniperf、
  或解读 profiling 结果时使用此技能。
---

# Kernel Profiling 工作流

## 分析流程

```
1. 运行 benchmark 获取基线数据
       ↓
2. 用 profiling 工具收集硬件计数器
       ↓
3. 判断瓶颈类型（计算/内存/延迟）
       ↓
4. 针对性优化
       ↓
5. 重新 benchmark 验证效果
```

## Step 1: 基线 Benchmark

```bash
# Attention
python3 benchmark/ops/bench_attention_turbo.py

# GEMM
python3 benchmark/ops/bench_gemm_turbo.py

# Grouped GEMM
python3 benchmark/ops/bench_grouped_gemm_turbo.py

# 全套
python3 benchmark/ops/run_suite.py -d output/
```

## Step 2: Profiling 工具

### rocprof（基础）
```bash
# Kernel 级计时
rocprof --stats python3 benchmark/ops/bench_attention_turbo.py

# 输出 results.stats.csv 包含每个 kernel 的调用次数和耗时
```

### omniperf（推荐，深度分析）
```bash
# 收集
omniperf profile -n attention_run -- python3 benchmark/ops/bench_attention_turbo.py

# 分析（交互式）
omniperf analyze -p workloads/attention_run/

# 导出报告
omniperf analyze -p workloads/attention_run/ --report-format csv
```

### PyTorch Profiler（端到端）
```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    # 运行你的 kernel
    output = model(input)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

## Step 3: 瓶颈诊断

### 判断方法

| 指标 | 计算受限 | 内存受限 | 延迟受限 |
|------|----------|----------|----------|
| MFMA Utilization | 高 (>60%) | 低 | 低 |
| HBM Bandwidth | 低 | 高 (>70% 峰值) | 低 |
| Occupancy | 适中 | 适中 | 低 |
| Wavefront Stalls | 少 | 等待内存 | 等待依赖/同步 |

### 优化策略

**计算受限**:
- 减少非 MFMA 计算（softmax、mask 等辅助计算）
- 使用更低精度（BF16 → FP8）
- 优化指令级并行

**内存受限**:
- Kernel 融合减少 HBM 访问
- 增加数据复用（tiling、LDS 缓存）
- 向量化加载（128bit / 256bit）
- 合并内存访问

**延迟受限**:
- 提高占用率（减少 VGPR/LDS 使用）
- 减少同步点（barrier、fence）
- 指令预取和 double buffering

## Step 4: 常用 Profiling 脚本

```bash
# 快速诊断脚本
bash .cursor/skills/kernel-profiling/scripts/quick_profile.sh <script.py>
```

## 参考
- omniperf 文档: https://rocm.docs.amd.com/projects/omniperf/
- rocprof 文档: https://rocm.docs.amd.com/projects/rocprofiler/
