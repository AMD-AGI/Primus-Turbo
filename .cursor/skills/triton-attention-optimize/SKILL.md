---
name: triton-attention-optimize
description: >-
  优化 Primus-Turbo 中的 Triton FlashAttention kernel 性能。
  当用户要求优化 attention kernel、修改 attention_kernel.py、
  讨论 FlashAttention 性能、或分析 attention 相关 benchmark 结果时使用。
---

# Triton Attention 优化指南

## 核心文件

| 文件 | 作用 |
|------|------|
| `primus_turbo/triton/attention/attention_kernel.py` | Triton kernel 实现（~1700行） |
| `primus_turbo/pytorch/kernels/attention/attention_triton_impl.py` | PyTorch 封装和 launch 逻辑 |
| `primus_turbo/pytorch/kernels/attention/attention_aiter_impl.py` | AITER 后端（对比参考） |
| `primus_turbo/pytorch/ops/attention/` | 高级 API |
| `benchmark/ops/bench_attention_turbo.py` | 性能 benchmark |
| `benchmark/accuracy/eval_attention_accuracy.py` | 精度验证 |
| `benchmark/ops/config.py` | benchmark 配置（batch/seq_len/heads 等） |

## 当前实现概况

基于 FlashAttention v2，核心参数：
- `FIXED_BLOCK_M = 64`, `FIXED_BLOCK_N = 64`
- 支持布局: `bhsd`, `bshd`, `thd`（变长序列）
- 支持 causal masking、GQA、FP8
- 前向 + 反向传播

## 优化方向（按优先级）

### P0: Block Size 与 Autotune
- 当前 BLOCK_M/N 固定为 64，不同 head_dim 和 seq_len 下可能非最优
- 启用 autotune: `PRIMUS_TURBO_TRITON_AMD_AUTOTUNE=1`
- 尝试 BLOCK_M/N ∈ {32, 64, 128} 的组合

### P1: Causal Masking 优化
- 跳过完全 masked 的 K-block（当前可能仍在计算）
- 对 fully-unmasked 块使用更轻量的路径（无需 mask 计算）
- 参考 FA3/FA4 的 bitmask causal masking 技术

### P2: 内存访问模式
- Q/K/V 加载的合并度（coalescing）
- 利用 `eviction_policy` 控制 L2 缓存行为
- 考虑 Q-tile 的 double buffering / prefetch

### P3: Softmax 计算
- Online softmax 的 running max/sum 更新效率
- 减少不必要的 `tl.debug_barrier()` 同步
- 累加器重缩放的分支消除（参考 AVO 论文的 branchless rescaling）

## 标准验证流程

```bash
# 1. 编译
pip3 install --no-build-isolation -e . -v

# 2. 正确性（必须通过）
python3 benchmark/accuracy/eval_attention_accuracy.py

# 3. 性能
python3 benchmark/ops/bench_attention_turbo.py

# 4. 快速 benchmark（仅测关键配置）
bash .cursor/skills/triton-attention-optimize/scripts/bench_quick.sh
```

## 参考资料
- FlashAttention 算法细节: [flash-attention-algo.md](flash-attention-algo.md)
- AVO 论文的优化发现可作为灵感来源（见 AI-summary 笔记）
