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
- **参考**: AITER `flash_attn_triton_amd/` 使用 JSON 配置，按 arch 分别调优
- **参考**: AITER 同时调优 `waves_per_eu` (2 or 3) 和 `num_stages`

### P0: Causal Masking 三分法
- 将 KV 块分为三类，分别处理:
  1. **完全 unmasked**: 无需 mask 计算，直接 QK^T → softmax → V
  2. **完全 masked**: 直接跳过 (zero compute)
  3. **部分 masked**: 完整 mask 逻辑
- 判断依据: Q-block 行范围 vs K-block 列范围的 causal 关系
- **参考**: FA3 `hopper/mask.h`, FA4 bitmask causal
- 预期收益: 15-25% (causal attention, seq_len ≥ 2048)

### P1: Branchless Rescaling (AVO)
- Online softmax 中 m_new > m_old 时 rescale O 累加器
- 有分支版本导致 warp divergence (wavefront=64 更严重)
- 无分支: 始终计算 alpha = exp(m_old - m_new)，当 m_new==m_old 时 alpha=1.0
- **参考**: AVO 论文，non-causal 场景收益最大 (+8.1%)

### P1: exp2 域 Softmax (ThunderKittens)
- 预乘 scale × log₂(e): scale_log2 = 1.44269504 / √d
- 用 exp2(x) 代替 exp(x)，减少 transcendental 指令开销
- **参考**: ThunderKittens `mha_h100.cu` 中 `1.44269504089f * 0.08838834764f`
- 需验证 Triton AMD 后端 tl.math.exp2 是否映射到快速硬件指令

### P1: KV Double Buffering
- 计算当前 KV block 的同时预取下一个 KV block 到 LDS
- Triton 实现: 手动展开 2 次循环迭代，交替使用缓冲区
- **参考**: ThunderKittens multi-stage pipeline, FA3 Hopper path
- 注意: 增加寄存器/LDS 压力，需平衡 occupancy

### P2: 内存访问模式
- Q/K/V 加载的合并度（coalescing）
- 利用 `eviction_policy` 控制 L2 缓存行为
- Q tile 可用 `.ca` (cache all)，K/V 流式用 `.cs` (cache streaming)

### P2: Lean Attention (Stream-K for Decode)
- **来源**: AITER `lean_atten.py`
- 将 KV 维度切分为多个 partition, 不同 CU 处理不同 partition
- 需要 lock-based partial softmax merge
- 对 decode 场景 (short Q, long KV) 收益大

### P3: Paged Attention (推理场景)
- **来源**: vLLM, AITER `pa_decode.py`
- KV cache 分页管理，非连续内存布局
- V1 (简单) vs V2 (sequence partitioning, SEQ_PARTITION_SIZE=1024)
- 推理部署的必要功能

### P3: SageAttention (INT8 Q/K)
- **来源**: AITER `fav3_sage.py`
- Q/K 量化到 INT8, V 保持 FP16, 按 arch 调参
- 精度影响大，需要仔细验证

## 标准验证流程

```bash
# 1. 编译
pip3 install --no-build-isolation -e . -v

# 2. 精度验证（必须通过）
python3 benchmark/accuracy/eval_attention_accuracy.py
pytest tests/pytorch/ops/test_attention.py -x

# 3. 性能 benchmark
python3 benchmark/ops/bench_attention_turbo.py

# 4. 快速 benchmark（仅测关键配置）
bash .cursor/skills/triton-attention-optimize/scripts/bench_quick.sh

# 5. 多后端对比
python3 benchmark/ops/bench_attention_turbo.py   # Triton
python3 benchmark/ops/bench_attention_fa.py      # FlashAttention
python3 benchmark/ops/bench_attention_torch.py   # PyTorch SDPA
```

## 参考资料
- FlashAttention 算法细节: [flash-attention-algo.md](flash-attention-algo.md)
- SOTA 知识库: `.cursor/skills/sota-knowledge-base/SKILL.md`
- AITER AMD FA: `ref/OP_optimize/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/`
- ThunderKittens MHA: `ref/OP_optimize/ThunderKittens/kernels/attention/`
- vLLM Triton Attention: `ref/OP_optimize/vllm/vllm/v1/attention/`
