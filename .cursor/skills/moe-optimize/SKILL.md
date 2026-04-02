---
name: moe-optimize
description: >-
  优化 Primus-Turbo 中的 MoE (Mixture of Experts) 相关算子。
  当用户要求优化 MoE kernel、讨论 fused MoE、grouped GEMM for MoE、
  或分析 MoE 性能瓶颈时使用。
---

# MoE 优化指南

## 核心文件

| 文件 | 作用 |
|------|------|
| `primus_turbo/triton/moe/fused_router_kernel.py` | Fused routing kernel (topk + scoring) |
| `primus_turbo/triton/moe/permutation.py` | Token permutation |
| `primus_turbo/triton/moe/tokens_per_expert_to_mask_kernel.py` | Expert mask 生成 |
| `primus_turbo/triton/moe/multihot_to_indices.py` | Multi-hot 转索引 |
| `primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py` | MoE 的核心 GEMM |
| `primus_turbo/triton/grouped_gemm/grouped_gemm_fp8_kernel.py` | FP8 Grouped GEMM |
| `primus_turbo/pytorch/kernels/moe/` | PyTorch 封装层 |
| `primus_turbo/pytorch/ops/moe/` | 高级 API |
| `primus_turbo/pytorch/modules/moe/token_dispatcher.py` | Token Dispatcher |
| `benchmark/ops/bench_grouped_gemm_turbo.py` | Grouped GEMM benchmark |
| `benchmark/ops/config.py` | MoE 模型配置 |

## MoE 数据流

```
Input Tokens [S, H]
      │
      ▼
┌─────────────┐
│ Router      │  fused_scaling_group_sum_routing_kernel
│ (Top-K)     │  → scores, topk_idx, routing_map
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Dispatch    │  Token permutation → 按 expert 重排
│ (Permute)   │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Expert GEMM │  Grouped GEMM: Gate-UP projection
│ (Gate+UP)   │  [tokens_per_expert, 2*intermediate, hidden]
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Activation  │  SwiGLU / GeGLU
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Expert GEMM │  Grouped GEMM: Down projection
│ (Down)      │  [tokens_per_expert, hidden, intermediate]
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Combine     │  Un-permute + weighted sum
│ (Unpermute) │
└─────┬───────┘
      │
      ▼
Output [S, H]
```

## 当前实现概况

- **Router**: Fused scoring + group TopK (sigmoid/softmax)
- **Grouped GEMM**: Persistent kernel (BF16/FP8)，各 expert 的 M 可不等
- **Token Dispatch**: PyTorch 层面的 permutation
- **模型覆盖**: DeepSeek-V2/V3, Mixtral-8x7B/22B, Qwen3, Grok-2, Kimi-K2, MoE-1T

## 优化方向（按优先级）

### P0: Fused MoE E2E Kernel

**来源**: vLLM `fused_moe.py`, AITER `moe_op_e2e.py`

将 dispatch → GEMM → activation → GEMM → combine 融合为单个或两个 kernel:

**方案 A: vLLM 式 Fused MoE**
- 一个 Triton kernel 处理: gather tokens → GEMM1 → activation → GEMM2 → scatter
- 按 expert 分组，每组 token 连续处理
- 关键: `moe_align_block_size` 确保每个 expert 的 token 数对齐到 BLOCK_M

**方案 B: AITER 式 2-Stage MoE**
- Stage 1: Gate-UP projection (所有 expert 的 GEMM)
- Stage 2: Activation + Down projection
- 通过 tuned CSV 选择最优配置

**收益**: 减少 dispatch/combine 的 HBM 读写，消除中间张量分配。

### P1: Grouped GEMM 优化 (MoE 核心路径)

Grouped GEMM 是 MoE 的性能关键:

**不均衡 M 处理**:
- DeepSeek-V3 有 256 个 expert，top-8 选择，各 expert 分到的 token 数高度不均
- 当前 persistent kernel 按组划分 tile，但 load imbalance 可能浪费 CU
- AITER 方案: 将所有组的 tile 展平，按全局 tile ID 调度

**Gate-UP Fused (G1U1)**:
- Gate 和 UP projection 共享输入，可合并为一个 GEMM: [M, 2*N, K]
- 减少一次 input 的 HBM 读取
- 当前已部分实现 (config.py 中 `2 * moe_intermediate_size`)

### P1: Router Kernel 优化

`fused_scaling_group_sum_routing_kernel` 当前用法:
- 对 expert 维度做 sigmoid/softmax + group topk
- 当 expert 数量大 (256/384) 时，E_ALIGNED 可能导致过多 padding
- 优化: 分块处理 expert 维度，减少 padding 浪费

### P2: 通信重叠 (Expert Parallel)

**来源**: AITER Iris, Primus-Turbo DeepEP

- all-to-all dispatch 可与上一层计算重叠
- GEMM + reduce-scatter 融合
- 需要跨 kernel 流编排，超出单 kernel 优化范围

## 关键模型配置

| 模型 | Experts | TopK | Intermediate | Hidden |
|------|---------|------|-------------|--------|
| DeepSeek-V3 | 256 | 8 | 2048 | 7168 |
| Kimi-K2 | 384 | 8 | 2048 | 7168 |
| MoE-1T | 224 | 8 | 1920 | 8192 |
| Mixtral-8x7B | 8 | 2 | 14336 | 4096 |
| Qwen3-235B | 128 | 8 | 4096 | 4096 |

注意: 大 expert 数 (≥128) 和小 expert 数 (8) 的优化策略截然不同。

## 标准验证流程

```bash
# 1. 编译
pip3 install --no-build-isolation -e . -v

# 2. 精度验证
pytest tests/pytorch/ops/test_grouped_gemm.py -x
pytest tests/pytorch/ops/test_grouped_gemm_fp8.py -x
pytest tests/pytorch/ops/test_fused_moe_router.py -x
pytest tests/pytorch/ops/test_permutation.py -x

# 3. 性能 benchmark
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON python3 benchmark/ops/bench_grouped_gemm_turbo.py
```

## 参考
- vLLM Fused MoE: `ref/OP_optimize/vllm/vllm/model_executor/layers/fused_moe/`
- AITER MoE: `ref/OP_optimize/aiter/aiter/ops/moe_op.py`
- AITER E2E MoE: `ref/OP_optimize/aiter/aiter/ops/triton/moe/moe_op_e2e.py`
- CK MoE 2-stage: `ref/OP_optimize/aiter/csrc/ck_gemm_moe_2stages_codegen/`
