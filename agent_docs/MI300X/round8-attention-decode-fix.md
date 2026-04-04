# Round 8: FP8 Attention Decode Fix + BLOCK_N Investigation

## 优化概述

修复 FP8 Attention 的 `block_scaling_node` 以支持 decode 场景 (seqlen_q < BLOCK_M)。
同时调研了 BLOCK_N=128 对 decode 性能的影响。

## 修改文件

| 文件 | 变更 |
|------|------|
| `primus_turbo/pytorch/ops/attention/attention_utils.py` | `block_scaling_node` 添加 L 对 BLOCK_M 的 padding |

## Bug Fix: block_scaling_node

### 问题

```python
# 原始代码
tensor = tensor.reshape(B, H, L // BLOCK_M, BLOCK_M, D)
# 当 L < BLOCK_M (如 decode seqlen_q=1) 时, L // BLOCK_M = 0
# → RuntimeError: shape '[1, 32, 0, 64, 128]' is invalid
```

### 修复

```python
padded_L = ((L + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
if padded_L != L:
    tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padded_L - L))
num_blocks = padded_L // BLOCK_M
tensor = tensor.reshape(B, H, num_blocks, BLOCK_M, D).reshape(...)
# 量化后截断回原始长度
tensor = tensor.reshape(B, H, padded_L, D)[:, :, :L, :].permute(...)
```

## 正确性验证

### Decode 场景（新增可用）

| seqlen_q | Output Shape | max_diff | cosine_sim | 状态 |
|----------|-------------|----------|------------|------|
| 1 | [2,1,8,128] | 0.0088 | 0.9961 | **PASS** |
| 4 | [2,4,8,128] | 0.0084 | 1.0000 | **PASS** |
| 16 | [2,16,8,128] | 0.0117 | 0.9961 | **PASS** |
| 32 | [2,32,8,128] | 0.0166 | 1.0000 | **PASS** |
| 64 | [2,64,8,128] | 0.0117 | 1.0000 | **PASS** |

### 回归验证

现有 FP8 attention 测试结果与基线完全一致：
- 18 passed, 42 failed（42 failures 为预存在的边缘 SNR 问题，与本修改无关）

## BLOCK_N=128 调研（已撤回）

### 假设

对 decode (seqlen_q ≤ BLOCK_M)，将 BLOCK_N 从 64 增大到 128 可减半 KV 内循环迭代次数，
从而减少 loop overhead 并提升内存带宽利用率。

### 实测结果

| 场景 | BLOCK_N=64 | BLOCK_N=128 | 差异 |
|------|-----------|------------|------|
| B=1 sq=1 sk=2K | 1.128 ms | 1.133 ms | +0.4% |
| B=1 sq=1 sk=4K | 1.106 ms | 1.139 ms | +3.0% |
| B=1 sq=1 sk=8K | 1.125 ms | 1.161 ms | +3.2% |
| B=4 sq=1 sk=4K | 1.311 ms | 1.306 ms | -0.4% |
| B=4 sq=64 sk=4K | 1.320 ms | 1.316 ms | -0.3% |

### 结论

BLOCK_N=128 在 decode 场景无明显收益（差异在 ±3% 噪声范围内），
部分 case 有轻微回退。原因分析：
1. Decode 受 **autotuned kernel 参数** (num_stages, num_warps) 主导，非 BLOCK_N
2. 总 KV 数据量相同，内存带宽是瓶颈
3. BLOCK_N=128 可能增加寄存器压力，降低 occupancy

**已撤回 BLOCK_N 修改**，仅保留 block_scaling_node bug fix。
