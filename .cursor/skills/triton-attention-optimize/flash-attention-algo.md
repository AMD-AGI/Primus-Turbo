# FlashAttention 算法参考

## 核心算法：Online Softmax + 分块计算

### 标准 Attention
$$O = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

朴素实现需要物化 $N \times N$ 的分数矩阵 $S = QK^\top$，内存复杂度 $O(N^2)$。

### FlashAttention 分块算法

将 Q 按行分块（BLOCK_M），K/V 按列分块（BLOCK_N），逐块计算：

```
对每个 Q-block i:
    初始化: O_i = 0, m_i = -inf, l_i = 0
    对每个 K-block j:
        S_ij = Q_i @ K_j^T / sqrt(d)          # 局部分数
        m_new = max(m_i, rowmax(S_ij))         # 更新 running max
        P_ij = exp(S_ij - m_new)               # 局部 softmax 分子
        l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)  # 更新 running sum
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j      # 重缩放并累加
        m_i = m_new, l_i = l_new
    O_i = O_i / l_i                            # 最终归一化
```

### 关键优化点

1. **累加器重缩放 (Rescaling)**
   - 当 `m_new > m_i` 时需要重缩放 O_i
   - AVO 发现：无分支推测路径（始终计算 rescale 因子，不需要时替换为 1.0）可消除 warp 同步开销
   - Non-causal 场景收益最大（+8.1%）

2. **Causal Masking**
   - 完全 masked 的 K-block（所有 entry 都被 mask）可直接跳过
   - 完全 unmasked 的 K-block 无需 mask 计算
   - 仅部分 masked 的 K-block 需要完整 mask 逻辑

3. **双 Q-stage (Dual Q-stage)**
   - FA4 同时处理两个 Q-tile，通过 warp specialization 重叠计算
   - Triton 中可通过增大 BLOCK_M 近似实现类似效果

## FlashAttention 版本演进

| 版本 | 核心改进 | 硬件目标 |
|------|----------|----------|
| FA1 | 分块 + online softmax，避免物化 N×N 矩阵 | A100 |
| FA2 | 更好的并行化（沿 seq_len 维度）、减少非矩阵乘 FLOPs | A100/H100 |
| FA3 | Warp specialization、异步数据搬运 (TMA)、FP8 支持 | H100 |
| FA4 | Dual Q-stage、Blackwell 特化、bitmask causal | B200 |

## AMD GPU 上的 Attention 优化要点

- 无 TMA（Tensor Memory Accelerator），数据搬运通过标准 load/store
- Wavefront = 64，影响 reduction 操作的粒度
- LDS 64KB/CU，需要合理分配给 Q/K/V/S 各 tile
- MFMA 指令支持的矩阵形状与 NVIDIA HMMA 不同
- CK 库中有高度优化的 attention 实现可作为参考
