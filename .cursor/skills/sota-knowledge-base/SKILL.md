---
name: sota-knowledge-base
description: >-
  汇总 SOTA 算子优化技术和参考实现。
  当 Agent 需要选择优化策略、查阅行业最佳实践、
  或对比不同技术方案时使用此知识库。
---

# SOTA 算子优化知识库

## 参考项目索引

| 项目 | 参考路径 | 核心价值 |
|------|----------|----------|
| **AITER** | `ref/OP_optimize/aiter/` | AMD 官方高性能算子库，Triton/CK/ASM 多后端 |
| **FlashAttention** | `ref/OP_optimize/flash-attention/` | FA2/FA3/FA4 算法演进 |
| **ThunderKittens** | `ref/OP_optimize/ThunderKittens/` | Tile-based 编程，exp2 softmax |
| **Liger-Kernel** | `ref/OP_optimize/Liger-Kernel/` | Triton 训练算子，kernel 融合 |
| **KernelBench** | `ref/OP_optimize/KernelBench/` | Kernel 性能评估框架 |
| **Triton** | `ref/OP_optimize/triton/` | AMD 后端编译器优化 |
| **vLLM** | `ref/OP_optimize/vllm/` | 推理服务，多后端 attention |

---

## Attention 优化技术

### 1. Block Size Autotune
- **来源**: AITER, Triton AMD backend
- **原理**: 不同 (seq_len, head_dim, batch) 组合下最优 BLOCK_M/N 不同
- **实现**: `triton.autotune` + 配置搜索 {32, 64, 128}
- **预期收益**: 10-30%
- **参考代码**: AITER `flash_attn_triton_amd/` JSON 配置文件

### 2. Causal Masking 三分法
- **来源**: FlashAttention FA3/FA4
- **原理**: KV 块按 causal 关系分为三类:
  - **完全 unmasked**: 无需 mask 计算，直接矩阵乘
  - **完全 masked**: 直接跳过，零计算
  - **部分 masked**: 完整 mask 逻辑
- **预期收益**: 15-25% (causal attention)
- **参考**: FA3 `hopper/mask.h`, AITER causal 路径

### 3. Branchless Rescaling (AVO)
- **来源**: AVO 论文
- **原理**: Online softmax 中 `m_new > m_old` 时需要 rescale O 累加器。
  分支版本导致 warp divergence; 无分支版本始终计算 rescale 因子，
  不需要时替换为 1.0
- **预期收益**: 5-10% (non-causal 场景最大)
- **Triton 实现思路**:
  ```python
  # 有分支 (当前)
  if m_new > m_old:
      alpha = tl.exp(m_old - m_new)
      acc *= alpha
  # 无分支 (优化)
  alpha = tl.exp(m_old - m_new)  # 当 m_new == m_old 时 alpha = 1.0
  acc *= alpha
  ```

### 4. exp2 域 Softmax
- **来源**: ThunderKittens
- **原理**: 预乘 scale × log₂(e)，用 exp2 代替 exp
- **数学**: scale_log2 = (1/√d) × log₂(e) ≈ 1.44269504 / √d
  S = Q·Kᵀ × scale_log2, 然后 exp2(S - m) 代替 exp(S/√d - m)
- **预期收益**: 3-8% (减少 transcendental 指令)
- **注意**: 需验证 Triton AMD 后端 exp2 是否映射到快速硬件指令

### 5. KV Double Buffering
- **来源**: ThunderKittens, FlashAttention FA3
- **原理**: 在计算当前 KV block 的同时预取下一个 KV block
- **Triton 实现**: 循环展开 2 次，交替使用两组 KV 寄存器/LDS
- **预期收益**: 5-15% (隐藏内存延迟)
- **注意**: 会增加寄存器压力，可能降低 occupancy

### 6. Lean Attention (Stream-K for Attention)
- **来源**: AITER `lean_atten.py`
- **原理**: 将 attention 的 KV 维度切分为多个 partition，
  不同 CU 处理不同 partition，最后归约 (类似 Split-K for GEMM)
- **预期收益**: 10-20% (decode 场景，short Q long KV)
- **注意**: 需要额外的 lock/barrier 机制做 partial softmax merge

### 7. SageAttention (INT8 Q/K)
- **来源**: AITER `fav3_sage.py`
- **原理**: Q/K 量化到 INT8 做矩阵乘，V 保持 FP16
- **预期收益**: 20-40% (QK 矩阵乘算力翻倍)
- **注意**: 精度影响较大，需按 arch (gfx942/gfx950) 调参

---

## GEMM 优化技术

### 1. M-Bucketed 配置
- **来源**: AITER
- **原理**: 按 M 值分段选配置 (M_LEQ_16, M_LEQ_256, M_GEQ_256 等)
- **参考**: AITER `aiter/ops/triton/README.md` JSON 配置格式

### 2. Split-K
- **来源**: AITER, Triton tutorials
- **原理**: 小 M 时 (M·N)/tile 不足以占满所有 CU，
  沿 K 维切分增加并行度
- **参考**: AITER `compute_splitk_params`

### 3. XCD Swizzle
- **来源**: AITER, 当前 gemm_kernel.py
- **原理**: MI300X 8 个 XCD，相邻 tile 分配到不同 XCD 提升 L2 效率
- **已实现**: 是（NUM_XCDS = 8），可验证 GROUP_M 是否各 shape 最优

### 4. Persistent Kernel
- **来源**: Triton tutorials, ThunderKittens
- **原理**: Grid 大小固定为 SM 数，每个 block 循环处理多个 tile
- **已实现**: 是，可优化调度策略 (Super-M 行)

### 5. Preshuffle Weight Layout
- **来源**: AITER CK
- **原理**: 离线重排 weight 矩阵，使其 layout 匹配 MFMA 读取模式
- **收益**: 减少运行时 layout 转换，5-10%

---

## MoE 优化技术

### 1. Fused MoE E2E
- **来源**: vLLM, AITER
- **详见**: `skills/moe-optimize/SKILL.md`

### 2. 2-Stage vs 1-Stage 自动选择
- **来源**: AITER CK `ck_gemm_moe_2stages_codegen/`
- **原理**: 1-stage 减少 kernel launch，2-stage 可分别优化每个 GEMM

### 3. Gate-Up Fused (G1U1)
- **来源**: AITER
- **原理**: Gate 和 UP projection 合并为单个 [M, 2N, K] GEMM

---

## 通用 Triton 优化技术

### 1. AMD HIP-specific num_warps
- **来源**: Liger-Kernel, vLLM
- **原理**: AMD wavefront=64，通常 num_warps=16 比 32 效果好
- **适用**: 所有 Triton kernel

### 2. int64 索引
- **来源**: AITER, Liger-Kernel
- **原理**: program_id × stride 可能溢出 int32 (大矩阵/长序列)
- **实现**: `tl.program_id(0).to(tl.int64)`

### 3. Cache Modifier
- **来源**: Liger-Kernel, Triton AMD backend
- **原理**: `.ca` (cache all), `.cs` (cache streaming), `.wb` (write-back)
- **适用**: 区分读一次 vs 读多次的数据

### 4. waves_per_eu
- **来源**: Triton AMD `compiler.py`
- **原理**: 控制每个执行单元的 wavefront 数，影响 occupancy vs 寄存器压力
- **典型值**: attention 用 2-3, GEMM 用 2

### 5. Triton AMD schedule_hint
- **来源**: Triton AMD backend
- **可选值**: `"attention"`, `"memory-bound-attention"`
- **原理**: 编译器级别的指令调度优化 (IGLP/sched barriers)

---

## AMD 架构限制提醒

- **无 TMA**: 数据搬运通过标准 load/store，无 NVIDIA TMA 加速器
- **无 WGMMA**: 矩阵乘通过 MFMA 指令，而非 NVIDIA warpgroup MMA
- **Wavefront = 64**: reduction / shuffle 粒度不同于 NVIDIA warp=32
- **LDS 64KB/CU**: 需合理分配给各 tile，bank conflict 32 banks
- **8 XCD**: 需要 swizzle 确保负载均衡
- **Infinity Cache 256MB**: 利用数据复用模式 (tiling L2 友好)
