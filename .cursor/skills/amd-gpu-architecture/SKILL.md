---
name: amd-gpu-architecture
description: >-
  AMD Instinct GPU (MI300X/MI350X) 的架构知识和性能优化要点。
  当用户讨论 GPU 硬件特性、分析性能瓶颈、需要硬件级优化建议、
  或涉及 CK/hipBLASLt 后端选择时使用此技能。
---

# AMD GPU 架构知识

## 硬件规格速查

### MI300X (CDNA3, GFX942)

| 参数 | 值 |
|------|-----|
| Compute Units (CU) | 304 |
| 流处理器 | 19456 |
| HBM3 容量 | 192 GB |
| HBM3 带宽 | 5.3 TB/s |
| Infinity Cache | 256 MB |
| 峰值 BF16 | 1307.4 TFLOPS |
| 峰值 FP8 | 2614.9 TFLOPS |
| LDS / CU | 64 KB |
| 寄存器 / CU | 512 KB (256 VGPR × 64 × 32bit) |
| Wavefront Size | 64 |
| Max Waves / CU | 32 (occupancy 上限) |
| TDP | 750W |

### MI350X (CDNA4, GFX950) — 新一代

| 参数 | 值 |
|------|-----|
| HBM3E 容量 | 288 GB |
| HBM3E 带宽 | ~8 TB/s |
| 峰值 FP8 | ~4x MI300X |
| 新增特性 | MXFP4/MXFP8 原生支持 |

## 核心概念

### Wavefront vs Warp
- AMD Wavefront = **64 线程**同步执行
- NVIDIA Warp = 32 线程
- 影响: reduction 操作、shuffle 指令、block size 选择都需要考虑 64 的倍数

### 内存层次

```
HBM3 (5.3 TB/s, 192GB)
  └── Infinity Cache (256 MB, ~高带宽)
       └── L1 Cache / LDS (64 KB/CU, ~高带宽低延迟)
            └── VGPR 寄存器 (512 KB/CU, ~最快)
```

### Matrix Core (MFMA 指令)
- 等价于 NVIDIA Tensor Core
- 通过 MFMA (Matrix Fused Multiply-Add) 指令访问
- 支持形状: 16×16×16, 32×32×8 等（取决于数据类型）
- Triton 中通过 `tl.dot` 自动映射到 MFMA

### LDS (Local Data Share)
- 等价于 NVIDIA Shared Memory
- 每 CU 64KB，所有 wavefront 共享
- 32 banks，bank conflict 规则与 NVIDIA 类似
- Triton 中通过 `tl.load` 到 block 变量隐式使用

## 性能优化检查清单

### 内存受限场景
- [ ] 数据访问是否合并（连续线程访问连续地址）
- [ ] 是否可以使用向量化加载（一次加载 128bit / 256bit）
- [ ] LDS 是否有 bank conflict
- [ ] 是否充分利用 Infinity Cache（数据复用模式）

### 计算受限场景
- [ ] 是否在使用 Matrix Core（通过 tl.dot / MFMA）
- [ ] 占用率是否足够（waves/CU）
- [ ] 寄存器压力是否导致 spilling
- [ ] 是否有不必要的类型转换开销

### 后端选择指南

| 场景 | 推荐后端 | 理由 |
|------|----------|------|
| GEMM BF16/FP16 | hipBLASLt | 厂商深度优化 |
| GEMM FP8 tensorwise | hipBLASLt 或 CK | 视具体 shape |
| GEMM FP8 rowwise/blockwise | CK 或 Triton | hipBLASLt 支持有限 |
| Grouped GEMM | CK | CK 有专门的 grouped gemm 优化 |
| Attention | Triton 或 AITER | 灵活性 vs 性能 |
| Activation/Reduce | Triton | 简单算子 Triton 足够 |

## 详细参考
- CDNA3 架构详情: [cdna3-reference.md](cdna3-reference.md)
