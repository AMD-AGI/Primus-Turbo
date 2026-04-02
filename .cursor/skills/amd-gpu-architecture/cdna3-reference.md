# CDNA3 (MI300X) 架构详细参考

## Compute Unit (CU) 内部结构

每个 CU 包含:
- 4 个 SIMD 单元（每个执行一个 wavefront 的 16 个线程/cycle）
- 1 个 Scheduler（每 cycle 发射 1 条指令到 1 个 SIMD）
- 1 个 LDS (64KB)
- 1 组 VGPR (Vector General Purpose Registers)
- 1 组 SGPR (Scalar General Purpose Registers)
- Matrix Core 单元（MFMA 指令执行）

## MFMA 指令详情

### BF16 MFMA 指令
| 指令 | 输入形状 | 输出形状 | 输出类型 | Cycles |
|------|----------|----------|----------|--------|
| `v_mfma_f32_32x32x8_bf16` | 32×8 × 8×32 | 32×32 | FP32 | 64 |
| `v_mfma_f32_16x16x16_bf16` | 16×16 × 16×16 | 16×16 | FP32 | 32 |

### FP8 MFMA 指令 (CDNA3)
| 指令 | 输入形状 | 输出形状 | 输出类型 | Cycles |
|------|----------|----------|----------|--------|
| `v_mfma_f32_32x32x16_fp8` | 32×16 × 16×32 | 32×32 | FP32 | 64 |
| `v_mfma_f32_16x16x32_fp8` | 16×32 × 32×16 | 16×16 | FP32 | 32 |

## 占用率 (Occupancy) 计算

每个 CU 的资源限制:
- VGPR: 512 KB total → 每个 wavefront 最多 256 个 VGPR
- LDS: 64 KB total
- Max wavefronts: 32

占用率 = 实际 waves / 最大 waves

影响因素:
1. **VGPR 使用量**: 每个线程用的 VGPR 越多，能同时运行的 wave 越少
   - 128 VGPRs → 最多 8 waves/SIMD
   - 256 VGPRs → 最多 4 waves/SIMD
2. **LDS 使用量**: workgroup 使用的 LDS 越多，能同时运行的 workgroup 越少
3. **Workgroup 大小**: 影响 wave 的分组

## 内存带宽分析

### 理论带宽
- HBM3: 5.3 TB/s
- LDS: ~每 CU 约 32 TB/s (理论峰值)

### 实际可达带宽
- HBM3 连续读取: ~4.5-5.0 TB/s (85-95% 效率)
- HBM3 随机读取: 大幅下降，取决于访问模式
- LDS 无 bank conflict: 接近理论峰值
- LDS 有 bank conflict: 按冲突度线性下降

### Arithmetic Intensity 分析
- BF16 MFMA 峰值: 1307.4 TFLOPS
- HBM3 带宽: 5.3 TB/s
- 平衡点: 1307.4 / 5.3 ≈ **247 FLOP/Byte**
- 低于此值 → 内存受限，高于此值 → 计算受限

## Profiling 工具

### rocprof
```bash
# 基础 kernel 计时
rocprof --stats python3 your_script.py

# 硬件计数器
rocprof -i counters.txt python3 your_script.py
```

### omniperf (推荐)
```bash
# 收集数据
omniperf profile -n my_run -- python3 your_script.py

# 分析
omniperf analyze -p workloads/my_run/
```

关键指标:
- **VALU Utilization**: 向量 ALU 利用率
- **MFMA Utilization**: Matrix Core 利用率
- **LDS Bandwidth**: LDS 实际带宽
- **HBM Bandwidth**: HBM 实际带宽
- **Occupancy**: 实际占用率
- **Wavefront Stalls**: 停顿原因分析
