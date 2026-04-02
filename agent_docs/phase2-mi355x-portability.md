# Phase 2 优化在 MI355X (gfx950) 上的适用性分析

## 结论

**部分有效，核心 tile 降级逻辑需要移植到 gfx950 代码路径。**

## 逐项分析

### 1. 动态 Tile 降级 (核心优化)

```
gfx950 路径:  ❌ 未生效 — tile 降级代码在 `else` (MI300X) 分支内
gfx942 路径:  ✅ 已生效
```

**原因:** `_get_gg_bf16_fwd_config` 和 `_get_gg_bf16_vk_config` 的 `if _is_gfx950():` 分支没有 tile 降级逻辑。

**是否需要移植:** **需要**。MI355X 有 256 CU (vs MI300X 304 CU)，tile 不足问题同样存在：

| 配置 (256×256 blocks) | MI300X (304 CU) | MI355X (256 CU) | 状态 |
|------------------------|:-:|:-:|:---:|
| DSv2-Lite B=2 M=512 | 44 tiles / 304 = **14%** | 44 tiles / 256 = **17%** | ❌ 都严重不足 |
| Qwen3-30B B=4 M=512 | 128 tiles / 304 = **42%** | 128 tiles / 256 = **50%** | ⚠️ MI355X 稍好 |
| DSv3 B=8 M=512 | 256 tiles / 304 = **84%** | 256 tiles / 256 = **100%** | ✅ MI355X 刚好够 |

MI355X 256 CU 使得 60% 阈值 = 154 tiles (vs MI300X 的 182)。有些在 MI300X 需要降级的 shape 在 MI355X 可能不需要，但 DSv2-Lite 等小 shape 仍然需要。

### 2. avg_m 修正 (移除 max(..., 256))

```
gfx950 路径:  ✅ 已生效 — 修改在 launch 函数中，两个路径共享
gfx942 路径:  ✅ 已生效
```

`grouped_gemm_triton_kernel()` 中 `avg_m = M_total // max(G, 1)` 影响传入 config 函数的参数，两个架构都受益。

### 3. num_warps 动态选择

```
gfx950 路径:  ✅ 已生效 — return 语句在 if/else 之后，两个路径共享
gfx942 路径:  ✅ 已生效
```

`num_warps = 4 if BLOCK_M * BLOCK_N <= 128 * 128 else 8` 在最终 return 前，对所有架构生效。但在 gfx950 上只有当 origami 选择了 128×128 时才会触发 (条件 `min(om, on) >= 128` 允许)。

### 4. Variable-K Backward Origami 补全

```
gfx950 路径:  ❌ 未受影响 — gfx950 已有 origami 调用
gfx942 路径:  ✅ 新增的 origami + tile 降级
```

gfx950 的 VK backward 原本就有 origami，但同样缺少 tile 降级。

### 5. GEMM 分析发现 (hipBLASLt 弱点)

```
gfx950 路径:  ✅ 适用 — AutoTune 机制与架构无关
gfx942 路径:  ✅ 适用
```

`PRIMUS_TURBO_AUTO_TUNE=1` 在任何架构上都可用。hipBLASLt 在非标准 N 维度上的弱点在 MI355X 上可能同样存在 (需实测验证)。

## 汇总

| 优化项 | MI300X (gfx942) | MI355X (gfx950) | 移植工作量 |
|--------|:---:|:---:|:---:|
| Tile 降级 (forward) | ✅ | ❌ | 中 (复制逻辑到 gfx950 分支) |
| Tile 降级 (VK backward) | ✅ | ❌ | 中 |
| avg_m 修正 | ✅ | ✅ | 0 |
| num_warps 动态 | ✅ | ✅ | 0 |
| GEMM AutoTune | ✅ | ✅ | 0 |

## 移植建议

在 `_get_gg_bf16_fwd_config` 的 gfx950 分支末尾添加：

```python
if _is_gfx950():
    # ... existing origami logic ...

    # [NEW] Tile downgrade for CU underutilization
    total_tiles = _estimate_total_tiles(avg_m, N, G, BLOCK_M, BLOCK_N)
    cu_count = num_sms  # 256 for MI355X
    if total_tiles < cu_count * 3 // 5:
        for bm, bn in [(128, 256), (256, 128), (128, 128)]:
            cand_tiles = _estimate_total_tiles(avg_m, N, G, bm, bn)
            if cand_tiles >= cu_count:
                BLOCK_M, BLOCK_N = bm, bn
                group_m = min(8, max(2, group_m))
                break
```

### 注意事项

1. **LDS 容量差异:** MI355X 有 160KB LDS (vs 64KB)，128×128 tiles 绝对不会溢出
2. **MFMA 差异:** gfx950 的 MFMA 指令可能对 tile shape 有不同偏好，需实测验证
3. **BLOCK_K 差异:** gfx950 TN 用 `BLOCK_K=64`，非 TN 用 `BLOCK_K=32`，降级时应保持 BLOCK_K 不变
4. **建议:** 在 MI355X 上移植后，用同样的 `bench_gg_ck_vs_triton.py` 脚本验证
