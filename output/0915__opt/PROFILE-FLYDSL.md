# Profile 验证：转置确实消失了

日期 2026-09-16 · `repro_l8b_turbo_conv_8L_prof.yaml` · nkfix 规则 1 + 规则 3 + ASM 反向

## 先说方法上的限制

上一版 profile（`PROFILE-POST-NKFIX.md`）是用**另一个分桶脚本**产出的，而旧的 trace 已被本次
运行覆盖。**跨脚本比较占比是不可靠的**，所以下面只报这份 trace 自身能证明的东西，
不与旧数字做逐项对照。

## 这份 trace（`iteration_10`，GPU kernel 合计 691.9 ms，1659 个 kernel）

| 桶 | ms | 占比 | n |
|---|--:|--:|--:|
| GEMM | 437.8 | 63.3% | 171 |
| elementwise/copy | 119.3 | 17.2% | 951 |
| attention | 90.5 | 13.1% | 32 |
| other | 44.3 | 6.4% | 505 |

## 待检验的说法

"FlyDSL 之所以赢，是因为那 137 ms 的操作数拷贝不再发生，而不只是它的 GEMM 更快。"
这在此前一直是**推断**。

## 三条直接证据

1. **FlyDSL 的 kernel 确实在承担 wgrad**，且 tile 与离线表逐一对应：

   | kernel | ms | n |
   |---|--:|--:|
   | `pt_gemm_gfx1250_bf16_t256x128x32_w2x4_nb2_g8` | 85.6 | 16 |
   | `pt_gemm_gfx1250_bf16_t256x128x32_w4x2_nb2_g8` | 43.9 | 1 |
   | `pt_gemm_gfx1250_bf16_t128x128x32_w2x2_nb2_g8` | 43.8 | 8 |
   | `pt_gemm_gfx1250_bf16_t128x256x32_w2x4_nb2_g8` | 22.4 | 16 |

   `w4x2` 那个 n=1，正是 lm_head（表里 M=128256 调出的就是 `[256,128,32] mw=4 nw=2`）。

2. **规则 2 的拷贝一次都没发生**：`wgrad: 0, wgrad_flydsl: 570, declined: 0, no_config: 0`。

3. **没有任何一个 kernel 还像原来那种大转置**。此前记录的 nkfix 转置是 **187.05 ms**；
   这份 trace 里最大的 elementwise kernel 是 **15.8 ms 分摊在 57 次调用上（单次 0.28 ms）**。
   残留的 119.2 ms 是普通长尾：dtype 转换（`bfloat16_copy` 13.2 ms / 148 次，
   `bfloat16to` 11.0 ms / 148 次）、二元算子、激活。

## 顺带确认规则 1 仍然有效

最大的单个 kernel 是 `Cijk_Alik_Bljk_..._MT256x256x128_MI16x16x1_SN_LDSB`，231.7 ms / 114 次 ——
**`MT256x256x128` 正是我们要的那个 tile**（而不是缺陷路径上的 `MT32x16x32` GEMV tile），
且 `Alik_Bljk` 是 TN，说明 dgrad 的重写照常生效。

## 结论

机制说法从推断变成证实：**转置消失了，不是被摊薄了。**
