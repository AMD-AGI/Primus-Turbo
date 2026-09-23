# 阶段 2 · 前向 —— KV 块宽度扫描（2026-09-23）

计划里 S1 的头号候选是「`n_block` 64 → 128 →（评估 256）」，理由是
`MIN_KV_BLK_BYTES = 64 KB` 已经把 LDS 地板抬好、提到 256 一个字节额外 LDS 都不要，
而 **ASM 的 tile 恰恰就是 `128x256`**。

**用一个 10 分钟的同会话实验把它打死了**，而不是先花 58 分钟建作业再发现方向不对。

## 结果

同会话、轮转交错、每 rep 256 MiB L2 flush、n=20、先验 SQNR 后计时。

| 臂 | median | TF/s | SQNR | vs 出货 |
|---|--:|--:|--:|--:|
| aiter ASM（bar） | **1.576 ms** | 1416.9 | 53.62 dB | — |
| FlyDSL n_block=64（出货） | 2.400 ms | 930.7 | 51.63 dB | 1.000× |
| FlyDSL n_block=128 | 7.741 ms | 288.5 | 51.66 dB | **0.310×** |
| FlyDSL n_block=256 | 23.753 ms | 94.0 | 51.69 dB | **0.101×** |

**128 慢 3.2 倍，256 慢 9.9 倍。** 死因不是 LDS——LDS 确实一字节没涨，计划那半是对的——
是 **VGPR**。LDS 已经把占用率钉死在 1 WG/CU，8 wave/WG 摊到 4 个 SIMD 就是 2 wave/SIMD，
而 n_block=128 下每 wave 约 584 个 VGPR，2 × 584 = 1168 > 1024，直接溢出到 scratch。

## 第一次跑出来的是个空实验，不算数

第一版 harness 报出三个臂**时间与 SQNR 完全相同**、后两次「构建」耗时 **0.0 s**。
那不是「n_block 无所谓」，那是 **n_block 根本没传进去**：

- `n_block: int = DEFAULT_N_BLOCK` 是**关键字默认值，在 def 时就绑定**，事后改模块常量无效；
- `build_fmha_fwd_prefill_a16w16_m32x8` 上面有 `@functools.cache`，而 `_ensure_bshd_kernel`
  每次传的参数完全相同 → 直接返回同一个 kernel。

改成直接改 `__wrapped__.__kwdefaults__` 并 `cache_clear()`，才出现了真实的构建耗时
（0.1 / 4.1 / 8.2 s）和真实各异的 SQNR（51.63 / 51.66 / 51.69）。
harness 现在带两道绊线：构建耗时 < 0.5 s 报 SUSPICIOUS，所有变体是同一对象则直接中止。

> 规矩：**一个「无差别」的结果，先怀疑实验没生效，再怀疑机制没效果。**
> 三个不同 tile 尺寸给出小数点后两位完全相同的 SQNR，是不可能的巧合。

## 但它指出了正确的形状

| | Q tile | KV tile |
|---|--:|--:|
| aiter ASM bar | **128** | **256** |
| FlyDSL 出货 | 256 | 64 |

bar 不是「KV 块更大」，是**用减半的 Q tile 换来 4 倍的 KV 块**。
单独放大 KV 必然撞寄存器墙，因为 Q 那边的 O 累加器一分不让。

所以下一个实验是**联动**：`WMMA_ROW_PER_WAVE` 2 → 1（`BLOCK_M` 256 → 128，O 与 S 累加器同时减半）
与 `n_block` 一起扫，逼近 bar 的长宽比。
