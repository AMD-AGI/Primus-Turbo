# Profiling 总结 — round 5，op/current（`kn_fmha_fwd_prefill_a16w16_m32x8_bshd`）

> **Benchmark 说明。** 以下所有速率都来自 `benchmark.py` 的一次**未开 profiler** 的运行：每组取 101 次调用的中位数，每次计时前 flush L2，arm 按回文顺序执行，`beat`（aiter gfx1250 ASM forward）在同一次运行中测量。本文不使用任何 profiler 下得到的速率。这张卡受 VR 限频：sclk DPM 最高 1100 MHz，prod 持续频率约 988–1030 MHz。

## Benchmark

| shape | (B, Sq, Skv, Hq, Hkv, D) | current TFLOP/s | beat TFLOP/s | current/beat | sclk MHz |
|---|---|---:|---:|---:|---|
| fast | (1, 1024, 1024, 8, 2, 128) | 38.27 | 69.96 | 0.547 | 1100 |
| proxy | (1, 4096, 4096, 32, 8, 128) | 703.21 | 1008.13 | 0.698 | 1027–1062 |
| prod | (4, 8192, 8192, 32, 8, 128) | 1083.33 | 1410.87 | **0.768** | 1019–1031 |

详情见 [benchmark-results.md](benchmark-results.md)。该 kernel 占 prod 时间的 94.4%、fast 时间的 85.6%（[kernel.yaml](kernel.yaml)）。

## 结论

- **prod：`latency`，置信度 low。** 瓶颈在片上执行效率，不能排除 compute/issue。
  - HBM 已排除：时钟变化时，每次调用的 cycles 保持不变。
  - 提高时钟最多能换回 +11%。
  - 在相同的卡状态下，beat 每秒完成的工作量是 current 的 1.30 倍。
  - 见 [6-bound-analysis](6-bound-analysis/analysis.md)、[5-power-wall-analysis](5-power-wall-analysis/analysis.md)。
- **fast：`latency`，置信度 medium-high。**
  - grid 填不满机器：32 个 WG 对 256 个 CU。
  - 单次调用和冷 cache 的开销大：warm 35.3 µs，flushed 56.2 µs。
  - 该 kernel 不受功耗限制。
  - 见 [6-bound-analysis](6-bound-analysis/analysis.md)。
- `bound: latency` 通过 `verdict.bound`（即 prod 的结论）写入 state.yaml。

## 候选方向（已排序；按 correction 4 优先 prod；都未定价，定价需上卡测量）

1. **c1（prod）：** KV 主循环的每 cycle 效率，重点是指令调度中的 WMMA 供数，以及 softmax VALU 与 WMMA 的重叠。与 beat 的差距中约 23% 不是时钟造成的。见 [6-bound-analysis](6-bound-analysis/analysis.md)、[benchmark-results.md](benchmark-results.md)。
2. **c2（prod）：** 每 cycle 的能耗。
   - 随机数据下时钟降到 988 MHz；全零数据下为 1053 MHz，时间减少 7.2%。
   - 降低翻转活动有双重收益：既减少 cycles，又抬高时钟。时钟部分的收益上限约 +11%。
   - 见 [5-power-wall-analysis](5-power-wall-analysis/analysis.md)。
3. **c3（fast）：** grid 填不满，只有 ≤12.5% 的 CU 有活干。小 grid 时可拆分 Q 行或 KV 来增加 WG 数，但不得让 prod 回退。见 [kernel.yaml](kernel.yaml)、[6-bound-analysis](6-bound-analysis/analysis.md)。
4. **c4（fast）：** 单次调用和冷 cache 的开销（warm→flushed 为 1.59 倍）。定价之前，需要先对 current 和 beat 都测 warm 与 flushed，这是最便宜的下一项测量。

Non-findings 见 [profiling.yaml](profiling.yaml) 的 `non_findings`：没有 spill；I-cache 不是因素；HBM 不是 prod 的瓶颈；prod 的 grid 能填满机器；仅靠 occupancy 解释不了差距；功耗没有到 socket 上限。

## 没有数据的部分

- **各级内存的字节数：** gfx1250 没有字节计数器，所以没有 roofline，L2/LDS 带宽既不能确认也不能排除。
- **WMMA 利用率：** 相关计数器读数为 0。
- **stall/issue 归因：** 已关闭，且禁止使用 PC sampling。
- **Thread trace：** [4-thread-trace](4-thread-trace/provenance.yaml) 已跳过，因为 ATT 看不到 FlyDSL kernel。
- **Panel 指标：** [3-kernel-metrics](3-kernel-metrics/provenance.yaml) 已跳过，因为 rocprof-compute 不可用。
- **计算峰值：** bf16 WMMA 每 cycle 的 FLOP 数没有文档。
- **VGPR 数：** 读数为 232，按 correction 1 约为 464，未从 ISA 确认。
- **fast 下 beat 的 warm 时间：** 未测量。

计数器证据（SQ busy、waves、I-cache）见 [2-kernel-profiling](2-kernel-profiling/kernel-kn_fmha_fwd_prefill_a16w16_m32x8_bshd/analysis.md)。
