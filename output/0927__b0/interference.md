# B0 并行干扰探针结论（2026-09-27）

prod 形状，op-evolve 同款 harness。fwd：r4 vs ASM(beat)，101 iters；bwd：r20(current) vs ASM，51 iters。
比值 = ASM ms / FlyDSL ms（= FlyDSL/ASM 的 TF/s 比）。

| 场景 | fwd r4 ms | fwd ASM ms | fwd 比值 | bwd r20 ms | bwd ASM ms | bwd 比值 |
|---|---|---|---|---|---|---|
| 单卡（A1,S1,S2） | 1.491-1.503 | 1.176-1.188 | 0.787-0.796 | 8.662-8.701 | 6.413-6.458 | 0.737-0.742 |
| fwd(g0)+bwd(g1) 同时（P1,P2） | 1.491-1.499 | 1.178-1.183 | 0.789-0.790 | 8.695-8.707 | 6.428-6.435 | 0.739 |
| 4 卡同时跑 attention（Q1） | 1.482-1.500 | 1.167-1.188 | 0.787-0.792 | 8.603-8.846 | 6.425-6.583 | 0.744-0.747 |
| 其余 3 卡跑持续 bf16 GEMM（B1） | **16.82** | **16.73** | 0.995 | **24.29** | **37.73** | 1.553 |

- attention 负载之间互不干扰：比值和绝对值的变化都在单卡自身的波动范围内（约 0.5-1%）。**fwd/bwd 两个 job + 两个 lab 可以 4 卡并行。**
- 邻卡跑持续 GEMM burn 时，被测卡慢 3-11 倍，而且比值会反转。此时被测卡 busy 100%、sclk 约 2 GHz，属于 GPU 侧停顿，原因未知。**规则：任何卡上都不许跑持续的 GEMM 压测或长时间 hipBLASLt GEMM 循环。**
- 原始数据：`probe/*.log`、`*.json`；脚本：`probe.sh`（GEMM burn）、`probe2.sh`（solo/pair）、`probe4.sh`（4 卡）。
