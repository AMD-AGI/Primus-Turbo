# e2e Llama-3.1-8B 32L（A0，2026-10-02 10:10–10:21，驱动 7.1.0-2412954）

driver `e2e/drive.sh`（MODE=3arm，RUNS="opc opcprod p1 p2"）：opcheck fast（串行化）与 prod（side stream 并发）门槛 PASS（50.8–52.8 dB）；
两个训练进程各 92 步、三个 arm 交替（p1：asm;asm,fly,flyr29,asm,flyr29,fly；p2 反序），全部 rc=0，0 非有限、0 BLAS 改写、0 GPU 故障、显存峰值 88.99%。
完整表：`e2e/runs/TABLE.1002_095504.md`；分析：`e2e/runs/analysis.1002_095504.txt`。

| arm | 单步 ms（p1 / p2） | 对 ASM（相邻配对） | attn fwd ms/步 | attn bwd ms/步 |
|---|---|--:|--:|--:|
| asm | 1350.6 / 1350.6 | 1 | 34.7 | 166.5 |
| flyr29 = fwd r16 + bwd r29 | 1389.6 / 1388.5 | 1.0269 / 1.0273 | 43.9 | 214.8 / 215.4 |
| **fly = fwd r16 + bwd s6** | **1349.4 / 1349.9** | **0.9990 / 0.9985** | 43.9 | **162.2 / 162.6** |

- s6 vs r29：单步 −39.1 / −37.0 ms（0.972），attention bwd −53.1 / −52.9 ms/步，与真实数据 op 级预测（−53 ms/步）一致。
- fly vs ASM：单步持平（B0 09-28 为 1.032）；bwd 比 ASM 每步快 4.5 ms，fwd 每步慢 9.2 ms。剩余差距在 fwd。
- 训练时 sclk 中位数约 1.50 GHz（介于 op 级 blk ~1.45–1.5 与 gb ~1.28 之间）。
- kineto trace 在 A0 上仍无效（每步 4–15 个 kernel），fwd/bwd 拆分依据 CUDA event。
