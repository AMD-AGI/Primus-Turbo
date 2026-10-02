# 真实训练数据 op 级 A/B（2026-10-02 09:14，A0，驱动 7.1.0-2412954）

工具 `tools/realab.sh`（2 个卡进程，147 s 卡时，rc=0，dmesg 干净）；输入 = B0 step 43 dump 的 6 层真实 q/k/v（L00/01/02/08/16/31）+ randn；
bwd 的 o/lse 来自同一 q/k/v 上的 ASM fwd，dO 为固定种子 randn。fwd arm：asm / r16 / r16_aa；bwd arm：asm / s6(0.3.4.1) / r29 / s6_aa。
blk = job 尺子（blocked 9 + lead 4，palindromic，256 MB flush）；gb = 每次计时前 10 个 bf16 32768×4096×14336 GEMM（image hipBLASLt，~21.5 ms），即训练工作点。
日志 `runs/realab_driver_1002_*.txt`、`runs/realab_{blk,gb}_o1_1002_091405.{log,json}`。

| | blk ms（fly / asm） | blk fly/asm | gb ms（fly / asm） | gb fly/asm | gb 时钟 |
|---|---|--:|---|--:|--:|
| fwd r16，6 层真实（几何平均） | 1.308 / 1.209 | 1.081 | 1.535 / 1.137 | **1.349** | ~1.28 GHz |
| fwd r16，randn | 1.285 / 1.190 | 1.080 | | 1.343 | |
| bwd s6，6 层真实 | 5.324 / 5.475 | **0.972** | 5.743 / 5.530 | **1.038** | ~1.28 GHz |
| bwd s6，randn | 5.244 / 5.480 | 0.957 | | 1.022 | |
| bwd s6 / r29，真实 | | 0.806 | 5.743 / 7.421 | **0.774** | |

A/A：fwd fly_aa/fly 0.985 / 0.999（blk/gb），bwd s6_aa/s6 0.999 / 0.999。correctness：所有 set 的输出有限、无 CORR FAIL/WARN。

结论：真实数据与 randn 一致；但**训练工作点（GEMM burst 后，~1.28 GHz）**下 ASM 几乎不受影响，FlyDSL 两个方向都变慢：
bwd 从领先 2.8% 变为落后 3.8%，fwd 从落后 8% 变为落后 35%。s6 相对 r29 在训练工作点仍快 22.6%。
预期 e2e（32 层/步）：bwd 相对 r29 约 −53 ms/步，相对 ASM 约 +7 ms/步；fwd 相对 ASM 约 +13 ms/步。
