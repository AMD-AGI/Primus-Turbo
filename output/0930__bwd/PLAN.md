# bwd 追平/超越 aiter ASM：计划与决策记录（2026-09-30 起，A0 heliosr-1b114-c07-1，分支 dev/lhz/flydsl-attn-b0）

起点（A0 09-30 blocked 尺子，randn，`0928__a0_repro/REPORT.md` §6.2）：r29 6.583 ms / ASM 5.503 ms = **83.5%**。
cycle 视角（`0930__roofline/REPORT.md` §6）：fly 11.44e6 cyc/SIMD（k_dkdv 7.26e6 + k_dq 4.00e6 + k_delta 0.18e6）vs ASM 7.71e6；
k_dkdv 每轮 1766 cyc = WMMA 512 + 发射 ~450 + WMMA↔DS 切换 ~200（7 次）+ 访存等待 ~600（1 wave/SIMD 无法遮盖）。
目标：时间 ≤ 5.50 ms（~-17%），即 cycle 降到 ~9.5e6 以下（fly 时钟高 ~8%）。

## 阶段

| 阶段 | 内容 | 上卡 | 风险 |
|---|---|---|---|
| P0 | 工作区 `output/0930__bwd/`、锁/KFD/dmesg 包装（复用 0928/0930 工具）、champion r29 + ASM beat 作为 arm | 否 | — |
| P1 | **profiler 能力矩阵**（新固件下 deep 轮被裁掉的分析能否恢复）：`--kernel-trace` 行数、`--pmc` 可用计数器（`-L`）及 stall 类计数器是否非零、ATT 能否抓到 FlyDSL JIT kernel、rocprof-compute 能否出面板、PC sampling（**最后做**，先 `-L` 查配置，再只在 toy HIP kernel 上试 host_trap） | 是，每项独立进程，toy 优先 | PC sampling 旧固件 3/3 挂 MES → 放在本轮 GPU 工作的最后 |
| P2 | **bwd 光速消融（M8 式）**：k_dkdv / k_dq 各做 nosm、nowait（不改地址）、nobar、noDS-switch 等 arm；PMC cycle + （若 P1 通过）ATT 每指令 stall | 是 | 只加开关、不动 index；compile-only + toy 先行 |
| P3 | 杠杆设计 + 实现：多方案并行设计（CPU，workflow），每个 arm compile-only（0 spill/scratch）、CPU 越界证明、toy、prod blocked A/B（同进程 r29 A/A + ASM） | 是 | 同上 |
| P4 | 赢家晋升；重建 A0 bwd op-evolve job（旧目录停在 r23；B0 的 r24–32 历史因 B0 SSH host key 变更拿不到）到 r29/新冠军，恢复 P1 验证可用的 deep 分析，跑 deep 轮 + fast 轮 | 是 | job 与手工实验互斥 |

## 决策记录（用户开会期间按建议执行，不等确认）

- D1 B0 `ssh` 报 REMOTE HOST IDENTIFICATION HAS CHANGED：**不绕过 host key 校验**，不从 B0 拉 job 历史；r29 代码用 git 里的 `0927__b0/champions/`。
- D2 PC sampling 在旧固件上 3/3 挂卡：本次放在所有 GPU 工作之后，只用 toy HIP kernel，先 `-L` 确认支持；挂卡则停 GPU、记录 dmesg、转 CPU 工作。
- D3 fwd 不动（e2e 上 bwd 占差距大头，HANDOFF §5）。
- D4 尺子：blocked（lead 4 + block 9，palindromic），同进程 r29 + r29_aa + ASM；每进程新 JIT cache；一个 shape 一个进程；判定阈值 0.5%。
- D5 op-evolve 目标从 1.00× 提到 **1.20× ASM**（validation.py `BEAT_MARGIN=20.0` + final.yaml "beaten by 20%"）：s6 已是 1.038×，维持 1.00× 会让第一个通过的 round 直接以 target_met 结束、deep 轮跑不到；1.20× 取自 w4f 无原子写的 1.30× 上限。
- D6 op-evolve bwd job 在 A0 以 s6 为起点重启（旧目录备份为 `*.a0-stale-0930`），round 24 为 deep（ATT 已恢复进 deep prompt，OE 未提交改动，补丁在 `oejob/deep_att_enable.patch`），之后 4 fast + 1 deep 循环，max_rounds 48。
