## (d) 过程中遇到的问题及解决方案

**阅读约定**

- 机器与时钟态（不同态的绝对数不能直接比较，凡并列处均另行说明）：
  - **A0-限频**：heliosr-1b114-c07-1（单卡），2026-09-29 刷固件之前，VR 限频，DPM 只有 500/1100 MHz 两档，空闲 1100 MHz，负载约 0.95–1.07 GHz。
  - **A0-新固件**：同一张卡，09-29 刷 VBIOS 630A→700E、SMU 125.7.1→125.12.0 之后。op 级 1.74–2.03 GHz，训练中约 1.5 GHz（2,500 W 功耗墙）。10-01 起驱动为 amdgpu-dkms 7.1.0-2412954。
  - **B0**：ctheliosp-1b112-a37-1，4×gfx1250，负载 2,133–2,244 MHz。09-28 nkfix 之后训练中约 1.25–1.69 GHz。
- 来源写法：
  - 路径省略前缀 `output/`。
  - `s:<会话id前8位> <时间Z>` 指会话记录。
  - `OE:` 指 op-evolve 仓库 `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`。
  - `skill:` 指 `.claude/skills/gfx1250-attn-campaign/references/*.md` 或 `~/.claude/skills/*`。
  - 8 位十六进制是 Primus-Turbo 的 commit。
  - `hNN` 指 hint 编号：A0 bwd 在 `0923__flydsl/hint.md`，A0 fwd 在 `0925__flydsl/fwd-job/hint.md`，B0 在 `0927__b0/{fwd,bwd}-hint.md`，A0 09-30 之后（h83/h85）在 bwd job 的 `job_context/hint.md`。
- 时间一律 UTC。bwd 轮次编号 r24–r27 有两套：B0 09-27/28 一套，A0 09-30/10-02 一套。本节都会标明机器。

---

### d.1 GPU wedge 与 AC power cycle

在这两台机器上，卡挂死（wedge）之后，软件手段都救不回来：驱动自己的 reset 完不成，容器里有不可杀的 D 状态进程，`/dev/kfd` 的引用计数也不会归零。唯一的恢复办法是用户本人到机器前给整机断电（AC power cycle）。因此挂卡首先是排程约束，其次才是技术问题。

#### d.1.1 事件表（跨来源去重）

类别图例：

- **A**：参考 GEMM（fp32 Tensile）地址越界写，即"内存越界类"。
- **B**：TLB/队列类。首条是 `INVALIDATE_TLBS` 超时，此前没有任何内存故障。
- **C**：候选 kernel 越界读引发故障风暴，伴随 `IH ring buffer overflow`。
- **S**：启动期 MES 故障族，未归因。
- **K**：已知操作诱因，包括在线调优、inductor、坏候选、profiler、新 kernel 直接上 prod。
- **R**：资源或进程问题，如 SIGBUS、KFD 残留、占卡。
- **F**：进程级故障，可恢复，不算挂卡。
- **D**：降级，不是挂死。
- **X**：非 kernel 原因，如驱动、固件、协作。

| 日期 UTC | 机器 | 当时在跑什么 | 触发/根因 | dmesg 特征 | 类别 | 恢复方式 | 代价 | 来源 |
|---|---|---|---|---|---|---|---|---|
| 09-04（活动前） | A0 | HipKittens bf16 GEMM 阶梯第 16/25 轮，没开 profiler | 未知，普通 GEMM 负载 | MES 无响应，reset 失败 | 平台 | 主机重启 | 1 次重启 | OE:`output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md` §2.1-A |
| 09-06 20:33→09-07 07:42（活动前） | A0 | 无人值守的 op-evolve grouped-GEMM job | 未知 | 11 h 内 `SMU: No response` 857 次 | 平台 | 主机重启 | 1 次重启 | 同上 §2.1-B |
| 09-11 ~08:31 | A0-限频 | E3 探针：在卡上对 FlexAttention `mask_mod` 求 8192×8192 外积 | 在卡上物化超大布尔外积 | `GPU Hang` | K | 运维重启 | 1 次重启；该分析改在 CPU 上用 numpy 做 | 同上 §2.2 |
| 09-11 10:32→10:48 | A0-限频 | E4 探针：rocprofv3 PC sampling（同系列 PMC/ATT 都正常） | PC sampling | 10:32 `GCVM_L2_PROTECTION_FAULT` PF 0x3 RW 0x0 TCP，涉及 3 个 XCD，进程死、卡活；10:48 重试后 `GPU Hang` → MES 无响应 → `wait for reset ack` | K（profiler） | ~11:10 运维重启 | 1 次重启；PC sampling 永久禁用 | 同上 §2.3 |
| 09-13 ~13:19（13:30 发现） | A0-限频 | `forever_queue.sh` 4 个 worker，加独占窗口探针 | 卡本身脆弱；探针半小时内多次 `pkill -9` 正在跑 kernel 的队列，这是可避免的诱因 | `MES(0,0) … INVALIDATE_TLBS` → `MES(7,0) REMOVE_QUEUE` → `wait for reset ack`；随后 `Mode2 reset failed`、`ASIC reset failed -62` | B | 主机重启（时间未记录）；用 `docker cp` 抢救出 1063 行账本 | 下午起无卡可用；09-14 转到 B0 | `0913__opt_plan__claude/phase2/INCIDENT-2026-09-13-wedge.md` |
| 09-14 | B0 GPU0 | 当天工作开始前已坏（uptime 14647 s 起） | 非我方触发 | 120 条 `MES failed to respond`，0 条 `wait for reset ack` | D | 从调度中摘除，改用 GPU1–3；不重启 | 0 次 AC；当天 4 次误判卡状态；一个孤儿循环给坏卡喂了 2 h 15 min | `0914__campaign/HANDOFF.md:274-360` |
| 09-15 04:47→06:07 | A0-限频 | 两臂 e2e 相隔 65 s 启动；随后执行 `docker kill` + `modprobe -r amdgpu` | 没等 KFD 清空；`timeout` 只杀了父进程，torchrun 孤儿仍占卡和 1234 端口；反复探测；在已经不能计算的卡上卸驱动 | 首臂第一次 attention fwd 100% 占用 24 min，dmesg 0 故障；`torch.cuda.init` 45.2 s；之后整机 SSH 失联 | R + 操作 | 操作者 AC（07:18 前） | ~1 h 卡时 + ~2 h 整机不可达；助手曾误报"驱动重载修好了"，已更正 | `0915__opt/INCIDENT-2026-09-15-machine-death.md`；457593d8 |
| 09-15 ~08:08 | A0-限频 | 32L e2e + ASM bwd scratch 常驻缓存 | 显存 88.3% 的配置上再加 1.38 GiB | step 3 SIGBUS；死进程泄漏 KFD 上下文和 411 GB 显存；dmesg 干净 | R | AC（~08:20，08:25 时 up 4 min） | 改用 8L 配置 | s:b596bddb 08:08–08:25Z；`0915__opt/E2E-AB.md` |
| 09-15 08:38 | A0-限频 | AC 后的第一个 8L run | 未定 | worker 首次 attention 后静默死亡；3 个陈旧 KFD 条目；`docker exec` 报 setns 失败；dmesg 0 | 未归因 | AC（09:00） | 当天剩余 GPU 工作暂停 | s:b596bddb 08:38–09:00Z |
| 09-15 ~10:47 | A0-限频 | noconv 对照（`converters: []`） | 退回 torchtitan Flex → 被强制走 inductor → `flex_decoding` autotune | `MES(6,0)/(7,0) REMOVE_QUEUE` 反复约 30 s → `Suspending ip block ih_v7_0` → `wait for reset ack` | K（inductor） | AC（11:16） | 结论：A0 上测不了"不用 turbo attention"的基线 | s:11052787 10:47–11:16Z；8ad25967 |
| 09-15 12:35–12:54 | A0-限频 | nkfix ON 第 4 次连跑（cooldown 10 s） | 未归因 | 38 条 `MES(0,0) ring buffer is full` → `MES(4,0) failed to respond … WAIT_REG_MEM` + `reg_write_reg_wait` 失败；进程 Zsl | S | AC（13:07） | cooldown 改 20 s 后连跑 6 次都干净（不构成安全结论） | `0915__opt/MES-WEDGE.md`；s:11052787 12:51Z |
| 09-16 01:19 | A0-限频 | 第 3 次连续 profile run（v2prof） | 未归因 | 故障计数 0→72；`REMOVE_QUEUE` 反复 + `wait for reset ack` + hung task；rocm-smi 全部 N/A | S | AC（02:26） | — | s:11052787 01:19–02:26Z |
| 09-16 03:16 | A0-限频 | 32L 基线启动 | 并行 workflow 中一个"只做 CPU"的 agent 自写了 GPU 探针 `warps16.py`；其中 num_warps=16 候选挂起，两个实例各占 385 MB，把卡占死 | `ring buffer is full` 持续；`gpu_health.sh` 仍报 HEALTHY；D 状态持有者，无 reset ack | R（编排） | AC（07:35） | "挂卡与 nkfix 无关"这条证据被污染，已撤回（4e03bd57） | `0915__opt/MES-WEDGE.md:11-29` |
| 09-16 07:38 | A0-限频 | 32L b=4 + nkfix（显存 88%） | 起初归因为 SIGBUS/显存，为此加了 `_headroom_ok`，后来证实它从未触发 | 首条 `INVALIDATE_TLBS`；`GPU reset begin` 之后没有 end；训练日志 SIGBUS | S | AC（07:56） | — | s:11052787 07:38–07:57Z |
| 09-16 08:01 | A0-限频 | 32r1（显存仅 6.93%）。07:59 刚有一次 32L+nkfix 跑完 20 步（11,608 tps，5.85×），08:00 起连跑三次做定性，第一次就挂 | 启动期族；把两份日志对齐到同一时钟后，MES 失败比 SIGBUS 早 27 s，说明 SIGBUS 只是症状 | MES 17 条 + SIGBUS | S | AC（08:20） | 推翻上一条归因；此后只用 8L。07:59 那次 run 在 13:13 查出从 step 3 起 NaN（20 步里 17 步），11,608/5.85× 已撤回（见 d.2.1） | s:11052787 07:59–08:20Z、13:13Z |
| 09-16 09:02 | A0-限频 | Triton fwd sweep 第 58 个候选 | 候选 `num_warps=8, waves_per_eu=0, num_stages=2, PRE_LOAD_V=0` | `MES(7,0) REMOVE_QUEUE` → `wait for reset ack`；内核线程卡在 `svm_migrate_to_ram` | K（坏候选） | AC（09:18） | 账本记为 `card-wedge` 终态，不再重试 | e4730158；`0915__opt/FWD-SWEEP.md:42-51` |
| 09-16 09:35 | A0-限频 | e2e 规则 3（FlyDSL wgrad GEMM）第一次真跑 | 在训练步内做 autotune，`except Exception: continue` 吞掉了 launch failure | step 2 `hipErrorLaunchFailure` → MES 对 INVALIDATE_TLBS/REMOVE_QUEUE/SUSPEND/RESET/RESUME 全部无响应 → `GPU reset begin` 未完成 | K（在线调优） | AC（11:17） | 规则 3 改为离线建表 | f08ccffe；`0915__opt/FLYDSL-WEDGE.md` |
| 09-16 11:23、12:28、13:12 | A0-限频 | fly3-fly、copy4（不经过 FlyDSL 的基线臂）、nanhunt | 启动期族，与规则 3 无关 | 三次都是 0 步；nanhunt 为启动期 SIGBUS | S | AC 11:57、12:44、~13:15–13:30（最后一次为推断：13:30 读到的 dmesg 已是新开机） | 当日 14 次启动有 3 次启动期失败（21%） | 6d0cc872；`0915__opt/MES-WEDGE.md:84-103`；s:11052787 13:30Z |
| 09-17 09:12 | A0-限频 | 给已知会越界写的 `dkdv_heads='kv'` 臂计时，当作"下界" | 实验设计错误 | `GCVM_L2_PROTECTION_FAULT` PF 0x5 RW 0x1 TCP → MES REMOVE_QUEUE/SUSPEND 无响应 → `Queues reset on process python3`；没有 GPU reset | F | 不需要 AC；KFD 为空，matmul 5.08 ms（故障前 5.09） | 损失 1 个进程 | `0917__flydsl/GQA-WORKAROUND-COST.md`；s:c4b79aa6 09:13Z |
| 09-21 ~14:12（14:35 报警） | A0-限频 | bwd job r3 opt | 闸门里的 eager fp32 参考走 Tensile GEMM（`Cijk_*`）越界 | 2 条不可恢复签名（reset ack / ring gfx timeout / GPU reset begin 一类） | A | AC（~14:55 开机） | 被打断的 opt 归档为 `1-opt.stale-20260921T141238` | s:a9a96fef 14:35–15:09Z；`0923__flydsl/wedge4/ANALYSIS.md:45-51` |
| 09-21 16:01:54 | A0-限频 | r3 验收 `gpu5.sh`（detach 运行） | 同属 r3 路径，事后按 A 类归档；`patrol.sh` 告警不锁存，`release_guard` 从未被拉起 | 会话判为不可恢复；4547915c 称这次 dmesg 没有任何 reset 签名，supervisor 停在 `gpu_ok()` 的 900 s 退避里 | A?（签名矛盾） | AC 09-22 00:49:22 | 整夜无卡；09-21..22 五轮墙钟里有 17.8 h（82%）耗在挂卡和等 AC | 617481d7；4547915c；s:a9a96fef 09-22T00:59Z |
| 09-22 14:26 | A0-限频 | bwd r8 armB（`bitwise.py` 在 prod 形状直接调 fp32 参考） | 首条是 INVALIDATE_TLBS，此前没有内存故障 | INVALIDATE_TLBS→REMOVE_QUEUE→SUSPEND→RESET→RESUME→`MES might be in unrecoverable state`→`GPU reset begin!. Source: 3`→MES(0..6) REMOVE_QUEUE 无界级联 | B | AC 14:40:38 | r8 与 incumbent 逐位相同，以此继承正确性，手工收口 | `0922__flydsl/ROUND3-CLOSEOUT.md` Round 8 |
| 09-22 16:02 | A0-限频 | bwd r9 `d_final` 验收 sweep；挂前约 2 min `bitwise.py` 在 prod 跑过约 2000 次 fp32 Tensile | 首行签名没保存（ring buffer 随断电丢失） | 会话读到 `unrecoverable` + `GPU reset begin!. Source: 3`；`ps` 诊断命令本身挂住 | 未定 | 节点交接，夜间 AC（~23:47 开机） | 断电截断 255 个 git object，丢失 70efc407；ROUND3-CLOSEOUT.md 被截在 16384 B | `0922__flydsl/HANDOVER.md`；s:a9a96fef 09-23T07:13Z |
| 09-23 09:21 | A0-限频 | bwd r11 `d_s4`（开机后 9.5 h 零异常） | 直接原因不明 | INVALIDATE_TLBS→`failed to suspend all gangs`→`Failed to detect hung queues`→unrecoverable→GPU reset begin→14 次 REMOVE_QUEUE；没有 sub-4GB 地址，没有 `Cijk_` | B | AC（09:42 前） | 新增早停信号（出现在不可恢复前约 5 s） | 98eaa09e；`0923__flydsl/wedge4/ANALYSIS.md` |
| 09-23 13:35 | A0-限频 | bwd r14 的 scratch 候选臂 | 候选 kernel 越界读到已映射但不可读的页 | `GC_UTCL2` 故障风暴：TCP、RW 0x0、PF 0x3、高地址、MORE_FAULTS 0x1、3 个 XCD、6 次 `IH ring buffer overflow`，1 s 后 MES 不可恢复 | C | AC（~14:00） | 当天 `dmesg_restrict=1`，全天的"零故障"检查都在读空输出；证据最后从 `/var/log/kern.log` 找回 | d2bb6a87；`0923__flydsl/hint.md` h22 |
| 09-24 ~05:24 | A0-限频 | bwd r17 deep 轮在 prod 形状跑 PC sampling | PC sampling | 131 行 page-fault burst；`GPU core dump skipped because PC Sampling active`；`Timeout while waiting for queue sync` | K（profiler），卡存活 | 不需要 AC | 离挂卡只差一步 | 82420bdf |
| 09-24 08:58:53 | A0-限频 | bwd r18 `validation.py` 在单进程里跑 fast,proxy,prod | 单进程跨 shape 运行。"一 shape 一进程"在 `facts.md:241` 早有记录，但没进 `measure()`；写 fault 本身仍未解释 | RW 0x1 PF 0x5 写 fault，31 s 后 MES 死（MES(6)→MES(7)），没有 IH overflow | A 型写 fault | AC（09:14:30 开机） | 改成一 shape 一进程后，6 次完整测量 0 故障 | bf58bec0；`0924__flydsl/DAY-SUMMARY.md:82-104` |
| 09-28（Day2 开局发现） | B0 GPU1 | 夜间没有我方任务 | 未知；uptime 69813 s 时有一个非我方 python3 在 libamdhip64 里 GPF | 持续刷 `MES(0,0) failed to respond` / `ring buffer full` | 未知 | 弃用 GPU1（fwd 移到 GPU2，bwd 移到 GPU3）；交还时注明需要 AC（同一 hive 的 4 卡会一起断电） | B0 一直少一张卡到收工 | `0927__b0/LAB-RULES.md`；`0927__b0/HANDOFF-A0.md` §9 |
| 09-28 ~11:55 | B0 GPU2 | fwd job r20 act 的 `adv_m32x8.py` | 在卡上跑 fp32 hipBLASLt 参考 GEMM | TCP PF 0x3（AID1.XCD2），page not present 0x280000，rc=134；无 MES hang | A（未升级为挂卡） | 硬停 fwd loop | r20 act 作废；h50 禁止在卡上算 fp32 参考 | d13160ba；`0927__b0/fwd-hint.md:862-870` |
| 09-28 11:55 | A0-限频 | e2e `a0_p4a` 32L（ASM 先跑），紧接在 p3b 的 86 步 NaN 运行之后 | 怀疑上一个进程已把卡弄坏，加上显存 89% | step 2 `hipErrorLaunchFailure` → MES unrecoverable → GPU reset begin | S? | AC（12:10 开机） | 改为 24L，每次开机只跑一个训练进程 | `0928__a0_repro/REPORT.md` §3 |
| 09-28 12:32 | A0-限频 | e2e `a0_p4b` 24L，FlyDSL 在 step 1 运行 | fwd 树的 `_env.py` 把 `HIPBLASLT_TENSILE_LIBPATH` 改到宿主库（与证据最吻合，但未在卡上证实） | INVALIDATE_TLBS → unrecoverable → GPU reset begin → `wait for reset ack`；KFD 残留 2 个 | 宿主库 | AC（13:01 开机） | 当天 5 次训练启动：2 次有效、1 次 NaN、2 次挂卡 | `0928__a0_repro/REPORT.md` §5.2–5.3 |
| 09-28 19:39 → 09-29 22:19 | A0 | 无（管理员 asierrag 更换驱动和固件包） | dkms 7.1.1-2397345 → 7.1.0-2410994；推断新固件的 TOC 与 VBIOS 630A 不匹配 | PSP `LOAD_TOC failed (0x11)`、`LOAD_IP_FW failed (0xFFFF0006)`、`SMU: No response`、`hw_init of IP block <smu> failed -62` | X | 06:35 用户做了 AC，无效；管理员刷 VBIOS 700E + dkms 7.1.0-2411946（主机约 10 次重启） | 约 1.5 天不可用；09-29 之前 A0 的绝对数全部作废 | s:2dafe0d2 06:25–06:50Z；`0928__a0_repro/REPORT.md` §6.3 |
| 10-02 11:40:59 | A0-新固件 | bwd job r27 opt agent 生成的 `build_cmd.sh` | 在 prod 形状直接调用两个从未上过卡的 s6 基底变体，目的只是读 `vgpr_count`（compile-only 就能拿到）：A_g74 让 k_dkdv 按 KV band 组成 (1,4,1) cluster、多播 Q/dO；B_g82 让 k_dqg 按 GQA 组组成 (4,1,1) cluster、多播 K/V。卡挂在第一个臂 A_g74 上，B_g82 没跑到。探针 `build_probe.py` 与 r26 读 w4f VGPR 的 `vgpr_probe.py` 逐字节相同（docstring 仍写 w4f）；1ee0dd59 提交信息、WEDGE-1002.md（以及 h85 正文、`gfx1250-card-safety` §1 #13）把这两个变体记成"w4f 融合变体（4-wave + split barrier + dQ 原子）"，是误记。没有 toy 先行、没有 `AMD_SERIALIZE_KERNEL`、没有锁/KFD 包装，LIBPATH 指向宿主库 | REMOVE_QUEUE→SUSPEND 失败→queue reset→RESET 失败→unrecoverable→`GPU reset begin!. Source: 3`→ADD_QUEUE 失败 | K（新 kernel 直接上 prod） | 只写 `.stop`，不 kill；AC（时间未记录） | r27 作废；新增 h85 | 1ee0dd59；`1002__oe/incident/WEDGE-1002.md`；OE bwd job `rounds/027/_scratch/arms/{A_g74,B_g82}/kernels.py`、`_scratch/build_run/out`、`1-opt/raw/build_probe.py`（= `rounds/026/1-opt/raw/vgpr_probe.py`） |
| 10-06（窗口外，其它项目） | A0-新固件 | `hipblaslt-bench --algo_method all`，bf16 TN 7168×8192×576 | 预编译 Tensile 解（约第 122/133 个）本身是坏的 | `unspecified launch failure` → MES REMOVE_QUEUE/SUSPEND/RESET/RESUME 无响应 → GPU reset begin，未完成 | 坏 Tensile 解 | AC | 仅作为 hipBLASLt 的风险参考 | skill:`gfx1250-card-safety` §1 #14 |

还有几次可存活的故障，它们不需要 AC，但都曾被误判：

- 09-17 r1 首次尝试时的越界读，PF 0x3 RW 0x0。
- 09-23 ~14:15 那批，以及 09-24 03:58、07:58、10:51 的 shape 切换处故障。

这些都只损失进程。来源：s:c4b79aa6 09-17；0e9ca37c；`0924__flydsl/DAY-SUMMARY.md:84-87`。

#### d.1.2 挂卡类别与认识的演进

1. **09-04..09-13（A0-限频）：先当成平台问题。**
   - 四种互不相关的负载都触发过同一种挂法：MES 无响应、reset 完不成。这四种负载是 HipKittens GEMM、grouped-GEMM、E3 mask 外积、E4 PC sampling。
   - 当时的结论：卡本身脆弱；PC sampling 必挂；驱动重载不可行，因为 D 状态进程不释放 `/dev/kfd`、amdgpu refcount 不归零。
   - 同时发现挂卡时 `rocm-smi`、`ps -eo …wchan`、`pgrep`、`docker exec` 都会自己挂住。
   - 处理：提交 HARDWARE-ISSUE / PLATFORM-ESCALATION，09-14 换到 B0。
2. **09-14..09-17：学会区分三种状态。**
   - "降级"与"挂死"不同。B0 GPU0 有 120 条 `MES failed to respond`，但仍能跑完 kernel。只有 `wait for reset ack`、`ring gfx timeout`、`GPU reset begin` 才是不可恢复。
   - "进程级故障"与"挂卡"不同。09-17 那次 `Queues reset on process` 不需要 AC。
   - e2e 期间启动期挂卡很多，09-16 当天 14 次启动有 3 次挂（21%），由此得出：**风险按启动次数计价，不按运行时长**。
   - 识别出的确定诱因：训练步内 autotune、`converters: []`→inductor、Triton `num_warps=16`、坏候选 #58、在 KFD 残留时启动新 run、`modprobe -r`、反复 SIGKILL 正在跑的 kernel、"只做 CPU"的 agent 私自上卡。
   - 另一个结论：SIGBUS 是症状，不是病因。
3. **09-21..09-24（op-evolve bwd job）：按 dmesg 首行分出 A/B/C 三类，判别式被逐个证伪。**
   - 先后提出过四个"挂卡前兆"判别式，全部被实测推翻：
     - h15：sub-4GB 截断地址 + PF 0x5/RW 0x1 + `copy_context_work_handler`。B 类挂卡完全不触发它。
     - h22：`IH ring buffer overflow`。09-24 那次开机出现 8 次 overflow，卡仍然活着；而 08:58 的挂卡反而没有 overflow。
     - MORE_FAULTS / 多 XCD。可存活的批次里 8/8 都有 MORE_FAULTS。
     - 故障到达率"6000×"。这个数字其实是驱动约 97 ms 的节流（7436cd29）。
   - 结论（3fd48059）：**不存在可用的挂卡预测指标**，只对挂卡本身告警。
   - 这期间真正定位到的触发器有两个：一是闸门在卡上跑的 fp32 Tensile 参考，用 refcache 解决；二是单进程跑多个 shape，约 44%/进程会出故障，改成一 shape 一进程后连续 6 次 0 故障。
   - 同期还发现 `dmesg_restrict=1` 使此前的"零故障"检查全部无效。
4. **09-28..10-06：宿主库与新 kernel 成为主要诱因，刷固件没有让卡变结实。**
   - 宿主 hipBLASLt 库 `~/.local/hipblaslt-gfx1250` 与多起事件相关：A0 09-28 一次 NaN、一次挂卡；B0 step 1 卡住；10-02 挂卡。
   - 在卡上跑 fp32 参考 GEMM 仍会出故障（B0 GPU2）。
   - 09-29 刷固件后时钟恢复，ATT 也能用了，但 10-02 一个新 kernel 直接上 prod 照样挂卡。10-06 `hipblaslt-bench` 枚举到坏的 Tensile 解也会挂。
   - 规则收敛为 h85：新 kernel 按 toy→prod 逐级放大；只读 ISA/VGPR 的探针一律 compile-only；只用镜像自带的 hipBLASLt 库。
   - 另一组数据：B0 09-27/28 共 ≥12 次训练启动，0 次挂卡。把各段合起来，启动期挂卡率约为 ≤5/31≈16%（`1002__e2e/E2E-PLAN.md:377`）。

#### d.1.3 有效的对策

| 措施 | 针对的失败 | 引入 | 证据/效果 | 来源 |
|---|---|---|---|---|
| 不做驱动级恢复：从恢复阶梯里删掉 `modprobe -r amdgpu`。挂卡时只读一次带超时的 `timeout 20 sudo -n dmesg`（再 tail）和有界的 `/sys`，不给 D 状态进程发信号，写 `.stop` 哨兵后转做 CPU 工作 | 09-15 整机失联；反复 kill 留下更多不可杀进程 | 09-15 起 | 此后再没有因恢复操作而加重的事件 | 457593d8；skill:`gfx1250-card-safety` §4–§5 |
| 风险按启动计价：多个配置合进一个进程；用 60 步长 run 代替多次短 run；两次运行之间等 KFD 清空，再冷却 25–45 s；单卡上 fwd/bwd job 不并行；每卡一把 flock 锁 | 启动期族（21%/启动） | 09-16 起 | B0 ≥12 次启动 0 次挂卡；10-02 e2e 0 故障 | skill:`gfx1250-card-safety` §0；`1002__e2e/E2E-PLAN.md` |
| compile-only 先行（`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 HIP_VISIBLE_DEVICES=-1`）；只要 spill>0 或 `private_segment_fixed_size>0` 就直接淘汰 | spill 的 build 首次 launch 即挂起（代价是一次断电） | 09-21（b1572877） | VGPR ≤951 从不 spill；=1024 必 spill | skill:`env-and-pitfalls` §2b §6 |
| 新地址算术先在 CPU 上做越界证明，全枚举而不是抽样 | C 类越界读 | 09-23..09-24 | k_dq 全枚举 134,217,728 元组，结论为双射；另找到 3 个潜在缺陷（见 d.6） | 41c916b7；h22/h33 |
| 新 kernel 上卡的顺序：toy 形状 → 独立进程 → `AMD_SERIALIZE_KERNEL=3` → 不做进程内 autotune → 经锁/KFD 包装 → 只用镜像 hipBLASLt。h85 必须写成 hint 表格行 | 10-02 挂卡 | 09-17 起写进 skill，10-02 升级为 h85 | 09-30 手工 campaign s1–s6 期间 0 挂卡（w4f 首次上卡只用 toy、nsp=1、串行单进程） | 1ee0dd59；skill:`gfx1250-card-safety` §1 #13、§2 |
| 一 shape 一进程，首个 rc≠0 即中止 | 09-24 挂卡；shape 边界故障约 44%/进程 | 09-24（bf58bec0） | 之后 6 次完整测量 0 故障 | `0924__flydsl/wedge-rootcause/` |
| 参考值离开 GPU（refcache），并且只用镜像 hipBLASLt 库 | A 类挂卡 | 09-22（376b9fd3 等），10-02（h85） | prod 参考在卡上直接 FAULT，改在 CPU 上算，10m26s，产出 1.1 GB | 376b9fd3（commit message）；`0922__flydsl/gate-patch/README.md` |
| 禁用 PC sampling | 09-11 挂卡；09-24 差一点挂卡 | 09-11 起；09-24 重申（82420bdf） | 旧固件上 3 次尝试全部 fault，其中 1 次挂卡（09-11 10:48），2 次卡存活（09-11 10:32、09-24 ~05:24）。`0930__bwd/PLAN.md` D2 和 skill:`gfx1250-attn-campaign/SKILL.md:106` 写成"3/3 挂卡"，说重了。刷固件后 rocprofv3 1.3.2 启动时直接拒绝该配置 | skill:`env-and-pitfalls` §2c；OE:`output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md` §2.3；82420bdf |
| 只按显式 PID 杀进程，绝不用 `pkill -f`（会匹配到自己，已杀过 4 次以上自己的 shell）；长循环用 `setsid nohup` 启动，停止用 `op-evolve stop` 或 `kill -TERM -<PGID>`；watchdog 绑定 `E2E_RUN_MARKER` 或 PID；定期查 ppid=1 的孤儿 | 误杀、孤儿占卡、监控盯错 run | 09-14..09-25 | 6ad9bda0（按进程组回收）；6029b430 | skill:`gfx1250-card-safety` §6；`op-evolve-ops` §3 |
| 监控与守护：`patrol.sh` 挂卡告警改为锁存并与开机时间比对（617481d7）；`release_guard` 写 pidfile、由 patrol 拉起，新增"supervisor 停在退避""state.yaml 冻结"检查（dd1aec29、4547915c）；`card_ok.sh` 看 KFD 持有者、`/sys` VRAM，并跑一次带硬超时的玩具 matmul（ac95f1c7）；免 root 的活性探针 `timeout 20 docker exec fa-repro true`；只对 MES failed/unrecoverable 和 docker exec 超时报警 | 告警只响一次；守护进程从未启动；健康脚本误报 HEALTHY | 09-21..09-24 | 09-21 起每次挂卡都在一个巡检周期内报出 | 617481d7；3fd48059 |
| 在线调优改为离线建表，坏候选记为终态 | 训练步内 autotune；sweep 候选挂卡 | 09-16 | `flydsl_table.py`：每形状一个子进程 + 硬超时 + SQNR 门；`num_warps=16` 和 #58 永久排除 | f08ccffe；e4730158 |
| e2e 防线：禁止 `converters: []`；`compile.enable: false`；留显存余量（8L 或 `E2E_MEM_STOP=89.5`）；前两步先跑 ASM；BLAS guard（exit 94）；非有限值 watchdog（exit 96）；外来 KFD 进程（exit 91） | 09-15/16、09-28 的 e2e 挂卡 | 09-15..10-02 | 10-02 e2e 两个 92 步进程 0 故障 | `1002__e2e/E2E-PLAN.md` §3 |
| 委派任务时明确禁止上卡，并在委派前后检查 KFD 持有者 | 09-16 agent 私自上卡 | 09-16 | 此后再没有私自上卡 | 4e03bd57 |
| 每轮结束都 push；每次 AC 后跑 `git fsck --no-dangling` | 断电截断 git 对象（见 d.7） | 09-23 07:36 | 09-23 09:42 那次 AC 没有损坏仓库 | memory `push-after-every-round` |

**AC 之后的恢复步骤**（gfx1250-card-safety §5、`1002__oe/incident/WEDGE-1002.md`）：

```bash
# 0. 用户 AC 断电（只有这一步能恢复 wedge；绝不 modprobe -r）
sudo modprobe amdgpu                    # 内核 cmdline 里 modprobe.blacklist=amdgpu，不会自动加载
sleep 6
sudo sysctl -w kernel.dmesg_restrict=0  # 每次重启都会回到 1；确认 dmesg | wc -l 是几千行
cat /sys/class/drm/card*/device/pp_dpm_sclk   # 检查 DPM 表/VBIOS/dkms 是否又被人改过
ls /sys/class/kfd/kfd/proc/             # 必须为空
docker start fa-repro                   # 复用，不重建；确认容器内没有镜像版 primus_turbo
# 一次带超时的 4096^3 bf16 matmul（A0-限频 约 5.07 ms）——这才算"卡能用"，rocm-smi 不算
git fsck --no-dangling; find .git/objects -size 0   # 断电可能截断 git 对象
# 先补 hint（如 h85）再 op-evolve resume（不带 --config）
```

#### d.1.4 AC 断电 / 重启次数统计

| 日期（UTC） | A0 因挂卡引发的人工断电/重启 | 说明 |
|---|---|---|
| 09-11 | 2 | E3 外积、E4 PC sampling；运维执行，是 AC 还是热重启没有记录 |
| 09-13 | 1 | 第 5 次 wedge（从 09-04 起累计）；重启时间未记录 |
| 09-15 | 5 | 机器失联（07:18 前）、SIGBUS（~08:20）、8L worker（09:00）、noconv（11:16）、nk4（13:07） |
| 09-16 | 9 | 02:26、07:35、07:56、08:20、09:18、11:17、11:57、12:44 有用户确认；~13:15–13:30 为推断 |
| 09-21 | 2 | ~14:55、09-22 00:49（都是 r3） |
| 09-22 | 2 | 14:40（r8）、~23:47（r9） |
| 09-23 | 2 | 09:42 前（r11）、~14:00（r14） |
| 09-24 | 1 | 09:14（r18） |
| 09-28 | 2 | 12:10（p4a）、13:01（p4b），都在刷固件之前 |
| 10-02 | 1 | r27，刷固件之后；AC 时间未记录 |
| **合计** | **27（区间约 25–28）** | 下界 25：09-11 两次若是热重启则不计。上界 28：再加 09-29 那次无效 AC |

- 另有 1 次：09-29 驱动/固件不匹配，用户 AC 无效，随后管理员刷固件，期间主机约 10 次重启。把它算进去，人工断电合计 **28 次**（即上表的上界；不含它为 27 次）。全报告统一用这两个数。
- 活动开始前 A0 还有 2 次挂卡重启（09-04、09-06/07）。窗口外 10-06 有 1 次（其它项目）。
- B0：我方没有发起过 AC。GPU1 在交还时仍需 AC；GPU0 只是降级。
- 文档口径要更正：skill `gfx1250-card-safety` 写"09-16 一天五次"，DAY-0916-SUMMARY 也只记了 5 次，那只是 08:20 之后那一段。按 UTC 算，09-16 实际是 9 次。
- 另有约 13 次原因不明或由他人造成的主机重启，不计入上面的统计：09-13 ~04:11、09-17 ~07:46（造成 6 个 git 对象被截断，见 d.7）、09-22 06:47:43、09-23 ~18:15、09-24 ~21:55、09-25 03:10、09-25 10:10:51、09-28 ~02:10 和 08:13 前（这两次是 CPU BERT 致命错误导致主机崩溃）、09-28 21:50（管理员）、09-29 01:47、09-30 01:31、10-01 20:41（CPU L3 致命错误）。其中部分时间取自 journalctl 首条记录，可能与实际开机时间有偏差。
- 时间代价举例：09-21..22 五轮墙钟 21.65 h，其中 17.8 h（82%）花在挂卡和等待 AC 上，实际工作只有 3.85 h（s:a9a96fef 09-22）。

---

### d.2 GEMM / hipBLASLt 问题与优化

#### d.2.1 时间线

| 日期 | 认识/现象 | 数字（机器/时钟） | 处理 | 来源 |
|---|---|---|---|---|
| 09-11 | 镜像 `primus-turbo:gfx1250-20260831-extended-v2` 上 hipBLASLt 能用，但很慢 | A0-限频：bf16 8192³ 91.5 TF/s（各布局 60–92），Triton GEMM 1002.7，差 10.96× | 不用 hipBLASLt，改用 Triton GEMM 作 roof | OE:`output/0911__fa_gfx1250_phase1/E1.md` §5 §8 |
| 09-13 | 换成 `amdprimus/amdprimus:gfx1250-20260910`（fa-tune:deps）后，任何 matmul 都报 `HIPBLAS_STATUS_INVALID_VALUE` 和 `Cannot read …/TensileLibrary_lazy_gfx1250.dat`。当时误判为"镜像缺 gfx1250 Tensile 库"，按"~37×"写进了 PLATFORM-ESCALATION | A0-限频：`TORCH_BLAS_PREFER_HIPBLASLT=0` 走 rocBLAS，8192³ 27.4 TF/s；e2e 32L 245 tps、133.7 s/step | harness 默认 PREFER=0，只保证参考能跑；得出"e2e 无法验证 attention" | `0913__opt_plan__claude/phase1/RESULTS.md` §5 §11 |
| 09-14 | B0 上更正：库是在的（46 个 bf16 解）；换环境变量的三种组合结果相差 <1%；当时结论是"hipBLASLt 在 gfx1250 上本来就慢" | B0：torch.mm 112.98 TF/s，朴素 Triton 1190.07（差 10.5×）；flex eager e2e 2,392 tps，compile + 强制 Triton GEMM 后 9,602（4.01×）；只开 compile 反而 −4% | 设 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` 和 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`（后来证明在 A0 上开 compile 会挂卡） | `0914__campaign/RESULTS.md:39-57,132-146` |
| 09-15 | 真因①：路径错位。加载器在 `library/` 下找索引，402 个文件实际放在 `library/gfx1250/` 子目录（镜像打包缺陷）。指向 `_rocm_sdk_devel` 那份会 core dump | A0-限频：8192³ 从 rocBLAS 27.64 到 68.74 TF/s（2.49×）；e2e 32L 单步 135/134 s → 17/17 s（7.9×），吞吐 244 → 2,027 tps（8.3×；两臂各 3 步、非稳态、n=1） | 设 `HIPBLASLT_TENSILE_LIBPATH=<镜像>/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`；写 VENDOR-REPORT-hipblaslt.md 上报 | `0915__opt/BLAS-FINDING.md:15-31,49-58`；`0915__opt/VENDOR-REPORT-hipblaslt.md:58-61` |
| 09-15 | 真因②：开 `HIPBLASLT_LOG_LEVEL=4` 和 `TENSILE_DB=0x6` 后看到，NN（`Ailk_Bljk`，即 dgrad）只有稀疏的 GridBased 表，缺纯 bf16 GEMM 库 `BB_BB_UA_Type_BB_HPA`（TN 有 3 份，含 CU96/CU192 变体；文件数 TN 68、NN 40）。查表最近邻落到 N=1，选中 GEMV 用的 tile `MT32x16x32`。这也是 e2e 三模态噪声的来源 | A0-限频：GEMM 占单步 94%，其中 97% 在 MT32x16x32；单次 592–681 ms，grid 6567 万 WG；同形状走 TN 快 16.5–21.5×；同一张卡 torch NT（库内为 TN）跑到 1502 TF/s | nkfix v1（改 dgrad 的 B 布局）：8L 6,128 → 11,604 tps（1.894×，n=9）；跨运行 sd 从 3.91%（三模态）降到 0.42% | `0915__opt/GEMM-NN-FINDING.md`；`0915__opt/VENDOR-REPORT-hipblaslt.md` |
| 09-16 | wgrad（`Ailk_Bjlk`）仍然落在 MT32x16x32，57 次调用共 2185.6 ms，占单步 77%。v2 规则的微基准把布局造错了，e2e 里命中 0 次。实测只改 A 或只改 B 都只能到 60–70 TF/s | A0-限频：两个操作数都改后 1145.8–1627.0 TF/s；8L 37,746 tps（6.16×，n=5）。原报 38,043/6.21×（n=6），其中混入了一次 NaN 运行，已更正 | nkfix v3：wgrad 判据放在前面，A 改为连续、B 改为 N-major；统计写到 `NKFIX_STATS_FILE`（atexit 的输出会被启动器吞掉） | `0915__opt/PROFILE-POST-NKFIX.md`；`0915__opt/GEMM-WGRAD-FINDING.md:24-76`；`PROGRESS-REPORT-0917.html` |
| 09-16 | 规则 3：用同事的 FlyDSL gfx1250 WMMA GEMM 接管 wgrad，去掉约 137 ms/步的转置拷贝。第一次把 autotune 放在训练步里跑，卡挂了 | A0-限频：8L 46,374 tps，同会话对照 42,445（+9.26%，n=3，累计 7.57×）；32L 1,984 → 14,050（7.08×，n=1；同会话对照 12,340） | 改成离线建表 `flydsl_table.py`，表的 key 用完整 (M,N,K)；写 VENDOR-REPORT-flydsl-gemm.md | `0915__opt/DAY-0916-SUMMARY.md`；`0915__opt/FLYDSL-WEDGE.md` |
| 09-16/17 | 扫描 74 次 e2e：带 nkfix 的 39 次里有 8 次 loss NaN，不带的 35 次为 0（Fisher 单尾 p=0.004）。NaN 运行都是各自组里最快的 | A0-限频：撤回 49,878、32L 11,608/5.85×、38,043/6.21×；ASM bwd 的 +14.40% 改为 +12.72% | `e2e.sh` 自动统计 NaN 并打印 DISCARD；加累加器式有限性检查 `NKFIX_CHECK`；规则 3 默认关闭；nkfix 判为不可交付 | `0915__opt/NKFIX-NAN-RATE.md`；`0915__opt/NAN-FINDING.md` |
| 09-17 | 为了绕开 INVALID_VALUE，在 op-evolve 发车前往 fa-repro 的 `/usr/lib/python3.12/sitecustomize.py` 追加了 `setdefault("TORCH_BLAS_PREFER_HIPBLASLT","0")` | — | 后果：接下来 12 轮里，job 进程中 `_env.py` 的 setdefault 全部无效，torch GEMM 一直走回落路径 | s:c4b79aa6 09-17T12:01Z；`0923__flydsl/STAGE2-S0-PROBE.md` S0-b |
| 09-21..22 | 闸门、benchmark、bitwise 里的 eager fp32 参考在卡上走 Tensile（`Cijk_Ailk_Bljk_SB_MT128x64x8`），触发 aperture violation，造成 A 类挂卡 | A0-限频：r3 挂卡两次；r8/r9 挂卡前也都跑过这条路径 | refcache（376b9fd3），并补上漏掉的 4 个调用点（00128018、a5693e9c、93ad3b89） | `0922__flydsl/gate-patch/README.md` |
| 09-23 | S0-b：确认"缺库"的诊断不成立，并列出容器里三个库目录：`/opt/rocm/lib/hipblaslt/library` 是空的（容器的 ROCm 不在 `/opt/rocm`）；默认搜索路径 `_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library` 这一层没有 gfx1250 载荷，是 INVALID_VALUE 的真因；`_rocm_sdk_devel/lib/hipblaslt/library/gfx1250` 有 326 个文件但残缺（无 `Kernels.so-000`、无 extop/transform，`.dat` 未压缩），指向它会 SIGSEGV（exit 139）。表里没有 09-15 已验证可用的 `library/gfx1250/` 子目录（402 个文件），于是判为"只有宿主库完整"，把宿主 ROCm 10.1.0 的 328 个文件拷到 `~/.local/hipblaslt-gfx1250/gfx1250` 使用，而且只验证了 512/2048/4096 的方阵。此后镜像子目录库（09-15 方案，09-28 B0 e2e 也用它）与宿主库两条路径并存，直到 10-02 h85 才统一为镜像库 | A0-限频：fp32/bf16 方阵和当年 fault 过的 fp32 QK^T 都跑干净 | `TORCH_BLAS_PREFER_HIPBLASLT=1` 改为显式赋值，保留 `OPEVOLVE_KEEP_BLAS_ENV=1` 作逃生口 | d9160519；`0923__flydsl/STAGE2-S0-PROBE.md` S0-b；`0915__opt/BLAS-FINDING.md:15-31` |
| 09-24 | LIBPATH 设对之后，refcache 的前提不复存在 | A0-限频：prod 上 `forward_reference` 0.88 s + `eager_attn_bwd` 0.89 s = 1.77 s，跨进程逐位一致 | 仍保留 refcache，因为偶尔还有非致命 page fault | 1b19fbe4；`0924__flydsl/REFCACHE-PREMISE-GONE.md` |
| 09-28 | B0 e2e 的反向 GEMM 同样全部落在 MT32x16x32。另外 A0 的 `0915__opt/bin/nkfix.py` 在 HEAD 上已经坏了（`_MM/stats/_big/_headroom_ok` 未定义） | B0：修复前 GEMM 15.6–16.5 s/步（占 96%），ASM e2e 2,016–2,022 tps（高档）；op 级 dgrad/wgrad 50–79 TF/s，lm_head 50–55 TF/s | 以 1255557f 为基础重写 `nkfix_b0.py`。修复后每步 997 次 GEMM 全部落在 `Alik_Bljk MT256x256x128`，GEMM 0.83–0.86 s，转置和检查另占 0.1 s；ASM 20,718 tps（10.3×），FlyDSL 19,762，Triton@1cb2e183 14,563；8 次运行共 486 步，0 次 NaN | `0927__b0/gemm/REPORT.md` §0–§4 |
| 09-28 | GEMM 修好后整板功耗顶到约 2.13 kW，bwd 阶段 sclk 从约 1,820 降到约 1,390 MHz。单步拆分里 attention kernel 本身慢了 10–16%：gemm/REPORT 记为"原因没查"，profile/REPORT §0.3 归因于时钟。同进程的配对比值不受影响 | B0：attn bwd ASM 197 → 229、FlyDSL 269 → 300 ms/步；k_dkdv 5.049 → 5.95–5.99 ms（+18%）；ASM bwd 主 kernel 6.008 → 7.00–7.09 ms（+17%）；k_dq +2% | 绝对 ms 一律用修复后的数；建议设 `NKFIX_CHECK=0` 降低 GEMM 功耗 | `0927__b0/profile/REPORT.md` §0.3 §3；`0927__b0/gemm/REPORT.md` §5 |
| 09-28 | 宿主库在 B0 上出问题：p1a_turbo 用宿主库，step 1 卡住 14 min（gdb 显示主线程在等 AQL 槽位，dmesg 干净）。GEMM 突发尺子用宿主库时，一次突发 47 ms（约 80 TF/s，慢约 20×），时钟也不降 | B0：`kb6_prod_gb_*` 数据作废 | launcher 在 `bash -c` 里给镜像库赋值；尺子加载 arm 之后再赋一次值（fwd 树的 `_env.py` 会改写它）。VERIFY 起初认为归因证据不足（同时变的还有 roctracer、每步 inductor block mask）；RESULT.md 随后用 p1a/p1b 单变量对照（只差 BLAS 库，p1b 正常跑完）排除了这两个嫌疑，归因到宿主库，每边 n=1 | `0927__b0/e2e/RESULT.md:50-59`；`0927__b0/fwd-nospec/REPORT.md` §2；`0927__b0/e2e/VERIFY.md:28-38` |
| 09-28 | 宿主库在 A0 上出问题：fwd 树的 `_env.py` 在首次 attention import 时把 LIBPATH 改到宿主库 | A0-限频：a0_p3b 在 step 4 出现 grad inf，之后 86 步 NaN，非有限值全部出在 bwd GEMM 输出；a0_p4b 在 step 1 挂卡；两个"ASM 先跑"的进程都正常（只是相关性，没有在卡上证实） | e2e 副本里删掉这两行（`fwd_r16_imglib`） | `0928__a0_repro/REPORT.md` §5.3 |
| 10-02 | A0-新固件 e2e 加了几道 BLAS 防线：preflight 用 grep 拒绝任何 BLAS 赋值、setdefault、putenv；运行时 guard 发现改写就打 `!! BLAS-REPOINT` 并 exit 94；前两步先跑 ASM；`NKFIX_CHECK=1` 遇到非有限值就停（exit 96）。同一天 bwd r27 的 opt agent 仍用宿主库在 prod 跑新 kernel，卡挂了 | A0-新固件，`NKFIX_CHECK=1`：0 次 BLAS 改写，0 故障；ASM+nkfix 1,350.6 ms/步、24,262 tps；fly（r16+s6）1,349.4/1,349.9 ms（24,283 tps） | h85：禁止使用 `~/.local/hipblaslt-gfx1250`，只用镜像库 | `1002__e2e/E2E-PLAN.md` §3；`1002__e2e/RESULT-e2e.md`；`1002__oe/incident/WEDGE-1002.md`；wt-llama31 README:189-190（tps） |
| 10-05 | 教程把配置固定下来：镜像库，`bash -c` 里显式赋值，nkfix hook，transpose_triton | A0-新固件：分支 FlyDSL attention + nkfix（CHECK=0）1,361.9 ms/步、24,061 tps；按提交版本重跑 1,360 ms、24,099 tps；漏了 `transpose_triton.py` 时 1,689 ms、19,406 tps。这三个数都是 CHECK=0，不能和 10-02 的 CHECK=1 数直接比：按 CHECK=1 约 +33 ms/步（B0 实测值，教程沿用）折算，ASM 在 CHECK=0 下约 1,318 ms，分支 FlyDSL 约慢 3%（推算，未实测）。固件更新前后 ASM+nkfix 1,947 → 1,351 ms（1.44×；两边都是 CHECK=1）。教程称前后"同代码同方法"，但这个倍数不纯是平台差异：1,947 来自 09-28 a0_p3a，该进程 step 2 起加载的 FlyDSL fwd 树 `_env.py` 把 `HIPBLASLT_TENSILE_LIBPATH` 改指到宿主库；两次之间驱动也从 7.1.1-2397345 换成了 7.1.0-2412954（见附录 A.2） | — | wt-llama31：`docs/gfx1250_llama31_8b_e2e/README.md` §3.2 §6–§7 |
| 10-06（窗口外） | 其它项目用 `hipblaslt-bench --algo_method all` 枚举所有解，碰到坏的 Tensile 解后挂卡 | A0-新固件 | 不再枚举全部解，只测预先筛过的短名单 | skill:`gfx1250-card-safety` §1 #14 |

#### d.2.2 nkfix 是什么、怎么装

- **问题**：镜像里的 hipBLASLt 只给前向 Linear 用到的 TN 布局（`Alik_Bljk`）发布了纯 bf16 调优库。反向的 dgrad（`Ailk_Bljk`）和 wgrad（`Ailk_Bjlk`）只能查稀疏的 GridBased 表，最近邻落在 N=1，结果选到 GEMV 用的 tile `MT32x16x32`，只有 50–80 TF/s。前向能到 1.5–1.9 PF/s，单次调用差 11–32×。
- **做法**：用 `TorchDispatchMode` 拦截 `aten::mm`，把反向 GEMM 的操作数物理布局改成前向那种，让它命中 `MT256x256x128`：
  - dgrad 在 B 连续时改为 `mm(A, B.t().contiguous().t())`。
  - wgrad 改为 `mm(A.contiguous(), B.t().contiguous().t())`。wgrad 的判据必须先于 dgrad 判断。
- **试过不行的方法**：
  - 零拷贝写法 `(BᵀAᵀ)ᵀ` 仍然落到坏 tile，只有 0.83–0.95×，因为这份库只有 TN 一种布局有好 tile。
  - 用 `torch.library` 覆盖 `aten::mm` 会无限递归。
  - 放进 `sitecustomize` 时 torch 还没加载，装不上。
- **数值**：op 级 SQNR 与原调用相同，都是 55.59–55.63 dB；99.996% 以上的元素差在 1 ulp 以内；3 次重复逐位确定。
- **版本**：
  - A0 v1（09-15）只改 dgrad。
  - A0 v3（09-16）加上 wgrad；规则 3 为 FlyDSL wgrad，走离线表，默认关闭。
  - B0 `nkfix_b0.py`（09-28）在 1255557f 基础上重写：
    - scratch 常驻，按 (device, stream) 区分。
    - 沿输出维分块，块大小 `NKFIX_CHUNK_BYTES` 默认 256 MiB，不拆 K 归约。
    - 转置用 Triton 分块 kernel `transpose_triton.py`，与 torch 拷贝逐位相同，快 5–7×。
    - `NKFIX_CHECK=0/1/2` 检查非有限值；`NKFIX_SHADOW` 在真实训练中做影子对照。
  - 教程使用的就是 B0 这一版。
- **安装**：
  - 教程方式：`primus-nkfix-hook.patch`，在 Primus 的 `primus/core/runtime/train_runtime.py` 的 `_run_trainer_lifecycle` 里加 14 行，`NKFIX_ENABLE=1` 时 import 并 `install()`。
  - B0 e2e kit 方式：shim 的 `primus_turbo/__init__.py`，配合 `E2E_NKFIX=1`。
  - A0 09-15/16 的 Primus 侧 hook 只备份在 `0915__opt/primus-side/`，没有推送。
  - `transpose_triton.py` 必须和 `nkfix.py` 放在同一目录。
- **开销**：每步约 70 ms；开 `NKFIX_CHECK=1` 再加约 33 ms/步。

#### d.2.3 e2e 效果（不同机器、时钟、层数，不可横比）

| 机器/时钟 | 配置 | 修复前 | 修复后 | 说明 | 来源 |
|---|---|---|---|---|---|
| A0-限频（负载约 1.0 GHz） | 8L，b4 s8192 | 6,128 tps（n=9，三模态，sd 3.91%） | v1 11,604（1.894×）→ v3 37,746（6.16×，n=5）→ v3+规则 3 46,374（7.57×，n=3） | 规则 3 有在线调优挂卡史，默认关闭 | `0915__opt/GEMM-NN-FINDING.md`；`0915__opt/DAY-0916-SUMMARY.md` |
| A0-限频 | 32L 生产配置（含 ASM bwd） | 1,984 tps（16.5 s/步，n=1） | 14,050 tps（约 2.33 s/步，7.08×，n=1；同会话对照 12,340） | 32L 的复现率从未有效测量过 | `0915__opt/RESULT-32L.md` |
| B0（满频；修复后训练中约 1.4 GHz） | 32L | ASM 2,016–2,022 tps | ASM 20,718（10.3×）；FlyDSL 19,762；Triton 14,563 | GEMM 占比从 96% 降到约 53%，另有 6% 是转置和检查 | `0927__b0/gemm/REPORT.md` |
| A0-新固件 | 32L | 刷固件前（A0-限频）ASM+nkfix 1,947 ms/步（09-28 a0_p3a；与修复后一行之间还差了 LIBPATH 改写和驱动版本，见附录 A.2） | ASM+nkfix 1,351 ms/步、24,262 tps（10-02，`NKFIX_CHECK=1`）；fly r16+s6 1,349 ms/步、24,283 tps（10-02，CHECK=1，未进分支）；分支 FlyDSL 1,362 ms/步、24,061 tps（10-05，CHECK=0） | 不开 nkfix 约 16 s/步、约 2k tps（B0 实测）。CHECK=0 与 CHECK=1 的行不能直接比（CHECK=1 约 +33 ms/步）；同为 CHECK=1 时 fly 对 ASM 是 0.999；按 +33 ms 折算，分支 FlyDSL 约比 ASM 慢 3%（推算，见 d.2.1） | wt-llama31 README §3.2 §7（:88-89、:184-191）；`1002__e2e/RESULT-e2e.md`；`0928__a0_repro/REPORT.md` §2、§5.3 |

#### d.2.4 NaN、`HIPBLASLT_TENSILE_LIBPATH` 与宿主库的风险

1. **nkfix 与 NaN（A0-限频，09-16）**
   - 带 nkfix 的运行 8/39 出现 NaN，不带的 0/35（p=0.004）。另有统计：带 nkfix 40 次中 7 次只出现 `grad_norm=inf`，不带的 36 次中 3 次。
   - 所有 NaN 都发生在 step ≤10，而且和当天的 MES TLB 故障混在一起。根因没有找到。
   - B0 重写版共 8 次 nkfix 运行、486 步（150 个不稳定步），0 次 NaN。上界只按其中 32 层的 7 次运行、466 步（138 个不稳定步）计算：按步 95% 上界是 2.2%/步；按次是 35%/次，还不足以排除 A0 的 21%/次，要压到 21% 以下约需 13 次干净运行（`0927__b0/gemm/REPORT.md` §0 §4）。
   - 因此 nkfix 保持 opt-in；长跑或排查时开 `NKFIX_CHECK=1`；只要出现 NaN 或 inf，这次运行的速度数据整次作废。
2. **库路径**
   - 默认搜索路径高了一层目录：`library/` 下只有一个 `gfx1250/` 子目录，402 个文件都在里面，结果是 `INVALID_VALUE`。09-23 S0-b 的目录表没有列这个子目录，因此转用了宿主库（见 d.2.1）。
   - `_rocm_sdk_devel/.../library/gfx1250` 是残缺版（326 个文件，无 `Kernels.so-000`、无 extop/transform），会 SIGSEGV，exit 139。
   - `/opt/rocm/lib/hipblaslt` 是空目录，容器的 ROCm 根本不在 `/opt/rocm`。
   - 回落到 rocBLAS 时 e2e 慢 8×，它的 fp32 Tensile kernel 还曾让卡 page fault。
3. **`TORCH_BLAS_PREFER_HIPBLASLT` 会被静默改写，至少有三处**
   - `sitecustomize.py` 用 `setdefault("0")`，是 09-17 我们自己加的。
   - `/etc/profile.d/zz-gfx1250.sh` 里 export 为 0，只在 login shell 生效，所以 `bash -lc` 会中招。
   - `tune_attention.py` 在 import torch 之前自己设成 0。
   - 规则：只用赋值，不用 `setdefault`；用 `bash -c`；在进程内打印 `torch.backends.cuda.preferred_blas_library()` 确认。
   - 另外，op-evolve spec 里的 `runtime.env` 实际不生效，必须写进 op 自己的代码或 exec 命令行。
4. **宿主库 `~/.local/hipblaslt-gfx1250`（宿主 ROCm 10.1.0 的拷贝，328 个文件）**
   - 只验证过方阵。
   - 与这些事件相关：B0 step 1 卡住（p1a/p1b 单变量对照，每边 n=1）；GEMM 突发慢约 20×；A0 09-28 的 NaN 和挂卡；10-02 挂卡。除 p1a/p1b 外，都只是相关性证据。
   - 10-02 起由 h85 禁用。
   - `skill:env-and-pitfalls` §1b 至今仍推荐宿主库，已过时，需要更新。
5. **卡上的 fp32 GEMM 参考**：会导致 A 类挂卡（refcache）和 B0 GPU2 故障（h50）。测试和闸门的参考一律放在 CPU 上算或走 refcache；refcache 的 sha 不一致时直接判 FAIL（见 d.5）。
6. **`hipblaslt-bench --algo_method all`**：会碰到坏的预编译解而挂卡，不要使用。

#### d.2.5 当前推荐配置（10-05 教程，A0-新固件实测）

```bash
# Primus 打 nkfix hook：git apply docs/gfx1250_llama31_8b_e2e/primus-nkfix-hook.patch
# 容器内：pip uninstall -y primus_turbo（镜像 editable 安装会遮蔽 checkout）；不要用 REBUILD_PRIMUS_TURBO=1
docker exec -w $PWD fa-repro bash -c '            # bash -c，不用 bash -lc
export TORCH_BLAS_PREFER_HIPBLASLT=1              # 显式赋值，绝不 setdefault
export HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250   # 只用镜像库
export NKFIX_ENABLE=1 NKFIX_CHECK=0               # 长跑/排查 NaN 时 NKFIX_CHECK=1（+~33 ms/步）
export PYTHONPATH=$TURBO/docs/gfx1250_llama31_8b_e2e/nkfix:$TURBO   # nkfix.py 与 transpose_triton.py 同目录
...'
```

- 检查点：
  - 日志里要有 `[nkfix] installed`，并且不能出现 `[nkfix] triton transpose unavailable`。
  - 稳态约 24k tps（A0-新固件）。稳定在约 19.4k 说明缺 `transpose_triton.py`；只有约 2k 说明 nkfix 没生效。
- yaml 配置：`flavor: 8B_flex`，`converters: ["primus_turbo"]`，`use_turbo_attention: true`，`compile.enable: false`，设置 `debug.seed`。
- 根本解决还得靠镜像方：补 NN/wgrad 的 bf16 调优库，修正库路径（`0915__opt/VENDOR-REPORT-hipblaslt.md`）。

---

### d.3 计时与尺子问题

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| ASM bwd 标杆 10.160 ms 是 shim 伪影 | 09-15..09-24 一直把 10.160 ms / 541 TF/s 当作 ASM bwd 标杆（A0-限频），FlyDSL bwd 因此被报为 ASM 的 0.92× | 09-15 那次测的是 `_AsmFwdAsmBwd` autograd shim，调用 `asm_backward` 时没传 `hip=`/`scratch=`，每次都做 3 次 hipModuleLoad 并新分配约 1 GiB scratch，还算进了 autograd 管路。修正后的路径（8.13–8.68 ms）当天就有，但没有传播开；这个数还被误标为"产品路径" | 改用 op-evolve beat 在同进程里普查：7.6766 ms（n=224 raw）/ 7.6769（n=146 去重），约 716 TF/s。n=177/7.6721 那一版也已撤回。规定只引用同进程 beat 的比值 | 进度被高估：0.92× 实际是 0.70–0.72×。11.726 ms/层、4.76×、1.74× 一并作废。开销拆分（+1.15 / +1.42 / +0.01 ms）只有文字描述，只能引总值 | skill:`baselines.md` §0 §5；77963d7e；`0924__flydsl/DAY-SUMMARY.md` |
| 逐次交错计时与分块计时（B0，09-28） | 同一份代码，换个位置或换个同进程 arm 组合，读数就差 2–10%。r6/beat 交错测 1.29×、稳态 1.035×；bwd beat/current 交错测 0.7618、分块测 0.7878 | 每次调用都继承前一个 arm 留下的功耗/时钟状态，2 GB flush 也消不掉，结果偏向功耗高的 arm | 改为分块计时：每 arm 先跑 4 次再计 9 次，回文排序（fwd 03:40 起用 h40/h41，bwd 约 05:00 起用 h66）。每个排名进程带一个 A/A 副本（fwd ±0.19%，bwd ±0.05%），偏离超过 0.5% 整个进程作废（h35） | 三个假赢：r6 报 +5.7%（实际 +4.6%）；L12 报 +6.4%（实际 −0.4%）；L21 报 +6.3%（实际 +0.3–0.6%，两份文档口径不同，都在噪声边缘）。r6 之前的 %ASM 系统性偏低（fwd 约 25%，bwd 约 3.4%）。新旧 harness 的绝对 ms 不能比 | `0927__b0/ruler/REPORT.md`；`0927__b0/ruler/bwd/REPORT.md`；`0927__b0/REPORT-0928.html` §6 |
| ASM 之后的 I-cache 惩罚（A0-限频 fwd job，09-25..27） | 同进程里先跑 aiter ASM，再跑 FlyDSL kernel，I$ miss 从 0 变成 808，冷启动固定多 25–42 µs；fast/proxy 的中位数随臂的位置漂移 | beat 与候选在同一进程里回文排列（A B C C B A），紧挨着 beat 的臂吃亏。这和功耗、循环次数、代码体积都无关。B0 审计后把同类现象解释为"状态继承"，两种解释对应的修法相同 | h28：候选和冠军在不含 beat 的进程里比较，至少轮转 3 个 session；beat 放到单独进程测 | r3/r6/r8 的判定都受了影响；r8 nodelay 的 proxy 0.956（去掉 beat 后为 1.004）被错误拒绝 | `0925__flydsl/fwd-job/hint.md` h28；skill:`fwd.md` |
| 用 randn 输入还是真实数据（B0 09-28；A0 10-02 复核） | B0：r6 投机 softmax 在 randn 上快 4.6%；但训练中 FlyDSL fwd（r6）对 ASM 逐层是 1.19（第 0 层）到 1.72（第 7 层），而且越训越慢（50.2→55.3→59.2 ms/步）。op 级用真实输入、在刚跑完 GEMM 的时钟下是 1.45–1.67；同一批数据用分块尺子测是 1.15–1.30，randn 分块是 1.02 | 真实 q/k/v 的 score 标准差为 21–53（randn 约为 1），13–25% 的 tile 步会触发重算（randn 为 0%） | 用 step 43 的 6 层真实 dump（`/home/lihuzhan/_prof_dump/qkv_call0*.pt`）作第二把尺子（h45/h47），r18 起两把尺子都报。关掉投机的 r13ns 经 refactor h44 采用：randn 上慢 4.8%，真实数据加训练时钟下快 12–23% | fwd r16 起，job 的 randn 读数按设计会偏低约 4.8%，r16 前后的 %ASM 不能比。A0 09-30 的 s1–s6 只用 randn 测（当时以为 dump 只在 B0，见 d.7）。10-02 在 A0-新固件上用同一批真实 dump 补测：真实数据与 randn 一致（bwd s6/ASM blk 0.972，randn 0.957；fwd r16/ASM 1.081，randn 1.080），r16/s6 的结论在真实输入上成立。拉开差距的是工作点：gb 下 ASM 几乎不动，FlyDSL 两个方向都变慢，bwd 从领先 2.8% 变成落后 3.8%，fwd 从落后 8% 变成落后 35%（见下一行） | `0927__b0/profile/REPORT.md` §0 §1 §2.2；`0927__b0/fwd-nospec/REPORT.md`；`1002__e2e/RESULT-realab.md` |
| 训练工作点的时钟（gb 尺子） | B0 训练中 attention 跑在 1,250–1,690 MHz（空闲 2,356），整板约 2.13 kW。B0 拟合的 1350/2350 MHz 时间比：FlyDSL fwd r6 1.25（r²=0.28）；ASM fwd 0.97（r²=0.01），基本不随 sclk 变。A0-新固件 10-02（真实数据）三种尺子的比值（blk / gb / e2e）：fwd r16/ASM 1.081 / 1.349 / 1.263；bwd s6/ASM 0.972 / 1.038 / 0.974；bwd r29/ASM 1.206 / 1.341 / 1.290；s6/r29 0.806 / 0.774 / 0.754 | A0-新固件上，blk 尺子的调用窗口约 1.45–1.5 GHz；gb 尺子按"GEMM 刚跑完"的工作点设计，调用窗口约 1.26–1.39 GHz（realab 记约 1.28 GHz）；A0 e2e 整步 sclk 中位数约 1.50 GHz。A0 上没测过训练中 attention 窗口的时钟，1,250–1,690 MHz 是 B0 的读数。在功耗墙下，耗时还随数据变化：A0 r24 全零操作数时间少 34%，cycle 数反而多 7.7% | 设计了 gb 尺子：每次计时前跑 10 个镜像库 bf16 GEMM 突发，proxy/prod 按 gb 计分。尚未安装，需要用户批准。另用 cycle（GRBM_GUI_ACTIVE/8）作第二把尺子 | 4 项比值里 3 项（s6/r29、r29/ASM、r16/ASM）是 gb 更接近 e2e；s6/ASM 是例外，gb 偏悲观约 6%（训练中 s6 比 ASM 快 2.6%）。单一 op 尺子代表不了训练 | `1002__oe/RULER.md` §0–§1；`1002__e2e/RESULT-realab.md`；`1002__e2e/RESULT-e2e.md`；`0927__b0/profile/REPORT.md` §0 §2.1；`0930__roofline/REPORT.md` |
| e2e 里 ASM 比两把 op 尺子测的都快（未解释） | A0-新固件 10-02：e2e 中 ASM 每层 fwd 1.07–1.11 ms、bwd 5.11–5.29 ms，比 blk 和 gb 两把尺子的读数都短（realab 6 层真实数据：fwd blk 1.209 / gb 1.137 ms，bwd blk 5.475 / gb 5.530 ms）。B0 09-28 有类似迹象：ASM fwd 不随 sclk 变，却随条件变（iso 1.22、blk 1.45、eburst 1.13、layer 1.00 ms） | 时钟解释不了。一种可能：训练中 q/k/v 刚由前面的 GEMM/RoPE 写出，还热在 MALL 里，而两把尺子每次调用前都会冲掉缓存。没测过 | 无；RULER §8.1 列为未决 | op 尺子和训练之间除了时钟还有缓存热度这个变量，ASM 的绝对 ms 不能直接外推到 e2e | `1002__oe/RULER.md` §1 末条、§8.1；`1002__e2e/RESULT-realab.md`；`0927__b0/profile/REPORT.md` §2.1 |
| fast 形状中位数噪声导致假接受 | A0-新固件 bwd r24（09-30）被判为 1.4263×（score 1.32077 对 0.91238），其中 fast 2.275×，prod 只有 0.9986。09-17 A0 上，两份相同目录在 fast 迭代 20 次时中位数差 7.8% | fast 每次约 55 µs，受 launch 开销主导，中位数被调度噪声左右：两个臂跑的是同一个 k_dq_sp，min 只差 0.05%，median 却差 56%。验收 gain 是三个 shape 的等权算术平均：(2.275+1.0051+0.9986)/3=1.426（几何平均只有约 1.317）。`0930__bwd/PLAN.md` D7、bwd hint h83 和 s:69864fc9 都写成"几何平均"，写错了；框架里只有 target_met 用几何平均（见 d.5） | D7：fast 只取 min，gain_weights 设为 prod 1 / proxy 0.25 / fast 0（4d61867f，h83）；迭代次数 20 → 51/101 | 新冠军在 prod 上与 s6 等价，没有造成损失。h83 因为缺索引行，到 10-02 才真正生效（见 d.5） | 4d61867f；20baa45e（commit message：gain 为三 shape 算术平均）；OE bwd job `rounds/024/3-act/act.yaml`、`job_context/state.yaml`（round 24 gain 1.4263）；s:69864fc9 09-30T13:41Z |
| 噪声地板 | 曾把 1.57% 当噪声地板。同一份代码跨 session 从 382.73 跳到 404.06（+5.6%）。B0 e2e 进程中途整体掉档约 5%（2,015 → 1,918 tps）。A0-限频 e2e 呈三模态，sd 3.91% | 1.57% 是 HipKittens bf16 GEMM 阶梯的地板。本算子同 session 的地板是 0.24–0.66%，fwd 跨 session 漂移约 1.5%。e2e 三模态来自 hipBLASLt NN 的 GridBased 选解，nkfix 后 sd 0.42%。B0 掉档原因未查 | 只用同 session 回文比值；min_gain 设 0.007。e2e 只用同进程 ABBA/BAAB 配对，剔除 step 1–7 和 profile 步 | 多轮判定要回头重看 | skill:`env-and-pitfalls` §3；`0923__flydsl/hint.md` h38-CORRECTION/h56；`0927__b0/e2e/RESULT.md` §2 |
| 同 session 回文 A/B 的槽位偏差与 beat 异常 | A0-限频 fwd r3：同一个 A 臂的 proxy 只因槽位不同就读到 858/759/718（单 session 虚涨 15%）。bwd r8 那个 session 里，beat 在 fast 只有 21.83（正常约 50.70），proxy 461.57（正常 573.31），score 因此从 0.42321 跳到 0.60244 | 5 个臂回文排列时，fast/proxy 随槽位有 ±8% 的变化；beat 在小形状上偶尔异常 | 臂的顺序轮转 3 个 session 后取均值；排名只看 prod 和同 session 的 gain；score 不跨 session 画图 | 这次 score 跳升不是真进步 | fwd job `rounds/003/1-opt/act.yaml`；`0922__flydsl/ROUND3-CLOSEOUT.md` Round 8 |
| JIT/配置缓存导致"改了没生效" | B0 r16 refactor 只改了模块常量 `SPEC_STALE_MAX`，`/tmp/flycache` 仍返回 r13 的二进制，逐字节相同的 A/A 副本差了 6.6%。09-23 n_block 扫描三个臂的时间和 SQNR 完全一样，构建耗时 0.0 s。09-11/13 aiter bwd 配置扫描结果平坦（27.45–27.49 ms） | FlyDSL 0.3.4.1 的 `_jit_function_cache_key` 只哈希函数源码和闭包标量（h46/h72）。n_block 是 def 时绑定的 kwdefault，又被 `functools.cache` 缓存。aiter 的 `_get_config` 带 lru_cache，wrapper 又用 `from … import` 绑定了旧引用 | 每进程、每 arm 用新的 `FLYDSL_RUNTIME_CACHE_DIR`，或清掉 `/root/.flydsl` 和 `~/.cache/comgr`。实验开关写成参数。在 launch 处验证：看 kernel 名、`--pmc` 描述符里的 VGPR、构建 <0.5 s 就报 SUSPICIOUS。扫描结果平坦就说明覆盖没生效 | 受影响的结果整批丢弃。陈旧缓存会返回旧的、正确的 SQNR，是最危险的假阴性 | 6c798efd；`0923__flydsl/STAGE2-FWD-SWEEP.md:24-38`；`0913__opt_plan__claude/phase2/DECISIONS.md` D7 |
| 时钟、功耗"已排除"的结论被撤回；限频代价的几种口径被混用 | 09-24 宣布 sclk 一直是 1100 MHz、853 W，排除了限频。09-13 估限频代价为 1.65×，09-14 又改称"实测约 1.93×，1.65× 作废"。09-15 的 JIRA-TRACE-ANALYSIS（后来进了 `PROGRESS-REPORT-0917.html` 和 `0922_summary/ASM-ATTENTION.md`）说 A0 与 B0 的 attention 只差约 13%（A0/B0：fwd 1.118、bwd 1.150、total 1.146），并称两边是"同一个 ASM 反向实现" | pw.sh 传了 `--arms cur_a`，benchmark.py 以 exit 2 退出，输出进了 /dev/null，wait 却返回 0，那 60 行其实是空载数据。真实 264 个样本显示 prod 窗口 998–1029 MHz，1100 只是空闲时钟。1.65× 是按 MAX_CLK 之比做的估算，不是实测。1.93× 是 B0 五行同版本对照（total）1.88–2.01×（中位 1.93×），即 A0-VR→B0 的跨卡比（B0 自身也有 VR 告警，代价约 9%），不是 A0 自身的限频代价；A0 自身刷固件前后同代码实测为 op 1.24–1.67×、e2e 1.44×（e2e 另有附加差异，见附录 A.2）。所以 1.65× 不算作废，只是估算。"13%"有两处错：A0 的 bwd 10.160 是被污染的 shim 数（见本表第 1 行）；B0 的 8.835 是融合 Triton bwd（09-14 B0 还没有 ASM bwd，HANDOFF 把它列为待办 T2，ledger 记录也都标 vendored fused backward）。fwd 1.118×（两边都是干净的 ASM fwd）成立；同代码的 bwd（两边都是融合 Triton bwd）由两份 RESULTS 推算为 17.735/8.835≈2.01× | 测量脚本要断言被测对象真的跑了（逐 shape echo RC）。每次计时附 sclk 见证。不做跨机换算。hwmon sclk 是 10–14 ms 平均值，看不到 kernel 内部时钟，改用 PMC 的 GRBM_GUI_ACTIVE/8 | 一条假结论进过 facts.md；"13%"和"A0 11.726 ms 在健康卡上约 10.2 ms"一起进过 0917 进度报告和 0922 总结 | `0924__flydsl/DAY-SUMMARY.md`（RETRACTED 段）；`0915__opt/JIRA-TRACE-ANALYSIS.md` §五；`0922_summary/ASM-ATTENTION.md` 页首；skill:`baselines.md` §8；`0914__campaign/HANDOFF.md:246`；`0914__campaign/ledgers/*.jsonl`；`0914__repro__c07/RESULTS.md:126-129`；`0915__repro__c07/RESULTS.md:17`；`0930__roofline/REPORT.md` |
| e2e 测到的不是想测的东西 | 09-13 turbo 245 tps 对 flex 244 tps。09-14 turbo_asm 13,970 对 turbo_noasm 13,930，并据此得出"turbo 比 flex 快 1.459×"。实际上 ASM fwd 在训练中从未被调用 | `converters: []` 让 turbo attention 根本没进 e2e，两边其实都是 flex。镜像里 editable 安装的 primus_turbo 通过 `.pth` MetaPathFinder 压过了 PYTHONPATH。`AITER_LOG_LEVEL=ERROR` 把横幅压掉了，"日志里没有 aiter"这个判据本身无效 | 容器内 `pip uninstall -y primus_turbo` 并打印 `__file__`；打开 converter；让门在入口处写 trace 文件 | 09-13/14 所有 turbo attention 的 e2e 结论作废（13,204 tps、1.459×、"38% 杠杆"） | `0914__campaign/RESULTS.md:277-318`；`0914__campaign/HANDOFF.md` §4 |
| e2e 统计方法 | "ASM bwd 让训练慢 6.78%（n=1）"、"快 3.42%"、"OFF 很稳定，0.06%"先后被推翻。MFU 显示 178–249% | 把单次运行内的步间抖动当成了跨运行方差；没有固定随机种子；三模态。torchtitan 的 `get_peak_flops` 对 gfx1250 兜底成 A100 的 312 TF/s | 设 `debug.seed 1234`；用 `bin/ab_replicate.sh` 交替跑 n≥9，再用 `ab_summary.py` 汇总；步数 20 → 10（step 4–8 的中位数差 ≤0.22%）；只引用 tps/tflops | 所有 MFU 列都不可引用 | `0915__opt/E2E-AB.md:180-260`；`0915__opt/GEMM-WGRAD-FINDING.md:78-94` |
| 卡上争用 | B0 09-14 四卡满载时，单卡计时漂移最高 50%（11.2–17.1 ms），还出现了假 SQNR 失败。B0 09-27 邻卡跑持续 GEMM 时，被测卡 attention 慢 3–11×，fwd/bwd 的比值甚至反转。A0 r23 遗留的 meas2 与框架的 validation 同时占卡约 2 min | 整板共享功耗和温度，同一 hive；孤儿进程 | 用独占窗口加交替测量；禁止在任何卡上跑持续 GEMM 压测；`rocm-smi --showpids` 和 KFD 持有者必须能一一说出来；放弃一个 session 时同步按 PID kill | — | `0914__campaign/RESULTS.md:84-97,217-233`；`0927__b0/interference.md`；6029b430 |
| 测量驱动/harness 自身的 bug | "turbo 仅配置调优"记成 25.695 ms（实际 18.916，错 1.36×）。flex 锚点测成 37.2 ms（应为 15.31，错 2.43×）。waves_per_eu=0 被判为无效旋钮。harness 测到的是镜像里的 Primus-Turbo。ledger 的 impl_note 是硬编码。clock_probe 9 s 的窗口变成约 55 min 的卡时。前向有 0.9 ms 差距 | `bash -lc "… $*"` 遇到分号截断了 `--tune`。flex 走 inductor 默认启发式，选了 num_warps=8。harness 把 0 当非法值拒掉。子进程的 sys.path 不对。impl_note 在运行前就写死了。enqueue 没有背压（多发 372×）。那 0.9 ms 里约 0.7 ms 是 host 侧开销 | 改成 `bash -lc 'cd "$0" && exec … "$@"'` 并回显实际收到的配置；flex 用 max-autotune；加 `_ZERO_MEANS_UNSET`（满频下 +4.2%）；ledger 记录 turbo_path；用 SQNR 指纹审计（ASM dk 50.61 / dv 50.83，融合 bwd 52.31 / 52.71）；加 Throttle 包装；两种测法结果不一致时，视为差距未确立 | 多行阶梯数据返工 | `0914__repro__c07/RESULTS.md:81-113`；`0915__opt/LEDGER-AUDIT.md`；OE:`output/0911__fa_gfx1250_phase1/E1.md` §3 |
| FLOP 口径不统一 | 同一个 2.356 ms 有时读成 947.8 TF/s，有时读成 933.5。早期 TF/s 用的是 fwd+bwd 合计 7.697e12。bwd 有"7-GEMM 实发"和"5-GEMM 名义"两种算法 | 旧 harness 的 fwd FLOP 约 2.233e12；`op_flops.py` 用 fwd 2.199292e12、bwd 5.498229e12 | 统一用 `op_flops.py`；引用时同时给 ms；issued TF/s 和算法 TF/s 不混在一起比 | — | `0925__flydsl/fwd-job/NOTES.md:299-304`；skill:`baselines.md` §1 |

---

### d.4 工具链问题

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| FlyDSL 三个版本并存 | 镜像自带 0.2.4（Primus-Turbo main 固定这个版本，用于 gfx950 树）；0.3.2 是 aiter 固定的版本（`~/.local/flydsl032`，bwd job 用）；0.3.4.1 在 `~/.local/flydsl0341`（fwd job 和 e2e 用）。0.3.2 在 PYTHONPATH 上时，`import primus_turbo` 报 `cannot import name 'buffer_ops'`。B0 smoke 报 `AssertionError: flydsl 0.2.4 @ /opt/venv` | 0.3.x 删掉了 `flydsl.expr.buffer_ops`。Primus 的 base_env 把镜像 site-packages 排在前面，而 `_ensure_flydsl0341()` 看到路径已在 sys.path 里就不再前移。用 pip --target 装 matplotlib 时，numpy 2.5.3 覆盖了 torch 依赖的 2.4.1 | `_env.py` 先 `sys.path.insert(0, dir)` 再 import，并断言版本和 `__file__`；job 进程里永远不 import primus_turbo。0.3.2 与 0.3.4.1 在同 session A/B 下性能持平：fly/asm 分别为 0.657–0.664 和 0.664–0.666（A0-限频）；r29/s6 在两个版本下 ISA 逐字节相同 | Stage 4 的派发接线必须重新设计。cherry-pick 70607aa9（兼容补丁）出现 13 个文件冲突，已中止 | skill:`env-and-pitfalls` §1a；`0917__flydsl/STAGE1-FWD.md` §4；`0925__flydsl/fwd341/ab_prod.log` |
| 0.2.4 能力不足（产品移植） | 09-17 "0.2.4 加 4 个 shim 即可"的结论被推翻，报错 `state variable 'result' is list`。09-30 产品 fwd 在 0.2.4 上，toy 和中等 shape 都出现 memory access fault（进程级） | 0.2.4 的 ast_rewriter 不接受 list 作为 stateful-if 的状态变量。0.2.4 缺 `fx.ceildiv`、`to_llvm_ptr`、max/min、`shuffle_xor`、`global_store_async_from_lds_b128`。手写 intrinsic 时，0.2.4 的 LLVM 把 LDS 操作数编成了全局地址，fault 地址就是 LDS 偏移。0.2.4 的 rocm.py:61 对 gfx1250 传了 wave64。COMPILE_ONLY 下 0.3.4.1 的 `flyc.compile` 会返回 None，0.2.4 没有这个提前返回 | 加兼容层 `_flydsl_compat/common.py`。dynamic-if 的状态改成具名变量。用特性检测 `HAS_ASYNC_LDS_STORE`：没有该 op 时改用 buffer_store 版 O writer v1，flydsl ≥0.3.4 时自动回到 v3。ISA 门断言 wave32、0 spill | fwd 慢约 1.3%（0.3.4.1 下 v3 1.279 ms，v1 1.296 ms，A0-新固件），bwd 慢约 1.8% | `0917__flydsl/API-DELTA.md`（RETRACTION）；`0930__port/PR_BODY.md:45-53`；s:fdc2534d 09-30T03:57–04:25Z |
| FlyDSL API 稳定性 | 维护者指出 Primus-Turbo 用了内部 API。审计发现：私有 `_raw/_ir` 约 90 处，llvm load/store，raw fastmath，对 `exe._cf` 的私有写入，已废弃的 shuffle_xor | 09-25 的清理只做在副本 op_clean 上，冠军是从未清理的 op0341 演进来的（fwd 差 181 行，bwd 差 590 行）。v0.2.4 本身没有 api_stability 策略 | 按 0.3.4.1 的策略重新审计（`API_AUDIT_SUMMARY.md`）。T1 已落地（1f74e662、58d5d798），两个版本下 ISA 都不变。T2 要等升级到 ≥0.3.4。T3（WMMA、TDM、ds_load_tr16、s_wait_dscnt、sched_barrier）没有稳定替代，标为 UNSTABLE(gfx1250)。`global_store_async_from_lds` 的 wrapper 在两个版本里都有 mask/cpol 参数顺序 bug | 在 gfx1250 上，FlyDSL 的部分功能只能依赖不稳定 API | `0930__port/API_AUDIT_SUMMARY.md` |
| gfx1250 上的 API 陷阱 | 能编译，但运行时静默出错或挂起 | `is_rdna_arch("gfx1250")` 返回 False：0.2.4 会因此误判成 wave64；0.3.x 中它只影响 V# flags，导致缺 OOB_SELECT，buffer descriptor 成了 CDNA 形态 0x00027000。`rocdl.s_waitcnt` 在 gfx1250 上会抛错（计数器是分开的）。`BufferAtomicAdd` 生成 SCOPE_CU，8 个 XCD 之间会静默丢更新。V# 的 num_records 以 128 B 为单位，int32 计算在 prod 形状下会回绕成 0。TDM 不加 pad 会有 64 路 bank 冲突。`make_tdm_atom(num_warps=8)` 用在 4-wave 下会静默漏载。aiter fwd 的 buffer manager 在 `num_waves != 8` 时抛 NotImplementedError。`fastmath=fast` 让 LLVM 删掉了因果 −inf mask。Python 的 `if` 被改写成 `scf.if` 分支函数后出现 NameError（需要 `const_expr`）。语句级 `for range` 里不能往 list append。`exec()` 会破坏源码内省 | 用 `UniversalAtomicAdd(Float32, SyncScope.Agent)` 得到 SCOPE_DEV；用裸 `fx.barrier()` 或 `rocdl.s_wait_dscnt`；用显式谓词和钳位，不依赖 descriptor 截断；extent 用 i64 计算；用 aiter 的 `tdm_ops_gfx1250`，并显式传 pad；把 `num_waves!=8` 记为路障，而不是记成"4-wave 更慢" | 造成多轮 build failure；h41 误判 atomic scope，由 4e02d3ae 更正 | skill:`env-and-pitfalls` §7；skill:`flydsl-api.md`；`0921__flydsl/A0-A1-ZERO-GPU-COMPILE.md` §3 |
| COMPILE_ONLY 的 arch 变量 | 只设一个 arch 变量时，会静默生成 target 混杂的 ISA。没有设备也没有 env 时，`get_rocm_arch()` 回落到 gfx942。fwd r5 漏设 `FLYDSL_DUMP_IR=1`，grep 改读 stdin，600 s 超时 | `ARCH` 决定编译后端，`FLYDSL_GPU_ARCH` 决定 buffer descriptor 的写法 | `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 HIP_VISIBLE_DEVICES=-1 FLYDSL_RUNTIME_CACHE_DIR=…`，并断言结果是 gfx1250；从 `*_final_isa.s` 读 vgpr、spill、LDS。bwd 每次 build 约 4 min | `rounds/002/_scratch/screen.py` 写死了三个 kernel 签名和 block=(32,1,1)，新 kernel 会被静默漏检 | b1572877；skill:`env-and-pitfalls` §6 |
| profiler 能用到什么程度 | rocprofv3 PC sampling：09-11 两次 fault、一次挂卡；09-24 r17 在 prod 上 fault；刷固件后 1.3.2 启动时直接拒绝。`--kernel-trace/--runtime-trace/--hip-trace` 拿到 0 条 dispatch，刷固件前后都一样。`--pmc` 能用，但 51 个计数器只有 9 个可信，`SQ_VALU_WMMA_FLOP_*` 恒为 0，没有 stall 和字节计数器。VGPR 列是 ISA 实际值的一半（ASM 显示 512，实际 1024）。ATT 在旧固件下抓不到 FlyDSL JIT，09-29 刷固件后能给出逐指令 Hitcount/Latency/Stall（FlyDSL 和 ASM .co 都行，但 csv 的 Latency 列会把共发射指令重复计入）。A0 上 kineto 每个 profile 步只记录到 4–15 个 kernel。`--stats` 在 teardown 时报 `corrupted double-linked list`。`--pmc` 配 1 s 不同步的 warmup 会积压几万次 dispatch，看起来像卡死。rocm-smi 在空闲卡上显示 13% busy。hwmon `power1_input` 在负载下偏低约 800 W。llvm-objdump 解不出 ASM 里的 TDM 指令（显示为 `.long 0xd031…`） | 工具链对 gfx1250 的支持不完整，再加上旧固件的限制 | 计时用 CUDA event，kernel 名和时间戳用 PMC csv；VGPR 从 ISA 读；按 ATT 配方采集（`--att-library-path …/_rocm_sdk_devel/lib --att-target-cu 1 --kernel-include-regex`）；`--warmup-seconds 0`；kernel 数 <1500 的 trace 判为 INVALID；功耗和 sclk 改为 100 Hz 读 raw gpu_metrics | PC sampling 永久禁用；静态 ISA 指标只能当 build gate，不能用来排序（d.5） | skill:`env-and-pitfalls` §2c；`0930__bwd/probe/P1-RESULTS.md`；`0930__roofline/REPORT.md:158-160`；`1002__e2e/RESULT-e2e.md` |
| Triton、inductor、torch 的问题 | Triton `num_warps=16` 会挂起，日志 0 字节。inductor autotune 在本卡上报 `hipErrorLaunchFailure`。融合 bwd 在 torch.compile 下先后撞上 `num_ctas` kwarg 被拒、inductor 生成的 f-string SyntaxError。upstream main 在 torch 2.11 下 import 失败（`register_opaque_type`）。8B flavor 关掉 converter 后，SDPA MATH 物化 [4,32,8192,8192]，导致 SIGBUS | 编译器和运行时对 gfx1250 支持不足；AOTAutograd 编译前向时会顺带追踪反向 | 网格里排除 num_warps=16；`compile.enable: false`；ad67a2cc 改为 num_ctas=1 时不传该参数；从分支拷 `low_precision.py` 过来；用 `8B_flex` 并打开 converter | 51 个测试都不开 compile，e2e 又因 `converters: []` 走不到这条路径，两道门的盲区正好错开 | skill:`gfx1250-card-safety` §1；`0914__campaign/HANDOFF.md:86-101`；wt-llama31 README §1 |
| 其它路线的判定反复 | FlyDSL Gate A 在 09-14 判为 BLOCKED（gfx950 的 MFMA、ds_read_tr16_b64、permlane32_swap 在 gfx1250 上都报 `Cannot select`），09-16 撤回。09-13 曾判断"没有任何 FlyDSL 版本支持 gfx1250"，因为只看了 0.2.4。HipKittens udna1 门判 RED：256 个元素错 126 个，原因是 3 条编译期障碍 | 探针回答的是"gfx950 的 wave64 MFMA kernel 能不能直接 retarget"，不是"FlyDSL 能不能用于 gfx1250" | 改为逐 kernel 移植到 gfx1250 的 WMMA 和 `ds_load_tr16_b128` | 18–36 工程日的估算先被"确证"、后又撤回 | `0914__campaign/RESULTS.md:148-184`；`0914__campaign/HANDOFF.md:133-202` |

---

### d.5 op-evolve 框架问题

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| 精度门设得连 baseline 都过不了 | fwd 的 50 dB 门连未修改的 baseline 都过不了：7 个边界 case 在 49.82–49.99 dB。r1 longest-first 在 prod 上快 8.6%，因此被拒。另外 `op/ut/gates.py` 里硬编码了 `GATE_DB=50.0`。0923 版 fwd spec 的 `precision_gate` 被并进了 `refcache: >` 块标量，读出来是 None | 门槛没先拿 baseline 校准；同一个门在两处定义 | 在 resolved yaml 里改为 49，不 bump spec；`GATE_DB` 手改为 49.0；修好 spec | r1 被拒；r5–r10 在 state 里记成 passed=false，账面失真 | skill:`fwd.md:107,112`；skill:`op-evolve-ops` §10 |
| UT 门一直空转 | pytest 输出"no tests ran in 0.72s" | `seed=hash(shape)&0xFFFF` 受 PYTHONHASHSEED 加盐影响，每个进程都不同，缓存永远命中不了；文件里也没有 `test_` 函数 | 93ad3b89 改成 seed=0；r11 是 UT 第一次真正检查到东西的一轮 | — | 93ad3b89；`0923__flydsl/CAMPAIGN-FINAL.md` |
| 验收规则：算术平均 + min_gain 0 + 固定 band | A0-限频 bwd r15 g47 在 prod 上慢 3.1%（0.9688），却因 gain 1.0035（由 fast 撑起）被晋升为冠军。r5 账面 1.15×，其实是 fast 拉起来的，prod 只有 +1.6%。fwd r10 nodelay 在 prod 上 4/4 次 +0.83%，平均后只剩 1.0022，被拒。fwd r1 fast 0.934 低于 band 0.993，否决了 prod 的 +8.6% | acceptance 对三个 shape 等权平均；band = max(0.95, 1−min_gain)，对每个 shape 一视同仁；min_gain 不是 tune 能改的参数 | 20baa45e 手工回滚冠军；h7/h24 规定按 prod 排名、prod 低于本轮地板就不 ship；加 `evolve.gain_weights {prod 1, proxy 0.25, fast 0}` 和 `shape_band`（fast 0.90，proxy 0.98）；min_gain 设 0.007 | 改规则要手改 final.yaml 再普通 resume | 20baa45e；skill:`op-evolve-ops` §5；skill:`fwd.md:101-112` |
| 冠军记录混入未晋升的 arm | B0 fwd r9 BG 在三个 shape 上都胜过 r6，却得 0.9984 被拒。r19 在真实 dump 上 +1.5%，被 r18 一个未发布 arm 的记录挡住（0.9998）。bwd champions 里 fast/proxy=30 是被拒的 r30 的记录 | 每个 shape 的 best-ever 在被拒的轮次也会更新 | operator 用 refactor hint 手工晋升（h39 → r10，h44，h48/h68）；建议 champions 只记已晋升的轮次 | — | `0927__b0/OP-EVOLVE-SUGGESTIONS.md` #1 |
| target_met 用几何平均判定 | B0 bwd r27：fast 比 beat 快 1.66×（ASM 在小 shape 上慢），盖住了 prod/proxy 约 0.78× 的落后，geomean 1.007，job 自行结束 | 达标判据是三个 shape 的几何平均 | 经用户批准改为 h69：proxy 和 prod 各自 ≥ 同进程 beat，fast 只报告。A0 之后又设 `BEAT_MARGIN=20`（目标 1.20×），避免第一个达标的轮次就结束 job | — | `0927__b0/bwd-hint.md` h69；`0930__bwd/oejob/README.md` |
| hint 送不到 | fwd hint 用 L#/f# 编号，而 `core/hints.py` 只解析首格为 `h<N>` 的表格行，结果一条都没读到；hint.md 还是在 r1 开始之后才放进去的。deep 轮的 profiling/plan/act 根本不读 hint.md（ae6dd218）。deep_loop-trim 补丁（OE 58b2134）把 bwd 专用的 CAMPAIGN CORRECTIONS 写死在 3 个 preamble 里，所以 fwd 的 deep 轮（A0 r5/r10、B0 r15/r20）读到的都是 bwd 的内容。h83 只有段落、没有索引行，从 09-30 到 10-02 一直没生效。`core/route.py` 在整行里匹配类型，条件列里的 "must be predicated" 被当成了 must，fwd r1 因此报 RouteError | 解析器只认索引表；deep 模块不 import hints；preamble 被所有 job 共用 | 改成 id、type、title、status 四列的索引表；每个 hint 都必须有索引行，并用 `core/hints.py` 自检；route.py 只读 type 列；准备了 fwd 版 deep prompt 补丁和切换脚本 | 无法确认哪些轮次真的读到了哪些规则 | skill:`fwd.md:34-35,106-109`；s:69864fc9 10-02T11:41Z；`1002__oe/FWDJOB.md` |
| deep 轮"盲飞" | h4（09-22）以 rocprofv3 下出现 APERTURE_VIOLATION 为由，禁了 profiling 和 deep 轮；09-23 S0-c 查明那是 eager Tensile 造成的混淆。stock 的 01_select 强制用 `--kernel-trace`，拿到 0 行。r17 deep 轮的 PC sampling 在 prod 上 fault。deep act 只允许一个候选 | 禁令建立在被混淆的证据上；框架默认的 profiler 用法在 gfx1250 上无效 | af33a8fd 启用 deep 轮（ATT 已测）；82420bdf 禁 PC sampling；09-30 刷固件后把 ATT 接入 04_thread_trace；fast 轮改用 PMC csv（h80） | — | af33a8fd；82420bdf；`0930__bwd/REPORT.md:35` |
| `resume --config` 会重建一切 | `apply_spec` 会 bump spec_version，导致两个 setup 阶段重跑（约 58 min，op_setup 实测 2850 s），并重新生成 `op/{baseline,eager,ut,benchmark.py,validation.py}`，把 refcache 和手工补丁全部冲掉。min_gain 不是 tune 的选项；tune 不改 state.yaml。bwd 的 `validation.py` 硬编码了 `BEAT_MARGIN=0.0`，所以 `tune --beat-margin` 只改了文字，不改门 | 框架设计如此 | 只用 `op-evolve tune`，或者先备份 final.yaml（`.bak.<原因>`）再手改，然后普通 `resume` | 用这个办法，bwd 手改三次后 spec 仍是 v000 | skill:`op-evolve-ops` §4 |
| refcache 与在卡上算参考 | 闸门、benchmark、check_determinism、bitwise.py、test_correctness.py 共 5 处在卡上跑 fp32 Tensile 参考，这是 A 类挂卡的来源。B0 r24 起，`ut/common.py` 只改了一行 toy 形状，common_sha 就对不上了，于是 gate 每轮都打出"IGNORED … recomputing"，在卡上重算 prod 参考（正是 09-22 那次断电的触发路径）；A0 09-30 也一样。B0 fwd r20 act 自写的脚本也在卡上跑 fp32 参考，GPU2 因此 fault | 参考计算设计在 GPU 上；sha 不一致时静默回退到重算 | 376b9fd3、00128018、a5693e9c、93ad3b89 把所有调用点改为走 refcache（prod 参考在 CPU 上算，10m26s，1.1 GB）。B0 把 provenance 从 988c14ca 改为 d441e55a；A0 接受已记录的 sha。cache miss 直接判 FAIL。h50 禁止在卡上算 fp32 参考 | 09-21..22 那条路径占了当时墙钟的 82% | 376b9fd3；`0927__b0/bwd-hint.md` h69；`0930__bwd/oejob/README.md` |
| 进程管理与监管 | `supervise_job.sh` 里的 VENV 写死成别人的路径，找不到 bin 就 exit 127，然后无限重启。MAX_RESTARTS=60，backoff 只增不减。supervisor 只看进程是否死亡，不看挂卡和停滞。用 `kill <pid>` 杀 loop 会留下孤儿 agent，继续写 job_context。`op-evolve stop` 打断的 act 下次从头重跑（B0 bwd r25 因此跑了两遍）。stop 生效前框架已经建好了下一轮的空壳。09-25 resume 没用 setsid，日志放在 /tmp，10:10:51 重启后丢失。r23 遗留的 meas2 与 validation 并发占卡。09-24 `.stop` 晚了几十秒，下一轮 opt 已经启动 | 框架只有"进程死亡"这一个监控维度；手工操作习惯不对 | 用 `setsid nohup … resume`；停止用 `op-evolve stop` 或 `kill -TERM -<PGID>`；部署 card_ok.sh、patrol、release_guard；监控里用 `[o]p-evolve` 这种括号模式避免匹配到自己 | — | skill:`op-evolve-ops` §3 §11；`0917__flydsl/op-evolve/LAUNCHED.md` |
| 账本和状态的缺陷 | state.yaml 的 flops 键错位（A0 从 r18 起，B0 从 r20 起），progress.md 的 ms 列因此是错的，TF/s 本身没错。改了 shape 名之后 champions 分叉（A0 r20 同时存在 `prod:19` 和长名 `:20`）。refactor 会清空全部 champions。refactor 之后同一轮仍复用 `rounds/N/op`：fwd job 的 `rounds/016/op` 其实是被拒的 g61，r17–19 都把 g61 当成冠军重测。FastError：一个想法没写进 facts/dead_ends，整轮就被记成 failed（A0 r22、r26）。setup 不给 round 0 的速度 | 框架实现缺陷 | ms 一律用 FLOP ÷ TF/s 推导；act.yaml 里只用 fast/proxy/prod 三个名字（h35）；用 `--arms current` 测 incumbent（h51）；state_edit.py 做 rebase 前先核对 `rounds/<best>/op` 与 `op/current` 逐字节相同 | 根治要改框架，需要用户批准，这次没做 | `1002__oe/RULER.md` §3.3–3.4；`0923__flydsl/hint.md` h35 |
| 多卡调度与多棵代码树 | gpu_id 只写进 prompt，gate 的 docker exec 不带设备变量，结果总是跑在 device 0。preflight 要求整机空闲，power_wall 因此永远被跳过。09-14 同一进程里加载多棵 vendored primus_turbo 树：sys.modules 守卫被触发；opaque-type 注册表是进程全局的；两棵树都声明 `primus_turbo::attention_triton_forward_impl`，第二个 torch.library 片段析构时把 schema 从第一个的活句柄下抽走（use-after-free），表现为间歇性的内部断言、bad_alloc、SIGSEGV 139 | 框架默认单卡、单树 | B0 每卡一个容器（用 docker commit 做快照，只挂 /dev/kfd 和一个 renderD）；模块改用私有前缀 `_op_evolve_<n>.`，算子注册到 `primus_turbo_opevolve::`；每个臂一个进程 | use-after-free 吃掉了 4 次验证，约 2 h | `0927__b0/README.md`；`0927__b0/OP-EVOLVE-SUGGESTIONS.md` #4 #6；`0914__campaign/op-evolve-artifacts/…/dead_ends.md` |
| agent 基础设施与安全 | `.venv` 里的 claude-agent-sdk 0.2.152 自带的 CLI 不支持 claude-opus-5-5。B0 上没有 `~/.op_evolve_openai`，codex reviewer 返回 401，deep 轮失败。spec 里的 `runtime.env` 静默无效。review 10 min 后自动批准。opt agent 自己写探针，在 prod 上跑从未上过卡的 kernel，导致 10-02 挂卡 | 依赖版本、账号配置、框架默认值不合适；agent 不知道卡的安全规则 | 升级到 claude-agent-sdk ≥0.2.159；reviewer 改为 claude；环境变量写进 op 代码；h85 写进 hint 表和 deep 轮的 CAMPAIGN CORRECTIONS | agent 的权限要靠 hint 约束，没写成表格行就等于没写 | skill:`fwd.md:110`；`0927__b0/HANDOFF-A0.md` §3；1ee0dd59 |

---

### d.6 正确性与精度门

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| 需要四张量 SQNR 门 | 09-13（A0-限频）出现过一次 dk/dv = −inf，而 out 53.67、dq 52.24 dB 都正常（当时跑的是 `fwd:num_stages=2; bwd:num_warps=2,waves_per_eu=1`）。这是一次性事件：约 1/100 次，立即重复 3 次都通过，之后 2,300 次调用（进程内 1000+1000，150 个新进程共 300 次）都没复现，原文判为"不是任何配置的属性"，疑为瞬时硬件故障；但它说明只看 out/dq 的门会漏检。aiter 只改 `BLOCK_N1=256` 快 1.31×，但 dq 只有 9.59 dB。dkdv 的 BLOCK_M 改成 128 后结果"平滑地错"。num_block_m 没跟着 BLOCK_M 变时，dk/dv 尾部一直停在 zeros 初值。`sequence_parallel=False` 只算了 1/128 | 只看 out 的门看不到反向错误；grid 和切分参数之间相互耦合 | out/dq/dk/dv 各自 ≥50 dB；派发前断言 `BLOCK_N1==BLOCK_M2`、`BLOCK_M1==BLOCK_N2`（拒掉了 3 个配置）；输出预填 NaN，并检查 isfinite 覆盖率；fwd 门对 o 和 lse 分别要求（后来降到 49 dB，见 d.5） | 错的配置往往看起来更快 | `0913__opt_plan__claude/phase1/RESULTS.md` §4 §12（:89-95、:380-393）；`0913__opt_plan__claude/phase2/DECISIONS.md` D1 D4 D8 |
| 确定性测试测到了自己的尺子 | 09-13 默认配置跑 25 次，dk 出现 3 个不同的 SQNR，被报告为"dk 非确定" | 每次都重算 fp32 参考，而参考走的 rocBLAS 本身就不确定 | 改成与第 0 次结果逐位比较（`--determinism-reps`，每次 0.47 s，原来 30 s）；冠军和出厂各跑 1000 次，再各开 150 个新进程跑两轮，0 mismatch | 不要拿新导出的参考来测确定性 | `0913__opt_plan__claude/phase1/RESULTS.md` §4 §12 |
| 确定性门拆分：dk/dv 逐位，dq 用 SQNR | spec 写着"引入 split-k 或 atomic 即失败"，可 validation.py 只做 200 次逐位比对，没有检测 split-k 的手段，固定顺序的 split-k 从 r13 起每轮都在 ship。check_determinism 只跑 fast 一个 shape。aiter ASM 的 dq 本身就是 fp32 atomic 累加（514 条 `buffer_atomic_add_f32`），过不了我们自己的门；7-GEMM 对 5-GEMM 的结构差距（1.386–1.408×）因此在旧规则下无法缩小 | 规则的本意是"每个输出元素只写一次"，200 次逐位只是一个便宜的观测手段 | 09-25 c741fa8d：dk/dv 保持 200 次逐位一致；dq 改为同输入多次运行之间 SQNR ≥70 dB，用 aiter 实测的 dq（fast 113.0 / prod 98.0 dB）校准。冠军不改代码即可通过 | 遗留：确定性测试还要扩到 (q_split, BLOCK_KV) × {fast, prod} | c741fa8d；`0925__flydsl/PARITY-STRATEGY.md` R0；`0925__flydsl/AITER-5GEMM-STUDY.md` §5 |
| `bench_attention_turbo` 的参考值有 bug | 09-30 跑产品分支 bench：non-causal s8192 下所有 backend 的 dq 都只有 12.0–12.4 dB，判 FAIL（triton 那几行 out 60.6、dk 61.0；flydsl 那几行 out 48.8、dk 50.0） | benchmark 自带的参考值在这个形状上不准；用 CPU fp32 参考实测是 52 dB | 记进 memory，未修；PR 里说明这不是 kernel 错误 | 很容易误判为产品 kernel 出错 | `0930__port/runs/bench_turbo_triton.log`；`0930__port/runs/bench_turbo_flydsl.log`；s:fdc2534d 09-30T05:29Z |
| 毒化分配器（poison allocator） | 一个候选越界写冲掉了中断环，poison_allocator 却没抓到"没写过的元素" | 最大毒块只有 64 MiB，而 prod 的 dq 是 256 MiB，caching allocator 永远不会把毒块分给 dq，新页面全是 0。填充值是 fp32 NaN 0x7FC00000，按 bf16 读时有一半是 0.0。自己的修复里 0x7FC07FC0−2^32 又越出了 int32。另外，离线表的平均 SQNR 门会放过 NaN（`nan < 50` 为 False） | 改用 0x7FC07FC0（三种 dtype 下都是 NaN），通过 int32 视图写入，块大小从最大张量往下覆盖；放到 `op/poison_util.py`（686c9927），不改 ut/common.py，免得 refcache 失效；预填 NaN 并检查 isfinite 覆盖（720 次调用 0 个非有限值） | — | cba2abc2；686c9927；`0923__flydsl/poison-fix/README.md`；`0915__opt/NAN-FINDING.md:66-73` |
| 争用造成假 SQNR 失败 | B0 09-14 孤儿扫描进程抢卡时：llama31-8b-s4096 fused 的 out 43.23，gate-s2048 42.88，gate-s1024 asm 48.48（门限 50）。在独占窗口里重测是 53.74/53.87/53.93 | 共用一张卡 | 任何 SQNR 失败，都要先在独占窗口里复现一次才算数 | 差点误杀正确的配置 | `0914__campaign/RESULTS.md:217-233`；skill:`gfx1250-card-safety` §7 |
| SQNR 余量薄、区分度低 | bf16 梯度的 SQNR 上限约 52–53 dB，离 50 dB 门只有约 2.5 dB。ASM bwd 的 dk/dv 只有 50.2–51.0 dB（s4096 下 dk 50.56、dv 50.84），原因是 host 端把 g=4 个部分和先各自舍入成 bf16 再相加。armN 改了 fp32 累加顺序，SQNR 仍读 52.61/52.65/52.83，看不出区别。causal 被跳过的区域真值接近 0，off-by-one 或竞态可能照样过门 | SQNR 是全局平均量 | 以逐元素逐位比较为主：armAB 与 incumbent 的 134,217,728 个元素零差异。SQNR 只作辅助；用 SQNR 指纹审计实际跑的是哪个实现 | 门槛上调或换形状时，ASM 会最先撞线 | `0922_summary`（xlsx NOTES 4/12）；`0925__flydsl/AITER-5GEMM-STUDY.md` §2 |
| 用证明找越界 | 09-24 对 prod 的 134,217,728 个元组做全枚举，证明 k_dq 和 k_dq_sp 的读写都是到真实张量的双射。同时找到 3 个潜在缺陷：(1) impl.py:115 只断言 sq%32，launcher 向下取整而 kernel 向上取整，sq%64==32 时会越界写 262,144 B；(2) dqp 的 num_records 用 int32 计算，nsp≥8 时回绕成 0，所有 dQ 写入被丢弃；(3) k_dkdv 的预取没有钳位，最多读过末尾 982,272 B，只是被真实 extent 挡住了。B0 r19 在 9 个 UT 形状上发生 1,580 次越界预取（prod 上 256 次） | 地址算术改动没有配套的边界证明 | B0 用 h33 的三处修复得到 r19h：0 次越界，输出逐位相同，开销测不出来（1.0001）。w4f 的证明模型修正后（之前把部分等待误当成整批 TDM 已退休）重新证明。注意：不要把 k_dq 的描述符改成真实 extent，那会丢掉 75% 的 dQ | — | 41c916b7；`0924__flydsl/kdq-enumeration/SYNTHESIS.md`；`0927__b0/lab-bwd-r19/REPORT.md` §3–§4 |
| 产品移植的门 | 回归测试发现 gate 放行了 G=6/7，fwd 在构建时 assert；G=32 时 n_seq=0。对抗式 review 又确认了 3 处问题：apply 路径的 window 检查、autotune 缓存 key、COMPILE_ONLY 下的 launch memo | fwd 的 O store 要求 G 整除 32 且 ≤16 | gate 收紧到 G∈{1,2,4,8,16}，补上 gqa16 的精度测试和 G=7/32 的拒绝测试；3 处 review 问题修完后才提交 | — | s:fdc2534d 09-30T04:37–04:56Z |
| e2e 正确性：NaN 运行反而最快 | 74 次 e2e 里 8 次 loss NaN，每次都是所在组最快的：fly 49,487 对 46.5k（+6.4%），asm-on 44,294 对 42.5k，v3 38,713 对 37.7k，nkfix2-on 12,101 对 11,650 | 只按 tps 打分，从来不看 loss；算坏的计算往往更快 | `e2e.sh` 统计 nan_steps 并打印 DISCARD；先判正确性再看速度；一次 run 出问题就重新审查所有 run；后续 e2e 的 watchdog 遇到非有限值即退出（exit 96） | 已报出的 4 个数字被撤回（见 d.2） | `PROGRESS-REPORT-0917.html`；skill:`gfx1250-card-safety` §8；`1002__e2e/E2E-PLAN.md` |

ASM 自身的正确性问题，见 (b) 节：GQA 下 dk/dv 越界写（−0.94/−0.65 dB，绕法是 `dkdv_heads=q` 加 host 规约），`_perf.co` 的 dq 只有 5.84 dB，ASM fwd 的 LSE 布局与两 kernel 版 bwd 不兼容（混用会静默算出错梯度）。

---

### d.7 基础设施与协作

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| 断电截断 git 对象 | 09-17 约 07:46 主机重启（原因未记录）：dev/lhz/attn 的 HEAD 3412b2b 等 6 个 loose object 变成 0 字节，`git status` 报 `fatal: bad object HEAD`；SURVEY.md 被填到 16384 B，尾部 1969 个 NUL。09-22 夜间那次 AC：255 个 object 被截断，分支 ref 指向已损坏的 70efc407（无法恢复）；ROUND3-CLOSEOUT.md 也被截在 16384 B（尾部 2277 个 NUL），而且这个损坏状态还被提交过一次。当时有 4 个提交从未 push | 断电时 commit 正写到一半 | 先隔离 0 字节对象，再 `git update-ref <branch> <最后一个完好提交>`，然后 `rm .git/index && git reset`，最后 `git fsck --no-dangling`。09-23 07:36 起规定每轮结束必须 push（memory `push-after-every-round`）；每次 AC 后都查 fsck；op-evolve 仓库的改动另存成 patch | 丢了 2 个提交（内容还在工作区）和若干文件尾部。skill 里把第一次记成"2026-09-16 cycle"，按会话记录实际是 09-17 | s:c4b79aa6 09-17T08:02Z；s:a9a96fef 09-23T07:13–07:29Z；43eb5d89；f940fac6 |
| 推送通道 | 09-15 本机没有 GitHub 凭据（gh 未登录，remote 用 HTTPS）。09-17 13:25 的 `git push` 被 auto mode 分类器判为数据外泄，29 个提交在 4 天里只有本地一份 | 环境未配置；安全分类器拦截 | 09-15 把 remote 改为 SSH（`git@github.com`）。09-21 用户在 `~/.claude/settings.json` 里加了 Bash 权限规则，09:22 推送成功 | 4 天没有异地备份 | s:b596bddb 09-15T03:18Z；s:c4b79aa6 09-17T13:25Z、09-21T09:22Z |
| A0 与 B0 之间来回切换 | 时间线：09-10..13 在 A0（限频单卡）；09-14 去 B0（4 卡，GPU0 降级）；09-15..17 回 A0（B0 打包时做了 sha256 校验，落成分支 b0-20260914=3742d591 并快进，SSH 只通 B0→A0 方向）；09-17..25 在 A0，与同事分时用；09-27..28 在 B0（4 卡隔离，fwd job 从 A0 r4 克隆，bwd 从 r24 续跑，于是出现 B0 的 r24–r32，与 A0 09-30 之后的 r24–r27 重名）；09-28 12:45 交还 B0，留下冠军快照 5188c9da、HANDOFF-A0.md、备份 `0928__bak_b0`，B0 历史 squash 成 90508b9b 推送，约 40 MB 的 bulk 没推；此后都在 A0 | 两台机器交替可用 | 用 devsync 做 backup/restore（skill xmachine-sync）；跨机文件进 git；Primus 侧的配置在 Primus-Turbo 里另存一份 | 绝对数不能跨机比较，只能迁移同进程比值。Primus 侧配置漏备过一次。09-30 B0 的 ssh host key 变了（会话推测被重装过），按 D1 不绕过校验，于是认定 B0 bwd job 的 r24–r32 历史和真实 dump 都拿不到：r29 代码改从 git 里的 `0927__b0/champions/` 取，09-30 的 s1–s6 只用 randn 测。其实 B0 交还时打的备份包 `/home/lihuzhan/code/0928__bak_b0.tar.gz`（5.3 GB）09-28 14:05 起就在 A0 上（stat），里面有真实 dump 和两个 op-evolve job 的完整目录（含 B0 bwd 各轮）。10-02 07:52–07:55 从中解出真实 dump 到 `/home/lihuzhan/_prof_dump`（sha256 已核对），当天 realab 用它在 A0 上补测（见 d.3）；fwd job 也从这个包恢复。B0 bwd job 目录至今没有解包 | s:b596bddb 09-15T02:20–04:07Z；`0927__b0/HANDOFF-A0.md`；`0928__bak_b0/MANIFEST.md` §2–§4；`0930__bwd/PLAN.md:16,20`；`0930__bwd/REPORT.md:16`；s:69864fc9 10-02T07:51–07:55Z；`1002__oe/FWDJOB.md:11`；`1002__e2e/RESULT-realab.md:3` |
| 固件、驱动与限频 | 09-04 A0 负载下还能跑到 1,699–1,703 MHz。09-10 起出现 VR 限频，DPM 只剩 500/1100 MHz（当时 VBIOS 630A、SMU 125.7.1、dkms 7.1.1-2397345），重启也不消失，负载下约 1.0 GHz。09-28 19:39 有人换了驱动和固件包，09-29 卡无法初始化，AC 也没用。管理员随后刷 VBIOS 700E、SMU 125.12.0、dkms 7.1.0-2411946（主机约 10 次重启），之后 sclk 档位为 500/2356/2400，op 级 1.74–2.03 GHz，训练中约 1.5 GHz。10-01 dkms 又换成 7.1.0-2412954。开机仍会打印 VR 警告，但 DPM 表不再被截断 | 平台侧状态，我方无法控制 | 每次开机检查 pp_dpm_sclk、VBIOS、dkms；所有数字都标上时钟态；时钟变化的问题提交给平台方 | 09-29 之前 A0 的绝对数全部作废。刷固件后（同代码、同分块尺子）fwd FlyDSL r13ns 提速 1.47–1.50×（o1 2.023→1.353 ms 为 1.50×，o2 1.992→1.356 ms 为 1.47×，均值约 1.48×，即 REPORT §6.3 汇总写的 1.48×；教程 README §7 只取 o1，写作 1.50×）、ASM 1.24×（o2 1.23×）；bwd r29 1.67×、ASM 1.39×；e2e 1,947→1,351 ms（1.44×，不纯是平台差异：1,947 那个进程里 fwd 树 `_env.py` 改写过 LIBPATH，驱动也从 7.1.1-2397345 换成了 7.1.0-2412954，见附录 A.2）。09-28 得出的"u2n 在 A0 上退化"结论作废。ATT 恢复可用 | OE:`output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md` §1；`0928__a0_repro/REPORT.md` §2、§5.3、§6.1–§6.4；`0930__bwd/REPORT.md` §6；`1002__e2e/RESULT-e2e.md`（驱动）；wt-llama31 README §7 |
| 容器 | fa-repro（镜像 fa-tune:deps）标为 `owned: false`，不能重建，不能 pip install，只挂载 /home/lihuzhan。具体问题：镜像里 editable 安装的 primus_turbo 会遮蔽 checkout；容器的 /tmp 与宿主不通；sitecustomize.py 被追加过 BLAS 设置；容器里以 root 生成的文件和 core dump 在宿主上删不掉（每次 fault 约 1.1 GB core；09-13 仓库里出现过两个 8 GB 的 GPU core，e2e 还写出过 311 GB 的 coredump）；AC 后不会自动启动。B0 09-14 的 HIP 初始化要 58–172 s（推测 KFD 上下文创建被串行化）。B0 每卡一个容器 fa-g0..g3，09-27 15:43 被他人删除 | 共享镜像，不归我们管 | 改名或移走过期的 core 并关闭 coredump；重新生成产物前先确认旧的已删（必要时 chown）；AC 后用 docker start 复用容器；镜像库路径和补丁写进教程 | 有过磁盘被 core dump 撑满的风险 | skill:`env-and-pitfalls` §1；`0927__b0/README.md`；`0927__b0/STOPPED.md`；s:541e7bc3 09-13T05:25Z |
| 和同事共用卡与机器 | A0 只有一张卡，交接记录：09-17 13:20 交出（TERM 进程组、归档、停 docker）；09-21 发现他人容器 triton-7ff97e-20260910；09-22 16:10 交出；09-24 08:34 linxwang 的 jolly_easley 占着卡（用户授权以后直接停 linxwang 的容器）；09-25 14:00、09-27 15:21、09-30 14:34 分别交出；10-07 同事仍在用。B0：09-27 容器被删；09-28 有他人的 hipblaslt-bench 和容器（weihuan、andyye12）；邻卡 GEMM 会让被测卡慢 3–11× | 共享机器 | 交接流程：`op-evolve stop`、写当日总结、push、`docker stop fa-repro`；交接之后遇到挂卡、重启、容器被停，都不自动重启。KFD 持有者必须能一一说出来；每卡一把 flock 锁；B0 禁止跑持续 GEMM 压测 | 单卡时 fwd 和 bwd job 不能同时跑，op-evolve 和交互工作只能分时；本次总结全程不碰 GPU | skill:`env-and-pitfalls` §11；`0927__b0/LAB-RULES.md`；`0927__b0/REPORT-0928.html` |
| 宿主机硬件与资源 | A0 的 CPU 8 MC60 bank 持续报 corrected L3 MCE（09-25 那次开机报了 163 次）。09-28 02:15 和 08:13 两次开机都带有上一次的 BERT fatal L3 取指错误，说明是主机崩溃；10-01 20:41 也是同类重启。pcie_pl 可纠正 RAS 错误：09-30 共 68 行，10-02 先后 57 条和 58 条。根分区 99% 满（09-22 只剩 53 GB，09-23 有人清理 docker 后剩 828 GB）。`dmesg_restrict` 每次重启都会回到 1。amdgpu 被 blacklist | 主机 CPU 硬件问题，与 GPU 无关 | 报给机器管理员；dmesg 过滤分成 INFO 和 FAULT 两类；上卡前先 push | 主机崩溃同样会截断 git | `0928__a0_repro/REPORT.md` §5.3；`1002__e2e/E2E-PLAN.md` |
| 无人值守与编排 | 09-13 GPU 空转了 2 h 45 min，因为把"提交并汇报"当成了停止点；cron 只在 REPL 空闲时才触发，而 cron 的提示里还写着"GPU busy 就什么也不做"。用户多次要求"不要停"。并行 agent 默认会自己上卡。监控输出刷屏 | 编排设计不当 | 用 `forever_queue.sh` 加 supervisor；PROGRESS.md 作为唤醒入口；委派时明确禁止上卡；监控约 30 s 轮询一次，正常时不输出 | — | s:541e7bc3 09-13T12:29Z；skill:`env-and-pitfalls` §11；memory `e2e-monitoring-style` |

---

#### 仍未解决或待核实（d 节范围）

- B 类挂卡（INVALIDATE_TLBS 起头）和启动期 MES 故障族的直接原因都还不知道。也没有可用的前兆指标。
- nkfix 在 A0 上 21% 运行出 NaN 的根因没有找到。宿主 hipBLASLt 库与 NaN 和挂卡之间，除了 B0 p1a/p1b 这一对单变量对照（每边 n=1，B0 VERIFY 的质疑已由它回应）外，只有相关性证据，没在卡上反复证实。
- 09-21 16:01:54 那次挂卡，会话判为不可恢复，但 4547915c 说 dmesg 里没有任何 reset 签名。09-22 16:02 那次的首行签名没保存下来。这两次的类别都定不了。
- AC 计数中，09-11 的两次（是 AC 还是热重启）和 09-16 第 9 次（推断）不确定。09-13 和 10-02 的重启时间没有记录。
- 需要更正的文档：
  - `env-and-pitfalls` §1b 仍推荐宿主库，与 h85 和教程冲突。
  - `gfx1250-card-safety` 写"09-16 一天五次 AC"，实际是 9 次。
  - 同一份 skill 写"09-16 cycle 截断 6 个 object"，实际是 09-17。
  - skill:`gfx1250-attn-campaign/SKILL.md:106` 和 `0930__bwd/PLAN.md` D2 写 PC sampling"3/3 挂卡"；原始记录是 3 次都 fault、1 次挂卡（`env-and-pitfalls` §2c 的"3 attempts 3 faults"是准确的）。
  - `0930__bwd/PLAN.md` D7 和 bwd hint h83 说 A0 r24 是被"几何平均"接受的，实际是算术平均（20baa45e）。
  - `0930__bwd/PLAN.md` P4/D1 说 B0 bwd job 的 r24–r32 历史拿不到，`0930__bwd/REPORT.md:16` 说真实 dump 只在 B0；其实两者 09-28 起都在 A0 的 `/home/lihuzhan/code/0928__bak_b0.tar.gz` 里（见 d.7）。
  - 1ee0dd59 提交信息、`1002__oe/incident/WEDGE-1002.md`、h85 正文（`0930__bwd/oejob/hint.md:4806`）和 `gfx1250-card-safety` §1 #13 把 10-02 r27 挂卡的两个变体记成 w4f 融合 kernel；按 `rounds/027/_scratch/arms` 源码，实为 s6 基底的 cluster multicast（A_g74 改 k_dkdv，B_g82 改 k_dqg），见 d.1.1。
- gb 尺子已经设计好但没有安装（需要用户批准；可选先花约 6 min 卡时校准 burst 长度）。s6 相对 ASM 的幅度已在 10-02 实测（A0-新固件、真实数据）：blk 0.972、gb 1.038、e2e 内 attention bwd 每步耗时比 0.974（p1：162.15 / 166.54 ms；这是 bwd 耗时比，不是单步比。bwd 每步比 ASM 快约 4 ms：相邻配对差 4.50 / 4.56 ms，即 `1002__e2e/RESULT-e2e.md` 正文的 4.5；两臂中位数相减 3.8–4.4 ms），gb 在这一项上偏悲观约 6%。
- 同一次 e2e 里，fly 的 attention 合计每步比 ASM 多约 4.3–5.2 ms（fwd 43.9 对 34.7，+9.2；bwd 约 −4），单步却快 0.7–2.0 ms（1,349.4 / 1,349.9 对 1,350.6）。两进程的 step − FA 都是 −6.3 ms，超出 E2E-PLAN §4 判定 2 的 5 ms 门限；s6 对 r29 同样如此（attention −53.0 ms，单步只快 37.0–39.1 ms）。差额来自 attention 以外的部分或噪声，没有拆解 [`1002__e2e/e2e/runs/analysis.1002_095504.txt`]。
- e2e 里 ASM 每层（fwd 1.07–1.11 ms、bwd 5.11–5.29 ms）比两把 op 尺子测的都短，时钟解释不了；"输入在 MALL 里是热的"这一假设没测（RULER §8.1）。
- L21 的真实增益两份文档说法不一：REPORT-0928.html 写 +0.3%，ruler/REPORT.md 的分块读数是 +0.6%。
