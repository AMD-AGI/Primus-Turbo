# 卡恢复后的执行清单（0916 收尾）

卡当前被一个 D 状态进程占死（`kill -9` 无效，它在等 GPU 而 GPU 被它占着）。
**但 dmesg 里零 `wait for reset ack` / `GPU reset`** —— 与上一次不同，卡本身可能没坏。
AC-cycle 后按下面顺序做。

## 先做（有序，前两项互为前提）

### 1. 清场并确认卡真的空闲

```bash
sudo -n modprobe amdgpu && sleep 6
bash tools/gfx1250/gpu_health.sh
for p in $(ls /sys/class/kfd/kfd/proc/); do ps -o args= -p $p; done   # 必须为空
```

**每一个 KFD 持有者都要能叫出名字。** 叫不出名字的持有者是丢弃测量的理由，不是好奇心 ——
0916 正是因为没做这一步，把一个被抢卡污染的故障当成了决定性证据。

`gpu_health.sh` 会在卡被挂起进程占死时仍报 `HEALTHY degraded=0 wedged=0`。
**不要只信它**，用一次真实 GEMM（带 timeout）确认。

### 2. 32 层验证 —— 今天唯一没做成的高价值实验

b=2 探路已成功：32 层真实模型 + wgrad 规则，10,646 tps，**峰值显存 56.41%**，零故障。
b=4 那次是被抢卡打断的，不是显存问题。

```bash
bash output/0915__opt/bin/run32nk.sh      # 只跑带 patch 的那一臂，89% 显存阈值
```

基线不用重跑：`e2e.32-base.log` 那次虽然被打断，但 32L 无 patch 的历史数据是
1,858 / 1,984 tps（20 步，n=1，**未播种**）。若要严格对照，32L 配置现已加 `debug.seed: 1234`，
可以重跑一个干净基线。

**中止条件**：显存 >89%，或 dmesg 故障计数增加。

### 3. Triton 两 kernel sweep —— 陷阱已排除，可以跑了

**此前任何在 `auto` 下做的 bwd sweep 都是假结果。** dispatcher 在训练形状下永远选融合反向
（`flash_attn_interface.py:387`，`fused_backward_eligible()` 在 b·Hq≥32 且 seqlen_k≥512 时恒真），
而融合反向不读 `PRIMUS_TURBO_ATTN_TRITON_TUNE` —— 所以每个候选返回相同时间，
读起来就是"这个旋钮没用"，且 `assert_config_applied` 抓不到（它只验 Config 列表，不验哪个 kernel 真的跑了）。

已修：`tune_attention.py --bwd-path twokernel`（patch dispatcher 读的那个**名字**，
不是源模块的定义 —— 后者会被静默忽略）。call-spy 验证：`auto` 是 `{fused:1, twokernel:0}`，
强制后是 `{fused:0, twokernel:1}`。

跑之前：
- **先单独确认 `num_warps=16` 能不能编译**，带硬超时。它在 0916 **挂起**（不是编译失败：
  日志 0 字节、11 分钟零输出）并占死了卡。挂起不是"一个失败的候选"，重试包装救不了它。
- 在代理形状 `llama31-8b-s4096`（¼ FLOPs）上扫，每次钉死一个配置。
- `rc=139` 当重试而非判负；赢家落在区间边缘 = 区间太小，要扩。
- 四张量分别过 SQNR 门（已抓到过只查 output 会放行的情况，且它不是确定性的）。

## 顺带修正的几处认知

### FlyDSL 的 33% / 28% / 59% 是**调优前**的数字，不能用来判断它的性能

`67beab6f`（9-08）发布"NN 33% of roofline、TN 28%、NT 59%"。
**下一个 commit `f0b583be`（9-09）加了 autotune，并在自己的 message 里说：
"手选的默认值最多差 46%，每个 (dtype, layout) 的最优都不同。"**

所以那三个百分比测的是 `_TRANSPOSED_CFG` 的手选默认值。
**该分支没有发布任何调优后的 bf16 数字**（`git log 67beab6f..HEAD` 里 grep `TFLOP|roofline|%` 无结果）。
唯一的调优后数据是 fp8（NT 3601 / NN 2855 / TN 2158 TFLOPS @ 4096×2048×7168），
其 layout 间比例远好于未调优的 bf16。

**结论：不能用 33% 判断 FlyDSL 不如现有路径。** 要比就得自己测。

### 首次保存了卡故障的 dmesg 证据

`output/0915__opt/logs/dmesg.32-base.fault.txt`（+ `.late`）——
前三次故障都只有事后回忆，这次有完整缓冲区。

03:14:30 起的故障组成：38× `MES(0,0) ring buffer is full`、
25× `MES(N,0) failed to respond to msg=MISC (WAIT_REG_MEM)`、25× `failed to reg_write_reg_wait`、
13× `MES(0,0) failed to respond to msg=INVALIDATE_TLBS`、1× `tawk_ipc mailbox idle too long`。
**零 `wait for reset ack` / `GPU reset` / `amdgpu_device_gpu_recover`** ——
驱动没有尝试恢复，这与 0915 那次（有 reset ack）不同。

## 今天完成的（供交接）

| | 结果 |
|---|---|
| post-nkfix profile | GEMM 85.9% / attention 6.0% / 其余 8.1% |
| wgrad 修法 | **38,043 tps，6.21×**（n=6），单步 861 ms，640.7 TF/s |
| b=2 探路 | 32 层真实模型跑通，显存 56.41% |
| MFU 更正 | 分母是 A100 的 312 TF/s，全部材料已改为 TFLOP/s |
| FlyDSL 结论更正 | BLOCKED 是错的，已在原文件更正 |
| 上报文档 | `VENDOR-REPORT-hipblaslt.md`，196 行 |
| sweep dispatcher 陷阱 | 已定位并修好 |
