# gfx1250 bwd 追平并超过 aiter ASM：总结（2026-09-30，A0 heliosr-1b114-c07-1）

分支 `dev/lhz/flydsl-attn-b0`；逐 arm 记录见 `PROGRESS.md`，决策见 `PLAN.md`（D1–D7），源码见 `armsrc/`（md5 在 `armsrc/MD5SUMS`）。
形状：Llama-3.1-8B bwd，b4 s8192 hq32 hkv8 d128 bf16，causal bottom-right，BSHD；FLOP 5.498229e12。

## 1. 结果（randn 输入，同进程 blocked 尺子，含 r29 A/A 与 ASM）

| | prod ms | 占 ASM | proxy ms | 占 ASM |
|---|--:|--:|--:|--:|
| r29（起点） | 6.58 | 83.5% | 0.455 | 83.8% |
| **s6（手工冠军，`armsrc/s6`）** | **5.295–5.300** | **103.8%** | 0.338（s5） | 112.8% |
| ASM | 5.497–5.500 | 100% | 0.381 | 100% |
| op-evolve job 冠军 r24（`armsrc/job_r24` = s6 + `DQT_VT_KEEP`） | 5.292–5.295 | ≈104% | 0.346–0.348 | ≈110% |

- 正确性：prod/proxy/fast SQNR 52.5–52.6 dB；dk/dv 与 r29 逐位一致；s5 lab_validate prod 200 次全逐位；s6 的 dq 与 s5 相差 81 dB（packed 顺序），run-to-run 逐位。
- **未测**：真实训练 q/k/v dump（h47 尺子，dump 只在 B0）与 e2e。round 24 的功耗实验表明时间强烈依赖数据（操作数全零 −34%），真实数据上的领先幅度必须复测。

## 2. 杠杆（按贡献）

1. **TDM LDS ring + 下一轮操作数提前读回寄存器**：k_dkdv（`dkdv_tdm3`，−11.2%），k_dqg（`dqg_tdm`，−2.1%）。消融 A2（Q/dO 已在寄存器）−18.5% 指出方向；只有 TDM 不回读寄存器是负收益（+1.7%）。
2. k_dqg：tailpf（−3.3%，后被 TDM 版取代）、side stream 与 k_dkdv 并发（−0.9%）、VALU 精简（−0.9%）。
3. k_dkdv：divide-free 计数器（−1.6%）、VALU/v_nop/地址精简（−3.2%）。
4. 否决：ku2 +8.2%、lse_late +4.1%、flip −0.5%、trorder_b、dkdv_trim2 +1.3%、abl_l2 证明 L2 带宽不是限制。
5. **4-wave 融合 dQ（w4f）**：正确；无原子写 4.24 ms（ASM 的 1.30×）；原子写按 ASM 的 fragment 顺序写满整条 128 B line 后 9.27→6.63 ms，最好 6.51 ms（r4b）；剩余是整芯片原子吞吐 → 暂停。packed bf16 原子在数值预筛中被判死（round 25）。

## 3. profiler 能力（新固件，`probe/P1-RESULTS.md`）

ATT **可用**（FlyDSL JIT 与 ASM 均有逐指令 stall；op-evolve deep 轮 04_thread_trace 也跑通）；PMC 可用（51 个计数器，无 stall/LDS/字节计数器）；kernel-trace 仍 0 行；PC sampling 被 rocprofv3 1.3.2 拒绝（未采样，未伤卡）；rocprof-compute 未安装且在 gfx1250 上无派生指标。

## 4. op-evolve job 状态（已停止）

- job `gfx1250-flydsl-attn-bwd-20260917-115934`：round 24（deep，refactor h75 采用 s6）被接受为 best_round（其"1.4263×"是 fast 中位数噪声造成的假阳性，prod 实为 0.9986；树与 s6 等价）；round 25（fast）拒绝（prod +0.19%）。
- **14:34 UTC 用 `op-evolve stop` 在 round 26 开始前停下**；`rounds/026/` 为空目录，下次 `resume` 会从 round 26（fast）重新开始。KFD 为空，无残留进程。
- 本次手工改动（全部有 `.bak.*` 备份，diff 在 `oejob/patches/`）：B0 的 blocked/sclk/validation 补丁、min_gain 0.007、目标 **1.20× ASM**（D5）、round 24 起 deep、**fast 用 min 计分 + gain_weights prod 1/proxy 0.25/fast 0**（D7，h83）、refcache 永不在卡上算 fp32 参考、hint h74–h83。state 快照 `oejob/state.yaml.snapshot-0930`。
- **op-evolve 仓库有未提交改动**（deep 轮 prompt 启用 ATT、更新 CAMPAIGN CORRECTIONS；bwd 专用）：完整 diff 在 `oejob/oe_uncommitted_0930.diff`。fwd job 跑 deep 轮前要换回。

## 5. 下一步（建议顺序）

1. 在有真实 dump 的机器上用 h47 尺子（真实 q/k/v、GEMM burst 后）测 s6 vs ASM；再跑 e2e（s6 用 flydsl 0.3.2 的 TDM API，e2e 树目前 pin 0.3.4.1，需先确认 0.3.4.1 下 ISA 一致）。
2. 继续 op-evolve job（resume 即可；目标 1.20× 下 prod 还差约 13%，plan 判断只有融合 dQ 路线够量级）。
3. w4f：减少整芯片原子吞吐压力（ASM 同样付 ~430 cyc/步但藏在 4 wave 里）。

## 6. 节点软硬件参数（2026-09-30 14:35 UTC 采集；这台机器近期常变，任何数字引用前先对照）

| 项 | 值 |
|---|---|
| 主机 | heliosr-1b114-c07-1（A0），Ubuntu 24.04.3 LTS，kernel 6.14.0-37-generic，255 CPU 线程，251 GiB 内存；本次开机 2026-09-29 22:19:42 |
| 内核参数 | `modprobe.blacklist=amdgpu,device_dax,dax_hmem iommu=pt`（amdgpu 不会自动加载；重启后需 `sudo modprobe amdgpu`） |
| GPU | 1 × gfx1250（MI455X），PCI 0001:01:00.0，sysfs `card1`；PCIe 32 GT/s ×16 |
| VBIOS | **113-M4500001-700E**（build 00204667，2026/09/25；09-29 由 asierrag 刷新，之前 630A） |
| SMU fw | **125.12.0**（0x027d0c00），smu driver if 0x7d0000 / fw if 0x7d0001 |
| 驱动 | amdgpu-dkms **7.1.0.31300009-2411946**（24.04），amdgpu-dkms-firmware 31.30.0.9.31300009-2411946；linux-firmware 20240318.git3b128b60-0ubuntu2.26 |
| 时钟档位 | sclk 500 / **2355** / 2400 MHz；fclk 1250 / 1900；mclk 1900；power cap 2500 W（hwmon power1_cap）。满载 WMMA 实测约 1.4–1.8 GHz（功耗墙） |
| 开机告警 | 仍打印 `WARN: GPU is throttled, expect performance decrease. VR.`（但 DPM 表不再被截到 1100 MHz） |
| RAS | 本次开机 dmesg 有 68 行 correctable hardware errors（pcie_pl 块，非致命）；CPU 侧 L3 MCE 也在报 |
| 容器 | `fa-repro`（镜像 `fa-tune:deps`），python `/opt/venv/bin/python3`；ROCm SDK 7.14.0a20260625（rocm_sdk_core/devel/libraries_gfx1250），torch 2.11.0+rocm7.14.0a20260625（HIP 7.14.60850），triton 3.6.0+rocm7.14.0a20260625，rocprofv3 1.3.2（feb9c98），镜像自带 flydsl 0.2.4 |
| flydsl | `~/.local/flydsl032`（0.3.2，bwd 冠军 pin），`~/.local/flydsl0341`（0.3.4.1，fwd/e2e） |
| aiter | `/home/lihuzhan/code/aiter-src` @ 6963ae9 |
| op-evolve | `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve` @ ee16d5b（branch lhz/gfx1250）+ 未提交 deep prompt 改动 |
| ATT decoder | `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib`（librocprof-trace-decoder 0.2.0） |
| 其它 | `kernel.dmesg_restrict` 本次开机被设为 0（重启后恢复 1）；fa-repro 内 FlyDSL 缓存旧目录已移到 `/root/.flydsl/cache.pre-bwd-0930` |
