# 更新（05:30 UTC）：GEMM 修好之后（nkfix，E2E_NKFIX=1），attention 在 e2e 里能看出差别了

| attn 实现 | tps | step ms | GEMM ms（+nkfix 开销） | FA path ms | attention 占单步 |
|---|--:|--:|--:|--:|--:|
| ASM（n5 / n6） | 20,718 / 20,792 | 1,582 / 1,576 | 850–859（+99） | 270–273 | 17.0–17.3% |
| FlyDSL（n5 / n6） | 19,762 / 19,809 | 1,658 / 1,654 | 843–848（+98） | 352–364 | 21.4–21.8% |
| turbo baseline（n7） | 14,563 | 2,250 | 830（+98） | 990 | 43.9% |

- **FlyDSL 对 ASM（同进程配对）**：fly/asm 单步时间比 1.0475（n5，ABBA，17 组）和 1.0500（n6，BAAB，17 组），也就是 FlyDSL 每步慢 4.8–5.0%，约 77 ms。修复前这个差只有 0.42–0.45%。
- **吞吐和 GEMM**：tps 约为修复前的 10 倍（turbo baseline 7.8 倍）。GEMM 占单步从 96% 降到约 53%，全部落在 MT256x256x128。
- **数值**：step-1 loss 与未修复的运行逐位相同；8 次 nkfix 运行（486 步）没有出现 nan/inf。NaN 的根因没有找到，上界见 `../gemm/REPORT.md` §4。
- **注意**：修复后 attention kernel 本身慢了 10–16%。比较绝对 ms 时要用修复后的数字。
- **用法**：`E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1 bash run_e2e.sh train …`，峰值显存 88.65–88.99%。

---

# B0 e2e 结果（2026-09-28）：Llama-3.1-8B BF16，MBS=GBS=4，seq 8192，1 卡，AC none，compile 关

## 0. 结论

| arm | 进程 | 稳态 tps（低档，中位数） | step ms | 高档 tps | FA path ms/step（trace） | 相对 baseline（低档） |
|---|---|--:|--:|--:|--:|--:|
| 1 turbo baseline（Triton @1cb2e183） | P1 `p1b_turbo` | **1,856**（n=31） | 17,655 | 没有高档步 | **961**（fwd 190 / bwd 770） | 1.000 |
| 2 aiter ASM | P2 `p2a`+`p2b` | **1,922**（n=9+15） | 17,049 | 2,016-2,022 | **240**（fwd 39 / bwd 197 / GQA 求和 3.8） | **+3.56%** |
| 3 FlyDSL（fwd r6 + bwd r20@0.3.4.1） | P2 `p2a`+`p2b` | **1,914**（n=10+15） | 17,120-17,125 | 2,008-2,013 | **321 / 328**（fwd 52.5 / 59.2，bwd 268.6） | **+3.13%** |

- **FlyDSL vs ASM（同进程配对，这是唯一干净的比较）**：两个进程、两种顺序（ABBA 和 BAAB）结果相同。按 4 步一组配对，
  fly/asm 的单步时间比是 **1.0045**（p2a，11 组，全部 >1，范围 1.0025-1.0052）和 **1.0042**（p2b，11 组，范围 1.0022-1.0296）。
  也就是 **FlyDSL 慢 0.42-0.45%，约 71-76 ms/步**。trace 里 FA path 的差是 81-88 ms/步，二者一致。**结论：loss（小，但可复现）。**
- **FlyDSL vs baseline**：+3.1% tps（-530 ms/步），**ASM vs baseline**：+3.6%（-606 ms/步）。这一项是跨进程比较：
  P1 整个运行都在"低档"（见 §2），所以拿 P2 低档的步来比。trace 里 FA path 的差是 720 ms（asm）和 633-640 ms（fly），
  能解释 tps 差的约 85%，剩下的约 100 ms 是跨进程的 GEMM 波动。
- **e2e 是 GEMM-bound，而且是被一个 hipBLASLt 的回退 kernel 卡住**：GEMM 占每步 **15.6-16.5 s，也就是 96%**。反向的 dgrad/wgrad
  （两种布局 `Ailk_Bjlk`、`Ailk_Bljk`）全部落在 **`MT32x16x32` 这个很小的回退 tile** 上，一步 15.3 s；前向的 GEMM（`Alik_Bljk`）
  用的是 `MT256x256x128`，一步只要 **0.31 s**（例如 32768×4096×14336 每次 2.47 ms，约 1.56 PF/s）。反向 GEMM 的 FLOPs
  是前向的 2 倍，按前向的效率算应该约 0.62 s。**如果修好，一步能从约 17 s 降到约 1.5 s（约 21-22k tps，是估算，没有测）**。
  这时 attention 的占比会从 1.4-5.4% 升到约 16-40%，FlyDSL 和 ASM 之间的 80 ms 就会变成约 5% 的 tps 差。
  JIRA MI455X eager 是 19,795 tps、GEMM 655 ms；我们这里的 GEMM 是它的 24 倍。**这是 e2e 最大的问题，而且和 attention 无关。**
- **FlyDSL 路径上没有吃掉 kernel 优势的额外开销**：适配层拷贝 0 次（debug.log 每层都记，全程 0 MiB），没有 cast、contiguous、
  GQA 求和、scratch 清零。fly 路径上除了主 kernel 以外只有 `k_delta_bshd` 2.1 ms/步，这是算法本身需要的一步。
  差距**全部在 kernel 本身**：fwd 每层 1.64 ms（高档）/ 1.85 ms（低档），ASM 是 1.22 ms；bwd 每层 8.4 ms
  （`k_dkdv` 5.05 + `k_dq` 3.27 + delta 0.07），ASM 是 6.0 ms（单个融合 kernel）。
  反过来，**ASM 路径有 9.2 ms/步的适配开销**（GQA 求和 3.84、dq_convert 2.27、dq_acc 清零 1.62、odo 1.44），fly 没有。
- **BUILD §3 的"训练模式下 fwd fly/asm≈0.98"在 e2e 里不成立**：e2e 里 fwd 的 fly/asm 是 **1.34**（高档）和 **1.52**（低档）。
  另外，**fly fwd 对时钟档位敏感**（52.5 → 59.2 ms，+13%），ASM fwd 不敏感（39.2 / 38.9 ms）。
- **loss 正常**：同一个 seed，三个进程 step 1 的 loss 分别是 12.25955（turbo）、12.25951（asm）、12.25957（fly），差异 < 1e-4。
  之后的轨迹在 bf16 噪声范围内慢慢分开（step 44：4.068 / 4.000 / 4.081）。没有 nan，arm 切换点上没有台阶。
- **卡的安全**：本次一共 5 次启动（1 次 smoke 失败在 Python 层，1 次 smoke 成功，3 次完整运行），**没有挂卡**。每次运行后 dmesg 里
  0001:04:00.0 都没有 amdgpu 的新行，watchdog 和 memguard 都没有触发。
- **第一次运行（p1a）为什么卡住**：和 p1b 相比只改了 BLAS 库这一个变量（profiler、每步 flex block mask、Triton attention 都一样，
  而 p1b 正常跑完）。所以卡住应当归因到**宿主 hipBLASLt 库**（`~/.local/hipblaslt-gfx1250`）在训练形状上 hang 或者极慢。
  两边各只有 1 个样本，但 VERIFY 里提到的另外两个嫌疑（roctracer、inductor block mask）已经排除。

## 1. 本轮做了什么

1. `run_e2e.sh`：
   - BLAS 改用**镜像库** `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`
     （仍然是 `bash -c` 里赋值 `TORCH_BLAS_PREFER_HIPBLASLT=1`）。进程内的证据是 debug.log 里这一行：
     `BLAS … LIBPATH=/opt/venv/…/gfx1250 preferred_blas_library=_BlasBackend.Cublaslt`。
   - `clk.<tag>.csv` 加了 `gpu_busy_percent` 这一列（card0）。
   - 加了 step-time watchdog：从第一次 attention fwd 开始 360 s 内没有 step 1，或者任意一步超过 max(3×上一步, 240 s)，
     就收集 docker top、busy/sclk、dmesg、worker 的 gdb bt，然后按 PID 给 torchrun 发**一次** SIGTERM。另外，训练开始后 900 s
     内没有 attention fwd 也会触发。本轮没有触发过。
   - 可选的 `E2E_NLAYERS`（通过 Primus 的 model_override patch 生成 `n_layers:`），用于 smoke。
   - 训练结束后把 `Primus/outputs/profile_traces/iteration_*` 拷贝到 `traces/<tag>/`（torchtitan 把 trace 写在共享目录，下次运行会覆盖）。
2. 修了 `attn_backends/e2e_attn/arms.py` 的一个 bug：第一次 smoke（`s1_p2smoke4L`）在第一次 attention 调用时报错
   `AssertionError: flydsl 0.2.4 @ /opt/venv/...`。原因是 `~/.local/flydsl0341` 已经通过 PYTHONPATH 进了 sys.path，
   但被 base_env.sh 排到了镜像 site-packages **后面**，而 `_ensure_flydsl0341()` 看到它"已在 sys.path 里"就没有再往前挪。
   现在改成每次都把它移到 `sys.path[0]`。这次失败发生在 Python 层，没有任何 FlyDSL kernel 上卡。在容器里用 CPU 验证过修复后
   解析到 0.3.4.1。旧版本在 `logs/arms.py.v1.bak`。
3. Smoke `s2_p2smoke4L`（4 层，6 步，asm/fly 交替，profile 第 3、6 步）：一步 3.2 s，10.2k tps，GEMM 路径能跑完，profiler 也正常。
4. 完整运行（32 层）：`p2a_asmfly`（64 步，schedule `asm,fly,asm,fly;asm,fly,fly,asm`，profile 第 22 步=fly、第 44 步=asm）→
   `p1b_turbo`（44 步，profile 第 21、42 步）→ `p2b_flyasm`（64 步，反序 `fly,asm,fly,asm;fly,asm,asm,fly`，第 22 步=asm、第 44 步=fly）。
   稳态窗口：去掉 step 1-7，以及每个被 profile 的步 F 的 F-1、F、F+1（F+1 包含 trace 导出）。

占卡时间约 55 分钟（03:06-04:01），和 ruler operator 通过 flock 交替使用。

## 2. 两个"档位"：同一进程里 tps 会整体跳 5%

两个 P2 进程里，tps 都会在运行中途从约 2,015 **整体掉到**约 1,918，然后一直保持：p2a 在 step 45（03:21），p2b 在 step 32。
fly 和 asm 一起掉，所以同进程配对的比值不受影响。p2a 的 clk.csv 显示，掉档时 sclk 的分钟均值从约 1,840 MHz 降到 1,725-1,790 MHz。
trace 显示掉档的是 GEMM：高档 15.6-15.7 s，低档 16.5 s。P1 整个运行都在低档（GEMM 16.4-16.5 s，tps 1,848-1,861，很平），
所以和 baseline 比时用 P2 的低档步。原因没有查（邻卡负载、功耗或温度都有可能）。它说明**跨进程的 tps 比较天然带着约 5% 的档位噪声**，
而同进程 ABBA 可以把它消掉。

| 进程 | arm | 高档 n / tps 中位数 | 低档 n / tps 中位数 |
|---|---|--:|--:|
| p2a_asmfly | asm | 17 / 2,022 | 9 / 1,922 |
| p2a_asmfly | fly | 15 / 2,013 | 10 / 1,914 |
| p2b_flyasm | asm | 10 / 2,016 | 15 / 1,922 |
| p2b_flyasm | fly | 11 / 2,008 | 15 / 1,914 |
| p1b_turbo | turbo | 0 / — | 31 / 1,856 |

## 3. 单步拆分（kineto trace，按 CPU 祖先归类，ms/步）

| 项 | turbo（p1b s21 / s42） | asm（p2a s44 / p2b s22） | fly（p2a s22 高档 / p2b s44 低档） |
|---|--:|--:|--:|
| 单步（GPU span） | 17,664 / 17,750 | 16,284 / 16,289 | 16,239 / 17,133 |
| **GEMM** | 16,406 / 16,490 | 15,733 / 15,737 | 15,611 / 16,500 |
| 　其中反向 MT32x16x32（两种布局） | 16,122 | 15,422 / 15,427 | 15,308 / — |
| 　其中前向 MT256x256x128 | 284 | 311 | 304 |
| **FA path 合计** | **960.6 / 962.7** | **240.2 / 240.7** | **320.9 / 328.0** |
| 　attn fwd | 190.2（Triton `attn_fwd`） | 39.2 / 38.9（`fmha_bf16_…128x256_mask`） | 52.5 / 59.2（`kn_fmha_fwd_prefill_a16w16_m32x8`） |
| 　attn bwd 主 kernel | 567.2 dkdv + 195.5 dq | 191.8（`fmha_bwd_…causal_br_a32_pssk`） | 161.5 `k_dkdv` + 104.8 `k_dq` |
| 　bwd 附属 kernel | preprocess 1.3、bf16 cast 3.9、清零 2.5 | dq_convert 2.27、dq_acc 清零 1.62、odo 1.44 | `k_delta` 2.1 |
| 　GQA 求和（适配层） | 0 | 3.84 | 0 |
| 　适配层拷贝 | 0 | 0 | 0 |
| elementwise | 254.7 | 269.1 / 268.2 | 264.3 / 262.6 |
| optimizer | 26.9 | 26.5 | 26.7 |
| memcpy | 10.0 | 10.0 | 10.0 |
| GPU idle | 5.5 | 5.5 | 5.3 |

- elementwise（约 265 ms，三个 arm 都有，与 attention 无关）：AMP 下 fp32 权重每步 cast 成 bf16（492 次，27 ms）、bf16 梯度累加到 fp32
  （21 ms）、rms_norm 没走 fused 路径（权重是 fp32，日志里有 `Mismatch dtype … Cannot dispatch to fused implementation`：
  pow/mean/rsqrt/mul 约 25 ms）、silu fwd/bwd 28 ms、softmax/CE 9 ms。每步的 flex block mask 在 GPU 上只有 0.05 ms。
  GPU idle 只有 5 ms/步，所以 CPU 侧的 block mask 也没有让 GPU 停下来（它的代价在启动时：第一次 inductor 编译约 5 分钟，现在已经缓存）。

## 4. 建议去掉 / 要改的东西（按收益排序）

1. **反向 GEMM 的回退 tile（与 attention 无关，但决定一切）**：镜像 hipBLASLt 库对 `Ailk_Bjlk`、`Ailk_Bljk` 两种布局在这些形状上
   只选得到 `MT32x16x32`。可以试的做法（都还没试）：(a) 在 Linear 的 autograd 里把 dgrad/wgrad 改写成前向那种布局
   （`Alik_Bljk`，比如先显式转置或 contiguous 一次，代价是一次拷贝）；(b) 给 hipBLASLt 一份覆盖这两种布局的 tuning / solution；
   (c) 换一个 GEMM 后端。先用 op 级 A/B 在 32768×4096×14336 等形状上确认，再上 e2e。预计能从约 17 s/步降到约 1.5 s/步。
2. **FlyDSL bwd kernel 本身**（-70 ms/步 才能追平 ASM）：`k_dkdv` 5.05 ms/层 + `k_dq` 3.27 ms/层，对比 ASM 的 6.0 ms/层（单 kernel）。
   这是 bwd job 的主攻方向。适配层没有可以省的东西。
3. **FlyDSL fwd kernel**（-13 到 -20 ms/步）：e2e 里是 ASM 的 1.34-1.52 倍，而且对时钟档位敏感。op-evolve 用的 harness
   比值（约 1.26）比 BUILD §3 的"训练模式 0.98"更接近 e2e。fwd job 以后应当以 harness 的比值为准，不要引用那个 0.98。
4. `k_delta_bshd`（2.1 ms/步）可以融合进 `k_dkdv` 的开头，收益很小。
5. 三个 arm 共有的 elementwise：rms_norm 的 dtype 问题（约 25 ms/步）、AMP 每步 cast 权重（27 ms/步）。等 GEMM 修好以后才值得做。
6. ASM 路径（如果要把 ASM 当产品路径）：GQA 求和 + dq_convert + 清零 + odo = 9.2 ms/步，可以用 `dkdv_heads="kv"` 或融合来省。

## 5. 需要 operator 知道的事

- **e2e 期间 GPU0 一直有持续的 GEMM 负载**（RECON §6.2、LAB-RULES 规则 6 的冲突；这次是用户要求跑的）。时间窗：
  03:06:36-03:07:50（smoke）、03:08:45-03:27:00（p2a）、03:27:09-03:41:30（p1b）、03:41:36-04:01:00（p2b）。
  这些时间窗内 GPU2/3 完成的 op-evolve 轮次，建议看一下 beat（ASM）的 ms 是否偏离正常带（fwd 1.14-1.19，bwd 6.41-6.58）。
- 显存：32 层 P2 是 381.44 GiB（**88.30%**，正好是 A0 上 SIGBUS 的那个值），P1 是 382.06 GiB（**88.44%**，离 memguard 的 88.5%
  只差 0.06 个百分点）。这次都没出问题。如果以后的改动再多占 0.3 GiB，memguard 就会在 P1 上触发。
- dmesg：本轮所有运行之前，在 74444 s（约 03:01:30）有 4 行 `ifoe 000N:04:00.1: MC command 0x1b4 inlen 0 failed rc=-11`
  （4 张卡的 .1 function 都有，包括 0001）。时间在我第一次启动之前，也不是 amdgpu 的行，之后没有再出现。
- 宿主机上没有 py-spy。watchdog 的现场采样用的是宿主机的 `sudo gdb -batch`（上次记录过，gdb 对容器 PID namespace 的线程列表不可靠）。

## 6. 证据文件（都在 `output/0927__b0/e2e/` 下）

- 日志：`logs/e2e.{s1_p2smoke4L,s2_p2smoke4L,p2a_asmfly,p1b_turbo,p2b_flyasm}.log`；driver 输出：`logs/driver.*.txt`；
  时钟 + busy：`logs/clk.*.csv`；树快照：`logs/tree.*.sha256`；rank-0 debug.log：`/home/lihuzhan/_dbg_l8b/output/amd/root/<tag>/logs/pre_trainer/rank-0/debug.log`
- 稳态：`logs/steady.{p2a_asmfly,p1b_turbo,p2b_flyasm}.txt`（`tools/steady_arms.py <log> <E2E_ATTN> <pfreq> 7`，在容器里跑，不用卡）
- trace：`traces/<tag>/iteration_N/rank0_trace.json`；拆分：`traces/<tag>/breakdown.{txt,json}`（`tools/trace_breakdown.py`）
- 上一版：`logs/RESULT.v1.md`、`logs/run_e2e.sh.v1_hostlib.bak`、`logs/arms.py.v1.bak`

---

# 历史：第一次运行 p1a_turbo（宿主 BLAS 库，在 step 1 卡住 14 分钟）

以下是第一版 RESULT 的原文（02:58），保留作记录。归因见上面 §0 最后一条。


### 0. 结论

- **没有拿到任何 tps**。只启动了 1 次训练（P1 = arm 1 turbo baseline，`p1a_turbo`，44 步，profile_freq 21）。
  它卡在 step 1：02:41:15 第一层 attention fwd 发射完以后，GPU 一直 100% busy，sclk 一直在 2353-2355 MHz，
  **到 02:56 的 14 分钟里没有打出 step 1**（A0 同配置一步约 17 s）。
- 按 LAB-RULES 第 3 条（进程挂住就停掉所有卡上工作、不重试、上报），**P2（asm/fly）和第二个 P1 都没有跑**，
  profiler 的 trace 也没有拿到。
- 处理：02:56:07 按 PID 给 torchrun（2263448）发了**一次 SIGTERM**。26 s 内进程全部退出，KFD 里没有残留。
  **dmesg 里 0001:04:00.0 没有新行**：没有 fault，没有 queue reset，没有 `MES failed`。
  之后另一个 operator 的 ruler benchmark 拿到锁，在 GPU0 上**正常跑完一个进程**并开始下一个，所以卡本身是好的。
- **首要嫌疑是 BLAS 库，不是 attention**：这次用的是 `HIPBLASLT_TENSILE_LIBPATH=~/.local/hipblaslt-gfx1250/gfx1250`，
  也就是宿主 ROCm 10.1.0 的 328 文件库。这份库**只验证过 512/2048/4096 的方阵 GEMM**
  （`0923__flydsl/STAGE2-S0-PROBE.md` S0-b），训练里的形状从来没有在它上面跑过：M=32768，K 最大到 14336，
  lm_head 的 N=128256，还有 wgrad 的 TN/NT。A0 上跑通的 e2e 用的是**镜像里的库**
  `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`，
  32 层 17 s/步、2,027 tps（`0915__opt/BLAS-FINDING.md`）。这份库在 B0 的 fa-g0 里也有，402 个文件。
  另外，`2026_0927__bak/PERF-A0-vs-B0.md` 记录过：B0 上走这份宿主库的 fp32 GEMM 在 GPU0 上 memory fault 过一次。
  **是"卡死"还是"极慢"（比如某个形状落到很差的 kernel 上）区分不了**：gdb 看到主线程停在同一个
  `hipLaunchKernel`（grad clipping 的 `_foreach_mul_`）里 busy-wait AQL 槽位，前后两次采样隔了 50 s，
  说明 GPU 上排队的 step 1 fwd+bwd 工作一直没有消化完。

### 1. 表（本次没有数据）

| arm | tokens/s | step ms | attn ms/step（op 级预估，32 层） | 相对 baseline |
|---|--:|--:|--:|--:|
| 1 turbo baseline（Triton @1cb2e183） | — 阻塞 | — | 939（op 级） | 1.00 |
| 2 aiter ASM | 未跑 | — | 268（op 级） | — |
| 3 FlyDSL（fwd r6 + bwd r20@0.3.4.1） | 未跑 | — | 325（op 级） | — |

op 级数字来自 `BUILD.md`，不是 e2e 的测量值。

**先验预期（用来决定下一步值不值）**：如果 GEMM 和 A0 一样走镜像 hipBLASLt，一步约 17 s。那么
arm 1 → arm 3 省下的约 0.61 s/步，折合 tps 大约 **+3.5%**；arm 2 和 arm 3 之间的约 57 ms/步只有 **~0.3%**，
落在跨进程噪声里（A0 为 3.9%）。所以要比较 arm 2 和 arm 3，只能靠同一个进程里按步交替（P2 的 ABBA schedule），
再加上 trace 里的 FA path ms；单看 tps 是分不出来的。

### 2. 时间线（p1a_turbo）

| 时间 | 事件 |
|---|---|
| 02:36:11 | 拿到 `/tmp/b0-gpu0.lock`，启动 |
| 02:36:26 | Training started |
| 02:36:28-02:41:14 | `create_block_mask` 的 inductor 编译（`triton_per_fused_…sort…` 等 kernel，纯 CPU，写进 `/tmp/triton_cache_e2e`）。**不是 autotune**，config 里 compile 仍然是关的。这是 flavor `8B_flex` 每一步都会算 block mask 带来的，三个 arm 都一样。下次启动命中缓存 |
| 02:41:14 | 第一次 attention fwd：`[primus_turbo] … fallback backend TRITON is selected`（和 opcheck 一致）；BLAS 证据 `PREFER=1 LIB=~/.local/hipblaslt-gfx1250/gfx1250 preferred_blas_library=Cublaslt` |
| 02:41:15 → 02:56 | GPU 100% busy，没有 step 1。gdb：主线程在 `_foreach_mul_` → `hipLaunchKernel` → `BusyWaitSignal::WaitRelaxed` |
| 02:56:07 | SIGTERM torchrun 2263448（只发了一次） |
| 02:56:33 | 进程全部退出，rc=1，dmesg 没有新行 |

### 3. 证据文件

- 训练日志：`logs/e2e.p1a_turbo.log`；rank 日志（有 `[e2e_attn]` 的行）在
  `/home/lihuzhan/_dbg_l8b/output/amd/root/p1a_turbo/logs/pre_trainer/rank-0/debug.log`
- 所有线程的 gdb 栈：`logs/hang.p1a_turbo.gdb.txt`；处理记录：`logs/hang.p1a_turbo.actions.txt`；
  时钟：`logs/clk.p1a_turbo.csv`；树快照：`logs/tree.p1a_turbo.sha256`
- trace 分析脚本（已写好，还没用上）：`tools/trace_breakdown.py`。它按 CPU 祖先
  （`e2e::attn_fwd/bwd`、`e2e::asm_gqa_sum`、`e2e::adapter_contiguous`、Optimizer）来归类 GPU kernel，
  输出 GEMM / FA path / elementwise / optimizer / idle。只依赖 json，放在容器里跑，不需要卡。

### 4. 建议的下一步（需要 operator 放行；每一步都要上卡）

1. **先做 GEMM 形状探针，不启动训练**：一个进程，训练里的 bf16 GEMM 形状每个跑 1 次，每个都加 hard timeout 和
   `AMD_SERIALIZE_KERNEL=3`。形状包括 M=32768 × {4096→4096, 4096→1024, 4096→14336, 14336→4096, 4096→128256}，
   以及对应的 dgrad/wgrad。两份库（宿主 `~/.local/hipblaslt-gfx1250` 和镜像 `_rocm_sdk_libraries_gfx1250/…/gfx1250`）
   分开各跑一个进程。GEMM 次数很少，不算持续的 GEMM 负载（规则 6）。目的是定位到底是哪个形状、哪份库慢或者挂住。
2. 如果问题在宿主库：训练改用镜像库。只需要改 `run_e2e.sh` 里 `BLAS_EXPORT` 的 LIBPATH，这是 A0 上验证过的组合。
   然后按原计划跑：P1、P2（`asm,fly,asm,fly;asm,fly,fly,asm`，44 步，pfreq 21，这样第 21 步 profile 到 asm、
   第 42 步 profile 到 fly）、P2 反序（`fly,asm,fly,asm;fly,asm,asm,fly`）、P1，一共 4 次启动。
3. 如果两份库都正常：嫌疑转到 step 1 里其它第一次上卡的东西（inductor 生成的 block-mask sort kernel、
   turbo Triton bwd 在训练里的调用），再单独 bisect。
