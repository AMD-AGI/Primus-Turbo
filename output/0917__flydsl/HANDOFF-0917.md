# 0917 交接：机器交还他人使用

停机时间 2026-09-17 ~13:20 UTC · 分支 `dev/lhz/flydsl-attn` · 宿主 `heliosr-1b114-c07-1`

**读这个文件，不要重新推导。** 一天的结论和被推翻的结论都在这里，细节在同目录的分文档里。

---

## 0. 机器状态（交还时）

| | |
|---|---|
| GPU | **空闲**，零 KFD 持有者，零不可恢复故障签名 |
| op-evolve 作业 | **已停**（`kill -TERM -86137`，杀进程组），进程组清空，无孤儿 |
| docker `fa-repro` | **已停** |
| 定时任务 / 监控 | 已全部取消 |
| 全天挂卡账目 | **0 次 AC-cycle**；可恢复故障若干，见下面「订正」 |

### 订正：dmesg 里的故障不止我早上那一次

我在停机核对时先说了一句「MES 降级行全部早于今日实验」，**那是错的**，按时间戳核实后订正：

| 内核时间戳 | 距停机 | 来源 |
|---|---|---|
| ts≈5047–5049 | 约 249 分钟前 | **我的**：给已知越界写的 `dkdv_heads="kv"` 计时那次 |
| ts≈17510–18782 | 20–42 分钟前 | **op-evolve round 1 的候选内核** |

后者的完整形态是 `GCVM_L2_PROTECTION_FAULT` / `no-retry page fault`
（`PERMISSION_FAULTS: 0x3`、`RW: 0x0` = 读越界），随后
`MES(0,0) failed to respond to REMOVE_QUEUE` —— 那是驱动**按进程**拆队列的过程，不是卡故障。

**这对这个作业是正常的，不是异常**：一个在写新内核的优化器本来就会产出越界候选，
每个只损失它自己的进程。**全天 `wait for reset ack` / `GPU reset begin` / `ring gfx timeout`
计数始终为 0** —— 从未接近需要 AC-cycle 的那一类。

接手的人若在 dmesg 里看到这些，不必紧张，但也不该把它们当成「本来就有的背景噪声」——
它们是今天这份工作产生的，日期可查。

### `rocm-smi` 报 13% 占用，但卡是空闲的

停机后 `rocm-smi --showuse` 连采三次都读到 `GPU use (%): 13`，而同时
**KFD 持有者 = 0、VRAM = 0%**。没有进程、没有显存分配，不可能有计算在跑 ——
这是那个「% time GPU is busy」计数器的读数不可信，不是残留负载。

**直接信号是 KFD 持有者和显存占用，不是这个百分比。** 写在这里是为了让接手的人
不要去追一个假数。

**我对这台机器做过的三处持久改动**（交给别人之前应当知道）：

1. 容器 `fa-repro` 的 `/usr/lib/python3.12/sitecustomize.py` **被追加了**
   `TORCH_BLAS_PREFER_HIPBLASLT=0`。原因：镜像没有 gfx1250 的 hipBLASLt Tensile 库，
   不设它任何 matmul 都会抛 `HIPBLAS_STATUS_INVALID_VALUE`。**这对所有人都是必要的**，
   但如果有人依赖 hipBLASLt 行为，需要知道它被关掉了。注释里写了移除条件。
2. 容器里装了 `psutil`（aiter 的 import 链需要），以及
   `/home/lihuzhan/.local/flydsl032` 下的 **flydsl 0.3.2**（与镜像的 0.2.4 **并存，未覆盖**）。
3. 容器 `/etc/profile.d/zz-gfx1250.sh` 也设了同一个变量 —— 对 login shell 有效，
   但 op-evolve 走非 login shell，所以真正起作用的是第 1 条。留着无害。

---

## 1. 一天做了什么（26+ 提交，全在 `dev/lhz/flydsl-attn`）

### 结论性的结果

| | 结果 | 文档 |
|---|---|---|
| FlyDSL **前向** vs 现有 ASM 前向 | **慢 1.51×**（2.373 vs 1.569 ms，两边正确性都过） | `STAGE1-FWD.md` |
| FlyDSL **反向**：三个内核 | **全部写出并验证通过**，每个都一次通过 | `kernels/README.md` |
| FlyDSL 反向基线性能（op-evolve 实测） | **prod 57.2 / proxy 36.4 / fast 3.0 TFLOP/s** | `op-evolve/artifacts/.../progress.md` |
| 对 ASM 反向（我另测约 540 TFLOP/s）的差距 | **约 9.4×** | 下面 §3 |
| GQA 越界写绕法的代价 | 峰值 **1.254 GiB** + 反向 **5.2%** | `GQA-WORKAROUND-COST.md` |
| flydsl 版本 | **0.3.2 与 turbo 的 0.2.4 树不能共存** | `STAGE1-FWD.md` §4 |

### 反向三个内核（`output/0917__flydsl/kernels/`）

| | SQNR | 范围 |
|---|--:|---|
| `odo_gfx1250.py`（`delta = rowsum(dO*O)`） | 159.24 / 156.49 dB | 单 tile |
| `dkdv_gfx1250.py`（dV + dK） | 150.73 / 145.41 dB | 单 tile |
| `dkdv_loop_gfx1250.py`（运行时 q 循环 + causal） | 141.9 / 141.0 dB | **真内核** |
| `dq_gfx1250.py` | 144.90 dB | 单 tile |

**没有的**：GQA、多 wave 调度、尾块处理、bank 冲突 swizzle、varlen、launcher、派发接线。

### 顺带修掉的两个 main 上的真实缺陷

1. `flydsl` 名义可选却被**无条件 import** —— 缺了它 `import primus_turbo.pytorch` **整个失败**，
   连 gfx1250 上唯一能用的 Triton 路径也一起拖下水。已修 attention 与 sparse_mla 两处。
   **MoE / GEMM / quantization 下还有 10 个文件同样问题**，范围明确、独立有价值，留给下次。
2. `_flydsl_common_ok` 用 `>= _GFX950` 判断，而 gfx1250 报 `(12,5)` —— **它能通过**。
   只是碰巧被 GQA 门挡住才没出事。已改为 `is_gfx950()`。

---

## 2. 我自己被推翻的三个结论（写下来，免得被继承为事实）

1. **「flydsl 0.2.4 够用，4 个 shim 即可」——错。** 差的不是第五个符号，是编译器前端语义。
   教训：**完整的符号审计不等于兼容性审计。**
2. **「5.8 TFLOP/s，慢 93×」——错，错了十倍。** 那次计时是**单头单 batch**，
   grid 只有 256 个 workgroup × 32 线程，按构造就填不满卡。生产形状下同一份内核是 57.2。
   教训：**直接测 X，不要用一个人为欠并行的配置去推断它**（这正是我 skill 第 1 条）。
3. **「模板的免费操作数技巧不可移植」——只对了一半。** dq **成立**（151.32 dB 实测），
   dkdv 不成立。方向不同结论就不同，不能从一个推另一个。

---

## 3. 现在这件事的账目，和需要人做的取舍

| | |
|---|---|
| 前向惯用法竞争力 | **证伪**：慢 1.51×，且**两边都已成熟**，这是最好的估计 |
| 反向数据通路 | **正确**，三个内核全过 |
| 反向性能 | 基线 57.2 vs ASM ~540 TFLOP/s，**约 9.4×**，差距全在未做的外围 |
| 版本共存 | **否** |
| 立项理由 | 只剩结构性三条，**其中只有 varlen 是「有没有」的问题** |

**取舍**：按今天全部证据，一个优化到位的 FlyDSL 反向**最可能是性能持平**。
继续投入换来的不是速度，而是源码可控、GQA 越界写根治、varlen 能力。
**这是花几个 session 换一个已知非性能收益的决定，应当由人来做。**

三个选项：**(a)** 继续做外围优化（多 wave → causal tile 跳过 → 更宽收缩 → TDM 流水 → swizzle，
每步单独计时）；**(b)** 重挂 op-evolve 让它替我们回答「能不能追平」；
**(c)** 到此为止，把 varlen 单独立项。

---

## 4. op-evolve 作业：停在哪，怎么续

作业 `gfx1250-flydsl-attn-bwd-20260917-115934`，跑了 **1 小时 18 分**后被主动停止（**不是失败**）。

- setup **完成**（job_setup 618 s + op_setup 2850 s）
- round 0 基线**已测**（上面那三个数）
- **round 1 在 `opt` 模块中途被打断**，未完成

**全部产物已归档**到 `output/0917__flydsl/op-evolve/artifacts/`（1.1 MB，含
setup agent 花 47 分钟生成的 `op/{baseline,beat,eager,ut}` 与 `benchmark.py`/`validation.py`
—— 重做要一小时，不要重新生成）。**已确认不含任何凭据值**，只有变量名与文件路径。

续跑：

```bash
cd ~/code/2026_0910__op-evolve/op-evolve
export PATH="$PWD/.venv/bin:$PATH"     # 必须：supervise_job.sh 里的 VENV 是别人的硬编码路径
setsid nohup env PATH="$PATH" tools/supervise_job.sh \
  --job gfx1250-flydsl-attn-bwd-20260917-115934 >/dev/null 2>&1 &
```

**resume 从模块边界恢复，不会重跑已落盘的工作**；被打断的 round 1 会重做（框架不信任
留在 `running` 状态的阶段，这是对的）。

前提：容器 `fa-repro` 要先 `docker start`，且卡要空闲。

---

## 5. 下一个会话先读什么

1. 本文件
2. `PLAN.md` 末尾的「0917 晚间修订」—— 计划原文哪里错了
3. `kernels/README.md` —— 内核现状与每个探针回答了什么
4. `~/.claude/skills/gfx1250-card-safety` 与 `flydsl-gfx1250` —— 方法论与领域知识，已固化

**不要**照 `PLAN.md` 正文行动 —— 它的 Stage 0/1/3 都已被执行结果改写。
