# gfx1250 注意力优化 — 交接文档

**唤醒后读这个文件，不要重新推导。** 每次被唤醒，从 TODO 顶部取第一条没被认领的做掉，
然后更新本文件。**GPU 忙不是停下的理由**——GPU 跑测量时，CPU 侧永远有代码/文档/下一批可推进。
**永远不要以「等待」结束回合。**

最后更新：2026-09-13（离线 ISA 分析后）

---

## ⚠ 当前硬件状态：卡 wedged，GPU 工作全部阻塞

第五次 wedge（前四次在此之前的八天内）。`MES failed to respond` → `wait for reset ack`，
驱动自身的 reset 不会完成。

**恢复步骤**：
1. 重启主机
2. **`sudo modprobe amdgpu`** —— 本节点内核参数 blacklist 了它，不会自动加载。
   不做这步就没有 `/dev/kfd`，torch 报 "No CUDA GPUs are available"
3. `cat /sys/class/drm/card*/device/pp_dpm_sclk` 看限频（每次重启后都还在）
4. `rocm-smi --showpids` 必须显示无进程、VRAM 0%
5. 容器 `fa-tune-0913` 和镜像 `fa-tune:deps`（含全套 torchtitan 依赖）都不受影响

**诊断 wedge 时只能用带 timeout 的 dmesg，而且 `timeout` 对 rocm-smi 无效**
——sudo 父进程被杀掉，rocm-smi 子进程还活着占着管道，命令替换永不返回。
GPU 数量和占用进程改读 `/sys/class/kfd/`（纯读，不会阻塞在驱动上）。
另外 dmesg 要搜**整个缓冲区**，`tail -300` 会被 apparmor 日志刷掉而把挂掉的卡报成 clean。 `ps ... wchan`、`rocm-smi`、
`docker exec`、**`pgrep`**、任何遍历设备/进程状态的命令**都会挂住**。
判断进程存活要用 PID 文件 + `kill -0`（单个信号检查，不遍历 /proc）。这很反直觉——
怀疑卡挂了的时候本能想跑的命令，正是会挂住的那些。

详见 `phase2/INCIDENT-2026-09-13-wedge.md`。**可能的诱因是我自己的操作模式**：
为抢独占测量窗口反复 `pkill -9` 正在跑内核的进程。已修：`forever_queue.sh` 现在轮询
`STOP` 哨兵，`exclusive.sh` 负责「发信号→等静默→执行→释放」，不再打断内核。

---

## 成绩

| | ms | TFLOP/s | |
|---|---:|---:|---|
| turbo 出厂（起点） | 59.642 | 129.0 | |
| turbo 仅配置调优 | 36.405 | 211.4 | |
| torch flex（对照锚点） | 31.337 | 245.6 | |
| **树内 vendored（已提交）** | **24.435** | **315.0** | **2.44× 起点，1.28× flex** |
| 纯 aiter 调优（参考上界） | 21.684 | 354.9 | |

形状 b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal。**51 个测试在 gfx1250 上通过**（原本 0）。

**报告纪律**：反向要按**两种 FLOP 基准**报。这个内核发 **7 趟** score-matrix GEMM，
不是 FA-2 的 5 趟（用多一趟换掉 dq 原子累加）。按 FA-2 的 FLOP 数去除，
**低估真实机器利用率 1.22×**。20 ms 反向 = 385 TFLOP/s 实发 MFMA = 上限的 38.4%，
这个结构自己的下界是 7.68 ms 而不是 5.48。

已提交：`f8c45dee` `df20f1d6` `49257416` `5eba2cf4` `1cb2e183` `03a76f61` `1f83408f`
`8dd2fd32` `fe8d1f4a` `45ec2f76` `52276bd4`

---

## TODO（卡恢复后，按此顺序）

- [ ] **T0. `torch.compile` GEMM 探针。几分钟，决定端到端能否用。**
      镜像的 rocBLAS 只有 27.4 TFLOP/s（Triton 有 1002.7），attention 占一个 step 不到 1%，
      所以现在 tps 既不能验证也不能证伪任何改动。假设：inductor 生成 Triton GEMM 能绕过它。
      repro 配置里 `compile.enable: false` 是因为**旧镜像上** autotune 打死过 GPU，没重试过。
      详见 `phase2/E2E-STRATEGY.md`。
- [ ] **T1. E4：融合内核的 `num_warps=8`。零代码改动，最高价值未测项。**
      `PRIMUS_TURBO_FUSED_MHA_BWD_TUNE=num_warps=8`。理由：BLOCK_N1=256 + HEAD_DIM=128 +
      wave32 下，dk/dv 的 fp32 累加器占 512 VGPR/lane，加常驻 k/v 再 256 = 1024 里用掉 768，
      还没算 score tile；1 wave/SIMD + num_stages=1，占用率和流水都不提供延迟隐藏。
      同时扫 num_warps × num_stages。**离线筛已给出编译期证据支持这一项**：
      在冠军 tile 家族（32/256 bsf=1）内，warps=8 把 VGPR 从顶满的 1024 降到 512、
      `s_set_vgpr_msb` 从 2,299 降到 715（**少 3.2 倍**），代价是 spill 328→426。
      `num_stages=2` 在两种 warps 下都明显更差，与已知悬崖一致，不必优先。
      **离线 ISA 已证实寄存器压力确实是瓶颈**：冠军 vgpr 顶满 1024，spill 113/245，
      且 `s_set_vgpr_msb`（>256 VGPR 的存储体选择，纯开销）有 **1834 条 = 全部指令的 15.7%，
      是 WMMA 条数的 4.1 倍**。注意缓存里已有的 warps=8 变体（BSF=2）编译器选了 512 VGPR
      预算换占用率、**仍然 spill**，所以 warps=8 不是自动的解，要实测。
- [ ] **T1b（已降级，顺手测即可）。`waves_per_eu` 不设。**
      我之前用「10,831 行不同机器码」描述它，**那个说法误导**——diff 行数主要是重排。
      64 配置网格里 32 对同配置对照：总指令变化**中位数 0.00%**、最大 1.32%。
      一半的配置对这个旋钮一条指令都不差。不要再当成有希望的免费收益。
      **但解析器的修复是必要的**：`waves_per_eu=0` 此前被当作非法值拒绝，
      这个设置根本无法从 harness 到达（已修，见 `_ZERO_MEANS_UNSET`）。
- [ ] **T2. aiter 预编译 gfx1250 ASM 反向探针。一小时——考古部分已离线做完，
      见 `phase2/T2-ASM-BACKWARD-SPEC.md`，明天那一小时只用来测量。**
      离线解剖结果：LDS **327,680 B（整个 CU 的 320 KB，每 CU 只能驻留一个 workgroup）**、
      **零 spill**、864 条 WMMA（Triton 冠军 448）、存储体切换/WMMA 比 **1.5 对 4.1**。
      同 tile 同 wave 数下每次发射干 1.9 倍矩阵活。**这是 Triton 表达不了的结构**
      （它这里只分配 64 KB LDS）。注意这只是结构推断——限频态下单 workgroup 设计
      也可能因占用率太低而输。
      **不是 drop-in：一次反向要发三个内核**（odo 预处理 → 主体 → dq fp32→bf16 转换），
      需要一块 fp32 dq_acc 暂存。调用契约已由 `tools/gfx1250/asm_bwd_abi.py` 从 ELF 读出。
      **唯一剩下的未知是 `dq_convert` 的 208 B 布局**（打包函数不在 `mha_bwd.cu` 里）。
      `aiter-src/hsa/gfx1250/fmha_v3_bwd/bwd_hd128_bf16_causal_br_a32_pssk.co` 已确认存在，
      CSV 行与生产形状逐项匹配，C++ host 五处特判 gfx1250，**只缺 Python 侧架构门**。
      注意：必须先 `dq.zero_()`（gfx1250 只有 atomic32，绕过门控直调会得到静默垃圾 dq），
      且要在**脚本里**调，不要 patch 安装好的 aiter（会污染 21.684 ms 这个参考）。
- [x] ~~**T3-E2. 消四次 `tl.trans`**~~ —— **已离线判死，不要上卡。** ISA 显示后端已经
      发了 32 条 `ds_load_tr16_b128`、0 条普通 `ds_read`，转置已被免费折叠进加载指令。
      手工消除的是一个不存在的开销。见 `phase2/isa/ISA-FINDINGS.md`。
- [ ] **T3. E3** 折掉 exp 前的逐元素 VALU，仅 dq 那趟可用累加器初值法。
- [ ] **T4. vendored 与纯 aiter 的 1.85 ms 差。** 已排除四个假设（gather 开销 0.018 ms、
      配置相同、布局相同、sliding_window 相同）。同内核同配置，vendored 19.727 vs
      aiter 18.063，非内核开销两边都是 0.138 —— 差异在**编译产物本身**。缓存元数据已 diff 完：
      找到了一个同阶的机制候选 `waves_per_eu`（见 T1b）。7.6% 的量，优先级低于上面。
- [ ] **T5. 非因果 / varlen / sink 覆盖。** 融合内核支持 sink 但 dsink 没接；
      varlen 在 gfx1250 上仍无 Triton 路径。
- [ ] **T6. HipKittens udna1。** 仅在 Triton 撞到天花板后启动。见 `phase2/PLAN-4GPU-TOMORROW.md`。

**明天 4 卡机**：完整分卡排程、每项的接受/拒绝规则和放弃触发条件，
在 `phase2/PLAN-4GPU-TOMORROW.md`。第 0 件事是四卡同跑冠军配置重新标定基线——
今天所有绝对数字都在 VR 限频态下。

---

## 不要重做（每一条都花了真实 GPU 时间才推翻）

1. **turbo 非融合 dkdv 的非对称 tile** —— 64×64 本来最优，aiter 的 32×128 在那里慢 59%。
   **tile 偏好是内核结构的属性，不是形状的属性**：同一个 N=256，非融合下是灾难，融合下是胜负手。
2. **`sequence_parallel=False`** —— 只算 1/128 的梯度，其余停在 zeros 初值。**纯计时会报成巨大胜利。**
3. **标准 XCD remap** —— gridDim0 本来就是 8 的倍数，同 (batch,head) 的 tile 已免费落在同一 XCD，
   分块反而会打散。
4. **跳过非对角块的因果掩码** —— **慢 9%**，标量分支的代价超过它省掉的比较+选择。
5. **收紧 dkdv 的 `lo`、折叠 `log_p_scale`** —— 性能中性。
6. **前向「0.9 ms 差距」** —— profiler 下只差 0.18 ms（4.219 vs 4.035）。
   **两种方法论不一致 = 差距未确立。** 不要据此 vendor 前向。
7. **HipKittens 的「固定零参考最大值」用在反向** —— 这个反向**没有 running max**，
   `m` 来自前向存好的 LSE，要删的东西不存在。其对偶（把 LSE 偏移折进 GEMM 累加器初值）
   可用，但**仅 dq 那一趟**。
9. **E2 / `tl.trans` 消除** —— 后端已发 `ds_load_tr16_b128`，转置是免费的，无可优化。
8. **FlyDSL** —— 上游 v0.3.3 **确实支持 gfx1250**（`tdm_ops`/`s_wait_tensorcnt`/
   `ds_load_tr16_b128` 都在），所以工具链门是开的。但**任何树里都不存在 gfx1250 的 attention
   反向**，而 Primus-Turbo 的 gfx950 FlyDSL 反向是 5,581 行、106 处 MFMA，
   其手排调度在 wave64→wave32 下一条都不成立。同样人天投 udna1 期望更高（少一个外部依赖）。

---

## 明天 4 卡机的第一优先方法改动：**先离线筛，再上卡**

工具已做好并验证：`tools/gfx1250/isa_screen.py`。
**注意它只是灾难探测器，不是排序器**——对「是否灾难性 spill（>1000 条）」23 个配置里对 22 个，
但对「是否 spill」只有 19/23，且错的三个**方向偏悲观**（`M1=64 N1=128` 一族真值 0 spill 被预测成 133）。
**拿它做硬性拒绝会误杀好配置**，只能用来剔除 >1000 条 spill 的那一类。
今天以 2400–3800 条 spill 收场的六个配置它全部预测正确。

**64 配置全网格已跑完**（`phase2/isa/screen-default.jsonl`）：
16/64 因 spill >1000 在碰到卡之前判死，48 个进入基准测试。
**读表限制：不要跨 tile 家族比 `wmma` 条数**——tile 变小则每 workgroup 活少、
workgroup 变多，筛子只在同一 tile 家族内部有排序意义。

跑法（不需要 GPU，用同镜像起一个不挂 `/dev/kfd` 的新容器；
`docker exec` 对 wedged 容器失效但 `docker run` 可用）：
```bash
docker run --rm -v /home/lihuzhan/code:/home/lihuzhan/code \
  -w $PWD fa-tune:deps python tools/gfx1250/isa_screen.py --grid default
```
43 个变体里规律是二值的：vgpr ≤ 951 的 **21 个变体 spill 恰好为 0**；
vgpr = 1024 的 **14 个全部 spill**（28 到 1814 条不等），没有中间态。
今天那些以 1600–1900 条 spill 收场的配置每一个都烧掉了真实 GPU 分钟才被判死，
而它们在编译期就能看出来。

**另外：卡挂了也能解剖。** `docker cp` 在 `docker exec` 失效后仍然可用，
容器的 Triton 缓存（`/root/.triton/cache`）里有今天每一个内核的完整编译产物。

## 六条必须保留的纪律

1. **四张量 SQNR 门**（out/dq/dk/dv 分别，对 fp32 参考，50 dB）。
   实测抓到过 `out 53.67 ✓ / dq 52.24 ✓ / dk −inf ✗ / dv −inf ✗` —— **只看输出会放过它**。
2. **在 launch 处验证配置，不在源头。** aiter 把 tile 编进 kernel name。
   一轮 sweep 结果「平坦」= 覆盖没生效。我自己的断言曾经放过一次：
   验证「配置源返回对的值」≠「内核用这个值跑了」。
3. **两种测法不一致 = 差距未确立。** 不要基于只有一种方法能产生的数字去动代码。
4. **配对约束** `BLOCK_N1==BLOCK_M2`、`BLOCK_M1==BLOCK_N2`。破坏它会「更快但 dq 只算一半」。
5. **健康检查用 dmesg（带 timeout）**，不用 rocm-smi。**禁止 PC sampling**
   （三次尝试三次 GPU fault，一次需重启）。
6. **端点获胜 = 范围不够，要扩。** 硬下界上的获胜（如 `BLK_SLICE_FACTOR=1`）除外。

---

## 两个不是代码能解决的阻塞项

| 问题 | 量化价值 |
|---|---|
| VR 限频（重启后仍在，1100 MHz 上限） | **1.65×** |
| 缺失的 gfx1250 BLAS | **~37×** |

稠密 GEMM 只有 27.4 TFLOP/s，而同卡同会话里注意力内核跑到 315——
**GEMM 比 flash-attention 慢 8 倍，方向是反的**。升级单在 `phase0/PLATFORM-ESCALATION.md`。
