# gfx1250 注意力优化 — 交接文档

**唤醒后读这个文件，不要重新推导。** 每次被唤醒，从 TODO 顶部取第一条没被认领的做掉，
然后更新本文件。**GPU 忙不是停下的理由**——GPU 跑测量时，CPU 侧永远有代码/文档/下一批可推进。
**永远不要以「等待」结束回合。**

最后更新：2026-09-13 14:05 UTC

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

**诊断 wedge 时只能用带 timeout 的 dmesg。** `ps ... wchan`、`rocm-smi`、
`docker exec`、任何遍历设备/进程状态的命令**都会挂住**。这很反直觉——
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
      同时扫 num_warps × num_stages（寄存器压力若是瓶颈，stages=2 可能在 warps=8 下才可行）。
- [ ] **T2. aiter 预编译 gfx1250 ASM 反向探针。一小时。**
      `aiter-src/hsa/gfx1250/fmha_v3_bwd/bwd_hd128_bf16_causal_br_a32_pssk.co` 已确认存在，
      CSV 行与生产形状逐项匹配，C++ host 五处特判 gfx1250，**只缺 Python 侧架构门**。
      注意：必须先 `dq.zero_()`（gfx1250 只有 atomic32，绕过门控直调会得到静默垃圾 dq），
      且要在**脚本里**调，不要 patch 安装好的 aiter（会污染 21.684 ms 这个参考）。
- [ ] **T3. E2/E3（见 `phase2/prepared-experiments/README.md`）。**
      E2 消四次 `tl.trans`——先花 15 分钟 dump ISA 看后端是否发 `ds_load_tr16_b128`
      （gfx1250 有，gfx950 只有 b64），**这一个检查决定它值 2% 还是 10%**。
      E3 折掉 exp 前的逐元素 VALU，仅 dq 那趟可用累加器初值法。
- [ ] **T4. vendored 与纯 aiter 的 1.85 ms 差。** 已排除四个假设（gather 开销 0.018 ms、
      配置相同、布局相同、sliding_window 相同）。同内核同配置，vendored 19.727 vs
      aiter 18.063，非内核开销两边都是 0.138 —— 差异在**编译产物本身**。下一步 diff
      两条路径的 Triton 缓存元数据。7.6% 的量，优先级低于上面。
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
8. **FlyDSL** —— 上游 v0.3.3 **确实支持 gfx1250**（`tdm_ops`/`s_wait_tensorcnt`/
   `ds_load_tr16_b128` 都在），所以工具链门是开的。但**任何树里都不存在 gfx1250 的 attention
   反向**，而 Primus-Turbo 的 gfx950 FlyDSL 反向是 5,581 行、106 处 MFMA，
   其手排调度在 wave64→wave32 下一条都不成立。同样人天投 udna1 期望更高（少一个外部依赖）。

---

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
