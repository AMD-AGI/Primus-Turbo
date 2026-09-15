# B0-1P4G 10 小时持续优化计划（08:20 → 18:00 北京时间）

## Context

昨天在 c07-1（单卡、VR 限频 1100 MHz）做 gfx1250 attention 优化，产出了完整的成绩阶梯和交接文档，
但**反复停下来**——不是机制不存在（`forever_queue.sh` 设计是对的），而是三个真实原因：

1. 决策环节（写代码、判优劣、提交）只能由 agent 做，而 agent 每回合就结束；`forever_queue.sh`
   只能跑预设网格，跑完一轮就重跑同一个网格。
2. 卡 wedge 了（9 天内第 5 次），`pkill -9` 抢独占窗口是可能诱因。
3. 端到端验证从未跑通，所以没有任何改动能被证明有用。

今天换到 B0-1P4G：**4 张健康 gfx1250，负载下 2133–2244 MHz**（c07-1 是 1100）。
今早已完成阶梯复现（`output/0914__repro__c07/RESULTS.md`），全部行 1.88–2.12×，排序与昨天一致，
SQNR 逐位吻合。复现过程中修正了三项：VR 限频代价 1.65×→**1.95×**、所有天花板除以 2.136、
**aiter 差距的一半以上已转移到前向**（fwd 1.31× vs bwd 1.12×）。

目标：让 4 张卡在 10 小时里**一刻不停**地推进，且推进过程不依赖我这个会话活着。

---

## 一、不会停的机制（四层，逐层独立）

核心原则：**GPU 工作和代码演化都是脱离会话的外部进程；我只是监工，我死了它们继续跑。**

### L0 — 无 agent 的网格扫描（分钟级起步，永不跑干）
复用 `tools/gfx1250/forever_queue.sh` 的模式：ledger 幂等（tag 已存在就跳过）、
`STOP` 哨兵优雅暂停、dmesg 健康门、外层 `while true` 永不耗尽。
**开工第一分钟就起**，保证在 agent 底座还在安装时 GPU 已经在干活。

### L1 — op-evolve（GPU1，自主演化）
`/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`，"成功是退出码不是意见"：
框架自己跑 `validation.py` 读退出码，agent 不给自己打分。自带
`tools/supervise_job.sh`（崩溃重启、60s→900s 退避、最多 60 次）和
`tools/job_health.sh`（stall 检测）。模块边界可停可续。

### L2 — claude -p campaign（GPU2，自主演化，与 L1 互为保险）
`/home/lihuzhan/code/2026_0903__kyle_flydsl/campaign_example/flydsl_campaign.py`（2356 行，同事已验证）：
ANALYZE / OPTIMIZE / REVIEW 三相机、manager 二次裁决、`state.json` 账本、
`say.sh` 收件箱（不杀进程就能改方向）、stagnation 触发 REPLAN、3 次连续崩溃才收尾。

### L3 — 监工（我的会话 + watchdog）
- **watchdog.sh**（`setsid nohup`，60s 周期）：读各流心跳（ledger mtime + PID `kill -0`），
  重启死流，写 `fleet_status.json`。**检测到 GPU wedge 只停止向该卡派活并告警，绝不重启主机。**
- **Monitor 工具**：只把 watchdog 的 actionable 行流进对话 → 出错立刻叫醒我。
- **CronCreate 每 15 分钟**：叫醒我读 `fleet_status.json` + 各 ledger，做决策、改方向、提交、更新 PROGRESS。

### 铁律（来自昨天的 wedge 事故报告）
- **永远不用 `pkill -9` 抢独占窗口**，用 `exclusive.sh` 的 STOP 哨兵（在内核之间停，不在内核中间停）。
- 健康检查**只用带 timeout 的 `dmesg`**，绝不用 `rocm-smi` / `ps`/`pgrep`（wedge 时全会挂住）。
- 进程存活用 PID 文件 + `kill -0`，不遍历 `/proc`。
- **禁止 PC sampling**（3 次尝试 3 次 GPU fault）。gfx1250 上 ATT thread trace 也解不了码。
- 每条流独立 `TRITON_CACHE_DIR=/tmp/triton_cache_<N>`（共享缓存会 `.triton.lock` 撞车）。

---

## 二、GPU 分配（3 卡优化 + 1 卡专职 e2e）

| GPU | 角色 | 内容 |
|---|---|---|
| **0** | 探针 → 短脉冲宿主 | E4 扫描（立即起）→ T7 ASM 前向 → T2 ASM 反向 → **HipKittens 短验证 + FlyDSL Gate A 收尾** |
| **1** | **op-evolve** | 融合反向自主演化，全天不改任务 |
| **2** | **claude -p campaign** | 融合反向 + VALU 改造（E1/E3 已就绪补丁），与 GPU1 不同假设 |
| **3** | **e2e 专职** | hipBLASLt 探针 → 依赖补齐 → torchtitan repro → JIRA 复现 |
| — | **CPU 流（不占卡）** | HipKittens 单元测试移植、FlyDSL Gate A、ISA 离线筛、文档 |

4 个**独立假设**，不是一个扫描切四份：一条流是哑弹只赔 2.5 GPU-小时，不赔一整天。

**HipKittens 和 FlyDSL 不单独占卡**：两者都是 CPU 为主、GPU 只需短脉冲（编译产物验证、
GEMM ladder smoke）。它们通过 GPU0 的 `exclusive.sh` STOP 哨兵借卡，在 T7/T2 探针的空隙里跑，
不与任何 campaign 抢占。

---

## 三、时间线

### Hour 0（08:20–09:10）并行开工，CPU 与 GPU 同时推进

**立刻起 GPU（不需要任何 agent）**
- GPU0/GPU2：`forever_queue` 变体跑 **E4 — 融合内核 `num_warps` × `num_stages`**。
  昨天 `day1.sh` STEP 2 写好了但一次没测过，零代码改动、最高未测 EV。
  离线 ISA 证据：冠军 tile 家族里 `warps=8` 把 VGPR 从顶满 1024 降到 512、
  `s_set_vgpr_msb` 少 3.2 倍（代价是 spill 328→426），必须实测。
- GPU3：**hipBLASLt 路径探针（5 分钟，今天最高信息密度的实验）**
  ```
  LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib:$LD_LIBRARY_PATH \
  TORCH_BLAS_PREFER_HIPBLASLT=1   →  重跑 8192³ bf16 GEMM
  ```
  镜像里**有** 46 个 gfx1250 bf16 Tensile 解，但该目录不在 `LD_LIBRARY_PATH` 上。
  若 27.4 TFLOP/s 动了 → `PLATFORM-ESCALATION.md` 的「缺失 gfx1250 BLAS ~37×」是打包 bug 不是平台缺陷，
  e2e 当天就有意义，且 T0（torch.compile 探针）可能不必做。

**同时做 CPU 侧（不占卡）**
- `python3.12 -m venv ~/.venv-op-evolve` → `pip install -e .` + `claude-agent-sdk` + `openai-codex` + `cursor-sdk`。
  **关键：codex/cursor 不需要 node/CLI 二进制**——`openai-codex` 自带 Codex binary，
  `cursor-sdk` 自带 vendored node bridge。本机 python3 是 3.9，op-evolve 要 ≥3.10，必须建 venv。
- `tools/check_llm_account.py` 验三个后端（它会做一次真实调用**再做一次 resume**，
  因为 3-Act 续 2-Plan 的会话，不可 resume 的 session id 会静默废掉两个模块）。
- **装 `rocminfo` PATH shim**：`supervise_job.sh:120` 的 `gpu_ok()` 在**宿主机**跑 `rocminfo`，
  而本机没有（只在容器里）。不修这个，supervisor 永远在退避里睡着、一次都不会重启。
  写一个 shim 转发到 `docker exec fa-repro rocminfo`。
- 从 c07-1 抢救 `repro_l8b_bf16_mbs4_seq8k_v5.yaml`——**本机全盘没有，git 里也没有**，
  只在 `RUNBOOK.md:191` 有一行注释。它是 e2e 的硬阻塞。
- 拉 `kyle_skill` 的 `origin/optimizer` 分支（47 文件 / 1.1 MB 注意力知识库），
  作为两个 campaign 的 `--kb-dir`。重点读 `methodology/15-attention-optimize.md`、
  `pitfalls/13-meta-hd64-flash-bwd.md`、决策索引 §D2/§D3 的 **DEAD 清单**。
  另外 `origin/conductor_455` 有 `gfx1250-gemm/SKILL.md`，是 wave32 / WMMA / TDM / 320 KB LDS
  的架构简报——§15 的 PMC 计数器和 DEAD 清单全是 gfx950 测的，**方法可迁移、数字和死路清单不可迁移**。
- **rsync `3rdparty/hipkittens`**（本机 submodule 未初始化，c07-1 已检出 `a288366e`），
  为 HipKittens 流备料。

### Hour 1（09:10–10:10）点火两个自主底座
- **GPU1 op-evolve**：昨天已写好并校验的 job spec 在
  `output/0913__opt_plan__claude/phase0/gfx1250-attn-llama31-8b-bwd.yaml`（410 行）。
  改三处：`agent.account.*` 指向 `~/claude/.amd_llm_env`；
  `runtime.runner.docker.image` → `amdprimus/amdprimus:gfx1250-20260910`（spec 原写的
  `tasimage/primus:op-evolve-v1` 本机没有）；`gpu_id: 1, gpu_pool: []` 硬隔离。
  `evolve: {max_timeout: 9h, schedule: {fast_rounds: 12, fast_per_deep: 5}}`——
  gfx1250 的 profiling 面几乎是瞎的（51 计数器、0 derived、无访存计数器；
  ATT 解码器把 gfx1250 误认成 gfxip 9），deep round 性价比低，**fast-heavy 是对的**。
  spec 铁律：`target.tflops`/`roofline_pct` 必须留 `null`（任一被设置会静默关掉相对门槛）；
  `beat:` 字符串必须字面含 `beaten by N%`；SQNR 检查必须分别覆盖 out/dq/dk/dv。
- **GPU2 campaign**：`flydsl_campaign.py` 唯一缺的是 `LocalHarness`——它只有
  `RemoteHarness`（ssh+docker）。写一个 ~100 行的子类放在**他们仓库外面**
  （`sync()`→no-op、`clear_cache()`→本地 `rm -rf`、`run_bench`/`exec_remote`→`subprocess.run`、
  `gpu_free_map()`→容器内 rocm-smi），phase 机、决策链、prompt、账本全部白拿。
  评分用 `tune_attention.py` 的 JSON 最后一行（`ok` + `total_ms`），
  **计分必须是对冻结参照臂的比值、两臂在同一进程里交替计时**（抗 DVFS 漂移）。
- **GPU0**：T7 ASM 前向探针。阻塞已知：`import aiter.ops.mha` 会拉进
  `pa_decode_gluon.py` 无条件 `import jax`，镜像里没有 jax → 装 jax 或直接走 ctypes 入口。
- **GPU3**：按探针结果走 e2e。

### Hour 2–8（10:10–17:00）监督式自治
- 每 15 分钟 cron 唤醒：读 `fleet_status.json` + 各 ledger → 决策 → 提交 → 更新 PROGRESS。
- **GPU0 的排班**：T7（~10:10–11:10）→ T2（~11:10–12:10）→ 之后主要是 **HipKittens 7b 短脉冲**，
  探针空隙通过 `exclusive.sh` 借卡，每次只几分钟。
- **CPU 流全天并行**（不占卡）：HipKittens 7a 构建接线 → 7c 单元测试移植（3–4h，最耗时也最有价值）；
  FlyDSL Gate A（timebox 2h）；离线 ISA 筛（`isa_screen.py` 起一个不挂 `/dev/kfd` 的容器就能跑，
  能在碰卡之前判死 spill >1000 条的那一类）；文档修订。
- **Hour 4（12:00）中场闸门**，15 分钟，按五个事实重新排期：
  T7/T2 是否落地；E4 是否 ≥1.08×；e2e 是否跑通一个 step；
  HipKittens 的 `mma_ABt` 是否编得出正确 GEMM；FlyDSL Gate A 结论。
  **任何没在 12:00 前过掉验收线的项目直接放弃，不延期。**
- **13:00 HipKittens 硬闸**：还没有数值正确的 `mma_ABt` GEMM 就停掉这条流，把 CPU 时间还给 campaign。

### Hour 9–10（17:00–18:00）冻结与交接
停止派新活 → 冠军配置四卡复测 → 51 测试全绿 → 提交 → 重写 `PROGRESS.md` 为明天的交接文档。

---

## 四、技术优先级（按今早修正后的证据重排）

| # | 项目 | 卡 | 依据 | 验收 |
|---|---|---|---|---|
| 1 | **hipBLASLt 路径探针** | 3 | 镜像有 46 个 gfx1250 Tensile 解但不在搜索路径上 | GEMM 脱离 27.4 TFLOP/s |
| 2 | **E4 融合内核 warps×stages** | 0/2 | 零代码改动，从未测过，ISA 证据支持 | ≥1.08% 且 SQNR 不变 |
| 3 | **T7 ASM 前向**（提级） | 0 | **aiter 差距一半以上已在前向**（1.31× / 0.639 ms） | SQNR ≥40 dB 且 fwd < 2.095 ms |
| 4 | **T2 ASM 反向** | 0 | 昨天 rank 1，门槛按满频重算 | **必须打赢 9.174 ms；值得 vendor 要 ~8.5** |
| 5 | **VALU 改造 E1/E3** | 2 | 补丁已就绪，从未测 | 每项 >2% 且 SQNR 不变，否则回滚 |
| 6 | **e2e + JIRA 复现** | 3 | 用户点名 | 见下 |
| 7 | **HipKittens udna1 门** | CPU+0 | 二值问题，闸住 7–10 工程周 | `transpose` + reductions 单测绿或红 |
| 8 | **FlyDSL Gate A** | CPU | 工具链门比昨天判断的更开 | 「blocked / not blocked」一个结论 |

### 7. HipKittens（udna1）—— 交付物是一个决定，不是加速

`3rdparty/hipkittens` 是 submodule，**本机未初始化**（`git submodule status` 显示 `-`），
c07-1 上已检出（commit `a288366e`，72 个 udna1 头 + `tests/unit/{cdna4,udna1}`）——rsync 过来即可，
不必联网拉。

要回答的是**一个二值问题**：*字节相同的 register tier 在 wave32 + WMMA 下到底算不算得对？*
`conversions.cuh::transpose` 有 11 个 gfx950 反向调用点依赖它，而它是**纯寄存器重标号、没有跨 lane 操作**——
这个恒等式成立的前提是 MFMA 的 A-operand 与 C-accumulator 映射在 CDNA wave64 下互为对偶。
WMMA `f32_16x16x32_bf16`（A 是 v16bf16、C 是 v8f32、跨 32 lane）对偶关系不同。
**断言过，但从来没有被执行过。**

- **7a（CPU, 0.5h）** 构建接线：`setup.py:348` 无条件加 `-DKITTENS_CDNA4`，
  `-DBUILD_HIPKITTENS_BACKEND` 只对 gfx950 开（:360），所以今天的 gfx1250 构建是拿 wave64 头去编的。
  约 20 行 + `*_gfx1250.cu` 后缀（`filter_files_by_arch` 在 :152 已支持后缀分派）。
- **7b（GPU0 短脉冲, 1h）** GEMM ladder smoke：`gemm_naive` → `gemm_expert`，
  `-DKITTENS_UDNA1 --offload-arch=gfx1250`。**`gemm_tdm_arrive` 第一轮跳过、最后单独带 timeout 跑**
  ——它自己的头文件警告在不支持 `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` 的 runtime 上会挂。
  这会产出**任何 udna1 内核的第一个 TFLOP/s 数字**。
- **7c（CPU 为主, 3–4h）** 把 `tests/unit/cdna4/warp/{register,shared}/` 拷进 `udna1/`，
  改 Makefile 的 define。机械工作，因为被测头文件本身字节相同。
  优先级严格按：`transpose` → `reductions`（`permlane32_swap` 的注释描述的是 **64 lane** 累加器，
  wave32 下原样未改）→ `mma_ABt`/`mma_AtB` → `maps.cuh::exp2`。

**接受**：transpose + reductions 绿 → 7–10 周的估算变成一个计划。
**拒绝**：transpose 红 → **这也是好结果**，它把反向重新定价 +5–8 天，
并把 `ds_load_tr16_b128` 接线变成硬前置（现在 `grep -rn 'ds_load_tr' include/udna1/` 返回零）。
**放弃触发**：到 13:00 还编不出一个数值正确的 `mma_ABt` GEMM 就停——
整个移植所依赖的原语不工作，再移植测试也改变不了这一点。

### 8. FlyDSL —— 排最后，只做 Gate A，硬性 2 小时超时

**昨天 plan 的第 1 条反对理由已被证伪**：它说「pinned 0.2.4 不暴露 `tdm_ops` /
`s_wait_tensorcnt` / `ds_load_tr16_b128` / `permlanex16`」。实测镜像里装的 **0.2.4 四个全有**
（`flydsl/expr/rocdl/tdm_ops.py` 存在；后三个加 `gfx1250` 本身都在
`flydsl/_mlir/dialects/_rocdl_ops_gen.py` 里），pip index 还能装到 0.3.2。
**工具链门是开的**，这和 `PROGRESS.md` 第 8 条的更正一致，而与 `PLAN-4GPU-TOMORROW.md` 的第 1 条矛盾。

**但另外三条反对理由仍然成立，所以它依然排最后**：
- `setup.py:530` 对 `--offload-arch=gfx1250` 的构建**显式跳过 flydsl 安装**（连 triton 一起跳）。
- 现存的 FlyDSL attention 反向（`origin/dev/kyle/flydsl-attn-bwd-nd`，`flash_attn_bwd.py` 3289 行）
  是 **gfx950 wave64 + MFMA 手排调度**，wave32/WMMA 下调度表一条都不成立——这正是 18–36 工程日的来源。
- 前向即便完美也只值 0.639 ms / 4.9%，而 **T7（ASM 前向）用 1 小时就能拿同一块收益**。

**今天只做 Gate A，交付物是一个事实，不是内核**（timebox 2h，CPU 为主）：
1. 拿现有的 gfx950 FlyDSL **前向**，翻掉 `attention_flydsl_impl.py` 的 `is_gfx950()` 门
   （今早的分支已把它从 `get_device_compute_capability() >= (9,5)` 收紧成 `is_gfx950()`，
   因为 gfx1250 报 (12,5) 会被旧比较放行然后死在 gfx950 JIT 里）。
2. 试着为 gfx1250 构建，记录第一个真实的失败点。
3. 顺带记一笔：`origin/dev/kyle/flydsl-attn-gqa4`（"Admit GQA groups below 8, so llama 7B and 8B
   reach the flydsl attention"）**正是我们这个形状需要的门**——`_gqa_group_ok` 目前拒绝 G=4。
   如果 Gate A 通过，这条分支是下一步的起点。

**放弃触发**：2 小时到点，或第一个失败点落在「需要重写调度表」这一类上——立刻停，写下结论。
**不要**在今天尝试移植任何 FlyDSL 反向。

**JIRA 目标要说清是哪个**：`jira_2`（seq 8192）是 19,795 → ~30,000 tps；
`jira_1`（seq **4096**）已经报了 30,188 tps。两张表不是一回事。今天对标 **seq 8192 那张**。
注意 `RESULTS.md:152` 指出 30k tps 蕴含 1282.8 TFLOP/s——按今天实测满频 roof 2142 TFLOP/s 算是
**59.9%**，不再是「structurally impossible」（限频态算出来是 128%）。

**不要重做**（昨天花真实 GPU 时间推翻的，`PROGRESS.md` 有 9 条）：
非融合 dkdv 非对称 tile、`sequence_parallel=False`、标准 XCD remap、跳过非对角因果掩码、
收紧 dkdv 的 `lo`、前向「0.9 ms 差距」、HipKittens「固定零参考最大值」用在反向
（这个反向没有 running max，`m` 来自前向存好的 LSE，要删的东西不存在）、
`tl.trans` 消除（后端已发 `ds_load_tr16_b128`，转置已被折进加载指令）。
**FlyDSL 从「完全不碰」下调为「只做 Gate A」**——理由见上面第 8 条，
昨天那条反对理由的第 1 点已被实测证伪，但结论方向没变，只是把门槛从「禁止」改成「2 小时定论」。
**唯一值得在满频重测的**：`T1b waves_per_eu` 和 `num_stages` 类延迟隐藏旋钮——
它们的结论在 1100 MHz 下得出，而满频下访存延迟按周期算大 2.14×。

**一条方向反了的顾虑要纠正**：`PROGRESS.md` 担心 T2 的 320 KB LDS / 每 CU 单 workgroup 设计
「在限频态可能因占用率太低而输」。满频下低占用率设计只会**更**吃亏，这条顾虑在满频更强不是更弱。

---

## 五、关键文件

**新建**（都在 `output/0914__campaign/`）：
`watchdog.sh`、`fleet_status.json`、`local_harness.py`（flydsl_campaign 的 LocalHarness 子类）、
`bench_ratio.py`（campaign 评分器：冻结参照臂 + 同进程交替计时）、
`goal_fused_bwd.txt`（campaign 种子目标）、`jobs/fa-gfx1250.yaml`（op-evolve job）、
`rocminfo` shim。

**复用**（已存在，勿重写）：
`tools/gfx1250/tune_attention.py`（四张量 SQNR 门 + launch 级配置验证 + 逐位确定性模式）、
`tools/gfx1250/forever_queue.sh`、`tools/gfx1250/exclusive.sh`、`tools/gfx1250/isa_screen.py`、
`tools/gfx1250/day1.sh`（STEP 2/3 直接可跑）、`output/0914__repro__c07/run_at.sh`（已修引号 bug）、
`output/0913__opt_plan__claude/phase2/prepared-experiments/E{1,3}*.patch`。

**会被自动修改的内核**：`primus_turbo/triton/attention/fused_mha_bwd_kernel.py`（1911 行，冠军）、
`primus_turbo/triton/attention/attention_kernel.py`。

**HipKittens / FlyDSL 涉及**：`setup.py`（:348 无条件 `-DKITTENS_CDNA4`、:360 gfx950-only 门、
:530 gfx1250 跳过 flydsl）、`3rdparty/hipkittens/{include/udna1,tests/unit}`（rsync 自 c07-1）、
`primus_turbo/pytorch/kernels/attention/attention_flydsl_impl.py`（`is_gfx950()` 门）。

---

## 六、验证

每个改动三道门，缺一不可：
1. **四张量 SQNR ≥50 dB**（out/dq/dk/dv 分别，对 fp32 参考）。
   只看输出会放过真实抓到过的 `dk/dv = −inf` 配置。
2. **`pytest tests/pytorch/ops/test_attention.py -m gfx1250`** 51/51 绿。
3. **生产形状 A/B**：`--shape llama31-8b`，对当日冠军，>2% 才算赢（实测重复离散 0.78%）。
   配对约束 `BLOCK_N1==BLOCK_M2`、`BLOCK_M1==BLOCK_N2` 不可破（破了会「更快但 dq 只算一半」）。

端到端：`pytest -m gfx1250` 全绿 + torchtitan repro 跑出 ≥10 个 step 且 tps 稳定，
与 flex 基线做同镜像同会话对照。

收尾自检：四卡复测冠军，离散 <3%；`git log` 每个提交都能指回它自己的 ledger 行
（**没有 ledger 行的数字不算证据**）。
