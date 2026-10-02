# A0 e2e 运行计划（2026-10-02）：fwd r16 + bwd s6 对 aiter ASM，Llama-3.1-8B 32 层

写于 2026-10-02 08:35–08:50 UTC。写这份计划时没有上卡，只做了 CPU 工作：
- md5 校验、读 ISA dump、跑分析脚本自测；
- 在容器里用 `HIP_VISIBLE_DEVICES=-1` 做 import 检查和适配层功能测试；
- 用桩 launcher 把 driver 的正常路径和全部停止路径空跑了一遍。

**一处需要说明的事**：适配层功能测试的第一个版本调用了 autograd 的 `backward()`。PyTorch 第一次 backward 会为每种设备起工作线程，这时会去问 HIP 运行时有几张卡，结果即使设了 `HIP_VISIBLE_DEVICES=-1`，进程也在 KFD 里**登记了一下**。
- 08:3x 时在 `/sys/class/kfd/kfd/proc` 里看到一个短命的条目，随即消失。
- 没有任何 GPU 工作；前后 KFD 都是空的，当时也没有其他客户端在跑。
- 现在的版本不经过 autograd 引擎；它和 preflight 的 import 检查都按 0.1 s 间隔轮询 KFD 验证过，没有登记。

**09:30–10:00 按独立审查改过**（审查原文：workflow journal 里 key 为 `e2e` 的 review 结果）。上卡前必须改的 3 处、分析前必须改的 1 处、应改的 1 处都已改；建议项改了 5 条，1 条说明了不改的理由。逐条见 §10。改的时候同样没有上卡：只做了 `bash -n`、桩 launcher 空跑（19 种情形）、launcher 守卫的 CPU 单元测试和 preflight。

脚本都在 `output/1002__e2e/e2e/`（下文简称 `KIT`）。它们都是改过的副本，原件一个都没动。
同一目录下的 `arms_src/`、`isa/`、`tools/`、`arms/aa/`、`runs/` 属于做 s6 re-pin 和真实 dump 尺子的另一个 agent，本计划只读这些目录。

## 0. 结论

- **要回答的问题**：bwd s6 在 op 级比 r29 快 19%，比 ASM 快 4%。这个提升在真实训练的每步时间里能不能看到，幅度是否符合预期？
- **设计**：2 个训练进程，每个进程 3 个 arm，按步交替：
  - `asm`：aiter ASM
  - `fly`：fwd r16 + bwd s6
  - `flyr29`：fwd r16 + bwd r29，作为 B0 09-28 的对照
  - 每进程 92 步，每 11 步 profile 一次，step 1–2 只跑 ASM。
  - 训练前先跑两次 opcheck，都经过 E2EAttention 模块和 autograd：
    - fast：`AMD_SERIALIZE_KERNEL=3`，出错能定位到 kernel；
    - prod：**不串行化**。训练实际用的 k_dkdv（主 stream）+ k_dqg（side stream）在这里真并发，`FLY_BWD_RECORD_STREAM=0` 的显存复用路径也先在 op 级跑一遍。fast 形状走的是另一组 kernel（k_dq_sp/k_redsp_q + k_dkdv_sp/k_redsp），与训练只共用 k_delta。
  - `fly` 对 `flyr29` 在同一进程里直接给出 bwd s6 的 e2e 收益；`fly` 对 `asm` 给出与 B0 1.0323 可比的单步比。
- **一条命令**（先满足 §5.0 的前提）：
  ```bash
  K=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
  setsid nohup bash $K/drive.sh > $K/runs/driver.$(date +%m%d_%H%M).log 2>&1 &
  ```
  默认 `MODE=3arm RUNS="opc opcprod p1 p2"`。
- **时间**：墙钟约 31–37 分钟，其中占卡约 12 分钟，其余是 4 次 300 s 冷却。
- **挂卡暴露**：一共 4 次卡上启动，2 次 op 级、2 次训练。
  - 训练启动的历史风险是每次 16–21%。A0 刷固件后还没有过训练启动，这次是第一次。
  - 合计约 **32–40% 的概率至少挂一次**（op 级按 95% 上界 1.8%/次计），挂了需要人去 AC 断电。
  - 两次 opcheck 每次 <2%：刷固件后约 170 个 op 级进程，0 次挂卡。
  - 详见 §7。
- **op-evolve 必须等**：driver 打出 `E2E_DRIVE_DONE` 或 `E2E_DRIVE_STOP` 之前，编排**不得启动或 resume op-evolve**。
  - op-evolve 不拿 `/tmp/a0-gpu0.lock`。它的 benchmark 一旦和训练同时上卡，测量作废，而且属于 card-safety #6 那一类挂卡风险。
  - driver 在每次启动前查 op-evolve loop；训练中 launcher 每 5 s 查 KFD，发现不属于本次运行的进程就停掉本次运行、标 INVALID（exit 91）。这只是兜底，规则本身靠编排遵守。
- **中途停下**：`touch $K/STOP`。driver 在下一次启动前退出（exit 90），不杀正在跑的进程。
- **产出**：
  - `KIT/runs/TABLE.<stamp>.md`：§6 的各张表（每个 arm 的稳态数字、两两对比、两进程一致性、预期对实测、trace、运行健康）。
  - `KIT/runs/analysis.<stamp>.txt`：完整分析输出。
  - 结束后用中文汇报，并 commit + push（traces 除外）。

## 1. 起点

| 量 | 值 | 来源 |
|---|---|---|
| B0 e2e 终版（fwd r16 + bwd r29 对 ASM，09-28，32L） | 单步 fly/asm **1.0323 / 1.0328**；bwd 285.4–290.0 对 233.1–236.7 ms/步；fwd 48.0–49.3 对 38.8–39.3 ms/步；FA 占单步 20.5% / 17.3%；sclk 1342 / 1371 MHz | `output/0927__b0/e2e/RESULT-final.md` §0 |
| A0 op 级（今天 07:54–07:56，randn，blocked 尺子，另一 agent 的 `base1002_*`） | bwd prod：**s6 5.295 / r29 6.567 / ASM 5.513 ms**（s6 = ASM 的 104.1%，s6/r29 = 0.806）；fwd prod：r13ns 1.362 / ASM 1.262（1.079） | `output/0930__bwd/runs/base1002_{bwd,fwd}_prod.log` |
| A0 节点（今天） | <ul><li>amdgpu-dkms 7.1.0-2412954（10-01 换），VBIOS 700E，SMU 125.12.0</li><li>amdgpu 07:43:23 加载，fa-repro 07:50:52 启动</li><li>sclk 档位 500/2357/2400 MHz；空闲时 busy 38%、2355 MHz、1.13 kW；HBM 432 GiB</li><li>GPU RAS 在 07:58:50 和 08:09:46 各报了一批 pcie_pl **可纠正**错误（57+58 条）</li><li>上一次开机以 CPU L3 取指致命错误结束（BERT，10-01 20:41 重启）</li></ul> | dmesg、sysfs、`last` |
| A0 09-28 e2e（刷固件前，作废数字，只取教训） | 5 次训练启动：1 次有效，1 次 NaN，2 次挂卡，1 次 24L 有效 | `output/0928__a0_repro/REPORT.md` §2、§5 |

## 2. arm 与代码树

| arm | fwd | bwd | 说明 |
|---|---|---|---|
| `asm` | aiter `fmha_fwd_with_sink_asm` | 手工发射的 ASM bwd（`output/0927__b0/e2e/arms/asm/_asm_bwd_kernargs.py`）+ 主机 GQA 求和 | 与 B0 相同 |
| `fly` | `KIT/arms/fwd_r16_imglib` | `output/0927__b0/e2e/arms/bwd_s6_0341` | 见下面两段 |
| `flyr29` | 同上 | `output/0927__b0/e2e/arms/bwd_r29_0341` | 与 B0 终版 e2e 的 bwd 相同（kernels `37f37052`、impl `08cb8533`、`_env` `7df61bba`） |

**fwd `fwd_r16_imglib`**
- 来源：`champions/fwd_r16_r13ns` 的逐字节副本，唯一改动是 `_env.py`：删掉了把 `HIPBLASLT_TENSILE_LIBPATH` 指向宿主库的两行赋值。
- 所有 kernel 文件逐字节相同，`MD5SUMS` / `MD5SUMS.champion` 只差 `_env.py`。

**bwd `bwd_s6_0341`**
- `kernels.py` 等于 s6（`5e61678d`），`_env.py` 等于 0.3.4.1 pin（`7df61bba`）。
- `impl.py`（`46e0e812`）是另一 agent 做的 e2e 移植：在 s6 的基础上加了两个环境开关，默认值等于 s6。
- 这一份必须与 `output/1002__e2e/arms_src/bwd_s6_0341/MD5SUMS` 一致，preflight 会检查。
- 本计划用 `FLY_BWD_SIDE_STREAM=1 FLY_BWD_RECORD_STREAM=0`：
  - side stream 照常开，GPU 上的工作与 op 级完全相同；
  - 只关掉 join 处的 `record_stream`。
- 关掉 `record_stream` 的理由：
  - side stream 用到的每块显存，在调用返回、主 stream `wait_stream(s2)` 之前一直有引用，所以复用天然排在 s2 之后；
  - `record_stream` 只会让每次调用约 1.1 GiB 的块推迟复用，而主机在两次 nkfix 同步之间能领先约 4.6 层，也就是约 5 GiB；
  - 32L 时 memguard 只剩 2.2 GiB 余量。
  - 需要完全照搬 s6 时设 `FLY_BWD_RECORD_STREAM=1`。

**s6 在 0.3.2 和 0.3.4.1 下逐字节相同**，证据在 `output/1002__e2e/isa/`：
- 4 个 kernel 的 final ISA、LLVM IR、gpu.binary 都逐字节相同：delta 272、dkdv 2116、dqg 3419 条指令，0 处不同，0 spill/scratch。
- `isa_compare.txt`：两份 launch runtime 也相同。
- `jitkey_r29_vs_s6.log`：r29 与 s6 同进程时只有 `launch_delta` 的 JIT key 相同，而它的 code object 逐字节相同，所以 3-arm 同进程是安全的。

**适配层 `KIT/attn_backends`**（B0 版的副本，改了 3 处）：
1. **BLAS guard**：每加载一个 arm，就检查 `HIPBLASLT_TENSILE_LIBPATH` 和 `TORCH_BLAS_PREFER_HIPBLASLT` 是否仍是 launcher 设的值（镜像库、`1`）。被改了就恢复，并打出 `!! BLAS-REPOINT`，watchdog 见到这行就停。
2. **按步的 attention CUDA event 计时**：
   - `E2E_ATTN_EVENTS`：每次 fwd/bwd 调用前后各记一个 event，范围与 `e2e::attn_*` profiler range 相同。s6 的 side stream 也被包在里面，因为结束 event 在 join 之后。
   - 每步写一行 JSONL；不做同步，异常时自动关闭，不影响训练。
   - 为什么需要：A0 09-28 的 kineto trace 每个 profile 步只有约 4 个 GPU kernel，靠 trace 拆不出 fwd/bwd。
3. **没有默认 FlyDSL 树**：缺 `E2E_FLY_TREES` 时，建模型阶段就报错。B0 版这时会悄悄退回 r6+r20。

## 3. 09-28 A0 的失效模式与对策

| # | 09-28 发生了什么 | 本计划的对策 |
|---|---|---|
| 1 | FlyDSL fwd 树的 `_env.py` 在第一次 attention 时把 hipBLASLt 改指宿主库。两个"step 1 就跑 FlyDSL"的进程一个 NaN（a0_p3b，坏值出在 bwd GEMM），一个挂卡（a0_p4b） | <ul><li>(a) launcher 在 python 启动前用 `bash -c` 赋值镜像库：`TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/opt/venv/.../hipblaslt/library/gfx1250`</li><li>(b) fwd 用不改 BLAS 环境的副本</li><li>(c) preflight 静态 grep 所有树，有任何赋值、setdefault、putenv、update 就拒绝启动</li><li>(d) 适配层运行时 guard 加 watchdog</li><li>(e) **ASM 先跑**：step 1–2 只跑 ASM，第一次 FlyDSL 在 step 3。这时所有 GEMM 问题类型都已经在镜像库下初始化好了</li></ul> |
| 2 | p3b 跑了 86 步 NaN，下一个进程 p4a（32L）在 step 2 挂卡 | <ul><li>`NKFIX_CHECK=1`（每 64 次改写调用同步检查一次），加上 watchdog 在第一个非有限 loss/grad_norm 或第一条 `[nkfix] NON-FINITE` 时发**一次** SIGTERM</li><li>driver 随即停下（exit 96），不再启动任何进程</li><li>之后先 `health_gemm.sh`（§5.4），再做别的卡上工作</li></ul> |
| 3 | 前一个进程刚结束就启动下一个、KFD 里有残留 | driver 全程持有 `/tmp/a0-gpu0.lock`，冷却期间也不放。每次启动前：<ul><li>KFD 为空，`IDLE_MAX` 1800 s 内等不到就退出（exit 97）</li><li>冷却 300 s，期间每秒查一次 STOP 哨兵</li><li>守卫：STOP、op-evolve loop、realab driver、KFD 仍为空、代码树（第 10、11 行）</li></ul>launcher 在 docker exec 之前把同样的守卫再跑一遍 |
| 4 | 显存 89%（A0 曾在 88.30% SIGBUS，后来撤回了"是显存问题"的结论） | memguard 89.5%（B0 终版取值），超过就 SIGTERM 一次（exit 95）。s6 的 `record_stream` 已关（§2）。峰值显存按进程进健康表（reserved 显存是进程级的，只增不减，不按 arm 解读） |
| 5 | JIT 缓存串用（h46/h72） | 每个进程用新的 `FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_<tag>_<hhmmss>` |
| 6 | kineto 只记录约 4 个 kernel/步 | <ul><li>CUDA event 计时是主数据</li><li>`trace_breakdown2.py` 对 kernel 数 <1500、attention range 没归到 kernel、或调用数 ≠ 32 的 trace 标 **INVALID**，不使用</li></ul> |
| 7 | dmesg 过滤会误停 | 刷固件后 A0 在正常工作中也会打 GPU RAS **可纠正** pcie_pl 计数（今天 07:58:50、08:09:46），以及 `svm_range_restore_work ... hogged CPU`。09-28 的过滤（任何含 amdgpu 的新行都算）会把它们当故障停掉。现在分两类：<ul><li>**INFO**：CPU MCE、`[Hardware Error]` 寄存器转储、"N new correctable hardware errors"、ring buffer full、workqueue hog，只记录</li><li>**FAULT**：其余所有含 amdgpu/MES/GCVM/reset/Queues reset/uncorrectable 等的行，停止（exit 99）</li></ul>已用今天的 dmesg 和合成的故障行验证过 |
| 8 | 新因素：s6 的 side stream 第一次进训练进程（多一个硬件队列） | <ul><li>opcheck fast 覆盖不了训练的路径：<ul><li>fast 形状（b1 s1024 hq8 hkv2）下，s6 的 nsp_q=8、nsp=16，走 k_dq_sp+k_redsp_q（side stream）和 k_dkdv_sp+k_redsp（主 stream），与训练的 k_dkdv + k_dqg 只共用 k_delta</li><li>fast 加了 `AMD_SERIALIZE_KERNEL=3`，每个 kernel 前后都等设备空闲，两个 stream 不会真并发</li></ul></li><li>所以默认 RUNS 加了 **opcheck prod**（`opcprod`）：<ul><li>prod 形状，同样的 `FLY_BWD_SIDE_STREAM=1 FLY_BWD_RECORD_STREAM=0`</li><li>**不加** `AMD_SERIALIZE_KERNEL`，硬件队列数用 HIP 默认值（容器环境里没有设 `GPU_MAX_HW_QUEUES`）。训练的 k_dkdv + k_dqg 在这里真并发，record_stream=0 的复用也真发生。这比训练里的 `GPU_MAX_HW_QUEUES=2`（第 9 行）更严</li><li>opcheck 先用 NaN 填满再释放一批显存，读到没写过的显存会变成非有限值；链路结果对 fp32 参考打分，复用出错会体现在 SQNR 上</li><li>门槛同 fast：链路 o/dq/dk/dv、kernel bwd（参考 o/lse）dq/dk/dv、kernel fwd o/lse 全部有限且 ≥47 dB</li></ul></li><li>s6 在 09-30 和今天已在 op 级跑过几十个进程，没有出事</li><li>每个进程把 `[flydsl_bwd ...] DQ_SIDE_STREAM=1 DQ_SIDE_RECORD=0` 这行证据记进 post 文件</li></ul> |
| 9 | 新发现：训练进程里 side stream 可能没有并发 | <ul><li>Primus `runner/helpers/envs/base_env.sh:209` 在训练进程里设 `GPU_MAX_HW_QUEUES=2`（`:218` 还有 `CUDA_DEVICE_MAX_CONNECTIONS=1`）。op 级尺子没设，是 HIP 默认值</li><li>PyTorch 第一次建 stream 时会一次建一整池 stream，2 个硬件队列要被它们轮流共用。所以 s6 的 k_dqg 在 e2e 里**可能与 k_dkdv 同队列、退化成串行**</li><li>最坏情况就是 `FLY_BWD_SIDE_STREAM=0`：op 级损失约 0.9%，即约 0.05 ms/调用、约 1.6 ms/步，比预期收益（约 −41 ms/步）小一个数量级</li><li>为了与 B0（同样是 2）可比，**不改**这个值</li><li>worker 的 `GPU_MAX_HW_QUEUES` 记进 `logs/env.<tag>.txt`。trace 有效时，`attn_bwd` 的 wall < sum 说明有并发</li><li>要单独量并发收益，需要再加一个进程 `E2E_ENV+=" -e GPU_MAX_HW_QUEUES=4"`，另行决定</li></ul> |
| 10 | 审查发现：另一个 GPU 客户端不拿锁 | <ul><li>op-evolve 的 loop 不拿 `/tmp/a0-gpu0.lock`。realab 的每个进程拿锁，但两个进程之间（30 s 冷却）会放锁</li><li>**冷却之后**（driver）和 **docker exec 之前**（launcher）各查一次：<ul><li>`oe_loops`：按进程自己的 comm/argv 匹配 `op-evolve run\|resume`（comm 是 `op-evolve`，或 python 的脚本参数是 `.../op-evolve` 或 `-m op_evolve`）。不会匹配到检查自己的 shell，也不会匹配命令行里只是提到 op-evolve 的 shell 或 `tail`</li><li>`realab_drivers`：realab.sh / realab_run.sh</li><li>KFD 为空</li></ul>任何一项不满足就 exit 97，不启动</li><li>运行中，launcher 每 5 s 检查每个 KFD holder：<ul><li>判断依据是 `/proc/<pid>/environ` 里有没有本次的 `E2E_RUN_MARKER=<tag>`。docker exec -e 设置它，torchrun 和 worker 都继承</li><li>正常运行本来就有 torchrun + worker 两个 holder（09-28 a0_p4b 是 11714 和 12041），所以不按数量判断</li><li>没有标记的就是外来进程：记 `!! FOREIGN KFD <pid> <cmd>`，对本次运行的 torchrun（opcheck 是 python）各发**一次** SIGTERM，post 里写 `valid: INVALID`，driver exit 91，不再启动任何进程</li><li>外来进程本身不动</li><li>clk CSV 多一列 `kfd`，是每 5 s 的 holder PID</li></ul></li><li>编排规则：`E2E_DRIVE_DONE` / `E2E_DRIVE_STOP` 之前不得启动或 resume op-evolve（§0、§5.0）</li></ul> |
| 11 | 审查发现：中途可能换成另一个 s6 | <ul><li>`bwd_s6_0341` 属于另一个 agent，今天 08:08 已改过一次 impl.py。p2 要在 driver 启动后约 25 min 才开始</li><li>`trees_check`（trees.sh）：<ul><li>fwd 副本等于它的 MD5SUMS</li><li>s6：kernels `5e61678d`、impl `46e0e812`、_env `7df61bba`，整棵等于 `arms_src/bwd_s6_0341/MD5SUMS`</li><li>r29：kernels `37f37052`、impl `08cb8533`、_env `7df61bba`</li></ul></li><li>`run_files_sha`：所有会运行的文件（3 棵树、适配层、ASM launcher、nkfix 和 transpose，共 31 个）的 sha256。driver 拿锁后存一份 `runs/tree.<stamp>.sha256`</li><li>每次启动前（driver 冷却后、launcher docker exec 前）两项都要通过，否则 exit 93，不启动</li><li>训练结束后 launcher 再算一次。与本进程启动前的 `logs/tree.<tag>.sha256` 不同就打 `!! TREE CHANGED DURING RUN`，这次运行作废，exit 93</li><li>`analyze.sh` 把每个进程的 `tree.<tag>.sha256` 与第一个进程比，结果进健康表的"代码树"列</li></ul> |

其它沿用 B0 终版配方，不改：
- nkfix（`E2E_NKFIX=1`；不开的话每个 bwd GEMM 都落到 MT32x16x32，单步慢约 10 倍）
- `compile.enable=false`，`converters: ["primus_turbo"]`
- seed 1234，MBS=GBS=4，seq 8192，AC none
- 只用 `bash -c`，不用 `-lc`
- 只按 PID 发 SIGTERM，每个进程最多一次，共三处：watchdog、memguard、外来 KFD。
  - 唯一的 SIGKILL 是 `timeout -k` 的最后兜底：硬超时发出 SIGTERM 后，进程过 20 s 还在才发（health_gemm 是 10 s）。硬超时是 opcheck 600 s、训练 2600 s、health_gemm 100 s。
  - 训练里 timeout 的子进程是 primus-cli 的 bash，不是 GPU 进程；opcheck 里是 python 本身，这时它已经 600 s 没有结束。

## 4. 预期，以及怎样算"正常"

**预期**：op 级（randn、blocked，今天）× 32 层。

| 对比 | Δbwd ms/步 | Δfwd ms/步 | Δ单步 ms | 单步比 |
|---|--:|--:|--:|--:|
| fly − flyr29 | −40.7（按 B0 的 e2e/blocked 系数 1.07–1.09 约为 −44） | 0（同一 fwd 树） | ≈ −41 到 −45 | ≈ 0.972–0.975 |
| fly − asm | −7（s6/ASM 0.96，若在训练工作点保持） | 约 +10（B0 e2e 同源码 r16：48.7 对 39.1） | ≈ +3 | ≈ 1.002（B0 为 1.032） |
| flyr29 − asm | +34 到 +52（A0 r29/ASM 1.19；B0 e2e 为 +52） | 约 +10 | ≈ +44 到 +62 | ≈ 1.03–1.04 |

**用今天 part 1 的真实 dump + GEMM burst 数字替换**：part 1 是另一个 agent 跑的 `output/1002__e2e/tools/realab*` 的 gb 条件。
- 把它们传给 `OP=... bash analyze.sh`（§6）。表 3 会按 32 × Δ(op) 自动生成预期值和判定。
- asm 的 bwd 数字要包含主机 GQA 求和，与 e2e 的口径一致。

**判定"正常"**（逐进程）：
1. `fly − flyr29` 的 Δbwd（events）与 32 × Δop(gb) 的差在 max(5 ms, 20%) 以内。
2. 每对 arm 的 Δ单步 − ΔFA 在 max(5 ms, 15% ΔFA) 以内，即 attention 以外没有新增代价（例如 s6 的 side stream 或时钟与 GEMM 的相互作用）。
3. 两个进程的单步比相差 ≤0.3%（B0 为 0.05%）。表 2b 自动给出。
   - 分析工具的 arm 顺序固定为 asm、flyr29、fly，两个进程的配对方向一致：都报 fly/flyr29，即 fly − flyr29。
   - 改之前按 schedule 里首次出现的顺序排：p1 报 flyr29/fly，p2 报 fly/flyr29，方向相反（审查第 4 条）。
4. loss 全程有限，`nonfinite_events: []`，fly 与 asm 的 step 62/92 loss 差在 bf16 噪声内（B0 的 step-62 loss 为 3.530 / 3.517）。

**不正常时下一步**：
- 第 1 条不满足（e2e 收益明显小于 op 级）：
  - 先看 events 的逐层 bwd 和 sclk；
  - 可加一个进程，用 `FLY_BWD_SIDE_STREAM=0` 把 s6 串行化，判断是否是 side stream 在训练工作点（功耗墙下）失效。这要多一次启动，需要另行决定。
- 第 2 条不满足：attention 以外的代价，查 trace（若有效）的 gemm / elementwise / idle。

## 5. 运行步骤

### 5.0 前提（每条都要满足；driver 会核对其中能自动核对的）

1. 卡上没有其它客户端：
   - part 1 的真实 dump A/B（`realab`）已跑完：`runs/realab_*.driver.txt` 的最后一行是 `REALAB_DRIVER_DONE`。
     - 写这一版时最近一次是 09:17:34，`realab_1002_091405`，rc=0，blk + gb。
     - realab.sh 在两个进程之间会放锁，所以它的 driver 还活着时，preflight 判 FAIL，driver 退出 97。
   - op-evolve job 没在跑（`op-evolve stop --job ...`）。
     - **driver 打出 `E2E_DRIVE_DONE` 或 `E2E_DRIVE_STOP` 之前，不得启动或 resume op-evolve**。task 2 的 op-evolve 排在 e2e 之后。
     - 它不拿锁。driver 只能在每次启动前发现它；训练中只能靠外来 KFD 检测事后停下，那次运行就作废了。
   - `ls /sys/class/kfd/kfd/proc` 为空。
   - 卡是单张，一次只能一个客户端。driver 拿锁后等 KFD 为空；有 op-evolve loop 或 realab driver 就退出 97。
2. 先 commit + push `output/1002__e2e`（代码、配置；traces 已在 `.gitignore`；`KIT/logs/` 已由 `KIT/.gitignore` 重新纳入，仓库顶层的 `logs/` 规则不再挡住它）。A0 主机自己会崩（BERT fatal L3），断电时提交到一半会损坏 git 对象。**不要在 e2e 进行中做 git 操作。**
3. 跑 `MODE=3arm bash $K/preflight.sh`，最后一行必须是 `PREFLIGHT_OK`。
   - driver 拿锁**之后**还会再跑一次，所以它的容器 python 不会与别的客户端同时运行。
   - 单独运行时，如果锁被别的进程拿着，容器检查跳过（WARN）。锁只通过 `/proc/locks` 读，从不去拿。
   - 09:50:53–09:50:56 实跑结果（改完之后）：PREFLIGHT_OK，41 项 ok，0 WARN，0 FAIL；前后 KFD 都为空，锁空闲，没有 op-evolve loop 和 realab driver。
   - 它检查：
     - 主机和 DPM 表、KFD、op-evolve loop、realab driver、锁的持有者、STOP 哨兵、容器、镜像 hipBLASLt 库、磁盘、加载 amdgpu 以来的故障类 dmesg；
     - 三棵树的 md5，包括 driver 和 launcher 每次启动前用的同一个 `trees_check`；
     - 不改 BLAS 环境、s6 ISA 证据、依赖路径（含 refcache 的 fast.pt 和 prod.pt）；
     - 适配层编译、计时器自测、5 个脚本的 `bash -n`、schedule；
     - 容器内 import 检查和适配层功能测试（`HIP_VISIBLE_DEVICES=-1`，不加载真 arm，不初始化 HIP）。
4. 可选：改了 driver 后，用桩 launcher 在 CPU 上空跑一遍控制流。
   ```bash
   LOCKFILE=/tmp/e2e_dry.lock LAUNCHER=$K/tools/stub_launcher.sh SKIP_PREFLIGHT=1 COOL=1 IDLE_MAX=5 bash $K/drive.sh
   ```
   - 不用 docker，不上卡，用另一把锁文件；`LOCKFILE` 设成真锁、或 `LAUNCHER` 设成桩以外的东西，driver 都直接拒绝（exit 2）。
   - KFD 目录和 STOP 文件换成私有的临时路径。真实的 op-evolve、realab、KFD 只打印出来，不算数；只有命令行里带 `e2e_dry` 的假客户端算数。
   - `STUB_FAIL=<情形>` 模拟各种停止，列表在 `tools/stub_launcher.sh` 开头。
   - 09:49–09:50 跑过 19 种情形，全部符合预期；任何一次停止之后都没有再启动下一个进程：
     - 3arm、2arm 正常路径到 `E2E_DRIVE_DONE`，4 次启动依次是 opc fast、opc prod、p1、p2；
     - 在 opcheck fast 注入：门槛不过 1、launcher 拒绝 97、假 KFD holder 97、假 op-evolve loop 97、假 realab 97、代码漂移 93、STOP 90；
     - 在 opcheck prod 注入：门槛不过 1、外来 KFD 91；
     - 在 p1 注入：非有限 96、挂起 98、BLAS 94、memguard 95、dmesg 99、外来 KFD 91、运行中代码改变 93；
     - 未知 RUNS 记号 2；`LAUNCHER=/bin/true` 也是 2，在打开锁文件之前就退出。
   - launcher 自己的守卫和外来 KFD 检测另做了 CPU 单元测试（假 docker、假 KFD 目录）：
     - KFD 里有 holder → 97；快照不符 → 93；
     - 外来 holder 只报一次；只对带本次标记的进程发一次 SIGTERM，不带标记的进程不动。
   - 空跑产生的文件要删掉：`runs/`、`logs/` 里带空跑时间戳和 `STUB` 的文件（这次已删）。

### 5.1 启动

```bash
K=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
MODE=3arm bash $K/preflight.sh | tail -3                                         # PREFLIGHT_OK
setsid nohup bash $K/drive.sh > $K/runs/driver.$(date +%m%d_%H%M).log 2>&1 &     # MODE=3arm RUNS="opc opcprod p1 p2" 为默认
```

**监视**：每 30 s 看一次，正常时不出声。每个进程结束时有一行 `rc=`，整个流程以 `E2E_DRIVE_DONE` 或 `E2E_DRIVE_STOP <code> <原因>` 结束。

```bash
D=$(ls -t $K/runs/driver.*.log | head -1)
tail -n0 -F $D | grep --line-buffered -E "E2E_DRIVE_(DONE|STOP)|!!|WATCHDOG|rc=[0-9]"   # 作为 Monitor 命令，超时后重新挂
```

### 5.2 driver 做什么（`KIT/drive.sh`）

1. 持锁（fd 9，`flock -w 3600`）→ `export E2E_LOCK_HELD=1`（launcher 不再自己拿锁）→ preflight（在锁内跑）→ 存 `runs/tree.<stamp>.sha256`（所有会运行的文件），并核对 `trees_check`。
2. `idle`，每次启动前都做：
   - 先查 STOP 哨兵、op-evolve loop、realab driver；
   - 等 KFD 为空（最多 `IDLE_MAX`）；
   - 冷却 300 s，期间每秒查一次 STOP；
   - 守卫：再查 STOP、op-evolve、realab、KFD，再核对 `trees_check` 和 sha256 快照。
   - 任何一项不过就停（90 / 97 / 93）。launcher 在 docker exec 之前把守卫再跑一遍（97 / 93），运行中每 5 s 查外来 KFD（91）。
3. `opcheck fly fast`：
   - 新的 JIT 缓存；`AMD_SERIALIZE_KERNEL=3 FLY_BWD_SIDE_STREAM=1 FLY_BWD_RECORD_STREAM=0`；
   - 硬超时 600 s，B0 09-28 上 python 墙钟约 9 s（原来是 1400 s，挂住要 23 min 才会被发现）；
   - 内容：E2EAttention fwd + autograd bwd 对 fp32 refcache；kernel fwd；bwd kernel 用参考 o/lse。
   - 门槛：链路 o/dq/dk/dv、kernel bwd dq/dk/dv、kernel fwd o/lse 全部有限且 ≥47 dB，没有 BLAS re-point，计时器正常。
     - 参考是 fp32 refcache，ASM arm 也是对同一份参考打分：B0 09-28 prod 上 ASM 为 50.6–53.2 dB，r29 为 50.8–52.8 dB。
4. `idle`，然后 `opcheck fly prod`：
   - 同上，但**不加** `AMD_SERIALIZE_KERNEL`：k_dkdv + k_dqg 真并发，record_stream=0 的复用真发生；
   - 门槛相同。B0 09-28 r29 prod 的 python 墙钟约 12 s。
5. `idle`，然后 **p1**：`asm;asm,fly,flyr29,asm,flyr29,fly`，92 步，每 11 步 profile 一次。
   - 各 arm 的 profile 步：asm 11/44/77，flyr29 22/66/88，fly 33/55。
   - 稳态窗口 asm 21、fly 21、flyr29 19 步。
   - 这个 6 步周期里，每一对 arm 的两种相邻顺序都各出现一次。
6. `idle`，然后 **p2**：`asm;asm,flyr29,fly,asm,fly,flyr29`，即 fly 与 flyr29 位置互换。profile：asm 11/44/77，fly 22/66/88，flyr29 33/55。
7. 每个进程结束后依次检查：
   - dmesg FAULT → 99
   - 外来 KFD → 91
   - launcher 拒绝启动（docker exec 前守卫）→ 97 / 93
   - 运行中代码改变 → 93
   - watchdog：非有限 → 96，BLAS → 94，挂起 → 98
   - memguard → 95
   - 非有限步、nkfix 事件 → 96
   - rc≠0 → 92
   - 每一项都**立即停下**，不再启动下一个进程。
8. 放锁，跑 `analyze.sh`（CPU）。

**每个训练进程的环境**：

```
E2E_NKFIX=1 NKFIX_CHECK=1 E2E_MEM_STOP=89.5
E2E_FLYCACHE=/tmp/flycache_<tag>_<hhmmss>
E2E_FLY_TREES={"fly":{...s6},"flyr29":{...r29}}
E2E_EXPECT_BLAS_LIB=<镜像库>
E2E_ATTN_EVENTS=KIT/logs/attn_ev.<tag>.jsonl
E2E_RUN_MARKER=<tag>                      # 外来 KFD 检测靠它认出本次运行的进程
-e FLY_BWD_SIDE_STREAM=1 -e FLY_BWD_RECORD_STREAM=0
```

- 卡：`fa-repro`；时钟从 `/sys/class/drm/card1` 每 5 s 采一次（sclk、fclk、busy、功耗、温度，以及 KFD holder 的 PID）。
- watchdog 原有的挂起阈值不变：第一次 attention 900 s、step 1 360 s、单步 max(3× 上一步, 240 s)。
- 启动后会把 worker 的 `/proc/<pid>/environ` 存到 `logs/env.<tag>.txt`，包括 E2E_*、FLYDSL_*、FLY_BWD_*、HIPBLASLT_*、TORCH_BLAS*、NKFIX_*、GPU_MAX_HW_QUEUES、CUDA_DEVICE_MAX_CONNECTIONS、HSA_*、HIP_*。

### 5.3 产物（`KIT/`）

仓库顶层 `.gitignore` 忽略所有 `logs/` 目录，`KIT/.gitignore` 用 `!/logs/` 把这里的 logs 重新纳入。这些文件都小（09-28 的训练主日志约 110 KB），直接 `git add` 就能提交。

| 路径 | 内容 |
|---|---|
| `logs/e2e.<tag>.log` | 训练主日志 |
| `logs/attn_ev.<tag>.jsonl` | 每步 attention fwd/bwd 时间（ms），含逐层值 |
| `logs/clk.<tag>.csv` | 时钟采样，最后一列 `kfd` 是 KFD holder 的 PID |
| `logs/nkfix.<tag>.txt` | nkfix 统计 |
| `logs/env.<tag>.txt` | worker 环境变量证据 |
| `logs/tree.<tag>.sha256` | 本进程启动前所有运行文件的 sha256，加配置文件 |
| `logs/e2e.<tag>.log.foreign` | 只在出现外来 KFD holder 时生成 |
| `logs/hang.<tag>.*` | 只在 watchdog 触发时生成 |
| `runs/<tag>.post.txt` | rc、步数、非有限、nkfix、BLAS、watchdog、memguard、外来 KFD、运行中代码是否改变、`valid`、dmesg FAULT/INFO 行数、新 CPU MCE、s6 stream 模式、墙钟 |
| `runs/<tag>.dbg.txt` | rank-0 debug.log 里的 `[e2e_attn]`、`[nkfix]`、`[flydsl_bwd`、`BLAS-REPOINT` 行（debug.log 在仓库外） |
| `runs/<tag>.treecmp.txt` | analyze.sh 写：本进程运行的代码与第一个进程是否相同 |
| `runs/tree.<stamp>.sha256` | driver 拿锁时所有运行文件的 sha256，每次启动前都与它比对 |
| `runs/opc.{fast,prod}.<stamp>.out`、`runs/opcheck.fly.{fast,prod}.*.json` | opcheck 输出 |
| `traces/<tag>/iteration_*` | kineto trace（gitignored） |
| `/home/lihuzhan/_dbg_l8b/output/amd/root/<tag>/logs/pre_trainer/rank-0/debug.log` | rank-0 日志全文 |

### 5.4 停下以后怎么办

| exit | 含义 | 动作 |
|---|---|---|
| 99 dmesg FAULT | 卡可能已坏 | 只读一次有界 dmesg（`timeout 20 sudo -n dmesg \| tail -80`）分类（card-safety §2a/§3）：<ul><li>有 `wait for reset ack` / `GPU reset begin` / `ring ... timeout` / MES unrecoverable：**需要人 AC 断电**。通知用户并附 dmesg，停掉所有卡上 loop，转 CPU 工作，不再探测卡，**不要** `modprobe -r`</li><li>只有 `Queues reset on process` + 保护错：只是这个进程死了，KFD 空后跑 `health_gemm.sh`</li></ul> |
| 96 非有限 | 本次运行作废 | 不重跑。KFD 空、冷却 300 s、dmesg 干净后 `bash $K/health_gemm.sh` 必须 `HEALTH_OK`，才能做任何卡上工作（包括 op-evolve）。哪个 arm、哪一步先出现非有限：看 debug.log 里 `[nkfix] NON-FINITE` 前最近的 `step_idx`。第二次尝试需要用户同意（09-28 NaN 之后的启动挂过卡） |
| 98 watchdog 挂起 | 证据在 `logs/hang.<tag>.*` | 同 99 的分类；dmesg 干净时同 96 |
| 95 memguard | 显存 >89.5% | 不是故障。确认 `FLY_BWD_RECORD_STREAM=0` 已生效（post 文件里的 `bwd_stream_mode`）。可选重跑 `E2E_NLAYERS=24`（09-28 峰值 71%），但单步比与 B0 不直接可比，且多一次启动 |
| 94 BLAS re-point | 有代码改了 hipBLASLt 环境 | 按 debug.log 的 `!! BLAS-REPOINT by <where>` 找到并修正那棵树，再跑 preflight |
| 97 卡不空 | 另一客户端在用卡，或 op-evolve loop / realab driver 在跑，或 launcher 在 docker exec 前发现 KFD 不空 | 协调后重启 driver |
| 91 外来 KFD | 运行中出现了不属于本次运行的 KFD holder（op-evolve 的 benchmark、realab、宿主上的进程） | <ul><li>本次运行作废（post 里 `valid: INVALID`），数字不用</li><li>先按 99 的方法读一次 dmesg 分类</li><li>dmesg 干净：找出并停掉那个客户端的 loop（不 kill 它的 GPU 进程，等它自己结束）。两个客户端同时上过卡，所以先跑 `health_gemm.sh`，`HEALTH_OK` 后再重启 driver</li></ul> |
| 93 preflight / 代码漂移 | preflight 没过；或代码树与钉住的 md5、与 driver 启动时的快照不同；或运行中代码改变（那个进程作废） | 看 driver 日志里列出的文件，找出是谁改的。确认要跑哪一版后重新跑 preflight，再启动 driver。p1 和 p2 必须是同一份代码 |
| 90 STOP | `$K/STOP` 存在 | 正常停止，不是故障。删掉 `$K/STOP` 再启动 driver；已完成的进程可以手动分析 |
| 1 opcheck | 正确性或计时器门槛没过 | 不进训练。看 `runs/opcheck.fly.{fast,prod}.*.json` |
| 92 rc≠0 | 其它失败 | 看主日志尾部 |

driver 中途停下时不会自动做分析。对已完成的进程，手动运行 `bash $K/analyze.sh $K/runs/tags.<stamp>.txt`；非有限或 watchdog 停下的那个进程只能用来定位问题，它的数字不能用。

## 6. 分析和要出的表

driver 结束时会自动运行下面的命令，也可以手动再跑，比如拿到 part 1 的 op 数字之后：

```bash
K=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
OP="asm_bwd=<ms>,s6=<ms>,r29=<ms>,asm_fwd=<ms>,r16=<ms>" OP_LABEL="realab gb, 6 real dumps, median" \
  bash $K/analyze.sh $K/runs/tags.<stamp>.txt > $K/runs/analysis.<stamp>.txt
```

每个 tag 依次跑下面几步，都用宿主 python3，不需要 torch，不上卡：

```bash
cd $K
python3 $K/tools/steady_arms3.py  logs/e2e.<tag>.log "<spec>" 11 --order asm,flyr29,fly --json runs/<tag>.steady.json
python3 $K/tools/attn_events.py   logs/attn_ev.<tag>.jsonl "<spec>" 11 --log logs/e2e.<tag>.log --order asm,flyr29,fly --json runs/<tag>.events.json
python3 $K/tools/trace_breakdown2.py --json runs/<tag>.trace.json traces/<tag>/iteration_*/rank0_trace.json*
python3 $K/tools/clk_summary.py   logs/clk.<tag>.csv --json runs/<tag>.clk.json
python3 $K/tools/make_table.py    runs <tag1> <tag2> [--op ... --op-label ...]  > runs/TABLE.<stamp>.md
```

**各工具做什么**：
- `steady_arms3.py`：合并了 `steady_arms.py` 和 `e2e_arms3.py`，规则相同。
  - 去掉 step ≤7，以及每个 profile 步 F 的 F−1/F/F+1；step ms 由 tps 换算。
  - 给出周期比，以及相邻配对的比值和差值（两种顺序分开报）。
  - arm 顺序固定（`--order`，默认 asm,flyr29,fly；没列出的 arm 按首次出现排在后面）。配对总是"后者/前者"，所以两个进程都报 flyr29/asm、fly/asm、fly/flyr29。
  - 峰值显存只按进程给：reserved 显存是进程级的，只增不减（p3a 从 step 2 起所有 arm 都是 88.99%），不是 arm 的属性。
  - 用 A0 09-28 p3a 的日志复算，与原报告**逐位一致**：old/asm 1.0671（17 对），new/asm 1.0647（16），new/old 0.9987（17），周期 0.9990（4）。改了 arm 顺序之后又复算了一次，配对结果逐行相同（old/new 不在 `--order` 里，仍按首次出现排序）。
- `attn_events.py`：按同一窗口给出每个 arm 每步的 attention fwd/bwd/FA ms、FA 占单步、逐层中位数，以及相邻配对的 Δfwd/Δbwd/ΔFA 与 Δ单步 − ΔFA。arm 顺序与 `steady_arms3.py` 相同。
- `trace_breakdown2.py`：同 B0 的分桶，另外给出：
  - 每桶的 **wall**（kernel 区间并集）和每次调用的 **span**，s6 有两条 stream，按求和会重复计时；
  - attention 各 stream 的耗时；
  - 有效性判定。
  - 已用 A0 09-28 的坏 trace 验证（判为 INVALID），也用合成的双 stream trace 验证过（sum 259.2 / wall 163.2 / span 163.5，符合构造值）。

**交付给用户的汇总表**（中文，按 RESULT-final §0 的格式，每个进程一行，每个 arm 一组）：

| arm | 进程 | 单步 ms（中位数，IQR） | tps | 对 asm 单步比（相邻配对 n / 周期 n） | 对 flyr29 单步比（相邻配对 n / 周期 n） | attn fwd ms/步（events） | attn bwd ms/步（events） | FA ms/步 | FA 占单步 | trace FA wall ms（有效时） | loss 首/末 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|

`make_table.py` 的表 1 就是这张（"对 flyr29 单步比"一列原来漏了，审查后补上）。峰值显存不按 arm 列，放在健康表里，按进程给。

另外 4 张表：
- **Δ 表**：fly−flyr29、fly−asm、flyr29−asm。Δ单步、ΔFA、Δfwd、Δbwd、Δ单步−ΔFA。
- **两进程一致性（表 2b）**：每个对比在两个进程里的相邻配对单步比、最大相对差，以及 §4 判定 3（≤0.3%）。
- **预期对实测**：32 × Δop(gb) 对 events，以及 §4 的判定。
- **运行健康**：有效与否（`valid`）、rc、非有限、nkfix、BLAS、watchdog、memguard、外来 KFD、代码树（与第一个进程比，以及运行中是否改变）、dmesg FAULT/INFO、新 MCE、负载 sclk 中位数（B0 1342/1371 MHz）、功耗、峰值显存（进程级）、墙钟。

结论要明确回答三件事：
1. s6 的 bwd 收益在 e2e 里是多少 ms/步，是否等于 op 级 × 32；
2. fly/asm 单步比从 B0 的 1.032 变到多少；
3. 剩下的差距在 fwd 还是 bwd。

## 7. 墙钟与挂卡暴露

**墙钟**（3arm 默认）：

| 阶段 | 时间 |
|---|---|
| preflight（在锁内） | 约 10 s |
| 首次冷却 | 300 s |
| opcheck fast | 约 1 min |
| 冷却 | 300 s |
| opcheck prod | 约 1–2 min（B0 09-28 python 墙钟约 12 s，加上容器启动、import、JIT） |
| 冷却 | 300 s |
| p1 | 4–6 min（启动约 1 min；92 步 × 约 1.6 s；step 3/4 的 JIT 各约 +10–20 s；8 个 profile 步） |
| 冷却 | 300 s |
| p2 | 4–6 min |
| 分析 | <1 min |
| **合计** | **约 31–37 min**，其中占卡约 12 min |

- 2arm 方案：约 28–32 min。
- 单步时间按 B0 的约 1.6 s 估；A0 刷固件后还没有 e2e 数，09-28 限频时为 1.95 s。

**挂卡暴露**：风险附着在启动上（card-safety §0），步数本身几乎不加风险。所以本计划用 92 步的长进程，而不是多跑几个短进程。

| 启动 | 次数 | 每次风险 | 依据 |
|---|--:|--:|---|
| 训练启动 | 2 | 16–21% | <ul><li>09-16：3/14</li><li>B0 09-27/28：≥12 次，0 次挂卡</li><li>A0 09-28：2/5，两次都有已识别的诱因（宿主库 + step 1 跑 FlyDSL；前一个进程 NaN），本计划都已消除</li><li>合并 ≤5/31 ≈ 16%</li><li>A0 刷固件后：0 次训练启动，**无数据**</li></ul> |
| op 级（opcheck fast + prod） | 2 | 每次 <2% | A0 刷固件后约 170 个 op 级进程，0 次挂卡（95% 上界约 1.8%） |
| **合计** | 4 | **P(至少挂一次) ≈ 32–40%**（训练 16–21%/次，op 级按 1.8%/次），即约 0.3–0.4 次 AC 断电的期望 | 挂一次后计划就停止 |

- 比原来的 3 次启动（约 31–39%）多约 1 个百分点。换来的是：训练之前，prod kernel + 并发 side stream + record_stream=0 + E2EAttention 这个组合在卡上先跑过一次（审查第 2 条）。

降低暴露的备选：
- 只跑 p1（`RUNS="opc opcprod p1"`）：约 19–24%。p1 的 6 步周期里已经包含每对 arm 的两种相邻顺序，单进程就能回答 §0 的问题。
- 不跑 prod opcheck（`RUNS="opc p1 p2"`）：约 31–39%，但训练启动前没有任何一步在卡上测过上面那个组合。不建议。
- p2 是进程级重复：B0 两个进程相差 0.05%。HANDOFF 的规则是"始终两个进程"。

A0 主机本身还有独立于我们的崩溃风险：10-01 BERT fatal L3，09-28 两次。

## 8. 备选

| 选项 | 命令 | 用途 / 代价 |
|---|---|---|
| B0 原配方 2 arm | `MODE=2arm bash drive.sh` | <ul><li>p1 `asm,fly,asm,fly;asm,fly,fly,asm`（与 B0 fin_p2a 相同）</li><li>p2 `asm,fly,asm,fly;fly,asm,asm,fly`（B0 的 BAAB，warm-up 改成 ASM 先）</li><li>62 步，每 10 步 profile 一次</li><li>与 B0 数字可比性最强，但看不到 s6 对 r29 的同进程差值</li></ul> |
| 只跑一个进程 | `RUNS="opc opcprod p1"` | 暴露约 19–24%，见 §7 |
| 不跑 prod opcheck | `RUNS="opc p1 p2"` | 少一次 op 级启动和约 7 min；训练前没在卡上测过 prod kernel 并发 + record_stream=0。不建议 |
| 不做 kineto profile | `PFREQ=0 bash drive.sh` | <ul><li>A0 刷固件后 rocprofiler 的 kernel-trace 仍是 0 行（0930 REPORT §3），kineto trace 很可能仍 INVALID（09-28 每个 profile 步只有约 4 个 kernel）</li><li>不 profile 时每个 arm 的稳态窗口从约 20 步增到约 28 步，启动次数不变</li><li>代价：失去判断 side stream 是否并发的唯一证据（trace 有效时 attn_bwd 的 wall < sum），以及 gemm/elementwise/idle 的分桶</li><li>默认保留 11，理由见 §10</li></ul> |
| s6 串行消融 | `FLY_BWD_SIDE_STREAM=0 RUNS="p1" bash drive.sh`（另起，需决定） | 分离 side stream 在训练工作点的贡献；多一次训练启动 |
| s6 原样（record_stream 开） | `FLY_BWD_RECORD_STREAM=1 ...` | 完全照搬 op 级；显存可能多约 5 GiB，有触发 memguard 的风险 |
| 24 层 | `E2E_NLAYERS=24 E2E_MEM_STOP=80 ...` | 只在 memguard 触发后才用；峰值约 71% |
| NaN 定位 | `NKFIX_CHECK=2` | 输入输出都查，能区分 GEMM 产生还是传入的非有限值；每步更慢，两个 arm 一样 |

## 9. 文件（`output/1002__e2e/e2e/`）

| 文件 | 作用 |
|---|---|
| `drive.sh` | 驱动（持锁、preflight、冷却、守卫、停止规则、自动分析） |
| `run_e2e_a0.sh` | 单进程 launcher（train/opcheck），改自 `0928__a0_repro/run_e2e_a0.sh`；docker exec 前守卫、运行中外来 KFD 检测、运行后代码复核 |
| `trees.sh` | arm 树、md5 期望值、schedule、s6 stream 开关；三个脚本共用的守卫函数：`kfd_holders`、`oe_loops`、`realab_drivers`、`kfd_owner`、`run_files_sha`、`trees_check` |
| `preflight.sh` | CPU 前置检查 |
| `analyze.sh` | 分析入口 |
| `health_gemm.sh` | 出事后的卡健康检查（GPU，一个进程，4096³ bf16 GEMM），只在 §5.4 的情况下用 |
| `arms/fwd_r16_imglib/` | fwd r16 副本（不改 BLAS 环境），带 `MD5SUMS`、`MD5SUMS.champion`、PROVENANCE |
| `attn_backends/` | 适配层副本（shim、`e2e_attn/{__init__,arms,attn_timer}.py`、opcheck.py） |
| `tools/steady_arms3.py`、`attn_events.py`、`trace_breakdown2.py`、`clk_summary.py`、`make_table.py` | 分析工具 |
| `tools/selftest_timer.py`、`import_check_1002.py`、`functest_adapter_cpu.py` | CPU 自测；后两个在容器里跑，`HIP_VISIBLE_DEVICES=-1`，不初始化 HIP |
| `tools/stub_launcher.sh` | driver 空跑用的桩 launcher（不用 docker，不上卡），`STUB_FAIL` 共 16 种情形，见 §5.0 第 4 条 |
| `configs/`、`logs/`、`runs/`、`traces/` | 运行时生成；`traces/` 已 gitignore，`logs/` 由 `KIT/.gitignore` 重新纳入 |
| `.gitignore` | `!/logs/`（抵消仓库顶层的 `logs/`）、`traces/`、`__pycache__/`、`/STOP` |

## 10. 审查修改记录（10-02 09:30–10:00）

审查结论是"安全设计大体到位"。要求上卡前改 3 处、分析前改 1 处，另有 1 处应改、2 条建议。逐条如下。

| # | 审查意见 | 怎么改的 | 文件 |
|---|---|---|---|
| 1 | 可能出现两个 GPU 客户端：op-evolve 不拿锁；`idle()` 只在冷却前查一次 op-evolve；训练中没有外来 KFD 检测 | <ul><li>(a) 冷却之后（driver 的守卫）和 docker exec 之前（launcher 的守卫）都查 op-evolve loop、realab driver、KFD。op-evolve 改为按进程自己的 comm/argv 匹配；原来的 `ps \| awk '/op-evolve/ && /(resume\|run)/'` 会把命令行里同时含 op-evolve 和 run 的任何非 bash 进程（例如 `tail -F .../op-evolve/.../runs/x.log`）当成 loop</li><li>(b) launcher 运行中每 5 s 检查每个 KFD holder 的 `E2E_RUN_MARKER`。外来的记 `!! FOREIGN KFD <pid> <cmd>`，停掉本次运行，标 INVALID，exit 91。clk CSV 加 `kfd` 列</li><li>(c) §0、§5.0 写明：`E2E_DRIVE_DONE` / `E2E_DRIVE_STOP` 之前不得启动或 resume op-evolve</li><li>另加：realab driver 也在检查之列（它在两个进程之间放锁）</li></ul> | trees.sh、drive.sh、run_e2e_a0.sh、preflight.sh |
| 2 | opcheck 没覆盖 e2e 实际跑的 fly 路径 | <ul><li>默认 `RUNS="opc opcprod p1 p2"`。prod 不加 `AMD_SERIALIZE_KERNEL`，fast 保留</li><li>门槛都是全部有限且 ≥47 dB，参考是 fp32 refcache（ASM 对同一参考为 50.6–53.2 dB）。另加 kernel fwd 的 o/lse，并核对 json 的 arm/shape</li><li>§0、§7 的暴露更新为 4 次启动、约 32–40%</li></ul> | drive.sh、E2E-PLAN.md |
| 3 | 可能悄悄换成别的 s6：md5 只在 driver 开始时查一次 | <ul><li>`trees_check`（钉住的 md5）和 `run_files_sha`（与 driver 拿锁时的快照比）在每次启动前都查，driver 和 launcher 各一次，不符就 exit 93</li><li>训练结束后再算一次，运行中改变就作废</li><li>`analyze.sh` 跨进程比对 `tree.<tag>.sha256`，结果进健康表</li></ul> | trees.sh、drive.sh、run_e2e_a0.sh、analyze.sh、tools/make_table.py |
| 4 | fly 对 flyr29 两个进程方向相反；表 1 缺"对 flyr29 单步比" | <ul><li>`steady_arms3.py`、`attn_events.py` 加 `--order`（默认 asm,flyr29,fly）</li><li>`make_table.py` 表 1 加"对 flyr29 单步比"，新增表 2b（两进程一致性，即 §4 判定 3）</li></ul> | tools/steady_arms3.py、tools/attn_events.py、tools/make_table.py、analyze.sh |
| 5 | 证据不会进 git（仓库顶层 `.gitignore` 的 `logs/`） | <ul><li>`KIT/.gitignore` 加 `!/logs/`，`git check-ignore` 确认已不再忽略；另忽略 `/STOP`</li><li>launcher 把 rank-0 debug.log 里的 `[e2e_attn]`、`[nkfix]`、`[flydsl_bwd`、`BLAS-REPOINT` 行抽到 `runs/<tag>.dbg.txt`</li></ul> | .gitignore、run_e2e_a0.sh |
| 6a | "从不 SIGKILL"与 `timeout -k` 不符 | 保留 `-k`，把计划写准确（§3 末尾）：我们自己只发 SIGTERM；`-k` 只在硬超时的 SIGTERM 之后 20 s 进程仍在时才发。不去掉的理由：去掉之后，一个不响应 SIGTERM 的进程会让 driver 一直等到外层 timeout | E2E-PLAN.md |
| 6b | opcheck fast 的 1400 s 超时太长 | 改为 600 s（`E2E_OPC_TIMEOUT`，外层 docker exec 660 s）。B0 上 fast/prod 的 python 墙钟约 9/12 s | run_e2e_a0.sh |
| 6c | driver 没有 STOP 哨兵 | `$KIT/STOP`，不与 realab 的 `output/1002__e2e/STOP` 同名。idle 开始、冷却期间每秒、守卫里都查，exit 90 | drive.sh |
| 6d | preflight 的容器 python 在拿锁之前跑 | driver 先拿锁再跑 preflight。单独运行 preflight 时，锁若被别的进程持有，就跳过容器检查（WARN）；锁只读 `/proc/locks` | drive.sh、preflight.sh |
| 7a | 逐 arm 的峰值显存不是 arm 的属性 | 表 1 去掉这一列；`steady_arms3.py` 只给进程级峰值，放进健康表 | tools/steady_arms3.py、tools/make_table.py |
| 7b | kineto 可能仍 INVALID，可考虑 `PFREQ=0` 或只 profile 1 次 | **不改**默认值 11，在 §8 加了 `PFREQ=0` 一行。理由：<ul><li>A0 刷固件后还没有过训练进程的 kineto trace，这次正好知道它能不能用</li><li>trace 是判断 s6 的 side stream 在训练里是否并发的唯一证据（§3 第 9 行：`GPU_MAX_HW_QUEUES=2` 时可能退化成串行）</li><li>每个 arm 19–21 步的窗口与 B0 终版相当，B0 两个进程只差 0.05%</li><li>不影响启动次数，也就不影响挂卡暴露</li></ul> | E2E-PLAN.md |
