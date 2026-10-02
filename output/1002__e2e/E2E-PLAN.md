# A0 e2e 运行计划（2026-10-02）：fwd r16 + bwd s6 对 aiter ASM，Llama-3.1-8B 32 层

写于 2026-10-02 08:35–08:50 UTC。写这份计划时没有上卡，只做了 CPU 工作：
- md5 校验、读 ISA dump、跑分析脚本自测；
- 在容器里用 `HIP_VISIBLE_DEVICES=-1` 做 import 检查和适配层功能测试；
- 用桩 launcher 把 driver 的正常路径和全部停止路径空跑了一遍。

**一处需要说明的事**：适配层功能测试的第一个版本调用了 autograd 的 `backward()`。PyTorch 第一次 backward 会为每种设备起工作线程，这时会去问 HIP 运行时有几张卡，结果即使设了 `HIP_VISIBLE_DEVICES=-1`，进程也在 KFD 里**登记了一下**。
- 08:3x 时在 `/sys/class/kfd/kfd/proc` 里看到一个短命的条目，随即消失。
- 没有任何 GPU 工作；前后 KFD 都是空的，当时也没有其他客户端在跑。
- 现在的版本不经过 autograd 引擎；它和 preflight 的 import 检查都按 0.1 s 间隔轮询 KFD 验证过，没有登记。

脚本都在 `output/1002__e2e/e2e/`（下文简称 `KIT`）。它们都是改过的副本，原件一个都没动。
同一目录下的 `arms_src/`、`isa/`、`tools/`、`arms/aa/`、`runs/` 属于做 s6 re-pin 和真实 dump 尺子的另一个 agent，本计划只读这些目录。

## 0. 结论

- **要回答的问题**：bwd s6 在 op 级比 r29 快 19%，比 ASM 快 4%。这个提升在真实训练的每步时间里能不能看到，幅度是否符合预期？
- **设计**：2 个训练进程，每个进程 3 个 arm，按步交替：
  - `asm`：aiter ASM
  - `fly`：fwd r16 + bwd s6
  - `flyr29`：fwd r16 + bwd r29，作为 B0 09-28 的对照
  - 每进程 92 步，每 11 步 profile 一次，step 1–2 只跑 ASM。
  - 训练前先跑一次 opcheck fast，经过 E2EAttention 模块、`AMD_SERIALIZE_KERNEL=3`。
  - `fly` 对 `flyr29` 在同一进程里直接给出 bwd s6 的 e2e 收益；`fly` 对 `asm` 给出与 B0 1.0323 可比的单步比。
- **一条命令**（先满足 §5.0 的前提）：
  ```bash
  K=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
  setsid nohup bash $K/drive.sh > $K/runs/driver.$(date +%m%d_%H%M).log 2>&1 &
  ```
- **时间**：墙钟约 25–30 分钟，其中占卡约 10 分钟，其余是 3 次 300 s 冷却。
- **挂卡暴露**：一共 3 次卡上启动，1 次 op 级、2 次训练。
  - 训练启动的历史风险是每次 16–21%。A0 刷固件后还没有过训练启动，这次是第一次。
  - 两次训练合计约 **30–38% 的概率至少挂一次**，挂了需要人去 AC 断电。
  - opcheck 那次风险 <2%：刷固件后约 170 个 op 级进程，0 次挂卡。
  - 详见 §7。
- **产出**：
  - `KIT/runs/TABLE.<stamp>.md`：§6 的 5 张表。
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
| 3 | 前一个进程刚结束就启动下一个、KFD 里有残留 | driver 全程持有 `/tmp/a0-gpu0.lock`，冷却期间也不放。每次启动前要求 KFD 为空，再冷却 300 s，然后复查；`IDLE_MAX` 1800 s 内等不到就退出（exit 97）。有 `op-evolve` loop 在跑也退出 97 |
| 4 | 显存 89%（A0 曾在 88.30% SIGBUS，后来撤回了"是显存问题"的结论） | memguard 89.5%（B0 终版取值），超过就 SIGTERM 一次（exit 95）。s6 的 `record_stream` 已关（§2）。逐 arm 的峰值显存进表 |
| 5 | JIT 缓存串用（h46/h72） | 每个进程用新的 `FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_<tag>_<hhmmss>` |
| 6 | kineto 只记录约 4 个 kernel/步 | <ul><li>CUDA event 计时是主数据</li><li>`trace_breakdown2.py` 对 kernel 数 <1500、attention range 没归到 kernel、或调用数 ≠ 32 的 trace 标 **INVALID**，不使用</li></ul> |
| 7 | dmesg 过滤会误停 | 刷固件后 A0 在正常工作中也会打 GPU RAS **可纠正** pcie_pl 计数（今天 07:58:50、08:09:46），以及 `svm_range_restore_work ... hogged CPU`。09-28 的过滤（任何含 amdgpu 的新行都算）会把它们当故障停掉。现在分两类：<ul><li>**INFO**：CPU MCE、`[Hardware Error]` 寄存器转储、"N new correctable hardware errors"、ring buffer full、workqueue hog，只记录</li><li>**FAULT**：其余所有含 amdgpu/MES/GCVM/reset/Queues reset/uncorrectable 等的行，停止（exit 99）</li></ul>已用今天的 dmesg 和合成的故障行验证过 |
| 8 | 新因素：s6 的 side stream 第一次进训练进程（多一个硬件队列） | <ul><li>opcheck fast 用同样的 stream 设置，先在 op 级经 E2EAttention 跑一遍</li><li>s6 在 09-30 和今天已在 op 级跑过几十个进程，没有出事</li><li>每个进程把 `[flydsl_bwd ...] DQ_SIDE_STREAM=1 DQ_SIDE_RECORD=0` 这行证据记进 post 文件</li></ul> |
| 9 | 新发现：训练进程里 side stream 可能没有并发 | <ul><li>Primus `runner/helpers/envs/base_env.sh:209` 在训练进程里设 `GPU_MAX_HW_QUEUES=2`（`:218` 还有 `CUDA_DEVICE_MAX_CONNECTIONS=1`）。op 级尺子没设，是 HIP 默认值</li><li>PyTorch 第一次建 stream 时会一次建一整池 stream，2 个硬件队列要被它们轮流共用。所以 s6 的 k_dqg 在 e2e 里**可能与 k_dkdv 同队列、退化成串行**</li><li>最坏情况就是 `FLY_BWD_SIDE_STREAM=0`：op 级损失约 0.9%，即约 0.05 ms/调用、约 1.6 ms/步，比预期收益（约 −41 ms/步）小一个数量级</li><li>为了与 B0（同样是 2）可比，**不改**这个值</li><li>worker 的 `GPU_MAX_HW_QUEUES` 记进 `logs/env.<tag>.txt`。trace 有效时，`attn_bwd` 的 wall < sum 说明有并发</li><li>要单独量并发收益，需要再加一个进程 `E2E_ENV+=" -e GPU_MAX_HW_QUEUES=4"`，另行决定</li></ul> |

其它沿用 B0 终版配方，不改：
- nkfix（`E2E_NKFIX=1`；不开的话每个 bwd GEMM 都落到 MT32x16x32，单步慢约 10 倍）
- `compile.enable=false`，`converters: ["primus_turbo"]`
- seed 1234，MBS=GBS=4，seq 8192，AC none
- 只用 `bash -c`，不用 `-lc`
- 只按 PID 发信号，从不 SIGKILL

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
3. 两个进程的单步比相差 ≤0.3%（B0 为 0.05%）。
4. loss 全程有限，`nonfinite_events: []`，fly 与 asm 的 step 62/92 loss 差在 bf16 噪声内（B0 的 step-62 loss 为 3.530 / 3.517）。

**不正常时下一步**：
- 第 1 条不满足（e2e 收益明显小于 op 级）：
  - 先看 events 的逐层 bwd 和 sclk；
  - 可加一个进程，用 `FLY_BWD_SIDE_STREAM=0` 把 s6 串行化，判断是否是 side stream 在训练工作点（功耗墙下）失效。这要多一次启动，需要另行决定。
- 第 2 条不满足：attention 以外的代价，查 trace（若有效）的 gemm / elementwise / idle。

## 5. 运行步骤

### 5.0 前提（每条都要满足；driver 会核对其中能自动核对的）

1. 卡上没有其它客户端：
   - part 1 的真实 dump A/B（`realab`）已全部跑完；
   - op-evolve job 没在跑（`op-evolve stop --job ...`；task 2 的 op-evolve 要排在 e2e **之后**）；
   - `ls /sys/class/kfd/kfd/proc` 为空。
   - 卡是单张，一次只能一个客户端。driver 拿锁后等 KFD 为空，有 op-evolve 进程会直接退出。
2. 先 commit + push `output/1002__e2e`（代码、配置；traces 已在 `.gitignore`）。A0 主机自己会崩（BERT fatal L3），断电时提交到一半会损坏 git 对象。**不要在 e2e 进行中做 git 操作。**
3. 跑 `MODE=3arm bash $K/preflight.sh`，最后一行必须是 `PREFLIGHT_OK`。driver 启动时还会再跑一次。
   - 08:44 实跑结果：PREFLIGHT_OK，37 项 ok，0 WARN；期间 KFD 一直为空。
   - 它检查：主机和 DPM 表、KFD、op-evolve、容器、镜像 hipBLASLt 库、磁盘、加载 amdgpu 以来的故障类 dmesg、三棵树的 md5、不改 BLAS 环境、s6 ISA 证据、依赖路径、适配层编译和计时器自测、schedule、容器内 import 检查和适配层功能测试（`HIP_VISIBLE_DEVICES=-1`，不加载真 arm，不初始化 HIP）。
4. 可选：改了 driver 后，用桩 launcher 在 CPU 上空跑一遍控制流。不用 docker，不上卡，用另一把锁文件，`STUB_FAIL=nonfinite|hang|blas|memguard|dmesg|opcheck` 可以模拟各种停止：
   ```bash
   LOCKFILE=/tmp/e2e_dry.lock LAUNCHER=$K/tools/stub_launcher.sh SKIP_PREFLIGHT=1 COOL=1 IDLE_MAX=5 bash $K/drive.sh
   ```
   - 08:39–08:42 已跑过：3arm、2arm 正常路径都到 `E2E_DRIVE_DONE`；6 种故障分别得到 exit 1/96/98/94/95/99，都没有启动 p2。
   - 空跑产生的文件要删掉：`runs/`、`logs/` 里带 `_08` 时间戳和 `STUB` 的文件。

### 5.1 启动

```bash
K=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
MODE=3arm bash $K/preflight.sh | tail -3                                         # PREFLIGHT_OK
setsid nohup bash $K/drive.sh > $K/runs/driver.$(date +%m%d_%H%M).log 2>&1 &     # MODE=3arm RUNS="opc p1 p2" 为默认
```

**监视**：每 30 s 看一次，正常时不出声。每个进程结束时有一行 `rc=`，整个流程以 `E2E_DRIVE_DONE` 或 `E2E_DRIVE_STOP <code> <原因>` 结束。

```bash
D=$(ls -t $K/runs/driver.*.log | head -1)
tail -n0 -F $D | grep --line-buffered -E "E2E_DRIVE_(DONE|STOP)|!!|WATCHDOG|rc=[0-9]"   # 作为 Monitor 命令，超时后重新挂
```

### 5.2 driver 做什么（`KIT/drive.sh`）

1. preflight → 持锁（fd 9，`flock -w 3600`）→ `export E2E_LOCK_HELD=1`（launcher 不再自己拿锁）。
2. `idle`：KFD 为空 → 300 s 冷却 → 复查。
3. `opcheck fly fast`：
   - 新的 JIT 缓存；`AMD_SERIALIZE_KERNEL=3 FLY_BWD_SIDE_STREAM=1 FLY_BWD_RECORD_STREAM=0`；
   - E2EAttention fwd + autograd bwd 对 fp32 refcache，以及 bwd kernel 用参考 o/lse。
   - 门槛：o/dq/dk/dv 全部有限且 ≥47 dB（09-28 r29 为 51.2–52.8 dB），没有 BLAS re-point，计时器正常。
4. `idle`，然后 **p1**：`asm;asm,fly,flyr29,asm,flyr29,fly`，92 步，每 11 步 profile 一次。
   - 各 arm 的 profile 步：asm 11/44/77，flyr29 22/66/88，fly 33/55。
   - 稳态窗口 asm 21、fly 21、flyr29 19 步。
   - 这个 6 步周期里，每一对 arm 的两种相邻顺序都各出现一次。
5. `idle`，然后 **p2**：`asm;asm,flyr29,fly,asm,fly,flyr29`，即 fly 与 flyr29 位置互换。profile：asm 11/44/77，fly 22/66/88，flyr29 33/55。
6. 每个进程结束后依次检查：
   - dmesg FAULT → 99
   - watchdog：非有限 → 96，BLAS → 94，挂起 → 98
   - memguard → 95
   - 非有限步、nkfix 事件 → 96
   - rc≠0 → 92
   - 每一项都**立即停下**，不再启动下一个进程。
7. 放锁，跑 `analyze.sh`（CPU）。

**每个训练进程的环境**：

```
E2E_NKFIX=1 NKFIX_CHECK=1 E2E_MEM_STOP=89.5
E2E_FLYCACHE=/tmp/flycache_<tag>_<hhmmss>
E2E_FLY_TREES={"fly":{...s6},"flyr29":{...r29}}
E2E_EXPECT_BLAS_LIB=<镜像库>
E2E_ATTN_EVENTS=KIT/logs/attn_ev.<tag>.jsonl
-e FLY_BWD_SIDE_STREAM=1 -e FLY_BWD_RECORD_STREAM=0
```

- 卡：`fa-repro`；时钟从 `/sys/class/drm/card1` 每 5 s 采一次（sclk、fclk、busy、功耗、温度）。
- watchdog 原有的挂起阈值不变：第一次 attention 900 s、step 1 360 s、单步 max(3× 上一步, 240 s)。
- 启动后会把 worker 的 `/proc/<pid>/environ` 存到 `logs/env.<tag>.txt`，包括 E2E_*、FLYDSL_*、FLY_BWD_*、HIPBLASLT_*、TORCH_BLAS*、NKFIX_*、GPU_MAX_HW_QUEUES、CUDA_DEVICE_MAX_CONNECTIONS、HSA_*、HIP_*。

### 5.3 产物（`KIT/`）

| 路径 | 内容 |
|---|---|
| `logs/e2e.<tag>.log` | 训练主日志 |
| `logs/attn_ev.<tag>.jsonl` | 每步 attention fwd/bwd 时间（ms），含逐层值 |
| `logs/clk.<tag>.csv` | 时钟采样 |
| `logs/nkfix.<tag>.txt` | nkfix 统计 |
| `logs/env.<tag>.txt` | worker 环境变量证据 |
| `logs/tree.<tag>.sha256` | 所有运行文件的 sha256 |
| `logs/hang.<tag>.*` | 只在 watchdog 触发时生成 |
| `runs/<tag>.post.txt` | rc、步数、非有限、nkfix、BLAS、watchdog、memguard、dmesg FAULT/INFO 行数、新 CPU MCE、s6 stream 模式、墙钟 |
| `runs/opc.fast.<stamp>.out`、`runs/opcheck.fly.fast.*.json` | opcheck 输出 |
| `traces/<tag>/iteration_*` | kineto trace（gitignored） |
| `/home/lihuzhan/_dbg_l8b/output/amd/root/<tag>/logs/pre_trainer/rank-0/debug.log` | rank-0 日志，含 `[e2e_attn]`、`[nkfix]`、`[flydsl_bwd]` 行 |

### 5.4 停下以后怎么办

| exit | 含义 | 动作 |
|---|---|---|
| 99 dmesg FAULT | 卡可能已坏 | 只读一次有界 dmesg（`timeout 20 sudo -n dmesg \| tail -80`）分类（card-safety §2a/§3）：<ul><li>有 `wait for reset ack` / `GPU reset begin` / `ring ... timeout` / MES unrecoverable：**需要人 AC 断电**。通知用户并附 dmesg，停掉所有卡上 loop，转 CPU 工作，不再探测卡，**不要** `modprobe -r`</li><li>只有 `Queues reset on process` + 保护错：只是这个进程死了，KFD 空后跑 `health_gemm.sh`</li></ul> |
| 96 非有限 | 本次运行作废 | 不重跑。KFD 空、冷却 300 s、dmesg 干净后 `bash $K/health_gemm.sh` 必须 `HEALTH_OK`，才能做任何卡上工作（包括 op-evolve）。哪个 arm、哪一步先出现非有限：看 debug.log 里 `[nkfix] NON-FINITE` 前最近的 `step_idx`。第二次尝试需要用户同意（09-28 NaN 之后的启动挂过卡） |
| 98 watchdog 挂起 | 证据在 `logs/hang.<tag>.*` | 同 99 的分类；dmesg 干净时同 96 |
| 95 memguard | 显存 >89.5% | 不是故障。确认 `FLY_BWD_RECORD_STREAM=0` 已生效（post 文件里的 `bwd_stream_mode`）。可选重跑 `E2E_NLAYERS=24`（09-28 峰值 71%），但单步比与 B0 不直接可比，且多一次启动 |
| 94 BLAS re-point | 有代码改了 hipBLASLt 环境 | 按 debug.log 的 `!! BLAS-REPOINT by <where>` 找到并修正那棵树，再跑 preflight |
| 97 卡不空 | 另一客户端在用卡或 op-evolve 在跑 | 协调后重启 driver |
| 1 opcheck | 正确性或计时器门槛没过 | 不进训练。看 `runs/opcheck.fly.fast.*.json` |
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
python3 $K/tools/steady_arms3.py  logs/e2e.<tag>.log "<spec>" 11 --json runs/<tag>.steady.json
python3 $K/tools/attn_events.py   logs/attn_ev.<tag>.jsonl "<spec>" 11 --log logs/e2e.<tag>.log --json runs/<tag>.events.json
python3 $K/tools/trace_breakdown2.py --json runs/<tag>.trace.json traces/<tag>/iteration_*/rank0_trace.json*
python3 $K/tools/clk_summary.py   logs/clk.<tag>.csv --json runs/<tag>.clk.json
python3 $K/tools/make_table.py    runs <tag1> <tag2> [--op ... --op-label ...]  > runs/TABLE.<stamp>.md
```

**各工具做什么**：
- `steady_arms3.py`：合并了 `steady_arms.py` 和 `e2e_arms3.py`，规则相同。
  - 去掉 step ≤7，以及每个 profile 步 F 的 F−1/F/F+1；step ms 由 tps 换算。
  - 给出周期比，以及相邻配对的比值和差值（两种顺序分开报）。
  - 用 A0 09-28 p3a 的日志复算，与原报告**逐位一致**：old/asm 1.0671（17 对），new/asm 1.0647（16），new/old 0.9987（17），周期 0.9990（4）。
- `attn_events.py`：按同一窗口给出每个 arm 每步的 attention fwd/bwd/FA ms、FA 占单步、逐层中位数，以及相邻配对的 Δfwd/Δbwd/ΔFA 与 Δ单步 − ΔFA。
- `trace_breakdown2.py`：同 B0 的分桶，另外给出：
  - 每桶的 **wall**（kernel 区间并集）和每次调用的 **span**，s6 有两条 stream，按求和会重复计时；
  - attention 各 stream 的耗时；
  - 有效性判定。
  - 已用 A0 09-28 的坏 trace 验证（判为 INVALID），也用合成的双 stream trace 验证过（sum 259.2 / wall 163.2 / span 163.5，符合构造值）。

**交付给用户的汇总表**（中文，按 RESULT-final §0 的格式，每个进程一行，每个 arm 一组）：

| arm | 进程 | 单步 ms（中位数，IQR） | tps | 对 asm 单步比（相邻配对 n / 周期 n） | 对 flyr29 单步比 | attn fwd ms/步（events） | attn bwd ms/步（events） | FA ms/步 | FA 占单步 | trace FA wall ms（有效时） | 峰值显存 | loss 首/末 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|

另外 3 张表：
- **Δ 表**：fly−flyr29、fly−asm、flyr29−asm。Δ单步、ΔFA、Δfwd、Δbwd、Δ单步−ΔFA。
- **预期对实测**：32 × Δop(gb) 对 events，以及 §4 的判定。
- **运行健康**：rc、非有限、nkfix、BLAS、watchdog、memguard、dmesg FAULT/INFO、新 MCE、负载 sclk 中位数（B0 1342/1371 MHz）、功耗、墙钟。

结论要明确回答三件事：
1. s6 的 bwd 收益在 e2e 里是多少 ms/步，是否等于 op 级 × 32；
2. fly/asm 单步比从 B0 的 1.032 变到多少；
3. 剩下的差距在 fwd 还是 bwd。

## 7. 墙钟与挂卡暴露

**墙钟**（3arm 默认）：

| 阶段 | 时间 |
|---|---|
| preflight | 约 10 s |
| 首次冷却 | 300 s |
| opcheck fast | 约 1 min |
| 冷却 | 300 s |
| p1 | 4–6 min（启动约 1 min；92 步 × 约 1.6 s；step 3/4 的 JIT 各约 +10–20 s；8 个 profile 步） |
| 冷却 | 300 s |
| p2 | 4–6 min |
| 分析 | <1 min |
| **合计** | **约 25–30 min**，其中占卡约 10 min |

- 2arm 方案：约 22–25 min。
- 单步时间按 B0 的约 1.6 s 估；A0 刷固件后还没有 e2e 数，09-28 限频时为 1.95 s。

**挂卡暴露**：风险附着在启动上（card-safety §0），步数本身几乎不加风险。所以本计划用 92 步的长进程，而不是多跑几个短进程。

| 启动 | 次数 | 每次风险 | 依据 |
|---|--:|--:|---|
| 训练启动 | 2 | 16–21% | <ul><li>09-16：3/14</li><li>B0 09-27/28：≥12 次，0 次挂卡</li><li>A0 09-28：2/5，两次都有已识别的诱因（宿主库 + step 1 跑 FlyDSL；前一个进程 NaN），本计划都已消除</li><li>合并 ≤5/31 ≈ 16%</li><li>A0 刷固件后：0 次训练启动，**无数据**</li></ul> |
| op 级（opcheck fast） | 1 | <2% | A0 刷固件后约 170 个 op 级进程，0 次挂卡（95% 上界约 1.8%） |
| **合计** | 3 | **P(至少挂一次) ≈ 30–38%**，即约 0.3–0.4 次 AC 断电的期望 | 挂一次后计划就停止 |

降低暴露的备选：
- 只跑 p1（`RUNS="opc p1"`）：16–21%。p1 的 6 步周期里已经包含每对 arm 的两种相邻顺序，单进程就能回答 §0 的问题。
- p2 是进程级重复：B0 两个进程相差 0.05%。HANDOFF 的规则是"始终两个进程"。

A0 主机本身还有独立于我们的崩溃风险：10-01 BERT fatal L3，09-28 两次。

## 8. 备选

| 选项 | 命令 | 用途 / 代价 |
|---|---|---|
| B0 原配方 2 arm | `MODE=2arm bash drive.sh` | <ul><li>p1 `asm,fly,asm,fly;asm,fly,fly,asm`（与 B0 fin_p2a 相同）</li><li>p2 `asm,fly,asm,fly;fly,asm,asm,fly`（B0 的 BAAB，warm-up 改成 ASM 先）</li><li>62 步，每 10 步 profile 一次</li><li>与 B0 数字可比性最强，但看不到 s6 对 r29 的同进程差值</li></ul> |
| 只跑一个进程 | `RUNS="opc p1"` | 暴露减半，见 §7 |
| 加 prod opcheck | `RUNS="opc opcprod p1 p2"` | 多一次 op 级启动，约 2 min，得到 prod SQNR 和适配层开销 |
| s6 串行消融 | `FLY_BWD_SIDE_STREAM=0 RUNS="p1" bash drive.sh`（另起，需决定） | 分离 side stream 在训练工作点的贡献；多一次训练启动 |
| s6 原样（record_stream 开） | `FLY_BWD_RECORD_STREAM=1 ...` | 完全照搬 op 级；显存可能多约 5 GiB，有触发 memguard 的风险 |
| 24 层 | `E2E_NLAYERS=24 E2E_MEM_STOP=80 ...` | 只在 memguard 触发后才用；峰值约 71% |
| NaN 定位 | `NKFIX_CHECK=2` | 输入输出都查，能区分 GEMM 产生还是传入的非有限值；每步更慢，两个 arm 一样 |

## 9. 文件（`output/1002__e2e/e2e/`）

| 文件 | 作用 |
|---|---|
| `drive.sh` | 驱动（持锁、冷却、停止规则、自动分析） |
| `run_e2e_a0.sh` | 单进程 launcher（train/opcheck），改自 `0928__a0_repro/run_e2e_a0.sh` |
| `trees.sh` | arm 树、md5 期望值、schedule、s6 stream 开关 |
| `preflight.sh` | CPU 前置检查 |
| `analyze.sh` | 分析入口 |
| `health_gemm.sh` | 出事后的卡健康检查（GPU，一个进程，4096³ bf16 GEMM），只在 §5.4 的情况下用 |
| `arms/fwd_r16_imglib/` | fwd r16 副本（不改 BLAS 环境），带 `MD5SUMS`、`MD5SUMS.champion`、PROVENANCE |
| `attn_backends/` | 适配层副本（shim、`e2e_attn/{__init__,arms,attn_timer}.py`、opcheck.py） |
| `tools/steady_arms3.py`、`attn_events.py`、`trace_breakdown2.py`、`clk_summary.py`、`make_table.py` | 分析工具 |
| `tools/selftest_timer.py`、`import_check_1002.py`、`functest_adapter_cpu.py` | CPU 自测；后两个在容器里跑，`HIP_VISIBLE_DEVICES=-1`，不初始化 HIP |
| `tools/stub_launcher.sh` | driver 空跑用的桩 launcher（不用 docker，不上卡），见 §5.0 第 4 条 |
| `configs/`、`logs/`、`runs/`、`traces/` | 运行时生成；`traces/` 已 gitignore |
