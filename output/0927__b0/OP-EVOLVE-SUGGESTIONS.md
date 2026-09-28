# op-evolve 框架改进建议（依据 B0 2026-09-27/28 两天的 fwd/bwd 运行）

背景：fwd job 从 r4 开始跑到 r12，bwd job 从 r24 跑到 r27。bwd 已经连续 6 轮以上没有提升。
下面每一条都附了在本次运行里实际发生的证据，并按"浪费了多少轮次、会不会导致误判"排了优先级。

## P0：会直接导致误判的问题

### 1. 接受判定参照"每个 shape 的历史最佳"，而历史最佳里混进了没被接受的 arm
- **现象**：fwd r8 的 arm B（m16x8 小 grid 切换）让 fast +29%，但因为 proxy 噪声被拒。之后 fast 的"历史最佳"就停在 r8 这个没被接受的 arm 上（139.09）。r9 的 BG 在同一 session 里三个 shape 全面胜过冠军 r6（+0.4% / +1.0% / +27.5%），得分却只有 0.9984，被拒。这个改动此后**永远合入不了**，最后是 operator 用 refactor hint（h39）手动提升的。
- **建议**：`champions` 只记录已经提升（promoted）的轮次。或者把判定改成"和同一 session 重测的当前冠军比"，这样也和 `beat_measured_same_run` 的思路一致。被拒轮次的每个 shape 的记录，只作为信息保留，不进入基准。

### 2. 默认的 benchmark 模板用逐次交错计时，在功耗受限的卡上有系统性偏差
- **现象**：B0 的 fwd prod 上，同一份代码读数随位置和同进程 arm 组合变化 2–10%；FlyDSL 对 ASM 偏差约 25%，bwd 约 3.4%。由此产生了 3 个假"赢"（r6 被高估，L12 +6.4%、L21 +6.3% 都是假象）。以前说的"beat 之后的 I-cache 惩罚"也是这个效应。详见 `ruler/REPORT.md`、`ruler/bwd/REPORT.md`。
- **建议**：
  - setup 生成的 benchmark 模板默认用**分块计时**（lead 4 + block 9，轮次 palindromic）。
  - 每个排名进程自带一个 A/A 副本；A/A 偏离超过噪声底时，这次结果作废。
  - 框架每 N 轮自动跑一次"尺子体检"：A/A、冠军对 beat 用两种模式各测一次，偏差超过阈值就停下报告。

### 3. 接受判定是 3 个 shape 的算术平均，真正计分的 shape（prod）被稀释
- **现象**：A0 r10 在 prod 上 +0.83%，但因为 fast/proxy 持平被平均成 +0.22%，被拒。bwd r20 按三个 shape 的平均当选冠军，但 prod 上 r19 其实快 1.7%（分块计时、3 个进程确认）。e2e 用的正是 prod 这个 shape。
- **建议**：spec 支持 `evolve.gain_weights`（A0 已实现：prod 1.0、proxy 0.25、fast 0），并且默认按计分 shape 加权。B0 这边还没合入，因为改框架源码需要用户批准。

## P1：浪费轮次或卡时的问题

### 4. 多卡节点上无法把 job 固定到某张卡
- **现象**：`gpu_id` 只写进 prompt。gate 执行的是 `docker exec <container> python validation.py`，不带任何设备环境变量，所以一定落在 device 0。B0 只能靠"每卡一个容器 + 只挂载一个 renderD"硬隔离。rocm-smi 的第一行 sclk 读的是 GPU[0]。
- **建议**：runner 支持 `runtime.runner.docker.env` / `devices`；`gpu_id` 同时进入 gate 和 benchmark 的环境；sclk 见证读被测卡的 sysfs。

### 5. reviewer 的账号缺失会让整个 deep round 失败
- **现象**：B0 上没有 `~/.op_evolve_openai`，fwd r5 的 plan 在 reviewer 这一步 401，这一轮 deep 白费了。
- **建议**：resume 时先对所有 role 做账号预检（`tools/check_llm_account.py`）；reviewer 不可用时自动退回到同 SDK 的第二个 session，并记录下来，而不是让这一轮失败。

### 6. 在多卡共享节点上，deep round 的 power_wall 和 idle 检查按整机判断
- **现象**：`00_preflight` 要求整机空闲，4 卡并行时 power_wall 永远被跳过；`--showpids` 会把其它卡上的进程当成"争用"。
- **建议**：idle 检查和功耗遥测都只看本 job 的卡（按 kfd gpu_id 或 sysfs）。

### 7. regression band 是固定的，不随 shape 的实际噪声调整
- **现象**：proxy 在 6-arm 进程里的离散约 ±6%，而 band 是 0.993，fwd r8 因此被误拒；之前还为 fast 手动放宽到 0.90。
- **建议**：setup 阶段测出每个 shape 的 A/A 噪声，据此自动设置 band（比如 3σ），并写进 spec。

### 8. 平台期没有"回头看"
- **现象**：bwd 从 r21 起连续 6 轮以上没有提升，框架的 stall audit 只审计 pool 和 route。其实把历史冠军 r19 拿出来重测，就能发现它在 prod 上比 r20 快 1.7%。
- **建议**：连续 K 轮不提升时，自动在新尺子下重测最近几个被接受的轮次和几个"几乎赢"的 arm，重新选冠军；同时提示 operator 可能需要结构性的方向（比如 bwd 的 7 对 5 GEMM 结构差）。

## P2：让优化目标更贴近端到端

### 9. op 级的稳态计时和训练中的真实工作点不一致
- **现象**：e2e 里 FlyDSL fwd 每层 1.64–1.85 ms，ASM 1.22 ms；而分块计时的 op 级结果是 FlyDSL 比 ASM 快 3.5%。原因是训练中各个 attention 调用之间夹着 GEMM，卡处在不同的频率和功耗状态，FlyDSL fwd 对这个很敏感（+13%）。
- **建议**：benchmark 增加一个"in-context"模式：每次调用前插入一个固定的 GEMM 负载，模拟训练里的真实工作点。候选胜出后，两种模式都要报告；或者把 in-context 模式按权重计入 score。

### 10. 其它小问题
- `core/hints.py` 只解析 `h<N>` 编号，`L#` 和 `f#` 会被静默丢弃（fwd.md §2 有记录）：应该报错，而不是静默忽略。
- bwd `state.yaml` 从 r20 起 `flops` 的 shape 标签错位，progress.md 里的 ms 列因此是错的。
- refactor hint 的提升会清空全部 champions：这正好绕过了第 1 条的问题，但也把真实有效的记录一起清掉了。修好第 1 条之后，这里应该只清掉受影响的 shape。
- 被 `op-evolve stop` 中断的 act module 重跑时会从头开始（bwd r25 的 act 跑了两遍）。建议支持 step 级断点续跑，或者在 stop 前等当前 step 完成。

## 补充（09-28 profiling 之后）

### 11. 尺子用 randn 输入，会选出在真实数据上变慢的改动（P0）
- **现象**：fwd r6 的投机 softmax 在 randn 上是赢的（+4.6%），但真实训练数据的 score 标准差是 21–53，randn 只有 1，于是 13–25% 的 tile 要重算。在训练中，FlyDSL fwd 每层是 ASM 的 1.45–1.72 倍。关掉投机后，在真实数据上反而快 13–23%，在 randn 上则慢 5%，所以 randn 尺子会一直选中投机。详见 `profile/REPORT.md`。
- **建议**：setup 时从目标模型里 dump 几层真实的 q/k/v，作为计分 shape 之一，或者作为必须同时满足的第二把尺子。任何依赖数据分布的机制（投机、跳过、早停）只能在真实数据上判定。

### 12. 尺子的工作点和训练不一致（时钟）
- **现象**：训练中 sclk 只有 1250–1690 MHz，功耗顶在约 2.13 kW；分块尺子测的是高时钟稳态。FlyDSL fwd 从 2350 MHz 到 1350 MHz 慢了 1.25 倍，ASM 基本不变。
- **建议**：benchmark 增加一种"紧跟 GEMM 突发"的计时模式，候选胜出时两种模式都要报告，并把它按权重计入 score（见第 9 条）。

### 13. job 的达标判据用几何平均，launch-bound 的小 shape 会掩盖计分 shape
- **现象**：bwd r27 的 fast 是 ASM 的 1.66 倍，因为 ASM 在小 shape 上本身就很慢。这一项掩盖了 prod 和 proxy 只有约 0.78 的差距，几何平均 1.007，于是 job 被判"target_met"并自行结束。fwd 也快到这个临界点了。
- **建议**：达标判据要求每个计分 shape 各自不低于 bar；launch-bound 的 shape 只报告，不参与判定。

### 14. refcache 的 sha 失配后被静默忽略，每轮都在卡上重算 fp32 参考
- **现象**：bwd 的 `ut/common.py` 在 refcache 建好之后被改过，从 r24 起 gate 每轮都显示 "IGNORED … recomputing"，也就是在卡上重算 fp32 prod 参考。这正是 09-22 那次导致整机断电重启的触发条件。
- **建议**：sha 失配时让 gate 直接失败并报警，而不是静默地重算；同时提供一个工具，在 `make_inputs` 没变时自动更新 provenance。

### 15. FlyDSL 的 JIT 磁盘缓存 key 不包含模块级常量，同一路径下的新代码可能跑的还是旧二进制（P0）
- **现象**：fwd r16 发现，op/current 被 refactor 换成只改了一个模块级常量（`SPEC_STALE_MAX`）的树之后，`/tmp/flycache` 仍然按旧 key 返回 r13 的二进制。逐字节相同的 A/A 副本因此读数差了 6.6%。
- **原因**：`_jit_function_cache_key` 只哈希函数源码和闭包里的标量，不看函数里读到的模块全局变量。
- **建议**：
  - harness 每个进程用独立的 FLYDSL_RUNTIME_CACHE_DIR；
  - promotion 或 refactor 之后，框架自动清空 job 的 JIT 缓存；
  - 模板里的实验开关一律写成参数，不用模块全局变量；
  - 向 FlyDSL 上游报告这个问题，建议把模块全局变量纳入 key。
