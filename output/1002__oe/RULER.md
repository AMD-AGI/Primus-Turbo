# gb 尺子：两个 op-evolve job 改在"GEMM burst 之后"打分（设计、补丁、上线步骤）

2026-10-02，A0（heliosr-1b114-c07-1，单卡，容器 fa-repro）。**本文的改动都只是"备好"：没有应用到任何 live job 目录，没有用卡。**
按 HANDOFF-A0 §5.4，把 burst 条件放进验收/validation、以及给 fwd 加 `gain_weights`，**都需要用户批准**后才能上线。

写作期间的现场情况（都不是本工作做的，本工作也没有碰）：
- bwd job 在 10:21:42 被别的会话 `op-evolve resume`（PID 319624，round 26 fast，用的仍是 blk 尺子）。本工作只在它启动前读过其文件。
- fwd job 已由并行的 fwdjob 工作从 B0 备份恢复到 A0，并做了 A0 迁移改动（`PT/output/1002__oe/FWDJOB.md`、`fwdjob/`）：final.yaml 换成 fa-repro、新增 h49、退役 h29/h38、fp32 参考改到 CPU、round 20 将在 A0 上重做。本文的 fwd 部分以那份结果为基线。
- e2e 在 10:21 出了结果（`PT/output/1002__e2e/RESULT-e2e.md`），它修正了本任务的前提，见 §1。

缩写：`PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`，`OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`，
`J=$OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934`（bwd），`FJ=$OE/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927`（fwd），
`R=$PT/output/1002__oe/ruler`。blk = job 现在的 blocked 尺子（lead 4 + block 9，256 MB flush，回文）；gb = 每次计时前 10 个 bf16 GEMM。

## 0. 结论

| 问题 | 决定 |
|---|---|
| burst 放在哪 | **每一次**计时调用之前（per-call），不按 block。每次多 ~23 ms，整轮只多 ~1 分钟卡时（§4）；按 block 打每 arm 只省 1–2 s，但 block 内第 2–5 次调用前面是 attention 不是 GEMM，时钟已回升，不再是目标工作点 |
| 怎么计分 | proxy/prod：同一进程先跑原 blk 循环（报告为 `blk_*`），再跑 gb 循环，**按 gb 计分**（`latency_ms`/`tflops` 取 gb 中位数）。fast 不变（blk；bwd 仍取 min，h83）。不取两者混合 |
| validation / 门槛 | `validation.py`、`gates.py` 不改：正确性、确定性不受影响；speed 门槛自动读 gb 数字；目标 margin 不变（bwd 1.20×，fwd 1.0×） |
| 权重 | bwd 已是 `gain_weights {prod 1, proxy 0.25, fast 0}`；fwd 的 B0 分叉没有这一项，随 gb 一起补上同值（需批准） |
| 冠军记录 rebase | **bwd**：轮次边界停下后手改 `state.yaml` 的 champions → best_round，工具先核对 `rounds/<best>/op` 与 `op/current` 逐字节相同，零卡时。**fwd**：不改 ledger；在 round 20（A0 上用 blk 重做）的 act 结束之后、round 21 的 opt 开始之前装上 gb。h48 在 round 21 开头由框架把所有冠军指向 21，所以第一轮 gb 的比较对象就是 op/current。两边都**不用**"再采纳一次 op/current"的空 refactor：它会造成 `rounds/N/op ≠ op/current`，fwd r17–19 已经因此测错了冠军（§3.3） |
| 额外卡时 | bwd 每个 fast 轮约 +0.9 min，fwd 约 +1.3 min（约 +2–3%）；deep 轮约 +2–3 min |
| 证据是否站得住 | e2e 显示：在 FlyDSL 之间的排序（s6/r29）、r29/ASM、fwd 差距三项上，gb 都比 blk 更接近训练；但 bwd s6/ASM 偏悲观约 6%（训练里 s6 已比 ASM 快 2.6%）。验收比较的是 FlyDSL 之间，所以**建议用 gb 做验收尺子**；可选先用 ~6 min 卡时校准 burst 长度（§8.1） |

## 1. 证据：哪把尺子更接近训练

数据为同一批 trees、同一天、A0。op 级数字来自 `RESULT-realab.md`（6 层真实 q/k/v 几何平均，同进程回文 A/B，含 A/A）；e2e 数字来自 `RESULT-e2e.md`（Llama-3.1-8B 32 层，两个进程各 92 步，按层 CUDA event）。表中都是时间比，<1 表示前者快。

| 比值 | blk（现尺子） | gb（本尺子） | e2e（训练） | 更接近 e2e |
|---|--:|--:|--:|---|
| bwd s6 / r29（FlyDSL 之间，验收比的就是这类） | 0.806 | 0.774 | **0.754** | gb |
| bwd r29 / ASM | 1.206 | 1.341 | **1.290** | gb |
| bwd s6 / ASM | 0.972 | 1.038 | **0.974** | blk |
| fwd r16 / ASM | 1.081 | 1.349 | **1.263** | gb |
| 调用窗口 sclk | ~1.45–1.5 GHz | ~1.26–1.39 GHz | 整步中位数 ~1.50 GHz | |

- 四项里有三项 gb 更接近训练，其中包括验收真正依赖的 FlyDSL 之间的排序。s6 相对 r29 的真实收益，blk 低估了 5 个点，gb 只低估 2 个点。B0 上有同样的先例：r13ns 对 r13 在 blk 下慢 4.8%，在真实数据加 burst 下快 12–23%，e2e 里也是快（h44）。
- 唯一的例外是 s6/ASM。训练中 s6 比 ASM 快 2.6%（bwd 每步 −4.5 ms），blk 说对了，gb 却说 s6 慢 3.8%。所以上线后 bwd 的"x beat"会偏悲观约 6%：s6 会读成约 0.96–0.98×，实际约 1.03×。目标是 1.20×，短期内不会误触发 target_met。
- 现在每步唯一的剩余差距在 fwd：fly 对 ASM 整步为 0.999，其中 fwd 每步 +9.2 ms。blk 只看到 8% 的差距，训练中是 26%，gb 读 35%，略微高估。
- e2e 中 ASM 每层的时间（fwd 1.07–1.11 ms，bwd 5.11–5.29 ms）比两把 op 级尺子测出的都短。这说明时钟之外还有别的因素。一种可能是：训练中输入刚由前面的 GEMM/RoPE 写出，在 MALL 里是热的，而两把尺子每次调用前都会把缓存冲掉。这一点还没有测，列为未决（§8.1）。

## 2. 设计

### 2.1 方法

与 `realab.py RA_COND=gb` 和 `fwd-nospec/tools/ab.py AB_COND=gb` 完全一致，§1 的数字就是用它们测的。

- **burst**：每次计时调用前，先 `torch.cuda.synchronize()`，再跑 10 个 `F.linear(x[32768,4096], w[14336,4096])`（bf16，A0 上约 21.5 ms），然后立即调用被测 arm，中间不同步。CUDA event 只包住被测调用；burst 本身的耗时记为 `gb_burst_ms`。
- **image hipBLASLt**：
  - `preimport()` 在 `import torch` **之前**按需赋值 `TORCH_BLAS_PREFER_HIPBLASLT=1` 和 `HIPBLASLT_TENSILE_LIBPATH=<image 库>`。是直接赋值，不是 setdefault。
  - burst 的两次预热放在**加载任何 arm 之前**，这样 hipBLASLt 一开始就用 image 库初始化。
  - 加载完 arm 后再赋值一次。原因是 fwd 树的 `_env.py` 会把它改成 host 库；用 host 库时这个 GEMM 只有约 80 TF/s，时钟也降不下来，尺子作废。
  - 两个 job 的 runner 环境（final.yaml 的 `runtime.env`）传进来的本来就是 host 库，所以这一步两边都需要。
  - 只跑 blk 的进程（`--ruler blk`，或 auto 模式下只测 fast）环境与以前完全一致。
- **作废（void）**：
  - 加载 arm 后先计时 3 次 burst；每个 shape 的 gb 循环跑完后，再看所有调用的 burst 中位数。
  - 超过 `--gb-max-burst-ms`（80 ms）说明 image 库没生效。这时该 shape 不输出 RESULT，`exit 4`，stdout 和 stderr 都打印 `SPEED RULER VOID ...`。
  - 输出里含 "speed"，refactor 步骤的 `_check` 会把它算作速度问题，不会当成正确性问题。也不会产生 Traceback。
- **顺序**：
  - 回文轮，轮数为偶数。每轮每个 arm 计时 `--gb-block`（5）次，不加 lead，因为卡的状态由 burst 决定，而不是由上一次调用决定。
  - 每个 arm 的计时次数 = ceil(gb_iters / 5) 向上取到偶数轮，再乘 5：bwd `--iters 51` 得 60 次，fwd `--iters 101` 得 110 次。
  - gb 下不再做 256 MB flush，burst 本身 1.2 GB 的流量已经起到了 flush 的作用。
- **统计量**：gb 计时的中位数。bwd 的 fast 仍取 min。
- **时钟见证**：
  - 读 hwmon 的 `freq1_input`（A0 上是 card1）。只在 gb 循环期间由一个线程每 1 ms 采样一次；不采样时线程挂起，不影响 blk 循环。
  - 每次调用记两个值：调用窗口内的中位数（`gb_sclk`），以及调用前 3 ms 的中位数（`gb_sclk_pre`，即 burst 末尾）。另外给出 `gb_iqr_pct`。
  - `gb_sclk` 高于 `--gb-max-sclk`（1700 MHz）时只标记 `gb_clock_ok=False`，不作废。
  - A0 上的预期值：`gb_burst_ms` 21–23，`gb_sclk` 1260–1390。
- **A/A**：`--aa LABEL` 把该 arm 逐字节复制到新的临时目录，作为 `LABEL_aa` 加入同一组回文轮。它的结果行带 `aa_of`、`aa_ratio` 和 `blk_aa_ratio`。临时目录在进程退出时删除。

### 2.2 为什么每次调用前都打 burst

| | 每次计时的代价（A0） | 每 arm（bwd 60 / fwd 110 次） |
|---|--:|--:|
| per-call（采用） | bwd prod 28.7 ms，fwd prod 25.0 ms，proxy ~23.7 ms | bwd prod 1.72 s，fwd prod 2.75 s |
| per-block（每 5 次打一次 burst） | bwd 10.8 ms，fwd 6.6 ms | 少 1.1 s / 2.0 s |

per-block 每轮省下的卡时不到 1 分钟，但 block 里有四次调用不在目标工作点，不值得。

### 2.3 计分：为什么 proxy/prod 用 gb，同时保留 blk 报告

- 验收（`core/acceptance.py`）比较的是同一 session 中候选与冠军的吞吐，两者都是 FlyDSL。§1 中这类比值是 gb 更接近训练。
- proxy 也用 gb。k_dqg 只在 proxy/prod 上运行（h80），proxy 是 prod 那批 kernel 的哨兵；如果 proxy 留在 blk，两个 shape 对时钟敏感的改动会给出相反的信号。validation 的目标是"proxy 和 prod 各自达标"，两者也应该用同一把尺子。
- blk 在同一进程中照常跑，紧接在 warmup 之后，位置与以前相同，可以和历史数字对照，但只作报告。每个 arm 的 gb/blk 比值就是它的"时钟敏感度"。代价是 bwd prod 每个 arm 约 0.45 s。
- 不使用 blk 与 gb 的混合分数，因为那会奖励只在高时钟下才成立的改动。
- fast 不变。它是 launch-bound 的哨兵，权重为 0，在 burst 之后测没有额外信息。

### 2.4 validation 与 gates 为什么不用改

- 两个 `validation.py` 都是每个 shape 起一个 `benchmark.py` 子进程（candidate + beat，`--json`），只读 `tflops`、`latency_ms`、`stat`、`iters`、`sclk_*` 这几个字段。默认 `--ruler auto` 下，proxy/prod 的这些字段就是 gb 的数字，字段名不变。§7 的 CPU 冒烟测试已按 validation 的调用方式核对过。
- 正确性（refcache / CPU 参考）和确定性（200 次逐位比较 / dq SQNR）都在 validation 自己的进程里跑，不经过 benchmark.py，所以完全不受影响；validation 进程的 BLAS 环境也不变。
- void 时 benchmark.py 返回 4：
  - bwd validation 打印 "benchmark.py exited 4 ... aborting"；
  - fwd validation 把输出尾部（含 VOID 行）写进 `FAILED` 列表，gate 的 `_failures()` 会把它记进 ledger；
  - 两种情况都只判 speed FAIL，不会误判成正确性问题。
- 目标：bwd 为 1.20× beat（gb 下 s6 约 0.96–0.98×，有 §1 所说的偏差）；fwd 为 1.0×（r16 约 0.74×）。两者都离目标很远，这次不改。以后如果想让 bwd 的目标反映训练中的 s6/ASM，可以另议，例如改用 blk 数字计算目标。

### 2.5 实现

| 文件 | 内容 |
|---|---|
| `R/gbruler.py` | 共用模块，两个 job 各放一份，逐字节相同，装到 `job_context/op/gbruler.py`。包含参数、`preimport()`、时钟见证、burst、`run_gb()`、void 判定和 A/A。模块级不 import torch |
| `R/bwd/benchmark.py` / `R/bwd/bwd_benchmark_gb.diff` | bwd 的现成文件，以及相对 live 文件（md5 `292ac70e`）的补丁，共 10 处改动 |
| `R/fwd/benchmark.py` / `R/fwd/fwd_benchmark_gb.diff` | fwd 的现成文件，以及相对 job 里文件（md5 `17e8804a`，fwdjob 没有改它）的补丁 |
| `R/fwd/gfx1250-flydsl-attn-fwd_final.yaml` / `R/fwd/fwd_final_yaml_gainweights.diff` | 在 fwdjob 迁移后的 final.yaml（md5 `39e3c33c`）上只加 `gain_weights` 一行 |
| `R/tools/make_ready.py` | 用锚点精确替换，从任一基线重新生成上面三个现成文件。锚点对不上就停下并指出是哪一处（说明基线被某轮改过，需要手工合并） |
| `R/tools/state_edit.py` | `rebase-champions`、`realign-best`、`close-round` 三个动作。用框架同样的 dump 设置（`safe_dump(width=100, sort_keys=False)`），并先校验文件能逐字节往返。默认只打印 diff；`--write` 时先备份。loop 还在跑就拒绝执行 |
| `R/tools/hintadd.py` | 把表格行插到 hint.md 索引表末尾，把 `## hN` 段落追加到文末，也可以退役旧的 standing hint。写入前先用 OE 自己的 `core/hints.py` 解析校验 |
| `R/bwd/hint_add.md`、`R/fwd/hint_add.md` | bwd：补上缺失的 h83 表格行，新增 h84。fwd：新增 h51（h49 已由 fwdjob 写好） |
| `R/test/test_gbruler.py`、`R/test/smoke_fake.py` | 纯 CPU 测试（§7） |
| `R/{bwd,fwd}/base/` | 补丁的基线文件，用于复核 md5、重跑 make_ready 和测试 |

benchmark.py 的改动要点。两个 job 结构相同，完整内容见 .diff 文件：

```python
sys.path.insert(0, str(HERE))                                      # gbruler.py
import gbruler  # stdlib only; it must run BEFORE torch is imported
gbruler.preimport(sys.argv[1:], default_shapes="fast,proxy,prod")  # image hipBLASLt env iff gb runs
import torch
...
def measure(..., rulers=("blk",), gb=None):
    ...                                   # warmup unchanged; blk loop unchanged (skipped when "blk" not in rulers)
    gbrec = gb.run(labels, call) if "gb" in rulers else None      # a burst before every timed call
    if gbrec is not None and (void := gb.check(gbrec, shape)): return void
    ...  # rows: latency_ms/tflops from gb; plus ruler, gb_sclk, gb_sclk_pre, gb_burst_ms, gb_iqr_pct, gb_clock_ok, blk_*
def main():
    gbruler.add_args(ap)    # --ruler {auto,blk,gb,both} --gb-iters --gb-block --gb-ng --gb-max-burst-ms --gb-max-sclk --aa
    arms = gbruler.add_aa_arms(arms, args.aa)
    gb = gbruler.GbRuler(torch, args, default_iters=args.iters)   # before load_impl
    fns = {label: load_impl(path) ...}
    gb.after_load()         # re-assign the env, time 3 bursts, start the clock thread; void -> exit 4
```

RESULT 行示例（CPU 冒烟测试的假数据，只用来看字段）：
`RESULT shape=prod arm=current stat=median iters=10 latency_ms=2.0703 ... order=gb10x32768x4096x14336bf16+palindromic5 ruler=gb gb_sclk=1280 gb_sclk_pre=1280 gb_burst_ms=10.716 gb_iqr_pct=0.508 gb_clock_ok=True blk_latency_ms=2.0901 blk_min_ms=2.08144 blk_tflops=526.06 blk_iters=18`

## 3. 冠军记录的 rebase

### 3.1 验收怎么比

`acceptance.judge` 只用 `act.yaml` 里在同一 session 重测的数字：
- 每个 shape 算 `this_round_tflops / champion_tflops`，按 `gain_weights` 加权平均后要大于 1 + min_gain；
- 每个 shape 都不能低于各自冠军乘以 band。

冠军记录的是**轮次**，不是数字（见 `_champions()` 的注释）。每轮由 agent 把这些轮次的代码在同一个进程里重新测。所以换尺子之后，**验收的算术本身**会自动在 gb 下按同一口径比较。

### 3.2 换尺子后哪些东西过时了

1. **哪些轮次是冠军**是在 blk 下选出来的。10-02 时 bwd 的 champions 是 {prod 25, proxy 25, fast 24}，best_round 是 24。round 25 从未被提升（prod +0.19%，被拒），却仍是 prod/proxy 的标杆。fwd 的 best 是 16，champions 是 {fast 19, proxy 18, prod 18}，都是没被提升的轮次（h48 就是为这个问题写的）。如果不处理，第一轮 gb 会拿 blk 选出、且从未提升的代码当标杆。
2. ledger（state.yaml / progress.md 里的 tflops、incumbents、targets、beats）、findings 和 hint 里的 TF/s 全是 blk 下的数字：FlyDSL arm 比 gb 高 4–8%，ASM 高约 1%。agent 会读这些来推理，所以必须明确告诉它们这些数字不可比（h84 / h51 第 4 条）。

### 3.3 为什么不用 refactor 路径（h75 那种）来做"空 rebase"

`_promote_refactor` 会做三件事：设 `best_round = N`，设 `champions = {全部: N}`，并在该轮标上 `refactor`（之后 fast loop 的 `_superseded()` 会提醒 agent 不要把更早的轮次当标杆）。h68/h75 用的就是这条框架路径。

问题在于：refactor 在轮首执行完以后，**同一轮**的 opt/act 会继续改同一个工作目录 `rounds/N/op`（`job.working_copy()` 发现目录已存在就直接复用）。如果这一轮的候选被拒：
- `best_round` 仍是 N，但 `rounds/N/op` 里已经是被拒的候选，和 `op/current` 不一样；
- 之后每一轮的 prompt 都会写 "re-measure `rounds/N/op/` (the incumbent, and it is what `op/current/` is a copy of)"，而这句话是错的。

证据：fwd job 的 best_round 是 16。`op/current` 的 m32x8 md5 是 `370769c9`（r13ns），`rounds/016/op` 的是 `96583ca5`（被拒的 g61）。rounds 17–19 的 act.yaml 都写着 `champion_round: 16 # rounds/016/op re-measured`，也就是把 g61 当成了冠军。

大多数轮次都会被拒。为了换尺子专门做一次"再采纳 op/current"的 refactor，要花一次 agent 调用和一次 validation，而且很可能再次造成同样的缺陷。

### 3.4 最小安全做法

- **bwd**：
  - 在轮次边界停下 job，然后运行 `state_edit.py rebase-champions --round <best_round>`。它把 champions 全部指向 best_round，并追加一条 `hand_edit` lifecycle 事件。
  - 以下情况工具会拒绝执行：loop 还在跑；`best_round` 与给定轮次不同；`rounds/<best>/op` 与 `op/current` 不同。10-02 已核对过 `rounds/024/op == op/current`（diff -rq 为空）。
  - 如果停下时 best_round 正好是一个"refactor 后候选被拒"的轮次，先运行 `realign-best`：把 `rounds/<best>/op` 改名为 `op.rejected-cand-<时间戳>`，再复制一份 `op/current` 到原位置。
  - 这些操作都不占卡时。完成后，第一轮 gb 只会和已提升的冠军（即 op/current）在同一进程中用 gb 比较。h84 会告诉 agent：之前的数字都是 blk 下的，incumbent 就是 op/current。
- **fwd**：
  - 按 fwdjob 的计划，round 20（deep）在 A0 上用 blk 重做 act 和 reflect。这一轮整轮都在 blk 下，口径一致。
  - round 21 开头，框架执行已排队的 h48。按 fwdjob 的 A0 补充：如果 best 仍是 16，就采纳 `rounds/019/op`；如果 round 20 被接受，就做一次 no-op。`_promote_refactor` 随后把所有冠军指向 21。
  - 在 round 20 的 act 结束之后、round 21 的 opt 开始之前，任何一次停下都可以用来装 gb，因为 reflect 和 refactor 都不判速度。这样第一轮 gb（round 21 的 opt）只和 op/current 比较，不需要改 ledger。
  - 如果错过了这个窗口，round 21 就会用 blk 跑完。那就在下一个边界停下：先用 `realign-best` 处理（如需要），再用 `rebase-champions`，与 bwd 相同。
  - §3.3 的缺陷在 round 21 被拒时仍会出现。所以 h51 第 3 条要求 agent 一律用 `--arms current` 测 incumbent，`champion_tflops` 取 op/current 的数字。
  - h48 的依据本身就是"真实数据 + GEMM burst 下 +1.5%、randn 下 +1.7%"（在 B0 上测的），与新尺子方向一致。
  - 另一种做法是手工把 round 20 记为 failed（`state_edit.py close-round`）、让 h48 直接在 round 21 执行。但这会丢掉 fwdjob 为重做 round 20 所做的准备，所以不推荐。
- **框架层面的根治**（需要用户批准，**这次没做**）：让 fast/deep prompt 对 best_round 直接写 `job_context/op/current`；或者 refactor 之后给该轮的候选另开一个工作目录。

## 4. 每轮额外卡时（与现在只跑 blk 相比）

估算依据：
- gb 单次代价取 realab 在 A0 上的实测：bwd prod 为 2.3 s/80 次，即 28.7 ms；fwd prod 为 1.5 s/60 次，即 25.0 ms。proxy 按"burst 前置约 23.0 ms + 调用时间 + 0.5 ms"估算。
- 每个进程另有约 0.3 s 的初始化：hipBLASLt 初始化、2 次预热、3 次核对 burst。
- blk 循环保留不变，所以 gb 的代价全部是额外的。

| | 每 arm 的 gb 循环 | 一次 gate（validation，2 arm × proxy+prod） | 一个 4-arm prod 进程 | 一个 fast 轮 |
|---|--:|--:|--:|--:|
| bwd | prod 1.72 s，proxy 1.43 s | +6.9 s | +7.2 s | **+55 s ≈ 0.9 min**（按 round 25 的模式：4 个 prod 和 2 个 proxy 的 4-arm 进程，加 2 次 validation；round 25 共 42.5 min，约 +2.1%） |
| fwd | prod 2.75 s，proxy 2.59 s | +11.3 s | +11.3 s | **+78 s ≈ 1.3 min**（3 个 prod 和 2 个 proxy 的 4-arm 进程，加 2 次 validation；约 +2–3%） |

- deep 轮测量更多，估计 +2–3 min。deep 轮本身约 3.5 h，占比不到 1.5%。
- refactor 的 validation 每次另加 ~7 s（bwd）或 ~11 s（fwd）。
- agent 每多开一个 4-arm prod 进程，bwd 多约 7 s，fwd 多约 11 s。
- A/A 复制 agent 本来就在做（round 25 的 cur_a/cur_b，fwd 的 champ2），所以 `--aa` 不会增加 arm 数。

## 5. 上线步骤：bwd

前提：
- 用户已批准；
- 负责 bwd job 的会话同意切换；
- 卡上没有其他客户端，fwd job 也没在跑（单卡，两个 job 永远不同时跑）。

```bash
PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo; R=$PT/output/1002__oe/ruler
OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve; J=$OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934
# 1. Stop at a module boundary (writes .stop). Wait until both lines below print nothing.
cd $OE && .venv/bin/op-evolve stop --job gfx1250-flydsl-attn-bwd-20260917-115934
pgrep -af '[o]p-evolve (run|resume)'; ls /sys/class/kfd/kfd/proc
# 2. Base check: must be 292ac70e...; otherwise regenerate (an anchor mismatch = a round edited the harness: merge by hand)
md5sum $J/job_context/op/benchmark.py
#    python3 $R/tools/make_ready.py bwd-benchmark $J/job_context/op/benchmark.py /tmp/bwd_benchmark_gb.py
# 3. Install
cp -p $J/job_context/op/benchmark.py $J/job_context/op/benchmark.py.bak.pre-gb-1002
cp $R/gbruler.py $J/job_context/op/gbruler.py
cp $R/bwd/benchmark.py $J/job_context/op/benchmark.py          # md5: see §10
# 4. Hints: read the diff first, then write; then sync the git copy
python3 $R/tools/hintadd.py --hint $J/job_context/hint.md --add $R/bwd/hint_add.md
python3 $R/tools/hintadd.py --hint $J/job_context/hint.md --add $R/bwd/hint_add.md --write
cp $J/job_context/hint.md $PT/output/0930__bwd/oejob/hint.md
# 5. Champion rebase (dry run first; if it refuses with "rounds/N/op != op/current", run realign-best first)
B=$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["best_round"])' $J/job_context/state.yaml)
python3 $R/tools/state_edit.py rebase-champions --job $J --round $B --reason "gb ruler installed (h84)"
python3 $R/tools/state_edit.py rebase-champions --job $J --round $B --reason "gb ruler installed (h84)" --write
# 6. Card smoke (~1 min, through the job's own runner, same env as the rounds)
cd $J/job_context/op && $OE/.venv/bin/python3 drive.py --key gbsmoke1002 --timeout 900 --cmd \
  "cd $J/job_context/op && ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_gbsmoke_bwd \
   /opt/venv/bin/python3 benchmark.py --arms current,beat --aa current --shapes prod --iters 10 --json /tmp/gbsmoke_bwd.json"
# 7. Resume (plain resume, never --config)
cd $OE && setsid nohup env PATH="$PWD/.venv/bin:$PATH" op-evolve resume --job gfx1250-flydsl-attn-bwd-20260917-115934 \
  >> $PT/output/0930__bwd/oejob/oe-gfx1250-flydsl-attn-bwd-20260917-115934.log 2>&1 < /dev/null &
```

第 6 步冒烟的期望结果：
- rc 0；输出有 `# gb ruler: ... burst 21–23 ms`；
- prod 行 `ruler=gb`，`gb_burst_ms` 21–23，`gb_sclk` 1260–1390；
- current/beat 的 gb 时间比约 1.02，blk 时间比约 0.96（realab 的 randn 数字）；
- `current_aa` 的 `aa_ratio` 在 0.995–1.005 之间。

第一轮 gb 要确认：
- route.md 的约束表里有 h83 和 h84；
- `rounds/<N>/1-opt/raw` 的 RESULT 行里，proxy/prod 是 `ruler=gb`，并且带 `aa_ratio`；
- act.yaml 的 `champion_round` 等于第 5 步的 best_round；
- `gate.log` 里 prod/proxy 是 gb 数字；
- 没有 `SPEED RULER VOID`。

## 6. 上线步骤：fwd（建立在 FWDJOB.md 的恢复和启动步骤之上）

前提：
- 用户已批准（gb 计分和 `gain_weights` 两项）；
- FWDJOB.md §5 的启动前清单已通过（bwd 已停、deep prompt 已切到 fwd 版、FlyDSL cache 已清空、`prelaunch_check.sh` 输出 READY）。

1. 按 FWDJOB.md §6 resume。round 20 在 A0 上用 blk 重做 act 和 reflect。
2. 在日志出现 round 20 的 `3-act: done` 之后执行 `op-evolve stop --job gfx1250-flydsl-attn-fwd-b0-20260927`。job 会停在 reflect 之前，或者停在 round 21 的 h48 refactor 之后、opt 之前，两种都可以。如果已经看到 round 21 的 opt 开始，就改用 §3.4 的"错过窗口"做法。
3. 停下后安装：

```bash
PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo; R=$PT/output/1002__oe/ruler
OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve; FJ=$OE/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927
pgrep -af '[o]p-evolve (run|resume)'; ls /sys/class/kfd/kfd/proc                     # both empty
md5sum $FJ/job_context/op/benchmark.py $FJ/job_context/gfx1250-flydsl-attn-fwd_final.yaml   # 17e8804a / 39e3c33c
cp -p $FJ/job_context/op/benchmark.py $FJ/job_context/op/benchmark.py.bak.pre-gb-1002
cp -p $FJ/job_context/gfx1250-flydsl-attn-fwd_final.yaml $FJ/job_context/gfx1250-flydsl-attn-fwd_final.yaml.bak.pre-gb-1002
cp $R/gbruler.py $FJ/job_context/op/gbruler.py
cp $R/fwd/benchmark.py $FJ/job_context/op/benchmark.py
cp $R/fwd/gfx1250-flydsl-attn-fwd_final.yaml $FJ/job_context/gfx1250-flydsl-attn-fwd_final.yaml   # + gain_weights only
python3 $R/tools/hintadd.py --hint $FJ/job_context/hint.md --add $R/fwd/hint_add.md            # read the diff
python3 $R/tools/hintadd.py --hint $FJ/job_context/hint.md --add $R/fwd/hint_add.md --write
cp $FJ/job_context/hint.md $PT/output/1002__oe/fwdjob/hint.md                                     # git copy
# If round 21's opt already ran under blk (window missed): realign-best (if needed) + rebase-champions, as in §5 step 5
# Card smoke (~1.5 min): the header must show "re-assigned /home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250 -> <image>"
cd $FJ/job_context/op && $OE/.venv/bin/python3 drive.py --key gbsmoke1002 --timeout 900 --cmd \
  "cd $FJ/job_context/op && ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_gbsmoke_fwd \
   /opt/venv/bin/python3 benchmark.py --arms current,beat --aa current --shapes prod --iters 10 --json /tmp/gbsmoke_fwd.json"
cd $OE && setsid nohup env PATH="$PWD/.venv/bin:$PATH" op-evolve resume --job gfx1250-flydsl-attn-fwd-b0-20260927 \
  >> $PT/output/1002__oe/fwdjob/oe-gfx1250-flydsl-attn-fwd-b0-20260927.log 2>&1 < /dev/null &
```

冒烟的期望结果：
- burst 21–23 ms，`gb_sclk` 约 1260–1300；
- current/beat 的 gb 时间比约 1.34（若 h48 已采纳 r19，会略低），blk 比约 1.08；
- `aa_ratio` 在 0.99–1.01 之间。

round 21 的 opt 开始后要确认：RESULT 行里 proxy/prod 是 `ruler=gb`；act.yaml 的 `champion_round` 是 21。

## 7. 已做的验证（全部在 CPU 上，不 import torch）

- `python3 R/test/test_gbruler.py`：7 个单元测试全部通过，覆盖：
  - auto/blk/gb/both 四种模式下各 shape 用哪把尺子；
  - 轮数为偶数；
  - `preimport` 只在需要时改环境；
  - 回文顺序，每次计时前恰好一次 burst；
  - 时钟线程只在 gb 循环期间采样；
  - void 判定；
  - A/A 复制不带 `__pycache__`；
  - 比值字段。
- `python3 R/test/smoke_fake.py`：用假的 torch 包，把两个现成的 benchmark.py 各跑 5 个场景，38 项检查全部通过：
  - auto：fast 用 blk，prod 用 gb 且带 `blk_*` 和时钟见证；bwd 的 fast 仍取 min；A/A 行有 `aa_ratio`；加载 arm 后环境从 host 库改回 image 库。
  - `--ruler blk`：不打 burst，结果行里没有 gb 字段，fwd 仍用 host 库。
  - 加载时 burst 过慢：exit 4，stdout 和 stderr 都有 VOID，没有 Traceback。
  - 循环中途 burst 变慢：该 shape 作废，exit 4。
  - 按 validation 的方式调用（candidate + beat、单个 shape、`--json`）：JSON 字段齐全。
- 补丁：
  - 三个 diff 都能干净地打到各自的基线上，结果与现成文件 md5 一致；
  - `make_ready.py` 从基线重新生成的结果与现成文件逐字节相同；
  - 把 fwdjob 的 `fwd_final_yaml_a0.diff` 打到 B0 原件上，结果 md5 为 `39e3c33c`，与 fwdjob 记录的改动后 md5 一致。
- `state_edit.py`：
  - 在 fwd 的临时副本上，`rebase-champions 16` 因 `rounds/016/op ≠ op/current` 被拒，正好对应 §3.3 的缺陷；
  - 在临时夹具上，`realign-best` 后再 `rebase-champions` 通过；
  - `close-round` 只改了预期的字段，重复执行会拒绝；
  - 对 live bwd 的 dry run 因为 loop 正在跑而拒绝，没有写任何东西。
- `hintadd.py`：
  - 在 bwd hint.md 的 git 副本（与 10:15 时的 live 文件逐字节相同）和 fwdjob 的 hint.md 副本（`d464f93a`）上插入，再用 OE 的 `core/hints.py` 解析：h83/h84/h51 都是 must standing；h48 仍是待执行的 refactor；h49 是 standing；
  - 重复运行会拒绝。

**还没做（需要卡）**：§5/§6 的冒烟测试，以及可选的校准（§8.1）。方法本身已由 realab 在卡上验证过（约 1000 次 burst，rc 0，dmesg 干净）。

## 8. 风险与未决

1. **尺子与训练有偏差**（§1）：gb 对 bwd s6/ASM 偏悲观约 6%，对 fwd r16/ASM 偏高约 7%。训练时整步时钟约 1.5 GHz，gb 的调用窗口约 1.28 GHz。
   - 可选的校准：卡空闲时跑一次 `RA_CONDS=gb RA_GB_NG=3 RA_KINETO=0 bash $PT/output/1002__e2e/tools/realab.sh`，再用 `RA_GB_NG=5` 跑一次，共约 6 min 卡时。
   - 选使四个 e2e 比值的最大对数误差最小的 burst 长度。只要 FlyDSL 之间的比值（s6/r29）不变差，就把 `gbruler.add_args` 里 `--gb-ng` 的默认值改成它，两个 job 一起改。
   - 训练中输入的缓存热度是另一个还没测的因素。
2. **proxy 在 gb 下还没在卡上测过**：第一轮的 `aa_ratio` 会给出答案。如果 fwd proxy 的 A/A 偏差超过 1%，`shape_band.proxy` 可能要从 0.98 放宽到 0.97。
3. **噪声**：
   - gb 单次调用的 IQR：fwd prod 2–3%，bwd 0.4–1.0%。按 110 / 60 次计，中位数的标准误约 0.3% / 0.1%。
   - realab gb 的 A/A（每个 set 20 次）：fwd 0.991–1.00（7 个 set 几何平均 0.999），bwd 0.998–1.000。
   - min_gain 0.007 保留。如果 fwd 第一轮的 |1 − aa_ratio| > 0.35%，就加大 `--gb-iters`。
4. **负载**：gb 循环期间 GEMM 占空比约 75%。一个 4-arm prod 进程多出 7–11 s 的 GEMM 负载，相当于几步训练，用的也是训练同款的 GEMM 和库，不算持续烧卡。
5. **benchmark.py 进程里所有 torch GEMM 都会走 image hipBLASLt**：两个 job 现有的 arm（bwd 的 current/beat，fwd 的各树）都没有 torch GEMM。以后如果有 arm 用 torch GEMM，它的计时会受影响；正确性不受影响，因为 validation 进程的环境没变。
6. **`gbruler.py` 必须和 benchmark.py 放在同一目录**：agent 如果把 benchmark.py 复制到别处，要一起复制，否则会报 ImportError（这个错误很显眼）。
7. **bwd hint.md 原本缺 h83 的表格行**：它有 `## h83` 段落但没有表格行，按 `core/hints.py` 的规则，h83 从未进入 route.md。`bwd/hint_add.md` 顺带补上了。
8. **hint 编号可能冲突**：如果 h84 / h51 在上线前已被占用，`hintadd.py` 会拒绝；改号即可，文中的交叉引用要一起改。
9. **bwd job 现在用 blk 在跑**（10:21 起），这期间各轮都是 blk 数字。rebase 用的是上线那一刻的 best_round，工具会现场核对。
10. **做 PMC/ATT 的轮次**：auto 默认会在 proxy/prod 上打 burst，trace 里会多出 GEMM。h84/h51 已写明：数周期用 `--ruler blk`，看训练时钟下的表现用 `--ruler gb --gb-iters 5`。
11. 原先担心的"fwd gate 在卡上算 13 个 fp32 eager 参考"，已由 fwdjob 的 `fwd_refcache_util_a0.diff` 和 `fwd_eager_guard_a0.diff` 解决（改为 CPU 计算）。fwd deep prompt 的切换也由 `fwdjob/swap_deep_prompts.sh` 负责。

## 9. 回滚

- 尺子：`cp -p <job>/job_context/op/benchmark.py.bak.pre-gb-1002 <job>/job_context/op/benchmark.py`，`gbruler.py` 留着无害。也可以临时让 agent 用 `--ruler blk`。
- fwd final.yaml：`...final.yaml.bak.pre-gb-1002`，只撤掉 `gain_weights`。
- hint / state：
  - 如果改动之后 loop 还没跑过，直接用 `hint.md.bak.pre-gb-*` 和 `state.yaml.bak.pre-gb-*` 覆盖回去；
  - 否则手工把 h84/h51 的类型去掉 standing，状态改为 `superseded (...)`。champions 不用改回，它们指向的就是已提升的冠军。

## 10. 文件清单与 md5（前 8 位）

| 文件 | md5 | 用途 |
|---|---|---|
| `ruler/gbruler.py` | `779b090e` | 两个 job 都装为 `job_context/op/gbruler.py` |
| `ruler/bwd/benchmark.py` | `ea53915f` | bwd 现成文件（基线 `292ac70e`，`bwd/base/`） |
| `ruler/bwd/bwd_benchmark_gb.diff` | | bwd 补丁（141 行） |
| `ruler/bwd/hint_add.md` | | h83 表格行 + h84 |
| `ruler/fwd/benchmark.py` | `63596479` | fwd 现成文件（基线 `17e8804a`，`fwd/base/`） |
| `ruler/fwd/fwd_benchmark_gb.diff` | | fwd 补丁（141 行） |
| `ruler/fwd/gfx1250-flydsl-attn-fwd_final.yaml` | `bbc83a9e` | fwd spec = fwdjob 的 A0 版（`39e3c33c`，`fwd/base/`）+ `gain_weights`；diff 为 `fwd_final_yaml_gainweights.diff` |
| `ruler/fwd/hint_add.md` | | h51 |
| `ruler/tools/{make_ready,state_edit,hintadd}.py` | | 重新生成现成文件 / 改 ledger 与轮次目录 / 改 hint |
| `ruler/test/{test_gbruler,smoke_fake}.py` | | CPU 测试 |
