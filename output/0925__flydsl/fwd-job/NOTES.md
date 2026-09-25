# 前向作业规格 · 操作手册与决策记录（2026-09-23）

本目录下的 `gfx1250-flydsl-attn-fwd.yaml` 是一份**尚未创建的**作业规格。
写这份文件时**没有碰过 GPU**：全部验证手段是 `grep` / `sed` / `python3 -c yaml.safe_load` /
`tools/op_flops.py`（纯 CPU 算术）。没有跑过 `op-evolve run`，没有 `artifacts/` 目录。

---

## ⚠⚠ 零号约束：只有一张卡，两个作业不能同时跑

`runtime.gpu_pool: []`、`runtime.observed.gpu_count: 1`——heliosr-1b114-c07-1 上
只有一张 gfx1250。
`gfx1250-flydsl-attn-bwd-20260917-115934` **此刻正在这张卡上跑 round 13**。

在后向作业释放卡之前创建前向作业，会让两个进程在同一个设备上交错下发 kernel。
两边的所有计时同时失效，而且**看起来不像出错**——这张卡本来就被 VR 限频、会漂移
（后向 spec 第 302-303 行记录过一个计时窗口内 1100 → 967 MHz），污染的结果长得跟时钟漂移一模一样。

**顺序是硬的：先让后向作业结束，或者 `op-evolve stop` 掉它，再创建前向作业。**
后向 spec 自己也从另一侧写了同一条规矩（第 61-63 行："While this job holds the card,
nothing else may use it."）。

---

## 一、操作员要做的事，按顺序

### 0. 等后向作业让出卡
```
op-evolve status --job gfx1250-flydsl-attn-bwd-20260917-115934
# 确认已终结；若未终结且要抢卡：
op-evolve stop  --job gfx1250-flydsl-attn-bwd-20260917-115934
```

### 1. 确认 vendored baseline 树已经就位
```
ls -la /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0923__flydsl/fwd-job/op_baseline/
```
**写这份 spec 开头的时候这个目录还不存在**（实测 `No such file or directory`）；
它在 2026-09-23 12:34 由并行的另一个 agent 落地，12:39 复查：10 个文件，
五个 vendored 的 aiter 文件在 `op_baseline/flydsl_fwd/` 下，
外层是 `impl.py` / `_env.py` / `__init__.py` / `PROVENANCE.md`。
`impl.py:81` 定义 `flydsl_attn_fwd(q, k, v, softmax_scale=None, causal=True)`，
`impl.py:124` 是 `attn_fwd = flydsl_attn_fwd`——**与 spec 的 `op.reference.api` 对得上**。
操作员仍然要自己确认一次。目录为空就**停**，不要继续——否则 setup 会安静地回退到
导入已安装的 aiter，整个作业的 baseline 臂就跑的是别人的树，而这恰好是本作业最贵的失败模式
（见下文 §三.3）。

### 2. 把 spec 放到 op-evolve 的 jobs/ 下
```
cp /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0923__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml \
   /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/jobs/
```
（不是必须放 jobs/，`--config` 接受任意路径；放过去只是为了跟其它作业同址。）

### 3. ⚠ **必须 cd 到 op-evolve 仓库根目录**再 run
```
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve
op-evolve run --config jobs/gfx1250-flydsl-attn-fwd.yaml
```
**为什么必须 cd**：`op_evolve/core/job.py:14` 写的是
`ARTIFACTS_ROOT = Path("artifacts")`——**相对路径**，`job.create()`
（`job.py:194`）默认吃它。在别的目录下执行，artifacts 会落在那个目录下面，
而 `status` / `resume` / `tune` 之后又按各自的 cwd 去找，结果是作业"消失"。

### 4. ⚠ 预算 58 分钟的 GPU 时间给 setup
`op-evolve run` 会**无条件先跑两个 setup 阶段**：`job_setup` 然后 `op_setup`。
本机实测（后向作业的 `job_context/state.yaml:8-15`）：

| 阶段 | 实测耗时 |
|---|--:|
| `job_setup` | 618.3 s（约 10.3 分钟） |
| `op_setup`  | 2850.0 s（约 47.5 分钟） |
| 合计 | **约 58 分钟** |

这不是文档值，是这台机器上量出来的。**这 58 分钟里卡是被占住的**，
所以不要在只剩半小时空窗的时候起作业。

### 5. setup 交出报告后，逐条检查（见 §二"setup 必须证明的四件事"）
`review.auto_approve_after: 10m`——**十分钟后自动批准**。
操作员如果人不在，报告就会自动通过。要审就要在这十分钟里审。

### 6. 之后调旋钮不要重跑 setup
```
op-evolve tune --job gfx1250-flydsl-attn-fwd-<id> \
    --max-rounds N --fast-rounds N --fast-per-deep N --beat-margin X
```
`tune` **只写 final yaml，完全不碰 state.yaml**（`core/tune.py:19-22`），
因此**不会 bump `spec_version`**。这很重要：一旦 bump，两个 setup 阶段都会被判为未完成，
setup agent 会按 `op_setup.md` 重写 `op/baseline`、`op/eager`、`op/ut`、
`op/benchmark.py`、`op/validation.py`——所有手工补丁被覆盖。
（先例：2026-09-23 对后向作业做过 `--max-rounds 40 --fast-rounds 40`，
spec_version 仍为 v000，refcache 与手工补丁完好。见 `STAGE2-S0-PROBE.md` S0-a。）

⚠ `min_gain` **不在 tune 的旋钮里**（`cli.py:76-86` 只有 `--fast-rounds`
`--fast-per-deep` `--max-rounds` `--max-timeout` `--beat-margin` `--target-tflops`）。
要改 `min_gain` 只能手改 resolved yaml。

---

## 二、setup 必须证明的四件事

这四条都写在 spec 里了，这里是给操作员的验收清单。

1. **`attn_fwd` 这个属性名，四个 impl 全改**
   loader 硬编码 `getattr(mod, "attn_bwd", None)`（后向作业
   `op/ut/common.py:104`，紧跟着 `:106` 抛 "defines no \`attn_bwd\`"）。
   前向要把它和四处别名一起改成 `attn_fwd`：
   `baseline/impl.py`、`beat/impl.py`、`eager/impl.py`、`current/impl.py`，
   外加 `benchmark.py` / `validation.py` 里凡是点到名字的地方。
   **改漏一个不会立刻炸**——只有轮到那条臂被加载时才 AttributeError，可能是几小时以后。

2. **FLOP 计数必须是 `backward=False`**
   后向作业的 `op/benchmark.py:72-76` 写死了 `backward=True`。照抄过来，
   前向吞吐会稳定地大 2.5 倍，而且**看不出任何异常**。
   spec 的 `op.shape.shapes` 里已经把三个 shape 的前向 flop/bytes_min 算好写死了，
   benchmark 里的调用也要同步改。

3. **deliberate-edit 证明：改的副本确实是被导入的那份**
   （规矩来自 `jobs/turbo-flydsl-attn-fwd.yaml:82-86`）
   建议用 `O_VARIANT`（`fmha_fwd_prefill_a16w16_m32x8.py:150`，出货 `"v3"`）：
   在**副本里**改成 `"v1"` → 实测中位数要动（101 次实测 v1 快 1.90%）→ 改回 `"v3"` → 数字回来。
   **反例的警告**：0.0–0.1 s 的构建耗时**不能**证明改动被忽略了；
   两条"不同"的臂给出相同计时，通常是参数根本没传进去。
   `n_block` 是 def 时绑定的关键字默认值（`:1422`）且 builder 带 `@functools.cache`，
   改模块常量是空操作。绊线要追踪 `build()` **返回的对象**，不是每次都新建的 `_launch` 闭包。
   最终用 `rocprofv3 --pmc` 读描述符确认（v3 VGPR=224 / v1 VGPR=232，LDS 都是 327680，scratch 都是 0）。

4. **环境变量要"赋值"不是 setdefault**
   `/usr/lib/python3.12/sitecustomize.py:18` 在**每个解释器启动时**
   `setdefault("TORCH_BLAS_PREFER_HIPBLASLT","0")`，早于任何项目代码。
   后向作业的 `op/*/_env.py` 用的也是 `setdefault`，所以**十二轮里一直是空操作，没人发现**。
   setup 必须显式赋值，并且**从测量进程内部**打印生效值。

---

## 三、每个字段是什么意思 / 改了什么

| 字段 | 后向值（`final.yaml` 行号） | 前向值 | 性质 |
|---|---|---|---|
| `job.name` | `gfx1250-flydsl-attn-bwd` (:67) | `gfx1250-flydsl-attn-fwd` | 决定 artifacts 目录名与 `<name>_final.yaml`（`job.py:76-77`、`:199`），必须不同否则撞目录 |
| `op.config.direction` | `[bwd]` + "forward … NOT part of the measured region" (:123-125) | `[fwd]`，注释整段重写 | 前向**就是**被测区域 |
| `op.reference.api` | `flydsl_attn_bwd(do,q,k,v,o,lse,…) -> (dq,dk,dv)` (:153-156) | `attn_fwd(q,k,v,softmax_scale=None,causal=True) -> (o,lse)` | 属性名同步改 |
| `op.reference.impl/logic` | delta/ds/dq/dk/dv 全链 (:141-145, :149-152) | `s=q@k.T*scale` → mask → `lse` → `p` → `o=p@v` | 已逐字存在于 `op/ut/common.py:53-82` (`forward_reference`)，直接抄 |
| `op.config.determinism_gate` | dq/dk/dv bitwise × 200 (:135-138) | **o 和 lse** bitwise × 200 | 200 次**保留** |
| `op.precision_gate` | dq/dk/dv 各 ≥50 dB + 2026-09-16 先例 (:168-174) | **o 和 lse** 各 ≥50 dB，先例删除 | 规则留，证据不留 |
| `op.baseline.root/sources` | 我们自己的三个 kernel (:160-165) | vendored 的 aiter 五个文件 | **别人的代码** |
| `op.baseline.reported_tflops` | `5.8` (:166) | `933.5` | 今天实测 |
| `op.target.beat` | ASM backward + 手写 launcher (:189) | `aiter.ops.mha.fmha_fwd_with_sink_asm` | 保留 `beaten by N%` 子串 |
| `op.target.beat_launcher` | `asm_bwd_launcher.py` (:190) | **删除** | 前向不需要手写 launcher |
| `op.target.beat_reference_figure` | 10.160 ms / 541.2 TF/s (:199-202) | 1.557 ms / 1412.5 TF/s | 今天实测 |
| `op.shape.shapes` flop/bytes_min | `--backward` 生成 (:181-186) | 六个数全部重算 | 见下 |
| `evolve.min_gain` | `0.0` (:211-215) | `0.015` | 见 §四.1 |
| `evolve.max_rounds` | `40` (:209) | `12` | 见 §四.2 |
| `evolve.schedule.fast_rounds` | `40` (:217) | `6` | 见 §四.2 |
| `evolve.schedule.fast_per_deep` | `5` (:219) | `3` | 同上 |
| `runtime`（整块） | (:221-303) | 原样，只加 hipBLASLt 注释 | 见 §四.5 |

### 行号是否移动了？

**移动的是总行数，不是这些字段的行号。**
任务里写的是"286 lines"，而 2026-09-23 实测
`wc -l gfx1250-flydsl-attn-bwd_final.yaml` = **307 行**（md5 `cf6d36c0...`，20803 字节，
mtime 2026-09-23 11:41）。增长来自 `runtime.env` 里 2026-09-23 新加的那段 hipBLASLt
S0-b 注释（现第 260-284 行），它位于**所有被改字段之后**，
所以上表里引用的每一个行号都是在当前这份 307 行文件里逐一核对过的、仍然有效。

### 重算的六个数（`tools/op_flops.py`，不去 --backward，不碰 GPU）

| shape | 前向 flop | 前向 bytes_min | 后向旧值 flop | 比 |
|---|--:|--:|--:|--:|
| fast  b1 s1024 hq8 hkv2   | `2.14958e+09` | `5.24288e+06` | 5.37395e+09 | 2.500 |
| proxy b1 s4096 hq32 hkv8  | `1.37473e+11` | `8.38861e+07` | 3.43681e+11 | 2.500 |
| prod  b4 s8192 hq32 hkv8  | `2.19929e+12` | `6.71089e+08` | 5.49823e+12 | 2.500 |

`bytes_min` 三个全部整 2.000 倍。两个比值都是教科书值，作为誊写错误的交叉检查。
命令与完整输出已逐字粘在 spec 的 `op.shape.shapes` 注释里。

---

## 四、我做的判断，读者可能不同意的地方

### 1. `min_gain = 0.015`，取自**跨会话漂移**而不是**单 rep 标准差**

两个候选数（都在 `STAGE2-FWD-SWEEP.md` 里）：
- **单臂 rep 标准差**：101 次下 1.17%（v1）/ 1.21%（v3）/ 1.39%（ASM）。
- **同一份代码跨会话漂移**：约 **1.5%**。

我取了大的那个，落在它上面一点：`0.015`。

**反方意见（我认为成立、但没采纳）**：101 次的**中位数**比单 rep 的 sd 紧得多
（粗算 ~0.12%），而且 `beat_measured_same_run: true` 意味着两条臂在同一次运行里一起测，
**轮内比值会把大部分会话漂移抵消掉**。按这个读法 0.005 也站得住，
而 0.015 会让已经确认为真的 `O_VARIANT` 1.90% 只剩 0.4 个点的余量。

**我仍然取 0.015 的理由**：漂移是**唯一一个被实际观察到、在代码没变的情况下让这个 kernel
的数字动起来**的量。被噪声推上去的冠军不重跑每一轮就救不回来；
而在第 N 轮被卡掉的真实收益，第 N+1 轮再提一次就行。**两类错误的代价不对称。**

⚠ 这个数**不能用 `op-evolve tune` 改**，只能手改 yaml。所以如果要改，越早越好。

### 2. `max_rounds = 12`，`fast_rounds = 6`，`fast_per_deep = 3`

后向的 40/40 的理由是"差距 93x + 搜索空间未开采"。**两条在这里都不成立**：
baseline 是厂商成熟内核，而且**搜索空间已经被开采过三刀**（全部在 2026-09-23、
建作业之前，见 `STAGE2-FWD-SWEEP.md`）：

- `n_block` 单独放大 → 慢 3.2× / 9.9×（VGPR 溢出，不是 LDS）；
- `(BLOCK_M, n_block)` 五点联动 → 出货点是**局部最优**；
- 4 波 → **路障**（flydsl 0.3.2 下 V1 loader 不能编译，V2/V3 manager 在 `num_waves != 8` 硬抛）。
  ⚠ 这条**不能**记成"4 波更慢"。

剩下的是：一个已确认的**免费收益**（`O_VARIANT` v1，+1.90%，还没进 baseline），
和两条从未在前向试过的结构性杠杆（KV 循环的深展开 / 软件流水；派发顺序——
后向 round 12 单靠派发顺序拿过 14%）。
12 轮够落地那个免费收益、给两条杠杆各一个 deep 轮、并在不成立时给出一份可信的证伪。

`fast_rounds` 从 40 砍到 6，**因为那条论据的另一半过期了**：
2026-09-17 写 40 的时候，这张卡上 profiling 是被全面禁止的，deep 轮基本白跑。
2026-09-23 禁令解除（六次 rocprofv3 全部 rc=0、KFD 干净、零 page fault），
唯一的限制是**只有 `--pmc` 有数据**（`--kernel-trace` 在这台机器上 dispatch 表 0 行），
约 2 分钟/组。deep 轮现在值这个钱，而剩下两条杠杆恰恰是需要计数器而不是猜的那种。

**可能的反对**：12 轮太少，前向是个 1.51x 的硬缺口。
**回应**：三个旋钮都能用 `op-evolve tune` 就地上调、不 bump spec_version、不重跑 setup。
起始值不必一次定死，**只需要小到"定错了也便宜"**。反过来定 40 却发现方向不对，
要么白烧单卡时间，要么中途 stop 留一个半截作业。

### 3. `beat_margin` 保持 `0%`（parity），没有软化

后向用 0%（跟 ASM 打平即过）。前向的起点比 ASM 慢 **1.513×**，
所以同一个"0%"在这里意味着**一次性补上 51%**——比后向的 0% 难得多。

**反对意见**：之前那个 gfx950 的 turbo 前向作业用的是 1.20，而且高不可及的门槛
会让每一轮的 score 都在 0.66 附近，看起来像没进展。
**我的理由**：`beat_margin` 是**目标**不是**接受标准**——round 是否被接受由
`min_gain` 与上一轮冠军的比值决定，不由 beat 门槛决定。
把目标设成"打平厂商 ASM"是这个作业真正的问题；设成一个能蒙对的数字会让结论没有意义。
而且这是**唯一一个 `op-evolve tune` 能直接改的门槛**（`--beat-margin`），改起来最便宜。

### 4. `precision_gate` 保留了两张量分别检查的规则，但**删掉了后向的先例**

后向用 2026-09-16 的 `out 53.67 / dq 52.24 / dk -inf / dv -inf` 来论证"必须分别检查"。
那条证据是关于 dk/dv 的，**本作业里这两个张量不存在**，引用它等于借别人的证据。
规则本身独立成立（未覆盖缓冲区读出 -inf 是一类 bug，不是后向专属），而且
NaN 预填充的机制（`poison_allocator`）已经在 `op/ut/common.py` 里现成。

**留了一个明确的未决**：50 dB 是为 bf16 的 `o` 定的；`lse` 是 fp32 自然对数，
对它算 SQNR 数值上合法但**不是同一个量纲**，50 dB 对它是否合适 **UNVERIFIED**。
spec 要求 setup 在 round 0 把 baseline 的 lse SQNR 和 o SQNR 一起报出来，
让操作员**拿着数字**重调一次阈值。

作为标定（不是门槛）：2026-09-23 实测 FlyDSL 前向的 `o` 是 51.63–51.69 dB，
ASM 前向是 53.62 dB。**50 dB 在 baseline 臂上只有约 1.6 dB 余量**——
这是一道真门，不是走过场，掉 2 dB 的轮次会被挡下。

### 5. `TORCH_BLAS_PREFER_HIPBLASLT` 留在 `'1'`

任务的措辞是"leave 它在能让所有臂一致的值上"。严格说两个值都"一致"——
runner 对每个进程导出同一个值。所以真正的问题是**哪个值是安全的**，
2026-09-23 之后答案是 `'1'`：

- v000 的诊断「这个镜像缺 hipBLASLt 的 gfx1250 Tensile 库」**是错的**。
- 容器里有**三个** hipBLASLt 库目录，只有一个可用；
  hipBLASLt 的**默认搜索路径**（`_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library`）
  缺 gfx1250 载荷——这才是 `HIPBLAS_STATUS_INVALID_VALUE` 的真因。
- `HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250`（宿主机
  ROCm 10.1.0 的完整 328 文件库）**让它工作**。实测干净：bf16/fp32 GEMM 于
  512/2048/4096，以及精度门当年 fault 的 fp32 QK^T `[2,16,4096,128]`；零 rocblaslt error、零 dmesg 故障。
- **没有**跑旧 `'0'` 做反向对照——那等于故意重现一次挂卡。

出处：`output/0923__flydsl/STAGE2-S0-PROBE.md` 的 S0-b 一节。

### 6. `determinism_gate` 的 200 次**保留**，但我把"这不要钱"标成了 source-level 论证

任务要求验证"aiter 前向没有 atomic、没有 split-k"。我自己 grep 了：

```
grep -rn "atomic\|split_k\|splitk\|split-k" --include=*.py \
    /home/lihuzhan/code/aiter-src/aiter/ops/flydsl/kernels/fmha_gfx1250/
→ 零输出
```

覆盖 `fmha_fwd_prefill_a16w16_m32x8.py`（2233 行）与
`fmha_b16_buffer_managers.py`（1914 行）全部。

⚠ 但我在 spec 里明确标了 **STILL UNVERIFIED**：**本机从来没有真的跑过 200 次 bitwise 比对**。
源码层面的论证很强，但它仍然只是源码层面的论证。
spec 要求 setup 实际跑一次并报结果，而不是把这句话当成"已通过"继承下去。

### 7. `op.baseline.root` 指向一个**当时还不存在**的目录

`root: .../fwd-job/op_baseline`——由另一个 agent 并行搭建，写 spec 时 `ls` 还是
`No such file or directory`。我没有等它，而是在 spec 里加了一条**硬停条件**：
setup 必须先确认该目录存在且非空，否则停。
理由是"安静地回退到已安装的 aiter"是本作业**最贵**的失败模式——
它会让 baseline 臂和 beat 臂读同一棵树，作业报出一个整洁的"无变化"，
同时已经毁掉了自己的 baseline。

### 8. reported_tflops 用 933.5 而不是报告里的 947.8

`STAGE2-FWD-SWEEP.md` 对同一个 2.356 ms 印的是 947.8 TF/s，因为它的 harness 用了
一个略大的 FLOP 数（约 2.233e12）。spec 全篇统一用 `tools/op_flops.py` 的
2.19929e12 → **933.5 TF/s**，这样作业里每一个数字都可比。
**时间是测量值，TF/s 是导出单位。** ASM 那边同样处理：1.557 ms → 1412.5 TF/s。

---

## 五、不要碰的东西

- ⚠ **不要编辑** `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/` 下的任何东西——那是正在跑的后向作业。
- ⚠ **不要整体复制** 后向作业的 `job_context/op/`：里面的 `core`（385 MB）与
  `core.gpu`（33 MB）是 **root 属主的 core dump**，不是代码。
  需要的只有 `op/ut/common.py:53-82` 的 `forward_reference` 这一个函数。
- ⚠ **不要编辑** `/home/lihuzhan/code/aiter-src`。它是 beat 臂**也在**导入的树；
  在那里改一行会让两条臂一起动，作业报"无变化"而 baseline 已毁。
  `op/current/` 是唯一可写的副本。
- ⚠ 后向作业的 `op/refcache/{fast,proxy,prod}.pt` **不能搬过来**：
  provenance 里绑的是后向的 `eager/impl.py` 与 `ut/common.py` 的 sha，
  validation 会判 "IGNORED, provenance differs"。
- ⚠ **不要开 rocprofv3 PC sampling**（2026-09-11 wedge 过 MES）。`--pmc` 可以。

---

## 六、本目录写出的文件

```
/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0923__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml   (721 行，yaml.safe_load 通过)
/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0923__flydsl/fwd-job/NOTES.md                        (本文件)
```

目录外没有写任何东西。
