# 0914 复现 — c07-1（VR 限频 1100 MHz）→ a37-1（4 卡，负载下 2133–2244 MHz）

复现昨天 `output/0913__opt_plan__claude/PROGRESS.md` 的成绩阶梯，
执行 `phase2/PLAN-4GPU-TOMORROW.md` 的 **HOUR 0 强制项**（四卡重新标定）。

形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`。harness、计时方法、SQNR 门全部沿用
`tools/gfx1250/tune_attention.py`：CUDA event、20 iters / 5 warmup、取中位数、每 rep 冲刷 256 MiB L2。

## 阶梯

| 版本 | n | fwd | bwd | **total ms** | TFLOP/s | 昨天 total | 加速 | 离散 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| turbo 出厂 @`c1325c7e` † | 4 | 5.226 | 25.025 | **30.258** | 254.4 | 59.642 | 1.97× | 1.35% |
| turbo 出厂 @`1cb2e183` | 3 | 5.179 | 22.942 | **28.121** | 273.7 | 59.642 | 2.12× | 1.20% |
| turbo 仅配置调优 | 4 | 2.738 | 16.195 | **18.916** | 406.9 | 36.405 | 1.92× | 2.17% |
| aiter 出厂 | 2 | 3.366 | 13.838 | **17.204** | 447.4 | 34.565 | 2.01× | 0.00% |
| torch flex（锚点） | 2 | 3.771 | 11.536 | **15.307** | 502.8 | 31.337 | 2.05× | 0.13% |
| **树内 vendored 融合（冠军）** | 4 | 2.734 | 10.270 | **12.994** | 592.3 | 24.435 | 1.88× | 0.78% |
| 纯 aiter 调优（上界） | 2 | 2.095 | 9.174 | **11.269** | 683.0 | 21.684 | 1.92× | 0.09% |

† `wt-main` 是昨天从未测过的代码状态，仅作对照。与昨天 59.642 对应的是 `@1cb2e183` 那一行。

**排序与昨天逐行一致**，无一处易位。所有 turbo/aiter 行 SQNR 恒为
`53.67 / 52.24 / 52.31 / 52.71`，与昨天冠军记录（`phase2/ledgers/wq3.jsonl`）
**吻合到小数点后 7 位** —— 数值路径逐位复现。

## HOUR 0 四卡标定

冠军配置：GPU0 13.006 / GPU1 12.963 / GPU2 12.981 / GPU3 13.065 ms。
离散 **0.79%**，接受门 3% → **ACCEPT**，全天用单一基线，不必按卡分别对照。

## 实测持续频率（不是 DPM 表上的数）

空闲 2358 MHz，**负载下只跑到 2133–2244 MHz**，随负载而定、极稳（p5–p95 带宽 <6 MHz）。
本机 dmesg 同样有 `WARN: GPU is throttled ... VR.`——限频在这里也真实存在，
只是代价是 9% 而不是 2×。mclk 1900 / fclk 1950 全程恒定。

对 1100 MHz 的时钟比 **1.94–2.04×**；各行实测加速达到时钟比的 **88–96%**。

## 复现环境

- 容器 `fa-repro`，镜像 `amdprimus/amdprimus:gfx1250-20260910`
  （image id `6e656de79e6c`，与昨天 `fa-tune:deps` 的基础镜像**同一个**）。
- 三个 checkout：`Primus-Turbo`(HEAD `d2f75576`)、`wt-bakeoff`(`1cb2e183`)、`wt-main`(`c1325c7e`)。
- `_C.so` / `libprimus_turbo_kernels.so` 从镜像内 `/workspace/Primus-Turbo` 拷入。
  与远端 checkout 的 md5 不同（各自构建），但 attention 路径纯 Triton/Python，经核查不触及 `_C`。
- `wt-main` 单独打了一个补丁：`primus_turbo/pytorch/core/low_precision.py` 取自分支。
  `c1325c7e` 的版本在 torch 2.11 上 `import` 即抛
  `TypeError: Opaque type ... must subclass torch._opaque_base.OpaqueBase`，进程起不来。
  该文件是量化配置的 opaque 类型注册，不在 attention 路径上。
- `aiter-src` 从 c07-1 rsync（195 MB），容器内补装 `psutil`。

## 三个必须记下的坑

### 1. HEAD 测不出基线 —— 反向已被路由到融合内核

`03a76f61` 让 gfx1250 在 `seqlen_k >= 512 且 batch*hq >= _MIN_PARALLEL_WORK` 时
把反向路由到融合内核（`flash_attn_interface.py` 的 `fused_backward_eligible`），
且**没有环境变量可以关掉**。所以在 HEAD 上 `--impl turbo` 和 `--impl fused` 的反向是同一个内核。
第一轮三行反向全是 10.27 ms 就是这个原因。

真基线必须在路由落地之前的树上测 → `wt-bakeoff` @ `1cb2e183`
（融合内核已 vendor、尚未接线，双内核反向仍是默认路径，且 `--tune` 可用）。
`wt-main` @ `c1325c7e` 连 `_parse_tune_spec` 都还没有，只能测出厂配置。

### 2. 驱动脚本吃掉了 `--tune` 的后半段（已修）

`run_row.sh` / `run_at.sh` 里写的是
`docker exec ... bash -lc "cd $R && python3 ... $*"`。`$*` 在双引号内被内层 shell 重新分词，
`--tune "fwd:num_stages=2; bwd:num_warps=2"` 的分号直接**截断了命令**，
`bwd:num_warps=2` 被当成另一条命令执行并 `command not found`，
Python 只收到 `--tune fwd:num_stages=2`。

后果：「turbo 仅配置调优」一度被记成 **25.695 ms**，实际是 **18.916 ms**，错了 **1.36×**。

`assert_config_applied` 没有失职——它证明的是「到达 Python 的 spec 确实到达了内核」，
它看不见一个在 argv 之前就被销毁的 spec。已修为
`bash -lc 'cd "$0" && exec python3 ... "$@"' "$R" "$@"`，
并让驱动回显 harness 实际收到的 `tune` 与编译出的 `bwd` 配置：

```
bakeoff|turbo-cfgtuned-FIXED  gpu0  ... 18.582 ms  414.2 TF/s  OK
  | seen tune='fwd:num_stages=2; bwd:num_warps=2' bwd=[{'num_warps': 2, 'num_stages': 1}]
```

**`bwd:num_warps=2` 的收益是真的、且几乎完整迁移**：限频态 1.52×，本机 1.39–1.42×
（22.426 → 16.114 ms）。它不是限频产物。

### 3. flex 锚点一开始用了未经 autotune 的配置

自写的 `flex_anchor.py` 走 inductor 默认启发式，反向模板拿到 `num_warps=8`；
inductor 自己的 autotuner 在**完全相同的 tile** 下选 `num_warps=4`，快 **2.43×**。
37.2 ms → **15.31 ms**。两条独立路径（`max-autotune-no-cudagraphs` 与显式
`kernel_options={"num_warps":4}`）结果一致到 0.1%，SQNR 53.67/52.23/52.29/52.71 过门。

本机没有昨天那支 Primus 侧 `bench_attention.py`（`find` 全盘无），所以这一行是**重建**而非重跑。
支持重建正确的证据：修正后 flex 的时钟缩放是 2.05×，与其余各行的 1.88–2.12× 同族；
而 37.2 ms 意味着 0.84×，即在快 2.14× 的卡上变慢，没有任何机制支持。

## 天花板与结论修订（按 2350 MHz 重算）

Roof：以 Phase-1 实测 Triton bf16 GEMM 1002.7 TFLOP/s @1100 MHz 为基
（该基准恰好复现 plan 里的 7.68 / 5.48 ms 两个数），换算到 2350 MHz = **2142 TFLOP/s**。

| plan 中的数（限频态） | 按满频修正 |
|---|---|
| 7-GEMM MFMA 下界 7.68 ms | **3.59 ms** |
| 5-GEMM MFMA 下界 5.48 ms | **2.57 ms** |
| 「真实墙」9.2–9.9 ms | **4.31–4.63 ms** |
| 「Triton 落点」bwd 12.5–14 ms / total 16.7–18.2 | **5.85–6.55 / 7.82–8.52 ms** |
| 「低于 13 ms total 就不是 Triton」 | **低于 ~6.1 ms** |

⚠ 冠军现在 total **13.0 ms**，与那句「13 ms」是**数字巧合**，不能读成「已触及 Triton 极限」。

**反向没有到墙**：10.270 ms 对修正后的 4.31–4.63 ms 是 **2.22–2.39×**；
昨天 20 ms 对 9.2–9.9 是 2.02–2.17×。位置没变，甚至略退。
且 plan 那句「well-scheduled Triton 落在 issue-slot 墙的 1.3–1.5×」
被我们自己的两个 Triton 内核（2.0× 和 2.2–2.4×）证伪。

冠军反向按两种 FLOP 基准（PROGRESS.md 要求的报告纪律）：
- 名义 5-GEMM（5.498 TFLOP）：**534.6 TFLOP/s = roof 的 25.0%**
- 实发 7-GEMM（7.697 TFLOP）：**748.4 TFLOP/s = roof 的 34.9%**

1.22× 的单基准低估被精确确认（7/5）。

### VR 限频的代价：1.65× → **~1.95×**

`phase0/PLATFORM-ESCALATION.md` 的 1.65× 低估了 18%。五行 like-for-like 比值
1.88 / 1.92 / 1.93 / 1.97 / 2.01，中位 1.93×。升级单里两行结论翻转：

- 「30,000 tps = roof 的 128%，**impossible by construction**」→ 满频下是 **59.9%**。
  不再是结构性不可能，但它基本就坐在结构墙上（最好情况 ~173 ms/step）。
- 「MI355X 对标 = roof 的 78%，遥不可及」→ **36.5%**，进入可讨论区间。

### 对今天排期的影响

- **aiter 差距是结构性的，不是时钟**：11.269 vs 12.994 = **1.16×**，昨天 1.13×，
  跨 2.14× 时钟变化后不降反升。
- **差距的一半以上现在在前向**：fwd 2.095 vs 2.734 = 1.31×（0.639 ms），
  bwd 9.174 vs 10.270 = 1.12×（1.096 ms）。
  → **T7（预编译 gfx1250 ASM 前向）应当提级**：它便宜、阻塞已清
  （`a34b9831` / `d2f75576`），且攻的正是变大的那一半。
- **ITEM 1（aiter ASM v3 反向探针）保持 rank 1，但接受门要重算**：
  「<18 ms 就是今天」→ **<8.4 ms**；「≥20 ms 关掉」→ **≥9.4 ms**。
  两条线都跨在纯 aiter 调优 Triton 的 9.174 ms 上，所以实际口径就是
  **必须打赢 9.174，要值得 vendor 得打到 ~8.5**。
- **T2 里有一条顾虑方向反了**：PROGRESS.md 担心 320 KB LDS / 每 CU 单 workgroup 的设计
  「在限频态可能因占用率太低而输」。满频下访存延迟按**周期**算大 2.14×，
  低占用率设计在满频只会更吃亏——这条顾虑在满频**更强**，不是更弱。

## 产物

```
RESULTS.md            本文件
FINAL-TABLE.md        阶梯表（脚本生成）
hour0.jsonl           四卡标定
true_ladder.jsonl     各版本原始测量
corrected.jsonl       修正后的 cfgtuned + flex
ladder.jsonl          第一轮（HEAD，反向被路由到融合内核，仅作记录）
gpu1_bwd_warps.jsonl  双内核反向 num_warps × num_stages 全扫
gpu1_combined.jsonl   修正后 cfgtuned 7 次重复
flex_ledger.jsonl     flex 变体扫描
run_at.sh run_row.sh  驱动（已修引号 bug，带配置回显）
flex_anchor.py        flex 锚点（--num-warps）
flex_correct.py       flex 四张量 SQNR 门
flex_variants.py      flex 变体 + 三种计时器交叉验证
flex_prof.py          flex 内核归因
gpu1_verify_*.py      tune spec 到编译产物的贯通验证
```
