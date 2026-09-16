# 0915 交接

分支 `dev/lhz/attn`（已推送到 `AMD-AGI/Primus-Turbo`）。本机 = c07-1，单卡 gfx1250，VR 限频 1100 MHz。

## 一句话

算子层面把 attention 从出厂 **55.785 ms** 推到 **11.726 ms（4.757×）**，
**但端到端 A/B 证明它在真实训练里是负的** —— 关掉 ASM 反向反而快 6.78%。
当天真正的正收益来自别处：一个 hipBLASLt 的路径错位，修好后训练吞吐 **8.3×**。

## 先读这个：算子级胜利 ≠ 端到端胜利

20 步训练，唯一差别是 `PRIMUS_TURBO_ATTN_DISABLE_ASM_BWD`：

| | tps | mfu |
|---|--:|--:|
| ASM 反向 **开** | 1858 | 34.49% |
| ASM 反向 **关** | **1984** | **36.83%** |

离散 0.4% / 0.2%，差异是噪声的 30 倍。

op 级反向 17.686 → 10.160 ms，32 层该省 241 ms，实际单步**多了约 1.2 秒**。
差的约 1.4 秒/步是分配开销：ASM 反向需要 fp32 `dq_acc`（融合反向不需要，它不做原子累加）
加 4 倍大的 dk/dv，每层每步 1.07 GB，32 层约 **34 GB 的分配 churn**。
op 级 harness 里这些张量被缓存分配器复用，这笔账完全看不见。

已实现 scratch 复用（只缓存中间缓冲，返回给 autograd 的仍是新张量），
op 级 SQNR 逐位不变、耗时中性，**端到端效果见 `E2E-AB.md` 的第二轮结果**。
若第二轮仍为负，**资格门必须默认关闭**。

**这个 campaign 在今天之前的所有结论都只有 op 级证据**，因为 e2e 从来跑不起来
（`enable_gqa` 的 TypeError 把 converter 关掉了）。第一次跑通它，就推翻了主线成果。

## 成绩阶梯（本机同口径，不经跨机换算）

| 版本 | n | fwd | bwd | total | 离散 |
|---|--:|--:|--:|--:|--:|
| turbo 出厂 `1cb2e183` | 3 | 10.651 | 45.134 | 55.785 | 0.93% |
| 强制 Triton 前向 + 融合反向 | 2 | 4.166 | 17.835 | 22.001 | 0.06% |
| ASM 前向 + 融合反向（晨间冠军） | 3 | 1.549 | 17.675 | 19.233 | 0.22% |
| **ASM 前向 + ASM 反向（产品路径）** | 7 | 1.568 | **10.118** | **11.687** | 0.5% |

反向 10.118 ms 略高于文档给的原结构"真实墙"9.2–9.9 ms。**这不代表到顶** ——
那条墙是给我们那个 7-GEMM Triton 结构算的，现在跑的是 aiter 的另一个结构，
对它相关的参照是 5-GEMM 下界 5.48 ms，还有约 1.74×。

## 明天可直接做的（按价值排序，已按端到端结果重排）

### 1. GEMM —— 今天唯一被证明有端到端价值的方向

hipBLASLt 修正路径后是 68.7 TF/s，Triton 是 897 TF/s，**还有 13×**。
而 e2e 是 GEMM-bound（32 层 attention 只占单步的 2.2%），所以这里的每一分收益都直接到账。

对比今天的实测：BLAS 路径一个环境变量换 **8.3×**（244 → 2027 tps），
而一整天的 attention 内核工作在端到端是 **−6.78%**。

**先做**：把 BLAS 路径错位报给镜像维护方（`BLAS-FINDING.md`，一行环境变量，影响所有用户）。
**再做**：评估把 GEMM 钉到 Triton。需要 `torch.compile`，而配置注释记载 inductor 对
TransformerBlock 做 autotune 会抛 `hipErrorLaunchFailure` **并打死 GPU** ——
**用 8 层配置试**（`repro_l8b_turbo_conv_8L.yaml`），不要在 32 层那个 88% 显存的配置上试。

### 2. 带两个修复重跑 e2e A/B —— 这是最可能翻盘的一件事

**做这个之前不要碰 ASM 反向的其他部分。** 代码审查在收尾时找到了第二个成因，
而 −6.78% 那个数字是**修复之前**测的：

| 成因 | 量级 | 状态 |
|---|---|---|
| **模块重载** | 每步 96 次 `hipModuleLoad`，从不卸载 | 已修（进程级 `HipModule` 单例） |
| 分配 churn | 每层 1.07 GB，每步约 34 GB | 已修（scratch 复用 + 清零） |

"scratch 单独修完没恢复回归"本身就是证据，说明**模块重载是两者中更大的那个**。
两个修复都**没有硬件验证**（写它们的时候卡不可用）。

```bash
D=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
BLAS_ENV="-e HIPBLASLT_TENSILE_LIBPATH=$D" E2E_ENV="-e PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD=1" \
  bash output/0915__opt/bin/e2e.sh fix2-on repro_l8b_turbo_conv_8L.yaml
BLAS_ENV="-e HIPBLASLT_TENSILE_LIBPATH=$D" \
  bash output/0915__opt/bin/e2e.sh fix2-off repro_l8b_turbo_conv_8L.yaml
```

**用 8 层配置**（`repro_l8b_turbo_conv_8L.yaml`，已验证峰值显存 141.00 GiB / 32.64%（早期记的 26% 与 step 行不符））。
32 层那个配置在 88% 显存上跑，今天两次挂卡都和它有关。
8 层单层形状与生产逐项相同，op 级 1.741× 的参照点不变。

若修复后 ON 反超 OFF，资格门就该改回默认开启（`_ENABLED` 那行）。
若仍落后，**默认关闭不变**，并把剩余差距列为未解释。

### 3. op-evolve 优化我们自己的融合反向

今天的结论把它的价值提高了：融合反向**不需要 `dq_acc`**（没有原子累加），
也不需要 4 倍大的 dk/dv，所以没有那个内存问题。
昨天在 B0 上 round 1 就拿到 −13.2%（`waves_per_eu 1→0` 加 `IN_THREAD_TRANSPOSE`）。
它占整块 GPU 跑一天，适合作为一整天的主线。
重启命令在 0914 的 `HANDOFF.md:114-118`，spec 陷阱在 `HOWTO-OP-EVOLVE.md:99-108`。

### 4. 5-GEMM 结构重构（只在 1–3 都推不动时）

`PLAN-4GPU-TOMORROW.md:160-178` 有完整设计和 accept/reject/abandon 判据。
交付物是**一个投资决策，不是一个内核**。

## 不要重做

除 `PROGRESS.md` 原有的 9 条外，今天新增：

- **`_perf` 变体**：aiter 自己不 dispatch 它们（CSV 里没有、源码里搜不到）。
  dq 只有 5.84 dB，位置 0–127 对、128 以后全错，128 正是 `TS_KV` ——
  它对 dq 是赋值而非原子累加。细节见 `PERF-CO-CLOSED.md`。
- **varlen**：gfx1250 **没有** varlen 的 ASM 反向。dispatch 表只有 `mode=0`（batch），
  host 源码对所有 seqstart 指针显式传 nullptr。不是"没接"，是"不存在"。
- **`b*nhead_q` 作为资格门判据**：错的。判据是 **seqlen**，见下。

## 今天踩的坑（都已修，但机制会复发）

1. **`TORCH_BLAS_PREFER_HIPBLASLT=0` 咬了三次**（T2 参考、第一次 e2e、`gemm_roof.py`）。
   `tune_attention.py` 在 import torch 前自己设了它，所以**所有 op 级测量从来免疫**，
   任何新路径都会撞上。必须在进程启动前进环境。
2. **参照臂被自己污染**：把 ASM 反向接进分发层后，走 `flash_attn_func` 取的"冠军"参照
   自己也变成了 ASM 反向。签名是**比值向 1.0 收敛而非报错**。
   修法：参照臂点名调 `dense_fused_backward`。
3. **孤儿进程**：`timeout` 只杀它 spawn 的进程，`torchrun` 是子进程会活下来，
   占着卡和端口 1234。已加 `timeout --foreground` + 随机端口 + 主动收割。
4. **root 所有的进程杀不掉**：容器以 root 跑，普通 kill 拿 EPERM，
   而 `2>/dev/null` 把它伪装成"进程不存在"。必须 `sudo -n`。
5. **我自己的两条后台命令竞争同一个哨兵文件**，导致调度器在探针运行时恢复、产生竞争测量。
   哨兵只能由一处管理。

## 资格门的判据（9 个测量点支撑）

比值 = 融合/ASM，>1 表示 ASM 赢：

| seqlen | b·hq=8 | =12 | =16 | =24 | =32 | =128 |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 0.660× | | | | 0.632× | 0.537× |
| 1536 | 0.869× | | | | | |
| 2048 | 1.748× | | | | 1.549× | |
| 4096 | 2.812× | 2.025× | 2.571× | 2.235× | 2.110× | |
| 8192 | 3.954× | | | | | 2.002× |

ASM 反向耗时 0.88/0.91/1.00/1.34 ms 跨 16 倍工作量 —— **近乎常数，固定地板约 0.85 ms**。
融合反向在 seqlen=1024 上 0.5806/0.5893/0.5785 ms（b·hq 8/32/128）—— **也是平的**，
那个规模下它延迟受限。两条平线相交 = 固定开销问题。**门槛 `seqlen >= 2048`**。

## 事故：机器被我搞挂，靠人工 AC-cycle 恢复

完整复盘在 `INCIDENT-2026-09-15-machine-death.md`。要点：

跑 20 步 e2e 时两臂之间没等卡安静 → 第一臂挂死 24 分钟 → `timeout` 只杀了它 spawn 的进程、
`torchrun` 作为子进程活下来占着卡和端口 → 第二臂绑不上端口 → 我反复探测、
每次多留一个 SIGKILL 不死的进程 → 最后在一块已经不能执行工作的卡上跑
`modprobe -r amdgpu`，**机器彻底失联，SSH 都连不上，由操作者 AC-cycle 恢复**。

我一度把重启后的正常状态误判为"我的驱动重载成功了"并写进文档 —— 那是错的，已更正。

**恢复阶梯第 3 步（驱动重载）应当删除。** 卡不能执行工作时，
驱动级操作的下界不是"没改善"而是"整台机器失联"。正确动作是报告并等 AC-cycle。

三个修复已进 `bin/e2e.sh`：等卡安静、`timeout --foreground` 杀进程组、随机 `MASTER_PORT`。

## 基础设施

- 队列：`output/0915__opt/`，`cat STATUS.md` 看状态，`>> queue.jsonl` 运行中可追加，
  `touch STOP` 暂停，`kill $(cat sched.pid)` 停止（**绝不要 `pkill -f`**，会匹配到杀手自己）。
- 全天：117 行、GPU 空转 0.3%、0 次 wedge。
- 健康探针统一定义在 `tools/gfx1250/gpu_health.sh`（树里原有四份互相矛盾的版本，
  其中三份还带着被撤回的过宽 `MES\(` 模式）。
