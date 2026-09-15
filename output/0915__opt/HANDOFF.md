# 0915 交接

分支 `dev/lhz/attn`（已推送到 `AMD-AGI/Primus-Turbo`）。本机 = c07-1，单卡 gfx1250，VR 限频 1100 MHz。

## 一句话

算子从出厂 **55.785 ms** 推到 **11.117 ms（5.018×）**，走的是产品路径而非专用测量入口；
但当天最大的实际收益来自别处 —— 一个 hipBLASLt 的路径错位，修好后端到端训练吞吐 **8.3×**。

## 成绩阶梯（本机同口径，不经跨机换算）

| 版本 | n | fwd | bwd | total | 离散 |
|---|--:|--:|--:|--:|--:|
| turbo 出厂 `1cb2e183` | 3 | 10.651 | 45.134 | 55.785 | 0.93% |
| 强制 Triton 前向 + 融合反向 | 2 | 4.166 | 17.835 | 22.001 | 0.06% |
| ASM 前向 + 融合反向（晨间冠军） | 3 | 1.549 | 17.675 | 19.233 | 0.22% |
| **ASM 前向 + ASM 反向（产品路径）** | 2 | 1.573 | **9.544** | **11.117** | |

反向 9.544 ms 落在文档给的原结构"真实墙"9.2–9.9 ms 内。**这不代表到顶** ——
那条墙是给我们那个 7-GEMM Triton 结构算的，现在跑的是 aiter 的另一个结构，
对它相关的参照是 5-GEMM 下界 5.48 ms，还有约 1.74×。

## 明天可直接做的（按价值排序）

### 1. 重跑 20 步 e2e A/B —— 今天唯一没拿到的数字

今天两次都失败，原因已查清并修好（`bin/e2e.sh`）：孤儿 `torchrun` 占着 rendezvous 端口 1234。
现在脚本会先收割残留、等卡安静、并给每次运行随机 `MASTER_PORT`。

```bash
D=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
BLAS_ENV="-e HIPBLASLT_TENSILE_LIBPATH=$D" \
E2E_ENV="-e PRIMUS_TURBO_ASM_BWD_TRACE=1 -e PRIMUS_TURBO_ASM_BWD_TRACE_FILE=/tmp/t.trace" \
  bash output/0915__opt/bin/e2e.sh on20 repro_l8b_turbo_conv.yaml
# 对照臂加  -e PRIMUS_TURBO_ATTN_DISABLE_ASM_BWD=1
```

预期差异约 **1.5%**（attention 占 17 秒步长的 2.2%），需要 20 步取稳态才测得出。
**两臂之间必须等卡安静** —— 今天正是没等导致的失败。

### 2. 把 BLAS 修复推给镜像维护方

`TensileLibrary_lazy_gfx1250.dat` 被放在 `library/gfx1250/` 而加载器在 `library/` 找。
细节和实测在 `BLAS-FINDING.md`。这是**一行环境变量换 8.3×**，影响所有用这个镜像的人。

### 3. GEMM 仍有 13×

hipBLASLt 68.7 vs Triton 897 TF/s。钉到 Triton 需要 `torch.compile`，
而配置注释记载 inductor 对 TransformerBlock 做 autotune 会抛 `hipErrorLaunchFailure`
**并打死 GPU**。这个矛盾未解，**不要在没有准备的情况下试**。

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

## 卡的状态（收尾时）

05:14 之后出现 degraded：dmesg 全干净（0 wedge / 0 degraded 标记），
但**三个独立作业都跑不完** —— 两个探针（170 s / 580 s 预算）和一个队列行（70 s，实际被杀于 101 s）。
"dmesg 干净"不等于"卡能算"，健康检查必须包含真实计算探针。

已执行恢复阶梯第 1 步（收割进程）和第 2 步（重建容器）。
若仍不恢复，第 3 步是 `sudo modprobe -r amdgpu && sudo modprobe amdgpu`；
再不行需要人工 AC-cycle。

## 基础设施

- 队列：`output/0915__opt/`，`cat STATUS.md` 看状态，`>> queue.jsonl` 运行中可追加，
  `touch STOP` 暂停，`kill $(cat sched.pid)` 停止（**绝不要 `pkill -f`**，会匹配到杀手自己）。
- 全天：117 行、GPU 空转 0.3%、0 次 wedge。
- 健康探针统一定义在 `tools/gfx1250/gpu_health.sh`（树里原有四份互相矛盾的版本，
  其中三份还带着被撤回的过宽 `MES\(` 模式）。
