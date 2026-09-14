# gfx1250 注意力优化 — 0914 交接

**读这个文件，不要重新推导。** 昨天的交接在 `output/0913__opt_plan__claude/PROGRESS.md`，
今天的完整账本在 `output/0914__campaign/RESULTS.md`。这里只放明天要用的结论。

宿主 `ctheliosp-1b112-a37-1`，4× gfx1250，负载下 2133–2244 MHz（c07-1 是 VR 限频 1100）。
形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`。

---

## 1. 成绩

| 阶段 | total ms | TFLOP/s | commit |
|---|--:|--:|---|
| 昨天冠军（折算到本机满频） | 13.000 | 592 | — |
| + `waves_per_eu` 不设 + 前向 `num_warps=2` | 12.412 | 620 | `566e7798` |
| + aiter 预编译 ASM 前向 | 11.198 | 687 | `0e5cb743` |
| **+ in-thread transpose** | **10.299** | **747** | `a009599b` |

累计 **1.26×**。纯 aiter 调优的参考上界是 11.269 ms——**已被越过**。

四张量 SQNR 全程 `53.62/52.24/52.31/52.71`（ASM 前向那条 out 是 53.62，树内前向是 53.67）。
51/51 测试在**有 aiter 和无 aiter 两种情况下**都绿。200 次逐位确定性通过。

### 形状门表：ASM 在所有形状上都赢

s1024 **1.65×**、s2048 1.33×、b1 1.16×、s4096 1.15×、b2 1.13×、b8 1.12×、
**s8192 1.11×**、s16384 1.13×、非因果 1.08×。
**不需要形状豁免**——`asm_forward_eligible` 按能力判定（dtype/维度/stride/GQA/sink/swa/arch）是对的。

---

## 2. 明天第一件事：全机静默下复测

今天所有 op 层数字都在**四卡同时满载**下取的，整板功耗/温度预算共享会让单卡测量漂移
**最高 50%**（同一配置从 11.2 到 17.1 ms 都见过）。**`exclusive.sh` 的单卡独占窗口挡不住这个。**

明天的第一个动作：停掉全部四条流 → 冷却 120s → 复测冠军 → 以此为全天基线。
交替测量（A/B 交错）是唯一能在漂移中保住可比性的手段，`best-of-N` 比 median 更接近未受扰动的卡。

---

## 3. 平台级结论（两条都推翻了昨天的说法）

### hipBLASLt 不是「缺库」，是慢 10.5×

同进程同张量，8192³ bf16：`torch.mm` **113.0 TFLOP/s** vs 一个**朴素** Triton GEMM
**1190.1**（`max_abs_err = 0.0`）。

昨天记的是「27.4 TFLOP/s，因为镜像缺 gfx1250 Tensile 库」。三处修正：
库**在**（46 个 bf16 解）；放上 `LD_LIBRARY_PATH` 与 `TORCH_BLAS_PREFER_HIPBLASLT` 0/1
三种组合全落在 110–120，差异 <1%；本机的数是 113 不是 27.4。
`PLATFORM-ESCALATION.md` 的措辞要按这个改写。

**推论**：gfx1250 上用 `torch.compile` 必须同时把 GEMM 后端钉死到 Triton
（`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` + `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`），
否则 inductor 把 matmul 留在 `aten.mm` 上，**单开 compile 是 −4% 负收益**。
钉死之后端到端 **4.01×**（2,392 → 9,602 tps）。

### VR 限频代价 1.65× → ~1.95×

五行 like-for-like：1.88 / 1.92 / 1.93 / 1.97 / 2.01，中位 1.93×。
所有天花板按 2.136 重算：7-GEMM 下界 7.68→**3.59 ms**，「真实墙」9.2–9.9→**4.31–4.63 ms**，
「低于 13 ms 不是 Triton」→**低于 6.1 ms**。
⚠ 冠军现在 10.3 ms 与那句「13 ms」是数字巧合，**不是**「已达 Triton 极限」。

---

## 4. e2e：路径已打通到第六层，但还没跑通

**最重要的事实：`converters: []` 让 turbo attention 从未被测到。**

链条是：`turbo_attention` patch 只替换 **Attention 类** → Primus 的子类只重写 `forward`、
**没重写 `__init__`**，并**假设** `inner_attention` 已被换掉 → 真正做替换的是**模型 converter**
（`primus_turbo_converter.py:22`）→ 配置把它关了。
所以 `inner_attention` 一直是 torchtitan 的 `FlexAttentionWrapper`。

已由 traceback 逐层确证（见 RESULTS.md §17）。**因此今天 e2e 的「turbo vs flex = 1.459×」
测的是 `turbo_float8_linear` + `turbo_mx_linear`，不是 attention。**

`converters: []` 的原因（0.2.2 传 `enable_gqa` 导致 TypeError）**在 turbo 配置下不成立**：
那是 torchtitan **自己的** `Attention.forward` 在传，Primus 的子类调 `inner_attention(xq,xk,xv)`
只有 3 个参数。v5 必须关是因为 v5 没启用 turbo_attention patch。
已写好 `examples/torchtitan/configs/MI455X/repro_l8b_turbo_conv.yaml`。

打开 converter 后依次撞到并修掉：

| 层 | 问题 | 状态 |
|---|---|---|
| 1 | 探针 `os.getpid()` 不可追踪 | 修（import 时求值） |
| 2 | 探针 `open()` 不可追踪 | 修（`is_compiling()` 守卫） |
| 3 | 探针 `@torch.compiler.disable` 也是硬错误 | 修（关掉追踪） |
| 4 | **`os.environ` 写在启动路径上** | **修，`ad67a2cc`** |
| 5 | **`num_ctas` 直传 kernel** | **修，`ad67a2cc`** |
| 6 | inductor 试图重编我们的 Triton 内核 | 加了 `PRIMUS_TURBO_ATTN_NO_COMPILE=1` 开关 |

**第 4、5 层是产品代码里的真 bug**：融合反向此前**完全无法在 `torch.compile` 下运行**。
两道门的盲区正好错开——51 个测试不编译模型，e2e 的 `converters: []` 让它走不到这条路径，
**「compile 开」和「融合反向在跑」在今天之前从未同时成立过**。

明天从第 6 层继续。

---

## 5. op-evolve：自主循环产出了真东西

`round 1 accepted，548.9 → 633.2 TFLOP/s`。它找到两个旋钮：
`waves_per_eu 1→0`（−4.2%，**独立复现了我早上提交的 `566e7798`**）和
**`TRITON_HIP_USE_IN_THREAD_TRANSPOSE`（−5.9%，我没测过）**。

**两套 harness、两个进程、零共享，在同一个旋钮上吻合**——这比任何一边单独的结果都值钱。
它自己还诚实写了「合并 −13.2% 超过可加的 −9.9%，我有故事没有测量，那是故事不是发现」。

跑法（`artifacts/` 是相对 CWD 的，**必须先 cd 到仓库根**）：
```bash
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve
PATH=/home/lihuzhan/.venv-op-evolve/bin:$PATH op-evolve status --job gfx1250-attn-llama31-8b-bwd-20260914-015415
```

**codex 通了**，需要两步：`~/.codex/config.toml` 里配 `model_providers` 指向 AMD 网关
（SDK 不读 `OPENAI_BASE_URL`），再用 `env_http_headers` 把 `Ocp-Apim-Subscription-Key`
指向 `LLM_GATEWAY_KEY`。**cursor 的 key 报 `Invalid User API Key`，需要刷新。**

---

## 6. 已判死，不要重做

- **HipKittens udna1**：RED，且红在预测之外。`transpose` 本身能编译且结构正确
  （离线穷举 4608 个三元组 0 违例），真正的墙是三条**编译期**事实：
  `reductions.cuh` 因 `permlane32_swap` 是 gfx950 独有而完全编不过；
  三个矩阵入口只活了 `mma_ABt`；`swap_layout` 在 wave32 下只覆盖一半寄存器却是必经环节。
  GPU gate 确认 256 元素错 126 个。**7–10 工程周的估算要按这三条重新定价。**
- **FlyDSL**：BLOCKED（确证）。gfx1250 **根本没有 MFMA**——
  `mfma.f32.32x32x16.bf16` / `ds.read.tr16.b64` / `permlane32.swap` 在 gfx950 全部选中、
  gfx1250 全部 `Cannot select`。18–36 工程日估算成立。
- **E4 融合反向 `num_warps`**：8 → 24.13 ms、16 → 23.54（出厂的 4 是 10.29）。
  离线 ISA 筛预测 warps=8 能把 VGPR 从顶满 1024 降到 512、`s_set_vgpr_msb` 少 3.2 倍——
  **实测 spill 代价压倒收益，这个内核的寄存器压力假说被证伪。**
- 昨天 `PROGRESS.md` 的 9 条「不要重做」仍然有效。

---

## 7. 纪律（今天每一条都是踩出来的）

1. **抢卡的测量不会报错**——它记录一个错的数字，**而且会产生假的 SQNR 失败**。
   今天中过两次：`s4096|fused` out 43.23、`s2048|fused` 42.88，独占窗口重测全部恢复 53.7+。
   **任何 SQNR 失败在当成结论前必须在独占窗口复现一次。**
2. **四卡满载会让单卡测量漂移最高 50%**，`exclusive.sh` 挡不住。要么全机静默，要么交替测量。
3. **缺席不是证据。** 今天七次「改动应该有效果却没有」，**七次根因全在测量链路上**：
   镜像里的 primus_turbo 顶替了 checkout（`.pth` 注册 MetaPathFinder，`sys.meta_path`
   先于 `sys.path`，`PYTHONPATH` 压不过）；`AITER_LOG_LEVEL=ERROR` 压掉 aiter 横幅；
   stdout 被 launcher 吞；探针自身引入致命 graph break（三次）。
   **凡是「改动应该有效果却没有」，第一件事是打印被 import 模块的 `__file__`。**
4. **会改变程序能否运行的诊断不是诊断。** 探针连续三次决定了训练跑不跑得起来。
5. **队列的幂等 tag 里不能有单调变化的字段**，否则网格跑完后会退化成只重跑那一个 tag
   （今天空转了 40 轮）。
6. **`pkill -f <pattern>` 会杀掉你自己的 shell**；`grep -c` 无匹配时既打印 `0` 又返回 1，
   `|| echo 0` 产生 `"0\n0"`（两次栽在这个上）；`set -u` 下从 `declare -A` 摘掉一条流会让
   写死的循环列表在下一周期炸掉，watchdog 静默死亡还留着显示 "ok" 的陈旧状态文件。
7. **孤儿进程**：agent 的子进程会变成 `ppid=1` 继续占卡、继续占端口
   （一个残留的 `pt_elastic` 占着 1234 让三条臂的 12 次 run 全部在第 1 步前失败却被记为完成）。
   每次 run 用独立端口；不足 8 步的 run 不标记完成。
8. **降级 ≠ 挂死**。`MES failed to respond` 是降级（卡还在产出），
   `wait for reset ack` 才是不可恢复。把降级当挂死，代价是白扔一张还在干活的卡。
9. **`_verify_launch` 这道防线现在是坏的**（既有问题，非今天引入）：它期待 tile 尺寸出现在
   kernel 名里，而这个 Triton 版本的 `compiled.name` 就是 `bwd_kernel_causal`，
   打开 `PRIMUS_TURBO_FUSED_MHA_BWD_VERIFY=1` 必然误报。**修它，它守的是「sweep 结果平坦
   = 覆盖没生效」这个真实失败模式。**

---

## 8. 明天的排期建议

1. **全机静默复测冠军**（0.5h，阻塞其余一切）
2. **e2e 第 6 层**：`PRIMUS_TURBO_ATTN_NO_COMPILE=1` + converter，拿到 turbo attention
   的**真实**端到端数字。这是今天唯一没拿到的关键数。
3. **修 `_verify_launch`**（0.5h）——防线坏着，任何 sweep 结论都少一层保护
4. **op-evolve 继续**，它已经证明能产出我没想到的旋钮
5. **T2 aiter ASM 反向**：门槛按满频重算——必须打赢 **8.893 ms**（今天的反向），
   要值得 vendor 得打到 ~8.0。昨天写的「<18 ms 就是今天」已经完全过时。

---

## 9. 一个被我写进待办、随后被自己推翻的「发现」

一度记录为「低并行度形状会吊死内核」。**那是错的，不要继承。**

复测：生产形状 `llama31-8b`，**热 Triton 缓存 + 独占窗口**，正常完成——
`bwd 8.881 / tot 10.280 ms / 748.7 TF/s`，SQNR 53.62/52.24/52.31/52.71，
**wall 475.1 秒**。

我先前给的超时是 **420 秒**，而且每次都用**全新的 `TRITON_CACHE_DIR`**（冷缓存 = 全量 autotune 编译），
最后一次还**没取独占窗口就在跑着队列的卡上测**。三者叠加超过超时，看起来就是吊死。

**一次生产形状的测量本来就要约 8 分钟**，大头是 b=4 s=8192 的 fp32 参考。
任何 bounded 测量的超时不能低于 900 秒，冷缓存时更长。

`par-b1h32 / par-b4h8 / par-b2h8 / par-b1h8` 四个跨 `_MIN_PARALLEL_WORK=32` 的形状仍然有用，
已固化在 `tools/gfx1250/tune_attention.py` 的 `SHAPES` 里——
**那条分发门至今没有在本机实测验证过，这仍是待办**，但它不是「有形状会挂」。

**为什么要留着这一节**：这个不存在的 bug 已经被写进过待办。如果没有复查，
明天会有人去追一个不存在的问题。**监控噪声会污染记录，而记录会被当作事实继承。**

---

## 10. ⚠ 明天开工前必须先看：GPU0 坏了，GPU1/2/3 是好的

**结论（分卡直接测量，17:15）**：

| GPU | 4096³ bf16 稳态 | 显存 |
|---|--:|--:|
| **0** | **超时，跑不出来** | — |
| 1 | **124.8 TFLOP/s** | 0.7–1.1 GiB 干净 |
| 2 | **123.9 TFLOP/s** | 干净 |
| 3 | **122.0 TFLOP/s** | 干净 |

两个独立证据指向同一张卡：GPU0 是唯一跑不出结果的，而今天 dmesg 里 **120 条
`MES failed to respond` 全部在 `0001:04:00.0`**。故障从 uptime 14647 就开始了，
**远早于今天的工作**，不是我们触发的。

### 明天怎么开工

**不要重启。把 GPU0 从调度里摘掉，用 GPU1/2/3 三张卡。** 重启会丢掉 op-evolve 的运行态
（artifacts 在盘上可 `resume`，但要重新起），而且未必修得好一张硬件层面就有问题的卡。
`wait for reset ack` 全天为 0，驱动没有走到必须重启的那一步。

开工检查，**按这个顺序**：

```bash
# 1) 终态检查(零风险)
timeout 20 dmesg | grep -c 'wait for reset ack'        # 非 0 才必须重启

# 2) 分卡活性 —— 给足初始化预算，这是关键
for g in 1 2 3; do
  timeout 620 docker exec -e HIP_VISIBLE_DEVICES=$g fa-repro bash -lc \
    'timeout 580 python3 -c "
import torch,time,statistics
torch.cuda.init()
a=torch.randn(4096,4096,device=\"cuda\",dtype=torch.bfloat16); torch.cuda.synchronize()
b=a@a; torch.cuda.synchronize()
ts=[]
for _ in range(5):
    s=time.time(); c=a@a; torch.cuda.synchronize(); ts.append(time.time()-s)
print(\"%.1f TFLOP/s\"%(2*4096**3/statistics.median(ts)/1e12))"'
done
```
**一个冷进程要花约 78 秒在 HIP 初始化(58s)和分配(20s)上，然后才轮到计算。**
超时低于 600 秒测的是运行时启动，不是卡。

### 我当天在这件事上错了四次，模式完全一样

| # | 判断 | 依据 | 错在哪 |
|---|---|---|---|
| 1 | 「卡 wedged」 | dmesg 匹配 `MES(` | 模式太宽，命中了 `ring buffer is full` 这种背压 |
| 2 | 「误报，卡是好的」 | `tail -12` 只看到背压 | 证据窗口太窄，没往前翻 |
| 3 | 「D 状态，建议重启」 | 进程卡死、`kill -9` 无效 | D 状态是暂时的；`kill -9` 无效其实是 **EPERM** |
| 4 | 「四张卡全死」 | 150s matmul 超时 | **超时把 78 秒初始化算了进去** |

四次全是**从间接信号推断一个可以直接测量的事实**。

### 三个工具陷阱（都是当天真栽的）

- **`timeout 5 kill -0 $p` 跑的是 `/bin/kill`，不是 shell 内建**，语义不同，会假报「进程已消失」
- **`kill -0` 的 EPERM 被 `2>/dev/null` 吞成了「进程不存在」**——容器以 root 运行，
  宿主机上的普通用户无权向它发信号
- **容器有独立 PID 命名空间**：宿主 PID 3346063 ↔ 容器 PID 187287，两边的 `kill` 不通用

### 一个仍未查明的

`amdgpu` 引用计数清理后从 30 降到 24，但仍偏高；**HIP 初始化 58 秒是异常的**（正常几秒）。
没查清是 GPU0 的故障拖累了整个驱动，还是这个镜像本来就慢。
**这直接决定测量吞吐**——今天下午每个测量光初始化就要一分半，我一直误以为是「冷缓存编译慢」。

## 11. 今天没做完的（明天照 §8 的顺序做）

| # | 项 | 为什么没做完 |
|---|---|---|
| 1 | **静默复测冠军** | 收尾时在 GPU1 上补跑了（见 §13）。今天其余 op 数字都在四卡满载下取，仍建议明天用 GPU1/2/3 重做一次交叉标定 |
| 2 | **turbo attention 的真实端到端数字** | 连拆六层障碍（第 4、5 层是产品真 bug，已修并提交 `ad67a2cc`），第 6 层 inductor 重编我们的 Triton 内核未过。`PRIMUS_TURBO_ATTN_NO_COMPILE=1` 开关已就位但未验证 |
| 3 | **`_MIN_PARALLEL_WORK=32` 分发门的边界验证** | 四个跨阈值形状已固化进 `SHAPES`，只测到 `par-b1h8` 一点（turbo: bwd 1.076 / tot 1.430） |
| 4 | **`_verify_launch` 的 tile 检查** | 已修成不再误报，但这个 Triton 版本确实读不到 tile，现报 `UNVERIFIABLE`。要恢复完整保护得换信号源 |

## 12. 给明天的两条方法建议

### 一、任何「X 是否在工作」的判断，先问「有没有直接测 X 的办法」

当天有**九次**「现象不符合预期」，**九次根因都在测量或编排链路上，零次在被测的改动上**。
代价最大的四次，共同点是**用间接信号推断一个可以直接测量的事实**：

| 想知道的事 | 我用的代理 | 代理为什么说谎 | 直接的办法 |
|---|---|---|---|
| 卡在不在干活 | ledger 时间戳 | 网格跑完后每轮仍写 `round_complete` | 跑个 matmul，或 `rocm-smi --showuse` |
| 卡是否恢复 | 进程是否还在 D 状态 | 退出 D 状态 ≠ 能算 | 分卡跑 4096³ matmul |
| 卡是否全死 | 一个 150s 超时的 matmul | 超时把 78s 初始化算了进去 | 分阶段计时，给足预算 |
| ASM 前向有没有被走到 | 日志里有没有 aiter 横幅 | `AITER_LOG_LEVEL=ERROR` 压掉了 | 门自己写文件；或读 traceback |

三张卡闲置数小时**是操作者一眼看出来的，不是监控报出来的**——因为监控看的是代理。

### 二、不要用一个会改变被测对象的诊断

探针连续三次让训练跑不起来（`os.getpid()`、`open()`、`@torch.compiler.disable`
在被追踪的 `autograd.Function` 里都是硬错误）。**会改变程序能否运行的诊断不是诊断。**

同类：`pkill -f <pattern>` 会杀掉发起者自己的 shell（当天中过 4 次，
其中一次直接终止了正在收集结果的回合）。
