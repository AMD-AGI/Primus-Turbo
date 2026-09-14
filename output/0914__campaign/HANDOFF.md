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
