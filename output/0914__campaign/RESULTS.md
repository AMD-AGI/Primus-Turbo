# 0914 持续优化 — 结果账本

宿主 `ctheliosp-1b112-a37-1`，4× gfx1250，负载下 2133–2244 MHz。
形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`，除非另注。

---

## 1. 冠军推进（已提交 `566e7798`）

独占窗口下三次取中位数，行内离散 0.3%：

| 配置 | fwd | bwd | total | TFLOP/s |
|---|--:|--:|--:|--:|
| 昨日冠军 | 2.716 | 10.292 | 13.000 | 592.0 |
| `waves_per_eu` 不设 | 2.718 | 9.861 | 12.580 | 611.8 |
| `fwd num_warps=2` | 2.583 | 10.289 | 12.875 | 597.8 |
| **两者合并（新默认值）** | **2.579** | **9.867** | **12.412** | **620.5** |

**+4.28%**，四张量 SQNR 全程不变（53.67/52.24/52.31/52.71），51/51 测试绿。

`waves_per_eu` 是有意思的那个。昨天在 64 配置网格里它把指令数的中位变化量测成 **0.00%**，
那个观测现在依然成立——收益不在指令数上，而在后端自选的占用率上。昨天那张卡被限在 1100 MHz，
访存延迟按**周期**算小 2.14 倍，因而没有东西需要隐藏，这个旋钮也就没有可买的东西。
它还是一口窄井而不是一条趋势：`waves_per_eu=2` 是 49.7 ms，`=3` 是 78.7 ms。

## 2. 已测并**否决**的（别再走一遍）

| 实验 | 结果 | 结论 |
|---|---|---|
| 融合反向 `num_warps=2` | bwd 46.98 ms | 出厂的 4 就是最优 |
| 融合反向 `num_warps=8` | bwd 24.13 ms | 离线 ISA 预测它能把 VGPR 从顶满 1024 降到 512、`s_set_vgpr_msb` 少 3.2 倍——**实测 spill 代价压倒收益，寄存器压力假说对这个内核被证伪** |
| 融合反向 `num_warps=16` | bwd 23.54 ms | — |
| 融合反向 `num_stages=2` | bwd 12.55 ms | 与前向方向相反 |
| `BLOCK_M1/N2 = 16` | bwd 12.86 ms | 出厂的 32 最优 |
| `BLOCK_M1/N2 = 64` | bwd 23.97 ms | — |
| `waves_per_eu = 2 / 3` | bwd 49.68 / 78.75 ms | 悬崖 |
| 前向 `num_warps=8` | fwd 5.18 ms | 比出厂的 4 还差，2 是井底 |

## 3. GEMM / hipBLASLt —— 昨天的归因是错的

同一进程、同一组张量、GPU3：

| | TFLOP/s |
|---|--:|
| `torch.mm`（hipBLASLt，`preferred_blas` 确认为 `Cublaslt`） | **112.98** |
| 一个**朴素**的 Triton GEMM（`max_abs_err = 0.0`） | **1190.07** |
| 差距 | **10.5×** |

昨天记的是「27.4 TFLOP/s，因为镜像缺 gfx1250 Tensile 库」。三点修正：

1. 库**在**镜像里（`_rocm_sdk_libraries_gfx1250/lib` 下 46 个 bf16 Tensile 解）。
2. 把它放上 `LD_LIBRARY_PATH`、以及 `TORCH_BLAS_PREFER_HIPBLASLT` 在 0/1 之间切换，
   三种组合全部落在 110–120 TFLOP/s，**彼此差异 <1%**。路径假设被证伪。
3. 本机的数字是 113 而不是 27.4（8192³：40.12 ms → 9.71 ms，4.13×，远超 2.14× 时钟比）。

所以**不是缺库、不是回退到 rocBLAS**——hipBLASLt 在 gfx1250 上就是比一个随手写的 Triton
kernel 慢 10.5×。`PLATFORM-ESCALATION.md` 的措辞需要按这个改写。

## 4. 端到端（GPU3，容器 `fa-e2e`）

Llama-3.1-8B、MBS=GBS=4、seq 8192、单卡、AC=none、torchtitan **0.2.2**、20 步：

| 配置 | tps | TFLOP/s | MFU | 峰值显存 |
|---|--:|--:|--:|--:|
| c07-1 昨日（限频，torchtitan 0.1.0） | 245 | 14.17 | 4.54% | 379.5 GiB |
| **flex 基线（compile 关）** | **2,394** | **138.64** | **44.44%** | 380.1 GiB |
| compile 开（默认 inductor） | 2,295 | 132.9 | 42.61% | **260.7 GiB** |

**比昨天快 9.8×。** 但注意 138.64 TFLOP/s 正好落在 hipBLASLt 的 GEMM 天花板附近
（113–120）——**这一步就是被 BLAS 卡住的**，与第 3 节的探针端到端吻合。

`compile.enable=true` **单独开没有用**，反而慢 4%：inductor 默认把 matmul 留在 `aten.mm`
上，只融合逐元素算子，所以拿不到那 10.5×。显存倒是降了 119 GiB。
正在测的是强制它下降到自己的 Triton 模板：
`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` + `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`
（把 ATEN 排除掉才是让这个选择生效的关键）。

**JIRA 对标要说清是哪张表**：`jira_2`（seq 8192）是 19,795 → ~30,000 tps；
`jira_1`（seq **4096**）已报 30,188 tps。今天对标 seq 8192 那张。
19,795 tps 蕴含约 1,025 TFLOP/s，而我们现在是 138.64——**差的这一个数量级就是 GEMM**。

## 5. 方法论（今天中过的两个坑）

**抢卡的测量不会报错，只会记录一个错的数字。** 合并配置的第一次 A/B 是在两条扫描队列
还占着 GPU0/GPU2 时跑的，读数 15.6 和 13.6 ms、SQNR 漂到 51.27 dB。全是假的；
走 `bin/exclusive.sh` 取独占窗口后是 12.44 和 53.67。**没有 GPU 锁，纪律就是锁。**

**`pkill -f <pattern>` 会杀掉你自己的 shell**——模式匹配到了发起者自己的命令行。中过一次。

**`grep -c` 无匹配时既打印 `0` 又返回退出码 1**，`|| echo 0` 于是产生 `"0\n0"`，
watchdog 的所有算术判断随之失效。

**队列跑完网格后会退化成只重跑带轮次号的那个 tag**——不空闲，但 40 轮都在重复同一个测量。
幂等靠 tag，那么 tag 里就不能有单调变化的字段。

**`set -u` 下把一条流从 `declare -A` 表里摘掉，会让写死的循环列表在下一周期炸掉**，
watchdog 静默死亡、留下一个仍然显示 "ok" 的陈旧状态文件。现在改成遍历表的键。

---

## 6. T7 — aiter 预编译 gfx1250 ASM 前向：**接受，且打穿参考上界**

GPU0 独占窗口，与冠军交替测量 3 次，`--impl asm`（ASM 前向 + 融合反向，一个 autograd
Function，所以总时是实测不是两半相加）：

| impl | fwd | bwd | **total** | TFLOP/s | SQNR out/dq/dk/dv |
|---|--:|--:|--:|--:|---|
| **asm** | **1.410** | 9.770 | **11.147** | **690.5** | 53.62/52.24/52.31/52.71 |
| 冠军（融合） | 2.607 | 9.799 | 12.406 | 620.4 | 53.67/52.24/52.31/52.71 |

**总时 −10.1%，并且越过了纯 aiter 调优的参考上界 11.269 ms**——这条上界从昨天立到今天，
现在被自家组合打穿了。前向单看是我们的 **1.82×**、aiter Triton 调优后的 **1.48×**。

三件考古没说准或没能定的事，现已定：

- **文档里那个阻塞根本不存在**。`import aiter.ops.mha` 原样成功，
  `pa_decode_gluon.py` 里**没有任何** `import jax`。既不用装 jax 也不用绕 ctypes。
- **gqa=4 之前没人测过**（aiter 自己的测试只覆盖 gqa=8），实测没问题：
  32 个 query head 的输出 SQNR 落在 53.54–53.72 dB 的均匀带里，没有某个子集塌掉——
  而 gqa 比例索引错误恰恰会产生那种塌陷。
- **LSE 是自然对数，实测确认**（不只是读 aiter 的测试源码）：
  对 fp32 参考的 logsumexp 是 140.5 dB，最大绝对误差 4.8e-6。

两个配对细节值得记：ASM 前向返回的是平铺 `[B, Hq, Sq]`，`dense_fused_backward` 的
docstring 正好接受这个形状，**所以这两者零适配层**。而树内双内核反向吃的是打包的
`[B, Hq, 2*Sq]` lse/delta 暂存（两者每 `FIXED_BLOCK_M` 行交错），换过去要做 scatter。

**尚未落到 dispatcher**。上线前还差两项测量：非因果的 `.co` 变体（这轮只验了 causal），
以及训练负载下的确认——探针期间出现过一次未能诊断的卡状态退化，ASM 内核**丢掉了 100%
的优势**而 turbo 只慢 13%。

## 7. e2e —— 强制 Triton GEMM 是 3.9×

| 配置 | tps | TFLOP/s | 峰值显存 |
|---|--:|--:|--:|
| flex 基线（compile 关） | 2,393 | 138.60 | 380.1 GiB |
| compile 开（inductor 默认） | 2,293 | 132.78 | 260.7 GiB |
| **compile + `MAX_AUTOTUNE_GEMM=1` + `GEMM_BACKENDS=TRITON`** | **9,599** | **555.90** | 261.8 GiB |

`compile.enable=true` **单独开是负收益**（−4%）：inductor 默认把 matmul 留在 `aten.mm` 上，
只融合逐元素算子。**把 ATEN 从后端列表里排除掉**，逼它下降到自己的 Triton 模板，才是
那 3.9× 的来源——与第 3 节同进程测到的 hipBLASLt 113 vs Triton 1190（10.5×）同源。

注：日志里 MFU 报到 178% 是 torchtitan 对这颗芯片的 `peak_flops` 设小了，属报告假象，
不影响 tps。**这是一条平台级建议**：gfx1250 上只要用 torch.compile，就该同时强制
GEMM 后端为 Triton，否则等于没开。

## 8. HipKittens udna1 —— **RED**，但红在别处，而且红得更彻底

判定：`{"test":"udna1_mma_transpose_duality","bad":126,"total":256,"worst_abs_err":2.73,"verdict":"RED"}`

有意思的是**预测错了位置**。`conversions.cuh::transpose` 本身**能编译、而且结构上是对的**
——行/列寄存器布局在 wave32 下也是精确对偶（离线穷举 4608 个 (lane, idx, half) 三元组，
0 处违例）。真正的阻塞在别处，且都是**编译期**的硬墙，根本走不到数值问题：

1. **`reductions.cuh` 在 gfx1250 上完全编不过**：`__builtin_amdgcn_permlane32_swap`
   是 gfx950 独有指令，udna1 的 reductions 里有 7 个调用点，全部被拒。
2. **三个矩阵入口只活了一个**：`mma_ABt` 走 WMMA 能编；`mma_AB` 和 `mma_AtB` 仍然
   派发到 wave64 的 CDNA MFMA builtin，连重载都匹配不上。
3. **`swap_layout` 在 wave32 下可证明是坏的**：源端 4 个 bf16_2 只读 2 个、
   目的端 8 个只写 4 个，`data[4..7]` 未初始化。而它在 gfx1250 上是
   transpose→mma 路径的**必经环节**（`transpose<rt_16x16>` = rt_16x16，8 f32/lane，
   永远落不到 A 操作数要的 rt_16x32 / 16 bf16/lane 形状上）。

**结论**：反向路线在编译期就被挡死，transpose 就算绿了也不解锁任何东西。
7–10 工程周的估算应按这三条重新定价，而不是按「transpose 能不能算对」。
另外 udna1 的 72 个头里有 **53 个与 cdna4 字节相同**，真正被移植的只有 6 个。

## 9. FlyDSL Gate A —— **BLOCKED**，从假设变成确证

工具链门确实是开的（0.2.4 有那四个原语），gfx950 的 FlyDSL 前向翻一行 arch 门之后
**能一路 build 并 lower 过 MLIR**。但计算核心够不着：**gfx1250 根本没有 MFMA**。
容器自带 LLVM 直接探针：

| intrinsic | gfx950 | gfx1250 |
|---|---|---|
| `llvm.amdgcn.mfma.f32.32x32x16.bf16` | 选中 `v_mfma_f32_32x32x16_bf16` | **Cannot select** |
| `llvm.amdgcn.ds.read.tr16.b64` | 选中 `ds_read_b64_tr_b16` | **Cannot select** |
| `llvm.amdgcn.permlane32.swap` | 选中 `v_permlane32_swap_b32_e64` | **Cannot select** |

内核的数学、累加器宽度、以及每一个手推的 LDS→VGPR 立即数 stride，全部表达在
MFMA 32x32x16 wave64 的 fragment 布局上（`warp_size = 64` 硬编码在
`attn_helper.py:382`）。这正是「需要重写调度表」那一类，放弃触发器命中。
**18–36 工程日的估算得到确证，FlyDSL 维持排最后。**

顺带：`origin/dev/kyle/flydsl-attn-gqa4` 只有 +12/−6 两个文件，是把 `_gqa_group_ok`
从 `8 <= g` 放宽到 `1 <= g`，外加 dkdv 暂存循环的一个局部索引修正。它是门放宽，不是重构。
而我们 G=4 目前正是被这个门拒掉的——也正因为这个意外的拒绝，昨天那个
`>= (9,5)` 会放行 gfx1250 的 bug 才一直没在本机炸出来。

## 10. 端到端阶梯 —— 5.5×，且 attention 的价值**终于可测**

Llama-3.1-8B、MBS=GBS=4、seq 8192、单卡、AC=none、torchtitan 0.2.2、20 步，取后 8 步中位：

| 配置 | tps | TFLOP/s | s/step | 峰值显存 |
|---|--:|--:|--:|--:|
| flex，eager（基线） | 2,392 | 138.5 | 13.70 | 380.1 GiB |
| flex，compile 默认 | 2,294 | 132.8 | 14.28 | 260.6 |
| flex，compile + Triton GEMM | 9,602 | 556.1 | 3.41 | 261.8 |
| **turbo attention + compile + Triton GEMM** | **13,204** | **764.7** | **2.48** | 259.6 |

**对自家基线 5.52×。其中 GEMM 修复 4.01×，turbo attention 再乘 1.375×。**

patch 状态逐条核对过，不是靠配置文件推断的：
`compile_mt` 那一跑日志里是 `[Patch] ⊘ Skipped: torchtitan.primus_turbo.turbo_attention
(condition not met)` 且 `use_turbo_attention: False`——确为 flex 路径；
`turbo_mt` 是 `[Patch] ✓ Applied: torchtitan.primus_turbo.turbo_attention`，13/13 全上。
昨天的失败模式正是「patch 因缺依赖而静默失败、日志只说 patch failed」，所以这一步不能省。

**这就是昨天拿不到的东西。** 昨天 attention 占一个 step 不到 1%（35.7 ms × 32 = 1.14 s
of 133.7 s），所以 turbo 和 flex 落在 0.4% 以内，任何内核改动都既不能被证实也不能被证伪。
GEMM 让出关键路径后，attention 变成了 **38% 的端到端杠杆**。

日志里的 MFU（178% / 245%）是 torchtitan 对这颗芯片的 `peak_flops` 设小了——
patch 列表里就有一个 `torchtitan.peak_flops`，值需要按 gfx1250 修正。不影响 tps。
